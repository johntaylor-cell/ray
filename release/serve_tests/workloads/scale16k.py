"""Pinned controller-scaling driver for the RAY_SERVE_RECON_SWEEP_FRACTION study.

Per checkpoint N: reach N replicas PINNED (min==max==N, no autoscaling ramp) with NO
load, then apply steady-state load and sample ControllerHealthMetrics every SAMPLE s
for MARIN s. All opts are unconditional in the image; RAY_SERVE_RECON_SWEEP_FRACTION is
set per job. Each sample row is augmented with the health-check deadline-staleness
fields (the image-resident row extractor doesn't know them). Serve restarts between
checkpoints so each N's cumulative deadline counters start fresh. Exit 0 as long as >=1
sample was banked -- post-measurement teardown races (in-flight request hits a killed
replica) are benign and must not fail the run.
"""
import asyncio
import json
import os
import statistics
import time

import ray
from ray import serve
import ray.serve._private.benchmarks.common as cmn

NUM_CPUS = float(os.environ.get("NUM_CPUS", "0.1"))
MAX_REPLICAS = int(os.environ.get("MAX_REPLICAS", "16384"))
CKPTS = [int(x) for x in os.environ.get("CKPTS", str(MAX_REPLICAS)).split(",")]
MARIN = int(os.environ.get("MARIN", "180"))
SAMPLE = int(os.environ.get("SAMPLE", "5"))
WAITER = int(os.environ.get("WAITER", "7200"))
PIN = int(os.environ.get("PIN", "1"))
LOAD_MULT = float(os.environ.get("LOAD_MULT", "1"))
FRACTION = os.environ.get("RAY_SERVE_RECON_SWEEP_FRACTION", "")

cmn._CONTROLLER_WAITER_TIMEOUT_S = WAITER

_DEADLINE_KEYS = (
    "health_deadline_misses",
    "health_checks_recorded",
    "health_max_lateness_s",
    "health_max_gap_s",
    "health_gap_p50_s",
    "health_gap_p90_s",
    "health_gap_p99_s",
    "health_gap_p99_9_s",
    "push_checks_recorded",
    "push_check_gap_p50_s",
    "push_check_gap_p99_s",
    "push_check_gap_max_s",
)


def _autoscaling(n):
    return {
        "min_replicas": n if PIN else 1,
        "max_replicas": n,
        "target_ongoing_requests": 1,
        "upscale_delay_s": 1,
    }


async def _pinned_checkpoint(handle, signal_actor, checkpoint, target_replicas, marin_s, sample_s):
    """Reach target replicas FIRST (no load), THEN apply steady load + sample."""
    start_time = time.time()
    await cmn._controller_wait_for_replicas_up(int(target_replicas * 0.95), timeout=WAITER)
    print("SCALE16K_PHASE replicas_up", flush=True)
    await cmn._controller_wait_for_deployment_healthy(timeout=WAITER)
    print("SCALE16K_PHASE deployment_healthy", flush=True)
    autoscale_duration_s = time.time() - start_time
    pending_requests = [handle.remote() for _ in range(int(target_replicas * LOAD_MULT))]
    print("SCALE16K_PHASE load_dispatched", flush=True)
    try:
        await cmn._controller_wait_for_waiters(
            signal_actor, len(pending_requests),
            timeout=float(os.environ.get("WAITER_WAIT_S", "180")),
        )
    except RuntimeError as _e:
        print("SCALE16K_WAITER_PARTIAL " + repr(str(_e)), flush=True)
    # Cancel requests the router never managed to place: their retry churn
    # (queue-len probes + native call buffers) grows the driver by ~GB/min at
    # 16K and OOM-kills it. The placed majority keeps the fleet loaded.
    unplaced = [
        r for r in pending_requests if not r._replica_result_future.done()
    ]
    for r in unplaced:
        r.cancel()
    if unplaced:
        print(f"SCALE16K_PHASE cancelled_unplaced n={len(unplaced)}", flush=True)
    print("SCALE16K_PHASE sampling_start", flush=True)
    samples = []
    _profile_controller_when_starved(samples)
    num_samples = marin_s // sample_s
    for sample_idx in range(num_samples):
        health_metrics = await cmn._controller_get_health_metrics()
        actual_replicas = await cmn._controller_get_replica_count()
        num_nodes = cmn._controller_get_active_nodes()
        row = cmn._controller_extract_metrics_row(
            health_metrics=health_metrics, checkpoint=checkpoint, sample=sample_idx,
            target_replicas=target_replicas, actual_replicas=actual_replicas,
            num_nodes=num_nodes, autoscale_duration_s=autoscale_duration_s,
        )
        for _k in _DEADLINE_KEYS:
            row[_k] = health_metrics.get(_k)
        row["recon_sweep_fraction"] = FRACTION
        samples.append(row)
        if sample_idx < num_samples - 1:
            await asyncio.sleep(sample_s)
    await signal_actor.send.remote(clear=True)
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending_requests, return_exceptions=True), timeout=30.0
        )
    except asyncio.TimeoutError:
        pass
    return samples


def _profile_controller_when_starved(sampled_flag):
    """If sampling produces nothing for 45s, py-spy the controller (same node)
    and print its stacks as SCALE16K_PYSPY lines -- names the ingest hog."""
    import glob
    import subprocess
    import threading

    def _controller_pid():
        for cmdline in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                args = open(cmdline, "rb").read().decode(errors="ignore")
            except Exception:
                continue
            if "ServeController" in args:
                return cmdline.split("/")[2]
        return None

    def _run():
        time.sleep(45)
        if sampled_flag:
            return
        pid = _controller_pid()
        print(f"SCALE16K_PYSPY controller_pid={pid}", flush=True)
        if pid is None:
            return
        try:
            subprocess.run(
                ["sudo", "-E", "env", "PATH=" + os.environ.get("PATH", ""),
                 "py-spy", "record", "--pid", str(pid), "-d", "20", "-r", "200",
                 "-f", "raw", "-o", "/tmp/controller.folded"],
                capture_output=True, text=True, timeout=60,
            )
            lines = open("/tmp/controller.folded").read().splitlines()
            def weight(line):
                try:
                    return int(line.rsplit(" ", 1)[1])
                except Exception:
                    return 0
            lines.sort(key=weight, reverse=True)
            print("SCALE16K_PYSPY folded top stacks:", flush=True)
            for line in lines[:25]:
                print("SCALE16K_PYSPY " + line[-400:], flush=True)
        except Exception as e:
            print(f"SCALE16K_PYSPY error={e!r}", flush=True)

    threading.Thread(target=_run, daemon=True).start()


def _start_driver_memory_monitor():
    """Diagnose driver OOM: log the cgroup limit once, then RSS + top Python
    allocators every 30s as SCALE16K_DRIVERMEM lines (readable post-mortem)."""
    import gc
    import threading
    import tracemalloc

    tracemalloc.start(1)

    def _cgroup_limit():
        for path in (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ):
            try:
                return open(path).read().strip()
            except Exception:
                continue
        return "unknown"

    def _rss_mb():
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
        except Exception:
            pass
        return -1

    print(f"SCALE16K_DRIVERMEM_LIMIT {_cgroup_limit()}", flush=True)

    def _loop():
        while True:
            try:
                snap = tracemalloc.take_snapshot()
                stats = snap.statistics("lineno")
                traced_mb = sum(st.size for st in stats) // (1024 * 1024)
                tops = "|".join(
                    f"{st.traceback[0].filename.split(chr(47))[-1]}:"
                    f"{st.traceback[0].lineno}={st.size // (1024 * 1024)}MB"
                    for st in stats[:8]
                )
                print(
                    f"SCALE16K_DRIVERMEM rss_mb={_rss_mb()} "
                    f"traced_mb={traced_mb} gc_counts={gc.get_count()} "
                    f"top={tops}",
                    flush=True,
                )
            except Exception as e:
                print(f"SCALE16K_DRIVERMEM error={e!r}", flush=True)
            time.sleep(30)

    threading.Thread(target=_loop, daemon=True).start()


async def main():
    _start_driver_memory_monitor()
    if not ray.is_initialized():
        ray.init()
    signal = cmn._SignalActorForController.remote()
    print("SCALE16K_START " + json.dumps({
        "ckpts": CKPTS, "num_cpus": NUM_CPUS, "marination_s": MARIN,
        "sample_s": SAMPLE, "pin": PIN, "recon_sweep_fraction": FRACTION,
    }), flush=True)
    total_samples = 0
    for i, N in enumerate(CKPTS):
        try:
            hello = cmn.ControllerBenchHelloWorld.bind(signal)
            app = cmn.ControllerBenchMetricsGenerator.options(
                ray_actor_options={"num_cpus": NUM_CPUS},
                autoscaling_config=_autoscaling(N),
            ).bind(hello)
            print("SCALE16K_PHASE serve_run_start", flush=True)
            handle = serve.run(app, name="default", route_prefix=None)
            print("SCALE16K_PHASE serve_run_done", flush=True)
            samples = await _pinned_checkpoint(handle, signal, i, N, MARIN, SAMPLE)
            for s in samples:
                print("SCALE16K_SAMPLE " + json.dumps(s), flush=True)
            total_samples += len(samples)
            if samples:
                loops = [s.get("loop_duration_mean_s", 0) for s in samples]
                last = samples[-1]
                print("SCALE16K_SUMMARY " + json.dumps({
                    "N": N, "fraction": FRACTION, "samples": len(samples),
                    "actual_replicas": last.get("actual_replicas"),
                    "loop_mean_s": round(statistics.mean(loops), 4),
                    "dep_update_mean_s": round(statistics.mean(
                        [s.get("deployment_state_update_mean_s", 0) for s in samples]), 4),
                    "app_update_mean_s": round(statistics.mean(
                        [s.get("application_state_update_mean_s", 0) for s in samples]), 4),
                    "loops_per_second_mean": round(statistics.mean(
                        [s.get("loops_per_second", 0) for s in samples]), 3),
                    "health_deadline_misses": last.get("health_deadline_misses"),
                    "health_checks_recorded": last.get("health_checks_recorded"),
                    "health_max_lateness_s": last.get("health_max_lateness_s"),
                    "health_max_gap_s": last.get("health_max_gap_s"),
                    "health_gap_p90_s": last.get("health_gap_p90_s"),
                    "health_gap_p99_s": last.get("health_gap_p99_s"),
                    "push_check_gap_p99_s": last.get("push_check_gap_p99_s"),
                    "push_check_gap_max_s": last.get("push_check_gap_max_s"),
                    "push_checks_recorded": last.get("push_checks_recorded"),
                }), flush=True)
        except Exception as e:
            print("SCALE16K_FAIL N=%d %r" % (N, e), flush=True)
        finally:
            try:
                serve.shutdown()
            except Exception:
                pass
            await asyncio.sleep(5)
    print("SCALE16K_DONE total_samples=%d" % total_samples, flush=True)
    os._exit(0 if total_samples > 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
