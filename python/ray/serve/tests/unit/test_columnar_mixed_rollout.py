"""Reviewer-flagged correctness (mixed columnar + cloudpickle stores).

During a rolling upgrade the columnar (array) and cloudpickle (object) metric stores
can BOTH hold data: the controller wire-detects the format independent of
RAY_SERVE_COLUMNAR_METRICS. The aggregation must count both, never double-count, and
be exact for every aggregation function.
"""
import random
import sys

import pytest

import ray.serve._private.autoscaling_state as A
from ray.serve._private import autoscaling_metrics_codec as codec
from ray.serve._private.autoscaling_state import DeploymentAutoscalingState
from ray.serve._private.common import (
    RUNNING_REQUESTS_KEY,
    DeploymentHandleSource,
    DeploymentID,
    HandleMetricReport,
    ReplicaID,
    ReplicaMetricReport,
    TimeStampedValue,
)
from ray.serve.config import AggregationFunction, AutoscalingConfig

NOW = 1000.0
DEP = DeploymentID("D", "default")


def _cfg(agg):
    return AutoscalingConfig(
        min_replicas=1,
        max_replicas=1000,
        target_ongoing_requests=1,
        aggregation_function=agg,
    )


def _state(agg=AggregationFunction.MEAN):
    st = DeploymentAutoscalingState(DEP)
    st._config = _cfg(agg)
    return st


def _replica_report(i, rng):
    npts = rng.randint(1, 6)
    series = [
        TimeStampedValue(round(NOW - 6.0 * (npts - 1 - j), 2), float(rng.randint(0, 9)))
        for j in range(npts)
    ]
    return ReplicaMetricReport(
        replica_id=ReplicaID(f"r{i}", DEP),
        aggregated_metrics={RUNNING_REQUESTS_KEY: 0.0},
        metrics={RUNNING_REQUESTS_KEY: series},
        timestamp=NOW,
    )


@pytest.mark.parametrize(
    "agg", [AggregationFunction.MEAN, AggregationFunction.MAX, AggregationFunction.MIN]
)
def test_mixed_rollout_equals_all_object(agg, monkeypatch):
    """B: a fleet split across columnar + cloudpickle stores totals the SAME as
    all-cloudpickle -- no drop, and exact for every aggregation function (the additive
    sum of two separate aggregations would be wrong for MAX/MIN)."""
    monkeypatch.setattr(A.time, "time", lambda: NOW + 3.0)
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(2, 6)
        reports = [_replica_report(i, rng) for i in range(n)]
        running = {r.replica_id for r in reports}
        ref = _state(agg)
        for r in reports:
            ref._replica_metrics[r.replica_id] = r
        ref._running_replicas = running
        ref_total = ref._calculate_total_requests_aggregate_mode()
        mix = _state(agg)
        k = rng.randint(0, n)
        for r in reports[:k]:
            mix._replica_metrics[r.replica_id] = r
        for r in reports[k:]:
            rid, ts, val, t = codec.decode_replica_running_requests(codec.encode(r))
            mix._replica_running_arrays[rid] = (ts, val, t)
        mix._running_replicas = running
        assert abs(ref_total - mix._calculate_total_requests_aggregate_mode()) < 1e-9


def test_columnar_counted_independent_of_flag(monkeypatch):
    """E: columnar arrays are aggregated whenever present -- the path does not gate on
    RAY_SERVE_COLUMNAR_METRICS, so a flag-off controller still counts columnar frames."""
    monkeypatch.setattr(A.time, "time", lambda: NOW + 3.0)
    rng = random.Random(3)
    reports = [_replica_report(i, rng) for i in range(4)]
    st = _state()
    for r in reports:
        rid, ts, val, t = codec.decode_replica_running_requests(codec.encode(r))
        st._replica_running_arrays[rid] = (ts, val, t)
    st._running_replicas = {r.replica_id for r in reports}
    assert st._replica_metrics == {}
    assert st._calculate_total_requests_aggregate_mode() > 0.0


def test_dedup_at_write_replica(monkeypatch):
    """B: a replica that switches wire format lives in exactly one store (no double)."""
    monkeypatch.setattr(A.time, "time", lambda: NOW + 3.0)
    st = _state()
    r = _replica_report(0, random.Random(1))
    st.record_request_metrics_for_replica(r)
    assert r.replica_id in st._replica_metrics
    nxt = ReplicaMetricReport(
        replica_id=r.replica_id,
        aggregated_metrics={RUNNING_REQUESTS_KEY: 0.0},
        metrics={RUNNING_REQUESTS_KEY: [TimeStampedValue(NOW, 5.0)]},
        timestamp=NOW + 1,
    )
    rid, ma, t = codec.decode_replica_all_metrics(codec.encode(nxt))
    st.record_columnar_metrics_for_replica(rid, ma, t)
    assert rid not in st._replica_metrics  # object entry cleared
    assert rid in st._replica_running_arrays
    st.record_request_metrics_for_replica(
        ReplicaMetricReport(
            replica_id=r.replica_id,
            aggregated_metrics={RUNNING_REQUESTS_KEY: 0.0},
            metrics={RUNNING_REQUESTS_KEY: [TimeStampedValue(NOW, 5.0)]},
            timestamp=NOW + 2,
        )
    )
    assert rid not in st._replica_running_arrays  # array entry cleared
    assert r.replica_id in st._replica_metrics


def _handle_report(hid, queued):
    return HandleMetricReport(
        deployment_id=DEP,
        handle_id=hid,
        actor_id="a",
        handle_source=DeploymentHandleSource.PROXY,
        aggregated_queued_requests=0.0,
        queued_requests=queued,
        aggregated_metrics={RUNNING_REQUESTS_KEY: {}},
        metrics={RUNNING_REQUESTS_KEY: {}},
        timestamp=NOW,
    )


def test_queued_from_both_stores(monkeypatch):
    """C: _get_queued_requests includes columnar handle queued, not just object."""
    monkeypatch.setattr(A.time, "time", lambda: NOW + 3.0)
    monkeypatch.setattr(A, "RAY_SERVE_AGGREGATE_METRICS_AT_CONTROLLER", True)
    q = [TimeStampedValue(NOW - 6, 2.0), TimeStampedValue(NOW, 2.0)]
    obj_h, col_h = _handle_report("h_obj", q), _handle_report("h_col", q)
    ref = _state()
    ref._handle_requests["h_obj"] = obj_h
    ref._handle_requests["h_col"] = col_h
    ref._running_replicas, ref._cached_running_replica_strs = set(), set()
    ref_q = ref._get_queued_requests()
    mix = _state()
    mix._handle_requests["h_obj"] = obj_h
    mix.record_columnar_metrics_for_handle(
        codec.decode_handle_flat(codec.encode(col_h))
    )
    mix._running_replicas, mix._cached_running_replica_strs = set(), set()
    assert mix._get_queued_requests() > 0.0
    assert abs(ref_q - mix._get_queued_requests()) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
