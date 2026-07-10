import asyncio

import pytest

import ray
from ray._raylet import GcsClient
from ray.serve._private.cluster_node_info_cache import DefaultClusterNodeInfoCache
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.test_utils import get_node_id
from ray.tests.conftest import *  # noqa


def test_get_alive_nodes(ray_start_cluster):
    cluster = ray_start_cluster
    cluster.add_node(resources={"head": 1})
    ray.init(address=cluster.address)
    worker_node = cluster.add_node(resources={"worker": 1})
    cluster.wait_for_nodes()

    head_node_id = ray.get(get_node_id.options(resources={"head": 1}).remote())
    worker_node_id = ray.get(get_node_id.options(resources={"worker": 1}).remote())

    gcs_client = GcsClient(address=ray.get_runtime_context().gcs_address)
    cluster_node_info_cache = create_cluster_node_info_cache(gcs_client)
    cluster_node_info_cache.update()
    assert set(cluster_node_info_cache.get_alive_nodes()) == {
        (head_node_id, ray.nodes()[0]["NodeName"], ""),
        (worker_node_id, ray.nodes()[0]["NodeName"], ""),
    }
    assert cluster_node_info_cache.get_alive_node_ids() == {
        head_node_id,
        worker_node_id,
    }
    assert (
        cluster_node_info_cache.get_alive_node_ids()
        == cluster_node_info_cache.get_active_node_ids()
    )

    cluster.remove_node(worker_node)
    cluster.wait_for_nodes()

    # The killed worker node shouldn't show up in the alive node list.
    cluster_node_info_cache.update()
    assert cluster_node_info_cache.get_alive_nodes() == [
        (head_node_id, ray.nodes()[0]["NodeName"], "")
    ]
    assert cluster_node_info_cache.get_alive_node_ids() == {head_node_id}
    assert (
        cluster_node_info_cache.get_alive_node_ids()
        == cluster_node_info_cache.get_active_node_ids()
    )


# Snapshots shaped like _apply_snapshot's tuple:
# (alive_nodes, alive_node_id_set, node_labels, total_resources, available_resources).
_OLD = ([("old", "old", "")], frozenset({"old"}), {}, {}, {})
_NEW = ([("new", "new", "")], frozenset({"new"}), {}, {}, {})


def test_shutdown_update_not_clobbered_by_stale_async_refresh():
    """A slow in-flight refresh_async must not overwrite a newer snapshot applied by
    the synchronous update() while the async was suspended in the executor -- the
    controller's shutdown path. The epoch guard rejects the stale late apply.
    """

    async def _run():
        cache = DefaultClusterNodeInfoCache(object())  # GCS is never called
        loop = asyncio.get_running_loop()
        # Route the executor step to a future we resolve by hand so we control
        # exactly when the in-flight async refresh resumes.
        exec_future = loop.create_future()
        loop.run_in_executor = lambda executor, fn: exec_future

        # An async refresh is issued first and suspends waiting on the executor.
        task = asyncio.create_task(cache.refresh_async())
        await asyncio.sleep(0)  # let it stamp its epoch and suspend on the future

        # Shutdown path: the synchronous update() applies a NEWER snapshot.
        cache._compute_snapshot = lambda: _NEW
        cache.update()
        assert cache.get_alive_nodes() == _NEW[0]

        # The stale executor result now lands; the epoch guard must reject it.
        exec_future.set_result(_OLD)
        await task
        assert cache.get_alive_nodes() == _NEW[0]  # NOT clobbered by stale _OLD

    asyncio.run(_run())


def test_async_refresh_issued_after_update_still_applies():
    """Mirror check: an async refresh issued AFTER an update() must still win -- the
    guard rejects only strictly-older applies, never a legitimately newer one.
    """

    async def _run():
        cache = DefaultClusterNodeInfoCache(object())
        loop = asyncio.get_running_loop()

        cache._compute_snapshot = lambda: _OLD
        cache.update()  # epoch 1
        assert cache.get_alive_nodes() == _OLD[0]

        exec_future = loop.create_future()
        loop.run_in_executor = lambda executor, fn: exec_future
        task = asyncio.create_task(cache.refresh_async())  # epoch 2, issued later
        await asyncio.sleep(0)
        exec_future.set_result(_NEW)
        await task
        assert cache.get_alive_nodes() == _NEW[0]  # newer refresh wins

    asyncio.run(_run())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
