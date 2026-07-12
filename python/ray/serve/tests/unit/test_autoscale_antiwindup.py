import sys

import pytest

from ray.serve.autoscaling_policy import (
    _apply_autoscaling_config,
    replica_queue_length_autoscaling_policy,
)
from ray.serve.config import AutoscalingConfig, AutoscalingContext


def _ctx(current, target, total_requests):
    """Minimal context for the core queue-length policy.

    desired = total_num_requests / target_ongoing_requests (the `current` base cancels),
    so with target_ongoing_requests=1, desired == total_requests before the cap.
    """
    config = AutoscalingConfig(
        min_replicas=1,
        max_replicas=100000,
        target_ongoing_requests=1,
        upscaling_factor=1.0,
        downscaling_factor=1.0,
        upscale_delay_s=0,
        downscale_delay_s=0,
    )
    return AutoscalingContext(
        config=config,
        current_num_replicas=current,
        target_num_replicas=target,
        total_num_requests=total_requests,
        capacity_adjusted_min_replicas=1,
        capacity_adjusted_max_replicas=100000,
        policy_state={},
        deployment_id=None,
        deployment_name=None,
        app_name=None,
        running_replicas=None,
        current_time=None,
        total_queued_requests=None,
        aggregated_metrics=None,
        raw_metrics=None,
        last_scale_up_time=None,
        last_scale_down_time=None,
        total_pending_async_requests=0,
    )


# Invoke the policy exactly as production does: the anti-windup cap lives in
# _apply_default_params, which runs only via the _apply_autoscaling_config wrapper
# (see DeploymentAutoscalingState._policy). Delays are zeroed above so the wrapper's
# delay logic is a pass-through and the assertions isolate the scaling-factor + cap.
_PRODUCTION_POLICY = _apply_autoscaling_config(replica_queue_length_autoscaling_policy)


def _desired(ctx):
    desired, _ = _PRODUCTION_POLICY(ctx)
    return desired


def test_antiwindup_caps_desired_while_starting():
    """A batch is still starting (target>current) and the transient backlog would push
    desired past the committed target (desired>target): the anti-windup cap holds desired
    at the target so the in-flight replicas drain the backlog instead of stacking more."""
    ctx = _ctx(current=100, target=200, total_requests=500)
    assert _desired(ctx) == 200


def test_antiwindup_does_not_block_initial_jump():
    """target==current (nothing starting): the initial jump to true demand is untouched."""
    ctx = _ctx(current=100, target=100, total_requests=500)
    assert _desired(ctx) == 500


def test_antiwindup_does_not_affect_scale_down():
    """desired<target while starting: only upward windup is capped; scale-down untouched."""
    ctx = _ctx(current=100, target=200, total_requests=50)
    assert _desired(ctx) == 50


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
