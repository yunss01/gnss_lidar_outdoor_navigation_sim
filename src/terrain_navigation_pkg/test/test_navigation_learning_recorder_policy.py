from types import SimpleNamespace

import pytest
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

from terrain_navigation_pkg.navigation_learning_recorder_node import (
    NavigationLearningRecorder,
    path_hard_valid_subscription_qos,
    perception_capture_completion_result,
    perception_capture_readiness,
)


class _Parameters:
    def __init__(self, **values):
        self.values = values

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])


def _policy(**overrides):
    values = {
        'controlled_traversability_actor_id': -1,
        'controlled_traversability_disposition': '',
        'controlled_traversability_blueprint': '',
        'controlled_traversability_policy_source': (
            'vehicle_clearance_policy'
        ),
    }
    values.update(overrides)
    return NavigationLearningRecorder._controlled_traversability_actor_policy(
        _Parameters(**values)
    )


def test_disabled_controlled_actor_policy_is_empty():
    assert _policy() == []


def test_controlled_actor_policy_is_normalized_for_session_metadata():
    assert _policy(
        controlled_traversability_actor_id=167,
        controlled_traversability_disposition=' Obstacle ',
        controlled_traversability_blueprint='static.prop.motorhelmet',
    ) == [{
        'actor_id': 167,
        'disposition': 'obstacle',
        'blueprint': 'static.prop.motorhelmet',
        'policy_source': 'vehicle_clearance_policy',
    }]


def test_controlled_actor_policy_rejects_unregistered_disposition():
    with pytest.raises(ValueError):
        _policy(
            controlled_traversability_actor_id=167,
            controlled_traversability_disposition='probably_obstacle',
        )


def _cloud(stamp_s):
    seconds = int(stamp_s)
    nanoseconds = int(round((stamp_s - seconds) * 1.0e9))
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=seconds, nanosec=nanoseconds)
        )
    )


def test_perception_capture_waits_for_aligned_semantic_cloud():
    latest = {
        'cloud': _cloud(10.0),
        'semantic_cloud': _cloud(10.2),
        'odom': object(),
    }
    latest_wall = {'cloud': 4.9, 'semantic_cloud': 4.9, 'odom': 4.9}

    ready, reason = perception_capture_readiness(
        latest,
        latest_wall,
        now=5.0,
        maximum_cloud_age_s=0.7,
        maximum_odometry_age_s=0.7,
        require_semantic_cloud=True,
        maximum_semantic_alignment_s=0.03,
    )

    assert not ready
    assert reason == 'semantic_cloud_not_aligned'


def test_perception_capture_accepts_all_required_fresh_inputs():
    latest = {
        'cloud': _cloud(10.0),
        'semantic_cloud': _cloud(10.01),
        'odom': object(),
    }
    latest_wall = {'cloud': 4.9, 'semantic_cloud': 4.9, 'odom': 4.9}

    assert perception_capture_readiness(
        latest,
        latest_wall,
        now=5.0,
        maximum_cloud_age_s=0.7,
        maximum_odometry_age_s=0.7,
        require_semantic_cloud=True,
        maximum_semantic_alignment_s=0.03,
    ) == (True, 'ready')


def test_zero_sample_perception_capture_is_a_failure():
    assert perception_capture_completion_result(0) == (
        'perception_capture_failed_no_samples'
    )
    assert perception_capture_completion_result(1) == (
        'perception_capture_completed'
    )


def test_path_validity_recorder_uses_volatile_reliable_qos():
    qos = path_hard_valid_subscription_qos()

    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.VOLATILE
