import math

import numpy as np
import pytest

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
)
from terrain_navigation_pkg.traversability_temporal_core import (
    build_ego_motion_compensated_lidar_bev,
    point_cloud_fingerprint,
    select_recent_unique_scan_indices,
    transform_sensor_points,
    transform_vehicle_points,
    vehicle_points_to_world,
    world_points_to_vehicle,
)


def _geometry():
    return BevGeometry(
        x_min_m=0.0,
        x_max_m=10.0,
        y_min_m=-5.0,
        y_max_m=5.0,
        resolution_m=0.5,
        z_min_m=-2.0,
        z_max_m=2.0,
    )


def test_vehicle_world_round_trip_preserves_points():
    points = np.asarray([[3.0, -0.4, -1.2], [5.0, 1.1, 0.2]])
    pose = np.asarray([10.0, -4.0, 0.5, 0.7])
    world = vehicle_points_to_world(points, pose)
    recovered = world_points_to_vehicle(world, pose)
    assert recovered == pytest.approx(points, abs=1e-5)


def test_transform_accounts_for_translation_and_rotation():
    point = np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32)
    translated = transform_vehicle_points(
        point,
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(translated, [[4.0, 0.0, 0.0]])

    rotated = transform_vehicle_points(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, math.pi / 2.0]),
    )
    np.testing.assert_allclose(
        rotated, [[0.0, -1.0, 0.0]], atol=1e-6
    )


def test_sensor_extrinsic_is_composed_during_turn():
    transformed = transform_sensor_points(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, math.pi / 2.0]),
        sensor_pose_vehicle_xyzyaw=np.asarray([1.0, 0.0, 1.5, 0.0]),
    )
    np.testing.assert_allclose(
        transformed, [[-1.0, -2.0, 0.0]], atol=1e-6
    )


def test_unique_scan_selection_skips_repeated_recorder_samples():
    selected = select_recent_unique_scan_indices(
        ['scan-a', 'scan-a', 'scan-b', 'scan-c'], 3, 3
    )
    # The newest copy of an identical scan is closest to the target pose.
    assert selected == (1, 2, 3)


def test_temporal_bev_aligns_static_point_and_counts_scan_support():
    # The same world point appears at x=5 and x=4 after the vehicle advances.
    points = [
        np.asarray([[5.0, 0.0, -1.0]], dtype=np.float32),
        np.asarray([[4.0, 0.0, -1.0]], dtype=np.float32),
    ]
    poses = [
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    ]
    result = build_ego_motion_compensated_lidar_bev(
        points, poses, 1, _geometry(), history_size=2
    )
    assert result.source_indices == (0, 1)
    occupied = np.argwhere(result.lidar_bev[0] > 0.5)
    assert occupied.shape[0] == 1
    row, column = occupied[0]
    assert result.scan_support_count[row, column] == 2
    assert result.aligned_points_xyz[:, 0] == pytest.approx([4.0, 4.0])


def test_temporal_bev_deduplicates_identical_scan_fingerprints():
    scan = np.asarray([[4.0, 0.0, -1.0]], dtype=np.float32)
    points = [scan, scan.copy()]
    poses = [np.zeros(4), np.zeros(4)]
    identity = point_cloud_fingerprint(scan)
    result = build_ego_motion_compensated_lidar_bev(
        points,
        poses,
        1,
        _geometry(),
        history_size=2,
        fingerprints=[identity, identity],
    )
    assert result.source_indices == (1,)
    assert result.aligned_points_xyz.shape == (1, 3)
