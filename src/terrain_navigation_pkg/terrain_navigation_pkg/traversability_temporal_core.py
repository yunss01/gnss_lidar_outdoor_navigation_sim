"""Ego-motion compensated LiDAR accumulation for temporal experiments.

The helpers in this module are deliberately independent from ROS and model
inference.  They turn a short history of vehicle-frame point clouds into one
current-frame BEV while rejecting byte-identical repeated scans.  This keeps
the first temporal experiment reproducible without granting it navigation
authority before scene-wise validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np

from .navigation_learning_recorder_core import BevGeometry, build_lidar_bev


@dataclass(frozen=True)
class TemporalLidarBev:
    """An aligned temporal point cloud and its rasterized evidence."""

    lidar_bev: np.ndarray
    scan_support_count: np.ndarray
    aligned_points_xyz: np.ndarray
    source_indices: tuple[int, ...]


def _points_array(points_xyz) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_xyz must have shape (N, 3)')
    return points


def _pose_array(vehicle_pose_xyzyaw) -> np.ndarray:
    pose = np.asarray(vehicle_pose_xyzyaw, dtype=np.float64)
    if pose.shape != (4,):
        raise ValueError('vehicle_pose_xyzyaw must have shape (4,)')
    if not np.all(np.isfinite(pose)):
        raise ValueError('vehicle pose must contain only finite values')
    return pose


def vehicle_points_to_world(
    points_xyz,
    vehicle_pose_xyzyaw,
) -> np.ndarray:
    """Transform x-forward/y-left vehicle points into the ROS odom frame."""

    points = _points_array(points_xyz)
    pose = _pose_array(vehicle_pose_xyzyaw)
    if not points.size:
        return points.copy()
    cosine = math.cos(float(pose[3]))
    sine = math.sin(float(pose[3]))
    result = np.empty_like(points, dtype=np.float32)
    result[:, 0] = (
        float(pose[0]) + cosine * points[:, 0] - sine * points[:, 1]
    )
    result[:, 1] = (
        float(pose[1]) + sine * points[:, 0] + cosine * points[:, 1]
    )
    result[:, 2] = float(pose[2]) + points[:, 2]
    return result


def world_points_to_vehicle(
    points_xyz,
    vehicle_pose_xyzyaw,
) -> np.ndarray:
    """Transform ROS odom-frame points into x-forward/y-left coordinates."""

    points = _points_array(points_xyz)
    pose = _pose_array(vehicle_pose_xyzyaw)
    if not points.size:
        return points.copy()
    dx = points[:, 0] - float(pose[0])
    dy = points[:, 1] - float(pose[1])
    cosine = math.cos(float(pose[3]))
    sine = math.sin(float(pose[3]))
    result = np.empty_like(points, dtype=np.float32)
    result[:, 0] = cosine * dx + sine * dy
    result[:, 1] = -sine * dx + cosine * dy
    result[:, 2] = points[:, 2] - float(pose[2])
    return result


def transform_vehicle_points(
    points_xyz,
    source_vehicle_pose_xyzyaw,
    target_vehicle_pose_xyzyaw,
) -> np.ndarray:
    """Express source-scan points in a target vehicle's coordinate frame."""

    return world_points_to_vehicle(
        vehicle_points_to_world(points_xyz, source_vehicle_pose_xyzyaw),
        target_vehicle_pose_xyzyaw,
    )


def compose_planar_sensor_pose(
    vehicle_pose_xyzyaw,
    sensor_pose_vehicle_xyzyaw,
) -> np.ndarray:
    """Compose a planar vehicle pose with a calibrated sensor extrinsic."""

    vehicle = _pose_array(vehicle_pose_xyzyaw)
    sensor = _pose_array(sensor_pose_vehicle_xyzyaw)
    cosine = math.cos(float(vehicle[3]))
    sine = math.sin(float(vehicle[3]))
    return np.asarray([
        vehicle[0] + cosine * sensor[0] - sine * sensor[1],
        vehicle[1] + sine * sensor[0] + cosine * sensor[1],
        vehicle[2] + sensor[2],
        vehicle[3] + sensor[3],
    ], dtype=np.float64)


def transform_sensor_points(
    points_xyz,
    source_vehicle_pose_xyzyaw,
    target_vehicle_pose_xyzyaw,
    sensor_pose_vehicle_xyzyaw=(0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Express source LiDAR-frame points in the target LiDAR frame."""

    sensor_pose = _pose_array(sensor_pose_vehicle_xyzyaw)
    source_sensor_pose = compose_planar_sensor_pose(
        source_vehicle_pose_xyzyaw, sensor_pose
    )
    target_sensor_pose = compose_planar_sensor_pose(
        target_vehicle_pose_xyzyaw, sensor_pose
    )
    return transform_vehicle_points(
        points_xyz, source_sensor_pose, target_sensor_pose
    )


def point_cloud_fingerprint(points_xyz) -> str:
    """Return a stable identity for exact repeated-scan rejection."""

    points = np.ascontiguousarray(_points_array(points_xyz))
    digest = hashlib.sha256()
    digest.update(str(points.shape).encode('ascii'))
    digest.update(points.tobytes())
    return digest.hexdigest()


def select_recent_unique_scan_indices(
    fingerprints: Sequence[str],
    target_index: int,
    history_size: int,
) -> tuple[int, ...]:
    """Select up to ``history_size`` distinct scans ending at target_index."""

    if history_size < 1:
        raise ValueError('history_size must be positive')
    if target_index < 0 or target_index >= len(fingerprints):
        raise IndexError('target_index is outside the scan sequence')
    selected: list[int] = []
    seen: set[str] = set()
    for index in range(target_index, -1, -1):
        fingerprint = str(fingerprints[index])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(index)
        if len(selected) >= history_size:
            break
    return tuple(reversed(selected))


def build_ego_motion_compensated_lidar_bev(
    points_by_scan: Sequence[np.ndarray],
    vehicle_poses_xyzyaw: Sequence[np.ndarray],
    target_index: int,
    geometry: BevGeometry,
    *,
    history_size: int = 5,
    fingerprints: Sequence[str] | None = None,
    sensor_pose_vehicle_xyzyaw=(0.0, 0.0, 0.0, 0.0),
) -> TemporalLidarBev:
    """Align recent unique scans to ``target_index`` and build one BEV.

    ``scan_support_count`` counts distinct source scans per occupied cell.  It
    is intentionally returned separately from the standard four-channel BEV
    so future models can distinguish persistent support from raw point count.
    """

    if len(points_by_scan) != len(vehicle_poses_xyzyaw):
        raise ValueError('point and pose sequence lengths must match')
    if not points_by_scan:
        raise ValueError('scan sequence cannot be empty')
    if fingerprints is None:
        identities = [
            point_cloud_fingerprint(value) for value in points_by_scan
        ]
    else:
        if len(fingerprints) != len(points_by_scan):
            raise ValueError('fingerprint sequence length must match scans')
        identities = list(fingerprints)
    indices = select_recent_unique_scan_indices(
        identities, target_index, history_size
    )
    target_pose = _pose_array(vehicle_poses_xyzyaw[target_index])
    aligned: list[np.ndarray] = []
    support = np.zeros(
        (geometry.height, geometry.width), dtype=np.uint16
    )
    for index in indices:
        transformed = transform_sensor_points(
            points_by_scan[index],
            vehicle_poses_xyzyaw[index],
            target_pose,
            sensor_pose_vehicle_xyzyaw,
        )
        aligned.append(transformed)
        support += (
            build_lidar_bev(transformed, geometry)[0] > 0.5
        ).astype(np.uint16)
    combined = (
        np.concatenate(aligned, axis=0)
        if aligned else np.empty((0, 3), dtype=np.float32)
    )
    return TemporalLidarBev(
        lidar_bev=build_lidar_bev(combined, geometry),
        scan_support_count=support,
        aligned_points_xyz=combined,
        source_indices=indices,
    )
