"""Evaluate a matched moving control/object temporal LiDAR pilot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .navigation_learning_recorder_core import BevGeometry, build_lidar_bev
from .traversability_temporal_core import (
    build_ego_motion_compensated_lidar_bev,
    point_cloud_fingerprint,
    select_recent_unique_scan_indices,
    transform_vehicle_points,
    vehicle_points_to_world,
)


@dataclass(frozen=True)
class SessionScans:
    path: Path
    metadata: dict
    sample_ids: np.ndarray
    ros_times_s: np.ndarray
    speeds_mps: np.ndarray
    poses_xyzyaw: np.ndarray
    points: tuple[np.ndarray, ...]
    semantic_points: tuple[np.ndarray, ...]
    semantic_object_ids: tuple[np.ndarray, ...]
    fingerprints: tuple[str, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate ego-motion-compensated 3/5-scan evidence on a '
            'matched moving control/object pair.'
        )
    )
    parser.add_argument('--control-session', required=True)
    parser.add_argument('--object-session', required=True)
    parser.add_argument('--actor-id', type=int, required=True)
    parser.add_argument('--history-sizes', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--output', required=True)
    return parser


def _geometry(metadata: dict) -> BevGeometry:
    bev = metadata['bev']
    return BevGeometry(
        x_min_m=float(bev['x_min_m']),
        x_max_m=float(bev['x_max_m']),
        y_min_m=float(bev['y_min_m']),
        y_max_m=float(bev['y_max_m']),
        resolution_m=float(bev['resolution_m']),
        z_min_m=float(bev['z_min_m']),
        z_max_m=float(bev['z_max_m']),
    )


def _load_session(path_value: str) -> SessionScans:
    path = Path(path_value).expanduser().resolve()
    metadata_path = path / 'metadata.json'
    frames_path = path / 'frames.csv'
    if not metadata_path.is_file() or not frames_path.is_file():
        raise FileNotFoundError('incomplete learning session: ' + str(path))
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    with frames_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('session contains no recorded frames: ' + str(path))

    sample_ids: list[int] = []
    times: list[float] = []
    speeds: list[float] = []
    poses: list[np.ndarray] = []
    points: list[np.ndarray] = []
    semantic_points: list[np.ndarray] = []
    semantic_ids: list[np.ndarray] = []
    fingerprints: list[str] = []
    for row in rows:
        sample_path = path / row['file']
        with np.load(sample_path, allow_pickle=False) as arrays:
            required = {
                'lidar_points_xyz', 'semantic_lidar_points_xyz',
                'semantic_lidar_object_idx', 'vehicle_pose_odom_xyzyaw',
            }
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(
                    f'{sample_path} is missing arrays: '
                    + ', '.join(sorted(missing))
                )
            raw = np.asarray(arrays['lidar_points_xyz'], dtype=np.float32)
            semantic = np.asarray(
                arrays['semantic_lidar_points_xyz'], dtype=np.float32
            )
            object_ids = np.asarray(
                arrays['semantic_lidar_object_idx'], dtype=np.uint32
            )
            pose = np.asarray(
                arrays['vehicle_pose_odom_xyzyaw'], dtype=np.float64
            )
        sample_ids.append(int(row['sample_id']))
        times.append(float(row['ros_time_s']))
        speeds.append(float(row['speed_mps']))
        poses.append(pose)
        points.append(raw)
        semantic_points.append(semantic)
        semantic_ids.append(object_ids)
        fingerprints.append(point_cloud_fingerprint(raw))
    return SessionScans(
        path=path,
        metadata=metadata,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        ros_times_s=np.asarray(times, dtype=np.float64),
        speeds_mps=np.asarray(speeds, dtype=np.float64),
        poses_xyzyaw=np.stack(poses, axis=0),
        points=tuple(points),
        semantic_points=tuple(semantic_points),
        semantic_object_ids=tuple(semantic_ids),
        fingerprints=tuple(fingerprints),
    )


def _trajectory_distance(poses: np.ndarray) -> np.ndarray:
    increments = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    return np.concatenate([np.zeros(1), np.cumsum(increments)])


def _angle_difference(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def _distribution(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            'count': 0, 'min': None, 'median': None,
            'p95': None, 'max': None,
        }
    return {
        'count': int(array.size),
        'min': float(np.min(array)),
        'median': float(np.median(array)),
        'p95': float(np.quantile(array, 0.95)),
        'max': float(np.max(array)),
    }


def _session_motion(session: SessionScans) -> dict:
    distance = _trajectory_distance(session.poses_xyzyaw)
    moving = np.flatnonzero(session.speeds_mps >= 0.5)
    span_s = 0.0
    span_m = 0.0
    if moving.size:
        span_s = float(
            session.ros_times_s[moving[-1]] - session.ros_times_s[moving[0]]
        )
        span_m = float(distance[moving[-1]] - distance[moving[0]])
    return {
        'frame_count': int(session.sample_ids.size),
        'unique_scan_count': int(len(set(session.fingerprints))),
        'path_distance_m': float(distance[-1]),
        'displacement_m': float(np.linalg.norm(
            session.poses_xyzyaw[-1, :2] - session.poses_xyzyaw[0, :2]
        )),
        'maximum_speed_mps': float(np.max(session.speeds_mps)),
        'moving_frame_count': int(moving.size),
        'moving_span_s': span_s,
        'moving_span_m': span_m,
        'yaw_change_deg': float(math.degrees(_angle_difference(
            session.poses_xyzyaw[-1, 3],
            session.poses_xyzyaw[0, 3],
        ))),
    }


def _pair_alignment(control: SessionScans, obstacle: SessionScans) -> dict:
    control_distance = _trajectory_distance(control.poses_xyzyaw)
    obstacle_distance = _trajectory_distance(obstacle.poses_xyzyaw)
    common = min(float(control_distance[-1]), float(obstacle_distance[-1]))
    position_errors: list[float] = []
    yaw_errors: list[float] = []
    for index, distance in enumerate(obstacle_distance):
        if distance > common:
            continue
        control_index = int(np.argmin(np.abs(control_distance - distance)))
        position_errors.append(float(np.linalg.norm(
            obstacle.poses_xyzyaw[index, :2]
            - control.poses_xyzyaw[control_index, :2]
        )))
        yaw_errors.append(abs(math.degrees(_angle_difference(
            obstacle.poses_xyzyaw[index, 3],
            control.poses_xyzyaw[control_index, 3],
        ))))
    return {
        'start_position_error_m': float(np.linalg.norm(
            obstacle.poses_xyzyaw[0, :2] - control.poses_xyzyaw[0, :2]
        )),
        'start_yaw_error_deg': abs(math.degrees(_angle_difference(
            obstacle.poses_xyzyaw[0, 3], control.poses_xyzyaw[0, 3]
        ))),
        'common_path_distance_m': common,
        'distance_matched_position_error_m': _distribution(position_errors),
        'distance_matched_yaw_error_deg': _distribution(yaw_errors),
    }


def _actor_world_points(
    session: SessionScans,
    actor_id: int,
) -> np.ndarray:
    result: list[np.ndarray] = []
    for points, ids, pose in zip(
        session.semantic_points,
        session.semantic_object_ids,
        session.poses_xyzyaw,
    ):
        selected = points[ids.astype(np.int64) == int(actor_id)]
        if selected.size:
            result.append(vehicle_points_to_world(selected, pose))
    return (
        np.concatenate(result, axis=0)
        if result else np.empty((0, 3), dtype=np.float32)
    )


def _actor_temporal_metrics(
    session: SessionScans,
    actor_id: int,
    history_size: int,
    geometry: BevGeometry,
) -> dict:
    current_counts: list[int] = []
    accumulated_counts: list[int] = []
    current_cells: list[int] = []
    accumulated_cells: list[int] = []
    raw_point_multipliers: list[float] = []
    complete_windows = 0
    for target_index in range(session.sample_ids.size):
        selected = select_recent_unique_scan_indices(
            session.fingerprints, target_index, history_size
        )
        if len(selected) != history_size:
            continue
        complete_windows += 1
        target_pose = session.poses_xyzyaw[target_index]
        actor_history: list[np.ndarray] = []
        for source_index in selected:
            mask = (
                session.semantic_object_ids[source_index].astype(np.int64)
                == int(actor_id)
            )
            actor_history.append(transform_vehicle_points(
                session.semantic_points[source_index][mask],
                session.poses_xyzyaw[source_index],
                target_pose,
            ))
        accumulated = np.concatenate(actor_history, axis=0)
        current = actor_history[-1]
        if not current.size:
            continue
        current_counts.append(int(current.shape[0]))
        accumulated_counts.append(int(accumulated.shape[0]))
        current_cells.append(int(np.count_nonzero(
            build_lidar_bev(current, geometry)[0] > 0.5
        )))
        accumulated_cells.append(int(np.count_nonzero(
            build_lidar_bev(accumulated, geometry)[0] > 0.5
        )))
        temporal = build_ego_motion_compensated_lidar_bev(
            session.points,
            session.poses_xyzyaw,
            target_index,
            geometry,
            history_size=history_size,
            fingerprints=session.fingerprints,
        )
        current_raw_count = max(1, session.points[target_index].shape[0])
        raw_point_multipliers.append(
            temporal.aligned_points_xyz.shape[0] / current_raw_count
        )
    current_distribution = _distribution(current_counts)
    accumulated_distribution = _distribution(accumulated_counts)
    median_gain = None
    if current_distribution['median'] and accumulated_distribution['median']:
        median_gain = float(
            accumulated_distribution['median'] / current_distribution['median']
        )
    return {
        'history_size': int(history_size),
        'complete_window_count': int(complete_windows),
        'actor_current_point_count': current_distribution,
        'actor_accumulated_point_count': accumulated_distribution,
        'actor_median_point_gain': median_gain,
        'actor_current_occupied_cell_count': _distribution(current_cells),
        'actor_accumulated_occupied_cell_count': _distribution(
            accumulated_cells
        ),
        'raw_aligned_point_multiplier': _distribution(raw_point_multipliers),
    }


def evaluate(
    control: SessionScans,
    obstacle: SessionScans,
    actor_id: int,
    history_sizes: list[int],
) -> dict:
    if any(value < 1 for value in history_sizes):
        raise ValueError('history sizes must be positive')
    control_geometry = _geometry(control.metadata)
    obstacle_geometry = _geometry(obstacle.metadata)
    if control_geometry != obstacle_geometry:
        raise ValueError(
            'control and object sessions use different BEV geometry'
        )
    actor_world = _actor_world_points(obstacle, actor_id)
    if not actor_world.size:
        raise ValueError(f'actor {actor_id} has no semantic LiDAR returns')
    actor_frame_counts = [
        int(np.count_nonzero(ids.astype(np.int64) == int(actor_id)))
        for ids in obstacle.semantic_object_ids
    ]
    centroid = np.median(actor_world, axis=0)
    radial_error = np.linalg.norm(actor_world[:, :2] - centroid[:2], axis=1)
    pair = _pair_alignment(control, obstacle)
    temporal = {
        str(size): _actor_temporal_metrics(
            obstacle, actor_id, size, obstacle_geometry
        )
        for size in sorted(set(history_sizes))
    }
    largest = temporal[str(max(history_sizes))]
    gain = largest['actor_median_point_gain'] or 0.0
    visible_ratio = float(
        np.count_nonzero(np.asarray(actor_frame_counts) > 0)
        / len(actor_frame_counts)
    )
    gates = {
        'start_position_error_at_most_0_10m': (
            pair['start_position_error_m'] <= 0.10
        ),
        'start_yaw_error_at_most_1deg': pair['start_yaw_error_deg'] <= 1.0,
        'common_path_at_least_3m': pair['common_path_distance_m'] >= 3.0,
        'actor_visible_in_at_least_90pct_frames': visible_ratio >= 0.90,
        'largest_window_median_point_gain_at_least_2x': gain >= 2.0,
    }
    qualified = all(gates.values())
    return {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'ego-motion compensated temporal LiDAR pilot',
        'deployment_authority': False,
        'semantic_actor_ids_used_for_evaluation_only': True,
        'control_session': str(control.path),
        'object_session': str(obstacle.path),
        'actor_id': int(actor_id),
        'control_motion': _session_motion(control),
        'object_motion': _session_motion(obstacle),
        'pair_alignment': pair,
        'actor_observation': {
            'visible_frame_count': int(np.count_nonzero(
                np.asarray(actor_frame_counts) > 0
            )),
            'frame_count': len(actor_frame_counts),
            'visible_frame_ratio': visible_ratio,
            'point_count_per_frame': _distribution(actor_frame_counts),
            'world_centroid_xyz_m': centroid.astype(float).tolist(),
            'world_radial_error_m': _distribution(radial_error),
        },
        'temporal_windows': temporal,
        'qualification_gates': gates,
        'qualified_for_representation_experiment': qualified,
        'recommended_next_step': (
            'freeze this pair as a pilot fixture and extend the dataset '
            'builder/model input with temporal evidence; do not collect a '
            'large temporal dataset until the representation test passes'
            if qualified else
            'repeat the matched control/object capture before changing '
            'the model'
        ),
    }


def main(args=None) -> int:
    options = _parser().parse_args(args)
    control = _load_session(options.control_session)
    obstacle = _load_session(options.object_session)
    result = evaluate(
        control, obstacle, options.actor_id, options.history_sizes
    )
    output = Path(options.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    largest = result['temporal_windows'][str(max(options.history_sizes))]
    print('temporal pilot: ' + ('PASS' if result[
        'qualified_for_representation_experiment'
    ] else 'FAIL'))
    print('start position error: {:.3f} m'.format(
        result['pair_alignment']['start_position_error_m']
    ))
    print('common path: {:.3f} m'.format(
        result['pair_alignment']['common_path_distance_m']
    ))
    print('actor visible: {}/{} frames'.format(
        result['actor_observation']['visible_frame_count'],
        result['actor_observation']['frame_count'],
    ))
    print('{}-scan actor point gain: {:.3f}x'.format(
        max(options.history_sizes), largest['actor_median_point_gain']
    ))
    print('report: ' + str(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
