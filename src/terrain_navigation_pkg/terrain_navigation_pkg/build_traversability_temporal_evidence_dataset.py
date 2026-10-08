#!/usr/bin/env python3
"""Build a non-destructive 17-channel temporal evidence archive.

Each output keeps the current single-scan eight channels, appends eight
channels built from an ego-motion-compensated unique-scan history, and adds a
distinct-scan support fraction.  Supervision is always generated from the
current semantic scan; historical semantic labels never become model input.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .build_traversability_dataset import (
    _atomic_json,
    _atomic_npz,
    _float_or_nan,
    _read_geometry,
    _sample_identifier,
)
from .build_traversability_evidence_dataset import (
    DEFAULT_RAW_ROOT,
    MANIFEST_FIELDS,
    _controlled_actor_policy,
    _empty_result,
    _process_sample,
    _safe_output_directory,
)
from .traversability_evidence_core import build_evidence_bev
from .traversability_learning_core import EgoFootprint
from .traversability_temporal_core import (
    build_ego_motion_compensated_lidar_bev,
    point_cloud_fingerprint,
    select_recent_unique_scan_indices,
)


DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_temporal_5scan'
)
SCHEMA_VERSION = 1
TEMPORAL_FIELDS = [
    'temporal_history_size', 'temporal_source_sample_ids',
    'temporal_window_span_s', 'temporal_window_yaw_span_deg',
    'input_channel_count',
]
OUTPUT_FIELDS = MANIFEST_FIELDS + TEMPORAL_FIELDS


def _angle_difference(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def _load_temporal_source(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(sample_path, allow_pickle=False) as arrays:
        required = {'lidar_points_xyz', 'vehicle_pose_odom_xyzyaw'}
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(
                'missing temporal arrays: ' + ','.join(sorted(missing))
            )
        points = np.asarray(arrays['lidar_points_xyz'], dtype=np.float32)
        pose = np.asarray(
            arrays['vehicle_pose_odom_xyzyaw'], dtype=np.float64
        )
    if points.ndim != 2 or points.shape[1] != 3 or not points.shape[0]:
        raise ValueError('lidar_points_xyz must have shape (N, 3)')
    if pose.shape != (4,) or not np.all(np.isfinite(pose)):
        raise ValueError('vehicle_pose_odom_xyzyaw must be finite shape (4,)')
    return points, pose


def _augment_temporal_sample(
    output_path: Path,
    points_by_scan: list[np.ndarray],
    poses: list[np.ndarray],
    fingerprints: list[str],
    source_sample_ids: list[int],
    target_index: int,
    history_size: int,
    geometry,
    *,
    sensor_pose_vehicle_xyzyaw: tuple[float, float, float, float],
    local_ground_radii_m: tuple[float, ...],
    local_ground_quantile: float,
    local_ground_minimum_support_cells: int,
) -> tuple[int, ...]:
    temporal = build_ego_motion_compensated_lidar_bev(
        points_by_scan,
        poses,
        target_index,
        geometry,
        history_size=history_size,
        fingerprints=fingerprints,
        sensor_pose_vehicle_xyzyaw=sensor_pose_vehicle_xyzyaw,
    )
    temporal_evidence, temporal_local = build_evidence_bev(
        temporal.lidar_bev,
        geometry,
        radii_m=local_ground_radii_m,
        ground_quantile=local_ground_quantile,
        minimum_support_cells=local_ground_minimum_support_cells,
    )
    support_fraction = (
        temporal.scan_support_count.astype(np.float32)
        / float(len(temporal.source_indices))
    )
    with np.load(output_path, allow_pickle=False) as arrays:
        derived = {key: arrays[key].copy() for key in arrays.files}
    current_evidence = np.asarray(
        derived['lidar_evidence_bev'], dtype=np.float32
    )
    if current_evidence.shape[0] != 8:
        raise ValueError('current evidence input must contain eight channels')
    derived.update({
        'current_lidar_evidence_bev': current_evidence.astype(np.float16),
        'temporal_lidar_bev': temporal.lidar_bev.astype(np.float16),
        'temporal_lidar_evidence_bev': temporal_evidence.astype(np.float16),
        'temporal_scan_support_fraction': support_fraction.astype(np.float16),
        'temporal_scan_support_count': temporal.scan_support_count,
        'temporal_source_sample_ids': np.asarray(
            [source_sample_ids[index] for index in temporal.source_indices],
            dtype=np.int64,
        ),
        'temporal_history_size': np.asarray(
            history_size, dtype=np.int16
        ),
        'temporal_sensor_pose_vehicle_xyzyaw': np.asarray(
            sensor_pose_vehicle_xyzyaw, dtype=np.float32
        ),
        'temporal_local_ground_relative_max_height_m': (
            temporal_local.relative_max_height.astype(np.float16)
        ),
        'lidar_evidence_bev': np.concatenate([
            current_evidence,
            temporal_evidence,
            support_fraction[None, ...],
        ], axis=0).astype(np.float16),
    })
    _atomic_npz(output_path, derived)
    return temporal.source_indices


def build_temporal_evidence_dataset(
    raw_root: Path,
    output_directory: Path,
    *,
    session_names: tuple[str, ...],
    history_size: int = 5,
    maximum_window_age_s: float = 1.25,
    maximum_window_yaw_deg: float = 5.0,
    sensor_pose_vehicle_xyzyaw: tuple[float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0,
    ),
    minimum_surface_points: int = 2,
    minimum_controlled_passable_points: int = 1,
    obstacle_vertical_span_m: float = 0.15,
    maximum_alignment_s: float = 0.03,
    ego_rear_m: float = 2.5,
    ego_front_m: float = 2.4,
    ego_half_width_m: float = 1.0,
    visibility_angular_bin_count: int = 720,
    local_ground_radii_m: tuple[float, ...] = (0.75, 1.50),
    local_ground_quantile: float = 0.25,
    local_ground_minimum_support_cells: int = 4,
) -> dict:
    """Build temporal samples from an explicit set of moving sessions."""

    raw_root = Path(raw_root).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if history_size < 2:
        raise ValueError('temporal history_size must be at least two')
    if maximum_window_age_s <= 0.0 or maximum_window_yaw_deg <= 0.0:
        raise ValueError('temporal window limits must be positive')
    requested = tuple(dict.fromkeys(session_names))
    if not requested:
        raise ValueError('explicit session_names are required')
    if not raw_root.is_dir():
        raise FileNotFoundError(
            'raw recording root not found: ' + str(raw_root)
        )
    _safe_output_directory(raw_root, output_directory)
    available = {
        path.name: path for path in raw_root.glob('session_*') if path.is_dir()
    }
    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(
            'requested sessions were not found: ' + ', '.join(missing)
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    footprint = EgoFootprint(
        rear_m=ego_rear_m,
        front_m=ego_front_m,
        half_width_m=ego_half_width_m,
    )

    rows_out: list[dict] = []
    written = skipped = 0
    for session_name in requested:
        session = available[session_name]
        metadata = json.loads(
            (session / 'metadata.json').read_text(encoding='utf-8')
        )
        geometry = _read_geometry(metadata)
        actor_dispositions, actor_policy = _controlled_actor_policy(metadata)
        with (session / 'frames.csv').open(
            newline='', encoding='utf-8'
        ) as stream:
            frame_rows = list(csv.DictReader(stream))
        points_by_scan: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        fingerprints: list[str] = []
        sample_ids: list[int] = []
        times: list[float] = []
        source_paths: list[Path] = []
        for row in frame_rows:
            source_path = session / row['file']
            points, pose = _load_temporal_source(source_path)
            points_by_scan.append(points)
            poses.append(pose)
            fingerprints.append(point_cloud_fingerprint(points))
            sample_ids.append(_sample_identifier(row, source_path))
            times.append(float(row['ros_time_s']))
            source_paths.append(source_path)

        emitted_fingerprints: dict[str, int] = {}
        for target_index, row in enumerate(frame_rows):
            source_path = source_paths[target_index]
            sample_id = sample_ids[target_index]
            output_path = (
                output_directory / 'samples' / session.name
                / ('sample_%06d.npz' % sample_id)
            )
            details = _empty_result(geometry)
            details['controlled_actor_count'] = len(actor_dispositions)
            selected = select_recent_unique_scan_indices(
                fingerprints, target_index, history_size
            )
            window_span_s = 0.0
            window_yaw_deg = 0.0
            temporal_ids = ''
            if len(selected) < history_size:
                details['reason'] = 'insufficient_unique_history'
            elif fingerprints[target_index] in emitted_fingerprints:
                details['reason'] = 'duplicate_identical_scan'
                details['scan_fingerprint'] = fingerprints[target_index]
                details['duplicate_of_source_sample_id'] = (
                    emitted_fingerprints[fingerprints[target_index]]
                )
            else:
                window_span_s = times[target_index] - times[selected[0]]
                window_yaw_deg = abs(math.degrees(_angle_difference(
                    poses[target_index][3], poses[selected[0]][3]
                )))
                if window_span_s > maximum_window_age_s:
                    details['reason'] = 'temporal_window_too_old'
                elif window_yaw_deg > maximum_window_yaw_deg:
                    details['reason'] = 'temporal_window_yaw_exceeds_limit'
                elif _float_or_nan(row.get('collision_event_count', 0)) > 0:
                    details['reason'] = 'collision_or_after'
                else:
                    details = _process_sample(
                        source_path,
                        output_path,
                        row,
                        geometry,
                        actor_dispositions=actor_dispositions,
                        actor_policy=actor_policy,
                        minimum_surface_points=minimum_surface_points,
                        minimum_controlled_passable_points=(
                            minimum_controlled_passable_points
                        ),
                        obstacle_vertical_span_m=obstacle_vertical_span_m,
                        maximum_alignment_s=maximum_alignment_s,
                        ego_footprint=footprint,
                        visibility_angular_bin_count=(
                            visibility_angular_bin_count
                        ),
                        local_ground_radii_m=local_ground_radii_m,
                        local_ground_quantile=local_ground_quantile,
                        local_ground_minimum_support_cells=(
                            local_ground_minimum_support_cells
                        ),
                    )
                    if details['status'] == 'written':
                        selected = _augment_temporal_sample(
                            output_path,
                            points_by_scan,
                            poses,
                            fingerprints,
                            sample_ids,
                            target_index,
                            history_size,
                            geometry,
                            sensor_pose_vehicle_xyzyaw=(
                                sensor_pose_vehicle_xyzyaw
                            ),
                            local_ground_radii_m=local_ground_radii_m,
                            local_ground_quantile=local_ground_quantile,
                            local_ground_minimum_support_cells=(
                                local_ground_minimum_support_cells
                            ),
                        )
                        temporal_ids = '|'.join(
                            str(sample_ids[index]) for index in selected
                        )
                        emitted_fingerprints[fingerprints[target_index]] = (
                            sample_id
                        )
            written += int(details['status'] == 'written')
            skipped += int(details['status'] != 'written')
            base = {
                'source_session': session.name,
                'source_session_result': metadata.get('result', ''),
                'source_sample_id': sample_id,
                'source_sample_path': str(source_path),
                'derived_sample_path': (
                    str(output_path) if details['status'] == 'written' else ''
                ),
                'status': details['status'],
                'reason': details['reason'],
                'ros_time_s': row.get('ros_time_s', ''),
                'route_index': row.get('route_index', ''),
                'route_status': row.get('route_status', ''),
                'semantic_alignment_delta_s': row.get(
                    'semantic_alignment_delta_s', ''
                ),
                'collision_event_count': row.get(
                    'collision_event_count', '0'
                ),
                **{
                    field: details[field]
                    for field in MANIFEST_FIELDS if field in details
                },
                'temporal_history_size': history_size,
                'temporal_source_sample_ids': temporal_ids,
                'temporal_window_span_s': window_span_s,
                'temporal_window_yaw_span_deg': window_yaw_deg,
                'input_channel_count': 17,
            }
            rows_out.append(base)

    manifest = output_directory / 'manifest.csv'
    temporary = manifest.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    temporary.replace(manifest)
    reason_counts: dict[str, int] = {}
    for row in rows_out:
        if row['reason']:
            reason_counts[row['reason']] = (
                reason_counts.get(row['reason'], 0) + 1
            )
    summary = {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'raw_root': str(raw_root),
        'output_directory': str(output_directory),
        'requested_sessions': list(requested),
        'input_variant': 'current_8_plus_temporal_8_plus_scan_support_1',
        'input_channel_count': 17,
        'history_size': history_size,
        'maximum_window_age_s': maximum_window_age_s,
        'maximum_window_yaw_deg': maximum_window_yaw_deg,
        'sensor_pose_vehicle_xyzyaw': list(sensor_pose_vehicle_xyzyaw),
        'target_policy': 'current semantic scan only',
        'semantic_labels_are_model_input': False,
        'written_samples': written,
        'skipped_samples': skipped,
        'skip_reasons': reason_counts,
        'manifest': str(manifest),
        'deployable': False,
    }
    _atomic_json(output_directory / 'summary.json', summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        '--output-directory', type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument('--session', action='append', required=True)
    parser.add_argument('--history-size', type=int, default=5)
    parser.add_argument('--maximum-window-age-s', type=float, default=1.25)
    parser.add_argument('--maximum-window-yaw-deg', type=float, default=5.0)
    parser.add_argument('--sensor-x-m', type=float, default=0.0)
    parser.add_argument('--sensor-y-m', type=float, default=0.0)
    parser.add_argument('--sensor-z-m', type=float, default=0.0)
    parser.add_argument('--sensor-yaw-deg', type=float, default=0.0)
    parser.add_argument('--minimum-surface-points', type=int, default=2)
    parser.add_argument(
        '--minimum-controlled-passable-points', type=int, default=1
    )
    parser.add_argument('--obstacle-vertical-span-m', type=float, default=0.15)
    parser.add_argument('--maximum-alignment-s', type=float, default=0.03)
    parser.add_argument('--ego-rear-m', type=float, default=2.5)
    parser.add_argument('--ego-front-m', type=float, default=2.4)
    parser.add_argument('--ego-half-width-m', type=float, default=1.0)
    parser.add_argument(
        '--visibility-angular-bin-count', type=int, default=720
    )
    parser.add_argument(
        '--local-ground-radii-m', type=float, nargs='+', default=(0.75, 1.50)
    )
    parser.add_argument('--local-ground-quantile', type=float, default=0.25)
    parser.add_argument(
        '--local-ground-minimum-support-cells', type=int, default=4
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = build_temporal_evidence_dataset(
        args.raw_root,
        args.output_directory,
        session_names=tuple(args.session),
        history_size=args.history_size,
        maximum_window_age_s=args.maximum_window_age_s,
        maximum_window_yaw_deg=args.maximum_window_yaw_deg,
        sensor_pose_vehicle_xyzyaw=(
            args.sensor_x_m,
            args.sensor_y_m,
            args.sensor_z_m,
            math.radians(args.sensor_yaw_deg),
        ),
        minimum_surface_points=args.minimum_surface_points,
        minimum_controlled_passable_points=(
            args.minimum_controlled_passable_points
        ),
        obstacle_vertical_span_m=args.obstacle_vertical_span_m,
        maximum_alignment_s=args.maximum_alignment_s,
        ego_rear_m=args.ego_rear_m,
        ego_front_m=args.ego_front_m,
        ego_half_width_m=args.ego_half_width_m,
        visibility_angular_bin_count=args.visibility_angular_bin_count,
        local_ground_radii_m=tuple(args.local_ground_radii_m),
        local_ground_quantile=args.local_ground_quantile,
        local_ground_minimum_support_cells=(
            args.local_ground_minimum_support_cells
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
