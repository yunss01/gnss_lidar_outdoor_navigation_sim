#!/usr/bin/env python3
"""Build LiDAR-BEV traversability pairs from privileged CARLA labels.

Raw recorder sessions are never modified.  The derived archive contains the
geometric LiDAR BEV used as the model input and a free/obstacle/unknown target
map generated from the synchronized semantic LiDAR scan.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .navigation_learning_recorder_core import BevGeometry
from .traversability_learning_core import (
    EgoFootprint,
    FREE_LABEL,
    OBSTACLE_LABEL,
    UNKNOWN_LABEL,
    build_conservative_visibility_evidence,
    build_semantic_traversability_targets,
)


DEFAULT_RAW_ROOT = Path('/home/sukja/terrain_nav_data/learning/raw')
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/v1'
)
SCHEMA_VERSION = 2

MANIFEST_FIELDS = [
    'source_session', 'source_session_result', 'source_sample_id',
    'source_sample_path', 'derived_sample_path', 'status', 'reason',
    'ros_time_s',
    'route_index', 'route_status', 'semantic_alignment_delta_s',
    'collision_event_count',
    'semantic_point_count', 'known_cell_count', 'free_cell_count',
    'obstacle_cell_count', 'unknown_cell_count', 'known_fraction',
    'ego_excluded_cell_count', 'visibility_free_cell_count',
]


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _read_geometry(metadata: dict) -> BevGeometry:
    values = metadata.get('bev')
    if not isinstance(values, dict):
        raise ValueError('metadata does not contain BEV geometry')
    return BevGeometry(
        x_min_m=float(values['x_min_m']),
        x_max_m=float(values['x_max_m']),
        y_min_m=float(values['y_min_m']),
        y_max_m=float(values['y_max_m']),
        resolution_m=float(values['resolution_m']),
        z_min_m=float(values['z_min_m']),
        z_max_m=float(values['z_max_m']),
    )


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _sample_identifier(row: dict, source_path: Path) -> int:
    try:
        return int(row.get('sample_id', ''))
    except (TypeError, ValueError):
        try:
            return int(source_path.stem.rsplit('_', 1)[-1])
        except ValueError:
            return -1


def _copy_if_present(source, output: dict, key: str) -> None:
    if key in source:
        output[key] = source[key]


def _empty_result(geometry: BevGeometry, reason: str = '') -> dict:
    return {
        'status': 'skipped',
        'reason': reason,
        'semantic_point_count': 0,
        'known_cell_count': 0,
        'free_cell_count': 0,
        'obstacle_cell_count': 0,
        'unknown_cell_count': geometry.height * geometry.width,
        'known_fraction': 0.0,
        'ego_excluded_cell_count': 0,
        'visibility_free_cell_count': 0,
    }


def _process_sample(
    source_path: Path,
    output_path: Path,
    row: dict,
    geometry: BevGeometry,
    minimum_surface_points: int,
    obstacle_vertical_span_m: float,
    maximum_alignment_s: float,
    ego_footprint: EgoFootprint,
    visibility_angular_bin_count: int,
) -> dict:
    result = _empty_result(geometry)
    if not source_path.is_file():
        result['reason'] = 'missing_source_sample'
        return result

    try:
        with np.load(source_path, allow_pickle=False) as source:
            required = {
                'lidar_bev', 'semantic_lidar_points_xyz',
                'semantic_lidar_object_tag',
            }
            missing = sorted(required.difference(source.files))
            if missing:
                result['reason'] = 'missing_arrays:' + ','.join(missing)
                return result

            lidar_bev = np.asarray(source['lidar_bev'])
            expected_shape = (4, geometry.height, geometry.width)
            if lidar_bev.shape != expected_shape:
                result['reason'] = (
                    'invalid_lidar_bev_shape:' + str(lidar_bev.shape)
                )
                return result

            xyz = np.asarray(source['semantic_lidar_points_xyz'])
            tags = np.asarray(source['semantic_lidar_object_tag'])
            if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
                result['reason'] = 'missing_semantic_points'
                return result
            if tags.shape != (xyz.shape[0],):
                result['reason'] = 'semantic_tag_shape_mismatch'
                return result

            alignment = _float_or_nan(
                source['semantic_alignment_delta_s'].reshape(-1)[0]
                if 'semantic_alignment_delta_s' in source
                and source['semantic_alignment_delta_s'].size
                else row.get('semantic_alignment_delta_s')
            )
            if not math.isfinite(alignment):
                result['reason'] = 'missing_semantic_alignment'
                return result
            if alignment > maximum_alignment_s:
                result['reason'] = 'semantic_alignment_exceeds_limit'
                return result

            targets = build_semantic_traversability_targets(
                xyz,
                tags,
                geometry,
                minimum_surface_points=minimum_surface_points,
                obstacle_vertical_span_m=obstacle_vertical_span_m,
                ego_footprint=ego_footprint,
            )
            visibility = build_conservative_visibility_evidence(
                xyz,
                geometry,
                targets,
                ego_footprint=ego_footprint,
                angular_bin_count=visibility_angular_bin_count,
            )
            labels = targets.labels
            known = int(np.count_nonzero(labels != UNKNOWN_LABEL))
            free = int(np.count_nonzero(labels == FREE_LABEL))
            obstacle = int(np.count_nonzero(labels == OBSTACLE_LABEL))
            unknown = int(np.count_nonzero(labels == UNKNOWN_LABEL))
            if known == 0:
                result['reason'] = 'no_known_target_cells'
                return result

            derived = {
                'lidar_bev': lidar_bev.astype(np.float16, copy=False),
                'target_labels': labels,
                'target_free_point_count': targets.free_point_count,
                'target_obstacle_point_count': targets.obstacle_point_count,
                'target_observed_point_count': targets.observed_point_count,
                'target_vertical_span_m': targets.vertical_span_m,
                'target_ego_exclusion_mask': targets.ego_exclusion_mask,
                'visibility_free_mask': visibility.free_mask,
                'visibility_ray_count': visibility.ray_count,
                'semantic_alignment_delta_s': np.asarray(
                    alignment, dtype=np.float32
                ),
                'source_sample_id': np.asarray(
                    _sample_identifier(row, source_path), dtype=np.int64
                ),
            }
            for key in (
                'goal_vehicle_xyz', 'vehicle_pose_odom_xyzyaw',
                'vehicle_twist_xyz_rpy', 'imu_linear_acceleration_xyz',
                'imu_angular_velocity_xyz', 'imu_orientation_xyzw',
            ):
                _copy_if_present(source, derived, key)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_npz(output_path, derived)
    except (OSError, ValueError, KeyError, IndexError) as error:
        result['reason'] = '{}:{}'.format(type(error).__name__, error)
        return result

    result.update({
        'status': 'written',
        'reason': '',
        'semantic_point_count': int(xyz.shape[0]),
        'known_cell_count': known,
        'free_cell_count': free,
        'obstacle_cell_count': obstacle,
        'unknown_cell_count': unknown,
        'known_fraction': known / float(labels.size),
        'ego_excluded_cell_count': int(np.count_nonzero(
            targets.ego_exclusion_mask
        )),
        'visibility_free_cell_count': int(np.count_nonzero(
            visibility.free_mask
        )),
    })
    return result


def build_dataset(
    raw_root: Path,
    output_directory: Path,
    *,
    minimum_surface_points: int = 2,
    obstacle_vertical_span_m: float = 0.15,
    maximum_alignment_s: float = 0.03,
    ego_rear_m: float = 2.5,
    ego_front_m: float = 2.4,
    ego_half_width_m: float = 1.0,
    visibility_angular_bin_count: int = 720,
    exclude_collision_and_after: bool = True,
) -> dict:
    """Build an auditable, non-destructive derived dataset."""
    raw_root = Path(raw_root).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(
            'raw recording root not found: ' + str(raw_root)
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    ego_footprint = EgoFootprint(
        rear_m=ego_rear_m,
        front_m=ego_front_m,
        half_width_m=ego_half_width_m,
    )

    manifest_rows = []
    written = skipped = 0
    session_count = semantic_session_count = 0
    for session in sorted(raw_root.glob('session_*')):
        metadata_path = session / 'metadata.json'
        frames_path = session / 'frames.csv'
        if not metadata_path.is_file() or not frames_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            geometry = _read_geometry(metadata)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        session_count += 1
        supervision = metadata.get('privileged_supervision', {})
        if not bool(supervision.get('semantic_labels_saved', False)):
            continue
        semantic_session_count += 1
        with frames_path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                file_name = str(row.get('file', '')).strip()
                if not file_name:
                    continue
                source_path = session / file_name
                sample_id = _sample_identifier(row, source_path)
                derived_path = (
                    output_directory / 'samples' / session.name
                    / ('sample_%06d.npz' % sample_id)
                )
                collision_count = _float_or_nan(
                    row.get('collision_event_count', 0)
                )
                if (
                    exclude_collision_and_after
                    and math.isfinite(collision_count)
                    and collision_count > 0.0
                ):
                    details = _empty_result(
                        geometry, 'collision_or_after'
                    )
                else:
                    details = _process_sample(
                        source_path,
                        derived_path,
                        row,
                        geometry,
                        minimum_surface_points,
                        obstacle_vertical_span_m,
                        maximum_alignment_s,
                        ego_footprint,
                        visibility_angular_bin_count,
                    )
                if details['status'] == 'written':
                    written += 1
                else:
                    skipped += 1
                manifest_rows.append({
                    'source_session': session.name,
                    'source_session_result': metadata.get('result', ''),
                    'source_sample_id': sample_id,
                    'source_sample_path': str(source_path),
                    'derived_sample_path': (
                        str(derived_path)
                        if details['status'] == 'written' else ''
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
                    'semantic_point_count': details['semantic_point_count'],
                    'known_cell_count': details['known_cell_count'],
                    'free_cell_count': details['free_cell_count'],
                    'obstacle_cell_count': details['obstacle_cell_count'],
                    'unknown_cell_count': details['unknown_cell_count'],
                    'known_fraction': '%.8f' % details['known_fraction'],
                    'ego_excluded_cell_count': details[
                        'ego_excluded_cell_count'
                    ],
                    'visibility_free_cell_count': details[
                        'visibility_free_cell_count'
                    ],
                })

    manifest_path = output_directory / 'manifest.csv'
    temporary_manifest = manifest_path.with_suffix('.csv.tmp')
    with temporary_manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(manifest_path)

    reason_counts = {}
    for row in manifest_rows:
        reason = row['reason']
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'raw_root': str(raw_root),
        'output_directory': str(output_directory),
        'model_inputs': ['lidar_bev'],
        'privileged_training_target_source': 'CARLA semantic LiDAR',
        'target_classes': {
            '-1': 'unknown/ignored', '0': 'free', '1': 'obstacle',
        },
        'surface_policy': (
            'road, sidewalk/paving, terrain/grass, road lines, and ground '
            'are equally free when no obstacle geometry is present'
        ),
        'parameters': {
            'minimum_surface_points': minimum_surface_points,
            'obstacle_vertical_span_m': obstacle_vertical_span_m,
            'maximum_alignment_s': maximum_alignment_s,
            'ego_rear_m': ego_rear_m,
            'ego_front_m': ego_front_m,
            'ego_half_width_m': ego_half_width_m,
            'visibility_angular_bin_count': visibility_angular_bin_count,
            'exclude_collision_and_after': exclude_collision_and_after,
        },
        'sessions_scanned': session_count,
        'semantic_sessions_scanned': semantic_session_count,
        'candidate_samples': len(manifest_rows),
        'written_samples': written,
        'skipped_samples': skipped,
        'skip_reasons': reason_counts,
        'manifest': str(manifest_path),
    }
    _atomic_json(output_directory / 'summary.json', summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        '--output-directory', type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument('--minimum-surface-points', type=int, default=2)
    parser.add_argument(
        '--obstacle-vertical-span-m', type=float, default=0.15
    )
    parser.add_argument('--maximum-alignment-s', type=float, default=0.03)
    parser.add_argument('--ego-rear-m', type=float, default=2.5)
    parser.add_argument('--ego-front-m', type=float, default=2.4)
    parser.add_argument('--ego-half-width-m', type=float, default=1.0)
    parser.add_argument(
        '--visibility-angular-bin-count', type=int, default=720
    )
    parser.add_argument(
        '--exclude-collision-and-after',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = build_dataset(
        args.raw_root,
        args.output_directory,
        minimum_surface_points=args.minimum_surface_points,
        obstacle_vertical_span_m=args.obstacle_vertical_span_m,
        maximum_alignment_s=args.maximum_alignment_s,
        ego_rear_m=args.ego_rear_m,
        ego_front_m=args.ego_front_m,
        ego_half_width_m=args.ego_half_width_m,
        visibility_angular_bin_count=args.visibility_angular_bin_count,
        exclude_collision_and_after=args.exclude_collision_and_after,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
