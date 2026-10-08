#!/usr/bin/env python3
"""Build the non-destructive v2 traversability evidence dataset.

The raw recorder sessions and the legacy v1 derived dataset are never
modified.  The v2 archive keeps obstacle and passable-surface evidence as
independent targets, preserves ambiguity, stores object-instance provenance,
and adds multi-scale local-ground-relative LiDAR features.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .build_traversability_dataset import (
    DEFAULT_OUTPUT_DIRECTORY as V1_OUTPUT_DIRECTORY,
    _atomic_json,
    _atomic_npz,
    _copy_if_present,
    _float_or_nan,
    _read_geometry,
    _sample_identifier,
)
from .navigation_learning_recorder_core import BevGeometry
from .traversability_evidence_core import (
    AMBIGUOUS_DISPOSITION,
    OBSTACLE_DISPOSITION,
    PASSABLE_DISPOSITION,
    VALID_DISPOSITIONS,
    build_evidence_bev,
    build_vehicle_evidence_targets,
)
from .traversability_learning_core import (
    EgoFootprint,
    FREE_LABEL,
    OBSTACLE_LABEL,
    UNKNOWN_LABEL,
    TraversabilityTargets,
    build_conservative_visibility_evidence,
)


DEFAULT_RAW_ROOT = Path('/home/sukja/terrain_nav_data/learning/raw')
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/v2'
)
SCHEMA_VERSION = 3

DISPOSITION_CODE = {
    AMBIGUOUS_DISPOSITION: -1,
    PASSABLE_DISPOSITION: 0,
    OBSTACLE_DISPOSITION: 1,
}

MANIFEST_FIELDS = [
    'source_session', 'source_session_result', 'source_sample_id',
    'source_sample_path', 'derived_sample_path', 'status', 'reason',
    'ros_time_s', 'route_index', 'route_status',
    'semantic_alignment_delta_s', 'collision_event_count',
    'scan_fingerprint', 'duplicate_of_source_sample_id',
    'semantic_point_count', 'passable_cell_count', 'obstacle_cell_count',
    'ambiguous_cell_count', 'unobserved_cell_count',
    'controlled_actor_count', 'controlled_actor_return_count',
    'ego_excluded_cell_count', 'visibility_free_cell_count',
]


def _safe_output_directory(raw_root: Path, output_directory: Path) -> None:
    if output_directory == V1_OUTPUT_DIRECTORY.expanduser().resolve():
        raise ValueError(
            'v2 builder refuses to write into the legacy v1 directory: '
            + str(output_directory)
        )
    try:
        output_directory.relative_to(raw_root)
    except ValueError:
        return
    raise ValueError(
        'derived output must not be inside the raw recording root: '
        + str(output_directory)
    )


def _controlled_actor_policy(metadata: dict) -> tuple[dict[int, str], list[dict]]:
    """Read auditable per-actor vehicle policy from session metadata."""

    raw_entries = metadata.get('controlled_traversability_actors', [])
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list):
        raise ValueError('controlled_traversability_actors must be a list')

    dispositions: dict[int, str] = {}
    normalized: list[dict] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f'controlled actor entry {index} must be an object'
            )
        raw_actor_id = entry.get('actor_id', entry.get('object_idx'))
        if raw_actor_id is None:
            raise ValueError(f'controlled actor entry {index} lacks actor_id')
        actor_id = int(raw_actor_id)
        disposition = str(entry.get('disposition', '')).strip().lower()
        if disposition not in VALID_DISPOSITIONS:
            raise ValueError(
                f'controlled actor {actor_id} has invalid disposition '
                f'{disposition!r}'
            )
        if actor_id in dispositions and dispositions[actor_id] != disposition:
            raise ValueError(
                f'controlled actor {actor_id} has conflicting dispositions'
            )
        dispositions[actor_id] = disposition
        normalized_entry = {
            'actor_id': actor_id,
            'disposition': disposition,
            'blueprint': str(entry.get('blueprint', '')),
            'policy_source': str(entry.get('policy_source', '')),
        }
        if 'bbox_extent_xyz_m' in entry:
            extent = [float(value) for value in entry['bbox_extent_xyz_m']]
            if len(extent) != 3:
                raise ValueError(
                    f'controlled actor {actor_id} bbox must have three values'
                )
            normalized_entry['bbox_extent_xyz_m'] = extent
        normalized.append(normalized_entry)
    return dispositions, normalized


def _empty_result(geometry: BevGeometry, reason: str = '') -> dict:
    return {
        'status': 'skipped',
        'reason': reason,
        'scan_fingerprint': '',
        'duplicate_of_source_sample_id': '',
        'semantic_point_count': 0,
        'passable_cell_count': 0,
        'obstacle_cell_count': 0,
        'ambiguous_cell_count': 0,
        'unobserved_cell_count': geometry.height * geometry.width,
        'controlled_actor_count': 0,
        'controlled_actor_return_count': 0,
        'ego_excluded_cell_count': 0,
        'visibility_free_cell_count': 0,
    }


def _update_array_fingerprint(digest, name: str, array: np.ndarray) -> None:
    """Add one typed, shaped array to a deterministic content fingerprint."""

    value = np.ascontiguousarray(array)
    digest.update(name.encode('utf-8'))
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(json.dumps(value.shape).encode('ascii'))
    digest.update(value.tobytes())


def _scan_fingerprint(source) -> str:
    """Fingerprint model input and privileged target sources for deduping.

    Deduplication is deliberately scoped to one recorder session.  Actor IDs
    and vehicle policy may legitimately differ across sessions even when the
    geometric scan happens to be identical.
    """

    digest = hashlib.sha256()
    for name in (
        'lidar_bev',
        'semantic_lidar_points_xyz',
        'semantic_lidar_object_tag',
        'semantic_lidar_object_idx',
    ):
        if name in source:
            _update_array_fingerprint(digest, name, np.asarray(source[name]))
        else:
            digest.update((name + ':missing').encode('ascii'))
    return digest.hexdigest()


def _legacy_visibility_adapter(targets) -> TraversabilityTargets:
    labels = np.full(targets.observed_mask.shape, UNKNOWN_LABEL, dtype=np.int8)
    labels[targets.passable_surface_mask] = FREE_LABEL
    labels[targets.obstacle_evidence_mask] = OBSTACLE_LABEL
    observed_count = (
        targets.passable_point_count
        + targets.obstacle_point_count
        + targets.ambiguous_point_count
    )
    return TraversabilityTargets(
        labels=labels,
        free_point_count=targets.passable_point_count,
        obstacle_point_count=targets.obstacle_point_count,
        observed_point_count=observed_count,
        vertical_span_m=targets.vertical_span,
        ego_exclusion_mask=targets.ego_mask,
    )


def _process_sample(
    source_path: Path,
    output_path: Path,
    row: dict,
    geometry: BevGeometry,
    *,
    actor_dispositions: dict[int, str],
    actor_policy: list[dict],
    minimum_surface_points: int,
    minimum_controlled_passable_points: int,
    obstacle_vertical_span_m: float,
    maximum_alignment_s: float,
    ego_footprint: EgoFootprint,
    visibility_angular_bin_count: int,
    local_ground_radii_m: tuple[float, ...],
    local_ground_quantile: float,
    local_ground_minimum_support_cells: int,
    seen_scan_fingerprints: dict[str, int] | None = None,
) -> dict:
    result = _empty_result(geometry)
    result['controlled_actor_count'] = len(actor_dispositions)
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

            if 'semantic_lidar_object_idx' in source:
                object_ids = np.asarray(source['semantic_lidar_object_idx'])
                if object_ids.shape != (xyz.shape[0],):
                    result['reason'] = 'semantic_object_idx_shape_mismatch'
                    return result
            else:
                object_ids = np.full((xyz.shape[0],), -1, dtype=np.int64)
                if actor_dispositions:
                    result['reason'] = (
                        'controlled_policy_requires_semantic_object_idx'
                    )
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

            fingerprint = _scan_fingerprint(source)
            result['scan_fingerprint'] = fingerprint
            if (
                seen_scan_fingerprints is not None
                and fingerprint in seen_scan_fingerprints
            ):
                result['reason'] = 'duplicate_identical_scan'
                result['duplicate_of_source_sample_id'] = (
                    seen_scan_fingerprints[fingerprint]
                )
                return result

            targets = build_vehicle_evidence_targets(
                xyz,
                tags,
                geometry,
                ego_footprint,
                semantic_object_ids=object_ids,
                actor_dispositions=actor_dispositions,
                minimum_surface_points=minimum_surface_points,
                minimum_controlled_passable_points=(
                    minimum_controlled_passable_points
                ),
                obstacle_height_span=obstacle_vertical_span_m,
            )
            passable_cells = int(np.count_nonzero(
                targets.passable_surface_mask
            ))
            obstacle_cells = int(np.count_nonzero(
                targets.obstacle_evidence_mask
            ))
            if passable_cells + obstacle_cells == 0:
                result['reason'] = 'no_positive_evidence_cells'
                return result

            evidence_bev, local_ground = build_evidence_bev(
                lidar_bev,
                geometry,
                radii_m=local_ground_radii_m,
                ground_quantile=local_ground_quantile,
                minimum_support_cells=local_ground_minimum_support_cells,
            )
            visibility = build_conservative_visibility_evidence(
                xyz,
                geometry,
                _legacy_visibility_adapter(targets),
                ego_footprint=ego_footprint,
                angular_bin_count=visibility_angular_bin_count,
            )

            controlled_actor_ids = np.asarray(
                sorted(actor_dispositions), dtype=np.int64
            )
            controlled_actor_codes = np.asarray(
                [
                    DISPOSITION_CODE[actor_dispositions[int(actor_id)]]
                    for actor_id in controlled_actor_ids
                ],
                dtype=np.int8,
            )
            controlled_returns = int(np.count_nonzero(
                np.isin(object_ids.astype(np.int64), controlled_actor_ids)
            )) if controlled_actor_ids.size else 0

            derived = {
                'lidar_bev': lidar_bev.astype(np.float16, copy=False),
                'lidar_evidence_bev': evidence_bev.astype(
                    np.float16, copy=False
                ),
                'local_ground_height_m': local_ground.ground_height.astype(
                    np.float16, copy=False
                ),
                'local_ground_relative_max_height_m': (
                    local_ground.relative_max_height.astype(
                        np.float16, copy=False
                    )
                ),
                'local_ground_support_count': local_ground.support_count,
                'local_ground_support_confidence': (
                    local_ground.support_confidence.astype(
                        np.float16, copy=False
                    )
                ),
                'local_ground_valid_mask': local_ground.valid_mask,
                'target_passable_surface_mask': (
                    targets.passable_surface_mask
                ),
                'target_obstacle_evidence_mask': (
                    targets.obstacle_evidence_mask
                ),
                'target_ambiguous_observed_mask': (
                    targets.ambiguous_observed_mask
                ),
                'target_observed_mask': targets.observed_mask,
                'target_passable_point_count': targets.passable_point_count,
                'target_obstacle_point_count': targets.obstacle_point_count,
                'target_ambiguous_point_count': targets.ambiguous_point_count,
                'target_controlled_passable_point_count': (
                    targets.controlled_passable_point_count
                ),
                'target_controlled_obstacle_point_count': (
                    targets.controlled_obstacle_point_count
                ),
                'target_controlled_ambiguous_point_count': (
                    targets.controlled_ambiguous_point_count
                ),
                'target_vertical_span_m': targets.vertical_span,
                'target_obstacle_instance_id': targets.obstacle_instance_id,
                'target_ego_exclusion_mask': targets.ego_mask,
                'visibility_free_mask': visibility.free_mask,
                'visibility_ray_count': visibility.ray_count,
                'controlled_actor_ids': controlled_actor_ids,
                'controlled_actor_disposition_code': controlled_actor_codes,
                'controlled_actor_policy_json': np.asarray(
                    json.dumps(actor_policy, sort_keys=True)
                ),
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

    ambiguous_cells = int(np.count_nonzero(
        targets.ambiguous_observed_mask
    ))
    unobserved_cells = int(np.count_nonzero(~targets.observed_mask))
    result.update({
        'status': 'written',
        'reason': '',
        'semantic_point_count': int(xyz.shape[0]),
        'passable_cell_count': passable_cells,
        'obstacle_cell_count': obstacle_cells,
        'ambiguous_cell_count': ambiguous_cells,
        'unobserved_cell_count': unobserved_cells,
        'controlled_actor_count': len(actor_dispositions),
        'controlled_actor_return_count': controlled_returns,
        'ego_excluded_cell_count': int(np.count_nonzero(targets.ego_mask)),
        'visibility_free_cell_count': int(np.count_nonzero(
            visibility.free_mask
        )),
    })
    if seen_scan_fingerprints is not None:
        seen_scan_fingerprints[result['scan_fingerprint']] = (
            _sample_identifier(row, source_path)
        )
    return result


def build_evidence_dataset(
    raw_root: Path,
    output_directory: Path,
    *,
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
    exclude_collision_and_after: bool = True,
    deduplicate_identical_scans: bool = True,
    session_names: tuple[str, ...] | None = None,
) -> dict:
    """Build an auditable v2 archive without mutating raw data or v1."""

    raw_root = Path(raw_root).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError('raw recording root not found: ' + str(raw_root))
    _safe_output_directory(raw_root, output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    ego_footprint = EgoFootprint(
        rear_m=ego_rear_m,
        front_m=ego_front_m,
        half_width_m=ego_half_width_m,
    )

    requested_sessions = tuple(dict.fromkeys(session_names or ()))
    available_sessions = {
        path.name: path for path in raw_root.glob('session_*') if path.is_dir()
    }
    if requested_sessions:
        missing_sessions = sorted(
            set(requested_sessions).difference(available_sessions)
        )
        if missing_sessions:
            raise ValueError(
                'requested sessions were not found: '
                + ', '.join(missing_sessions)
            )
        sessions = [available_sessions[name] for name in requested_sessions]
    else:
        sessions = [
            available_sessions[name] for name in sorted(available_sessions)
        ]

    manifest_rows: list[dict] = []
    written = skipped = session_count = semantic_session_count = 0
    controlled_session_count = 0
    empty_sessions: list[str] = []
    for session in sessions:
        metadata_path = session / 'metadata.json'
        frames_path = session / 'frames.csv'
        if not metadata_path.is_file() or not frames_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            geometry = _read_geometry(metadata)
            actor_dispositions, actor_policy = _controlled_actor_policy(metadata)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        session_count += 1
        supervision = metadata.get('privileged_supervision', {})
        if not bool(supervision.get('semantic_labels_saved', False)):
            continue
        semantic_session_count += 1
        controlled_session_count += int(bool(actor_dispositions))

        with frames_path.open(newline='', encoding='utf-8') as stream:
            frame_rows = list(csv.DictReader(stream))
        if not frame_rows:
            empty_sessions.append(session.name)
            continue

        seen_scan_fingerprints = (
            {} if deduplicate_identical_scans else None
        )
        for row in frame_rows:
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
                details = _empty_result(geometry, 'collision_or_after')
                details['controlled_actor_count'] = len(actor_dispositions)
            else:
                details = _process_sample(
                    source_path,
                    derived_path,
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
                    ego_footprint=ego_footprint,
                    visibility_angular_bin_count=(
                        visibility_angular_bin_count
                    ),
                    local_ground_radii_m=tuple(local_ground_radii_m),
                    local_ground_quantile=local_ground_quantile,
                    local_ground_minimum_support_cells=(
                        local_ground_minimum_support_cells
                    ),
                    seen_scan_fingerprints=seen_scan_fingerprints,
                )
            written += int(details['status'] == 'written')
            skipped += int(details['status'] != 'written')
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
                **{
                    field: details[field]
                    for field in MANIFEST_FIELDS
                    if field in details
                },
            })

    manifest_path = output_directory / 'manifest.csv'
    temporary_manifest = manifest_path.with_suffix('.csv.tmp')
    with temporary_manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(manifest_path)

    reason_counts: dict[str, int] = {}
    for row in manifest_rows:
        if row['reason']:
            reason_counts[row['reason']] = reason_counts.get(row['reason'], 0) + 1
    summary = {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'raw_root': str(raw_root),
        'output_directory': str(output_directory),
        'model_inputs': [
            'lidar_bev', 'local_ground_relative_max_height_m',
            'local_ground_support_confidence',
        ],
        'target_contract': {
            'passable_surface': 'independent positive evidence',
            'obstacle': 'independent positive evidence; wins conflicts',
            'unknown': 'neither positive evidence; never inferred from low obstacle probability',
        },
        'controlled_actor_dispositions': sorted(VALID_DISPOSITIONS),
        'parameters': {
            'minimum_surface_points': minimum_surface_points,
            'minimum_controlled_passable_points': (
                minimum_controlled_passable_points
            ),
            'obstacle_vertical_span_m': obstacle_vertical_span_m,
            'maximum_alignment_s': maximum_alignment_s,
            'ego_rear_m': ego_rear_m,
            'ego_front_m': ego_front_m,
            'ego_half_width_m': ego_half_width_m,
            'visibility_angular_bin_count': visibility_angular_bin_count,
            'local_ground_radii_m': list(local_ground_radii_m),
            'local_ground_quantile': local_ground_quantile,
            'local_ground_minimum_support_cells': (
                local_ground_minimum_support_cells
            ),
            'exclude_collision_and_after': exclude_collision_and_after,
            'deduplicate_identical_scans': deduplicate_identical_scans,
        },
        'requested_sessions': list(requested_sessions),
        'sessions_scanned': session_count,
        'semantic_sessions_scanned': semantic_session_count,
        'controlled_sessions_scanned': controlled_session_count,
        'empty_sessions': empty_sessions,
        'empty_session_count': len(empty_sessions),
        'candidate_samples': len(manifest_rows),
        'written_samples': written,
        'skipped_samples': skipped,
        'duplicate_samples_skipped': reason_counts.get(
            'duplicate_identical_scan', 0
        ),
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
        '--minimum-controlled-passable-points', type=int, default=1
    )
    parser.add_argument('--obstacle-vertical-span-m', type=float, default=0.15)
    parser.add_argument('--maximum-alignment-s', type=float, default=0.03)
    parser.add_argument('--ego-rear-m', type=float, default=2.5)
    parser.add_argument('--ego-front-m', type=float, default=2.4)
    parser.add_argument('--ego-half-width-m', type=float, default=1.0)
    parser.add_argument('--visibility-angular-bin-count', type=int, default=720)
    parser.add_argument(
        '--local-ground-radii-m', type=float, nargs='+', default=(0.75, 1.50)
    )
    parser.add_argument('--local-ground-quantile', type=float, default=0.25)
    parser.add_argument(
        '--local-ground-minimum-support-cells', type=int, default=4
    )
    parser.add_argument(
        '--exclude-collision-and-after',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--deduplicate-identical-scans',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='deduplicate exact model-input/target scans within each session',
    )
    parser.add_argument(
        '--session', action='append', default=None,
        help='source session directory name to include; may be repeated',
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = build_evidence_dataset(
        args.raw_root,
        args.output_directory,
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
        exclude_collision_and_after=args.exclude_collision_and_after,
        deduplicate_identical_scans=args.deduplicate_identical_scans,
        session_names=tuple(args.session or ()),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
