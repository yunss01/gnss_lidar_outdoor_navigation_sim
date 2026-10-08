#!/usr/bin/env python3
"""Audit navigation recordings and build leakage-resistant data manifests.

The recorder intentionally keeps successful, interrupted, and recovery
behaviour.  This utility does not delete or rewrite any raw recording.  It
creates derived CSV manifests that keep those behaviours separate so a later
training stage cannot silently treat every recorded frame as a clean teacher
example.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_RAW_ROOT = Path('/home/sukja/terrain_nav_data/learning/raw')
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/manifests/v1'
)

MANIFEST_FIELDS = [
    'session', 'split', 'session_result', 'route_signature', 'route_size',
    'sample_id', 'sample_path', 'sample_exists', 'npz_verification',
    'ros_time_s', 'route_index', 'route_status', 'speed_mps',
    'goal_vehicle_x', 'goal_vehicle_y', 'teacher_subgoal_x',
    'teacher_subgoal_y', 'nav2_plan_points', 'raw_lidar_points',
    'nav2_status', 'far_guide_status', 'safety_state',
    'collision_event_count', 'base_sample_valid', 'quality_tier',
    'initial_bc_eligible', 'recovery_candidate',
    'safety_negative_candidate', 'reason_codes',
]

SESSION_FIELDS = [
    'session', 'split', 'session_result', 'route_signature', 'route_size',
    'frame_rows', 'recorded_sample_rows', 'existing_sample_files',
    'base_valid_frames', 'clean_imitation_frames', 'recovery_frames',
    'safety_negative_frames', 'verified_samples', 'verification_failures',
    'metadata_path',
]

RECOVERY_NAV2_STATUSES = {
    'blocked_inefficient_path',
    'nav2_result_4',
    'nav2_result_6',
}

SAFETY_NEGATIVE_STATES = {
    'collision_stop',
    'invalid_lidar',
    'obstacle_recovery',
    'obstacle_stop',
    'waiting_for_lidar',
}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def _canonical_route(metadata: dict) -> dict:
    """Return only fields that identify the ordered waypoint route."""
    route = metadata.get('route_json', {})
    if isinstance(route, str):
        try:
            route = json.loads(route)
        except (TypeError, json.JSONDecodeError):
            route = {}
    if not isinstance(route, dict):
        route = {}

    waypoints = []
    for item in route.get('waypoints', []):
        if not isinstance(item, dict):
            continue
        try:
            waypoints.append({
                'latitude': round(float(item['latitude']), 10),
                'longitude': round(float(item['longitude']), 10),
                'altitude': round(float(item.get('altitude', 0.0)), 4),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return {
        'waypoints': waypoints,
        'loop': bool(route.get('loop', False)),
    }


def route_signature(metadata: dict) -> str:
    """Return a stable short hash for an ordered waypoint route."""
    payload = json.dumps(
        _canonical_route(metadata), sort_keys=True, separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def deterministic_split(
    group_key: str,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    """Assign an entire group to one reproducible data split."""
    if train_fraction <= 0.0 or validation_fraction < 0.0:
        raise ValueError('split fractions must be non-negative')
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError('train + validation fractions must be less than 1')
    digest = hashlib.sha256(
        '{}:{}'.format(int(seed), group_key).encode('utf-8')
    ).digest()
    value = int.from_bytes(digest[:8], 'big') / float(2 ** 64)
    if value < train_fraction:
        return 'train'
    if value < train_fraction + validation_fraction:
        return 'validation'
    return 'test'


def balanced_session_splits(
    session_rows: list[dict],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    """Balance clean-frame weight per route while keeping sessions intact."""
    test_fraction = 1.0 - train_fraction - validation_fraction
    if train_fraction <= 0.0 or validation_fraction < 0.0:
        raise ValueError('split fractions must be non-negative')
    if test_fraction <= 0.0:
        raise ValueError('test fraction must be positive')
    split_names = ('train', 'validation', 'test')
    fractions = {
        'train': train_fraction,
        'validation': validation_fraction,
        'test': test_fraction,
    }
    result = {}
    by_route = defaultdict(list)
    for row in session_rows:
        by_route[row['route_signature']].append(row)

    for signature, route_rows in sorted(by_route.items()):
        clean_rows = [
            row for row in route_rows
            if _integer(row.get('clean_imitation_frames')) > 0
        ]
        other_rows = [
            row for row in route_rows
            if _integer(row.get('clean_imitation_frames')) <= 0
        ]
        clean_rows.sort(key=lambda row: (
            -_integer(row.get('clean_imitation_frames')),
            hashlib.sha256(
                '{}:{}:{}'.format(
                    seed, signature, row['session']
                ).encode('utf-8')
            ).hexdigest(),
        ))
        total_clean = sum(
            _integer(row.get('clean_imitation_frames')) for row in clean_rows
        )
        clean_assigned = Counter()
        session_assigned = Counter()

        for row in clean_rows:
            weight = _integer(row.get('clean_imitation_frames'))
            # Largest sessions are placed first.  Normalizing by each split's
            # target makes the greedy assignment preserve the requested
            # proportions while ensuring non-empty validation/test sets when
            # at least three clean sessions are available for a route.
            chosen = min(
                split_names,
                key=lambda name: (
                    clean_assigned[name]
                    / max(total_clean * fractions[name], 1.0),
                    -fractions[name],
                    name,
                ),
            )
            result[row['session']] = chosen
            clean_assigned[chosen] += weight
            session_assigned[chosen] += 1

        other_rows.sort(key=lambda row: hashlib.sha256(
            '{}:{}:{}'.format(seed, signature, row['session']).encode(
                'utf-8'
            )
        ).hexdigest())
        target_sessions = {
            name: len(route_rows) * fractions[name] for name in split_names
        }
        for row in other_rows:
            chosen = min(
                split_names,
                key=lambda name: (
                    session_assigned[name]
                    / max(target_sessions[name], 1.0),
                    -fractions[name],
                    name,
                ),
            )
            result[row['session']] = chosen
            session_assigned[chosen] += 1
    return result


def _finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _integer(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def assess_frame(
    row: dict,
    *,
    session_result: str,
    sample_exists: bool,
    npz_verification: str = 'not_checked',
) -> dict:
    """Classify one frame without discarding ambiguous recovery data."""
    reasons = []
    if not row.get('file'):
        reasons.append('no_sample_recorded')
    elif not sample_exists:
        reasons.append('sample_file_missing')
    if _integer(row.get('raw_lidar_points')) <= 0:
        reasons.append('no_raw_lidar')
    if _integer(row.get('nav2_plan_points')) < 2:
        reasons.append('insufficient_nav2_plan')
    if not (
        _finite_number(row.get('goal_vehicle_x'))
        and _finite_number(row.get('goal_vehicle_y'))
    ):
        reasons.append('invalid_goal')
    if npz_verification.startswith('failed:'):
        reasons.append('npz_verification_failed')

    base_valid = not reasons
    nav2_status = str(row.get('nav2_status', ''))
    safety_state = str(row.get('safety_state', ''))
    far_status = str(row.get('far_guide_status', ''))
    clean_imitation = (
        base_valid
        and session_result == 'completed'
        and nav2_status == 'navigating'
        and safety_state == 'clear'
        and far_status == 'following_far_segment'
    )
    recovery_candidate = (
        base_valid
        and (
            session_result != 'completed'
            or nav2_status in RECOVERY_NAV2_STATUSES
            or 'replan' in far_status
            or 'blocked' in far_status
            or 'escape' in far_status
            or 'guide_not_found' in far_status
        )
    )
    safety_negative = base_valid and safety_state in SAFETY_NEGATIVE_STATES

    if clean_imitation:
        quality_tier = 'clean_imitation'
    elif recovery_candidate or safety_negative:
        quality_tier = 'recovery_or_negative'
    elif base_valid:
        quality_tier = 'review_required'
    else:
        quality_tier = 'unusable'

    return {
        'base_sample_valid': int(base_valid),
        'quality_tier': quality_tier,
        'initial_bc_eligible': int(clean_imitation),
        'recovery_candidate': int(recovery_candidate),
        'safety_negative_candidate': int(safety_negative),
        'reason_codes': ';'.join(reasons),
    }


def _verification_indices(frame_count: int, requested: int) -> set[int]:
    if requested == 0 or frame_count <= 0:
        return set()
    if requested < 0 or requested >= frame_count:
        return set(range(frame_count))
    return set(np.linspace(
        0, frame_count - 1, num=requested, dtype=np.int64,
    ).tolist())


def verify_sample_file(path: Path) -> str:
    """Check essential arrays and shapes in one compressed sample."""
    try:
        with np.load(path, allow_pickle=False) as arrays:
            required = {
                'lidar_bev', 'lidar_points_xyz', 'nav2_plan_vehicle_xy',
                'goal_vehicle_xyz', 'vehicle_pose_odom_xyzyaw',
                'vehicle_twist_xyz_rpy',
            }
            missing = sorted(required - set(arrays.files))
            if missing:
                return 'failed:missing_arrays={}'.format(','.join(missing))
            bev = arrays['lidar_bev']
            points = arrays['lidar_points_xyz']
            plan = arrays['nav2_plan_vehicle_xy']
            goal = arrays['goal_vehicle_xyz']
            if bev.ndim != 3 or bev.shape[0] < 1:
                return 'failed:invalid_lidar_bev_shape={}'.format(bev.shape)
            if points.ndim != 2 or points.shape[1] != 3:
                return 'failed:invalid_lidar_points_shape={}'.format(
                    points.shape
                )
            if plan.ndim != 2 or plan.shape[1] != 2:
                return 'failed:invalid_nav2_plan_shape={}'.format(plan.shape)
            if goal.ndim != 1 or goal.size < 2:
                return 'failed:invalid_goal_shape={}'.format(goal.shape)
    except Exception as error:  # Corrupt archives raise several subclasses.
        return 'failed:{}:{}'.format(type(error).__name__, str(error)[:120])
    return 'passed'


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})
    temporary.replace(path)


def build_manifests(
    raw_root: Path,
    output_directory: Path,
    *,
    seed: int = 20260904,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    split_unit: str = 'session',
    verify_samples_per_session: int = 5,
) -> dict:
    """Build manifests and return their JSON-serializable summary."""
    raw_root = raw_root.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if split_unit not in {'session', 'route'}:
        raise ValueError('split_unit must be session or route')
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)

    manifest_rows = []
    session_rows = []
    route_splits = defaultdict(set)
    route_counts = Counter()
    session_result_counts = Counter()
    quality_counts = Counter()
    reason_counts = Counter()

    session_directories = sorted(
        item for item in raw_root.glob('session_*') if item.is_dir()
    )
    for session_directory in session_directories:
        metadata_path = session_directory / 'metadata.json'
        frames_path = session_directory / 'frames.csv'
        if not metadata_path.is_file() or not frames_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        session = str(metadata.get('session') or session_directory.name)
        session_result = str(metadata.get('result', 'unknown'))
        signature = route_signature(metadata)
        route_size = _integer(metadata.get('route_size'))
        split_key = signature if split_unit == 'route' else session
        split = deterministic_split(
            split_key,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        route_counts[signature] += 1
        session_result_counts[session_result] += 1

        with frames_path.open(newline='', encoding='utf-8') as stream:
            frame_rows = list(csv.DictReader(stream))
        recorded = [row for row in frame_rows if row.get('file')]
        verification_positions = _verification_indices(
            len(recorded), verify_samples_per_session
        )
        verification_by_path = {}
        for position in verification_positions:
            relative = recorded[position].get('file', '')
            sample_path = session_directory / relative
            if sample_path.is_file():
                verification_by_path[str(sample_path)] = verify_sample_file(
                    sample_path
                )
            else:
                verification_by_path[str(sample_path)] = 'failed:file_missing'

        session_manifest_rows = []
        for row in frame_rows:
            relative = str(row.get('file', ''))
            sample_path = session_directory / relative if relative else None
            sample_exists = bool(sample_path and sample_path.is_file())
            verification = (
                verification_by_path.get(str(sample_path), 'not_checked')
                if sample_path else 'not_applicable'
            )
            assessment = assess_frame(
                row,
                session_result=session_result,
                sample_exists=sample_exists,
                npz_verification=verification,
            )
            output = {
                'session': session,
                'split': split,
                'session_result': session_result,
                'route_signature': signature,
                'route_size': route_size,
                'sample_id': row.get('sample_id', ''),
                'sample_path': str(sample_path) if sample_path else '',
                'sample_exists': int(sample_exists),
                'npz_verification': verification,
            }
            for field in MANIFEST_FIELDS:
                if field not in output and field not in assessment:
                    output[field] = row.get(field, '')
            output.update(assessment)
            manifest_rows.append(output)
            session_manifest_rows.append(output)
            quality_counts[output['quality_tier']] += 1
            for reason in output['reason_codes'].split(';'):
                if reason:
                    reason_counts[reason] += 1

        count = Counter()
        for item in session_manifest_rows:
            count['recorded'] += bool(item['sample_path'])
            count['existing'] += bool(item['sample_exists'])
            count['base_valid'] += bool(item['base_sample_valid'])
            count['clean'] += bool(item['initial_bc_eligible'])
            count['recovery'] += bool(item['recovery_candidate'])
            count['negative'] += bool(item['safety_negative_candidate'])
            count['verified'] += item['npz_verification'] == 'passed'
            count['verify_failed'] += item['npz_verification'].startswith(
                'failed:'
            )
        session_rows.append({
            'session': session,
            'split': split,
            'session_result': session_result,
            'route_signature': signature,
            'route_size': route_size,
            'frame_rows': len(session_manifest_rows),
            'recorded_sample_rows': count['recorded'],
            'existing_sample_files': count['existing'],
            'base_valid_frames': count['base_valid'],
            'clean_imitation_frames': count['clean'],
            'recovery_frames': count['recovery'],
            'safety_negative_frames': count['negative'],
            'verified_samples': count['verified'],
            'verification_failures': count['verify_failed'],
            'metadata_path': str(metadata_path),
        })

    if split_unit == 'session':
        split_by_session = balanced_session_splits(
            session_rows,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        for row in session_rows:
            row['split'] = split_by_session[row['session']]
        for row in manifest_rows:
            row['split'] = split_by_session[row['session']]

    route_splits.clear()
    for row in session_rows:
        route_splits[row['route_signature']].add(row['split'])

    _write_csv(
        output_directory / 'manifest_all.csv', MANIFEST_FIELDS, manifest_rows
    )
    for split in ('train', 'validation', 'test'):
        _write_csv(
            output_directory / 'manifest_{}.csv'.format(split),
            MANIFEST_FIELDS,
            (
                row for row in manifest_rows
                if row['split'] == split and row['initial_bc_eligible']
            ),
        )
    _write_csv(
        output_directory / 'manifest_recovery.csv',
        MANIFEST_FIELDS,
        (row for row in manifest_rows if row['recovery_candidate']),
    )
    _write_csv(
        output_directory / 'manifest_safety_negative.csv',
        MANIFEST_FIELDS,
        (row for row in manifest_rows if row['safety_negative_candidate']),
    )
    _write_csv(
        output_directory / 'sessions.csv', SESSION_FIELDS, session_rows
    )

    split_summary = {}
    for split in ('train', 'validation', 'test'):
        split_summary[split] = {
            'sessions': sum(row['split'] == split for row in session_rows),
            'sample_bearing_sessions': sum(
                row['split'] == split
                and _integer(row.get('existing_sample_files')) > 0
                for row in session_rows
            ),
            'clean_imitation_sessions': sum(
                row['split'] == split
                and _integer(row.get('clean_imitation_frames')) > 0
                for row in session_rows
            ),
            'all_frames': sum(row['split'] == split for row in manifest_rows),
            'clean_imitation_frames': sum(
                row['split'] == split and bool(row['initial_bc_eligible'])
                for row in manifest_rows
            ),
            'recovery_frames': sum(
                row['split'] == split and bool(row['recovery_candidate'])
                for row in manifest_rows
            ),
        }

    overlapping_routes = sorted(
        signature for signature, splits in route_splits.items()
        if len(splits) > 1
    )
    warnings = []
    if len(route_counts) < 6:
        warnings.append(
            'Only {} unique waypoint-route signatures were found; collect '
            'more routes before claiming unseen-route generalization.'.format(
                len(route_counts)
            )
        )
    if split_unit == 'session' and overlapping_routes:
        warnings.append(
            '{} route signatures occur in more than one split. This split '
            'is suitable for pipeline development, not final unseen-route '
            'evaluation.'.format(len(overlapping_routes))
        )
    if split_unit == 'route' and any(
        split_summary[split]['sessions'] == 0
        for split in ('train', 'validation', 'test')
    ):
        warnings.append(
            'Route-group splitting produced an empty split because too few '
            'unique routes are available.'
        )
    if reason_counts['no_sample_recorded']:
        warnings.append(
            '{} frame rows have no saved NPZ file and cannot be used for '
            'sensor-to-trajectory training.'.format(
                reason_counts['no_sample_recorded']
            )
        )
    sample_bearing_sessions = sum(
        _integer(row.get('existing_sample_files')) > 0 for row in session_rows
    )
    if sample_bearing_sessions < 30:
        warnings.append(
            'Only {} sessions contain saved NPZ samples. Consecutive frames '
            'must not be treated as equivalent to independent scene '
            'coverage.'.format(sample_bearing_sessions)
        )

    summary = {
        'schema_version': SCHEMA_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'raw_root': str(raw_root),
        'output_directory': str(output_directory),
        'split': {
            'unit': split_unit,
            'seed': seed,
            'train_fraction': train_fraction,
            'validation_fraction': validation_fraction,
            'test_fraction': 1.0 - train_fraction - validation_fraction,
        },
        'sessions': len(session_rows),
        'sample_bearing_sessions': sample_bearing_sessions,
        'frame_rows': len(manifest_rows),
        'existing_sample_files': sum(
            bool(row['sample_exists']) for row in manifest_rows
        ),
        'unique_route_signatures': len(route_counts),
        'route_session_counts': dict(sorted(route_counts.items())),
        'session_result_counts': dict(sorted(session_result_counts.items())),
        'quality_tier_counts': dict(sorted(quality_counts.items())),
        'reason_counts': dict(sorted(reason_counts.items())),
        'split_counts': split_summary,
        'verified_samples': sum(
            row['npz_verification'] == 'passed' for row in manifest_rows
        ),
        'verification_failures': sum(
            row['npz_verification'].startswith('failed:')
            for row in manifest_rows
        ),
        'route_signatures_shared_across_splits': overlapping_routes,
        'warnings': warnings,
    }
    _atomic_write_text(
        output_directory / 'summary.json',
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Audit navigation recordings and create non-destructive '
            'session-grouped learning manifests.'
        )
    )
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        '--output-directory', type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument('--seed', type=int, default=20260904)
    parser.add_argument('--train-fraction', type=float, default=0.70)
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument(
        '--split-unit', choices=('session', 'route'), default='session',
        help=(
            'Session is useful for initial development. Route is required '
            'for final unseen-route evaluation once enough routes exist.'
        ),
    )
    parser.add_argument(
        '--verify-samples-per-session', type=int, default=5,
        help='Evenly sample this many NPZ files per session; -1 checks all.',
    )
    args = parser.parse_args(argv)
    summary = build_manifests(
        args.raw_root,
        args.output_directory,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_unit=args.split_unit,
        verify_samples_per_session=args.verify_samples_per_session,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
