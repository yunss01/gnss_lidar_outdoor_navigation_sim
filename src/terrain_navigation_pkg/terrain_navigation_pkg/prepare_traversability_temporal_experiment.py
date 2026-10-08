#!/usr/bin/env python3
"""Freeze distance-aligned temporal scenes for representation comparison."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


OUTPUT_FIELDS = (
    'split', 'scene_id', 'pair_group', 'object_family', 'role',
    'evaluation_slice', 'source_session', 'source_sample_id',
    'source_sample_path', 'derived_sample_path', 'scan_fingerprint',
    'controlled_actor_return_count', 'controlled_obstacle_cell_count',
    'pair_position_error_m', 'pair_yaw_error_deg',
)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _angle_error_deg(left: float, right: float) -> float:
    delta = math.atan2(math.sin(left - right), math.cos(left - right))
    return abs(math.degrees(delta))


def _parse_scene(value: str) -> tuple[str, Path]:
    if '=' not in value:
        raise ValueError('--scene must use SCENE_ID=ARCHIVE_DIRECTORY')
    scene_id, path = value.split('=', 1)
    scene_id = scene_id.strip()
    if not scene_id:
        raise ValueError('scene ID cannot be blank')
    return scene_id, Path(path).expanduser().resolve()


def _load_written_rows(scene_id: str, archive: Path) -> list[dict]:
    manifest = archive / 'manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError(
            'temporal manifest not found: ' + str(manifest)
        )
    rows = []
    for source in _read_csv(manifest):
        if source.get('status') != 'written':
            continue
        sample_path = (
            Path(source['derived_sample_path']).expanduser().resolve()
        )
        if not sample_path.is_file():
            raise FileNotFoundError(
                'derived sample not found: ' + str(sample_path)
            )
        with np.load(sample_path, allow_pickle=False) as arrays:
            evidence = np.asarray(arrays['lidar_evidence_bev'])
            current = np.asarray(arrays['current_lidar_evidence_bev'])
            if evidence.shape[0] != 17 or current.shape[0] != 8:
                raise ValueError('temporal archive has an invalid input view')
            pose = np.asarray(
                arrays['vehicle_pose_odom_xyzyaw'], dtype=np.float64
            )
            policy = json.loads(
                str(arrays['controlled_actor_policy_json'].item())
            )
            controlled_cells = int(np.count_nonzero(
                arrays['target_controlled_obstacle_point_count'] > 0
            ))
        role = 'controlled' if policy else 'control'
        if role == 'controlled':
            if len(policy) != 1:
                raise ValueError('controlled frame must describe one actor')
            if policy[0].get('disposition') != 'obstacle':
                raise ValueError('controlled actor is not obstacle-labelled')
            family = (
                str(policy[0].get('blueprint', ''))
                .split('.')[-1].lower()
            )
            if not family or controlled_cells <= 0:
                raise ValueError(
                    'controlled frame lacks actor target evidence'
                )
        else:
            family = 'background'
            if controlled_cells:
                raise ValueError('control frame contains controlled evidence')
        row = dict(source)
        row.update({
            '_scene_id': scene_id,
            '_archive': str(archive),
            '_sample_path': str(sample_path),
            '_pose': pose,
            '_role': role,
            '_family': family,
            '_controlled_cells': controlled_cells,
        })
        rows.append(row)
    if not rows:
        raise ValueError('temporal archive contains no written samples')
    return rows


def _session_roles(rows: list[dict]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in rows:
        session = row['source_session']
        role = row['_role']
        if session in roles and roles[session] != role:
            raise ValueError('session mixes control and controlled frames')
        roles[session] = role
    return roles


def _output_row(
    row: dict,
    *,
    pair_group: str,
    position_error_m: float,
    yaw_error_deg: float,
) -> dict:
    return {
        'split': 'train',
        'scene_id': row['_scene_id'],
        'pair_group': pair_group,
        'object_family': row['_family'],
        'role': row['_role'],
        'evaluation_slice': 'temporal_development',
        'source_session': row['source_session'],
        'source_sample_id': int(row['source_sample_id']),
        'source_sample_path': row['source_sample_path'],
        'derived_sample_path': row['_sample_path'],
        'scan_fingerprint': row.get('scan_fingerprint', ''),
        'controlled_actor_return_count': int(
            row.get('controlled_actor_return_count', 0) or 0
        ),
        'controlled_obstacle_cell_count': row['_controlled_cells'],
        'pair_position_error_m': f'{position_error_m:.6f}',
        'pair_yaw_error_deg': f'{yaw_error_deg:.6f}',
    }


def prepare_temporal_experiment(
    scenes: list[tuple[str, Path]],
    output_directory: Path,
    maximum_pair_position_error_m: float,
    maximum_pair_yaw_error_deg: float,
) -> dict:
    output_directory = Path(output_directory).expanduser().resolve()
    scene_ids = [scene_id for scene_id, _ in scenes]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError('scene IDs must be unique')
    if len(scenes) < 2:
        raise ValueError('at least two development scenes are required')

    prepared = []
    scene_reports = {}
    source_manifests = []
    for scene_id, archive in scenes:
        rows = _load_written_rows(scene_id, archive)
        source_manifests.append(archive / 'manifest.csv')
        roles = _session_roles(rows)
        control_sessions = [
            session for session, role in roles.items() if role == 'control'
        ]
        if len(control_sessions) != 1:
            raise ValueError(
                f'{scene_id} needs exactly one control session'
            )
        control_session = control_sessions[0]
        control_rows = [
            row for row in rows if row['source_session'] == control_session
        ]
        controlled_rows = [row for row in rows if row['_role'] == 'controlled']
        if not controlled_rows:
            raise ValueError(scene_id + ' contains no controlled frames')
        control_xy = np.stack([row['_pose'][:2] for row in control_rows])

        matches = []
        excluded_position = 0
        excluded_yaw = 0
        for row in controlled_rows:
            distances = np.linalg.norm(control_xy - row['_pose'][:2], axis=1)
            index = int(np.argmin(distances))
            position_error = float(distances[index])
            control = control_rows[index]
            yaw_error = _angle_error_deg(
                float(row['_pose'][3]), float(control['_pose'][3])
            )
            if position_error > maximum_pair_position_error_m:
                excluded_position += 1
                continue
            if yaw_error > maximum_pair_yaw_error_deg:
                excluded_yaw += 1
                continue
            matches.append((row, control, position_error, yaw_error))
        if not matches:
            raise ValueError(
                scene_id + ' has no valid matched temporal frames'
            )

        used_controls = {}
        for _, control, _, _ in matches:
            control_key = int(control['source_sample_id'])
            used_controls[control_key] = control
        group_by_control = {
            sample_id: f'{scene_id}:control_{sample_id:06d}'
            for sample_id in sorted(used_controls)
        }
        for sample_id, control in sorted(used_controls.items()):
            prepared.append(_output_row(
                control,
                pair_group=group_by_control[sample_id],
                position_error_m=0.0,
                yaw_error_deg=0.0,
            ))
        for row, control, position_error, yaw_error in matches:
            prepared.append(_output_row(
                row,
                pair_group=group_by_control[int(control['source_sample_id'])],
                position_error_m=position_error,
                yaw_error_deg=yaw_error,
            ))
        errors = np.asarray([match[2] for match in matches], dtype=np.float64)
        yaw_errors = np.asarray(
            [match[3] for match in matches], dtype=np.float64
        )
        scene_reports[scene_id] = {
            'archive_directory': str(archive),
            'control_session': control_session,
            'controlled_sessions': sorted(
                session for session, role in roles.items()
                if role == 'controlled'
            ),
            'control_rows_used': len(used_controls),
            'controlled_rows_matched': len(matches),
            'controlled_rows_excluded_position': excluded_position,
            'controlled_rows_excluded_yaw': excluded_yaw,
            'pair_position_error_m': {
                'median': float(np.median(errors)),
                'p95': float(np.quantile(errors, 0.95)),
                'maximum': float(np.max(errors)),
            },
            'pair_yaw_error_deg': {
                'median': float(np.median(yaw_errors)),
                'p95': float(np.quantile(yaw_errors, 0.95)),
                'maximum': float(np.max(yaw_errors)),
            },
        }

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / 'split_manifest.csv'
    temporary = manifest.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(prepared)
    temporary.replace(manifest)
    summary = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'purpose': 'exact-sample current-only versus temporal comparison',
        'deployable': False,
        'frozen_test_accessed': False,
        'scene_split_policy': 'leave-one-scene-out development folds',
        'pairing_policy': 'nearest world pose within fixed position/yaw gates',
        'maximum_pair_position_error_m': maximum_pair_position_error_m,
        'maximum_pair_yaw_error_deg': maximum_pair_yaw_error_deg,
        'input_views': {
            'current_only': 'current_lidar_evidence_bev (8 channels)',
            'temporal': 'lidar_evidence_bev (17 channels)',
        },
        'split_manifest': str(manifest),
        'split_manifest_sha256': _sha256(manifest),
        'source_manifest_sha256': {
            str(path): _sha256(path) for path in source_manifests
        },
        'sample_rows': len(prepared),
        'control_rows': sum(row['role'] == 'control' for row in prepared),
        'controlled_rows': sum(
            row['role'] == 'controlled' for row in prepared
        ),
        'scenes': scene_reports,
    }
    _atomic_json(output_directory / 'summary.json', summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--scene', action='append', required=True,
        help='SCENE_ID=ARCHIVE_DIRECTORY; repeat for each development scene',
    )
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument(
        '--maximum-pair-position-error-m', type=float, default=0.20
    )
    parser.add_argument(
        '--maximum-pair-yaw-error-deg', type=float, default=1.0
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    result = prepare_temporal_experiment(
        [_parse_scene(value) for value in args.scene],
        args.output_directory,
        args.maximum_pair_position_error_m,
        args.maximum_pair_yaw_error_deg,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
