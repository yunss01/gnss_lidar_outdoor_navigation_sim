#!/usr/bin/env python3
"""Validate and freeze a scene-isolated v2 traversability experiment.

The split is supplied as an explicit dataset specification.  This command
never performs a random split: every sample recorded in one physical scene
must remain in exactly one of train, validation, or test.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SPLITS = ('train', 'validation', 'test')
ROLES = ('control', 'controlled')
OUTPUT_FIELDS = (
    'split', 'scene_id', 'pair_group', 'object_family', 'role',
    'evaluation_slice',
    'source_session', 'source_sample_id', 'source_sample_path',
    'derived_sample_path', 'scan_fingerprint',
    'controlled_actor_return_count', 'controlled_obstacle_cell_count',
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _sample_policy(sample_path: Path) -> list[dict]:
    with np.load(sample_path, allow_pickle=False) as arrays:
        if 'controlled_actor_policy_json' not in arrays:
            raise ValueError(
                f'v2 sample lacks controlled_actor_policy_json: {sample_path}'
            )
        return json.loads(str(arrays['controlled_actor_policy_json'].item()))


def _controlled_cells(sample_path: Path) -> int:
    with np.load(sample_path, allow_pickle=False) as arrays:
        key = 'target_controlled_obstacle_point_count'
        if key not in arrays:
            raise ValueError(f'v2 sample lacks {key}: {sample_path}')
        return int(np.count_nonzero(arrays[key] > 0))


def _normalize_blueprint_family(blueprint: str) -> str:
    return str(blueprint).strip().split('.')[-1].lower()


def _validate_spec(spec: dict) -> list[dict]:
    if int(spec.get('schema_version', 0)) != 1:
        raise ValueError('dataset spec schema_version must be 1')
    if spec.get('target_contract') != 'independent_passable_obstacle_evidence':
        raise ValueError('dataset spec has an unsupported target_contract')
    entries = spec.get('sessions')
    if not isinstance(entries, list) or not entries:
        raise ValueError('dataset spec sessions must be a non-empty list')
    required = {
        'source_session', 'scene_id', 'split', 'object_family', 'role',
        'evaluation_slice',
    }
    seen_sessions = set()
    scene_split: dict[str, str] = {}
    controls: dict[tuple[str, str], int] = {}
    for entry in entries:
        if not isinstance(entry, dict) or required.difference(entry):
            raise ValueError('every dataset spec session needs: ' + ', '.join(
                sorted(required)
            ))
        session = str(entry['source_session']).strip()
        scene = str(entry['scene_id']).strip()
        split = str(entry['split']).strip()
        family = str(entry['object_family']).strip().lower()
        role = str(entry['role']).strip()
        evaluation_slice = str(entry['evaluation_slice']).strip()
        pair_group = str(entry.get('pair_group', scene)).strip()
        if not all((session, scene, pair_group, family, evaluation_slice)):
            raise ValueError('dataset spec fields cannot be blank')
        if session in seen_sessions:
            raise ValueError('duplicate source_session in spec: ' + session)
        if split not in SPLITS:
            raise ValueError('invalid split for ' + session + ': ' + split)
        if role not in ROLES:
            raise ValueError('invalid role for ' + session + ': ' + role)
        if scene in scene_split and scene_split[scene] != split:
            raise ValueError(
                f'scene leakage: {scene} appears in both '
                f'{scene_split[scene]} and {split}'
            )
        if role == 'control' and family != 'background':
            raise ValueError('control session family must be background')
        if role == 'controlled' and family == 'background':
            raise ValueError('controlled session needs an object family')
        if evaluation_slice == 'family_holdout' and split != 'test':
            raise ValueError('family_holdout is allowed only in test')
        seen_sessions.add(session)
        scene_split[scene] = split
        pair_key = (scene, pair_group)
        controls[pair_key] = (
            controls.get(pair_key, 0) + int(role == 'control')
        )
    missing_splits = [
        split for split in SPLITS
        if not any(entry['split'] == split for entry in entries)
    ]
    if missing_splits:
        raise ValueError(
            'dataset spec lacks splits: ' + ', '.join(missing_splits)
        )
    invalid_controls = [
        f'{scene}/{pair_group}'
        for (scene, pair_group), count in controls.items() if count != 1
    ]
    if invalid_controls:
        raise ValueError(
            'every scene/pair_group needs exactly one control session: '
            + ', '.join(sorted(invalid_controls))
        )
    return entries


def prepare_experiment(
    manifest: Path,
    dataset_spec: Path,
    output_directory: Path,
) -> dict:
    manifest = Path(manifest).expanduser().resolve()
    dataset_spec = Path(dataset_spec).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError('source manifest not found: ' + str(manifest))
    if not dataset_spec.is_file():
        raise FileNotFoundError('dataset spec not found: ' + str(dataset_spec))
    spec = json.loads(dataset_spec.read_text(encoding='utf-8'))
    entries = _validate_spec(spec)
    spec_by_session = {entry['source_session']: entry for entry in entries}

    all_rows = _read_csv(manifest)
    written = [row for row in all_rows if row.get('status') == 'written']
    rows_by_session: dict[str, list[dict]] = {}
    for row in written:
        rows_by_session.setdefault(row['source_session'], []).append(row)
    source_sessions = set(rows_by_session)
    specified_sessions = set(spec_by_session)
    if source_sessions != specified_sessions:
        missing = sorted(specified_sessions - source_sessions)
        unexpected = sorted(source_sessions - specified_sessions)
        raise ValueError(
            'manifest/spec session mismatch; missing={} unexpected={}'.format(
                missing, unexpected
            )
        )
    multiple = sorted(
        session for session, rows in rows_by_session.items() if len(rows) != 1
    )
    if multiple:
        raise ValueError(
            'controlled experiment requires one written sample per session: '
            + ', '.join(multiple)
        )

    holdout_families = {
        str(entry['object_family']).lower() for entry in entries
        if entry['evaluation_slice'] == 'family_holdout'
    }
    development_families = {
        str(entry['object_family']).lower() for entry in entries
        if entry['split'] in ('train', 'validation')
        and entry['role'] == 'controlled'
    }
    overlap = holdout_families & development_families
    if overlap:
        raise ValueError(
            'family_holdout leaks into train/validation: '
            + ', '.join(sorted(overlap))
        )

    prepared_rows = []
    for entry in entries:
        source = rows_by_session[entry['source_session']][0]
        sample_path = (
            Path(source['derived_sample_path']).expanduser().resolve()
        )
        if not sample_path.is_file():
            raise FileNotFoundError(
                'derived sample not found: ' + str(sample_path)
            )
        policy = _sample_policy(sample_path)
        if entry['role'] == 'control':
            if policy:
                raise ValueError(
                    'control session has controlled policy: '
                    + entry['source_session']
                )
        else:
            if len(policy) != 1:
                raise ValueError(
                    'controlled session must have one actor policy: '
                    + entry['source_session']
                )
            actual_family = _normalize_blueprint_family(
                policy[0].get('blueprint', '')
            )
            if actual_family != str(entry['object_family']).lower():
                raise ValueError(
                    f"object family mismatch for {entry['source_session']}: "
                    f"spec={entry['object_family']} sample={actual_family}"
                )
            if policy[0].get('disposition') != 'obstacle':
                raise ValueError('controlled family is not obstacle-labelled')
        controlled_cells = _controlled_cells(sample_path)
        if entry['role'] == 'control' and controlled_cells:
            raise ValueError('control contains controlled obstacle cells')
        if entry['role'] == 'controlled' and controlled_cells <= 0:
            raise ValueError('controlled sample has no obstacle target cells')
        prepared_rows.append({
            'split': entry['split'],
            'scene_id': entry['scene_id'],
            'pair_group': str(
                entry.get('pair_group', entry['scene_id'])
            ),
            'object_family': str(entry['object_family']).lower(),
            'role': entry['role'],
            'evaluation_slice': entry['evaluation_slice'],
            'source_session': source['source_session'],
            'source_sample_id': source['source_sample_id'],
            'source_sample_path': source['source_sample_path'],
            'derived_sample_path': str(sample_path),
            'scan_fingerprint': source.get('scan_fingerprint', ''),
            'controlled_actor_return_count': int(
                source.get('controlled_actor_return_count', 0) or 0
            ),
            'controlled_obstacle_cell_count': controlled_cells,
        })

    output_directory.mkdir(parents=True, exist_ok=True)
    split_manifest = output_directory / 'split_manifest.csv'
    temporary = split_manifest.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(prepared_rows)
    temporary.replace(split_manifest)

    split_counts = {
        split: sum(row['split'] == split for row in prepared_rows)
        for split in SPLITS
    }
    scene_counts = {
        split: len({
            row['scene_id'] for row in prepared_rows if row['split'] == split
        }) for split in SPLITS
    }
    pair_group_counts = {
        split: len({
            (row['scene_id'], row['pair_group'])
            for row in prepared_rows if row['split'] == split
        }) for split in SPLITS
    }
    controlled_cells = {
        split: sum(
            row['controlled_obstacle_cell_count'] for row in prepared_rows
            if row['split'] == split
        ) for split in SPLITS
    }
    source_manifest_sha256 = _sha256(manifest)
    dataset_spec_sha256 = _sha256(dataset_spec)
    split_manifest_sha256 = _sha256(split_manifest)
    fingerprint = hashlib.sha256(
        (source_manifest_sha256 + dataset_spec_sha256 + split_manifest_sha256)
        .encode('ascii')
    ).hexdigest()
    summary = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'name': spec.get('name', ''),
        'purpose': spec.get('purpose', ''),
        'test_policy': spec.get('test_policy', ''),
        'deployable': False,
        'target_contract': spec['target_contract'],
        'source_manifest': str(manifest),
        'dataset_spec': str(dataset_spec),
        'split_manifest': str(split_manifest),
        'source_manifest_sha256': source_manifest_sha256,
        'dataset_spec_sha256': dataset_spec_sha256,
        'split_manifest_sha256': split_manifest_sha256,
        'dataset_fingerprint': fingerprint,
        'sample_counts': split_counts,
        'scene_counts': scene_counts,
        'pair_group_counts': pair_group_counts,
        'controlled_obstacle_cell_counts': controlled_cells,
        'family_holdout_families': sorted(holdout_families),
        'families_by_split': {
            split: sorted({
                row['object_family'] for row in prepared_rows
                if row['split'] == split and row['role'] == 'controlled'
            }) for split in SPLITS
        },
        'scene_leakage_count': 0,
    }
    _atomic_json(output_directory / 'summary.json', summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset-spec', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = prepare_experiment(
        args.manifest, args.dataset_spec, args.output_directory
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
