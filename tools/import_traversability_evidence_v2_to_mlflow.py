#!/usr/bin/env python3
"""Register v2 dataset, model, and development-only diagnostic milestones.

The importer is idempotent and reference-only.  Large datasets, checkpoints,
images, and audit files stay at their original paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


TRACKING_URI = 'sqlite:////home/sukja/terrain_nav_data/mlflow/mlflow.db'
ARTIFACT_ROOT = Path(
    '/home/sukja/terrain_nav_data/mlflow/artifacts/'
    'traversability_evidence_v2'
)
EXPERIMENT_NAME = 'traversability_evidence_v2'
DATASET_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_controlled_dataset_20260918'
)
EXPERIMENT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_controlled_experiment_20260918'
)
AUDIT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/visualizations/'
    'v2_controlled_dataset_20260918'
)
MODEL_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/pilot_paired_20260918'
)
CV_DIRECTORIES = {
    'local_evidence': Path(
        '/home/sukja/terrain_nav_data/learning/models/'
        'traversability_evidence_v2/cv_local_evidence_lr1e3_20260921'
    ),
    'unet_context': Path(
        '/home/sukja/terrain_nav_data/learning/models/'
        'traversability_evidence_v2/cv_unet_context_lr1e3_20260921'
    ),
}
SCENE06_DATASET_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_context_diagnostic_dataset_20260921'
)
SCENE06_EXPERIMENT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_scene06_diagnostic_experiment_20260921'
)
SCENE06_AUDIT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/visualizations/'
    'v2_scene06_context_diagnostic_20260921'
)
SCENE06_MODEL_DIRECTORIES = {
    'unet_context': Path(
        '/home/sukja/terrain_nav_data/learning/models/'
        'traversability_evidence_v2/'
        'scene06_diagnostic_unet_context_20260921'
    ),
    'bounded_context_cnn': Path(
        '/home/sukja/terrain_nav_data/learning/models/'
        'traversability_evidence_v2/'
        'scene06_diagnostic_bounded_context_20260921'
    ),
}
SCENE06_SELECTION_PATH = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'scene06_diagnostic_selection_20260921.json'
)
SCENE07_DATASET_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_scene07_context_replication_20260921'
)
SCENE07_AUDIT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/visualizations/'
    'v2_scene07_context_replication_20260921'
)
SCENE07_EVALUATION_PATH = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'scene07_fixed_unet_clearance_replication_20260921.json'
)
SCENE07_PASSABLE_EVALUATION_PATH = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'scene07_fixed_unet_clearance_poseB_passable_20260921.json'
)
SCENE07_SUMMARY_PATH = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'scene07_fixed_policy_replication_20260921_summary.json'
)
SCENE01_07_CV_DIRECTORIES = {
    'shared_unet_context': {
        'architecture': 'unet_context',
        'directory': Path(
            '/home/sukja/terrain_nav_data/learning/models/'
            'traversability_evidence_v2/'
            'cv_unet_context_scenes01_07_fixed_clearance_20260921'
        ),
    },
    'decoupled_unet_context': {
        'architecture': 'decoupled_unet_context',
        'directory': Path(
            '/home/sukja/terrain_nav_data/learning/models/'
            'traversability_evidence_v2/'
            'cv_decoupled_unet_scenes01_07_fixed_clearance_20260921'
        ),
    },
}
SCENE01_07_SELECTION_PATH = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'scene01_07_architecture_selection_20260921.json'
)
BRANCH_INDEPENDENT_CV_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'cv_decoupled_unet_independent_selection_scenes01_07_20260921'
)
TRAINING_PROTOCOL_SELECTION_PATH = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'scene01_07_training_protocol_selection_20260921.json'
)
TEMPORAL_EXPERIMENT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_temporal_representation_experiment_scenes09_11_20260921'
)
TEMPORAL_CV_DIRECTORIES = {
    'current_only': Path(
        '/home/sukja/terrain_nav_data/learning/models/'
        'traversability_evidence_v2/'
        'cv_temporal_representation_current_only_scenes09_11_20260921'
    ),
    'temporal': Path(
        '/home/sukja/terrain_nav_data/learning/models/'
        'traversability_evidence_v2/'
        'cv_temporal_representation_17ch_scenes09_11_20260921'
    ),
}
TEMPORAL_SELECTION_PATH = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'temporal_representation_selection_scenes09_11_20260921.json'
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('expected JSON object: ' + str(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _flatten(value, prefix='') -> dict[str, str]:
    output = {}
    if not isinstance(value, dict):
        return output
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(item, dict):
            output.update(_flatten(item, name))
        elif isinstance(item, (list, tuple)):
            output[name] = json.dumps(item, separators=(',', ':'))
        elif item is None:
            output[name] = 'null'
        else:
            output[name] = str(item)
    return output


def _metrics(value, prefix='') -> dict[str, float]:
    output = {}
    if not isinstance(value, dict):
        return output
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(item, dict):
            output.update(_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            number = float(item)
            if math.isfinite(number):
                output[name] = number
    return output


def _chunks(value: dict, size=90):
    items = list(value.items())
    for start in range(0, len(items), size):
        yield dict(items[start:start + size])


def _experiment_id() -> str:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=ARTIFACT_ROOT.as_uri(),
        tags={
            'project': '3d_lidar_gnss_outdoor_navigation',
            'target_contract': 'independent_passable_obstacle_evidence',
            'artifact_policy': 'reference_only_no_copy',
        },
    )


def _existing(client, experiment_id: str, import_key: str):
    escaped = import_key.replace("'", "\\'")
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.import_key = '{escaped}'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _log_params(values: dict) -> None:
    for chunk in _chunks(values):
        mlflow.log_params(chunk)


def _log_metrics(values: dict, step=None) -> None:
    for chunk in _chunks(values):
        mlflow.log_metrics(chunk, step=step)


def _dataset_run(experiment_id: str, client: MlflowClient) -> str:
    summary_path = EXPERIMENT_DIRECTORY / 'summary.json'
    audit_path = AUDIT_DIRECTORY / 'summary.json'
    summary = _load(summary_path)
    audit = _load(audit_path)
    fingerprint = summary['dataset_fingerprint']
    import_key = 'evidence_v2_dataset:' + fingerprint
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP dataset milestone:', existing)
        return existing
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='evidence_v2_dataset_20260918',
        description=(
            'Frozen scene-isolated controlled dataset milestone. Reference '
            'metadata only; raw/derived arrays and audit images are not '
            'copied.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'dataset_milestone',
            'deployable': 'false',
            'dataset_fingerprint': fingerprint,
            'target_contract': summary['target_contract'],
            'scene_leakage': 'false',
            'family_holdout': ','.join(summary['family_holdout_families']),
            'artifact_policy': 'reference_only_no_copy',
            'dataset_directory': str(DATASET_DIRECTORY),
            'experiment_directory': str(EXPERIMENT_DIRECTORY),
            'audit_directory': str(AUDIT_DIRECTORY),
        })
        _log_params({
            'dataset.source_manifest_sha256': (
                summary['source_manifest_sha256']
            ),
            'dataset.spec_sha256': summary['dataset_spec_sha256'],
            'dataset.split_manifest_sha256': summary['split_manifest_sha256'],
            'dataset.split_manifest': summary['split_manifest'],
            'dataset.spec': summary['dataset_spec'],
            'storage.files_copied': '0',
        })
        values = {}
        values.update(_metrics(summary['sample_counts'], 'samples'))
        values.update(_metrics(summary['scene_counts'], 'scenes'))
        values.update(_metrics(
            summary['controlled_obstacle_cell_counts'],
            'controlled_obstacle_cells',
        ))
        values.update(_metrics(audit, 'audit'))
        _log_metrics(values)
        return run.info.run_id


def _model_run(
    experiment_id: str,
    client: MlflowClient,
    dataset_run_id: str,
) -> str:
    paths = {
        name: MODEL_DIRECTORY / file_name
        for name, file_name in {
            'config': 'training_config.json',
            'result': 'result.json',
            'history': 'history.csv',
            'checkpoint': 'best.pt',
            'selection': 'model_selection.json',
            'validation': 'validation_metrics.json',
            'test': 'test_metrics_frozen.json',
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'missing v2 model outputs: ' + ', '.join(missing)
        )
    documents = {
        name: _load(path) for name, path in paths.items()
        if path.suffix == '.json'
    }
    checkpoint_hash = _sha256(paths['checkpoint'])
    import_key = 'evidence_v2_model:' + checkpoint_hash
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP model pilot:', existing)
        return existing
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='evidence_v2_paired_pilot_20260918',
        description=(
            'Non-deployable paired-counterfactual pilot. Frozen test includes '
            'a motorhelmet miss; this run must not be presented as validated.'
        ),
    ) as run:
        test_family = documents['test']['family_metrics']
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'paired_counterfactual_model_pilot',
            'deployable': 'false',
            'test_outcome': 'failed_controlled_motorhelmet_instance',
            'known_failure': 'scene05 motorhelmet 0/2 controlled cells',
            'target_contract': 'independent_passable_obstacle_evidence',
            'dataset_run_id': dataset_run_id,
            'dataset_fingerprint': _load(
                EXPERIMENT_DIRECTORY / 'summary.json'
            )['dataset_fingerprint'],
            'checkpoint_sha256': checkpoint_hash,
            'model_directory': str(MODEL_DIRECTORY),
            'artifact_policy': 'reference_only_no_copy',
            'family_holdout_interpretation': 'single-sample sentinel only',
        })
        params = _flatten(documents['config'], 'config')
        params.update(_flatten(documents['selection'], 'selection'))
        params.update({
            'checkpoint.path': str(paths['checkpoint']),
            'checkpoint.bytes': str(paths['checkpoint'].stat().st_size),
            'storage.files_copied': '0',
        })
        _log_params(params)
        with paths['history'].open(newline='', encoding='utf-8') as stream:
            history = list(csv.DictReader(stream))
        for row in history:
            epoch = int(row['epoch'])
            values = {}
            for key, item in row.items():
                if key == 'epoch' or item in ('', None):
                    continue
                number = float(item)
                if math.isfinite(number):
                    values[key] = number
            _log_metrics(values, step=epoch)
        _log_metrics(_metrics(documents['result'], 'result'))
        _log_metrics(_metrics(documents['validation'], 'validation'))
        _log_metrics(_metrics(documents['test'], 'test'))
        mlflow.log_metric(
            'test.motorhelmet_instance_detected',
            float(test_family['motorhelmet']['instance_any_detection_rate']),
        )
        mlflow.log_metric(
            'test.family_holdout_trashcan_instance_detected',
            float(test_family['trashcan01']['instance_any_detection_rate']),
        )
        return run.info.run_id


def _cross_validation_run(
    experiment_id: str,
    client: MlflowClient,
    dataset_run_id: str,
    architecture: str,
    directory: Path,
) -> str:
    summary_path = directory / 'cross_validation_summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError('missing CV summary: ' + str(summary_path))
    summary = _load(summary_path)
    if summary['architecture'] != architecture:
        raise ValueError(
            f'CV architecture mismatch: {architecture} != '
            f'{summary["architecture"]}'
        )
    if summary.get('frozen_test_accessed') is not False:
        raise ValueError('refusing to import CV that accessed frozen test')
    summary_hash = _sha256(summary_path)
    import_key = 'evidence_v2_development_cv:' + summary_hash
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP development CV:', architecture, existing)
        return existing

    first_fold = summary['folds'][0]
    config_path = (
        Path(first_fold['model_directory']) / 'training_config.json'
    )
    config = _load(config_path)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f'evidence_v2_cv_{architecture}_20260921',
        description=(
            'Development-only leave-one-scene-out comparison. Frozen test '
            'rows were retained in manifests but never loaded or evaluated.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'development_scene_cross_validation',
            'architecture': architecture,
            'deployable': 'false',
            'frozen_test_accessed': 'false',
            'target_contract': 'independent_passable_obstacle_evidence',
            'dataset_run_id': dataset_run_id,
            'summary_sha256': summary_hash,
            'cv_directory': str(directory),
            'artifact_policy': 'reference_only_no_copy',
        })
        _log_params({
            'cv.summary': str(summary_path),
            'cv.fold_count': str(summary['fold_count']),
            'cv.development_scenes': json.dumps(
                summary['development_scenes'], separators=(',', ':')
            ),
            'cv.test_policy': summary['test_policy'],
            'training.epochs': str(config['epochs']),
            'training.learning_rate': str(config['learning_rate']),
            'training.batch_size': str(config['batch_size']),
            'training.base_channels': str(
                config['model']['base_channels']
            ),
            'training.paired_counterfactual_weight': str(
                config['paired_counterfactual_weight']
            ),
            'storage.files_copied': '0',
        })
        _log_metrics(_metrics(summary['aggregate'], 'cv'))
        for fold in summary['folds']:
            scene = fold['validation_scene']
            evaluation = fold['evaluation']
            values = {
                f'fold.{scene}.obstacle_iou': (
                    evaluation['metrics']['obstacle_head']['iou']
                ),
                f'fold.{scene}.passable_iou': (
                    evaluation['metrics']['passable_head']['iou']
                ),
                f'fold.{scene}.controlled_cell_recall': (
                    evaluation['metrics'][
                        'controlled_obstacle_cell_recall'
                    ]
                ),
                f'fold.{scene}.controlled_instance_detection_rate': (
                    evaluation['metrics'][
                        'controlled_instance_any_detection_rate'
                    ]
                ),
            }
            _log_metrics(values)
        return run.info.run_id


def _scene06_dataset_run(
    experiment_id: str,
    client: MlflowClient,
) -> str:
    summary_path = SCENE06_EXPERIMENT_DIRECTORY / 'summary.json'
    audit_path = SCENE06_AUDIT_DIRECTORY / 'summary.json'
    summary = _load(summary_path)
    audit = _load(audit_path)
    fingerprint = summary['dataset_fingerprint']
    import_key = 'evidence_v2_scene06_dataset:' + fingerprint
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP scene06 dataset milestone:', existing)
        return existing

    if summary.get('test_policy') != (
        'scene05 is a consumed historical test and must not be evaluated '
        'or used for selection'
    ):
        raise ValueError('scene06 dataset has unexpected test policy')
    if summary.get('deployable') is not False:
        raise ValueError('scene06 diagnostic dataset must not be deployable')

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='evidence_v2_scene06_diagnostic_dataset_20260921',
        description=(
            'Development-only context diagnostic. Scenes 01-04 train, '
            'scene06 validation, and consumed scene05 retained in provenance '
            'but never loaded for evaluation or model selection.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'development_context_diagnostic_dataset',
            'deployable': 'false',
            'consumed_test_accessed': 'false',
            'validation_scene': 'scene06',
            'dataset_fingerprint': fingerprint,
            'target_contract': summary['target_contract'],
            'artifact_policy': 'reference_only_no_copy',
            'dataset_directory': str(SCENE06_DATASET_DIRECTORY),
            'experiment_directory': str(SCENE06_EXPERIMENT_DIRECTORY),
            'audit_directory': str(SCENE06_AUDIT_DIRECTORY),
        })
        _log_params({
            'dataset.source_manifest_sha256': (
                summary['source_manifest_sha256']
            ),
            'dataset.spec_sha256': summary['dataset_spec_sha256'],
            'dataset.split_manifest_sha256': (
                summary['split_manifest_sha256']
            ),
            'dataset.split_manifest': summary['split_manifest'],
            'dataset.spec': summary['dataset_spec'],
            'dataset.test_policy': summary['test_policy'],
            'storage.files_copied': '0',
        })
        values = {}
        values.update(_metrics(summary['sample_counts'], 'samples'))
        values.update(_metrics(summary['scene_counts'], 'scenes'))
        values.update(_metrics(summary['pair_group_counts'], 'pair_groups'))
        values.update(_metrics(
            summary['controlled_obstacle_cell_counts'],
            'controlled_obstacle_cells',
        ))
        values.update(_metrics(audit, 'scene06_audit'))
        _log_metrics(values)
        return run.info.run_id


def _scene06_model_run(
    experiment_id: str,
    client: MlflowClient,
    dataset_run_id: str,
    architecture_label: str,
    directory: Path,
) -> str:
    paths = {
        name: directory / file_name
        for name, file_name in {
            'config': 'training_config.json',
            'result': 'result.json',
            'history': 'history.csv',
            'checkpoint': 'best.pt',
            'validation': 'validation_metrics.json',
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'missing scene06 diagnostic outputs: ' + ', '.join(missing)
        )
    config = _load(paths['config'])
    result = _load(paths['result'])
    validation = _load(paths['validation'])
    selection = _load(SCENE06_SELECTION_PATH)
    checkpoint_hash = _sha256(paths['checkpoint'])
    validation_hash = _sha256(paths['validation'])
    import_key = (
        'evidence_v2_scene06_model:' + checkpoint_hash + ':' +
        validation_hash
    )
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP scene06 diagnostic model:', architecture_label, existing)
        return existing

    if validation.get('split') != 'validation':
        raise ValueError('scene06 model result is not a validation result')
    if validation.get('scenes') != ['scene06']:
        raise ValueError('scene06 model result has unexpected scenes')
    if validation.get('deployable') is not False:
        raise ValueError('scene06 diagnostic model must not be deployable')
    selected = architecture_label == selection['learned_component']
    outcome = (
        'selected_learned_component'
        if selected else 'rejected_as_replacement'
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=(
            f'evidence_v2_scene06_{architecture_label}_20260921'
        ),
        description=(
            'Fair scene06 context diagnostic trained from scratch on scenes '
            '01-04 only. Metrics include raw learned heads and the fixed '
            'vehicle-clearance hard-obstacle fusion. Consumed scene05 was '
            'not accessed.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'development_context_diagnostic_model',
            'architecture': architecture_label,
            'selection_outcome': outcome,
            'deployable': 'false',
            'consumed_test_accessed': 'false',
            'validation_scene': 'scene06',
            'dataset_run_id': dataset_run_id,
            'dataset_fingerprint': selection['dataset_fingerprint'],
            'checkpoint_sha256': checkpoint_hash,
            'validation_metrics_sha256': validation_hash,
            'model_directory': str(directory),
            'artifact_policy': 'reference_only_no_copy',
            'safety_fusion': selection['safety_fusion'],
        })
        params = _flatten(config, 'config')
        params.update({
            'checkpoint.path': str(paths['checkpoint']),
            'checkpoint.bytes': str(paths['checkpoint'].stat().st_size),
            'selection.path': str(SCENE06_SELECTION_PATH),
            'selection.next_gate': selection['next_gate'],
            'selection.outcome': outcome,
            'test.consumed_scene05_accessed': 'false',
            'storage.files_copied': '0',
        })
        _log_params(params)
        with paths['history'].open(newline='', encoding='utf-8') as stream:
            history = list(csv.DictReader(stream))
        for row in history:
            epoch = int(row['epoch'])
            values = {}
            for key, item in row.items():
                if key == 'epoch' or item in ('', None):
                    continue
                number = float(item)
                if math.isfinite(number):
                    values[key] = number
            _log_metrics(values, step=epoch)
        _log_metrics(_metrics(result, 'result'))
        _log_metrics(_metrics(validation, 'validation'))
        return run.info.run_id


def _scene07_dataset_run(
    experiment_id: str,
    client: MlflowClient,
) -> str:
    summary_path = SCENE07_DATASET_DIRECTORY / 'summary.json'
    manifest_path = SCENE07_DATASET_DIRECTORY / 'manifest.csv'
    evaluation_manifest_path = (
        SCENE07_DATASET_DIRECTORY / 'evaluation_manifest.csv'
    )
    audit_path = SCENE07_AUDIT_DIRECTORY / 'summary.json'
    summary = _load(summary_path)
    audit = _load(audit_path)
    identity = ':'.join((
        _sha256(summary_path),
        _sha256(manifest_path),
        _sha256(evaluation_manifest_path),
    ))
    import_key = 'evidence_v2_scene07_dataset:' + identity
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP scene07 replication dataset:', existing)
        return existing
    if summary.get('written_samples') != 8:
        raise ValueError('scene07 replication dataset must contain 8 scans')
    if audit.get('failed_samples') != 0:
        raise ValueError('refusing to import failed scene07 audit')

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='evidence_v2_scene07_replication_dataset_20260921',
        description=(
            'New-context replication dataset with matched Pose A control, '
            'six controlled obstacle scans, and a separate Pose B passable '
            'control. Reference metadata only.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'fixed_policy_replication_dataset',
            'deployable': 'false',
            'evaluation_scene': 'scene07',
            'artifact_policy': 'reference_only_no_copy',
            'dataset_directory': str(SCENE07_DATASET_DIRECTORY),
            'audit_directory': str(SCENE07_AUDIT_DIRECTORY),
            'evaluation_manifest': str(evaluation_manifest_path),
        })
        _log_params({
            'dataset.summary_sha256': _sha256(summary_path),
            'dataset.manifest_sha256': _sha256(manifest_path),
            'dataset.evaluation_manifest_sha256': (
                _sha256(evaluation_manifest_path)
            ),
            'dataset.raw_root': summary['raw_root'],
            'storage.files_copied': '0',
        })
        values = {}
        values.update(_metrics(summary, 'dataset'))
        values.update(_metrics(audit, 'audit'))
        _log_metrics(values)
        return run.info.run_id


def _scene07_evaluation_run(
    experiment_id: str,
    client: MlflowClient,
    dataset_run_id: str,
    scene06_unet_run_id: str,
) -> str:
    evaluation = _load(SCENE07_EVALUATION_PATH)
    passable_evaluation = _load(SCENE07_PASSABLE_EVALUATION_PATH)
    summary = _load(SCENE07_SUMMARY_PATH)
    identity = ':'.join((
        _sha256(SCENE07_EVALUATION_PATH),
        _sha256(SCENE07_PASSABLE_EVALUATION_PATH),
        _sha256(SCENE07_SUMMARY_PATH),
    ))
    import_key = 'evidence_v2_scene07_fixed_policy:' + identity
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP scene07 fixed-policy evaluation:', existing)
        return existing
    if summary.get('retraining_performed') is not False:
        raise ValueError('scene07 replication must not retrain the model')
    if summary.get('threshold_tuning_performed') is not False:
        raise ValueError('scene07 replication must not tune thresholds')
    if evaluation.get('scenes') != ['scene07']:
        raise ValueError('scene07 evaluation has unexpected scenes')
    if evaluation.get('checkpoint_sha256') != summary.get(
        'checkpoint_sha256'
    ):
        raise ValueError('scene07 checkpoint hash mismatch')

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='evidence_v2_scene07_fixed_policy_replication_20260921',
        description=(
            'No-retraining, no-threshold-tuning replication of the '
            'scene06-selected U-Net plus fixed 15 cm vehicle-clearance '
            'fusion on a new scene07 context.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'fixed_policy_context_replication',
            'deployable': 'false',
            'evaluation_scene': 'scene07',
            'retraining_performed': 'false',
            'threshold_tuning_performed': 'false',
            'raw_learned_motorhelmet_gate': 'failed',
            'fused_instance_safety_gate': 'passed',
            'dataset_run_id': dataset_run_id,
            'source_checkpoint_run_id': scene06_unet_run_id,
            'checkpoint_sha256': evaluation['checkpoint_sha256'],
            'artifact_policy': 'reference_only_no_copy',
        })
        _log_params({
            'evaluation.path': str(SCENE07_EVALUATION_PATH),
            'evaluation.sha256': _sha256(SCENE07_EVALUATION_PATH),
            'evaluation.passable_poseB_path': str(
                SCENE07_PASSABLE_EVALUATION_PATH
            ),
            'evaluation.passable_poseB_sha256': _sha256(
                SCENE07_PASSABLE_EVALUATION_PATH
            ),
            'replication.summary_path': str(SCENE07_SUMMARY_PATH),
            'replication.summary_sha256': _sha256(SCENE07_SUMMARY_PATH),
            'replication.next_step': summary['next_step'],
            'storage.files_copied': '0',
        })
        _log_metrics(_metrics(evaluation, 'scene07'))
        _log_metrics(_metrics(passable_evaluation, 'scene07_poseB'))
        return run.info.run_id


def _scene01_07_cross_validation_run(
    experiment_id: str,
    client: MlflowClient,
    candidate_label: str,
    architecture: str,
    directory: Path,
    scene06_dataset_run_id: str,
    scene07_dataset_run_id: str,
) -> str:
    summary_path = directory / 'cross_validation_summary.json'
    summary = _load(summary_path)
    selection = _load(SCENE01_07_SELECTION_PATH)
    if summary.get('architecture') != architecture:
        raise ValueError(
            f'scene01-07 architecture mismatch: {architecture} != '
            f'{summary.get("architecture")}'
        )
    if summary.get('untouched_test_accessed') is not False:
        raise ValueError('refusing to import CV that accessed untouched test')
    if summary.get('development_scenes') != [
        'scene01', 'scene02', 'scene03', 'scene04', 'scene05',
        'scene06', 'scene07',
    ]:
        raise ValueError('unexpected scene01-07 development scene set')
    if summary.get('promoted_consumed_test_scenes') != ['scene05']:
        raise ValueError('scene05 was not explicitly promoted to development')
    if selection.get('untouched_test_accessed') is not False:
        raise ValueError('selection record accessed untouched test')

    summary_hash = _sha256(summary_path)
    selection_hash = _sha256(SCENE01_07_SELECTION_PATH)
    import_key = ':'.join((
        'evidence_v2_scene01_07_cv', summary_hash, selection_hash,
    ))
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP scene01-07 development CV:', candidate_label, existing)
        return existing

    first_fold = summary['folds'][0]
    config_path = Path(first_fold['model_directory']) / 'training_config.json'
    config = _load(config_path)
    selected_for_development = (
        selection['selection']['development_architecture'] == architecture
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f'evidence_v2_scene01_07_cv_{candidate_label}_20260921',
        description=(
            'Seven-fold development-only scene holdout comparison. The '
            'historically consumed scene05 is explicitly development data; '
            'the future scene08 test is absent and untouched. No candidate '
            'passed every predeclared deployment gate.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'development_scene01_07_cross_validation',
            'candidate_label': candidate_label,
            'architecture': architecture,
            'selection_outcome': (
                'development_only_candidate'
                if selected_for_development else 'rejected_as_candidate'
            ),
            'deployable': 'false',
            'final_checkpoint_frozen': 'false',
            'scene05_status': 'promoted_consumed_test_to_development',
            'scene08_status': 'untouched_absent',
            'untouched_test_accessed': 'false',
            'summary_sha256': summary_hash,
            'selection_sha256': selection_hash,
            'cv_directory': str(directory),
            'artifact_policy': 'reference_only_no_copy',
        })
        _log_params({
            'cv.summary': str(summary_path),
            'cv.fold_count': str(summary['fold_count']),
            'cv.development_scenes': json.dumps(
                summary['development_scenes'], separators=(',', ':')
            ),
            'cv.source_split_manifests': json.dumps(
                summary['source_split_manifests'], separators=(',', ':')
            ),
            'cv.test_policy': summary['test_policy'],
            'cv.evaluation_only_summary_refresh': str(
                summary.get('evaluation_only', False)
            ).lower(),
            'training.epochs': str(config['epochs']),
            'training.learning_rate': str(config['learning_rate']),
            'training.batch_size': str(config['batch_size']),
            'training.base_channels': str(
                config['model']['base_channels']
            ),
            'training.paired_counterfactual_weight': str(
                config['paired_counterfactual_weight']
            ),
            'selection.path': str(SCENE01_07_SELECTION_PATH),
            'selection.development_architecture': selection[
                'selection'
            ]['development_architecture'],
            'selection.deployment_architecture': 'none',
            'dataset.scene06_run_id': scene06_dataset_run_id,
            'dataset.scene07_run_id': scene07_dataset_run_id,
            'storage.files_copied': '0',
        })
        _log_metrics(_metrics(summary['aggregate'], 'cv'))
        for fold in summary['folds']:
            scene = fold['validation_scene']
            evaluation = fold['evaluation']
            metrics = evaluation['metrics']
            fusion = evaluation['vehicle_clearance_fusion']
            _log_metrics({
                f'fold.{scene}.obstacle_iou': (
                    metrics['obstacle_head']['iou']
                ),
                f'fold.{scene}.passable_iou': (
                    metrics['passable_head']['iou']
                ),
                f'fold.{scene}.controlled_instance_detection_rate': (
                    metrics['controlled_instance_any_detection_rate']
                ),
                f'fold.{scene}.controlled_unsafe_passable_instance_rate': (
                    metrics['controlled_unsafe_passable_instance_rate']
                ),
                f'fold.{scene}.fused_instance_detection_rate': (
                    fusion['controlled_instance_detection_rate']
                ),
                f'fold.{scene}.fused_unsafe_passable_instance_rate': (
                    fusion['controlled_unsafe_passable_instance_rate']
                ),
                f'fold.{scene}.fused_control_false_obstacle_rate': (
                    fusion['paired_control_false_obstacle_rate']
                ),
            })
        return run.info.run_id


def _branch_independent_protocol_run(
    experiment_id: str,
    client: MlflowClient,
    scene06_dataset_run_id: str,
    scene07_dataset_run_id: str,
) -> str:
    summary_path = (
        BRANCH_INDEPENDENT_CV_DIRECTORY
        / 'cross_validation_summary.json'
    )
    summary = _load(summary_path)
    selection = _load(TRAINING_PROTOCOL_SELECTION_PATH)
    if summary.get('architecture') != 'decoupled_unet_context':
        raise ValueError('branch-independent CV architecture mismatch')
    if summary.get('untouched_test_accessed') is not False:
        raise ValueError('branch-independent CV accessed untouched test')
    if selection['fixed_conditions']['scene08_accessed'] is not False:
        raise ValueError('training protocol selection accessed scene08')
    if selection['selection']['deployable'] is not False:
        raise ValueError('training protocol candidate must not be deployable')

    summary_hash = _sha256(summary_path)
    selection_hash = _sha256(TRAINING_PROTOCOL_SELECTION_PATH)
    import_key = ':'.join((
        'evidence_v2_branch_independent_protocol',
        summary_hash,
        selection_hash,
    ))
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP branch-independent protocol:', existing)
        return existing

    first_fold = summary['folds'][0]
    config = _load(
        Path(first_fold['model_directory']) / 'training_config.json'
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=(
            'evidence_v2_scene01_07_branch_independent_protocol_20260921'
        ),
        description=(
            'Seven-fold development-only rerun with separate optimizers, '
            'schedulers, early stopping, and branch checkpoint assembly. '
            'Data, thresholds, clearance policy, and fold seeds are fixed. '
            'Scene08 is absent and untouched; the result is not deployable.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'development_training_protocol_comparison',
            'architecture': 'decoupled_unet_context',
            'training_controller_policy': 'branch_independent',
            'selection_outcome': 'preferred_development_protocol',
            'deployable': 'false',
            'final_checkpoint_frozen': 'false',
            'scene08_status': 'untouched_absent',
            'untouched_test_accessed': 'false',
            'summary_sha256': summary_hash,
            'selection_sha256': selection_hash,
            'cv_directory': str(BRANCH_INDEPENDENT_CV_DIRECTORY),
            'artifact_policy': 'reference_only_no_copy',
        })
        _log_params({
            'cv.summary': str(summary_path),
            'cv.fold_count': str(summary['fold_count']),
            'cv.development_scenes': json.dumps(
                summary['development_scenes'], separators=(',', ':')
            ),
            'training.controller_policy': config[
                'training_controller_policy'
            ],
            'training.checkpoint_selection_policy': config[
                'checkpoint_selection_policy'
            ],
            'training.epochs': str(config['epochs']),
            'training.learning_rate': str(config['learning_rate']),
            'training.batch_size': str(config['batch_size']),
            'training.minimum_controlled_instance_recall': str(
                config['minimum_controlled_instance_recall']
            ),
            'selection.path': str(TRAINING_PROTOCOL_SELECTION_PATH),
            'selection.next_experiment': selection['selection'][
                'next_experiment'
            ],
            'dataset.scene06_run_id': scene06_dataset_run_id,
            'dataset.scene07_run_id': scene07_dataset_run_id,
            'storage.files_copied': '0',
        })
        _log_metrics(_metrics(summary['aggregate'], 'cv'))
        _log_metrics({
            'selection.raw_gate_passing_folds': float(
                selection['candidate']['folds_passing_raw_selection_gates']
            ),
            'selection.fold_count': float(
                selection['candidate']['fold_count']
            ),
        })
        for fold in summary['folds']:
            scene = fold['validation_scene']
            training = fold['training_result']
            evaluation = fold['evaluation']
            _log_metrics({
                f'fold.{scene}.passable_epoch': training['passable_epoch'],
                f'fold.{scene}.obstacle_epoch': training['obstacle_epoch'],
                f'fold.{scene}.raw_selection_gate_passed': float(
                    training['selection_safety_gates_passed']
                ),
                f'fold.{scene}.passable_iou': evaluation['metrics'][
                    'passable_head'
                ]['iou'],
                f'fold.{scene}.obstacle_iou': evaluation['metrics'][
                    'obstacle_head'
                ]['iou'],
                f'fold.{scene}.controlled_instance_detection_rate': (
                    evaluation['metrics'][
                        'controlled_instance_any_detection_rate'
                    ]
                ),
                f'fold.{scene}.fused_instance_detection_rate': (
                    evaluation['vehicle_clearance_fusion'][
                        'controlled_instance_detection_rate'
                    ]
                ),
                f'fold.{scene}.fused_control_false_obstacle_rate': (
                    evaluation['vehicle_clearance_fusion'][
                        'paired_control_false_obstacle_rate'
                    ]
                ),
            })
        return run.info.run_id


def _temporal_representation_comparison_run(
    experiment_id: str,
    client: MlflowClient,
    protocol_run_id: str,
) -> str:
    experiment_summary_path = TEMPORAL_EXPERIMENT_DIRECTORY / 'summary.json'
    experiment_summary = _load(experiment_summary_path)
    selection = _load(TEMPORAL_SELECTION_PATH)
    summaries = {
        label: _load(directory / 'cross_validation_summary.json')
        for label, directory in TEMPORAL_CV_DIRECTORIES.items()
    }

    if experiment_summary.get('frozen_test_accessed') is not False:
        raise ValueError('temporal experiment accessed a frozen test')
    if selection['fixed_conditions']['scene08_accessed'] is not False:
        raise ValueError('temporal selection accessed scene08')
    if selection['selection']['deployable'] is not False:
        raise ValueError('temporal comparison must not be deployable')
    if not selection['fixed_conditions']['fold_manifests_byte_identical']:
        raise ValueError('temporal comparison fold manifests differ')
    for label, summary in summaries.items():
        if summary.get('architecture') != 'decoupled_unet_context':
            raise ValueError('temporal CV architecture mismatch: ' + label)
        if summary.get('untouched_test_accessed') is not False:
            raise ValueError('temporal CV accessed untouched test: ' + label)
        variants = {
            fold['evaluation']['evidence_input_variant']
            for fold in summary['folds']
        }
        if variants != {label}:
            raise ValueError('temporal CV input variant mismatch: ' + label)

    identity = ':'.join((
        _sha256(experiment_summary_path),
        _sha256(
            TEMPORAL_CV_DIRECTORIES['current_only']
            / 'cross_validation_summary.json'
        ),
        _sha256(
            TEMPORAL_CV_DIRECTORIES['temporal']
            / 'cross_validation_summary.json'
        ),
        _sha256(TEMPORAL_SELECTION_PATH),
    ))
    import_key = 'evidence_v2_temporal_representation:' + identity
    existing = _existing(client, experiment_id, import_key)
    if existing:
        print('SKIP temporal representation comparison:', existing)
        return existing

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=(
            'evidence_v2_temporal_representation_comparison_20260921'
        ),
        description=(
            'Exact-sample three-scene comparison of current-only 8-channel '
            'evidence and ego-motion-aligned 17-channel temporal evidence. '
            'The temporal replacement was rejected because gains were not '
            'consistent across scenes. Reference metadata only.'
        ),
    ) as run:
        mlflow.set_tags({
            'import_key': import_key,
            'run_kind': 'development_temporal_representation_comparison',
            'architecture': 'decoupled_unet_context',
            'training_controller_policy': 'branch_independent',
            'selected_representation': 'current_only',
            'temporal_replacement_accepted': 'false',
            'deployable': 'false',
            'navigation_authority': 'false',
            'scene08_status': 'untouched_absent',
            'untouched_test_accessed': 'false',
            'artifact_policy': 'reference_only_no_copy',
        })
        _log_params({
            'experiment.summary': str(experiment_summary_path),
            'experiment.summary_sha256': _sha256(
                experiment_summary_path
            ),
            'experiment.split_manifest': experiment_summary[
                'split_manifest'
            ],
            'experiment.split_manifest_sha256': experiment_summary[
                'split_manifest_sha256'
            ],
            'experiment.sample_rows': str(
                experiment_summary['sample_rows']
            ),
            'comparison.current_only_summary': str(
                TEMPORAL_CV_DIRECTORIES['current_only']
                / 'cross_validation_summary.json'
            ),
            'comparison.temporal_summary': str(
                TEMPORAL_CV_DIRECTORIES['temporal']
                / 'cross_validation_summary.json'
            ),
            'comparison.selection': str(TEMPORAL_SELECTION_PATH),
            'comparison.selection_sha256': _sha256(
                TEMPORAL_SELECTION_PATH
            ),
            'comparison.fold_manifests_byte_identical': 'true',
            'source.training_protocol_run_id': protocol_run_id,
            'storage.files_copied': '0',
        })
        for label, summary in summaries.items():
            _log_metrics(_metrics(summary['aggregate'], label))
            for fold in summary['folds']:
                scene = fold['validation_scene']
                evaluation = fold['evaluation']
                _log_metrics({
                    f'{label}.{scene}.obstacle_iou': (
                        evaluation['metrics']['obstacle_head']['iou']
                    ),
                    f'{label}.{scene}.passable_iou': (
                        evaluation['metrics']['passable_head']['iou']
                    ),
                    f'{label}.{scene}.controlled_cell_recall': (
                        evaluation['metrics'][
                            'controlled_obstacle_cell_recall'
                        ]
                    ),
                    f'{label}.{scene}.controlled_instance_recall': (
                        evaluation['metrics'][
                            'controlled_instance_any_detection_rate'
                        ]
                    ),
                })
        _log_metrics(_metrics(
            selection['temporal_minus_current_only'],
            'temporal_delta',
        ))
        return run.info.run_id


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_id = _experiment_id()
    client = MlflowClient()
    dataset_run_id = _dataset_run(experiment_id, client)
    model_run_id = _model_run(experiment_id, client, dataset_run_id)
    cv_run_ids = {
        architecture: _cross_validation_run(
            experiment_id,
            client,
            dataset_run_id,
            architecture,
            directory,
        )
        for architecture, directory in CV_DIRECTORIES.items()
    }
    scene06_dataset_run_id = _scene06_dataset_run(experiment_id, client)
    scene06_model_run_ids = {
        architecture: _scene06_model_run(
            experiment_id,
            client,
            scene06_dataset_run_id,
            architecture,
            directory,
        )
        for architecture, directory in SCENE06_MODEL_DIRECTORIES.items()
    }
    scene07_dataset_run_id = _scene07_dataset_run(experiment_id, client)
    scene07_evaluation_run_id = _scene07_evaluation_run(
        experiment_id,
        client,
        scene07_dataset_run_id,
        scene06_model_run_ids['unet_context'],
    )
    scene01_07_cv_run_ids = {
        label: _scene01_07_cross_validation_run(
            experiment_id,
            client,
            label,
            values['architecture'],
            values['directory'],
            scene06_dataset_run_id,
            scene07_dataset_run_id,
        )
        for label, values in SCENE01_07_CV_DIRECTORIES.items()
    }
    branch_independent_protocol_run_id = _branch_independent_protocol_run(
        experiment_id,
        client,
        scene06_dataset_run_id,
        scene07_dataset_run_id,
    )
    temporal_representation_run_id = _temporal_representation_comparison_run(
        experiment_id,
        client,
        branch_independent_protocol_run_id,
    )
    print(json.dumps({
        'experiment_id': experiment_id,
        'dataset_run_id': dataset_run_id,
        'model_run_id': model_run_id,
        'cv_run_ids': cv_run_ids,
        'scene06_dataset_run_id': scene06_dataset_run_id,
        'scene06_model_run_ids': scene06_model_run_ids,
        'scene07_dataset_run_id': scene07_dataset_run_id,
        'scene07_evaluation_run_id': scene07_evaluation_run_id,
        'scene01_07_cv_run_ids': scene01_07_cv_run_ids,
        'branch_independent_protocol_run_id': (
            branch_independent_protocol_run_id
        ),
        'temporal_representation_run_id': temporal_representation_run_id,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
