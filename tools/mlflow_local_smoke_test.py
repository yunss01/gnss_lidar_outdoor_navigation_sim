#!/usr/bin/env python3
"""Create one local-only MLflow smoke-test run."""

from pathlib import Path

import mlflow


TRACKING_URI = (
    'sqlite:////home/sukja/terrain_nav_data/mlflow/mlflow.db'
)
EXPERIMENT_NAME = 'mlflow_installation_test'
ARTIFACT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/mlflow/artifacts/'
    'mlflow_installation_test'
)


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_DIRECTORY.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='local_sqlite_smoke_test',
    ) as run:
        mlflow.log_params({
            'storage_mode': 'local_sqlite',
            'ros_integration': 'none',
        })
        mlflow.log_metric('installation_ok', 1.0)
        print('MLflow local SQLite smoke test passed.')
        print('run_id:', run.info.run_id)


if __name__ == '__main__':
    main()
