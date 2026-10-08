#!/usr/bin/env python3
"""Import existing traversability training results into local MLflow.

Only parameters, scalar/time-series metrics, hashes, and original filesystem
paths are stored. Checkpoints, datasets, plots, CSV files, and JSON files are
not copied into the MLflow artifact store.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = (
    "sqlite:////home/sukja/terrain_nav_data/mlflow/mlflow.db"
)
DEFAULT_ARTIFACT_DIRECTORY = Path(
    "/home/sukja/terrain_nav_data/mlflow/artifacts/"
    "traversability_model_training"
)
DEFAULT_MODEL_DIRECTORIES = (
    Path(
        "/home/sukja/terrain_nav_data/learning/models/"
        "traversability_pilot/v1"
    ),
    Path(
        "/home/sukja/terrain_nav_data/learning/models/"
        "traversability_pilot/v2_domain_aug"
    ),
)
EXPERIMENT_NAME = "traversability_model_training"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register existing traversability training results in local "
            "MLflow without copying datasets or model checkpoints."
        )
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        dest="model_directories",
        help=(
            "Model result directory to import. May be repeated. The known "
            "v1 and v2_domain_aug directories are used when omitted."
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (default: local project SQLite database).",
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Create another run even if this model directory was imported.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_parameters(
    value: Mapping[str, Any], prefix: str = ""
) -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_parameters(item, full_key))
        elif isinstance(item, (list, tuple)):
            flattened[full_key] = json.dumps(
                item, ensure_ascii=False, separators=(",", ":")
            )
        elif item is None:
            flattened[full_key] = "null"
        else:
            flattened[full_key] = str(item)
    return flattened


def numeric_metrics(
    value: Mapping[str, Any], prefix: str = ""
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        number = float(item)
        if math.isfinite(number):
            metrics[f"{prefix}{key}"] = number
    return metrics


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def chunks(mapping: Mapping[str, Any], size: int = 90) -> Iterable[Dict[str, Any]]:
    items = list(mapping.items())
    for start in range(0, len(items), size):
        yield dict(items[start : start + size])


def log_history(path: Path) -> Dict[str, float]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)

    if not rows:
        raise ValueError(f"Training history has no rows: {path}")

    for row in rows:
        epoch = int(float(row["epoch"]))
        metrics = {}
        for key, value in row.items():
            if key == "epoch" or value in (None, ""):
                continue
            number = float(value)
            if math.isfinite(number):
                metrics[key] = number
        for metric_chunk in chunks(metrics):
            mlflow.log_metrics(metric_chunk, step=epoch)

    best_row = min(rows, key=lambda row: float(row["validation_loss"]))
    final_row = rows[-1]
    summary = {
        "best_epoch": float(best_row["epoch"]),
        "best_history_validation_loss": float(best_row["validation_loss"]),
        "final_epoch": float(final_row["epoch"]),
    }
    for key, value in best_row.items():
        if key.startswith("validation_") and value not in (None, ""):
            summary[f"best_{key}"] = float(value)
    for key, value in final_row.items():
        if key in {
            "train_loss",
            "validation_loss",
            "validation_accuracy",
            "validation_obstacle_iou",
            "validation_obstacle_precision",
            "validation_obstacle_recall",
        }:
            summary[f"final_{key}"] = float(value)
    mlflow.log_metrics(summary)
    return summary


def log_validation_file(path: Path) -> None:
    document = load_json(path)
    file_stem = safe_component(path.stem)
    prefix = f"{file_stem}."

    metrics = numeric_metrics(document.get("metrics", {}), prefix)
    for key in ("samples", "mc_samples", "mean_mc_obstacle_probability_variance"):
        value = document.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{prefix}{key}"] = float(value)

    for item in document.get("selective_prediction", []):
        if not isinstance(item, dict):
            continue
        threshold = item.get("confidence_threshold")
        if not isinstance(threshold, (int, float)):
            continue
        threshold_name = f"{float(threshold):.2f}".replace(".", "p")
        metrics.update(
            numeric_metrics(item, f"{prefix}selective_{threshold_name}.")
        )

    metrics.update(
        numeric_metrics(document.get("shadow_policy", {}), f"{prefix}shadow.")
    )
    for metric_chunk in chunks(metrics):
        mlflow.log_metrics(metric_chunk)

    parameters = flatten_parameters(
        document.get("shadow_policy_parameters", {}),
        prefix=f"{file_stem}.shadow_policy",
    )
    for parameter_chunk in chunks(parameters):
        mlflow.log_params(parameter_chunk)


def log_stress_results(path: Path) -> None:
    document = load_json(path)
    results = document.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Stress results must be a list: {path}")

    obstacle_ious = []
    obstacle_recalls = []
    for result in results:
        if not isinstance(result, dict) or "case" not in result:
            continue
        case = safe_component(str(result["case"]))
        metrics = numeric_metrics(result, f"stress.{case}.")
        for metric_chunk in chunks(metrics):
            mlflow.log_metrics(metric_chunk)
        if isinstance(result.get("obstacle_iou"), (int, float)):
            obstacle_ious.append((float(result["obstacle_iou"]), case))
        if isinstance(result.get("obstacle_recall"), (int, float)):
            obstacle_recalls.append((float(result["obstacle_recall"]), case))

    summary: Dict[str, float] = {"stress.case_count": float(len(results))}
    if obstacle_ious:
        summary["stress.minimum_obstacle_iou"] = min(obstacle_ious)[0]
    if obstacle_recalls:
        summary["stress.minimum_obstacle_recall"] = min(obstacle_recalls)[0]
    mlflow.log_metrics(summary)
    if obstacle_ious:
        mlflow.set_tag("stress.worst_obstacle_iou_case", min(obstacle_ious)[1])
    if obstacle_recalls:
        mlflow.set_tag(
            "stress.worst_obstacle_recall_case", min(obstacle_recalls)[1]
        )


def find_existing_run(
    client: MlflowClient, experiment_id: str, import_key: str
) -> str | None:
    escaped = import_key.replace("'", "\\'")
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.import_key = '{escaped}'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def ensure_experiment(name: str) -> str:
    DEFAULT_ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        name,
        artifact_location=DEFAULT_ARTIFACT_DIRECTORY.as_uri(),
        tags={
            "project": "3d_lidar_gnss_outdoor_navigation",
            "artifact_policy": "reference_only_no_copy",
        },
    )


def import_model_directory(
    model_directory: Path,
    experiment_id: str,
    client: MlflowClient,
    allow_duplicate: bool,
) -> str:
    model_directory = model_directory.expanduser().resolve()
    required = (
        model_directory / "training_config.json",
        model_directory / "result.json",
        model_directory / "history.csv",
        model_directory / "best.pt",
        model_directory / "stress_test.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    version = model_directory.name
    import_key = f"traversability_training:{model_directory}"
    existing_run = find_existing_run(client, experiment_id, import_key)
    if existing_run is not None and not allow_duplicate:
        print(f"SKIP {version}: already registered as run {existing_run}")
        return existing_run

    config = load_json(model_directory / "training_config.json")
    result = load_json(model_directory / "result.json")
    checkpoint = model_directory / "best.pt"
    selected = version == "v2_domain_aug"

    description = (
        f"Imported existing traversability model {version}. The MLflow run "
        "contains parameters and metrics only. Dataset, checkpoint, plots, "
        "CSV, and JSON files remain at their original paths and were not "
        "copied into the artifact store."
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"traversability_{version}",
        description=description,
    ) as run:
        mlflow.set_tags(
            {
                "import_key": import_key,
                "model_version": version,
                "run_kind": "imported_existing_training_result",
                "project": "3d_lidar_gnss_outdoor_navigation",
                "task": "lidar_bev_free_obstacle_segmentation",
                "original_model_directory": str(model_directory),
                "original_checkpoint": str(checkpoint),
                "original_history": str(model_directory / "history.csv"),
                "original_training_config": str(
                    model_directory / "training_config.json"
                ),
                "original_result": str(model_directory / "result.json"),
                "artifact_policy": "reference_only_no_copy",
                "selected_for_current_shadow_evaluation": str(selected).lower(),
                "data_scope": "Town10 feasibility pilot",
                "generalization_status": "not a held-out-map or real-vehicle test",
            }
        )

        parameters = flatten_parameters(config, prefix="config")
        parameters.update(
            {
                "storage.artifact_files_copied": "0",
                "storage.model_checkpoint_copied": "false",
                "storage.dataset_copied": "false",
                "checkpoint.best_path": str(checkpoint),
                "checkpoint.best_bytes": str(checkpoint.stat().st_size),
                "checkpoint.best_sha256": sha256(checkpoint),
            }
        )
        for parameter_chunk in chunks(parameters):
            mlflow.log_params(parameter_chunk)

        result_metrics = numeric_metrics(result, prefix="result.")
        split_counts = result.get("split_sample_counts", {})
        if isinstance(split_counts, dict):
            result_metrics.update(
                numeric_metrics(split_counts, prefix="result.samples.")
            )
        mlflow.log_metrics(result_metrics)

        log_history(model_directory / "history.csv")
        log_stress_results(model_directory / "stress_test.json")
        for validation_path in sorted(model_directory.glob("validation_*.json")):
            log_validation_file(validation_path)

        visualization_summary = (
            model_directory / "prediction_visualizations" / "summary.json"
        )
        if visualization_summary.is_file():
            summary = load_json(visualization_summary)
            worst_samples = summary.get("worst_samples", [])
            if worst_samples:
                worst = max(
                    worst_samples,
                    key=lambda item: float(item.get("false_free_rate", 0.0)),
                )
                mlflow.log_metrics(
                    numeric_metrics(worst, prefix="visual_audit.worst_sample.")
                )
                mlflow.set_tag(
                    "visual_audit.worst_sample.session",
                    str(worst.get("session", "")),
                )
            mlflow.set_tag(
                "original_prediction_visualizations",
                str(visualization_summary.parent),
            )

        print(f"IMPORTED {version}: run {run.info.run_id}")
        return run.info.run_id


def main() -> None:
    arguments = parse_arguments()
    model_directories = (
        tuple(arguments.model_directories)
        if arguments.model_directories
        else DEFAULT_MODEL_DIRECTORIES
    )

    mlflow.set_tracking_uri(arguments.tracking_uri)
    experiment_id = ensure_experiment(arguments.experiment_name)
    client = MlflowClient()

    print(f"Experiment: {arguments.experiment_name} (id={experiment_id})")
    print("Artifact policy: reference-only; no source files will be copied")
    for model_directory in model_directories:
        import_model_directory(
            model_directory,
            experiment_id,
            client,
            arguments.allow_duplicate,
        )


if __name__ == "__main__":
    main()
