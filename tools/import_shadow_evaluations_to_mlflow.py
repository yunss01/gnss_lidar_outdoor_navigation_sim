#!/usr/bin/env python3
"""Import route-level traversability shadow audits into local MLflow.

The importer pairs a subscriber-only shadow recorder run with the online
semantic-LiDAR evaluator that started at the same time. Only parameters,
metrics, hashes, and original paths are registered. No CSV, NPZ, JSON,
dataset, or checkpoint file is copied into the MLflow artifact store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient


TRACKING_URI = "sqlite:////home/sukja/terrain_nav_data/mlflow/mlflow.db"
EXPERIMENT_NAME = "traversability_shadow_navigation"
ARTIFACT_DIRECTORY = Path(
    "/home/sukja/terrain_nav_data/mlflow/artifacts/"
    "traversability_shadow_navigation"
)
SHADOW_ROOT = Path("/home/sukja/terrain_nav_data/learning/shadow_runs")
EVALUATION_ROOT = Path(
    "/home/sukja/terrain_nav_data/learning/shadow_evaluations"
)
CHECKPOINT = Path(
    "/home/sukja/terrain_nav_data/learning/models/"
    "traversability_pilot/v2_domain_aug/best.pt"
)

# These are the ten runs explicitly collected as five repeated F9 trials and
# five repeated F10 trials. Earlier runs remain useful, but are tagged as
# pilot/instrumentation runs so that they are not mixed into the repeated set.
REPEATED_EVALUATION_SESSIONS = {
    "shadow_20260915_185500_826020",
    "shadow_20260915_185755_408688",
    "shadow_20260915_190042_713436",
    "shadow_20260915_190426_471974",
    "shadow_20260915_190707_894886",
    "shadow_20260915_191719_285010",
    "shadow_20260915_192055_919415",
    "shadow_20260915_192521_413084",
    "shadow_20260915_193008_051250",
    "shadow_20260915_193424_364602",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register existing shadow-navigation summaries in local MLflow "
            "without copying source files."
        )
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--allow-duplicate", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
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


def flatten_metrics(
    value: Mapping[str, Any], prefix: str = ""
) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_metrics(item, full_key))
        elif isinstance(item, bool):
            flattened[full_key] = float(item)
        elif isinstance(item, (int, float)):
            number = float(item)
            if math.isfinite(number):
                flattened[full_key] = number
    return flattened


def chunks(mapping: Mapping[str, Any], size: int = 90) -> Iterable[Dict[str, Any]]:
    items = list(mapping.items())
    for start in range(0, len(items), size):
        yield dict(items[start : start + size])


def started_at(document: Mapping[str, Any]) -> datetime:
    return datetime.fromisoformat(str(document["started_at"]))


def load_evaluations() -> Tuple[Tuple[Path, datetime, Dict[str, Any], Dict[str, Any]], ...]:
    evaluations = []
    for directory in sorted(EVALUATION_ROOT.iterdir()):
        if not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        summary_path = directory / "summary.json"
        if not metadata_path.is_file() or not summary_path.is_file():
            continue
        metadata = load_json(metadata_path)
        summary = load_json(summary_path)
        evaluations.append(
            (directory, started_at(metadata), metadata, summary)
        )
    return tuple(evaluations)


def matching_evaluation(
    shadow_start: datetime,
    evaluations: Tuple[
        Tuple[Path, datetime, Dict[str, Any], Dict[str, Any]], ...
    ],
) -> Optional[Tuple[Path, Dict[str, Any], Dict[str, Any], float]]:
    if not evaluations:
        return None
    nearest = min(
        evaluations, key=lambda item: abs((item[1] - shadow_start).total_seconds())
    )
    delta_s = abs((nearest[1] - shadow_start).total_seconds())
    if delta_s > 0.1:
        return None
    return nearest[0], nearest[2], nearest[3], delta_s


def ensure_experiment(name: str) -> str:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        name,
        artifact_location=ARTIFACT_DIRECTORY.as_uri(),
        tags={
            "project": "3d_lidar_gnss_outdoor_navigation",
            "artifact_policy": "reference_only_no_copy",
            "control_effect": "none_shadow_only",
        },
    )


def find_existing_run(
    client: MlflowClient, experiment_id: str, import_key: str
) -> Optional[str]:
    escaped = import_key.replace("'", "\\'")
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.import_key = '{escaped}'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def route_name(route_size: Any) -> str:
    return {4: "F9", 5: "F10"}.get(int(route_size), f"route_{route_size}")


def import_shadow_run(
    shadow_directory: Path,
    evaluations: Tuple[
        Tuple[Path, datetime, Dict[str, Any], Dict[str, Any]], ...
    ],
    experiment_id: str,
    client: MlflowClient,
    allow_duplicate: bool,
) -> str:
    shadow_metadata_path = shadow_directory / "metadata.json"
    shadow_summary_path = shadow_directory / "summary.json"
    shadow_frames_path = shadow_directory / "frames.csv"
    shadow_metadata = load_json(shadow_metadata_path)
    shadow_summary = load_json(shadow_summary_path)
    session = shadow_directory.name
    import_key = f"traversability_shadow:{shadow_directory.resolve()}"

    existing = find_existing_run(client, experiment_id, import_key)
    if existing is not None and not allow_duplicate:
        print(f"SKIP {session}: already registered as run {existing}")
        return existing

    route_size = shadow_summary.get(
        "route_size", shadow_metadata.get("route_size_at_start")
    )
    route = route_name(route_size)
    result = str(shadow_summary.get("result", "unknown"))
    success = result == "completed"
    phase = (
        "repeated_route_evaluation_5x"
        if session in REPEATED_EVALUATION_SESSIONS
        else "pilot_or_instrumentation"
    )
    paired = matching_evaluation(started_at(shadow_metadata), evaluations)

    description = (
        f"Subscriber-only learned-traversability shadow audit for {route}. "
        "The AI output did not control the vehicle. Navigation outcome is "
        "context for the perception audit and is not closed-loop AI driving "
        "performance. Source files remain at their original paths."
    )

    run_name = f"{route}_{session.removeprefix('shadow_')}"
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        description=description,
    ) as run:
        tags = {
            "import_key": import_key,
            "project": "3d_lidar_gnss_outdoor_navigation",
            "run_kind": "imported_shadow_navigation_evaluation",
            "route": route,
            "route_source": f"inferred_from_route_size_{route_size}",
            "evaluation_phase": phase,
            "navigation_result": result,
            "navigation_success": str(success).lower(),
            "model_version": "v2_domain_aug",
            "control_effect": "none_subscriber_only",
            "semantic_evaluation_attached": str(paired is not None).lower(),
            "artifact_policy": "reference_only_no_copy",
            "original_shadow_directory": str(shadow_directory.resolve()),
            "original_shadow_frames": str(shadow_frames_path.resolve()),
        }
        if paired is not None:
            tags["original_semantic_evaluation_directory"] = str(
                paired[0].resolve()
            )
        mlflow.set_tags(tags)

        parameters = flatten_parameters(shadow_metadata, "shadow_config")
        parameters.update(
            {
                "storage.artifact_files_copied": "0",
                "storage.source_csv_copied": "false",
                "storage.source_npz_copied": "false",
                "storage.source_json_copied": "false",
                "model.checkpoint_path": str(CHECKPOINT),
                "model.version": "v2_domain_aug",
                "source.shadow_metadata_sha256": sha256(shadow_metadata_path),
                "source.shadow_summary_sha256": sha256(shadow_summary_path),
            }
        )
        if shadow_frames_path.is_file():
            parameters["source.shadow_frames_bytes"] = str(
                shadow_frames_path.stat().st_size
            )
        if paired is not None:
            evaluation_directory, evaluation_metadata, _, delta_s = paired
            parameters.update(
                flatten_parameters(evaluation_metadata, "semantic_config")
            )
            parameters["pairing.start_time_delta_ms"] = str(delta_s * 1000.0)
            parameters["source.semantic_metadata_sha256"] = sha256(
                evaluation_directory / "metadata.json"
            )
            parameters["source.semantic_summary_sha256"] = sha256(
                evaluation_directory / "summary.json"
            )
            semantic_frames = evaluation_directory / "frames.csv"
            if semantic_frames.is_file():
                parameters["source.semantic_frames_bytes"] = str(
                    semantic_frames.stat().st_size
                )

        for parameter_chunk in chunks(parameters):
            mlflow.log_params(parameter_chunk)

        shadow_metrics = flatten_metrics(shadow_summary, "shadow")
        start = datetime.fromisoformat(str(shadow_summary["started_at"]))
        end = datetime.fromisoformat(str(shadow_summary["ended_at"]))
        shadow_metrics.update(
            {
                "navigation.success": float(success),
                "navigation.duration_s": (end - start).total_seconds(),
            }
        )
        for metric_chunk in chunks(shadow_metrics):
            mlflow.log_metrics(metric_chunk)

        if paired is not None:
            semantic_metrics = flatten_metrics(paired[2], "semantic")
            for metric_chunk in chunks(semantic_metrics):
                mlflow.log_metrics(metric_chunk)

        print(
            f"IMPORTED {session}: {route}, result={result}, "
            f"semantic={'yes' if paired else 'no'}, run={run.info.run_id}"
        )
        return run.info.run_id


def main() -> None:
    arguments = parse_arguments()
    mlflow.set_tracking_uri(arguments.tracking_uri)
    experiment_id = ensure_experiment(arguments.experiment_name)
    client = MlflowClient()
    evaluations = load_evaluations()

    print(f"Experiment: {arguments.experiment_name} (id={experiment_id})")
    print(f"Semantic evaluation directories found: {len(evaluations)}")
    print("Artifact policy: reference-only; no source files will be copied")

    imported = 0
    for shadow_directory in sorted(SHADOW_ROOT.iterdir()):
        if not shadow_directory.is_dir():
            continue
        if not (shadow_directory / "summary.json").is_file():
            continue
        import_shadow_run(
            shadow_directory,
            evaluations,
            experiment_id,
            client,
            arguments.allow_duplicate,
        )
        imported += 1
    print(f"Processed shadow directories: {imported}")


if __name__ == "__main__":
    main()
