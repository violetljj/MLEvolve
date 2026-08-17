"""Materialize the nine frozen OpenML tasks without exposing private labels."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from wave_common import PROTOCOL_PATH, atomic_json, load_protocol, sha256


def safe_columns(columns: list[object]) -> list[str]:
    result: list[str] = []
    used: set[str] = set()
    for index, value in enumerate(columns):
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or f"feature_{index}"
        candidate = base
        suffix = 2
        while candidate in used or candidate in {"id", "partition", "target"}:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


def encode_target(target: pd.Series, problem_type: str) -> tuple[pd.Series, dict]:
    if problem_type == "regression":
        numeric = pd.to_numeric(target, errors="raise").astype(float)
        if not np.isfinite(numeric).all():
            raise ValueError("Regression target contains non-finite values")
        return numeric, {"kind": "numeric"}
    labels = sorted(str(value) for value in pd.Series(target).dropna().unique())
    mapping = {label: index for index, label in enumerate(labels)}
    encoded = pd.Series(target).map(lambda value: mapping[str(value)]).astype(int)
    expected = 2 if problem_type == "binary_classification" else None
    if expected is not None and len(mapping) != expected:
        raise ValueError(f"Expected two target classes, found {len(mapping)}")
    if problem_type == "multiclass_classification" and len(mapping) < 3:
        raise ValueError("Expected at least three target classes")
    return encoded, {"kind": "class_index", "mapping": mapping}


def dummy_baseline(metric: str, fit_y: pd.Series, evaluation_y: pd.Series) -> float:
    fit_x = np.zeros((len(fit_y), 1))
    evaluation_x = np.zeros((len(evaluation_y), 1))
    if metric == "rmse":
        model = DummyRegressor(strategy="mean").fit(fit_x, fit_y)
        return float(root_mean_squared_error(evaluation_y, model.predict(evaluation_x)))
    model = DummyClassifier(strategy="prior").fit(fit_x, fit_y)
    if metric == "roc_auc":
        probability = model.predict_proba(evaluation_x)[:, 1]
        return float(roc_auc_score(evaluation_y, probability))
    return float(accuracy_score(evaluation_y, model.predict(evaluation_x)))


def description(task: dict, seed: int, features: list[str], target_meta: dict) -> str:
    metric_text = {
        "rmse": "root mean squared error (RMSE), lower is better",
        "roc_auc": "binary ROC AUC using the probability of encoded class 1, higher is better",
        "accuracy": "multiclass accuracy using encoded integer class labels, higher is better",
    }[task["metric"]]
    output_semantics = (
        "a finite numeric regression prediction"
        if task["metric"] == "rmse"
        else "a probability in [0, 1] for encoded class 1"
        if task["metric"] == "roc_auc"
        else "an encoded integer class label"
    )
    return f"""# Frozen OpenML task: {task['slug']}

Predict `target` from the supplied tabular features. The source is frozen OpenML data ID {task['openml_data_id']}. Target encoding metadata is `{json.dumps(target_meta, sort_keys=True)}`.

## Frozen data and evaluator

- `train.csv` contains `id`, `partition`, {len(features)} feature columns, and `target`.
- `partition` is frozen as `fit` or `validation`. Validation-scored models MUST train only on `fit` rows. The only development score is the official metric on exactly the validation rows.
- Validation labels may be used for model selection and iterative feedback, but never as training rows for the reported validation score.
- After choosing a method, refit that same method on all labeled rows to produce final test predictions.
- `test.csv` contains `id` and the same feature columns without labels. Private test labels are unavailable during search.

Use only local NumPy, pandas, SciPy, and scikit-learn. No internet, APIs, external data, pretrained artifacts, ID-based target lookup, target reconstruction, or files outside `./input`. Use seed {seed} wherever randomness exists and at most 8 CPU threads. Do not change the frozen partition or substitute cross-validation for the fixed validation score.

## Output and metric

The metric is {metric_text}. Write `./submission/submission.csv` and `./submission/validation_predictions.csv`, each with exactly `id,prediction`. Every prediction must be {output_semantics}. The final printed line must be `Final Validation Score: <score>`.
"""


def load_source(task: dict, task_root: Path, source_mode: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    if source_mode == "openml":
        source = fetch_openml(data_id=task["openml_data_id"], as_frame=True, parser="auto")
        return source.data.copy(), pd.Series(source.target), {
            "transport": "openml_api",
            "name": source.details.get("name"),
            "version": source.details.get("version"),
            "data_id": source.details.get("id"),
        }
    mirror = task_root / "source_mirror"
    url = f"https://gitlab.com/data/d/openml/{task['openml_data_id']}.git"
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(mirror)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Datagit clone failed for {task['slug']}: {completed.stderr[-2000:]}")
    metadata_path = mirror / "dataset" / "metadata.json"
    table_path = mirror / "dataset" / "tables" / "data.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["data_set_description"]
    if int(metadata["id"]) != task["openml_data_id"]:
        raise ValueError(f"Datagit ID mismatch for {task['slug']}")
    target_name = metadata.get("default_target_attribute")
    table = pd.read_csv(table_path)
    if not target_name or target_name not in table.columns:
        raise ValueError(f"Missing default target for {task['slug']}: {target_name}")
    commit = subprocess.run(
        ["git", "-C", str(mirror), "rev-parse", "HEAD"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60, check=True,
    ).stdout.strip()
    return table.drop(columns=[target_name]), table[target_name], {
        "transport": "datagit_gitlab_mirror",
        "repository": url,
        "commit": commit,
        "name": metadata.get("name"),
        "version": metadata.get("version"),
        "data_id": metadata.get("id"),
        "openml_md5_checksum": metadata.get("md5_checksum"),
        "mirror_data_csv_sha256": sha256(table_path),
        "mirror_metadata_sha256": sha256(metadata_path),
    }


def prepare_task(root: Path, task: dict, protocol: dict, source_mode: str) -> dict:
    task_root = root / "tasks" / task["slug"]
    if task_root.exists():
        raise FileExistsError(f"Refusing to overwrite task root: {task_root}")
    public = task_root / "benchmark" / "public"
    private = task_root / "benchmark" / "private_evaluator"
    evidence = task_root / "evidence"
    for directory in (public, private, evidence):
        directory.mkdir(parents=True)

    features, source_target, source_receipt = load_source(task, task_root, source_mode)
    features.columns = safe_columns(list(features.columns))
    target, target_meta = encode_target(source_target, task["problem_type"])
    frame = features.reset_index(drop=True)
    frame.insert(0, "id", np.arange(len(frame), dtype=np.int64))
    frame["target"] = target.reset_index(drop=True)
    if frame["target"].isna().any():
        raise ValueError(f"Missing target in {task['slug']}")

    seed = protocol["split"]["base_seed"] + task["sequence"]
    indices = np.arange(len(frame))
    stratify = target if task["problem_type"] != "regression" else None
    development_ids, test_ids = train_test_split(
        indices, test_size=protocol["split"]["test_fraction"], random_state=seed,
        stratify=stratify,
    )
    development_target = target.iloc[development_ids]
    relative_validation = protocol["split"]["validation_fraction"] / (
        protocol["split"]["fit_fraction"] + protocol["split"]["validation_fraction"]
    )
    fit_ids, validation_ids = train_test_split(
        development_ids, test_size=relative_validation, random_state=seed + 1000,
        stratify=development_target if stratify is not None else None,
    )

    fit = frame.iloc[fit_ids].copy()
    fit.insert(1, "partition", "fit")
    validation = frame.iloc[validation_ids].copy()
    validation.insert(1, "partition", "validation")
    development = pd.concat([fit, validation], ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    test_full = frame.iloc[test_ids].copy().reset_index(drop=True)
    test_public = test_full.drop(columns=["target"])

    development.to_csv(public / "train.csv", index=False)
    test_public.to_csv(public / "test.csv", index=False)
    pd.DataFrame({"id": test_public["id"], "prediction": 0.0}).to_csv(
        public / "sample_submission.csv", index=False
    )
    test_full[["id", "target"]].to_csv(private / "test_labels.csv", index=False)
    (public / "description.md").write_text(
        description(task, seed, list(features.columns), target_meta), encoding="utf-8"
    )

    validation_baseline = dummy_baseline(
        task["metric"], frame.iloc[fit_ids]["target"], frame.iloc[validation_ids]["target"]
    )
    test_baseline = dummy_baseline(
        task["metric"], frame.iloc[development_ids]["target"], frame.iloc[test_ids]["target"]
    )
    receipt = {
        "task": task,
        "source": source_receipt,
        "seed": seed,
        "rows": len(frame),
        "features": len(features.columns),
        "fit_rows": len(fit_ids),
        "validation_rows": len(validation_ids),
        "test_rows": len(test_ids),
        "target_encoding": target_meta,
        "dummy_validation_score": validation_baseline,
        "dummy_test_score": test_baseline,
        "files": {},
    }
    for path in sorted((task_root / "benchmark").rglob("*")):
        if path.is_file():
            receipt["files"][path.relative_to(task_root).as_posix()] = sha256(path)
    atomic_json(evidence / "frozen_task_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-root", type=Path, required=True)
    parser.add_argument("--source", choices=("openml", "datagit"), default="openml")
    args = parser.parse_args()
    root = args.wave_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing to prepare non-empty wave root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    protocol_copy = root / "FROZEN_PROTOCOL.json"
    protocol_copy.write_bytes(PROTOCOL_PATH.read_bytes())
    receipts = []
    try:
        for task in protocol["tasks"]:
            receipts.append(prepare_task(root, task, protocol, args.source))
    except Exception as exc:
        atomic_json(root / "PREPARATION_FAILED.json", {
            "protocol_sha256": sha256(protocol_copy),
            "completed_tasks": [item["task"]["slug"] for item in receipts],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "terminal": "WAVE_NOT_EVALUABLE",
        })
        raise
    atomic_json(root / "PREPARATION_COMPLETE.json", {
        "protocol_sha256": sha256(protocol_copy),
        "task_count": len(receipts),
        "tasks": [item["task"]["slug"] for item in receipts],
    })
    print(json.dumps({"prepared": len(receipts), "root": str(root)}, indent=2))


if __name__ == "__main__":
    main()
