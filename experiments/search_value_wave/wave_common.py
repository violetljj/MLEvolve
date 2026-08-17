"""Shared immutable-protocol and metric helpers for search-value wave R1."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, root_mean_squared_error, roc_auc_score


PROTOCOL_PATH = Path(__file__).with_name("wave_protocol.json")


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    tasks = protocol["tasks"]
    if len(tasks) != 9 or [task["sequence"] for task in tasks] != list(range(1, 10)):
        raise ValueError("Frozen protocol must contain exactly sequences 1..9")
    if len({task["slug"] for task in tasks}) != 9:
        raise ValueError("Frozen task slugs must be unique")
    return protocol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def validate_prediction_frame(truth: pd.DataFrame, prediction_path: Path) -> np.ndarray:
    predictions = pd.read_csv(prediction_path)
    if list(predictions.columns) != ["id", "prediction"]:
        raise ValueError(f"Invalid prediction columns: {prediction_path}")
    merged = truth[["id", "target"]].merge(
        predictions, on="id", how="left", validate="one_to_one", sort=False
    )
    values = pd.to_numeric(merged["prediction"], errors="coerce").to_numpy(dtype=float)
    if len(merged) != len(truth) or not np.isfinite(values).all():
        raise ValueError(f"Incomplete or non-finite predictions: {prediction_path}")
    return values


def score_predictions(metric: str, truth: pd.DataFrame, prediction_path: Path) -> float:
    predictions = validate_prediction_frame(truth, prediction_path)
    target = truth["target"].to_numpy()
    if metric == "rmse":
        return float(root_mean_squared_error(target.astype(float), predictions))
    if metric == "roc_auc":
        return float(roc_auc_score(target.astype(int), predictions))
    if metric == "accuracy":
        rounded = np.rint(predictions).astype(int)
        if not np.allclose(predictions, rounded, atol=1e-9):
            raise ValueError("Accuracy predictions must be encoded class labels")
        return float(accuracy_score(target.astype(int), rounded))
    raise ValueError(f"Unsupported frozen metric: {metric}")


def normalized_gain(score: float, baseline: float, maximize: bool) -> float:
    direction = 1.0 if maximize else -1.0
    return direction * (score - baseline) / max(abs(baseline), 1e-12)


def best_so_far(values: list[float | None], maximize: bool) -> list[float | None]:
    result: list[float | None] = []
    best: float | None = None
    for value in values:
        if value is not None and math.isfinite(value):
            if best is None or (value > best if maximize else value < best):
                best = value
        result.append(best)
    return result


def codex_usage(call_dir: Path) -> dict:
    """Return usage totals plus a timestamped cumulative token timeline."""
    items = []
    for path in call_dir.glob("call_*/metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        usage = metadata.get("usage")
        started = metadata.get("started_utc")
        if not usage or not started:
            continue
        items.append((datetime.fromisoformat(started), usage, path.parent.name))
    items.sort(key=lambda item: item[0])
    cumulative_input = cumulative_cached = cumulative_output = 0
    timeline = []
    for started, usage, call_name in items:
        cumulative_input += int(usage["input_tokens"])
        cumulative_cached += int(usage.get("cached_input_tokens", 0))
        cumulative_output += int(usage["output_tokens"])
        timeline.append({
            "call": call_name,
            "started_utc": started.isoformat(),
            "input_tokens": cumulative_input,
            "cached_input_tokens": cumulative_cached,
            "uncached_input_tokens": cumulative_input - cumulative_cached,
            "output_tokens": cumulative_output,
            "total_tokens": cumulative_input + cumulative_output,
        })
    return {
        "calls": len(items),
        "input_tokens": cumulative_input,
        "cached_input_tokens": cumulative_cached,
        "uncached_input_tokens": cumulative_input - cumulative_cached,
        "output_tokens": cumulative_output,
        "total_tokens": cumulative_input + cumulative_output,
        "timeline": timeline,
    }


def cumulative_at(timeline: list[dict], completed_utc: datetime) -> dict:
    eligible = [item for item in timeline if datetime.fromisoformat(item["started_utc"]) <= completed_utc]
    if not eligible:
        return {"total_tokens": 0, "uncached_input_tokens": 0}
    return eligible[-1]


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=60, check=True,
    ).stdout.strip()
