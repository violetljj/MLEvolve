"""Recompute final development RMSE from saved validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import root_mean_squared_error


def score(truth: pd.DataFrame, path: Path) -> float:
    predictions = pd.read_csv(path)
    merged = truth.merge(predictions, on="id", validate="one_to_one")
    if len(merged) != len(truth) or merged["prediction"].isna().any():
        raise ValueError(f"Incomplete predictions: {path}")
    return float(root_mean_squared_error(merged["target"], merged["prediction"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    train = pd.read_csv(root / "benchmark" / "public" / "train.csv")
    truth = train.loc[train["partition"].eq("validation"), ["id", "target"]]
    result = {
        "VANILLA_CODEX": score(
            truth,
            root / "VANILLA_CODEX_FORMAL" / "best_candidate" / "submission"
            / "validation_predictions.csv",
        ),
        "MLEVOLVE_CODEX": score(
            truth,
            root / "evidence" / "mlevolve_best_dev_replay" / "submission"
            / "validation_predictions.csv",
        ),
    }
    output = root / "evidence" / "development_score_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
