"""Create the frozen real-data benchmark and execute its one sanity check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 170817


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing to regenerate non-empty frozen root: {root}")
    public = root / "benchmark" / "public"
    private = root / "benchmark" / "private_evaluator"
    evidence = root / "evidence"
    for path in (public, private, evidence):
        path.mkdir(parents=True, exist_ok=True)

    dataset = load_diabetes(as_frame=True)
    frame = dataset.frame.copy()
    frame.insert(0, "id", np.arange(len(frame), dtype=int))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(frame))
    test_ids = order[:89]
    validation_ids = order[89:177]
    fit_ids = order[177:]

    fit = frame.iloc[fit_ids].copy()
    fit.insert(1, "partition", "fit")
    validation = frame.iloc[validation_ids].copy()
    validation.insert(1, "partition", "validation")
    development = pd.concat([fit, validation], ignore_index=True)
    development = development.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_full = frame.iloc[test_ids].copy().reset_index(drop=True)
    test_public = test_full.drop(columns=["target"])

    development.to_csv(public / "train.csv", index=False)
    test_public.to_csv(public / "test.csv", index=False)
    pd.DataFrame({"id": test_public["id"], "prediction": 0.0}).to_csv(
        public / "sample_submission.csv", index=False
    )
    test_full[["id", "target"]].to_csv(private / "test_labels.csv", index=False)

    description = """# Fixed Diabetes Progression Regression

Predict the one-year quantitative disease-progression measure for diabetes patients from ten baseline clinical variables. This is the real scikit-learn Diabetes dataset (442 patients), not the prior breast-cancer smoke task.

## Frozen data and evaluator

- `train.csv` has `id`, `partition`, ten numeric features, and `target`.
- `partition` is frozen as `fit` or `validation`. Every candidate MUST train validation-scored models only on `fit`; the only development score is RMSE on exactly the `validation` rows.
- Validation labels may be used for model selection and iterative feedback, but never as training rows for the reported validation score.
- After the method is chosen, refit that same method on all rows in `train.csv` to produce final test predictions.
- `test.csv` has `id` and the same ten features, without labels. Test labels are private and unavailable during search.

Use only local NumPy, pandas, SciPy, and scikit-learn. No internet, API, external data, pretrained artifacts, target reconstruction, ID-based target lookup, or inspection of files outside `./input`. Use seed 170817 wherever randomness exists and at most 8 CPU threads. Do not change the frozen partition or substitute cross-validation score for the fixed validation score.

## Output and metric

Write `./submission/submission.csv` with exactly `id,prediction`, in test row order, one finite model prediction per row. Also write `./submission/validation_predictions.csv` with exactly `id,prediction` for the frozen validation rows to make the reported score auditable. The metric is root mean squared error (RMSE), lower is better. The final printed line must be `Final Validation Score: <rmse>`.
"""
    (public / "description.md").write_text(description, encoding="utf-8")

    feature_cols = list(dataset.feature_names)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(fit[feature_cols], fit["target"])
    baseline_pred = model.predict(validation[feature_cols])
    baseline_rmse = float(root_mean_squared_error(validation["target"], baseline_pred))
    sanity = {
        "check_count": 1,
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "test_rows": len(test_full),
        "validation_rmse": baseline_rmse,
        "validation_target_std": float(validation["target"].std(ddof=0)),
        "saturated": bool(baseline_rmse < 5.0),
        "decision": "accept" if baseline_rmse >= 5.0 else "reject_as_saturated",
    }
    (evidence / "baseline_sanity.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    contract = {
        "benchmark": "sklearn_diabetes_fixed_split",
        "seed": SEED,
        "metric": "RMSE_lower_is_better",
        "formal_candidates_per_arm": 6,
        "codex_model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "cpu_cap": 8,
        "gpu_cap": 0,
        "test_evaluations_per_arm": 1,
        "files": {},
    }
    for path in sorted((root / "benchmark").rglob("*")):
        if path.is_file():
            contract["files"][str(path.relative_to(root)).replace("\\", "/")] = sha256(path)
    (evidence / "frozen_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
