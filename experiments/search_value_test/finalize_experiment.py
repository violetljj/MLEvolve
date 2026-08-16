"""One-shot test evaluation and evidence summary for both frozen arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import root_mean_squared_error


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_once(arm: str, submission: Path, labels_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"One-shot test receipt already exists: {output}")
    pred = pd.read_csv(submission)
    labels = pd.read_csv(labels_path)
    if list(pred.columns) != ["id", "prediction"]:
        raise ValueError(f"Invalid submission columns for {arm}: {pred.columns.tolist()}")
    merged = labels.merge(pred, on="id", how="left", validate="one_to_one")
    if merged["prediction"].isna().any() or len(merged) != len(labels):
        raise ValueError(f"Missing or duplicate test predictions for {arm}")
    receipt = {
        "arm": arm,
        "test_rmse": float(root_mean_squared_error(merged["target"], merged["prediction"])),
        "test_rows": len(merged),
        "submission_sha256": sha256(submission),
        "labels_sha256": sha256(labels_path),
        "evaluation_count": 1,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def usage(call_dir: Path) -> dict:
    calls = list(call_dir.glob("call_*/metadata.json"))
    total_in = total_out = 0
    available = True
    wall = 0.0
    for path in calls:
        meta = json.loads(path.read_text(encoding="utf-8"))
        wall += float(meta["elapsed_seconds"])
        item = meta.get("usage")
        if not item or "input_tokens" not in item or "output_tokens" not in item:
            available = False
        else:
            total_in += item["input_tokens"]
            total_out += item["output_tokens"]
    return {
        "calls": len(calls), "tokens_available": available,
        "input_tokens": total_in if available else None,
        "uncached_input_tokens": (
            total_in - sum(
                json.loads(path.read_text(encoding="utf-8")).get("usage", {}).get(
                    "cached_input_tokens", 0
                )
                for path in calls
            )
            if available else None
        ),
        "output_tokens": total_out if available else None,
        "codex_wall_seconds": wall,
    }


def best_so_far(values: list[float | None]) -> list[float | None]:
    result = []
    best = None
    for value in values:
        if value is not None and (best is None or value < best):
            best = value
        result.append(best)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--mlevolve-run", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    labels = root / "benchmark" / "private_evaluator" / "test_labels.csv"
    vanilla_dir = root / "VANILLA_CODEX_FORMAL"
    vanilla = json.loads((vanilla_dir / "arm_summary.json").read_text())
    journal = json.loads((args.mlevolve_run / "logs" / "journal.json").read_text())
    nodes = [node for node in journal["nodes"] if node.get("stage") != "root"]
    mle_values = [
        node.get("metric", {}).get("value") if node.get("is_buggy") is False and node.get("is_valid") is not False else None
        for node in nodes
    ]
    mle_best = min((value for value in mle_values if value is not None), default=None)
    mle_summary = {
        "arm": "MLEVOLVE_CODEX", "candidates": len(nodes),
        "valid": sum(value is not None for value in mle_values),
        "failed": sum(value is None for value in mle_values),
        "dev_best": mle_best,
        "compute_wall_seconds": sum(float(node.get("exec_time") or 0) for node in nodes),
        "trajectory": mle_values,
        "stages": [node.get("stage") for node in nodes],
    }
    vanilla_receipt = score_once(
        "VANILLA_CODEX",
        vanilla_dir / "best_candidate" / "submission" / "submission.csv",
        labels, root / "evidence" / "vanilla_test_once.json",
    )
    mle_receipt = score_once(
        "MLEVOLVE_CODEX",
        args.mlevolve_run / "workspace" / "best_submission" / "submission.csv",
        labels, root / "evidence" / "mlevolve_test_once.json",
    )
    vanilla_usage = usage(vanilla_dir / "codex_calls")
    mle_usage = usage(root / "MLEVOLVE_CODEX" / "codex_calls")
    mle_log = args.mlevolve_run / "logs" / "MLEvolve.log"
    mle_wall = None
    if mle_log.exists():
        # The run process wall is preserved separately by the launcher when available.
        launcher = root / "MLEVOLVE_CODEX" / "launcher_summary.json"
        if launcher.exists():
            mle_wall = json.loads(launcher.read_text()).get("wall_seconds")
    test_delta = mle_receipt["test_rmse"] - vanilla_receipt["test_rmse"]
    budget_ratio = None
    if vanilla_usage["calls"]:
        budget_ratio = mle_usage["calls"] / vanilla_usage["calls"]
    if budget_ratio is None or budget_ratio < 0.75 or budget_ratio > 1.25:
        verdict = "INVALID_COMPARISON"
    elif test_delta < -1.0:
        verdict = "MLEVOLVE_SEARCH_VALUE_SIGNAL_POSITIVE"
    elif test_delta > 1.0:
        verdict = "MLEVOLVE_SEARCH_VALUE_SIGNAL_NEGATIVE"
    else:
        verdict = "MLEVOLVE_SEARCH_VALUE_NOT_ESTABLISHED"
    final = {
        "verdict": verdict,
        "test_practical_equivalence_margin_rmse": 1.0,
        "arms": {
            "VANILLA_CODEX": {**vanilla, **vanilla_usage, "test": vanilla_receipt["test_rmse"], "best_so_far": best_so_far(vanilla["trajectory"])},
            "MLEVOLVE_CODEX": {**mle_summary, **mle_usage, "wall_seconds": mle_wall, "test": mle_receipt["test_rmse"], "best_so_far": best_so_far(mle_values)},
        },
        "budget_call_ratio_mlevolve_over_vanilla": budget_ratio,
        "budget_wall_ratio_mlevolve_over_vanilla": (
            mle_wall / vanilla["wall_seconds"] if mle_wall is not None else None
        ),
        "budget_uncached_input_ratio_mlevolve_over_vanilla": (
            mle_usage["uncached_input_tokens"] / vanilla_usage["uncached_input_tokens"]
            if vanilla_usage["uncached_input_tokens"] else None
        ),
        "test_rmse_delta_mlevolve_minus_vanilla": test_delta,
    }
    (root / "FINAL_SUMMARY.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
