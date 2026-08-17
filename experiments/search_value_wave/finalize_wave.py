"""Consume all private test labels once and issue the frozen wave verdict."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from wave_common import atomic_json, load_protocol, normalized_gain, score_predictions, sha256


def sign_test_p_one_sided(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def candidate_auc(trajectory: list[float | None], baseline: float, maximize: bool) -> float:
    best = None
    gains = []
    for score in trajectory:
        if score is not None and (best is None or (score > best if maximize else score < best)):
            best = score
        gains.append(normalized_gain(best, baseline, maximize) if best is not None else 0.0)
    return float(np.mean(gains))


def token_auc(checkpoints: list[dict], baseline: float, maximize: bool, cap: int) -> float:
    if cap <= 0:
        return 0.0
    points = [(0, 0.0)]
    for item in checkpoints:
        if item["best_score"] is not None and item["total_tokens"] <= cap:
            points.append((int(item["total_tokens"]), normalized_gain(item["best_score"], baseline, maximize)))
    points.sort()
    area = 0.0
    for index, (start, value) in enumerate(points):
        end = points[index + 1][0] if index + 1 < len(points) else cap
        if start < cap:
            area += max(0, min(end, cap) - start) * value
    return area / cap


def submission_path(task_root: Path, arm: str, receipt: dict) -> Path:
    if arm == "VANILLA_CODEX":
        return task_root / arm / "best_candidate" / "submission" / "submission.csv"
    return Path(receipt["best_submission_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.wave_root.resolve()
    output = root / "FINAL_WAVE_SUMMARY.json"
    if output.exists():
        raise SystemExit(f"Private evaluator already consumed: {output}")
    consumption = root / "PRIVATE_EVALUATION_CONSUMPTION.json"
    if consumption.exists():
        raise SystemExit(f"Private evaluator consumption already started: {consumption}")
    protocol = load_protocol()
    execution_start = json.loads((root / "EXECUTION_START.json").read_text(encoding="utf-8"))
    audited = []
    for task in protocol["tasks"]:
        task_root = root / "tasks" / task["slug"]
        task_receipt = json.loads((task_root / "evidence" / "frozen_task_receipt.json").read_text())
        arms = {}
        for arm in ("VANILLA_CODEX", "MLEVOLVE_CODEX"):
            receipt_path = task_root / arm / "ARM_TERMINAL.json"
            if not receipt_path.exists():
                raise SystemExit(f"Missing arm receipt: {receipt_path}")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "VALID":
                raise SystemExit(f"Invalid arm blocks private evaluation: {task['slug']} {arm}")
            if receipt.get("harness_git_head") != execution_start["harness_git_head"]:
                raise SystemExit(f"Harness commit mismatch: {task['slug']} {arm}")
            if receipt["usage"]["calls"] != protocol["expected_codex_calls_per_arm"]:
                raise SystemExit(f"Call-budget mismatch: {task['slug']} {arm}")
            candidate = submission_path(task_root, arm, receipt)
            if sha256(candidate) != receipt["best_submission_sha256"]:
                raise SystemExit(f"Submission changed after arm receipt: {candidate}")
            arms[arm] = {"receipt": receipt, "submission": candidate}
        labels = task_root / "benchmark" / "private_evaluator" / "test_labels.csv"
        if sha256(labels) != task_receipt["files"]["benchmark/private_evaluator/test_labels.csv"]:
            raise SystemExit(f"Private labels changed: {task['slug']}")
        audited.append({
            "task": task, "task_root": task_root, "task_receipt": task_receipt,
            "arms": arms, "labels": labels,
        })

    atomic_json(consumption, {
        "protocol_id": protocol["protocol_id"],
        "task_count": len(audited),
        "test_label_hashes": {
            item["task"]["slug"]: sha256(item["labels"]) for item in audited
        },
        "submission_hashes": {
            item["task"]["slug"]: {
                arm: data["receipt"]["best_submission_sha256"]
                for arm, data in item["arms"].items()
            }
            for item in audited
        },
    })

    pairs = []
    for item in audited:
        task = item["task"]
        task_receipt = item["task_receipt"]
        arms = item["arms"]
        labels = item["labels"]
        truth = pd.read_csv(labels)
        validation_baseline = float(task_receipt["dummy_validation_score"])
        test_baseline = float(task_receipt["dummy_test_score"])
        test_scores = {
            arm: score_predictions(task["metric"], truth, data["submission"])
            for arm, data in arms.items()
        }
        gains = {
            arm: normalized_gain(score, test_baseline, task["maximize"])
            for arm, score in test_scores.items()
        }
        delta = gains["MLEVOLVE_CODEX"] - gains["VANILLA_CODEX"]
        tie_band = protocol["normalization"]["practical_tie_band"]
        outcome = "win" if delta > tie_band else "loss" if delta < -tie_band else "tie"
        common_tokens = min(data["receipt"]["usage"]["total_tokens"] for data in arms.values())
        pair = {
            "task": task["slug"], "metric": task["metric"], "maximize": task["maximize"],
            "dummy_validation_baseline": validation_baseline,
            "dummy_test_baseline": test_baseline, "test_scores": test_scores,
            "normalized_test_gains": gains, "normalized_delta_mlevolve_minus_vanilla": delta,
            "practical_outcome": outcome, "common_token_budget": common_tokens,
            "candidate_axis_auc": {
                arm: candidate_auc(data["receipt"]["trajectory"], validation_baseline, task["maximize"])
                for arm, data in arms.items()
            },
            "token_axis_auc_at_common_budget": {
                arm: token_auc(data["receipt"]["candidate_checkpoints"], validation_baseline, task["maximize"], common_tokens)
                for arm, data in arms.items()
            },
            "usage": {arm: data["receipt"]["usage"] for arm, data in arms.items()},
            "wall_seconds": {arm: data["receipt"]["wall_seconds"] for arm, data in arms.items()},
            "invalid_candidate_rate": {
                arm: 1 - data["receipt"]["valid_candidates"] / data["receipt"]["candidates"]
                for arm, data in arms.items()
            },
        }
        pairs.append(pair)

    wins = sum(pair["practical_outcome"] == "win" for pair in pairs)
    ties = sum(pair["practical_outcome"] == "tie" for pair in pairs)
    losses = sum(pair["practical_outcome"] == "loss" for pair in pairs)
    median_delta = median(pair["normalized_delta_mlevolve_minus_vanilla"] for pair in pairs)
    p_value = sign_test_p_one_sided(wins, losses)
    if wins >= 8 and median_delta > 0.01 and p_value is not None and p_value < 0.05:
        verdict = "MLEVOLVE_SEARCH_VALUE_ADMIT"
    elif wins >= 6 and median_delta > 0.01:
        verdict = "MLEVOLVE_SEARCH_VALUE_CONDITIONAL_ADMIT"
    elif losses >= 6 or median_delta <= 0:
        verdict = "MLEVOLVE_SEARCH_VALUE_REJECT_FOR_INTEGRATION"
    else:
        verdict = "MLEVOLVE_SEARCH_VALUE_NOT_ESTABLISHED"
    candidate_auc_deltas = [pair["candidate_axis_auc"]["MLEVOLVE_CODEX"] - pair["candidate_axis_auc"]["VANILLA_CODEX"] for pair in pairs]
    token_auc_deltas = [pair["token_axis_auc_at_common_budget"]["MLEVOLVE_CODEX"] - pair["token_axis_auc_at_common_budget"]["VANILLA_CODEX"] for pair in pairs]
    final = {
        "protocol_id": protocol["protocol_id"], "verdict": verdict,
        "claim_ceiling": protocol["authority"]["claim_ceiling"],
        "win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
        "median_final_normalized_delta_mlevolve_minus_vanilla": median_delta,
        "paired_exact_sign_test_one_sided_p": p_value,
        "median_candidate_axis_auc_delta": median(candidate_auc_deltas),
        "median_token_axis_auc_delta_at_common_budget": median(token_auc_deltas),
        "pairs": pairs,
    }
    atomic_json(output, final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
