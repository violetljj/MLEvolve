"""Focused unit tests for frozen-wave scoring and accounting helpers."""

from __future__ import annotations

import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from finalize_wave import candidate_auc, sign_test_p_one_sided, token_auc
from prepare_wave import dummy_baseline, encode_target, safe_columns
from run_vanilla_arm import execute_candidate
from wave_common import (
    best_so_far,
    compare_prediction_files,
    load_protocol,
    normalized_gain,
    score_predictions,
    sha256,
)


class WaveHarnessTests(unittest.TestCase):
    def test_metric_scoring_and_direction_normalization(self) -> None:
        truth = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.csv"
            pd.DataFrame({"id": [1, 2, 3, 4], "prediction": [0.1, 0.2, 0.8, 0.9]}).to_csv(path, index=False)
            self.assertAlmostEqual(score_predictions("roc_auc", truth, path), 1.0)
        self.assertAlmostEqual(normalized_gain(0.75, 0.5, True), 0.5)
        self.assertAlmostEqual(normalized_gain(8.0, 10.0, False), 0.2)

    def test_best_so_far_respects_metric_direction(self) -> None:
        self.assertEqual(best_so_far([3.0, None, 2.0, 2.5], False), [3.0, 3.0, 2.0, 2.0])
        self.assertEqual(best_so_far([0.5, 0.4, None, 0.7], True), [0.5, 0.5, 0.5, 0.7])

    def test_prediction_replay_comparison_is_candidate_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.csv"
            replay = root / "replay.csv"
            pd.DataFrame({"id": [1, 2], "prediction": [0.1, 0.9]}).to_csv(original, index=False)
            pd.DataFrame({"id": [1, 2], "prediction": [0.1, 0.8]}).to_csv(replay, index=False)
            comparison = compare_prediction_files(original, replay)
            self.assertTrue(comparison["ids_match"])
            self.assertAlmostEqual(comparison["max_abs_prediction_delta"], 0.1)

    def test_exact_sign_test_threshold_is_frozen(self) -> None:
        self.assertAlmostEqual(sign_test_p_one_sided(8, 1), 10 / 512)
        self.assertGreater(sign_test_p_one_sided(7, 2), 0.05)

    def test_auc_helpers_use_best_available_quality(self) -> None:
        self.assertAlmostEqual(candidate_auc([9.0, 8.0, 8.5], 10.0, False), (0.1 + 0.2 + 0.2) / 3)
        checkpoints = [
            {"best_score": 9.0, "total_tokens": 20},
            {"best_score": 8.0, "total_tokens": 60},
        ]
        self.assertAlmostEqual(token_auc(checkpoints, 10.0, False, 100), 0.12)

    def test_preparation_helpers_are_deterministic(self) -> None:
        self.assertEqual(safe_columns(["a b", "a-b", "id"]), ["a_b", "a_b_2", "id_2"])
        encoded, metadata = encode_target(pd.Series(["yes", "no", "yes"]), "binary_classification")
        self.assertEqual(encoded.tolist(), [1, 0, 1])
        self.assertEqual(metadata["mapping"], {"no": 0, "yes": 1})
        baseline = dummy_baseline("accuracy", pd.Series([0, 0, 1]), pd.Series([0, 1]))
        self.assertEqual(baseline, 0.5)

    def test_candidate_timeout_is_consumed_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            public.mkdir()
            pd.DataFrame({"id": [1, 2], "partition": ["fit", "validation"], "x": [0, 1], "target": [0.0, 1.0]}).to_csv(public / "train.csv", index=False)
            pd.DataFrame({"id": [3], "x": [2]}).to_csv(public / "test.csv", index=False)
            import unittest.mock
            timeout = subprocess.TimeoutExpired([sys.executable, "candidate.py"], 300, output="partial")
            with unittest.mock.patch("run_vanilla_arm.subprocess.run", side_effect=timeout):
                result = execute_candidate("pass", root / "candidate", public, Path(sys.executable), {"metric": "rmse"})
            self.assertFalse(result["valid"])
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["returncode"], -1)

    def test_private_finalizer_audits_all_pairs_then_consumes_once(self) -> None:
        protocol = load_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "EXECUTION_START.json").write_text(
                json.dumps({"harness_git_head": "test-commit"}), encoding="utf-8"
            )
            for task in protocol["tasks"]:
                task_root = root / "tasks" / task["slug"]
                labels = task_root / "benchmark" / "private_evaluator" / "test_labels.csv"
                labels.parent.mkdir(parents=True)
                if task["metric"] == "rmse":
                    target = [0.0, 1.0, 2.0, 3.0]
                    vanilla_prediction = [10.0] * 4
                    mle_prediction = target
                    baseline = 2.0
                else:
                    target = [0, 1, 0, 1]
                    vanilla_prediction = [0.5] * 4 if task["metric"] == "roc_auc" else [0] * 4
                    mle_prediction = [0.1, 0.9, 0.1, 0.9] if task["metric"] == "roc_auc" else target
                    baseline = 0.5 if task["metric"] == "roc_auc" else 0.25
                pd.DataFrame({"id": range(4), "target": target}).to_csv(labels, index=False)
                evidence = task_root / "evidence"
                evidence.mkdir()
                (evidence / "frozen_task_receipt.json").write_text(json.dumps({
                    "dummy_validation_score": baseline,
                    "dummy_test_score": baseline,
                    "files": {"benchmark/private_evaluator/test_labels.csv": sha256(labels)},
                }), encoding="utf-8")
                for arm, prediction in (
                    ("VANILLA_CODEX", vanilla_prediction), ("MLEVOLVE_CODEX", mle_prediction)
                ):
                    arm_root = task_root / arm
                    submission = arm_root / "submission.csv"
                    submission.parent.mkdir()
                    pd.DataFrame({"id": range(4), "prediction": prediction}).to_csv(submission, index=False)
                    receipt = {
                        "status": "VALID", "harness_git_head": "test-commit",
                        "usage": {"calls": protocol["expected_codex_calls_per_arm"], "total_tokens": 100},
                        "best_submission_sha256": sha256(submission),
                        "trajectory": [baseline] * 6,
                        "candidate_checkpoints": [
                            {"best_score": baseline, "total_tokens": 10 * index}
                            for index in range(1, 7)
                        ],
                        "wall_seconds": 1.0, "valid_candidates": 6, "candidates": 6,
                    }
                    if arm == "MLEVOLVE_CODEX":
                        receipt["best_submission_path"] = str(submission)
                    else:
                        expected = arm_root / "best_candidate" / "submission" / "submission.csv"
                        expected.parent.mkdir(parents=True)
                        submission.replace(expected)
                        submission = expected
                        receipt["best_submission_sha256"] = sha256(submission)
                    (arm_root / "ARM_TERMINAL.json").write_text(json.dumps(receipt), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("finalize_wave.py")), "--wave-root", str(root)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / "FINAL_WAVE_SUMMARY.json").read_text())
            self.assertEqual(summary["verdict"], "MLEVOLVE_SEARCH_VALUE_ADMIT")
            self.assertEqual(summary["win_tie_loss"], {"loss": 0, "tie": 0, "win": 9})
            self.assertTrue((root / "PRIVATE_EVALUATION_CONSUMPTION.json").exists())
            repeated = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("finalize_wave.py")), "--wave-root", str(root)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == "__main__":
    unittest.main()
