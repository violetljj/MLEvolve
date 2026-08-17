"""Run one frozen six-candidate Vanilla Codex arm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from wave_common import (
    atomic_json,
    best_so_far,
    codex_usage,
    cumulative_at,
    load_protocol,
    score_predictions,
    sha256,
)


def strict_schema() -> dict:
    return {
        "type": "object",
        "properties": {"plan": {"type": "string"}, "code": {"type": "string"}},
        "required": ["plan", "code"],
        "additionalProperties": False,
    }


def codex_call(prompt: str, call_logs: Path, arm_name: str, schema: dict | None = None) -> str:
    python = Path(os.environ["SEARCH_VALUE_PYTHON"])
    runner = Path(__file__).parents[1] / "search_value_test" / "codex_logged_runner.py"
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as output_handle:
        output_path = Path(output_handle.name)
    schema_path = None
    try:
        command = [
            str(python), str(runner), "exec", "--ephemeral", "--sandbox", "read-only",
            "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
            "--color", "never", "-m", "gpt-5.6-sol", "-c",
            'model_reasoning_effort="medium"', "-o", str(output_path),
        ]
        if schema is not None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as schema_handle:
                json.dump(schema, schema_handle)
                schema_path = Path(schema_handle.name)
            command.extend(["--output-schema", str(schema_path)])
        command.append("-")
        env = os.environ.copy()
        env["SEARCH_VALUE_ARM"] = arm_name
        env["SEARCH_VALUE_CALL_LOG_DIR"] = str(call_logs)
        completed = subprocess.run(
            command, input=prompt, text=True, encoding="utf-8", errors="replace",
            capture_output=True, env=env, timeout=1800, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-4000:])
        return output_path.read_text(encoding="utf-8").strip()
    finally:
        output_path.unlink(missing_ok=True)
        if schema_path:
            schema_path.unlink(missing_ok=True)


def execute_candidate(code: str, candidate_dir: Path, public: Path, python: Path, task: dict) -> dict:
    candidate_dir.mkdir(parents=True)
    shutil.copytree(public, candidate_dir / "input")
    (candidate_dir / "submission").mkdir()
    (candidate_dir / "working").mkdir()
    (candidate_dir / "candidate.py").write_text(code, encoding="utf-8")
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "8"
    started = time.time()
    completed = subprocess.run(
        [str(python), "candidate.py"], cwd=candidate_dir, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, check=False,
    )
    elapsed = time.time() - started
    output = completed.stdout + completed.stderr
    (candidate_dir / "execution.txt").write_text(output, encoding="utf-8")
    valid = completed.returncode == 0
    score = None
    try:
        train = pd.read_csv(public / "train.csv")
        validation_truth = train.loc[train["partition"].eq("validation"), ["id", "target"]]
        score = score_predictions(
            task["metric"], validation_truth,
            candidate_dir / "submission" / "validation_predictions.csv",
        )
        test_ids = pd.read_csv(public / "test.csv")["id"].tolist()
        submission = pd.read_csv(candidate_dir / "submission" / "submission.csv")
        valid = valid and list(submission.columns) == ["id", "prediction"]
        valid = valid and submission["id"].tolist() == test_ids
        valid = valid and np.isfinite(pd.to_numeric(submission["prediction"], errors="coerce")).all()
    except Exception as exc:
        output += f"\nOfficial output validation failure: {exc}\n"
        valid = False
        score = None
    return {
        "valid": bool(valid), "score": score if valid else None,
        "returncode": completed.returncode, "execution_wall_seconds": elapsed,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "output": output[-12000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    task_root = args.task_root.resolve()
    task_receipt = json.loads((task_root / "evidence" / "frozen_task_receipt.json").read_text())
    task = task_receipt["task"]
    frozen = next(item for item in protocol["tasks"] if item["slug"] == task["slug"])
    if task != frozen:
        raise SystemExit("Task receipt differs from repository-frozen protocol")
    arm = task_root / "VANILLA_CODEX"
    if arm.exists():
        raise SystemExit(f"Refusing to overwrite arm: {arm}")
    arm.mkdir()
    call_logs = arm / "codex_calls"
    public = task_root / "benchmark" / "public"
    python = Path(os.environ["SEARCH_VALUE_PYTHON"])
    description = (public / "description.md").read_text(encoding="utf-8")
    train = pd.read_csv(public / "train.csv")
    preview = (
        f"train shape={train.shape}; columns={train.columns.tolist()}; "
        f"partition counts={train['partition'].value_counts().to_dict()}; "
        f"dtypes={train.dtypes.astype(str).to_dict()}"
    )
    arm_started = time.time()
    setup_task = codex_call(
        "Read the frozen task. Summarize its data isolation, metric, output semantics, "
        "and resource constraints. Do not write code or inspect files.\n\n" + description + "\n" + preview,
        call_logs, "VANILLA_CODEX",
    )
    setup_metric = codex_call(
        "State the optimization direction and exact fixed-partition metric computation for "
        "this frozen task. Do not write code or inspect files.\n\n" + description,
        call_logs, "VANILLA_CODEX",
    )
    history = []
    best_code = None
    best_score = None
    for round_index in range(1, protocol["formal_candidates_per_arm"] + 1):
        prior = "No previous candidate exists." if best_code is None else (
            f"Current best official validation score: {best_score}\n"
            f"Current best code:\n```python\n{best_code}\n```\n"
            f"Last official evaluator feedback:\n{history[-1]['output']}\n"
            f"Own post-evaluation analysis:\n{history[-1].get('feedback_analysis', 'unavailable')}"
        )
        generated = json.loads(codex_call(
            f"""You are the sole Vanilla Codex ML developer in round {round_index}/6.
Iteratively repair and improve only from your own prior code and evaluator feedback. You have no MLEvolve search state, candidates, Journal, memory, logs, or results. Do not inspect files outside the described input or fabricate scores.

Frozen task:
{description}

Data preview:
{preview}

Task interpretation:
{setup_task}

Metric interpretation:
{setup_metric}

Own prior state:
{prior}

Return a concise plan and full executable Python code. The code must create both prediction files and print the exact final score line.""",
            call_logs, "VANILLA_CODEX", strict_schema(),
        ))
        reviewed = json.loads(codex_call(
            f"""Act as the final pre-execution reviewer for Vanilla Codex round {round_index}/6. Review only this arm's proposed code against the frozen task. Repair data-split, metric, output, determinism, package, path, or runtime defects while preserving or improving the approach. Return a concise plan and the full final executable code, not a diff. Do not inspect files or invent results.

Frozen task:
{description}

Proposed plan:
{generated['plan']}

Proposed code:
```python
{generated['code']}
```""",
            call_logs, "VANILLA_CODEX", strict_schema(),
        ))
        candidate_dir = arm / "candidates" / f"candidate_{round_index:02d}"
        result = execute_candidate(reviewed["code"], candidate_dir, public, python, task)
        record = {
            "candidate": round_index, "plan": reviewed["plan"], "code_sha256": sha256(candidate_dir / "candidate.py"),
            **result,
        }
        feedback = codex_call(
            f"""Analyze only your own round {round_index}/6 official evaluator result. Give a concise next-round repair or improvement direction. Do not write full code, inspect files, fabricate results, or refer to MLEvolve.

Frozen task:
{description}

Executed code:
```python
{reviewed['code']}
```

Evaluator output:
{result['output']}
Valid={result['valid']}; official fixed-validation score={result['score']}""",
            call_logs, "VANILLA_CODEX",
        )
        record["feedback_analysis"] = feedback
        history.append(record)
        (candidate_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        better = result["valid"] and (
            best_score is None or (result["score"] > best_score if task["maximize"] else result["score"] < best_score)
        )
        if better:
            best_score = result["score"]
            best_code = reviewed["code"]
            shutil.copytree(candidate_dir, arm / "best_candidate", dirs_exist_ok=True)

    usage = codex_usage(call_logs)
    trajectory = [item["score"] if item["valid"] else None for item in history]
    best_trajectory = best_so_far(trajectory, task["maximize"])
    checkpoints = []
    for record, best in zip(history, best_trajectory):
        token_state = cumulative_at(usage["timeline"], datetime.fromisoformat(record["completed_utc"]))
        checkpoints.append({
            "candidate": record["candidate"], "score": record["score"], "best_score": best,
            "completed_utc": record["completed_utc"],
            "total_tokens": token_state.get("total_tokens", 0),
            "uncached_input_tokens": token_state.get("uncached_input_tokens", 0),
        })
    valid_arm = (
        len(history) == protocol["formal_candidates_per_arm"]
        and best_score is not None
        and usage["calls"] == protocol["expected_codex_calls_per_arm"]
    )
    terminal = {
        "arm": "VANILLA_CODEX", "task": task["slug"],
        "harness_git_head": os.environ.get("SEARCH_VALUE_HARNESS_COMMIT"),
        "status": "VALID" if valid_arm else "INVALID",
        "candidates": len(history), "valid_candidates": sum(item["valid"] for item in history),
        "best_validation_score": best_score, "trajectory": trajectory,
        "candidate_checkpoints": checkpoints,
        "wall_seconds": time.time() - arm_started,
        "candidate_compute_wall_seconds": sum(item["execution_wall_seconds"] for item in history),
        "usage": {key: value for key, value in usage.items() if key != "timeline"},
        "best_submission_sha256": sha256(arm / "best_candidate" / "submission" / "submission.csv") if best_score is not None else None,
    }
    atomic_json(arm / "ARM_TERMINAL.json", terminal)
    print(json.dumps(terminal, indent=2))


if __name__ == "__main__":
    main()
