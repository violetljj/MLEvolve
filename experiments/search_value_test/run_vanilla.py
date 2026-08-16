"""Budgeted multi-round Vanilla Codex arm with no MLEvolve search state."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd


SCORE_RE = re.compile(r"Final Validation Score:\s*([-+0-9.eE]+)")


def strict_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "code": {"type": "string"},
        },
        "required": ["plan", "code"],
        "additionalProperties": False,
    }


def codex_call(
    prompt: str, call_logs: Path, arm_name: str, schema: dict | None = None
) -> str:
    python = Path(os.environ["SEARCH_VALUE_PYTHON"])
    runner = Path(__file__).with_name("codex_logged_runner.py").resolve()
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


def execute_candidate(code: str, candidate_dir: Path, public: Path, python: Path) -> dict:
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
    match = SCORE_RE.search(output)
    if match:
        score = float(match.group(1))
    else:
        valid = False
    submission = candidate_dir / "submission" / "submission.csv"
    validation_predictions = candidate_dir / "submission" / "validation_predictions.csv"
    try:
        test = pd.read_csv(public / "test.csv")
        pred = pd.read_csv(submission)
        valid = valid and list(pred.columns) == ["id", "prediction"]
        valid = valid and pred["id"].tolist() == test["id"].tolist()
        valid = valid and pred["prediction"].notna().all()
        validation = pd.read_csv(validation_predictions)
        expected_ids = pd.read_csv(public / "train.csv").query(
            "partition == 'validation'"
        )["id"].tolist()
        valid = valid and list(validation.columns) == ["id", "prediction"]
        valid = valid and sorted(validation["id"].tolist()) == sorted(expected_ids)
        valid = valid and validation["prediction"].notna().all()
    except Exception as exc:
        output += f"\nOutput validation failure: {exc}\n"
        valid = False
    return {
        "valid": bool(valid), "score": score if valid else None,
        "returncode": completed.returncode, "execution_wall_seconds": elapsed,
        "output": output[-12000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--arm-name", default="VANILLA_CODEX")
    parser.add_argument("--setup-calls", type=int, choices=(1, 2), default=2)
    parser.add_argument("--post-eval-analysis", action="store_true")
    args = parser.parse_args()
    experiment_root = args.experiment_root.resolve()
    public = experiment_root / "benchmark" / "public"
    arm = experiment_root / args.arm_name
    if arm.exists():
        raise SystemExit(f"Refusing to overwrite arm: {arm}")
    arm.mkdir(parents=True)
    call_logs = arm / "codex_calls"
    python = Path(os.environ["SEARCH_VALUE_PYTHON"])
    description = (public / "description.md").read_text(encoding="utf-8")
    train = pd.read_csv(public / "train.csv")
    preview = (
        f"train shape={train.shape}; columns={train.columns.tolist()}; "
        f"partition counts={train['partition'].value_counts().to_dict()}"
    )
    arm_start = time.time()

    setup_task = codex_call(
        "Read the frozen task below. Summarize its exact data isolation, evaluator, output, "
        "and resource constraints. Do not write code and do not inspect any files or other arm artifacts.\n\n"
        + description + "\n\n" + preview,
        call_logs, args.arm_name,
    )
    if args.setup_calls == 2:
        setup_metric = codex_call(
            "For this frozen task, state the optimization direction and explain how a fixed-partition "
            "RMSE evaluator must be computed. Do not write code or inspect files.\n\n" + description,
            call_logs, args.arm_name,
        )
    else:
        setup_metric = setup_task
    (arm / "setup_task.txt").write_text(setup_task, encoding="utf-8")
    (arm / "setup_metric.txt").write_text(setup_metric, encoding="utf-8")

    history = []
    best_code = None
    best_score = float("inf")
    for round_index in range(1, args.rounds + 1):
        prior = "No previous candidate exists." if best_code is None else (
            f"Current best fixed-validation RMSE: {best_score}\n"
            f"Current best code:\n```python\n{best_code}\n```\n"
            f"Last evaluator feedback:\n{history[-1]['output']}\n"
            f"Own post-evaluation analysis:\n{history[-1].get('feedback_analysis', 'unavailable')}"
        )
        generation_prompt = f"""You are the sole Vanilla Codex ML developer in round {round_index}/{args.rounds}.
You may iteratively repair and improve only from your own prior code and evaluator feedback below. You have no MLEvolve candidates, Journal, memory, logs, or results; do not inspect any files outside the described input. Produce a complete executable candidate. Do not fabricate scores.

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

Return a concise plan and the full Python code. The code must create both required prediction files and print the exact final score line."""
        generated = json.loads(
            codex_call(generation_prompt, call_logs, args.arm_name, strict_schema())
        )
        review_prompt = f"""Act as the final pre-execution reviewer for Vanilla Codex round {round_index}/{args.rounds}. Review only this arm's proposed code against the frozen task. Repair any data-split, RMSE, path, determinism, package, or runtime defect while preserving or improving the ML approach. Return the full final executable code, not a diff. Do not inspect files and do not invent evaluator results.

Frozen task:
{description}

Proposed plan:
{generated['plan']}

Proposed code:
```python
{generated['code']}
```"""
        reviewed = json.loads(
            codex_call(review_prompt, call_logs, args.arm_name, strict_schema())
        )
        candidate_dir = arm / "candidates" / f"candidate_{round_index:02d}"
        result = execute_candidate(reviewed["code"], candidate_dir, public, python)
        record = {
            "candidate": round_index, "plan": reviewed["plan"],
            "valid": result["valid"], "score": result["score"],
            "execution_wall_seconds": result["execution_wall_seconds"],
            "output": result["output"],
        }
        history.append(record)
        if args.post_eval_analysis:
            feedback_prompt = f"""You are Vanilla Codex analyzing only your own round {round_index}/{args.rounds} evaluator result. Summarize the concrete modeling outcome or failure and give a concise next-round improvement or repair direction. Do not write full code, fabricate results, inspect files, or refer to MLEvolve or any other arm.

Frozen task:
{description}

Executed code:
```python
{reviewed['code']}
```

Evaluator result:
{result['output']}
Valid={result['valid']}; fixed-validation RMSE={result['score']}"""
            record["feedback_analysis"] = codex_call(
                feedback_prompt, call_logs, args.arm_name
            )
        (candidate_dir / "result.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        if result["valid"] and result["score"] < best_score:
            best_score = result["score"]
            best_code = reviewed["code"]
            shutil.copytree(candidate_dir, arm / "best_candidate", dirs_exist_ok=True)

    summary = {
        "arm": args.arm_name, "candidates": len(history),
        "valid": sum(1 for item in history if item["valid"]),
        "failed": sum(1 for item in history if not item["valid"]),
        "dev_best": best_score if best_code is not None else None,
        "wall_seconds": time.time() - arm_start,
        "compute_wall_seconds": sum(item["execution_wall_seconds"] for item in history),
        "trajectory": [item["score"] if item["valid"] else None for item in history],
    }
    (arm / "arm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
