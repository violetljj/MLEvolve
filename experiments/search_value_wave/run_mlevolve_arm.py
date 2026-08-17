"""Run and receipt one frozen six-candidate MLEvolve Codex arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from wave_common import (
    atomic_json,
    best_so_far,
    codex_usage,
    cumulative_at,
    load_protocol,
    sha256,
)
from run_vanilla_arm import execute_candidate


LOCAL_ZONE = ZoneInfo("Asia/Hong_Kong")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    repo = args.repo_root.resolve()
    task_root = args.task_root.resolve()
    receipt = json.loads((task_root / "evidence" / "frozen_task_receipt.json").read_text())
    task = receipt["task"]
    if task != next(item for item in protocol["tasks"] if item["slug"] == task["slug"]):
        raise SystemExit("Task receipt differs from frozen protocol")
    arm = task_root / "MLEVOLVE_CODEX"
    if arm.exists():
        raise SystemExit(f"Refusing to overwrite arm: {arm}")
    arm.mkdir()
    runtime = arm / "runtime"
    call_logs = arm / "codex_calls"
    python = Path(os.environ["SEARCH_VALUE_PYTHON"])
    runner = repo / "experiments" / "search_value_test" / "codex_logged_runner.py"
    env = os.environ.copy()
    env["SEARCH_VALUE_ARM"] = "MLEVOLVE_CODEX"
    env["SEARCH_VALUE_CALL_LOG_DIR"] = str(call_logs)
    env["MLEVOLVE_CODEX_COMMAND"] = json.dumps([str(python), str(runner)])
    env["MLEVOLVE_CODEX_REASONING_EFFORT"] = protocol["reasoning_effort"]
    seed = protocol["split"]["base_seed"] + task["sequence"]
    command = [
        str(python), str(repo / "run.py"),
        f"data_dir={task_root / 'benchmark' / 'public'}",
        f"desc_file={task_root / 'benchmark' / 'public' / 'description.md'}",
        f"log_dir={runtime}", f"workspace_dir={runtime}",
        f"exp_name=wave_{task['slug']}_mlevolve", "preprocess_data=false", "copy_data=true",
        f"agent.steps={protocol['formal_candidates_per_arm']}",
        f"agent.time_limit={protocol['arm_timeout_seconds']}", "agent.initial_drafts=2",
        f"agent.seed={seed}", f"exec.timeout={protocol['candidate_timeout_seconds']}",
        f"agent.code.model=codex:{protocol['model']}",
        f"agent.feedback.model=codex:{protocol['model']}",
        "agent.check_data_leakage=true", "agent.use_diff_mode=false",
        "agent.use_stepwise_generation=false", "agent.use_evolution=true",
        "agent.use_fusion=true", "agent.use_aggregation=true", "agent.use_global_memory=true",
        "agent.memory_embedding_device=cpu", "agent.memory_embedding_model_path=BAAI/bge-base-en-v1.5",
        "agent.search.parallel_search_num=1", "agent.search.num_gpus=0",
        "agent.search.num_drafts=2", "start_cpu_id=0",
        f"cpu_number={protocol['cpu_threads']}", "coldstart.use_coldstart=false",
    ]
    started = time.time()
    completed = subprocess.run(command, cwd=repo, env=env, check=False)
    wall = time.time() - started
    launcher = {
        "returncode": completed.returncode, "wall_seconds": wall, "command": command,
    }
    atomic_json(arm / "launcher_receipt.json", launcher)
    journals = list(runtime.glob("*/logs/journal.json"))
    if len(journals) != 1:
        atomic_json(arm / "ARM_TERMINAL.json", {
            "arm": "MLEVOLVE_CODEX", "task": task["slug"], "status": "INVALID",
            "reason": f"expected one journal, found {len(journals)}", "wall_seconds": wall,
        })
        raise SystemExit(1)
    journal_path = journals[0]
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    nodes = [node for node in journal["nodes"] if node.get("stage") != "root"]
    train = pd.read_csv(task_root / "benchmark" / "public" / "train.csv")
    truth = train.loc[train["partition"].eq("validation"), ["id", "target"]]
    trajectory = []
    checkpoints_raw = []
    for node in nodes:
        value = node.get("metric", {}).get("value")
        valid = node.get("is_buggy") is False and node.get("is_valid") is not False and value is not None
        trajectory.append(float(value) if valid else None)
        finish = node.get("finish_time") or node.get("created_time")
        checkpoints_raw.append(finish)
    best_trajectory = best_so_far(trajectory, task["maximize"])
    usage = codex_usage(call_logs)
    checkpoints = []
    for index, (score, best, finish) in enumerate(zip(trajectory, best_trajectory, checkpoints_raw), 1):
        token_state = {"total_tokens": 0, "uncached_input_tokens": 0}
        completed_utc = None
        if finish:
            local = datetime.fromisoformat(finish).replace(tzinfo=LOCAL_ZONE)
            utc = local.astimezone(ZoneInfo("UTC"))
            completed_utc = utc.isoformat()
            token_state = cumulative_at(usage["timeline"], utc)
        checkpoints.append({
            "candidate": index, "score": score, "best_score": best,
            "completed_utc": completed_utc,
            "total_tokens": token_state.get("total_tokens", 0),
            "uncached_input_tokens": token_state.get("uncached_input_tokens", 0),
        })
    best_score = best_trajectory[-1] if best_trajectory else None
    best_submission = journals[0].parents[1] / "workspace" / "best_submission" / "submission.csv"
    replay_score = None
    replay_wall = None
    replay_submission_sha256 = None
    replay_submission_ids_match = False
    replay_submission_max_abs_diff = None
    replay_error = None
    try:
        eligible = [
            node for node, value in zip(nodes, trajectory)
            if value is not None and abs(value - best_score) <= 1e-12 * max(1.0, abs(best_score))
        ]
        if not eligible:
            raise ValueError("No journal node matches the selected best score")
        replay_dir = arm / "selected_code_dev_replay"
        replay = execute_candidate(
            eligible[0]["code"], replay_dir,
            task_root / "benchmark" / "public", python, task,
        )
        replay_score = replay["score"] if replay["valid"] else None
        replay_wall = replay["execution_wall_seconds"]
        replay_submission = replay_dir / "submission" / "submission.csv"
        replay_submission_sha256 = sha256(replay_submission) if replay_submission.exists() else None
        original_frame = pd.read_csv(best_submission)
        replay_frame = pd.read_csv(replay_submission)
        replay_submission_ids_match = original_frame["id"].tolist() == replay_frame["id"].tolist()
        original_values = pd.to_numeric(original_frame["prediction"], errors="coerce").to_numpy(float)
        replay_values = pd.to_numeric(replay_frame["prediction"], errors="coerce").to_numpy(float)
        if len(original_values) == len(replay_values) and np.isfinite(original_values).all() and np.isfinite(replay_values).all():
            replay_submission_max_abs_diff = float(np.max(np.abs(original_values - replay_values)))
    except Exception as exc:
        replay_error = f"{type(exc).__name__}: {exc}"
    valid_arm = (
        completed.returncode == 0
        and len(nodes) == protocol["formal_candidates_per_arm"]
        and best_score is not None
        and replay_score is not None
        and abs(replay_score - best_score) <= 1e-8 * max(1.0, abs(best_score))
        and usage["calls"] == protocol["expected_codex_calls_per_arm"]
        and best_submission.exists()
        and replay_submission_ids_match
        and replay_submission_max_abs_diff is not None
        and replay_submission_max_abs_diff <= 1e-12
    )
    terminal = {
        "arm": "MLEVOLVE_CODEX", "task": task["slug"],
        "harness_git_head": os.environ.get("SEARCH_VALUE_HARNESS_COMMIT"),
        "status": "VALID" if valid_arm else "INVALID",
        "candidates": len(nodes), "valid_candidates": sum(value is not None for value in trajectory),
        "best_validation_score": best_score, "replayed_validation_score": replay_score,
        "trajectory": trajectory, "stages": [node.get("stage") for node in nodes],
        "candidate_checkpoints": checkpoints, "wall_seconds": wall,
        "candidate_compute_wall_seconds": sum(float(node.get("exec_time") or 0) for node in nodes),
        "selected_code_audit_replay_wall_seconds": replay_wall,
        "selected_code_audit_replay_submission_sha256": replay_submission_sha256,
        "selected_code_audit_replay_ids_match": replay_submission_ids_match,
        "selected_code_audit_replay_max_abs_diff": replay_submission_max_abs_diff,
        "selected_code_audit_replay_error": replay_error,
        "usage": {key: value for key, value in usage.items() if key != "timeline"},
        "journal_sha256": sha256(journal_path),
        "best_submission_path": str(best_submission),
        "best_submission_sha256": sha256(best_submission) if best_submission.exists() else None,
    }
    atomic_json(arm / "ARM_TERMINAL.json", terminal)
    print(json.dumps(terminal, indent=2))
    if not valid_arm:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
