"""Resume-safe orchestrator for the frozen paired wave (private test never read)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from wave_common import atomic_json, git_head, load_protocol, sha256


def terminal_valid(task_root: Path, arm: str) -> bool:
    path = task_root / arm / "ARM_TERMINAL.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "VALID"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--wave-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    root = args.wave_root.resolve()
    protocol = load_protocol()
    remote_config_path = Path(__file__).with_name(protocol["execution_backend"]["config_file"])
    remote_config = json.loads(remote_config_path.read_text(encoding="utf-8"))
    if int(remote_config["cpus_per_job"]) != int(protocol["cpu_threads"]):
        raise SystemExit("Remote worker CPU allocation differs from frozen protocol")
    if not (root / "PREPARATION_COMPLETE.json").exists():
        raise SystemExit("Wave preparation is not complete")
    frozen = root / "FROZEN_PROTOCOL.json"
    if sha256(frozen) != sha256(Path(__file__).with_name("wave_protocol.json")):
        raise SystemExit("Runtime protocol differs from repository-frozen protocol")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=60, check=True,
    ).stdout.strip()
    if status:
        raise SystemExit("Formal execution requires a clean harness worktree")
    head = git_head(repo)
    execution_start = root / "EXECUTION_START.json"
    if execution_start.exists():
        frozen_start = json.loads(execution_start.read_text(encoding="utf-8"))
        if frozen_start["harness_git_head"] != head:
            raise SystemExit("Harness commit changed after formal execution started")
        if frozen_start["remote_worker_config_sha256"] != sha256(remote_config_path):
            raise SystemExit("Remote worker configuration changed after formal execution started")
    else:
        atomic_json(execution_start, {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256(frozen), "harness_git_head": head,
            "remote_worker_config_sha256": sha256(remote_config_path),
        })
    python = Path(os.environ["SEARCH_VALUE_PYTHON"])
    child_env = os.environ.copy()
    child_env["SEARCH_VALUE_HARNESS_COMMIT"] = head
    child_env["MLEVOLVE_REMOTE_WORKER_JSON"] = json.dumps(remote_config)
    for task in protocol["tasks"]:
        task_root = root / "tasks" / task["slug"]
        order = [task["first_arm"], "VANILLA_CODEX" if task["first_arm"] == "MLEVOLVE_CODEX" else "MLEVOLVE_CODEX"]
        for arm in order:
            if terminal_valid(task_root, arm):
                continue
            arm_root = task_root / arm
            if arm_root.exists():
                raise SystemExit(f"Existing non-valid arm requires audit, not overwrite: {arm_root}")
            script = "run_vanilla_arm.py" if arm == "VANILLA_CODEX" else "run_mlevolve_arm.py"
            command = [str(python), str(Path(__file__).with_name(script)), "--task-root", str(task_root)]
            if arm == "MLEVOLVE_CODEX":
                command.extend(["--repo-root", str(repo)])
            completed = subprocess.run(command, cwd=repo, env=child_env, check=False)
            if completed.returncode != 0 or not terminal_valid(task_root, arm):
                raise SystemExit(f"Frozen arm failed and wave stopped: {task['slug']} {arm}")
    print(json.dumps({"status": "ALL_18_ARMS_VALID", "wave_root": str(root)}, indent=2))


if __name__ == "__main__":
    main()
