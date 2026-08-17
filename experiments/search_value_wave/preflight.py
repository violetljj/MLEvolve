"""Record local runtime, Codex CLI version/authentication, and capacity evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn

from wave_common import PROTOCOL_PATH, atomic_json, git_head, load_protocol, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    remote_config_path = Path(__file__).with_name(protocol["execution_backend"]["config_file"])
    remote_config = json.loads(remote_config_path.read_text(encoding="utf-8"))
    node = Path(r"E:\codex-tools\tools\nodejs\node.exe")
    cli = Path(r"E:\codex-tools\tools\node-global\node_modules\@openai\codex\bin\codex.js")
    version = subprocess.run(
        [str(node), str(cli), "--version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60, check=False,
    )
    smoke = subprocess.run(
        [
            str(node), str(cli), "exec", "--ephemeral", "--sandbox", "read-only",
            "--skip-git-repo-check", "--color", "never", "-m", protocol["model"],
            "-c", f'model_reasoning_effort="{protocol["reasoning_effort"]}"',
            "Reply with exactly CODEX_WAVE_SMOKE_OK",
        ], cwd=args.repo_root.resolve(), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300, check=False,
    )
    usage = shutil.disk_usage(args.runtime_root.resolve())
    with tempfile.TemporaryDirectory(prefix="wave-remote-preflight-") as temporary:
        temporary_root = Path(temporary)
        jobs = temporary_root / "jobs"
        results = temporary_root / "results"
        make_canary = Path(__file__).parents[2] / "scripts" / "remote_cpu_worker" / "make_canary.py"
        dispatcher = make_canary.with_name("dispatch.py")
        create = subprocess.run(
            [sys.executable, str(make_canary), str(jobs), "--jobs", "1"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        remote_smoke = subprocess.run(
            [
                sys.executable, str(dispatcher),
                "--host", remote_config["host"], "--port", str(remote_config["port"]),
                "--project", remote_config["project"],
                "--run-id", f"r3-preflight-{uuid.uuid4().hex[:12]}",
                "--jobs-root", str(jobs), "--results-root", str(results),
                "--remote-python", remote_config["remote_python"],
                "--concurrency", "1", "--cpus-per-job", str(remote_config["cpus_per_job"]),
                "--timeout", "60", "--cleanup-remote",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=300, check=False,
        ) if create.returncode == 0 else None
        worker_receipt = None
        if remote_smoke is not None and remote_smoke.returncode == 0:
            worker_receipt = json.loads((results / "worker_receipt.json").read_text(encoding="utf-8"))
        remote_valid = bool(
            worker_receipt
            and worker_receipt["cpus_per_job"] == protocol["cpu_threads"]
            and len(worker_receipt["jobs"]) == 1
            and worker_receipt["jobs"][0]["returncode"] == 0
            and not worker_receipt["jobs"][0]["timed_out"]
        )
    receipt = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "harness_git_head": git_head(args.repo_root.resolve()),
        "codex_version": version.stdout.strip(),
        "codex_version_returncode": version.returncode,
        "codex_smoke_returncode": smoke.returncode,
        "codex_smoke_marker_present": "CODEX_WAVE_SMOKE_OK" in smoke.stdout,
        "codex_smoke_stdout_tail": smoke.stdout[-2000:],
        "codex_smoke_stderr_tail": smoke.stderr[-2000:],
        "packages": {
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "runtime_free_bytes": usage.free,
        "runtime_total_bytes": usage.total,
        "remote_worker_config_sha256": sha256(remote_config_path),
        "remote_worker_smoke_returncode": remote_smoke.returncode if remote_smoke else None,
        "remote_worker_smoke_stderr_tail": remote_smoke.stderr[-2000:] if remote_smoke else create.stderr[-2000:],
        "remote_worker_receipt": worker_receipt,
        "valid": (
            version.returncode == 0
            and version.stdout.strip().startswith("codex-cli ")
            and version.stdout.strip() != "codex-cli unknown"
            and smoke.returncode == 0
            and "CODEX_WAVE_SMOKE_OK" in smoke.stdout
            and usage.free >= 10 * 1024 ** 3
            and remote_valid
        ),
    }
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2))
    if not receipt["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
