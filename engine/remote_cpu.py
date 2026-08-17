"""Optional stateless SSH execution transport for generated candidates."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


SAFE_FRAGMENT = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class RemoteExecution:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    wall_seconds: float
    receipt_path: Path


def load_remote_config() -> dict | None:
    raw = os.environ.get("MLEVOLVE_REMOTE_WORKER_JSON")
    if not raw:
        return None
    config = json.loads(raw)
    required = {"host", "port", "project", "run_prefix", "remote_python", "cpus_per_job"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Remote worker configuration missing: {missing}")
    return config


def safe_fragment(value: str, fallback: str) -> str:
    cleaned = SAFE_FRAGMENT.sub("-", value.lower()).strip("-.")
    return (cleaned or fallback)[:24]


def execute_remote(
    code: str,
    working_dir: Path,
    timeout: int,
    label: str,
    config: dict,
) -> RemoteExecution:
    """Upload code plus public input, execute it, and copy back allowlisted results."""
    working_dir = working_dir.resolve()
    public_input = working_dir / "input"
    if not public_input.is_dir():
        raise FileNotFoundError(f"Remote execution requires public input/: {public_input}")
    run_id = "-".join((
        safe_fragment(str(config["run_prefix"]), "run"),
        safe_fragment(label, "candidate"),
        uuid.uuid4().hex[:12],
    ))[:64]
    project = safe_fragment(str(config["project"]), "mlevolve")
    dispatcher = Path(__file__).parents[1] / "scripts" / "remote_cpu_worker" / "dispatch.py"
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="mlevolve-remote-") as temporary:
        temporary_root = Path(temporary)
        job = temporary_root / "jobs" / "candidate_01"
        job.mkdir(parents=True)
        (job / "candidate.py").write_text(code, encoding="utf-8")
        shutil.copytree(public_input, job / "input")
        results = temporary_root / "results"
        command = [
            sys.executable,
            str(dispatcher),
            "--host", str(config["host"]),
            "--port", str(int(config["port"])),
            "--project", project,
            "--run-id", run_id,
            "--jobs-root", str(temporary_root / "jobs"),
            "--results-root", str(results),
            "--remote-python", str(config["remote_python"]),
            "--concurrency", "1",
            "--cpus-per-job", str(int(config["cpus_per_job"])),
            "--timeout", str(int(timeout)),
            "--cleanup-remote",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 180,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Remote dispatcher failed: " + (completed.stderr or completed.stdout)[-4000:]
            )
        receipt = json.loads((results / "worker_receipt.json").read_text(encoding="utf-8"))
        job_receipt = receipt["jobs"][0]
        returned_job = results / "jobs" / "candidate_01"
        returned_submission = returned_job / "submission"
        destination_submission = working_dir / "submission"
        destination_submission.mkdir(parents=True, exist_ok=True)
        if returned_submission.is_dir():
            for path in returned_submission.iterdir():
                if path.is_file():
                    shutil.copy2(path, destination_submission / path.name)
        receipt_dir = working_dir / "working" / "remote_receipts"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"{run_id}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        stdout = (returned_job / "stdout.txt").read_text(encoding="utf-8", errors="replace")
        stderr = (returned_job / "stderr.txt").read_text(encoding="utf-8", errors="replace")
        return RemoteExecution(
            stdout=stdout,
            stderr=stderr,
            returncode=int(job_receipt["returncode"]),
            timed_out=bool(job_receipt["timed_out"]),
            wall_seconds=float(job_receipt.get("wall_seconds", time.time() - started)),
            receipt_path=receipt_path,
        )
