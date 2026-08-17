"""Package public candidate jobs, dispatch them over SSH, and fetch receipts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
REMOTE_BASE = "/root/autodl-tmp/workers"


def checked_name(label: str, value: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise SystemExit(f"Invalid {label}: {value!r}")
    return value


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, encoding="utf-8", errors="replace", **kwargs)


def extract_results(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise SystemExit(f"Unsafe result archive member: {member.name}")
        handle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--cpus-per-job", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cleanup-remote", action="store_true")
    args = parser.parse_args()

    project = checked_name("project", args.project)
    run_id = checked_name("run-id", args.run_id)
    if not SAFE_REMOTE_PATH.fullmatch(args.remote_python) or ".." in Path(args.remote_python).parts:
        raise SystemExit(f"Invalid remote Python path: {args.remote_python!r}")
    if args.concurrency < 1 or args.cpus_per_job < 1:
        raise SystemExit("concurrency and cpus-per-job must be positive")
    if args.concurrency * args.cpus_per_job > 32:
        raise SystemExit("concurrency * cpus-per-job must not exceed 32")
    jobs_root = args.jobs_root.resolve()
    jobs = sorted(path for path in jobs_root.iterdir() if path.is_dir())
    if not jobs or any(not (path / "candidate.py").is_file() for path in jobs):
        raise SystemExit("Each job directory must contain candidate.py")
    if any(not (path / "input").is_dir() for path in jobs):
        raise SystemExit("Each job directory must contain public input/")

    connection_options = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
    ]
    ssh = ["ssh", "-p", str(args.port), *connection_options, args.host]
    scp = ["scp", "-P", str(args.port), *connection_options]
    remote_run = f"{REMOTE_BASE}/{project}/runs/{run_id}"
    runtime = Path(__file__).with_name("worker_runtime.py")
    with tempfile.TemporaryDirectory(prefix="remote-worker-") as temporary:
        stage = Path(temporary) / "payload"
        staged_jobs = stage / "jobs"
        staged_jobs.mkdir(parents=True)
        manifest_jobs = []
        for job in jobs:
            name = checked_name("job directory", job.name)
            target = staged_jobs / name
            target.mkdir()
            shutil.copy2(job / "candidate.py", target / "candidate.py")
            shutil.copytree(job / "input", target / "input")
            (target / "submission").mkdir()
            (target / "working").mkdir()
            manifest_jobs.append(name)
        (stage / "dispatch_manifest.json").write_text(
            json.dumps({"project": project, "run_id": run_id, "jobs": manifest_jobs}, indent=2),
            encoding="utf-8",
        )
        archive = Path(temporary) / "payload.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(stage, arcname=".")

        run(ssh + [f"mkdir -p '{REMOTE_BASE}/{project}/runs' && mkdir '{remote_run}'"])
        operation_succeeded = False
        try:
            run(scp + [str(archive), str(runtime), f"{args.host}:{remote_run}/"])
            remote_command = (
                f"cd '{remote_run}' && tar -xzf payload.tar.gz && "
                f"flock -n '{REMOTE_BASE}/.cpu-worker.lock' "
                f"'{args.remote_python}' worker_runtime.py --run-dir '{remote_run}' "
                f"--python '{args.remote_python}' --concurrency {args.concurrency} "
                f"--cpus-per-job {args.cpus_per_job} --timeout {args.timeout}"
            )
            completed = run(ssh + [remote_command], capture_output=True)
            args.results_root.mkdir(parents=True, exist_ok=False)
            run(scp + [f"{args.host}:{remote_run}/results.tar.gz", str(args.results_root)])
            extract_results(args.results_root / "results.tar.gz", args.results_root)
            (args.results_root / "dispatch_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            operation_succeeded = True
        finally:
            if args.cleanup_remote:
                cleanup = subprocess.run(
                    ssh + [f"test '{remote_run}' != / && rm -rf -- '{remote_run}'"],
                    check=False,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if operation_succeeded and cleanup.returncode != 0:
                    raise RuntimeError(f"Remote cleanup failed: {cleanup.stderr[-2000:]}")


if __name__ == "__main__":
    main()
