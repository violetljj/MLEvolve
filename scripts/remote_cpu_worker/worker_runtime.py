"""Run an isolated batch of candidate programs on a CPU worker host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_job(job_dir: Path, python: Path, timeout: int, cpu_ids: list[int]) -> dict:
    thread_count = str(len(cpu_ids))
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
        "VECLIB_MAXIMUM_THREADS": thread_count,
        "PYTHONUNBUFFERED": "1",
    })
    started = time.time()
    taskset = shutil.which("taskset")
    command = [str(python), "candidate.py"]
    if taskset:
        command = [taskset, "-c", ",".join(map(str, cpu_ids)), *command]
    proc = subprocess.Popen(
        command,
        cwd=job_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
    elapsed = time.time() - started
    (job_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (job_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    result = {
        "job_id": job_dir.name,
        "cpu_ids": cpu_ids,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "wall_seconds": elapsed,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": file_hash(job_dir / "candidate.py"),
        "submission_sha256": file_hash(job_dir / "submission" / "submission.csv"),
        "validation_predictions_sha256": file_hash(
            job_dir / "submission" / "validation_predictions.csv"
        ),
    }
    (job_dir / "worker_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def safe_result_member(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if relative.name in {"candidate.py", "stdout.txt", "stderr.txt", "worker_result.json"}:
        return True
    return relative.parts[-2:-1] == ("submission",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--cpus-per-job", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    jobs_dir = run_dir / "jobs"
    jobs = sorted(path for path in jobs_dir.iterdir() if (path / "candidate.py").is_file())
    available = sorted(os.sched_getaffinity(0))
    if not jobs:
        raise SystemExit("No candidate jobs found")
    if args.concurrency < 1 or args.cpus_per_job < 1:
        raise SystemExit("concurrency and cpus-per-job must be positive")
    if args.concurrency * args.cpus_per_job > len(available):
        raise SystemExit(
            f"requested {args.concurrency * args.cpus_per_job} CPUs, only {len(available)} available"
        )

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                run_job,
                job,
                args.python,
                args.timeout,
                available[
                    (index % args.concurrency) * args.cpus_per_job:
                    (index % args.concurrency + 1) * args.cpus_per_job
                ],
            ): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["job_id"])
    receipt = {
        "schema_version": 1,
        "host": os.uname().nodename,
        "logical_cpus": len(available),
        "concurrency": args.concurrency,
        "cpus_per_job": args.cpus_per_job,
        "timeout_seconds": args.timeout,
        "python": str(args.python),
        "jobs": results,
    }
    receipt_path = run_dir / "worker_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    archive = run_dir / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(receipt_path, arcname="worker_receipt.json")
        for path in jobs_dir.rglob("*"):
            if path.is_file() and safe_result_member(path, jobs_dir):
                handle.add(path, arcname=str(Path("jobs") / path.relative_to(jobs_dir)))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
