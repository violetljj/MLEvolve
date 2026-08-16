"""Run Codex CLI while preserving per-call prompts, event streams, and usage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _next_call_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.glob("call_*"):
        try:
            existing.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    call_dir = root / f"call_{max(existing, default=0) + 1:03d}"
    call_dir.mkdir()
    return call_dir


def _usage_from_events(stdout: str) -> dict[str, int] | None:
    usages: list[dict] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage")
        if isinstance(usage, dict):
            usages.append(usage)
    if not usages:
        return None
    # Codex emits cumulative usage in turn.completed; the last usage is authoritative.
    usage = usages[-1]
    result = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result or None


def main() -> int:
    log_root_raw = os.environ.get("SEARCH_VALUE_CALL_LOG_DIR")
    if not log_root_raw:
        raise SystemExit("SEARCH_VALUE_CALL_LOG_DIR is required")
    call_dir = _next_call_dir(Path(log_root_raw))
    prompt = sys.stdin.read()
    (call_dir / "prompt.md").write_text(prompt, encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    if not args:
        raise SystemExit("Expected Codex CLI arguments")
    if args[0] == "exec" and "--json" not in args:
        args.insert(1, "--json")

    node_exe = os.environ.get(
        "SEARCH_VALUE_NODE_EXE", r"E:\codex-tools\tools\nodejs\node.exe"
    )
    codex_js = os.environ.get(
        "SEARCH_VALUE_CODEX_JS",
        r"E:\codex-tools\tools\node-global\node_modules\@openai\codex\bin\codex.js",
    )
    command = [node_exe, codex_js, *args]
    started_wall = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    elapsed = time.time() - started_wall
    (call_dir / "stdout.jsonl").write_text(completed.stdout, encoding="utf-8")
    (call_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    metadata = {
        "arm": os.environ.get("SEARCH_VALUE_ARM", "UNKNOWN"),
        "started_utc": started_iso,
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
        "command": command,
        "usage": _usage_from_events(completed.stdout),
    }
    (call_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
