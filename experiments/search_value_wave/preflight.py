"""Record local runtime, Codex CLI version/authentication, and capacity evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
        "valid": (
            version.returncode == 0
            and version.stdout.strip().startswith("codex-cli ")
            and version.stdout.strip() != "codex-cli unknown"
            and smoke.returncode == 0
            and "CODEX_WAVE_SMOKE_OK" in smoke.stdout
            and usage.free >= 10 * 1024 ** 3
        ),
    }
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2))
    if not receipt["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
