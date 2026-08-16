"""Launch the frozen MLEvolve arm and preserve launcher-level wall evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: launch_mlevolve.py OUTPUT_JSON COMMAND [ARGS...]")
    output = Path(sys.argv[1]).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite launcher receipt: {output}")
    started = time.time()
    completed = subprocess.run(sys.argv[2:], env=os.environ.copy(), check=False)
    receipt = {
        "returncode": completed.returncode,
        "wall_seconds": time.time() - started,
        "command": sys.argv[2:],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
