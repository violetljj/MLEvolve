"""Create a disposable public-only batch for remote worker validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATE = """import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

job = json.loads(Path('input/job.json').read_text())
rng = np.random.default_rng(job['index'])
matrix = rng.normal(size=(300, 300))
value = float(np.linalg.norm(matrix @ matrix.T))
time.sleep(0.25)
Path('submission').mkdir(exist_ok=True)
pd.DataFrame({'id': [job['index']], 'prediction': [value]}).to_csv(
    'submission/submission.csv', index=False
)
pd.DataFrame({'id': [job['index']], 'prediction': [value]}).to_csv(
    'submission/validation_predictions.csv', index=False
)
print(json.dumps({'job': job['index'], 'pid': os.getpid(), 'value': value}))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    for index in range(args.jobs):
        job = args.root / f"candidate_{index + 1:02d}"
        (job / "input").mkdir(parents=True)
        (job / "candidate.py").write_text(CANDIDATE, encoding="utf-8")
        (job / "input" / "job.json").write_text(
            json.dumps({"index": index}), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
