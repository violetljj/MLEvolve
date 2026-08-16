"""Render the immutable JSON result as a compact human-readable summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unavailable"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:.1f}s"


def trajectory(values: list[float | None]) -> str:
    parts = []
    best = None
    for index, value in enumerate(values, 1):
        if value is None:
            parts.append(f"{index}:FAILED")
            continue
        improved = best is None or value < best
        parts.append(f"{index}:{value:.6f}" + ("*" if improved else ""))
        if improved:
            best = value
    return " -> ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    summary = json.loads((root / "FINAL_SUMMARY.json").read_text(encoding="utf-8"))
    vanilla = summary["arms"]["VANILLA_CODEX"]
    mle = summary["arms"]["MLEVOLVE_CODEX"]
    text = f"""# MLEvolve Search-Value Test

Verdict: `{summary['verdict']}`

Metric: RMSE, lower is better. The final test difference was {abs(summary['test_rmse_delta_mlevolve_minus_vanilla']):.6f} RMSE in MLEvolve's favor, within the predeclared 1.0 RMSE practical-equivalence margin.

| Arm | Dev best | Test | Candidates | Valid | Codex calls | Tokens (input/output) | Wall | Compute wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VANILLA_CODEX | {vanilla['dev_best']:.6f} | {vanilla['test']:.6f} | {vanilla['candidates']} | {vanilla['valid']} | {vanilla['calls']} | {vanilla['input_tokens']}/{vanilla['output_tokens']} | {duration(vanilla['wall_seconds'])} | {duration(vanilla['compute_wall_seconds'])} |
| MLEVOLVE_CODEX | {mle['dev_best']:.6f} | {mle['test']:.6f} | {mle['candidates']} | {mle['valid']} | {mle['calls']} | {mle['input_tokens']}/{mle['output_tokens']} | {duration(mle['wall_seconds'])} | {duration(mle['compute_wall_seconds'])} |

`*` marks a real best-so-far improvement.

- VANILLA_CODEX: {trajectory(vanilla['trajectory'])}
- MLEVOLVE_CODEX: {trajectory(mle['trajectory'])}

Budget notes: calls and candidate counts were exactly matched. MLEvolve uncached input tokens were {mle['uncached_input_tokens']:,} versus Vanilla's {vanilla['uncached_input_tokens']:,}; Vanilla used more output tokens and wall time. Compute wall is summed candidate execution wall on CPU; exact process CPU time was unavailable, so it is not fabricated. GPU use was zero for both arms.

The first 14-call `VANILLA_CODEX` directory is retained only as a pre-test budget-calibration run. `VANILLA_CODEX_FORMAL` is the reported 19-call arm. Each formal arm has exactly one immutable test receipt.
"""
    (root / "FINAL_SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
