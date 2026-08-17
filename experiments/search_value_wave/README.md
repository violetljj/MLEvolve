# MLEvolve × Codex Search-Value Wave R1

This directory freezes and runs the nine-task paired admission screen that follows
the single diabetes result. The comparison is `VANILLA_CODEX` versus
`MLEVOLVE_CODEX` with the same Codex model, frozen task split, evaluator, candidate
count, CPU/GPU limits, and one private-test evaluation per completed pair.

The protocol is intentionally fail-closed:

- all nine OpenML data IDs, their order, metrics, arm order, budgets, normalization,
  tie band, endpoints, and verdict rules are frozen in `wave_protocol.json`;
- diabetes and the earlier breast-cancer smoke task are excluded;
- preparation records exact OpenML and generated-file hashes;
- an unavailable task is not replaced after seeing data or outcomes;
- private labels are not consumed until all 18 arm receipts exist;
- one task run per arm is an admission screen, not a stochastic-effect estimate.

Two questions remain separate. Equal candidate count tests whether search structure
produces a better solution. Actual token, wall, and candidate-compute receipts test
what that solution cost. Raw RMSE/AUC/accuracy values are never pooled across tasks;
each score is converted to improvement over a frozen task-local dummy baseline.

The canonical Windows runtime root is outside Git, for example:
`E:\MLEvolve-runtime\search-value-wave\r1`. Preparation and execution refuse to
overwrite non-empty or consumed evidence roots.

## Frozen execution sequence

1. Commit the protocol before retrieving any task payload.
2. Run `prepare_wave.py`; retain its hashes and baseline receipts.
3. Run the preflight, including Codex version/authentication smoke and one generated
   tiny local harness smoke that is not one of the nine formal tasks.
4. Run tasks in the frozen sequence and the frozen first-arm order. Complete both
   arms for a task before moving to the next task.
5. After all 18 arm terminal receipts pass the budget and integrity audit, run the
   one-shot private finalizer once.

No task outcome may be inspected to alter later task membership, metric, split,
budget, prompt, model, verdict rule, or failure handling.
