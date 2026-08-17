# Stateless SSH CPU worker

The local controller remains authoritative for search state, protocol, scoring, and evidence. The remote host receives only `candidate.py` and a job's public `input/`; it returns predictions, logs, hashes, timing, and a worker receipt.

Isolation rules:

- Remote paths are namespaced as `/root/autodl-tmp/workers/<project>/runs/<run-id>`.
- A run directory is create-once. Reusing a `run-id` fails instead of overwriting another run.
- Every candidate is pinned to one logical CPU and numerical-library threads are capped at one.
- Concurrency is explicitly bounded to 32; candidate processes have a hard timeout and process-group cleanup.
- The dispatcher never uploads private evaluator labels or the controller's Evidence Ledger.
- Remote cleanup targets only the validated project/run path and is opt-in.
- A host-wide `flock` lease permits only one CPU-saturating batch at a time, so separate projects fail fast instead of silently oversubscribing the same cores.

Example:

```powershell
python scripts/remote_cpu_worker/dispatch.py `
  --host root@connect.westb.seetacloud.com --port 16288 `
  --project mlevolve --run-id wave-r3-canary-001 `
  --jobs-root E:\candidate-batch --results-root E:\candidate-results `
  --remote-python /root/autodl-tmp/workers/mlevolve/envs/cpu-py311/bin/python `
  --concurrency 32 --cpus-per-job 1 --timeout 300
```

Do not use this backend in an already-started frozen experiment. Register the execution backend, remote environment fingerprint, concurrency, CPU/thread limits, timeout, artifact allowlist, and failure semantics before a new formal wave begins.

For an eight-thread sequential candidate budget, use `--concurrency 1 --cpus-per-job 8`.
