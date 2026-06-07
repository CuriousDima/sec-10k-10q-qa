# SIA task: SIA-FinCheck

This task directory was prepared from `/workspace/sec-10k-10q-sp100` for use with SIA from `/workspace/sec-10k-10q-qa`.

Run with:

```bash
cd /workspace/sec-10k-10q-qa
export SAKANA_API_KEY="..."

sia run \
  --task_dir ./tasks/sia-fincheck \
  --meta-agent-profile my-meta \
  --target-agent-profile my-target \
  --max_gen 3 \
  --run_id 1
```

The generated target agent should read `data/public/test.jsonl` through its `--dataset_dir` argument and write `submission.jsonl` to its `--working_dir`.

Evaluation is handled by `data/public/evaluate.py`, which scores `submission.jsonl` against `data/private/test_labels.jsonl` and writes `results.json`.
