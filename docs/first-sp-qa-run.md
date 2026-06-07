# First SIA-FinCheck S&P 100 QA Run

This project is configured to run SIA against the SIA-FinCheck SEC 10-K/10-Q QA task prepared from:

```text
/workspace/sec-10k-10q-sp100
```

The SIA working project is:

```text
/workspace/sec-10k-10q-qa
```

## Prepared task directory

The SIA task directory is:

```text
/workspace/sec-10k-10q-qa/tasks/sia-fincheck
```

Key files:

```text
tasks/sia-fincheck/
├── data/
│   ├── public/
│   │   ├── task.md
│   │   ├── evaluate.py
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   ├── test.jsonl
│   │   ├── sample_submission.jsonl
│   │   └── metadata/
│   └── private/
│       ├── train_labels.jsonl
│       ├── validation_labels.jsonl
│       └── test_labels.jsonl
└── reference/
    ├── reference_target_agent.py
    └── SAMPLE_TASK_DESCRIPTIONS.md
```

The target agent should read public data via `--dataset_dir` and write predictions to:

```text
<working_dir>/submission.jsonl
```

The evaluator scores that file against private `test_labels.jsonl` and writes:

```text
<working_dir>/results.json
```

## Model/API profiles

Local SIA profiles/providers were created in:

```text
providers/sakana.json
profiles/my-meta.json
profiles/my-target.json
profiles/default-meta.json
profiles/default-target.json
```

The local defaults point to Sakana AI's OpenAI-compatible API:

```text
https://api.sakana.ai/v1
```

- Meta/feedback profile: `my-meta` (`fugu-ultra`, OpenHands)
- Target profile: `my-target` (`fugu-mini`)

## Run command

Do not commit API keys. Export the key in your shell before running:

```bash
cd /workspace/sec-10k-10q-qa

export SAKANA_API_KEY="YOUR_API_KEY"

sia run \
  --task_dir ./tasks/sia-fincheck \
  --meta-agent-profile my-meta \
  --target-agent-profile my-target \
  --max_gen 3 \
  --run_id 1
```

Because local `default-meta` and `default-target` profiles were also created, this shorter command should work from `/workspace/sec-10k-10q-qa`:

```bash
cd /workspace/sec-10k-10q-qa

export SAKANA_API_KEY="YOUR_API_KEY"

sia run \
  --task_dir ./tasks/sia-fincheck \
  --max_gen 3 \
  --run_id 1
```

If `runs/run_1` already exists, use another run ID:

```bash
sia run \
  --task_dir ./tasks/sia-fincheck \
  --max_gen 3 \
  --run_id 2
```

## Visualize results

SIA normally starts the dashboard automatically during a run. To serve it manually:

```bash
cd /workspace/sec-10k-10q-qa

sia web --runs-dir ./runs --port 8000
```

Open:

```text
http://127.0.0.1:8000
```
