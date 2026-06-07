#!/usr/bin/env python3
"""Option-A / fake-RL-as-harness seed for SIA-FinCheck.

This target agent performs local self-training inside normal SIA harness mode:
1. builds weak/pseudo labels from the public SEC filing contexts,
2. LoRA-fine-tunes a local Gemma checkpoint using both GPUs via `accelerate launch`,
3. saves the adapter to `working_dir/checkpoints/adapter`,
4. runs inference DIRECTLY from `LOCAL_GEMMA_MODEL_PATH + working_dir/checkpoints/adapter`,
5. writes `submission.jsonl` and per-example trajectories for SIA evaluation/feedback.

IMPORTANT FOR SIA META/FEEDBACK AGENTS:
- Do NOT replace this with OpenAI/vLLM calls during inference.
- Do NOT call `LOCAL_GEMMA_BASE_URL` or `http://localhost:8000/v1`.
- The whole point of Option A is to train and then immediately load the newly saved PEFT adapter locally.
- For gen_N, continue from gen_{N-1}/checkpoints/adapter if it exists, then save the new adapter in gen_N.

Required runtime env:
  LOCAL_GEMMA_MODEL_PATH=/path/to/trainable/gemma/checkpoint

Useful optional env:
  CUDA_VISIBLE_DEVICES=0,1
  SIA_FINCHECK_MAX_TRAIN_EXAMPLES=512
  SIA_FINCHECK_MAX_STEPS=80
  SIA_FINCHECK_MAX_CONTEXT_CHARS=18000
  SIA_FINCHECK_TRAIN=1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_CONTEXT_CHARS = int(os.getenv("SIA_FINCHECK_MAX_CONTEXT_CHARS", "18000"))
MAX_TRAIN_EXAMPLES = int(os.getenv("SIA_FINCHECK_MAX_TRAIN_EXAMPLES", "512"))
MAX_STEPS = int(os.getenv("SIA_FINCHECK_MAX_STEPS", "80"))
PER_DEVICE_BATCH = int(os.getenv("SIA_FINCHECK_PER_DEVICE_BATCH", "1"))
GRAD_ACCUM = int(os.getenv("SIA_FINCHECK_GRAD_ACCUM", "8"))
LORA_R = int(os.getenv("SIA_FINCHECK_LORA_R", "32"))
LORA_ALPHA = int(os.getenv("SIA_FINCHECK_LORA_ALPHA", "64"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def question_terms(question: str) -> set[str]:
    stop = {
        "what", "were", "was", "the", "company", "companies", "for", "and", "of", "as", "at", "to", "in",
        "on", "ended", "year", "quarter", "fiscal", "total", "end", "did", "from", "with", "that", "this",
        "its", "their", "amount", "value", "reported", "report", "during", "how", "many", "much",
    }
    return {tok for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower()) if tok not in stop}


def context_excerpt(context: str, question: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(context) <= max_chars:
        return context
    terms = question_terms(question)
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        low = line.lower()
        score = sum(4 for term in terms if term in low)
        score += min(len(re.findall(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?", line)), 5)
        if score:
            scored.append((score, idx, line))
    selected_indices: set[int] = set()
    for _score, idx, _line in sorted(scored, reverse=True)[:100]:
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected_indices.add(j)
    pieces = ["\n".join(lines[:60])[:3500], "\n--- relevant excerpt lines ---"]
    pieces.extend(lines[idx] for idx in sorted(selected_indices))
    return "\n".join(pieces)[:max_chars]


def make_prompt(example: dict[str, Any]) -> str:
    excerpt = context_excerpt(example.get("context", ""), example.get("question", ""))
    return f"""You are solving one SIA-FinCheck numerical QA item from SEC filing excerpts.

Return JSON only with keys: answer, unit, confidence, reasoning.

Normalization rules:
- Currency answers must be raw USD. If a table is in millions, multiply by 1,000,000.
- Percent answers are percentage points, e.g. 12.5 for 12.5%.
- Per-share answers use unit USD/share.
- Ratios use unit ratio.

ID: {example.get('id')}
Company: {example.get('company_name')} ({example.get('ticker')})
Form: {example.get('form')}
Report date: {example.get('report_date')}
Expected unit: {example.get('expected_unit')}
Answer type: {example.get('answer_type')}
Question: {example.get('question')}

Filing context excerpt:
{excerpt}
"""


def parse_number_text(text: str) -> float | None:
    if not text:
        return None
    neg = False
    raw = text.strip()
    if re.fullmatch(r"\(.*\)", raw):
        neg = True
        raw = raw[1:-1]
    raw = raw.replace("$", "").replace(",", "").replace("%", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if not match:
        return None
    val = float(match.group(0))
    if neg and val > 0:
        val = -val
    low = text.lower()
    if re.search(r"\b(billion|billions|bn)\b", low):
        val *= 1_000_000_000
    elif re.search(r"\b(million|millions|mm)\b", low):
        val *= 1_000_000
    elif re.search(r"\b(thousand|thousands)\b", low):
        val *= 1_000
    return val if math.isfinite(val) else None


def weak_label(example: dict[str, Any]) -> dict[str, Any]:
    """Create a weak pseudo-label from public context only; no private labels are read."""
    question = example.get("question", "")
    terms = question_terms(question)
    lines = [line.strip() for line in str(example.get("context", "")).splitlines() if line.strip()]
    candidates: list[tuple[int, float, str]] = []
    for line in lines:
        nums = re.findall(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?(?:\s*(?:million|millions|billion|billions|thousand|thousands|bn|mm))?", line, flags=re.I)
        if not nums:
            continue
        low = line.lower()
        score = sum(5 for term in terms if term in low) + min(len(nums), 4)
        for num_text in nums:
            val = parse_number_text(num_text)
            if val is not None:
                candidates.append((score, val, line[:300]))
    candidates.sort(key=lambda x: x[0], reverse=True)
    unit = example.get("expected_unit") or {"currency": "USD", "percent": "percent", "ratio": "ratio"}.get(example.get("answer_type"), "")
    if candidates:
        answer = candidates[0][1]
        reason = f"Weak pseudo-label from context line: {candidates[0][2]}"
    else:
        answer = 0.0
        reason = "Weak fallback pseudo-label; no numeric candidate found."
    return {"answer": answer, "unit": unit, "confidence": 0.35, "reasoning": reason}


def training_text(example: dict[str, Any]) -> str:
    label = weak_label(example)
    return make_prompt(example) + "\n" + json.dumps(label, ensure_ascii=False)


def run_accelerate_training(dataset_dir: Path, working_dir: Path) -> None:
    if os.getenv("SIA_FINCHECK_TRAIN", "1") == "0":
        print("Training disabled with SIA_FINCHECK_TRAIN=0")
        return
    model_path = os.getenv("LOCAL_GEMMA_MODEL_PATH")
    if not model_path:
        print("LOCAL_GEMMA_MODEL_PATH is not set; skipping local fine-tuning.")
        return
    adapter_dir = working_dir / "checkpoints" / "adapter"
    if adapter_dir.exists():
        print(f"Adapter already exists at {adapter_dir}; skipping training.")
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", os.getenv("SIA_NUM_GPUS", "2"),
        "--mixed_precision", "bf16",
        str(Path(__file__).resolve()),
        "--dataset_dir", str(dataset_dir),
        "--working_dir", str(working_dir),
        "--train-worker",
    ]
    print("Launching 2-GPU local LoRA training:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def previous_adapter_dir(working_dir: Path) -> Path | None:
    """Return gen_{N-1}/checkpoints/adapter for cumulative weight evolution, if present."""
    match = re.fullmatch(r"gen_(\d+)", working_dir.name)
    if not match:
        return None
    gen_num = int(match.group(1))
    if gen_num <= 1:
        return None
    candidate = working_dir.parent / f"gen_{gen_num - 1}" / "checkpoints" / "adapter"
    if (candidate / "adapter_config.json").is_file():
        return candidate
    return None


def train_worker(dataset_dir: Path, working_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    model_path = os.environ["LOCAL_GEMMA_MODEL_PATH"]
    examples = (load_jsonl(dataset_dir / "train.jsonl") + load_jsonl(dataset_dir / "validation.jsonl"))[:MAX_TRAIN_EXAMPLES]
    texts = [training_text(ex) for ex in examples]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=int(os.getenv("SIA_FINCHECK_MAX_SEQ_LEN", "4096")))

    ds = Dataset.from_dict({"text": texts}).map(tok, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=os.getenv("SIA_ATTENTION_IMPL", "sdpa"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    prev_adapter = previous_adapter_dir(working_dir)
    if prev_adapter is not None:
        print(f"Continuing cumulative training from previous adapter: {prev_adapter}")
        model = PeftModel.from_pretrained(model, str(prev_adapter), is_trainable=True)
    else:
        print("No previous generation adapter found; starting a fresh LoRA adapter from base Gemma.")
        lora = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    out = working_dir / "training_output"
    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=float(os.getenv("SIA_FINCHECK_LR", "2e-4")),
        bf16=True,
        logging_steps=5,
        save_steps=MAX_STEPS,
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        adapter_dir = working_dir / "checkpoints" / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        (working_dir / "training_metrics.json").write_text(
            json.dumps(
                {
                    "max_steps": MAX_STEPS,
                    "n_examples": len(examples),
                    "base_model": model_path,
                    "continued_from_previous_adapter": str(previous_adapter_dir(working_dir)) if previous_adapter_dir(working_dir) else None,
                    "saved_adapter": str(adapter_dir),
                },
                indent=2,
            )
        )


def parse_model_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"answer": None, "unit": "", "confidence": 0.0, "reasoning": text[:500]}


def infer(dataset_dir: Path, working_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.getenv("LOCAL_GEMMA_MODEL_PATH")
    exec_dir = working_dir / "agent_execution"
    exec_dir.mkdir(parents=True, exist_ok=True)
    examples = load_jsonl(dataset_dir / "test.jsonl")
    predictions: list[dict[str, Any]] = []
    if not model_path:
        for i, ex in enumerate(examples):
            pred = {"id": ex.get("id"), "answer": 0, "unit": ex.get("expected_unit", ""), "confidence": 0, "reasoning": "LOCAL_GEMMA_MODEL_PATH missing"}
            predictions.append(pred)
            (exec_dir / f"execution_q{i}.json").write_text(json.dumps([{"role": "assistant", "content": pred}], indent=2), encoding="utf-8")
        write_jsonl(working_dir / "submission.jsonl", predictions)
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=os.getenv("SIA_ATTENTION_IMPL", "sdpa"),
    )
    adapter_dir = working_dir / "checkpoints" / "adapter"
    adapter_to_load = adapter_dir if (adapter_dir / "adapter_config.json").is_file() else previous_adapter_dir(working_dir)
    if adapter_to_load is not None:
        print(f"Loading tuned adapter for local inference: {adapter_to_load}")
        model = PeftModel.from_pretrained(model, str(adapter_to_load))
    else:
        print("No tuned adapter found; local inference will use the base checkpoint only.")
    model.eval()

    for i, ex in enumerate(examples):
        prompt = make_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(os.getenv("SIA_FINCHECK_MAX_SEQ_LEN", "4096"))).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_model_json(text)
        pred = {
            "id": ex.get("id"),
            "answer": parsed.get("answer"),
            "unit": parsed.get("unit") or ex.get("expected_unit", ""),
            "confidence": parsed.get("confidence", 0.5),
            "reasoning": parsed.get("reasoning", ""),
        }
        predictions.append(pred)
        (exec_dir / f"execution_q{i}.json").write_text(json.dumps([{"role": "user", "content": prompt[:8000]}, {"role": "assistant", "content": text}], indent=2), encoding="utf-8")
        print(f"[{i + 1}/{len(examples)}] {ex.get('id')} -> {pred.get('answer')} {pred.get('unit')}")
    write_jsonl(working_dir / "submission.jsonl", predictions)
    (working_dir / "summary.json").write_text(json.dumps({"n_predictions": len(predictions), "adapter": str(adapter_dir)}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--train-worker", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    if args.train_worker:
        train_worker(dataset_dir, working_dir)
    else:
        run_accelerate_training(dataset_dir, working_dir)
        infer(dataset_dir, working_dir)


if __name__ == "__main__":
    main()
