# Gemma fake-RL-as-harness reference

This reference is intentionally **local checkpoint only**.

The SIA meta/feedback agent should preserve this behavior:

- Train Gemma locally from `LOCAL_GEMMA_MODEL_PATH`.
- Save the PEFT/LoRA adapter to `<working_dir>/checkpoints/adapter`.
- Run inference by loading `LOCAL_GEMMA_MODEL_PATH` plus that adapter with `PeftModel.from_pretrained`.
- Continue from `<previous_generation>/checkpoints/adapter` when present.
- Do **not** call an OpenAI-compatible endpoint, vLLM endpoint, `LOCAL_GEMMA_BASE_URL`, or `http://localhost:8000/v1`.

The target profile intentionally uses a non-OpenAI provider (`local-gemma-trainable`) so SIA does not inject the OpenAI client refactor instructions into the meta prompt.
