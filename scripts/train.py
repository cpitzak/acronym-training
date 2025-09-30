# scripts/train.py
# Fine-tune a 4-bit Llama 3.1 with QLoRA (PEFT) for acronym expansion.

import argparse, os, json, warnings, yaml
from typing import Dict, List, Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Compatibility shim for Accelerate unwrap_model
try:
    from accelerate import Accelerator
    _orig_unwrap = Accelerator.unwrap_model
    def _unwrap_compat(self, model, *args, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return _orig_unwrap(self, model, *args, **kwargs)
    Accelerator.unwrap_model = _unwrap_compat
except Exception:
    pass


warnings.filterwarnings("ignore", category=UserWarning)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_dataset(train_file: str, eval_file: str) -> Dict[str, Dataset]:
    train_rows = read_jsonl(train_file)
    eval_rows = read_jsonl(eval_file) if os.path.exists(eval_file) else []
    return {
        "train": Dataset.from_list(train_rows),
        "eval": Dataset.from_list(eval_rows) if eval_rows else None,
    }

# Simple, template-free format; we’ll mask so only Assistant is trained.
def format_chat(tokenizer, example):
    user, assistant = None, None
    for m in example.get("messages", []):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user" and user is None:
            user = content
        elif role == "assistant" and assistant is None:
            assistant = content
    if not user or not assistant:
        return {"text": None}
    eos = tokenizer.eos_token or "</s>"
    return {"text": f"User: {user}\nAssistant: {assistant}{eos}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model_name"]

    # --- 4-bit base load (QLoRA) ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- Prep for k-bit training + apply LoRA (PEFT) ---
    model = prepare_model_for_kbit_training(model)
    l = cfg["lora"]
    peft_cfg = LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=l["target_modules"],  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        use_rslora=l.get("use_rslora", True),
    )
    model = get_peft_model(model, peft_cfg)
    # Optional: gradient checkpointing for longer context and lower VRAM
    model.gradient_checkpointing_enable()

    # --- Datasets ---
    ds = build_dataset(cfg["train"]["train_file"], cfg["train"]["eval_file"])
    train_ds = ds["train"].map(lambda ex: format_chat(tokenizer, ex))
    train_ds = train_ds.filter(lambda ex: ex["text"] is not None)
    eval_ds = None
    if ds["eval"] is not None:
        eval_ds = ds["eval"].map(lambda ex: format_chat(tokenizer, ex))
        eval_ds = eval_ds.filter(lambda ex: ex["text"] is not None)

    # Train only on Assistant span
    collator = DataCollatorForCompletionOnlyLM(
        response_template="Assistant:",
        instruction_template="User:",
        tokenizer=tokenizer,
    )

    t = cfg["train"]
    output_dir = t["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    base_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        bf16=t.get("bf16", True),
        fp16=False,
        optim="paged_adamw_8bit",   # good with 4-bit
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        report_to="none",
        seed=t["seed"],
    )
    with_eval_kwargs = dict(base_kwargs)
    if eval_ds is not None:
        with_eval_kwargs.update(dict(evaluation_strategy="steps", eval_steps=t["eval_steps"]))
    try:
        training_args = TrainingArguments(**with_eval_kwargs)
    except TypeError:
        training_args = TrainingArguments(**base_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        data_collator=collator,
        packing=False,
        max_seq_length=cfg["max_seq_len"],
        args=training_args,
    )

    trainer.train()

    # final eval if eval wasn’t wired
    if eval_ds is not None and "evaluation_strategy" not in training_args.to_dict():
        print("\nRunning final evaluation...")
        print(trainer.evaluate())

    ckpt_dir = os.path.join(output_dir, "checkpoint")
    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(output_dir)
    print("\nTraining complete.\nAdapter saved to:", ckpt_dir, "\nTokenizer saved to:", output_dir)

if __name__ == "__main__":
    main()
