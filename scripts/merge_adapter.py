# scripts/merge_adapter.py
# Purpose: Merge a PEFT LoRA adapter into its base model and save as one folder.

import argparse, os, torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model id (e.g. unsloth/Meta-Llama-3.1-8B-bnb-4bit)")
    p.add_argument("--adapter", required=True, help="Path to the adapter checkpoint folder (with adapter_config.json)")
    p.add_argument("--out", required=True, help="Where to save merged model")
    args = p.parse_args()

    print(f"Loading base {args.base} with adapter {args.adapter}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Merging adapter into base weights …")
    merged = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tok.save_pretrained(args.out)
    print(f"✅ Merged model saved at {args.out}")

if __name__ == "__main__":
    main()
