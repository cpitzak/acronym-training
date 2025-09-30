# scripts/eval.py
# Simple pass@1 exact-match evaluator for acronym expansions.
import argparse, json, os, re
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def load_pairs(path: str):
    gold = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]
            # expect pair [user, assistant]
            gold.append((msgs[0]["content"], msgs[1]["content"]))
    return gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    ap.add_argument("--file", default="data/acronyms.eval.jsonl")
    ap.add_argument("--model_name", default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name, max_seq_length=args.max_seq_len, load_in_4bit=True
    )
    model.load_adapter(args.adapter)

    pairs = load_pairs(args.file)
    correct = 0
    for (user, target) in pairs:
        messages = [{"role":"user","content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # naive: look for exact normalized equality of target tail
        if normalize(target) in normalize(text):
            correct += 1
    total = len(pairs)
    print(f"Exact/contains match: {correct}/{total} = {correct/total if total else 0:.2f}")

if __name__ == "__main__":
    main()
