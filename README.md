# Acronym SFT with QLoRA (Unsloth + Llama 3.1 8B 4-bit)

This repo fine-tunes **unsloth/Meta-Llama-3.1-8B-bnb-4bit** using **QLoRA** to teach your organization-specific acronyms.

- Framework: Unsloth + Transformers/TRL (+ PEFT)
- Target GPU: single RTX 5070 Ti (Blackwell, CUDA 12.8 wheels)
- Task: Supervised fine-tuning (SFT) on chat-style records that map acronyms to expansions and example usage.

## 0) Environment (Linux/Windows)
> Ensure you have the latest NVIDIA driver and PyTorch **cu128** wheels (or newer) working first.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# Install PyTorch cu128 (or the nightly cu128 if needed for Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# Optional (if kernels complain on sm_120): a newer Triton
pip uninstall -y triton && pip install -U --pre triton --index-url https://pypi.nvidia.com/
# Project deps
pip install -r requirements.txt
```

## 1) Put your data
Edit `data/acronyms.train.jsonl` and `data/acronyms.eval.jsonl`.
Each line is a JSON object with a `messages` array in Llama-3.1 chat format:
```json
{"messages":[{"role":"user","content":"What does GPU mean?"},
             {"role":"assistant","content":"GPU = Graphics Processing Unit."}]}
```

## 2) Configure
Edit `config.yaml` to adjust LoRA rank, LR, seq length, batch size, etc.

## 3) Train
```bash
python scripts/train.py --config config.yaml
```
Artifacts go to `./outputs/…` including the LoRA adapter. To merge adapters into a full FP16 model, see the script notes.

## 4) Quick inference test (loads base + adapter)
```bash
python scripts/infer.py --adapter outputs/last/checkpoint
```

## 5) Simple eval
```bash
python scripts/eval.py --adapter outputs/last/checkpoint
```

---

### Data design tips
- For ambiguous acronyms, include **context** in the user message and show the correct expansion in the assistant reply.
- Include **negative** examples where the acronym should not be expanded if already expanded or irrelevant.
- Keep answers **short and deterministic** for glossary behavior.


python scripts/infer.py \
  --model_name unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --adapter outputs/llama31-8b-acronyms/checkpoint \
  --prompt "What does SJL stand for?"

# -----------------------------------------------------------------

# Always watch VRAM usage:
```
watch -n 1 nvidia-smi
```

# Train:
```
python scripts/train.py --config config.gemma-2-9b-bnb-4bit.yaml
```

# Inference:
```
python scripts/infer.py \
  --model_name unsloth/gemma-2-9b-bnb-4bit \
  --adapter outputs/gemma2-9b-acronyms/checkpoint \
  --prompt "What does SJL stand for?"
 ```

 python scripts/infer.py \
  --model_name unsloth/gemma-2-9b-bnb-4bit \
  --adapter outputs/gemma2-9b-acronyms/checkpoint \
  --prompt "What does WTH stand for?"

 # Example run:
  ```
  python scripts/infer.py \
  --model_name unsloth/gemma-2-9b-bnb-4bit \
  --adapter outputs/gemma2-9b-acronyms/checkpoint \
  --prompt "What does SJL stand for?"
/home/clint/anaconda3/envs/acronym-training/lib/python3.10/site-packages/transformers/quantizers/auto.py:239: UserWarning: You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading already has a `quantization_config` attribute. The `quantization_config` from the model will be used.
  warnings.warn(warning_msg)
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

[BASELINE]
User: What does SJL stand for?
Assistant: I'm sorry, but I don't understand. Could you please rephrase your question?

[WITH ADAPTER]
User: What does SJL stand for?
Assistant: SJL = Strawberry Jam Lemonade.

[DEBUG] LoRA tensors: 588 | in_len: 13 | out_len: 21
```

# -----------------------------------------------------------------
# if wanting to provide one folder with the model plus adapter:
```
python scripts/merge_adapter.py \
  --base unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --adapter outputs/llama31-8b-acronyms/checkpoint \
  --out outputs/llama31-8b-merged-fp16
```

# then user does:
```
pip install "transformers==4.56.2" "accelerate==0.34.3" torch
python scripts/infer.py --model_name ./llama31-8b-merged-fp16 --prompt "What does SJL stand for?"
```
# -----------------------------------------------------------------