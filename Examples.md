python scripts/infer.py \
  --model_name unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --adapter outputs/llama31-8b-acronyms/checkpoint \
  --prompt "What does SJL stand for?"

# -----------------------------------------------------------------

# train:
python scripts/train.py --config config.gemma-2-9b-bnb-4bit.yaml

# inference:
python scripts/infer.py \
  --model_name unsloth/gemma-2-9b-bnb-4bit \
  --adapter outputs/gemma2-9b-acronyms/checkpoint \
  --prompt "What does SJL stand for?"
  
 # Example run:
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

# -----------------------------------------------------------------
# if wanting to provide one folder with the model plus adapter:
python scripts/merge_adapter.py \
  --base unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --adapter outputs/llama31-8b-acronyms/checkpoint \
  --out outputs/llama31-8b-merged-fp16

# then user does:
pip install "transformers==4.56.2" "accelerate==0.34.3" torch
python scripts/infer.py --model_name ./llama31-8b-merged-fp16 --prompt "What does SJL stand for?"
# -----------------------------------------------------------------