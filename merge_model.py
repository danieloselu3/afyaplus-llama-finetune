# merge_model.py
# Runs on the vast.ai instance, straight after training. Folds the trained LoRA
# adapter into the full-precision base model, producing one standard model
# directory with no PEFT overhead at inference time.
#
# Why on the GPU box rather than a laptop: the base model alone is ~16GB in
# 16-bit, and loading plus merging needs more RAM than a 16GB laptop has. The
# instance also already has the base weights cached from training.
import os
import time

import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ADAPTER_DIR, BASE_MODEL, MERGED_DIR

login(token=os.environ["HF_TOKEN"])

if not os.path.exists(f"{ADAPTER_DIR}/adapter_model.safetensors"):
    raise SystemExit(f"No adapter found in {ADAPTER_DIR}/. Run fine_tune.py first.")

# STEP 1: Load the base model in 16-bit, NOT 4-bit.
# Merging into quantised weights bakes rounding error into the result; a
# one-time 16-bit merge keeps full fidelity. bfloat16 is LLaMA 3.1's native
# dtype and matches the training compute dtype (fp16 on pre-Ampere cards).
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
started = time.time()
print(f"Loading base model in {DTYPE} for merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    device_map="auto",          # 16GB fits on a 24GB card; spills to CPU RAM otherwise
)

# STEP 2: Layer the LoRA adapter on top of the base model
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# STEP 3: Merge and unload.
# For every adapted layer: W_merged = W_original + (B x A) x (alpha / r).
# The adapter matrices are then discarded.
print("Merging adapter weights into base model...")
model = model.merge_and_unload()

# STEP 4: Save the merged model with the training tokeniser (carries the
# <|finetune_right_pad_id|> padding setting), so the folder is self-contained.
print(f"Saving merged model to {MERGED_DIR}/ ...")
model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(ADAPTER_DIR).save_pretrained(MERGED_DIR)

print(f"Merge complete in {(time.time() - started) / 60:.1f} min.")
print(f"Merged model saved to: {MERGED_DIR}/ (~16GB; not committed to git)")
