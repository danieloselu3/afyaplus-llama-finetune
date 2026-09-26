# fine_tune.py
# Runs ON THE VAST.AI GPU INSTANCE (24GB card: RTX 3090 / 4090 / A5000).
# Loads LLaMA 3 8B Instruct in 4-bit (QLoRA), injects LoRA adapters into the
# attention projections, and trains on the AfyaPlus splits from data_prep.py.
#
# Library versions: see requirements.txt. transformers 4.44.2 / trl 0.9.6 /
# peft 0.12.0 run on the vast.ai image's Python 3.12 + CUDA 12.8 torch and
# provide the SFTConfig API used below.
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from config import ADAPTER_DIR, BASE_MODEL, DATA_DIR, MAX_SEQ_LEN, OUTPUT_DIR

# ============================= HYPERPARAMETERS =============================
# Every value is justified here and in docs/training_report.md.

# --- LoRA -------------------------------------------------------------------
LORA_R = 16
# Adapter capacity. 16 is the course default for behaviour/style adaptation:
# enough to learn AfyaPlus terminology and answer structure, small enough not
# to memorise 160 examples. Raise to 32 only if the run underfits.
LORA_ALPHA = 32
# Update scale = alpha / r = 2. The standard 2 x rank convention keeps the
# effective step size stable if r is changed later.
LORA_DROPOUT = 0.05
# Light regularisation on the adapter. 160 examples is a small dataset; 0.05
# is the low end of the 0.05-0.1 range. Raise to 0.1 if the run overfits.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# All linear layers (attention + MLP), as the QLoRA paper recommends. Run 1
# adapted attention only: the model learned the AfyaPlus voice and disclaimer
# but invented procedures ("rebook within three working days" instead of 14
# days). Factual associations live mostly in the MLP layers, so run 2 adapts
# them too. Still ~0.5% of parameters; the base weights stay frozen.

# --- Optimisation -----------------------------------------------------------
LEARNING_RATE = 2e-4
# The reliable LoRA-on-LLaMA starting point (course range 2e-4 to 3e-4). Halve
# it if loss oscillates; double it if loss barely moves.
NUM_EPOCHS = 5
# Run 1 (3 epochs, 30 steps) was still improving at the final step: validation
# loss 1.34 -> 1.31 over the last 5 steps, best checkpoint = last checkpoint.
# Five epochs give the loss room to bottom out; load_best_model_at_end keeps
# whichever checkpoint generalised best, so extra epochs cannot make the saved
# adapter worse on validation.
BATCH_SIZE = 4
# Examples per forward pass. Fits comfortably in 24GB at 512 tokens in 4-bit.
GRAD_ACCUM = 4
# Effective batch = 4 x 4 = 16: smoother gradients without the activation
# memory of a real batch of 16. If CUDA OOMs: BATCH_SIZE=2, GRAD_ACCUM=8.
LR_SCHEDULER = "cosine"
# Decays the learning rate smoothly so late steps make small refinements.
WARMUP_RATIO = 0.03
# ~1 warm-up step: avoids a large first update on randomly initialised B x A.
WEIGHT_DECAY = 0.001
# Very light L2 on the adapter weights; a mild extra brake on memorisation.
OPTIMIZER = "paged_adamw_8bit"
# 8-bit optimiser state paged to CPU RAM under pressure: prevents OOM spikes.

# --- Monitoring -------------------------------------------------------------
LOGGING_STEPS = 2
EVAL_STEPS = 5
SAVE_STEPS = 5
# The course used 10/10/10, which yields only a handful of points per curve on
# a short run. Logging every 2 steps and evaluating every 5 gives ~5 training
# and 2 validation points per epoch: enough resolution to diagnose the curve.
SAVE_TOTAL_LIMIT = 3
# Keep disk bounded; the best checkpoint is always retained.

# --- Loss masking -----------------------------------------------------------
COMPLETION_ONLY_LOSS = True
# Compute loss only on the assistant's answer. Without this, ~40% of every
# example is the identical system prompt, so the loss mostly measures how well
# the model memorised that prompt and hides whether it learned the answers.

SEED = 42
# ===========================================================================

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Set your token first: export HF_TOKEN=hf_your_token_here")
login(token=HF_TOKEN)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU visible. Run this on the vast.ai instance, not locally.")
USE_BF16 = torch.cuda.is_bf16_supported()      # True on Ampere/Ada (3090, 4090, A5000)
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0)} | bf16 supported: {USE_BF16}")

# ------------------- 4-BIT QUANTISATION CONFIG (QLoRA) -------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4: suited to normally distributed weights
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,       # Quantise the quantisation constants too (~0.4GB saved)
)

# --------------------------- LOAD BASE MODEL -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=COMPUTE_DTYPE,
)
model.config.use_cache = False            # KV cache is useless in training and conflicts with checkpointing
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ---------------------------- LOAD TOKENISER -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Pad with a reserved special token, NOT eos. The collator masks padding out
# of the loss; if pad == eos (<|eot_id|>), the real end-of-turn token is masked
# too and the model never learns when to stop talking. LLaMA 3.1 ships a
# dedicated <|finetune_right_pad_id|>; LLaMA 3 has only the unused
# <|reserved_special_token_N|> slots, which are safe because padded positions
# are excluded from both attention and loss.
PAD_CANDIDATES = ["<|finetune_right_pad_id|>", "<|reserved_special_token_250|>"]
tokenizer.pad_token = next(t for t in PAD_CANDIDATES if t in tokenizer.get_vocab())
print(f"Pad token: {tokenizer.pad_token} (id {tokenizer.pad_token_id})")
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

# --------------------------- LORA CONFIGURATION --------------------------
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})")
# total is understated in 4-bit (two weights per packed byte), so compare
# against a generous ceiling: a failed freeze would show tens of percent.
if trainable / total > 0.05:
    raise SystemExit("More than 5% of parameters are trainable: the LoRA freeze failed. Stopping.")

# ------------------------------ LOAD DATASET -----------------------------
dataset = load_dataset(
    "json",
    data_files={"train": f"{DATA_DIR}/train.jsonl", "validation": f"{DATA_DIR}/val.jsonl"},
)


def apply_chat_template(example):
    """Render the messages exactly as LLaMA 3 Instruct was trained to see
    them. add_generation_prompt=False: the answer is already present. The
    leading <|begin_of_text|> is stripped because the trainer's tokeniser adds
    it again, which would otherwise double it."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)
    if text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    return {"text": text}


dataset = dataset.map(apply_chat_template, remove_columns=["messages"])
print(f"Dataset prepared. Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")
print("Sample training text (first 400 chars):")
print(dataset["train"][0]["text"][:400])

collator = None
if COMPLETION_ONLY_LOSS:
    # The assistant header is three special tokens, so it tokenises identically
    # wherever it appears. Everything before it is masked out of the loss.
    response_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template=response_ids, tokenizer=tokenizer)

# --------------------------- TRAINING ARGUMENTS ---------------------------
training_args = SFTConfig(
    output_dir=ADAPTER_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    optim=OPTIMIZER,
    bf16=USE_BF16,
    fp16=not USE_BF16,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,          # Revert to the lowest-eval-loss checkpoint
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=SEED,
    # SFT-specific
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,                        # One scenario per sequence: no blurring between protocols
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
)

# ----------------------------- LAUNCH TRAINING ---------------------------
steps_per_epoch = max(1, len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM))
print("Starting training...")
print(f"Total optimiser steps: ~{steps_per_epoch * NUM_EPOCHS} | eval every {EVAL_STEPS} steps")
print("-" * 60)

started = time.time()
start_iso = datetime.now(timezone.utc).isoformat()
train_result = trainer.train()
train_seconds = time.time() - started

# ----------------------------- SAVE EVERYTHING ---------------------------
trainer.model.save_pretrained(ADAPTER_DIR)       # adapter_model.safetensors + adapter_config.json
tokenizer.save_pretrained(ADAPTER_DIR)
trainer.save_state()                              # writes ADAPTER_DIR/trainer_state.json (deliverable)
shutil.copy(f"{ADAPTER_DIR}/trainer_state.json", "trainer_state.json")

price = os.getenv("VAST_PRICE_PER_HOUR")          # e.g. export VAST_PRICE_PER_HOUR=0.35
metadata = {
    "started_utc": start_iso,
    "train_seconds": round(train_seconds, 1),
    "train_minutes": round(train_seconds / 60, 2),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "precision": "bf16" if USE_BF16 else "fp16",
    "trainable_params": trainable,
    "total_params": total,
    "global_steps": trainer.state.global_step,
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_eval_loss": trainer.state.best_metric,
    "train_loss_avg": round(train_result.training_loss, 4),
    "vast_price_per_hour_usd": float(price) if price else None,
    "training_cost_usd": round(float(price) * train_seconds / 3600, 3) if price else None,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "hyperparameters": {
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES, "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM, "scheduler": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY, "optimizer": OPTIMIZER,
        "max_seq_len": MAX_SEQ_LEN, "completion_only_loss": COMPLETION_ONLY_LOSS, "seed": SEED,
    },
}
with open(f"{OUTPUT_DIR}/run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nTraining complete in {metadata['train_minutes']} min. Adapter saved to: {ADAPTER_DIR}")
print(f"Best checkpoint: {metadata['best_checkpoint']} (eval_loss {metadata['best_eval_loss']})")
print(f"trainer_state.json copied to project root; run metadata in {OUTPUT_DIR}/run_metadata.json")
