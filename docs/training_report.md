# Training report

## Setup

| Item | Value |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Method | QLoRA: 4-bit NF4 base with double quantisation, plus LoRA adapters |
| Compute | vast.ai, 1 x _GPU (pending run)_, _$/hr (pending)_ |
| Libraries | transformers 4.44.2, trl 0.9.6, peft 0.12.0, bitsandbytes 0.49.2 (see "Deviations from the course lab") |
| Data | 160 train / 20 validation examples (`data/train.jsonl`, `data/val.jsonl`) |

## Hyperparameters and justification

| Hyperparameter | Value | Why |
|---|---|---|
| `r` (rank) | 16 | Enough capacity to learn AfyaPlus terminology and answer structure. A higher rank on 160 examples raises the risk of memorising them. |
| `lora_alpha` | 32 | Update scale alpha/r = 2, the standard 2 x rank convention. |
| `lora_dropout` | 0.05 | Light regularisation at the low end of the 0.05-0.1 range, suited to a small dataset. The first knob to raise if validation loss climbs. |
| `target_modules` | q, k, v, o projections | Attention layers control what the model attends to, which is where tone and routing live. Leaving the MLP layers frozen protects general knowledge. |
| `learning_rate` | 2e-4 | The reliable LoRA starting point for LLaMA (course range 2e-4 to 3e-4). |
| `num_train_epochs` | 3 | 30 optimiser steps: enough for the loss to settle. `load_best_model_at_end` guards against a late overfit. |
| `per_device_train_batch_size` | 4 | Fits a 24GB card at 512 tokens in 4-bit. |
| `gradient_accumulation_steps` | 4 | Effective batch of 16 gives stable gradients without the memory of a real batch of 16. |
| `lr_scheduler_type` / `warmup_ratio` | cosine / 0.03 | A 1-step warm-up protects the freshly initialised adapter; cosine decay makes late updates small refinements. |
| `weight_decay` | 0.001 | A very light brake on adapter weight growth. |
| `optim` | `paged_adamw_8bit` | 8-bit optimiser state, paged to CPU RAM under pressure, which prevents out-of-memory spikes. |
| `max_seq_length` | 512 | The longest example is well under 512 tokens (see `data/validation_report.json`), so nothing is truncated. |
| `logging_steps` / `eval_steps` / `save_steps` | 2 / 5 / 5 | Gives 15 training and 6 validation points on a 30-step run. The course's 10/10/10 gives only 3 of each, too few to diagnose a curve. |
| `load_best_model_at_end` | True (by `eval_loss`) | The saved adapter is always the checkpoint that generalised best. |
| Loss masking | Answer tokens only | About 40% of each example is the same system prompt. Masking it means the loss measures how well the model learned the answers, not the prompt. |
| Padding token | `<|reserved_special_token_250|>` | Padding with `eos` (as in the course lab) would also mask the real end-of-turn token, and the model would never learn to stop. LLaMA 3 has no dedicated pad token, so an unused reserved token is used; padded positions are excluded from attention and loss. |
| Precision | bf16 (fp16 fallback) | bf16 is LLaMA 3's native dtype. The course's V100 has no bf16 support, so `bf16=True` would fail there. |

## Deviations from the course lab

1. **Base model.** The run uses LLaMA 3 8B Instruct rather than the brief's LLaMA 3.1 8B Instruct. The two share the 8B architecture, tokenizer family and chat format. LLaMA 3's 8K context is far above the 512-token training examples, so nothing in the pipeline depends on 3.1's longer context.
2. **Library versions.** The vast.ai PyTorch image ships Python 3.12 and CUDA 12.8 builds of torch, so the pins moved to transformers 4.44.2, trl 0.9.6, peft 0.12.0 and bitsandbytes 0.49.2. bitsandbytes 0.43.x has no CUDA 12.8 binary.
3. **Compute.** The run used vast.ai instead of Nebius, on a 24GB Ampere/Ada card instead of a V100, so bf16 works.
4. **Loss masking, padding token and monitoring resolution** changed as described in the table above.

## Loss curve and diagnosis

_Pending the vast.ai run: `outputs/loss_curve.png`, `trainer_state.json`, and the diagnosis from `monitor_training_loss.py`._
