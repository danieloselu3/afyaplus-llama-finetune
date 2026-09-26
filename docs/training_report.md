# Training report

## Setup

| Item | Value |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Method | QLoRA: 4-bit NF4 base with double quantisation, plus LoRA adapters |
| Compute | vast.ai, 1 x NVIDIA RTX 4090 (48GB variant), $0.78/hr; Python 3.12, torch 2.11 + CUDA 12.8 |
| Libraries | transformers 4.44.2, trl 0.9.6, peft 0.12.0, bitsandbytes 0.49.2 (see "Deviations from the course lab") |
| Data | Run 1: 160 train / 20 validation. Run 2: 176 train / 22 validation, adding the clinical-redirect records (`data/train.jsonl`, `data/val.jsonl`) |

## Hyperparameters and justification

| Hyperparameter | Value | Why |
|---|---|---|
| `r` (rank) | 16 | Enough capacity to learn AfyaPlus terminology and answer structure. A higher rank on 160 examples raises the risk of memorising them. |
| `lora_alpha` | 32 | Update scale alpha/r = 2, the standard 2 x rank convention. |
| `lora_dropout` | 0.05 | Light regularisation at the low end of the 0.05-0.1 range, suited to a small dataset. The first knob to raise if validation loss climbs. |
| `target_modules` | Run 1: q, k, v, o. Run 2: all linear layers (plus gate, up, down) | Run 1 adapted attention only. It learned the AfyaPlus voice but invented procedures, so run 2 also adapts the MLP layers, where factual associations mostly live, as the QLoRA paper recommends. The base weights stay frozen. |
| `learning_rate` | 2e-4 | The reliable LoRA starting point for LLaMA (course range 2e-4 to 3e-4). |
| `num_train_epochs` | Run 1: 3. Run 2: 5 | In run 1, validation loss was still falling at the last step (the best checkpoint was the final one). `load_best_model_at_end` keeps the checkpoint that generalised best, so extra epochs cannot make the saved adapter worse. |
| `per_device_train_batch_size` | 4 | Fits a 24GB card at 512 tokens in 4-bit. |
| `gradient_accumulation_steps` | 4 | Effective batch of 16 gives stable gradients without the memory of a real batch of 16. |
| `lr_scheduler_type` / `warmup_ratio` | cosine / 0.03 | A 1-step warm-up protects the freshly initialised adapter; cosine decay makes late updates small refinements. |
| `weight_decay` | 0.001 | A very light brake on adapter weight growth. |
| `optim` | `paged_adamw_8bit` | 8-bit optimiser state, paged to CPU RAM under pressure, which prevents out-of-memory spikes. |
| `max_seq_length` | 512 | The longest example is well under 512 tokens (see `data/validation_report.json`), so nothing is truncated. |
| `logging_steps` / `eval_steps` / `save_steps` | 2 / 5 / 5 | About 5 training and 2 validation points per epoch. The course's 10/10/10 gives too few points to diagnose a short run. |
| `load_best_model_at_end` | True (by `eval_loss`) | The saved adapter is always the checkpoint that generalised best. |
| Loss masking | Answer tokens only | About 40% of each example is the same system prompt. Masking it means the loss measures how well the model learned the answers, not the prompt. |
| Padding token | `<|reserved_special_token_250|>` | Padding with `eos` (as in the course lab) would also mask the real end-of-turn token, and the model would never learn to stop. LLaMA 3 has no dedicated pad token, so an unused reserved token is used; padded positions are excluded from attention and loss. |
| Precision | bf16 (fp16 fallback) | bf16 is LLaMA 3's native dtype. The course's V100 has no bf16 support, so `bf16=True` would fail there. |

## Deviations from the course lab

1. **Base model.** The run uses LLaMA 3 8B Instruct rather than the brief's LLaMA 3.1 8B Instruct. The two share the 8B architecture, tokenizer family and chat format. LLaMA 3's 8K context is far above the 512-token training examples, so nothing in the pipeline depends on 3.1's longer context.
2. **Library versions.** The vast.ai PyTorch image ships Python 3.12 and CUDA 12.8 builds of torch, so the pins moved to transformers 4.44.2, trl 0.9.6, peft 0.12.0 and bitsandbytes 0.49.2. bitsandbytes 0.43.x has no CUDA 12.8 binary.
3. **Compute.** The run used vast.ai instead of Nebius, on a 24GB Ampere/Ada card instead of a V100, so bf16 works.
4. **Loss masking, padding token and monitoring resolution** changed as described in the table above.

## Run 1: what happened and what changed

Run 1 used attention-only adapters and 3 epochs on 160 examples. Artefacts are in `outputs/run1/`.

| Measure | Value |
|---|---|
| Training time / pipeline time | 1.1 min / 8.6 min on the RTX 4090 |
| Cost at $0.78/hr | $0.014 training; $0.11 for the whole pipeline |
| Training loss | 3.52 -> 0.92 |
| Validation loss | 2.83 -> 1.31 (best at the final step, 30; never rose) |
| Overfitting gap | 0.39, above the 0.3 clinical threshold |
| Verification gate | 3 of 4 FAIL, stability PASS |

![Run 1 loss curve](../outputs/run1/loss_curve.png)

**Diagnosis:** not overfit and not unstable. Both curves fell together, and validation never turned upward. The script's clinical caution comes from the 0.39 gap. Part of that gap is by design: validation answers are rewordings the model never saw verbatim, so validation loss measures recall of procedures, not copying.

The verification gate exposed two real problems:

1. **Procedures were invented.** The model gave "Reschedule within three working days" (the SOP says 14 days) and "The administrator code is 0000". It had learned the AfyaPlus tone and disclaimer (19/20 test answers carry it) but not the facts. ROUGE-L rose from 0.12 (base) to 0.30, and answers shrank from 207 to 62 words.
2. **Unsafe clinical answer.** For a child's fever it gave "paracetamol or ibuprofen every six hours, up to the maximum dose". None of the 200 examples showed the assistant declining a clinical question. The scope filter also missed it, because it only looked for "N mg" and "take … tablets".

**Changes for run 2:**
- Adapters on all linear layers.
- 5 epochs.
- 20 clinical-redirect training examples.
- The scope filter now blocks named medicines, drug classes and dosing schedules.
- A verification safety question that is not in the training data.

## Run 2: loss curve and diagnosis

_Pending the second vast.ai run._
