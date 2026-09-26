# AfyaPlus Operational Assistant: a fine-tuned LLaMA 3 8B

## Overview

A fine-tuned LLaMA 3 8B Instruct (`meta-llama/Meta-Llama-3-8B-Instruct`) assistant for **AfyaPlus**, a community health platform in peri-urban Kenya. It answers staff and patient questions about operational workflows (booking, referrals, registration, records transfer, billing, dispensary stock, lab results, system access and triage escalation) in AfyaPlus's own procedures and voice. It is strictly non-diagnostic: it sends every clinical question to an AfyaPlus provider.

The model was trained with QLoRA (a 4-bit base model plus a rank-16 LoRA adapter) on a single RTX 4090 rented from vast.ai. It was then merged into a standalone model and evaluated against the untuned base model on 22 held-out questions using ROUGE-L, token F1 and a Claude judge (overall quality, accuracy, groundedness and safety).

| Deliverable | Where |
|---|---|
| 1. Dataset | `data/` · `data_prep.py` · `data/validation_report.json` · `docs/curation_note.md` |
| 2. Training run | `fine_tune.py` · `trainer_state.json` · `outputs/loss_curve.png` · `docs/training_report.md` |
| 3. Merge and inference | `merge_model.py` · `local_inference.py` · `verify_merge.py` · `outputs/inference_samples.md` · `outputs/verification_report.json` |
| 4. Evaluation | `evaluate_models.py` · `comparison_results.csv` · `docs/evaluation_report.md` |
| 5. Stakeholder memo | `memo.md` |

## Project Structure

- `data/`: `train.jsonl` (176), `val.jsonl` (22), `test.jsonl` (22), `validation_report.json`
- `data/raw/operational_data.json`: the curated AfyaPlus dataset (200 records)
- `data/raw/safety_refusals.json`: 20 clinical-redirect records that teach the assistant to decline clinical questions
- `data_prep.py`: disclaimer normalisation, formatting, validation, and a leakage-aware 80/10/10 split
- `config.py`: shared system prompt, mandatory disclaimer and scope guardrail
- `fine_tune.py`: QLoRA training script (runs on vast.ai)
- `monitor_training_loss.py`: loss-curve diagnosis and plot
- `merge_model.py`: adapter merge
- `local_inference.py`: inference with the scope filter and disclaimer guardrail
- `verify_merge.py`: verification gate
- `evaluator.py`, `llm_judge.py`: ROUGE-L / token F1 and the Claude judge (rebuilt Week 3 toolkit)
- `evaluate_models.py`: base vs fine-tuned evaluation
- `comparison_results.csv`: per-question evaluation metrics
- `memo.md`: stakeholder recommendation
- `run_pipeline.sh`, `RUNBOOK_VASTAI.md`: the GPU side, end to end
- `outputs/`: logs, loss curve, verification and inference samples from the GPU run (`outputs/run1/` holds the first run)
- Model weights are not committed: the LoRA adapter (~170MB) exceeds GitHub's file limit, and the merged model is 16GB. `run_pipeline.sh` regenerates both.

## How to Reproduce

1. **Environment setup.** GPU instance: `pip install -r requirements.txt`. Laptop: `pip install -r requirements-local.txt`. Copy `.env.example` to `.env` and fill in `HF_TOKEN` and `ANTHROPIC_API_KEY`.
2. **Data.** The curated dataset is already at `data/raw/operational_data.json`. Run `python data_prep.py`.
3. **Fine-tuning.** On a vast.ai 24GB GPU instance, inside tmux: `python fine_tune.py`.
4. **Merging.** On the same instance: `python merge_model.py` (it reads the adapter that `fine_tune.py` wrote to `afyaplus-llama-adapter/`). Then run `python verify_merge.py` and `python local_inference.py`.
5. **Evaluation.** On the instance: `python evaluate_models.py generate`. Anywhere with an Anthropic key: `python evaluate_models.py score`. This writes `comparison_results.csv`.

Steps 2-5 on the GPU run as one command, `bash run_pipeline.sh`. See [RUNBOOK_VASTAI.md](RUNBOOK_VASTAI.md) for instance selection, cost control and retrieving the results.

## Disclaimer

This model gives non-diagnostic operational guidance only: booking, records, billing, system access and escalation procedures. It does not diagnose, recommend treatment or give dosage advice. Clinical questions must go to an AfyaPlus provider. Answers must be checked against the current AfyaPlus SOP handbook before anyone relies on them.
