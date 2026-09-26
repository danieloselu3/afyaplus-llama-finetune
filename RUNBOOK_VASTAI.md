# vast.ai runbook

Every GPU step for this project, from renting the instance to destroying it. Expect about 30-45 minutes of instance time and well under $1.

## 1. Before you start (on your laptop)

- Hugging Face access to `meta-llama/Meta-Llama-3.1-8B-Instruct` is approved, and you have a read token (`hf_...`).
- This repo is pushed to GitHub so the instance can `git clone` it. A **private** repo needs credentials on the instance, so either make it public or upload the folder with `scp` instead (see step 3).
- You have at least $2 of vast.ai credit.

## 2. Rent the instance

In the vast.ai console, go to **Templates** and pick **PyTorch (Vast)**. Any official PyTorch image with CUDA 12.x works. Then filter the offers:

| Setting | Value |
|---|---|
| GPU | 1 x **RTX 4090** (first choice) or **RTX 3090 / RTX A5000** (24GB). Avoid V100s (no bf16) and anything under 24GB. |
| Disk | **100 GB**: base model 16GB + merged model 16GB + checkpoints + pip cache. |
| Reliability | 98% or higher. |
| Internet download | 500 Mbps or faster (the 16GB base model downloads twice as fast). |

Note the **$/hr** on the offer card; you need it for the cost figure. Rent the instance, and once it shows *Running*, connect with the SSH command from the instance card.

## 3. Set up (inside the instance)

```bash
# vast.ai SSH sessions usually open inside tmux already; if not:
tmux new -s capstone

cd /workspace 2>/dev/null || cd ~
git clone https://github.com/danieloselu3/afyaplus-llama-finetune.git afyaplus-llama-finetune
# (or from your laptop: scp -P <port> -r afyaplus-llama-finetune root@<host>:/workspace/)
cd afyaplus-llama-finetune

pip install -r requirements.txt

export HF_TOKEN=hf_your_token_here
export VAST_PRICE_PER_HOUR=0.35        # <- the $/hr from your offer card
```

## 4. Run everything

```bash
bash run_pipeline.sh
```

The pipeline runs these stages in order, and each one logs to `outputs/`:

1. Environment check: `nvidia-smi` and library versions
2. `data_prep.py`: regenerates the same splits with exact LLaMA token counts
3. `fine_tune.py`: QLoRA training, about 5-10 minutes after the model download
4. `monitor_training_loss.py`: prints the diagnosis and saves `outputs/loss_curve.png`
5. `merge_model.py`: adapter + base model become `afyaplus-llama-merged/`
6. `verify_merge.py`: the verification gate
7. `local_inference.py`: 8 sample answers and the stability test
8. `evaluate_models.py generate`: base and fine-tuned models each answer the 20 test questions

If the browser or SSH drops, reconnect and run `tmux attach`.

**Scoring stays on your laptop.** The pipeline skips the Claude judge unless `ANTHROPIC_API_KEY` is set. That's intentional: your API key never goes onto a rented machine, and the GPU clock stops sooner.

## 5. Download the results and destroy the instance

The pipeline ends by writing `results_bundle.tar.gz` (about 30MB). From your **laptop**, using the port and host from the instance card:

```bash
scp -P <port> root@<host>:/workspace/afyaplus-llama-finetune/results_bundle.tar.gz .
```

Then **destroy** the instance in the vast.ai console. Stopping it still bills for disk. The merged model is not needed afterwards: it can be rebuilt from the adapter in the bundle.

## 6. What to send back

Send `results_bundle.tar.gz`, or extract it into the repo folder and say so. Include:

- The full terminal output from `run_pipeline.sh`, if anything failed. The per-stage logs in `outputs/*.log` also cover this.
- The GPU model and $/hr, if `VAST_PRICE_PER_HOUR` wasn't set.

The bundle contains: `trainer_state.json`, `outputs/` (all logs, `run_metadata.json`, `loss_curve.png`, `loss_diagnosis.json`, `verification_report.json`, `inference_samples.md/.json`, `responses.jsonl`), `data/validation_report.json`, and the LoRA adapter.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `GatedRepoError` downloading LLaMA | `HF_TOKEN` is not exported, or the licence is not accepted on that HF account. |
| `CUDA out of memory` in training | In `fine_tune.py` set `BATCH_SIZE = 2` and `GRAD_ACCUM = 8`, then rerun `bash run_pipeline.sh`. |
| `No space left on device` | The disk is under 100GB. Rent a new instance with more disk; resizing is not supported. |
| `libbitsandbytes_cuda1xx.so` not found / `No module named 'triton.ops'` | An old bitsandbytes for the image's CUDA. Run `git pull && pip install -r requirements.txt` (pins bitsandbytes 0.49.2). |
| `no kernel image is available` | torch has no kernels for this GPU. Send `nvidia-smi` and the versions line from stage 0. |
| Loss is `nan` | Set `LEARNING_RATE = 1e-4` and rerun. |
| Verification prints `REVIEW REQUIRED` | The pipeline continues anyway. Send the results and we'll diagnose from `verification_report.json`. |
