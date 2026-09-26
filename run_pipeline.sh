#!/usr/bin/env bash
# run_pipeline.sh - the whole GPU side of the capstone, in order, on vast.ai.
# Every stage logs to outputs/ and the results are bundled at the end into
# results_bundle.tar.gz (small: no model weights except the ~30MB adapter).
#
#   export HF_TOKEN=hf_...
#   export VAST_PRICE_PER_HOUR=0.35        # the $/hr shown on your instance card
#   bash run_pipeline.sh
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
mkdir -p outputs
PIPELINE_START=$(date +%s)

stage () { echo; echo "================ $1 ================"; }

stage "0/7 Environment"
nvidia-smi | tee outputs/nvidia_smi.txt
python -c "import torch, transformers, trl, peft, bitsandbytes as bnb; print('torch', torch.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '| bitsandbytes', bnb.__version__)" | tee outputs/versions.txt

stage "1/7 Data prep (exact LLaMA token counts)"
python data_prep.py 2>&1 | tee outputs/data_prep.log

stage "2/7 Fine-tune (QLoRA)"
python fine_tune.py 2>&1 | tee outputs/fine_tune.log

stage "3/7 Loss-curve diagnosis"
python monitor_training_loss.py 2>&1 | tee outputs/monitor_training_loss.log

stage "4/7 Merge adapter into base model"
python merge_model.py 2>&1 | tee outputs/merge_model.log

stage "5/7 Verification gate"
python verify_merge.py 2>&1 | tee outputs/verify_merge.log || echo "[WARN] Verification flagged items for review - continuing so the results can be inspected."

stage "6/7 Sample inference"
python local_inference.py 2>&1 | tee outputs/local_inference.log

stage "7/7 Evaluation: generate base + fine-tuned answers"
python evaluate_models.py generate 2>&1 | tee outputs/evaluate_generate.log

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  stage "Scoring with Claude judge"
  python evaluate_models.py score 2>&1 | tee outputs/evaluate_score.log
else
  echo "[INFO] ANTHROPIC_API_KEY not set: run 'python evaluate_models.py score' locally after download."
fi

PIPELINE_END=$(date +%s)
python - <<PY
import json
p = "outputs/run_metadata.json"
m = json.load(open(p))
m["pipeline_minutes"] = round(($PIPELINE_END - $PIPELINE_START) / 60, 1)
price = m.get("vast_price_per_hour_usd")
m["pipeline_cost_usd"] = round(price * m["pipeline_minutes"] / 60, 3) if price else None
json.dump(m, open(p, "w"), indent=2)
print("Pipeline minutes:", m["pipeline_minutes"], "| pipeline cost USD:", m["pipeline_cost_usd"])
PY

stage "Bundling results"
tar czf results_bundle.tar.gz --exclude='afyaplus-llama-adapter/checkpoint-*' \
    outputs trainer_state.json data/validation_report.json afyaplus-llama-adapter \
    $( [ -f comparison_results.csv ] && echo comparison_results.csv )
ls -lh results_bundle.tar.gz
echo "Done. Download results_bundle.tar.gz, then DESTROY the instance."
