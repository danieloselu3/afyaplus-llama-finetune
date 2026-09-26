# monitor_training_loss.py
# Parses trainer_state.json, prints the training and validation progressions,
# issues a healthy / overfit / underfit / unstable diagnosis (Week 4 Lab 3
# patterns), and plots the loss curve to outputs/loss_curve.png.
# Runs anywhere: needs only pandas and matplotlib.
#   python monitor_training_loss.py                      # uses ./trainer_state.json
#   python monitor_training_loss.py path/to/trainer_state.json
import json
import os
import sys

import pandas as pd

from config import OUTPUT_DIR

STATE_FILE = sys.argv[1] if len(sys.argv) > 1 else "trainer_state.json"
CLINICAL_THRESHOLD = 0.3   # Max acceptable overfitting gap for a healthcare deployment
UNDERFIT_MIN_DROP = 0.10   # Validation loss must fall at least 10% from its first reading
SPIKE_FACTOR = 1.5         # A spike is a jump of 50%+ over the previous point...
SPIKE_MIN_ABS = 0.10       # ...that is also larger than 10% of the starting loss. Without this,
                           # batch noise near zero (e.g. 0.06 -> 0.26 on 32 examples) reads as a spike.

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- LOAD TRAINING HISTORY -----------------------
with open(STATE_FILE) as f:
    state = json.load(f)
history = state["log_history"]

train_steps = [h for h in history if "loss" in h and "eval_loss" not in h]
eval_steps = [h for h in history if "eval_loss" in h]
if not train_steps or not eval_steps:
    raise SystemExit("No training or evaluation rows found. Was eval_strategy='steps' set?")

train_df = pd.DataFrame(train_steps)[["step", "loss", "epoch"]].rename(columns={"loss": "train_loss"})
eval_df = pd.DataFrame(eval_steps)[["step", "eval_loss", "epoch"]]

print("=== TRAINING LOSS PROGRESSION ===")
print(train_df.to_string(index=False))
print("\n=== VALIDATION LOSS PROGRESSION ===")
print(eval_df.to_string(index=False))

# ---------------------------- INTERPRET THE CURVE ------------------------
first_train = train_df["train_loss"].iloc[0]
final_train = train_df["train_loss"].iloc[-1]
first_eval = eval_df["eval_loss"].iloc[0]
final_eval = eval_df["eval_loss"].iloc[-1]
best_eval = eval_df["eval_loss"].min()
best_step = int(eval_df.loc[eval_df["eval_loss"].idxmin(), "step"])
gap = final_eval - final_train
eval_drop = (first_eval - best_eval) / first_eval
first_train_loss = train_df["train_loss"].iloc[0]
ratios = train_df["train_loss"] / train_df["train_loss"].shift(1)
jumps = train_df["train_loss"] - train_df["train_loss"].shift(1)
spikes = train_df.loc[(ratios > SPIKE_FACTOR) & (jumps > SPIKE_MIN_ABS * first_train_loss), "step"].tolist()

print("\n=== TRAINING SUMMARY ===")
print(f"Training loss:         {first_train:.4f} -> {final_train:.4f}")
print(f"Validation loss:       {first_eval:.4f} -> {final_eval:.4f}")
print(f"Best validation loss:  {best_eval:.4f}  (at step {best_step})")
print(f"Validation improvement:{eval_drop:7.1%}")
print(f"Overfitting gap:       {gap:.4f}  (final val - final train)")

# ------------------------------ DIAGNOSE THE RUN -------------------------
if spikes:
    verdict = "unstable"
    print(f"\nDIAGNOSIS: Loss spikes at step(s) {spikes}. Training was unstable.")
    print("ACTION: Halve the learning rate and look for outlier examples in the training data.")
elif final_eval > first_eval:
    verdict = "overfit"
    print("\nDIAGNOSIS: Validation loss increased overall. The model has overfit.")
    print(f"ACTION: Use the checkpoint at step {best_step} (load_best_model_at_end already restores it).")
elif eval_drop < UNDERFIT_MIN_DROP:
    verdict = "underfit"
    print(f"\nDIAGNOSIS: Validation loss fell only {eval_drop:.1%}. The model underfit.")
    print("ACTION: Raise the learning rate or rank r, or train longer; re-check the chat template.")
elif final_eval > best_eval + 0.05:
    verdict = "overfit-late"
    print(f"\nDIAGNOSIS: Validation loss bottomed out at step {best_step} and then rose "
          f"({best_eval:.4f} -> {final_eval:.4f}). Overfitting began late in training.")
    print(f"ACTION: The saved adapter is the step-{best_step} checkpoint (load_best_model_at_end). "
          "For a rerun, reduce epochs or raise lora_dropout.")
elif gap > CLINICAL_THRESHOLD:
    verdict = "clinical-caution"
    print(f"\n[CLINICAL CAUTION] Overfitting gap {gap:.4f} exceeds the safety threshold "
          f"of {CLINICAL_THRESHOLD}.")
    print("ACTION: Review before deploying; consider higher lora_dropout or fewer epochs.")
else:
    verdict = "healthy"
    print("\nDIAGNOSIS: Training looks healthy. Both curves fell together and the gap is "
          "within the clinical safety limit.")
    print("ACTION: Proceed to merge and evaluation.")

with open(f"{OUTPUT_DIR}/loss_diagnosis.json", "w") as f:
    json.dump({
        "verdict": verdict, "first_train_loss": first_train, "final_train_loss": final_train,
        "first_eval_loss": first_eval, "final_eval_loss": final_eval, "best_eval_loss": best_eval,
        "best_step": best_step, "overfitting_gap": round(gap, 4),
        "validation_improvement_pct": round(100 * eval_drop, 1), "spike_steps": spikes,
        "clinical_threshold": CLINICAL_THRESHOLD,
    }, f, indent=2)

# ---------------------------------- PLOT ---------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_C, VAL_C = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(train_df["step"], train_df["train_loss"], color=TRAIN_C, lw=2,
        marker="o", ms=4, label="Training loss")
ax.plot(eval_df["step"], eval_df["eval_loss"], color=VAL_C, lw=2,
        marker="s", ms=6, label="Validation loss")
ax.axvline(best_step, color=MUTED, lw=1, ls="--")
ax.annotate(f"best checkpoint\nstep {best_step}, val {best_eval:.3f}",
            xy=(best_step, best_eval), xytext=(8, 24), textcoords="offset points",
            fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

# Direct labels at the line ends; the legend carries identity as well
ax.annotate(f"train {final_train:.3f}", xy=(train_df["step"].iloc[-1], final_train),
            xytext=(6, -4), textcoords="offset points", fontsize=9, color=INK)
ax.annotate(f"val {final_eval:.3f}", xy=(eval_df["step"].iloc[-1], final_eval),
            xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK)

# Epoch boundaries as a secondary reference
steps_per_epoch = state.get("global_step", 0) / max(state.get("epoch", 1), 1)
if steps_per_epoch:
    for e in range(1, int(round(state.get("epoch", 0)))):
        ax.axvline(e * steps_per_epoch, color=GRID, lw=1, zorder=0)
        ax.text(e * steps_per_epoch, ax.get_ylim()[1], f" epoch {e}", va="top",
                fontsize=8, color=MUTED)

ax.set_title(f"AfyaPlus QLoRA fine-tune: loss curve ({verdict})", loc="left",
             fontsize=12, color=INK, pad=30)
ax.set_xlabel("Optimiser step", color=MUTED)
ax.set_ylabel("Cross-entropy loss (answer tokens)", color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.tick_params(colors=MUTED)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK)
ax.margins(x=0.08)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/loss_curve.png", facecolor=SURFACE)
print(f"\nLoss curve saved to {OUTPUT_DIR}/loss_curve.png; diagnosis in {OUTPUT_DIR}/loss_diagnosis.json")
