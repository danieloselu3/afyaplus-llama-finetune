# evaluate_models.py
# The fair fight: base LLaMA 3.1 8B vs the fine-tuned AfyaPlus model on the 20
# held-out test questions neither model saw during training.
#
# Two stages, because two 16GB models do not fit on one 24GB GPU together and
# because scoring needs no GPU at all:
#   python evaluate_models.py generate   # GPU: both models answer all 20, one model at a time
#   python evaluate_models.py score      # anywhere: ROUGE-L, token F1, Claude judge -> CSV
#   python evaluate_models.py            # both stages back to back
import json
import os
import sys
import time

from config import BASE_MODEL, DATA_DIR, MERGED_DIR, OUTPUT_DIR, has_disclaimer, scope_violations

TEST_FILE = f"{DATA_DIR}/test.jsonl"
RESPONSES_FILE = f"{OUTPUT_DIR}/responses.jsonl"
RESULTS_CSV = "comparison_results.csv"
SUMMARY_FILE = f"{OUTPUT_DIR}/evaluation_summary.md"
GROUNDEDNESS_FLOOR = 3   # Safety gate: fine-tuned groundedness below this is flagged


def load_test_examples() -> list:
    with open(TEST_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------- Stage 1: generation ---------------------------
def generate_all() -> None:
    """Identical prompting for both models (same system prompt, chat template,
    greedy decoding, token cap), so every difference is the model."""
    from local_inference import generate, unload

    examples = load_test_examples()
    print(f"Loaded {len(examples)} held-out test examples")
    rows = [{"id": i + 1,
             "question": ex["messages"][1]["content"],
             "reference": ex["messages"][2]["content"],
             "system_prompt": ex["messages"][0]["content"]} for i, ex in enumerate(examples)]

    for label, path in (("base", BASE_MODEL), ("ft", MERGED_DIR)):
        started = time.time()
        for r in rows:
            r[f"{label}_response"] = generate(r["question"], model_path=path,
                                              system_prompt=r["system_prompt"])
            print(f"  [{label}] Q{r['id']:>2} done")
        print(f"{label} model: {len(rows)} answers in {time.time() - started:.0f}s")
        unload(path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            r.pop("system_prompt")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} paired responses to {RESPONSES_FILE}")


# ----------------------------- Stage 2: scoring ----------------------------
def score_all() -> None:
    import pandas as pd
    from tabulate import tabulate

    from evaluator import evaluate_response
    from llm_judge import llm_judge

    with open(RESPONSES_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"Scoring {len(rows)} paired responses (Claude judge: cached calls are free)")

    results, alerts = [], []
    for r in rows:
        q, ref = r["question"], r["reference"]
        b_auto, f_auto = evaluate_response(ref, r["base_response"]), evaluate_response(ref, r["ft_response"])
        b_j, f_j = llm_judge(q, ref, r["base_response"]), llm_judge(q, ref, r["ft_response"])

        if f_j["groundedness"] is not None and f_j["groundedness"] < GROUNDEDNESS_FLOOR:
            alerts.append(r["id"])
            print(f"  [ALERT] Low groundedness ({f_j['groundedness']}/5) on Q{r['id']}: {q[:60]}...")

        results.append({
            "id": r["id"],
            "question": q,
            "base_rouge_l": b_auto["rouge_l"], "ft_rouge_l": f_auto["rouge_l"],
            "rouge_delta": round(f_auto["rouge_l"] - b_auto["rouge_l"], 4),
            "base_token_f1": b_auto["token_f1"], "ft_token_f1": f_auto["token_f1"],
            "base_judge": b_j["overall"], "ft_judge": f_j["overall"],
            "judge_delta": _delta(f_j["overall"], b_j["overall"]),
            "base_accuracy": b_j["accuracy"], "ft_accuracy": f_j["accuracy"],
            "base_ground": b_j["groundedness"], "ft_ground": f_j["groundedness"],
            "ground_delta": _delta(f_j["groundedness"], b_j["groundedness"]),
            "base_safety": b_j["safety"], "ft_safety": f_j["safety"],
            "base_disclaimer": has_disclaimer(r["base_response"]),
            "ft_disclaimer": has_disclaimer(r["ft_response"]),
            "ft_scope_violation": bool(scope_violations(r["ft_response"])),
            "base_words": len(r["base_response"].split()), "ft_words": len(r["ft_response"].split()),
            "base_judge_reason": b_j["reason"], "ft_judge_reason": f_j["reason"],
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved per-question results to {RESULTS_CSV}")
    write_summary(df, alerts, tabulate)


def _delta(a, b):
    return None if a is None or b is None else a - b


def write_summary(df, alerts: list, tabulate) -> None:
    metrics = [
        ("ROUGE-L (avg, 0-1)", "base_rouge_l", "ft_rouge_l", 3),
        ("Token F1 (avg, 0-1)", "base_token_f1", "ft_token_f1", 3),
        ("LLM judge overall (avg /5)", "base_judge", "ft_judge", 2),
        ("Accuracy vs SOP (avg /5)", "base_accuracy", "ft_accuracy", 2),
        ("Groundedness (avg /5)", "base_ground", "ft_ground", 2),
        ("Safety (avg /5)", "base_safety", "ft_safety", 2),
    ]
    table = []
    for name, b, f, nd in metrics:
        bm, fm = df[b].mean(), df[f].mean()
        rel = f"{100 * (fm - bm) / bm:+.0f}%" if bm else "n/a"
        table.append([name, round(bm, nd), round(fm, nd), round(fm - bm, nd), rel])
    table.append(["Disclaimer present (% answers)",
                  f"{100 * df['base_disclaimer'].mean():.0f}%",
                  f"{100 * df['ft_disclaimer'].mean():.0f}%",
                  f"{100 * (df['ft_disclaimer'].mean() - df['base_disclaimer'].mean()):+.0f} pts", ""])
    table.append(["Answer length (avg words)", round(df["base_words"].mean()), round(df["ft_words"].mean()),
                  round(df["ft_words"].mean() - df["base_words"].mean()), ""])
    headers = ["Metric", "Base LLaMA 3.1 8B", "Fine-tuned", "Delta", "Relative"]

    ranked = df.sort_values(["judge_delta", "rouge_delta"], ascending=False)
    cols = ["id", "question", "base_judge", "ft_judge", "judge_delta", "base_rouge_l", "ft_rouge_l",
            "rouge_delta", "base_ground", "ft_ground"]
    top3, bottom3 = ranked.head(3)[cols], ranked.tail(3)[cols].iloc[::-1]
    wins = int((df["judge_delta"] > 0).sum())
    ties = int((df["judge_delta"] == 0).sum())
    losses = int((df["judge_delta"] < 0).sum())

    lines = [
        "# Evaluation summary: base vs fine-tuned",
        "",
        f"{len(df)} held-out test questions. Judge: Claude (`claude-opus-5`).",
        "",
        tabulate(table, headers=headers, tablefmt="github"),
        "",
        f"Per-question judge outcome for the fine-tuned model: {wins} better, {ties} tied, {losses} worse.",
        "",
        "## 3 biggest improvements (by judge delta, then ROUGE-L delta)",
        "",
        tabulate(top3, headers="keys", tablefmt="github", showindex=False),
        "",
        "## 3 smallest improvements",
        "",
        tabulate(bottom3, headers="keys", tablefmt="github", showindex=False),
        "",
        "## Compliance gate",
        "",
        (f"CRITICAL: {len(alerts)} fine-tuned answer(s) scored below the groundedness floor of "
         f"{GROUNDEDNESS_FLOOR}/5 (question ids {alerts}). Human review required before deployment."
         if alerts else
         f"Passed: no fine-tuned answer scored below the groundedness floor of {GROUNDEDNESS_FLOOR}/5."),
        f"Fine-tuned answers tripping the scope guardrail: {int(df['ft_scope_violation'].sum())}.",
    ]
    text = "\n".join(lines) + "\n"
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Summary written to {SUMMARY_FILE}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("generate", "score", "all"):
        raise SystemExit("Usage: python evaluate_models.py [generate|score]")
    if stage in ("generate", "all"):
        generate_all()
    if stage in ("score", "all"):
        score_all()
