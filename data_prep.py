# data_prep.py
# Load the curated AfyaPlus records, apply the safety normalisation, format for
# LLaMA instruction tuning, validate every example, and split 80/10/10 into
# JSONL files. Based on Week 4 Lab 1, with three additions:
#   1. Mandatory-disclaimer normalisation and check (capstone safety standard)
#   2. Scope-guardrail and max-sequence-length checks as blocking errors
#   3. A leakage-aware split: the 200 questions share only 97 distinct answers,
#      so a naive shuffle puts identical answer text in train AND test. Records
#      are grouped by answer and whole groups are assigned to one split.
#   4. 20 clinical-redirect records (data/raw/safety_refusals.json). Run 1 had
#      no example of declining a clinical question and answered a child's
#      fever with dosing advice; these teach the refusal itself.
import json
import os
import random
from collections import Counter, defaultdict

from config import (BASE_MODEL, DATA_DIR, DISCLAIMER, MAX_SEQ_LEN, RAW_DATA,
                    SAFETY_DATA, SYSTEM_PROMPT, has_disclaimer, scope_violations)

SEED = 42
SPLIT = (0.80, 0.10, 0.10)

# Keyword -> operational area, used only for the coverage report.
# Most specific areas first; the question is matched before the answer.
TOPICS = [
    ("Triage escalation",       ["triage", "queue escalation", "waiting"]),
    ("Referral codes",          ["referral"]),
    ("Records transfer",        ["transfer"]),
    ("Family & dependants",     ["guardian", "dependant", "family", "child"]),
    ("System access",           ["lockout", "locked out", "password", "credentials", "session", "log in", "login"]),
    ("Lab results",             ["laboratory", "results", "lab ", "pending review"]),
    ("Dispensary & stock",      ["dispensary", "stock", "restock", "pharmacy"]),
    ("Billing & coverage",      ["billing", "invoice", "coverage", "insurer", "refund", "bill"]),
    ("Profile updates",         ["profile", "phone number", "address"]),
    ("Registration & identity", ["register", "registration", "national id", "passport"]),
    ("Appointments & booking",  ["appointment", "booking", "book", "rebook", "reschedul", "slot"]),
]


def topic_of(record: dict) -> str:
    if record.get("category") == "clinical_redirect":
        return "Clinical redirect (safety)"
    for text in (record["question"].lower(), record["answer"].lower()):
        for name, keys in TOPICS:
            if any(k in text for k in keys):
                return name
    return "Other"


# ---------------------------------------------------------------
# STEP 1: Load and check the raw records
# ---------------------------------------------------------------
def load_and_validate_data(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) < 100:
        raise ValueError("Dataset too small: please provide at least 100 records.")
    return data


# ---------------------------------------------------------------
# STEP 2: Safety normalisation - every answer must defer clinical judgement
# ---------------------------------------------------------------
def normalise_disclaimer(records: list) -> int:
    added = 0
    for r in records:
        if not has_disclaimer(r["answer"]):
            r["answer"] = r["answer"].rstrip() + " " + DISCLAIMER
            added += 1
    return added


# ---------------------------------------------------------------
# STEP 3: Format each record into the LLaMA instruction-tuning shape
# ---------------------------------------------------------------
def format_example(qa: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
    }


# ---------------------------------------------------------------
# STEP 4: Validate every example before spending GPU credits
# ---------------------------------------------------------------
def _load_llama_tokenizer():
    """Real LLaMA tokeniser if HF access and transformers are available,
    otherwise None so validation falls back to an estimate."""
    try:
        from transformers import AutoTokenizer
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        token = os.getenv("HF_TOKEN")
        return AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    except Exception as exc:
        print(f"[WARN] Could not load the LLaMA tokeniser ({type(exc).__name__}: "
              f"{str(exc).strip().splitlines()[0][:200] if str(exc).strip() else 'no message'}).")
        print("[WARN] Falling back to an approximate token count (chars / 4).")
        return None


def count_tokens(msgs: list, tokenizer) -> int:
    if tokenizer is not None:
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False)
        return len(tokenizer.encode(formatted, add_special_tokens=False))
    chars = sum(len(m.get("content", "")) for m in msgs)
    return chars // 4 + 30


def validate_dataset(examples: list) -> dict:
    """
    Errors (block the split):  structure, empty content, missing disclaimer,
                               scope-guardrail trip, longer than MAX_SEQ_LEN.
    Warnings (review):         duplicate question, shorter than 64 tokens.
    """
    tokenizer = _load_llama_tokenizer()
    errors, warnings = [], []
    seen_questions = set()
    token_counts = []

    for i, example in enumerate(examples):
        msgs = example.get("messages", [])
        if len(msgs) != 3:
            errors.append(f"Example {i}: expected 3 messages, got {len(msgs)}")
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"Example {i}: wrong role order {roles}")
            continue
        for msg in msgs:
            if not msg.get("content", "").strip():
                errors.append(f"Example {i}: empty content in role '{msg['role']}'")

        user_content, answer = msgs[1]["content"], msgs[2]["content"]
        if user_content in seen_questions:
            warnings.append(f"Example {i}: duplicate question '{user_content[:60]}...'")
        seen_questions.add(user_content)

        if not has_disclaimer(answer):
            errors.append(f"Example {i}: answer lacks the mandatory clinical-deferral disclaimer")
        violations = scope_violations(answer)
        if violations:
            errors.append(f"Example {i}: answer trips the scope guardrail {violations}")

        tokens = count_tokens(msgs, tokenizer)
        token_counts.append(tokens)
        if tokens < 64:
            warnings.append(f"Example {i}: very short ({tokens} tokens)")
        if tokens > MAX_SEQ_LEN:
            errors.append(f"Example {i}: {tokens} tokens exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")

    return {
        "tokenizer": "llama-3 chat template (exact)" if tokenizer else "approximate (chars/4 + 30)",
        "total_examples": len(examples),
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "valid_examples": len(examples) - len({e.split(':')[0] for e in errors}),
    }


# ---------------------------------------------------------------
# STEP 5: Leakage-aware 80/10/10 split and save as JSONL
# ---------------------------------------------------------------
def grouped_split(records: list, examples: list) -> dict:
    """Assign whole answer-groups to one split so no test/val answer text is
    ever seen verbatim in training, drawing groups round-robin across
    operational areas so test and val cover the whole domain rather than
    whichever topic the shuffle happened to favour. Sizes are exactly 80/10/10."""
    rng = random.Random(SEED)
    n = len(examples)
    n_val, n_test = int(n * SPLIT[1]), int(n * SPLIT[2])

    groups = defaultdict(list)                       # answer text -> record indices
    for idx, r in enumerate(records):
        groups[r["answer"]].append(idx)
    by_topic = defaultdict(list)                     # topic -> list of groups
    for g in groups.values():
        by_topic[topic_of(records[g[0]])].append(g)
    for t in by_topic:
        rng.shuffle(by_topic[t])
    topics = sorted(by_topic)
    rng.shuffle(topics)

    def fill(target: int) -> list:
        """Round-robin one group per topic per pass; always leave each topic
        at least one group for training."""
        chosen, progress = [], True
        while len(chosen) < target and progress:
            progress = False
            for t in topics:
                pool = by_topic[t]
                for j, g in enumerate(pool[:-1]):    # never take a topic's last group
                    if len(chosen) + len(g) <= target:
                        chosen += pool.pop(j)
                        progress = True
                        break
                if len(chosen) == target:
                    break
        return chosen

    test = fill(n_test)
    val = fill(n_val)
    train = [i for pool in by_topic.values() for g in pool for i in g]
    if len(test) != n_test or len(val) != n_val:
        raise RuntimeError(f"Could not fill exact split sizes (test={len(test)}, val={len(val)})")

    for part in (train, val, test):
        rng.shuffle(part)
    return {"train": train, "val": val, "test": test}


def leakage_report(records: list, split_idx: dict) -> dict:
    train_answers = {records[i]["answer"] for i in split_idx["train"]}
    return {
        "distinct_answers": len({r["answer"] for r in records}),
        "val_answers_seen_in_train": sum(records[i]["answer"] in train_answers for i in split_idx["val"]),
        "test_answers_seen_in_train": sum(records[i]["answer"] in train_answers for i in split_idx["test"]),
    }


def save_splits(examples: list, split_idx: dict, output_dir: str = DATA_DIR) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    sizes = {}
    for name in ("train", "val", "test"):
        path = f"{output_dir}/{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in split_idx[name]:
                f.write(json.dumps(examples[i], ensure_ascii=False) + "\n")
        sizes[name] = len(split_idx[name])
        print(f"Saved {sizes[name]} examples to {path}")
    return sizes


# ---------------------------------------------------------------
# STEP 6: Run the whole pipeline end-to-end
# ---------------------------------------------------------------
if __name__ == "__main__":
    raw = load_and_validate_data(RAW_DATA)
    added = normalise_disclaimer(raw)
    with open(SAFETY_DATA, "r", encoding="utf-8") as f:
        safety = [dict(r, category="clinical_redirect") for r in json.load(f)]
    print(f"Loaded {len(raw)} curated records (disclaimer appended to {added}) "
          f"+ {len(safety)} clinical-redirect records")
    raw = raw + safety

    formatted = [format_example(r) for r in raw]
    report = validate_dataset(formatted)
    report["disclaimers_appended"] = added
    report["clinical_redirect_records"] = len(safety)
    report["coverage_by_topic"] = dict(Counter(topic_of(r) for r in raw).most_common())

    print("\n=== DATASET VALIDATION REPORT ===")
    print(f"Token counter:   {report['tokenizer']}")
    print(f"Total examples:  {report['total_examples']}")
    print(f"Valid examples:  {report['valid_examples']}")
    print(f"Errors:          {report['error_count']}")
    print(f"Warnings:        {report['warning_count']}")
    print(f"Avg token count: {report['avg_tokens']}")
    print(f"Token range:     {report['min_tokens']} to {report['max_tokens']}")
    for e in report["errors"]:
        print(f"  ERROR: {e}")
    for w in report["warnings"]:
        print(f"  WARN:  {w}")
    print("\nCoverage by operational area:")
    for topic, n in report["coverage_by_topic"].items():
        print(f"  {topic:<26} {n}")

    if report["error_count"] == 0:
        split_idx = grouped_split(raw, formatted)
        report["split_sizes"] = save_splits(formatted, split_idx)
        report["leakage"] = leakage_report(raw, split_idx)
        s = report["split_sizes"]
        print(f"\nSplit summary: {s['train']} train | {s['val']} val | {s['test']} test")
        print(f"Leakage check: {report['leakage']}")
        print("\nDataset preparation complete. Ready to upload to the GPU instance.")
    else:
        print("\nFix the errors above before splitting. Do not train on a broken dataset.")

    with open(f"{DATA_DIR}/validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report written to {DATA_DIR}/validation_report.json")
