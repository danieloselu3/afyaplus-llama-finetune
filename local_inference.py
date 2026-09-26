# local_inference.py
# Loads the merged AfyaPlus model and answers operational queries, with the two
# production guardrails the training never had: a scope filter and a mandatory
# clinical-deferral disclaimer. "Local" means on your own hardware rather than
# a hosted API; for an 8B model that is the vast.ai GPU box.
#
# Import-safe: `from local_inference import ask, generate` loads nothing until
# the first call, and the demo below only runs under __main__.
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (DISCLAIMER, MERGED_DIR, OUTPUT_DIR, SYSTEM_PROMPT,
                    has_disclaimer, scope_violations)

MAX_NEW_TOKENS = 300   # Curated answers are 36-76 words; 300 tokens never truncates them
REFUSAL = ("I can only help with AfyaPlus operational workflows such as booking, records, "
           "billing, system access, and escalation. " + DISCLAIMER)

_loaded = {}


def load(model_path: str = MERGED_DIR):
    """Load (once) and cache a model + tokeniser pair."""
    if model_path not in _loaded:
        print(f"Loading model from {model_path} ...")
        tok = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
        model.eval()
        _loaded[model_path] = (model, tok)
    return _loaded[model_path]


def unload(model_path: str) -> None:
    """Free GPU memory so a second 16GB model can be loaded on a 24GB card."""
    _loaded.pop(model_path, None)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stop_ids(tok) -> list:
    ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")}
    return sorted(i for i in ids if i is not None and i != tok.unk_token_id)


@torch.inference_mode()
def generate(question: str, model_path: str = MERGED_DIR,
             system_prompt: str = SYSTEM_PROMPT, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Raw model output with the training-time prompt format. Greedy decoding
    (do_sample=False) makes the answer deterministic, which consistent
    operational guidance requires."""
    model, tok = load(model_path)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True,   # True in INFERENCE: cue the assistant turn
        return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None, top_p=None,           # silence sampling-parameter warnings under greedy
        repetition_penalty=1.1,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        # LLaMA 3 ends an assistant turn with <|eot_id|>; older copies of its
        # config list only <|end_of_text|>, so stop on both explicitly.
        eos_token_id=_stop_ids(tok),
    )
    return tok.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def apply_guardrails(response: str) -> dict:
    """Scope filter first (refuse out-of-scope content), then guarantee the
    mandatory disclaimer is present."""
    violations = scope_violations(response)
    if violations:
        return {"final": REFUSAL, "blocked": True, "violations": violations,
                "raw_has_disclaimer": has_disclaimer(response), "disclaimer_appended": False}
    appended = not has_disclaimer(response)
    final = response if not appended else response.rstrip() + "\n\n" + DISCLAIMER
    return {"final": final, "blocked": False, "violations": [],
            "raw_has_disclaimer": not appended, "disclaimer_appended": appended}


def ask(question: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Production entry point: generate, then apply both guardrails."""
    return apply_guardrails(generate(question, max_new_tokens=max_new_tokens))["final"]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Fresh phrasings (not copied from the dataset) spanning the operational
    # areas, plus one clinical question the assistant must redirect.
    test_questions = [
        "What is the procedure for escalating a triage queue delay at the Kisumu facility?",
        "A patient missed their appointment last week. How do they reschedule?",
        "A staff member is locked out of the staff portal. What are the steps?",
        "How long do lab results take, and what does 'Pending review' mean?",
        "What documents are needed to register a new patient?",
        "A patient disputes a charge on their invoice. What should the front desk do?",
        "The dispensary has run out of an item on a patient's prescription. What do we tell the patient?",
        "I have had a fever and headache for three days. What medicine should I take?",
    ]
    samples = []
    for q in test_questions:
        raw = generate(q)
        guarded = apply_guardrails(raw)
        samples.append({"question": q, "raw_response": raw, **guarded})
        print(f"\nQ: {q}\nA: {guarded['final']}")
        print(f"   [disclaimer in raw output: {guarded['raw_has_disclaimer']} | "
              f"blocked: {guarded['blocked']}]")

    # Stability test: greedy decoding must give identical output twice
    r1 = generate(test_questions[0])
    r2 = generate(test_questions[0])
    stable = r1 == r2
    print("\nStability test:", "PASSED (identical output on repeat query)" if stable else "FAILED")

    with open(f"{OUTPUT_DIR}/inference_samples.json", "w", encoding="utf-8") as f:
        json.dump({"stability_test_passed": stable, "samples": samples}, f, indent=2, ensure_ascii=False)
    with open(f"{OUTPUT_DIR}/inference_samples.md", "w", encoding="utf-8") as f:
        f.write("# Sample responses from the merged AfyaPlus model\n\n")
        f.write(f"Greedy decoding, max_new_tokens={MAX_NEW_TOKENS}. "
                f"Stability test: {'passed' if stable else 'FAILED'}.\n\n")
        for i, s in enumerate(samples, 1):
            f.write(f"## {i}. {s['question']}\n\n{s['final']}\n\n")
            f.write(f"*Disclaimer in raw model output: {'yes' if s['raw_has_disclaimer'] else 'no (appended by guardrail)'}"
                    f"; scope filter: {'BLOCKED ' + str(s['violations']) if s['blocked'] else 'passed'}*\n\n")
    print(f"Samples written to {OUTPUT_DIR}/inference_samples.md and .json")
    assert stable, "Consistency test failed: outputs differ between runs!"
