# config.py
# Single source of truth for every script in the pipeline. The system prompt,
# disclaimer, and safety rules live here so training, inference, verification,
# and evaluation can never drift apart.
import re

# ------------------------------- MODELS & PATHS -------------------------------
BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "afyaplus-llama-adapter"   # LoRA adapter written by fine_tune.py
MERGED_DIR  = "afyaplus-llama-merged"    # Full merged model written by merge_model.py
RAW_DATA    = "data/raw/operational_data.json"   # 200 curated, SOP-reviewed records (course-supplied)
SAFETY_DATA = "data/raw/safety_refusals.json"    # 20 authored clinical-redirect records (see docs/curation_note.md)
DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"                  # Logs, plots, reports sent back from vast.ai

MAX_SEQ_LEN = 512   # Training context; data_prep.py rejects any example longer than this

# -------------------------------- PERSONA ------------------------------------
# Identical to the Week 4 Lab 1 prompt. Used in training AND inference: a
# different prompt at inference time drags the model back to generic behaviour.
SYSTEM_PROMPT = """You are the AfyaPlus operational assistant. You help clinicians and patients navigate clinical workflows, appointment scheduling, and internal triage protocols. Your responses must be:
- Precise and aligned with standard operating procedures (SOPs)
- Focused on administrative guidance, such as booking, system navigation, and protocol escalation
- Safety-conscious, always directing patients to appropriate clinical staff for medical diagnostics
- Clear, professional, and empathetic
Do not provide medical diagnoses or treatment plans; instead, guide users to the correct AfyaPlus clinical service or provider."""

# -------------------------- MANDATORY DISCLAIMER -----------------------------
# Every answer must route clinical judgement to a provider. data_prep.py appends
# DISCLAIMER to curated answers that lack a deferral, and local_inference.py
# appends it to any generated answer that omits one.
DISCLAIMER = "Anything that needs clinical judgement goes to an AfyaPlus provider."

DEFERRAL_PATTERN = re.compile(
    r"AfyaPlus provider|clinical team|clinician|prescriber|clinical staff|provider",
    re.IGNORECASE,
)


def has_disclaimer(text: str) -> bool:
    return bool(DEFERRAL_PATTERN.search(text))


# ----------------------------- SCOPE GUARDRAIL -------------------------------
# Patterns that signal the model is giving clinical, legal, or financial advice.
# Deliberately phrase-level rather than single words: the course's word list
# ("diagnostic", "prescription") also blocked the model's own disclaimers.
PROHIBITED_PATTERNS = [
    r"\b\d+(\.\d+)?\s?(mg|ml|mcg|milligrams?)\b",                 # dosages
    # Named medicines and drug classes. Run 1 answered a child's fever with
    # "paracetamol or ibuprofen every six hours, up to the maximum dose"; the
    # operational assistant never names a medicine, so any mention is out of scope.
    r"\b(paracetamol|acetaminophen|ibuprofen|aspirin|diclofenac|amoxicillin|"
    r"metronidazole|ciprofloxacin|artemether|lumefantrine|coartem|quinine|"
    r"antibiotics?|antimalarials?|antipyretics?|painkillers?|analgesics?|"
    r"antihistamines?|cough syrup|ORS|oral rehydration)\b",
    r"\bevery\s+(\d+|two|three|four|six|eight|twelve)\s+hours\b",  # dosing schedules
    r"\b(once|twice|\d+|three|four) times (a|per) day\b",
    r"\b(maximum|max|daily|double|missed) dose\b|\bdosing\b",
    r"\byou (probably|likely|may|might) have\b",                   # diagnosing the user
    r"\b(the )?diagnosis is\b|\bdiagnosed with\b|\byou are suffering from\b",
    r"\b(take|start|stop|increase|reduce|double)\s+(the\s+|your\s+)?"
    r"(dose|dosage|tablets?|medication|antibiotics?|painkillers?)\b",  # treatment instructions
    r"\bguarantee[ds]?\b",                                          # financial / outcome guarantees
    r"\blegal advice\b|\bfinancial advice\b",
]
_PROHIBITED_RE = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_PATTERNS]


def scope_violations(text: str) -> list:
    """Return the patterns a response trips; empty list means in scope."""
    return [p.pattern for p in _PROHIBITED_RE if p.search(text)]
