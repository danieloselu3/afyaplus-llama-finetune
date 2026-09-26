# evaluator.py
# The Week 3 lexical metrics suite, rebuilt: ROUGE-L and token-level F1
# between a reference answer and a candidate response.
import re
from collections import Counter

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(reference: str, candidate: str) -> float:
    """SQuAD-style bag-of-words F1: rewards shared content words regardless of order."""
    ref, cand = _tokens(reference), _tokens(candidate)
    if not ref or not cand:
        return 0.0
    common = sum((Counter(ref) & Counter(cand)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(cand), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def evaluate_response(reference: str, candidate: str) -> dict:
    """ROUGE-L F-measure (longest common subsequence: rewards matching
    structure and order) and token F1 (rewards shared vocabulary)."""
    rouge_l = _scorer.score(reference, candidate)["rougeL"].fmeasure
    return {"rouge_l": round(rouge_l, 4), "token_f1": round(token_f1(reference, candidate), 4)}


if __name__ == "__main__":
    ref = "Log the delay under Queue Escalation and tag the duty nurse practitioner."
    print(evaluate_response(ref, ref))
    print(evaluate_response(ref, "Escalate it to a manager."))
