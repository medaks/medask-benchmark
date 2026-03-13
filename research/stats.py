"""
Shared statistics computation for triage benchmark results.
"""
from typing import Dict, List

TRIAGE_ORDER = {"sc": 1, "ne": 2, "em": 3}


def compute_stats(results: List[Dict]) -> Dict[str, float]:
    total = len(results)
    correct_total = sum(1 for r in results if r["correct"])

    per_level = {}
    for lvl in ["em", "ne", "sc"]:
        lvl_results = [r for r in results if r["true_urgency"] == lvl]
        lvl_correct = sum(1 for r in lvl_results if r["correct"])
        per_level[lvl] = lvl_correct / len(lvl_results) if lvl_results else 0.0

    safe = sum(
        1 for r in results
        if r["llm_output"] in TRIAGE_ORDER and r["true_urgency"] in TRIAGE_ORDER
        and TRIAGE_ORDER[r["llm_output"]] >= TRIAGE_ORDER[r["true_urgency"]]
    )
    incorrect = total - correct_total
    over = sum(
        1 for r in results
        if not r["correct"]
        and r["llm_output"] in TRIAGE_ORDER and r["true_urgency"] in TRIAGE_ORDER
        and TRIAGE_ORDER[r["llm_output"]] > TRIAGE_ORDER[r["true_urgency"]]
    )

    return {
        "overall": correct_total / total if total else 0.0,
        "em": per_level["em"],
        "ne": per_level["ne"],
        "sc": per_level["sc"],
        "safety": safe / total if total else 0.0,
        "overtriage": over / incorrect if incorrect else 0.0,
    }
