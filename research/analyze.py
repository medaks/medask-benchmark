"""
Statistical analysis: McNemar's test, mean/SD summaries.
"""
from typing import Dict, Optional

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from research.db.schema import init_db
from research.db.store import ResultStore


def paired_comparison(
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
    vignette_set: str = "semigran",
    run_id_a: Optional[int] = None,
    run_id_b: Optional[int] = None,
) -> Optional[Dict]:
    """Run McNemar's test. Returns results dict, or None on error."""
    conn = init_db()
    store = ResultStore(conn)

    if run_id_a is None:
        if not model_a:
            conn.close()
            return None
        runs_a = store.get_runs(model_name=model_a, vignette_set=vignette_set)
        if not runs_a:
            conn.close()
            return None
        run_id_a = runs_a[0]["run_id"]

    if run_id_b is None:
        if not model_b:
            conn.close()
            return None
        runs_b = store.get_runs(model_name=model_b, vignette_set=vignette_set)
        if not runs_b:
            conn.close()
            return None
        run_id_b = runs_b[0]["run_id"]

    results_a = store.get_run_results(run_id_a)
    results_b = store.get_run_results(run_id_b)

    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    merged = pd.merge(
        df_a[["case_id", "correct", "llm_output"]],
        df_b[["case_id", "correct", "llm_output"]],
        on="case_id",
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        conn.close()
        return None

    b = int(((merged.correct_a == 1) & (merged.correct_b == 0)).sum())
    c = int(((merged.correct_a == 0) & (merged.correct_b == 1)).sum())
    discordant = b + c

    exact_flag = discordant < 25
    res = mcnemar([[0, b], [c, 0]], exact=exact_flag, correction=not exact_flag)

    acc_a = float(merged.correct_a.mean())
    acc_b = float(merged.correct_b.mean())

    conn.close()

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "a_right_b_wrong": b,
        "a_wrong_b_right": c,
        "p_value": float(res.pvalue),
        "exact": exact_flag,
        "odds_ratio": b / c if c > 0 else float("inf"),
        "accuracy_diff": acc_b - acc_a,
    }
