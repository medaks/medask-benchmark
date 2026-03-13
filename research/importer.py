"""
Import existing results into the SQLite research database.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from research.db.schema import init_db
from research.db.store import ResultStore
from research.prompts.manager import PromptManager
from research.stats import compute_stats


def _parse_filename(stem: str):
    """
    Parse model name, vignette set, and run number from a filename stem.

    Examples:
        "semigran_o3"           -> ("o3", "semigran", None)
        "semigran_medask_3"     -> ("medask", "semigran", 3)
        "20250703T134639_gpt-4o_semigran_triage" -> ("gpt-4o", "semigran", None)
    """
    ts_match = re.match(r"\d{8}T\d{6}_(.+)_(semigran|kopka)_triage", stem)
    if ts_match:
        return ts_match.group(1), ts_match.group(2), None

    for vs in ["semigran", "kopka"]:
        if stem.startswith(vs + "_"):
            rest = stem[len(vs) + 1:]
            run_match = re.match(r"(.+)_(\d+)$", rest)
            if run_match:
                return run_match.group(1), vs, int(run_match.group(2))
            return rest, vs, None

    return stem, "unknown", None


def import_jsonl_file(
    filepath: Path,
    model_name: Optional[str] = None,
    vignette_set: Optional[str] = None,
    run_number: int = 1,
    source: str = "imported_jsonl",
    batch_id: Optional[str] = None,
) -> List[int]:
    """Import a JSONL file. Handles multi-run files (split by run_id field).
    Returns list of database run_ids created."""
    stem = filepath.stem
    if model_name is None or vignette_set is None:
        m_name, v_set, r_num = _parse_filename(stem)
        model_name = model_name or m_name
        vignette_set = vignette_set or v_set
        if r_num is not None:
            run_number = r_num

    with filepath.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError(f"Empty file: {filepath}")

    if "model" in records[0]:
        model_name = model_name or records[0]["model"]

    # Group by run_id if present, otherwise treat as single run
    from collections import defaultdict
    runs_grouped = defaultdict(list)
    for rec in records:
        rid = rec.get("run_id", run_number)
        runs_grouped[rid].append(rec)

    pm = PromptManager()
    prompt_id, _ = pm.get_or_create("v1_original")

    conn = init_db()
    store = ResultStore(conn)

    created_ids = []
    for rid in sorted(runs_grouped.keys()):
        run_records = runs_grouped[rid]
        results = []
        for rec in run_records:
            results.append({
                "case_id": rec["case_id"],
                "true_urgency": rec["true_urgency"],
                "llm_output": rec["llm_output"],
                "correct": rec["correct"],
                "raw_response": None,
                "latency_ms": None,
            })

        stats = compute_stats(results)

        db_run_id = store.insert_run(
            model_name=model_name,
            prompt_id=prompt_id,
            vignette_set=vignette_set,
            num_vignettes=len(run_records),
            run_number=rid if isinstance(rid, int) else run_number,
            batch_id=batch_id or f"import_{stem}",
            source=source,
            overall_accuracy=stats["overall"],
            em_accuracy=stats["em"],
            ne_accuracy=stats["ne"],
            sc_accuracy=stats["sc"],
            safety_rate=stats["safety"],
            overtriage_rate=stats["overtriage"],
        )

        store.insert_results(db_run_id, results)
        created_ids.append(db_run_id)

    conn.close()
    return created_ids


def import_all_existing(results_root: Path):
    imported = 0
    for jsonl_file in sorted(results_root.rglob("*.jsonl")):
        parent = jsonl_file.parent.name
        try:
            run_ids = import_jsonl_file(
                jsonl_file,
                source="imported_jsonl",
                batch_id=f"import_{parent}",
            )
            print(f"  Imported {jsonl_file.name} -> {len(run_ids)} run(s), ids={run_ids}")
            imported += len(run_ids)
        except Exception as e:
            print(f"  FAILED {jsonl_file.name}: {e}")

    print(f"\nImported {imported} total runs.")


def import_table1_baselines():
    conn = init_db()

    baselines = [
        # (model, vignette_set, overall, em, ne, sc)
        ("MedAsk",      "semigran", 0.876, 0.844, 0.822, 0.956),
        ("o4-mini",     "semigran", 0.804, 0.844, 0.800, 0.778),
        ("o1-mini",     "semigran", 0.778, 0.778, 0.800, 0.756),
        ("o3",          "semigran", 0.756, 0.778, 0.844, 0.644),
        ("GPT-4.5",     "semigran", 0.733, 0.778, 0.800, 0.600),
        ("Claude-3.5",  "semigran", 0.689, 0.778, 0.778, 0.467),
    ]

    for model, vs, overall, em, ne, sc in baselines:
        conn.execute(
            """INSERT OR REPLACE INTO baseline_results
               (source_label, model_name, vignette_set,
                overall_accuracy, em_accuracy, ne_accuracy, sc_accuracy,
                num_runs, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("medask_blog_jul25", model, vs, overall, em, ne, sc, 5,
             "Published Table 1 from MedAsk blog, Jul 2025"),
        )

    conn.commit()
    conn.close()
    print(f"Imported {len(baselines)} baseline entries.")
