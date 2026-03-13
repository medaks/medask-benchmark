"""
Enhanced triage benchmark runner with metadata tracking and dual storage.
"""
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.ummon.lmstudio import UmmonLMStudio
from research.db.schema import init_db
from research.db.store import ResultStore
from research.prompts.manager import PromptManager
from research.stats import compute_stats


def build_client(
    model: str,
    temperature: float = 0.6,
    max_tokens: int = 300,
    base_url: str = "http://localhost:1234/v1",
) -> UmmonLMStudio:
    return UmmonLMStudio(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


def load_vignettes(vignette_set: str) -> List[Dict[str, Any]]:
    vignette_dir = Path(__file__).parent.parent / "triage_bench" / "vignettes"
    fp = vignette_dir / f"{vignette_set}_vignettes.jsonl"
    if not fp.exists():
        raise FileNotFoundError(f"Vignette file not found: {fp}")
    with fp.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_triage_answer(raw: str) -> str:
    """Extract em/ne/sc from model output, handling thinking models.

    Strategy:
      1. If the response is very short (≤20 chars) try a simple match.
      2. Otherwise search from the END of the text so we get the
         model's conclusion, not stray matches inside reasoning.
      3. Fall back to first match if nothing found at the end.
    """
    cleaned = re.sub(r"[`\s]", " ", raw.lower()).strip()

    # Short response — most likely just the label
    if len(cleaned) <= 20:
        match = re.search(r"\b(em|ne|sc)\b", cleaned)
        if match:
            return match.group(1)

    # Look for the LAST occurrence (the conclusion, not the reasoning)
    matches = list(re.finditer(r"\b(em|ne|sc)\b", cleaned))
    if matches:
        return matches[-1].group(1)

    return cleaned[:50]


def evaluate_single_vignette(
    client: UmmonLMStudio,
    prompt_template: str,
    case_description: str,
) -> Tuple[str, str, float]:
    prompt_text = prompt_template.format(vignette=case_description)

    t0 = time.perf_counter()
    history = [{"role": "user", "content": prompt_text}]
    content, reasoning = client._converse_full(history)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # For the parsed prediction, use content (the clean answer) first.
    # If content is blank or very long (fell back to reasoning),
    # the extract function handles both cases.
    prediction = extract_triage_answer(content)

    # Store full output for debugging: content + reasoning
    if reasoning:
        raw_response = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{content}\n</answer>"
    else:
        raw_response = content

    return prediction, raw_response, latency_ms


def run_benchmark(
    model: str,
    model_family: Optional[str] = None,
    quantization: Optional[str] = None,
    parameter_count: Optional[str] = None,
    temperature: float = 0.6,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: int = 2048,
    context_length: Optional[int] = None,
    gpu_layers: Optional[int] = None,
    prompt_version: str = "v1_original",
    vignette_set: str = "semigran",
    runs: int = 1,
    notes: Optional[str] = None,
    jsonl_dir: Optional[str] = None,
    base_url: str = "http://localhost:1234/v1",
    grader_model: Optional[str] = None,
    grader_base_url: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> str:
    """Run triage benchmark.

    progress_callback: optional callable(run_number, total_runs, case_idx,
                       total_cases, pred, gold, correct, errors) called
                       after each vignette.  Used by the Streamlit UI for
                       live progress display.
    """
    batch_id = str(uuid.uuid4())[:8]

    conn = init_db()
    store = ResultStore(conn)

    pm = PromptManager()
    prompt_id, prompt_template = pm.get_or_create(prompt_version)

    client = build_client(model, temperature, max_tokens, base_url)

    grader_client = None
    if grader_model:
        from research.grader import build_grader_client
        grader_client = build_grader_client(
            grader_model,
            base_url=grader_base_url or base_url,
        )

    vignettes = load_vignettes(vignette_set)
    num_vignettes = len(vignettes)

    if jsonl_dir is None:
        jsonl_dir = str(
            Path(__file__).parent.parent / "triage_bench" / "results"
        )
    os.makedirs(jsonl_dir, exist_ok=True)

    for run_number in range(1, runs + 1):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_model = model.replace("/", "_").replace("\\", "_")
        jsonl_path = os.path.join(
            jsonl_dir, f"{ts}_{safe_model}_{vignette_set}_run{run_number}.jsonl"
        )

        started_at = datetime.now().isoformat()
        t_run_start = time.perf_counter()

        results = []

        errors = 0
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            for idx, v in tqdm(
                list(enumerate(vignettes, 1)),
                desc=f"Run {run_number}/{runs}",
            ):
                gold = v["urgency_level"].strip().lower()

                try:
                    pred, raw_resp, latency = evaluate_single_vignette(
                        client, prompt_template, v["case_description"]
                    )
                except Exception as e:
                    errors += 1
                    print(f"\n  [!] Case {idx} failed: {e}")
                    pred = "ERROR"
                    raw_resp = f"ERROR: {e}"
                    latency = 0.0

                if grader_client and pred != "ERROR":
                    try:
                        from research.grader import grade_response
                        pred = grade_response(
                            grader_client, raw_resp, v["case_description"]
                        )
                    except Exception as e:
                        print(f"\n  [!] Grader failed on case {idx}: {e}")

                correct = pred == gold

                if progress_callback:
                    progress_callback(
                        run_number, runs, idx, num_vignettes,
                        pred, gold, correct, errors,
                    )

                rec = {
                    "run_id": run_number,
                    "case_id": idx,
                    "true_urgency": gold,
                    "llm_output": pred,
                    "correct": correct,
                    "model": model,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                results.append({
                    "case_id": idx,
                    "true_urgency": gold,
                    "llm_output": pred,
                    "correct": correct,
                    "raw_response": raw_resp,
                    "latency_ms": latency,
                })

        completed_at = datetime.now().isoformat()
        total_seconds = time.perf_counter() - t_run_start

        stats = compute_stats(results)

        run_id = store.insert_run(
            model_name=model,
            model_family=model_family,
            quantization=quantization,
            parameter_count=parameter_count,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            context_length=context_length,
            gpu_layers=gpu_layers,
            prompt_id=prompt_id,
            vignette_set=vignette_set,
            num_vignettes=num_vignettes,
            run_number=run_number,
            batch_id=batch_id,
            source="local",
            overall_accuracy=stats["overall"],
            em_accuracy=stats["em"],
            ne_accuracy=stats["ne"],
            sc_accuracy=stats["sc"],
            safety_rate=stats["safety"],
            overtriage_rate=stats["overtriage"],
            started_at=started_at,
            completed_at=completed_at,
            total_seconds=total_seconds,
            grader_model=grader_model,
            notes=notes,
        )

        store.insert_results(run_id, results)

        print(f"\n--- Run {run_number}/{runs} | {model} ---")
        print(f"  Overall: {stats['overall']:.1%}")
        print(f"  em: {stats['em']:.1%}  ne: {stats['ne']:.1%}  sc: {stats['sc']:.1%}")
        print(f"  Safety: {stats['safety']:.1%}  Over-triage: {stats['overtriage']:.1%}")
        print(f"  Time: {total_seconds:.1f}s")
        if errors:
            print(f"  Errors: {errors}/{num_vignettes} vignettes failed")

    conn.close()
    print(f"\nBatch {batch_id} complete. {runs} run(s) stored.")
    return batch_id
