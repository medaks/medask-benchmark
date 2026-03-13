"""
Database access layer for the triage research database.
"""
import math
import sqlite3
from typing import Any, Dict, List, Optional


class ResultStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_run(self, **kwargs) -> int:
        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        cur = self._conn.execute(
            f"INSERT INTO benchmark_runs ({columns}) VALUES ({placeholders})",
            list(kwargs.values()),
        )
        self._conn.commit()
        return cur.lastrowid

    def insert_results(self, run_id: int, results: List[Dict]) -> None:
        self._conn.executemany(
            """INSERT INTO run_results
               (run_id, case_id, true_urgency, llm_output, correct, raw_response, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (run_id, r["case_id"], r["true_urgency"], r["llm_output"],
                 int(r["correct"]), r.get("raw_response"), r.get("latency_ms"))
                for r in results
            ],
        )
        self._conn.commit()

    def get_runs(
        self,
        model_name: Optional[str] = None,
        vignette_set: Optional[str] = None,
        source: Optional[str] = None,
        quantization: Optional[str] = None,
        prompt_id: Optional[int] = None,
        batch_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM benchmark_runs WHERE 1=1"
        params = []

        for col, val in [
            ("model_name", model_name),
            ("vignette_set", vignette_set),
            ("source", source),
            ("quantization", quantization),
            ("prompt_id", prompt_id),
            ("batch_id", batch_id),
        ]:
            if val is not None:
                query += f" AND {col} = ?"
                params.append(val)

        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_model_summary(
        self,
        vignette_set: str = "semigran",
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where = "WHERE vignette_set = ?"
        params: list = [vignette_set]
        if source:
            where += " AND source = ?"
            params.append(source)

        query = f"""
            SELECT
                model_name,
                quantization,
                temperature,
                prompt_id,
                COUNT(*) as num_runs,
                AVG(overall_accuracy) as mean_overall,
                AVG(overall_accuracy * overall_accuracy) - AVG(overall_accuracy) * AVG(overall_accuracy) as var_overall,
                AVG(em_accuracy) as mean_em,
                AVG(ne_accuracy) as mean_ne,
                AVG(sc_accuracy) as mean_sc,
                AVG(safety_rate) as mean_safety,
                AVG(overtriage_rate) as mean_overtriage
            FROM benchmark_runs
            {where}
            GROUP BY model_name, quantization, temperature, prompt_id
            ORDER BY mean_overall DESC
        """
        rows = self._conn.execute(query, params).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            var = d.pop("var_overall", 0) or 0
            d["sd_overall"] = math.sqrt(max(var, 0))
            result.append(d)

        return result

    def get_baselines(
        self,
        vignette_set: str = "semigran",
        source_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM baseline_results WHERE vignette_set = ?"
        params: list = [vignette_set]
        if source_label:
            query += " AND source_label = ?"
            params.append(source_label)
        query += " ORDER BY overall_accuracy DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_run_results(self, run_id: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM run_results WHERE run_id = ? ORDER BY case_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
