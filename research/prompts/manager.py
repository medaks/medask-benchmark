"""
Versioned prompt management.

Prompts are stored as .txt files in research/prompts/library/.
Each prompt is registered in the SQLite prompts table on first use,
keyed by its SHA256 hash for deduplication.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

from research.db.schema import init_db

LIBRARY_DIR = Path(__file__).parent / "library"


class PromptManager:
    def __init__(self):
        self._conn = init_db()

    def get_or_create(self, version_tag: str) -> Tuple[int, str]:
        fp = LIBRARY_DIR / f"{version_tag}.txt"
        if not fp.exists():
            available = [p.stem for p in LIBRARY_DIR.glob("*.txt")]
            raise FileNotFoundError(
                f"Prompt file not found: {fp}\nAvailable: {available}"
            )

        template_text = fp.read_text(encoding="utf-8")

        if "{vignette}" not in template_text:
            raise ValueError(
                f"Prompt {version_tag} must contain {{vignette}} placeholder"
            )

        sha = hashlib.sha256(template_text.encode()).hexdigest()

        row = self._conn.execute(
            "SELECT prompt_id, template_text FROM prompts WHERE sha256 = ?",
            (sha,),
        ).fetchone()

        if row:
            return row["prompt_id"], row["template_text"]

        cur = self._conn.execute(
            """INSERT INTO prompts (version_tag, template_text, sha256)
               VALUES (?, ?, ?)""",
            (version_tag, template_text, sha),
        )
        self._conn.commit()
        return cur.lastrowid, template_text

    def list_prompts(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT prompt_id, version_tag, description, created_at FROM prompts"
        ).fetchall()
        return [dict(r) for r in rows]
