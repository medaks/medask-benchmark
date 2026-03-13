"""
SQLite schema creation for the triage research database.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "triage_research.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    version_tag     TEXT NOT NULL UNIQUE,
    template_text   TEXT NOT NULL,
    description     TEXT,
    sha256          TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Model identification
    model_name      TEXT NOT NULL,
    model_family    TEXT,
    quantization    TEXT,
    parameter_count TEXT,

    -- Inference settings
    temperature     REAL,
    top_p           REAL,
    top_k           INTEGER,
    max_tokens      INTEGER,
    context_length  INTEGER,
    gpu_layers      INTEGER,

    -- Prompt
    prompt_id       INTEGER NOT NULL REFERENCES prompts(prompt_id),

    -- Dataset
    vignette_set    TEXT NOT NULL,
    num_vignettes   INTEGER NOT NULL,

    -- Run metadata
    run_number      INTEGER NOT NULL DEFAULT 1,
    batch_id        TEXT,
    source          TEXT NOT NULL DEFAULT 'local',

    -- Results summary (denormalized for fast queries)
    overall_accuracy    REAL,
    em_accuracy         REAL,
    ne_accuracy         REAL,
    sc_accuracy         REAL,
    safety_rate         REAL,
    overtriage_rate     REAL,

    -- Timing
    started_at      TEXT,
    completed_at    TEXT,
    total_seconds   REAL,

    -- Grader
    grader_model    TEXT,

    -- Notes
    notes           TEXT,

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_model ON benchmark_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_runs_batch ON benchmark_runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_runs_vignette_set ON benchmark_runs(vignette_set);
CREATE INDEX IF NOT EXISTS idx_runs_source ON benchmark_runs(source);

CREATE TABLE IF NOT EXISTS run_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
    case_id         INTEGER NOT NULL,
    true_urgency    TEXT NOT NULL,
    llm_output      TEXT NOT NULL,
    correct         INTEGER NOT NULL,
    raw_response    TEXT,
    latency_ms      REAL,

    UNIQUE(run_id, case_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run ON run_results(run_id);

CREATE TABLE IF NOT EXISTS baseline_results (
    baseline_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    source_label    TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    vignette_set    TEXT NOT NULL,
    overall_accuracy REAL NOT NULL,
    em_accuracy      REAL,
    ne_accuracy      REAL,
    sc_accuracy      REAL,
    safety_rate      REAL,
    overtriage_rate  REAL,
    num_runs         INTEGER,
    std_dev          REAL,
    notes            TEXT,

    UNIQUE(source_label, model_name, vignette_set)
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply incremental migrations for columns added after initial release."""
    cur = conn.execute("PRAGMA table_info(benchmark_runs)")
    existing = {row[1] for row in cur.fetchall()}
    if "grader_model" not in existing:
        conn.execute("ALTER TABLE benchmark_runs ADD COLUMN grader_model TEXT")
        conn.commit()


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = get_connection(db_path)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    _migrate(conn)
    return conn
