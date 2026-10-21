
import sqlite3
from contextlib import closing

from backend.config import settings

DB_PATH = settings.database_path


def q(sql, args=(), fetch=None):
    """Execute a query and optionally return rows as dictionaries."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row

        with conn:
            cur = conn.execute(sql, args)

            if fetch == "all":
                return [dict(row) for row in cur.fetchall()]

            if fetch == "one":
                row = cur.fetchone()
                return dict(row) if row else None

            return cur.lastrowid


def init_db():
    """Create all application tables."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            closed INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            conv_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            kind TEXT NOT NULL
                CHECK (kind IN ('task', 'note', 'idea')),
            text TEXT NOT NULL,
            context TEXT,
            topic TEXT NOT NULL DEFAULT 'INBOX',
            status TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY,
            conv_id INTEGER NOT NULL,
            display_no INTEGER NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT 'INBOX',
            status TEXT NOT NULL DEFAULT 'pending',
            op TEXT NOT NULL DEFAULT 'new',
            item_id INTEGER,
            proposed_text TEXT,
            question TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id              INTEGER PRIMARY KEY,
            step            TEXT NOT NULL,
            model           TEXT NOT NULL,
            input_tokens    INTEGER NOT NULL DEFAULT 0,
            output_tokens   INTEGER NOT NULL DEFAULT 0,
            cached_tokens   INTEGER NOT NULL DEFAULT 0,
            latency_ms      INTEGER NOT NULL DEFAULT 0,
            input_cost_usd  REAL,
            output_cost_usd REAL,
            cost_usd        REAL,
            status          TEXT NOT NULL CHECK (status IN ('ok', 'error')),
            error           INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]

    for statement in statements:
        q(statement)
