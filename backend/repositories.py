
from backend.config import settings
from backend.database import q


# ---------------- conversations ----------------

def get_conversation(conv_id):
    return q(
        "SELECT * FROM conversations WHERE id=?",
        (conv_id,),
        "one",
    )


def get_latest_open_conversation():
    """Most recent open conversation that still has pending candidates.

    Used to resume an unresolved review after the UI loses its in-memory
    state (e.g. a browser refresh). Returns the conversation row or None.
    """
    return q(
        """
        SELECT c.*
        FROM conversations c
        WHERE c.closed = 0
          AND EXISTS (
              SELECT 1 FROM candidates cand
              WHERE cand.conv_id = c.id AND cand.status = 'pending'
          )
        ORDER BY c.id DESC
        LIMIT 1
        """,
        fetch="one",
    )


def get_or_create_conversation(conv_id=None):
    if conv_id:
        row = q(
            "SELECT * FROM conversations WHERE id=?",
            (conv_id,),
            "one",
        )
        if row and not row["closed"]:
            return conv_id

    return q("INSERT INTO conversations DEFAULT VALUES")


def close_conversation(conv_id):
    q(
        "UPDATE conversations SET closed=1 WHERE id=?",
        (conv_id,),
    )

def reset_conversation(conv_id):
    """Erase conversation content without deleting saved items."""
    conversation = get_conversation(conv_id)

    if not conversation:
        return False

    # Delete pending and resolved candidates, including their questions.
    q(
        "DELETE FROM candidates WHERE conv_id=?",
        (conv_id,),
    )

    # Delete the conversation's chat history.
    q(
        "DELETE FROM messages WHERE conv_id=?",
        (conv_id,),
    )

    # Retain the conversation record, but close it.
    close_conversation(conv_id)

    return True


def add_message(conv_id, role, content):
    q(
        """
        INSERT INTO messages(conv_id, role, content)
        VALUES (?, ?, ?)
        """,
        (conv_id, role, content),
    )


def get_history(conv_id, limit=10):
    rows = q(
        """
        SELECT role, content FROM messages
        WHERE conv_id=?
        ORDER BY id DESC LIMIT ?
        """,
        (conv_id, limit),
        "all",
    )
    return rows[::-1]


# ---------------- candidates ----------------

_CANDIDATE_SELECT = """
    SELECT
        c.*,
        i.text AS existing_item_text,
        i.kind AS existing_item_kind,
        i.status AS existing_item_status
    FROM candidates c
    LEFT JOIN items i
        ON i.id = c.item_id
"""


def get_pending_candidates(conv_id):
    return q(
        _CANDIDATE_SELECT + """
        WHERE c.conv_id=? AND c.status='pending'
        ORDER BY c.display_no
        """,
        (conv_id,),
        "all",
    )


def get_candidate(candidate_id):
    return q(
        _CANDIDATE_SELECT + " WHERE c.id=?",
        (candidate_id,),
        "one",
    )


def create_candidate(conv_id, display_no, kind, text, topic):
    return q(
        """
        INSERT INTO candidates(conv_id, display_no, kind, text, topic)
        VALUES (?, ?, ?, ?, ?)
        """,
        (conv_id, display_no, kind, text, topic),
    )


def update_candidate(candidate_id, **fields):
    allowed = {"text", "op", "item_id", "proposed_text", "question", "status", "kind", "topic"}
    updates = {k: v for k, v in fields.items() if k in allowed}

    if not updates:
        return

    columns = ", ".join(f"{key}=?" for key in updates)
    q(
        f"UPDATE candidates SET {columns} WHERE id=?",
        (*updates.values(), candidate_id),
    )

# ---------------- items ----------------

def create_item(kind: str, text: str, topic: str = "INBOX") -> int:
    status = "open" if kind == "task" else None

    return q(
        """
        INSERT INTO items (kind, text, topic, status)
        VALUES (?, ?, ?, ?)
        """,
        (kind, text.strip(), topic, status),
    )


def get_items(kind: str | None = None, status: str | None = None):
    sql = "SELECT * FROM items WHERE 1=1"
    args = []

    if kind is not None:
        sql += " AND kind = ?"
        args.append(kind)

    if status is not None:
        sql += " AND status = ?"
        args.append(status)

    sql += " ORDER BY id DESC"
    return q(sql, args, fetch="all")


def get_item(item_id: int):
    return q(
        "SELECT * FROM items WHERE id = ?",
        (item_id,),
        fetch="one",
    )

# ---------------- tasks ----------------

def get_open_tasks():
    return q(
        """
        SELECT id, text, topic, status, created_at
        FROM items
        WHERE kind = 'task' AND status = 'open'
        ORDER BY id DESC
        """,
        fetch="all",
    )


def create_task(text: str, topic: str = "INBOX") -> int:
    return create_item("task", text, topic)

def update_task_text(task_id: int, text: str):
    q(
        """
        UPDATE items
        SET text = ?
        WHERE id = ? AND kind = 'task'
        """,
        (text.strip(), task_id),
    )


def set_task_status(task_id: int, status: str):
    if status not in {"open", "done", "cancelled"}:
        raise ValueError("Invalid task status")

    q(
        """
        UPDATE items
        SET status = ?
        WHERE id = ? AND kind = 'task'
        """,
        (status, task_id),
    )

def get_task(task_id: int):
    return q(
        """
        SELECT *
        FROM items
        WHERE id = ? AND kind = 'task'
        """,
        (task_id,),
        fetch="one",
    )

# ---------------- settings ----------------

def get_topics():
    row = q(
        "SELECT value FROM settings WHERE key='topics'",
        fetch="one",
    )
    import json
    return json.loads(row["value"]) if row else list(settings.default_topics)


def set_topics(topics):
    import json

    cleaned = [
        topic.strip()
        for topic in topics
        if topic.strip() and topic.strip().upper() != "INBOX"
    ]

    q(
        "INSERT OR REPLACE INTO settings(key, value) VALUES('topics', ?)",
        (json.dumps(cleaned),),
    )



def get_max_display_no(conv_id):
    row = q(
        "SELECT MAX(display_no) AS max_no FROM candidates WHERE conv_id=?",
        (conv_id,),
        "one",
    )
    return row["max_no"] or 0



# ---------------- llm_calls ----------------

def insert_llm_calls(rows) -> int:
    """Insert each mapped row into llm_calls; return the number inserted.

    Rows come from metrics.record_to_row(), so cost fields may be None; the
    nullable REAL columns store those as NULL (null-as-zero is applied only at
    aggregation time, never here).
    """
    for row in rows:
        q(
            """
            INSERT INTO llm_calls(
                step, model, input_tokens, output_tokens, cached_tokens,
                latency_ms, input_cost_usd, output_cost_usd, cost_usd,
                status, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["step"],
                row["model"],
                row["input_tokens"],
                row["output_tokens"],
                row["cached_tokens"],
                row["latency_ms"],
                row["input_cost_usd"],
                row["output_cost_usd"],
                row["cost_usd"],
                row["status"],
                row["error"],
            ),
        )

    return len(rows)


def list_llm_calls(step=None, model=None, status=None, limit=100):
    """Return llm_calls rows filtered by any provided equality filters.

    Ordered by created_at DESC with id DESC as tiebreaker so rows sharing a
    timestamp still come back in a deterministic (newest-first) order.
    """
    sql = "SELECT * FROM llm_calls WHERE 1=1"
    args = []

    if step is not None:
        sql += " AND step = ?"
        args.append(step)

    if model is not None:
        sql += " AND model = ?"
        args.append(model)

    if status is not None:
        sql += " AND status = ?"
        args.append(status)

    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    args.append(limit)

    return q(sql, args, fetch="all")


def summarize_llm_calls(step=None, model=None):
    """Aggregate figures over llm_calls, filtered by optional step/model.

    Returns totals, average latency, and per-step / per-model breakdowns
    shaped for MetricsSummaryOut. Aggregates are computed in SQL so the
    summary never materializes every row just to total it.
    """
    where = " WHERE 1=1"
    args = []

    if step is not None:
        where += " AND step = ?"
        args.append(step)

    if model is not None:
        where += " AND model = ?"
        args.append(model)

    # COALESCE wraps the cost sum and average so an empty set (or a column of
    # all-NULL costs) collapses to 0 instead of NULL. Null cost is preserved at
    # storage time and only treated as zero here, at aggregation time.
    totals = q(
        """
        SELECT
            COUNT(*)                        AS call_count,
            COALESCE(SUM(COALESCE(cost_usd, 0)), 0) AS total_cost_usd,
            COALESCE(SUM(input_tokens), 0)  AS total_input_tokens,
            COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
            COALESCE(SUM(error), 0)         AS error_count,
            COALESCE(AVG(latency_ms), 0)    AS avg_latency_ms
        FROM llm_calls
        """
        + where,
        args,
        fetch="one",
    )

    by_step = q(
        """
        SELECT
            step                            AS key,
            COUNT(*)                        AS call_count,
            COALESCE(SUM(COALESCE(cost_usd, 0)), 0) AS cost_usd,
            COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens
        FROM llm_calls
        """
        + where
        + " GROUP BY step",
        args,
        fetch="all",
    )

    by_model = q(
        """
        SELECT
            model                           AS key,
            COUNT(*)                        AS call_count,
            COALESCE(SUM(COALESCE(cost_usd, 0)), 0) AS cost_usd,
            COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens
        FROM llm_calls
        """
        + where
        + " GROUP BY model",
        args,
        fetch="all",
    )

    return {
        "total_cost_usd": totals["total_cost_usd"],
        "total_input_tokens": totals["total_input_tokens"],
        "total_output_tokens": totals["total_output_tokens"],
        "avg_latency_ms": totals["avg_latency_ms"],
        "call_count": totals["call_count"],
        "error_count": totals["error_count"],
        "by_step": by_step,
        "by_model": by_model,
    }
