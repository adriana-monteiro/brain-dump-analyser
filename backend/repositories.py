
from backend.database import q


# ---------------- conversations ----------------

def get_conversation(conv_id):
    return q(
        "SELECT * FROM conversations WHERE id=?",
        (conv_id,),
        "one",
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
    allowed = {"text", "op", "item_id", "proposed_text", "question", "status"}
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


def update_item_text(item_id: int, text: str):
    q(
        "UPDATE items SET text = ? WHERE id = ?",
        (text.strip(), item_id),
    )


def set_item_status(item_id: int, status: str):
    q(
        "UPDATE items SET status = ? WHERE id = ?",
        (status, item_id),
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


def get_open_tasks_for_topics(topic: str):
    return q(
        """
        SELECT id, text, topic, status
        FROM items
        WHERE kind = 'task'
          AND status = 'open'
          AND topic = ?
        ORDER BY id DESC
        """,
        (topic,),
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

DEFAULT_TOPICS = ["Work", "Personal", "Health", "Finance"]


def get_topics():
    row = q(
        "SELECT value FROM settings WHERE key='topics'",
        fetch="one",
    )
    import json
    return json.loads(row["value"]) if row else DEFAULT_TOPICS


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


