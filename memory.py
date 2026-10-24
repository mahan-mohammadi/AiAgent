

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("memory.db")


def init_memory():
    """Create the memory table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)



def save_memory(key: str, value: str) -> str:
    """Save a fact, updating the most recent existing entry for its key."""
    key = key.strip()
    value = value.strip()

    if not key or not value:
        return "Memory key and value cannot be empty."

    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute(
            "SELECT id FROM memories WHERE key = ? ORDER BY id DESC LIMIT 1",
            (key,),
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE memories SET value = ?, created_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (value, existing[0]),
            )
            conn.execute(
                "DELETE FROM memories WHERE key = ? AND id != ?",
                (key, existing[0]),
            )
        else:
            conn.execute(
                "INSERT INTO memories (key, value) VALUES (?, ?)",
                (key, value),
            )

    return f"Saved memory: {key} = {value}"

def search_memory(query: str) -> list[dict]:
    """Find memories whose key or value contains the query."""
    terms = query.strip().split()

    if not terms:
        return []

    conditions = []
    parameters = []

    for term in terms:
        conditions.append("(key LIKE ? OR value LIKE ?)")
        pattern = f"%{term}%"
        parameters.extend([pattern, pattern])

    sql = (
        "SELECT id, key, value, created_at FROM memories WHERE "
        + " OR ".join(conditions)
        + " ORDER BY id DESC LIMIT 10"
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, parameters).fetchall()

    return [dict(row) for row in rows]


def get_all_memories() -> list[dict]:
    """Return all saved memories."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, key, value, created_at FROM memories ORDER BY id"
        ).fetchall()

    return [dict(row) for row in rows]


init_memory()



if __name__ == "__main__":
    print(save_memory("preferred_language", "Python"))
    print(search_memory("Python"))
    print(get_all_memories())
