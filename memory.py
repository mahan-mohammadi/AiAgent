

import re
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

STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "you", "your", "what",
    "when", "where", "which", "who", "how", "why", "can", "could",
    "would", "should", "does", "did", "has", "have", "had", "not",
    "but", "with", "from", "that", "this", "these", "those", "about",
    "into", "than", "then", "them", "they", "their", "there", "here",
    "tell", "please", "some", "any", "all", "out", "get", "got", "let",
    "its", "our", "his", "her", "him", "she", "too", "also", "just",
}

# Phrases that mean "show me what you have saved", where listing recent
# memories is useful even when no keyword matches.
RECALL_HINTS = (
    "do you remember", "what do you remember", "did i tell you",
    "what do you know", "know about me", "saved", "memories", "memory",
)


def asks_about_memory(request: str) -> bool:
    """True if the request is asking what the agent has saved."""
    lowered = request.lower()
    return any(hint in lowered for hint in RECALL_HINTS)


def extract_terms(query: str) -> list[str]:
    """Meaningful search words: no stopwords, nothing under 3 letters."""
    terms: list[str] = []
    for word in re.findall(r"\w+", query.lower()):
        if len(word) >= 3 and word not in STOPWORDS and word not in terms:
            terms.append(word)
    return terms


def _like_pattern(term: str) -> str:
    """Escape LIKE wildcards so '_' and '%' match literally."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_memory(query: str, limit: int = 10) -> list[dict]:
    """Find memories matching any meaningful word, best matches first."""
    terms = extract_terms(query)

    if not terms:
        return []

    conditions = []
    parameters = []

    for term in terms:
        conditions.append("(key LIKE ? ESCAPE '\\' OR value LIKE ? ESCAPE '\\')")
        pattern = _like_pattern(term)
        parameters.extend([pattern, pattern])

    sql = (
        "SELECT id, key, value, created_at FROM memories WHERE "
        + " OR ".join(conditions)
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, parameters).fetchall()]

    def score(row: dict) -> int:
        text = f"{row['key']} {row['value']}".lower()
        return sum(term in text for term in terms)

    rows.sort(key=lambda row: (score(row), row["id"]), reverse=True)
    return rows[:limit]


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
