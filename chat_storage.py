

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "chat_history.sqlite"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id)
                    REFERENCES conversations(thread_id)
                    ON DELETE CASCADE
            )
        """)


def create_conversation(thread_id):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (thread_id, title)
            VALUES (?, ?)
            """,
            (thread_id, "New conversation"),
        )


def save_message(thread_id, role, content):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (thread_id, role, content)
            VALUES (?, ?, ?)
            """,
            (thread_id, role, content),
        )

        if role == "user":
            conn.execute(
                """
                UPDATE conversations
                SET title = CASE
                    WHEN title = 'New conversation' THEN ?
                    ELSE title
                END
                WHERE thread_id = ?
                """,
                (content[:50], thread_id),
            )


def load_messages(thread_id):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE thread_id = ?
            ORDER BY id
            """,
            (thread_id,),
        ).fetchall()

    return [
        {"role": row["role"], "content": row["content"]}
        for row in rows
    ]



def list_conversations():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.thread_id, c.title, c.created_at
            FROM conversations c
            WHERE EXISTS (
                SELECT 1
                FROM messages m
                WHERE m.thread_id = c.thread_id
            )
            ORDER BY c.created_at DESC, c.rowid DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
