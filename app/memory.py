import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    created_at: str


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)"
            )

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )

    def recent_messages(self, conversation_id: str, limit: int) -> list[StoredMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [StoredMessage(*row) for row in reversed(rows)]

    def all_messages(self, conversation_id: str) -> list[StoredMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [StoredMessage(*row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
