"""Persistent review sessions and atomic daily provider-call limits."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DEFAULT_PATH = Path(__file__).resolve().parent / "runs" / "review_sessions.sqlite3"
SESSION_LIFETIME_SECONDS = 7 * 24 * 60 * 60


def database_path() -> Path:
    return Path(os.environ.get("SESSION_DB_PATH", str(DEFAULT_PATH))).expanduser()


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout = 30000")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS review_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            before_json TEXT NOT NULL,
            after_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily_usage (
            day_utc TEXT NOT NULL,
            action TEXT NOT NULL,
            used INTEGER NOT NULL,
            PRIMARY KEY (day_utc, action)
        );
        """
    )
    return db


@contextmanager
def database() -> Iterator[sqlite3.Connection]:
    db = connect()
    try:
        with db:
            yield db
    finally:
        db.close()


def create_session(session_id: str, payload: dict) -> None:
    now = int(time.time())
    with database() as db:
        db.execute(
            "DELETE FROM review_events WHERE session_id IN (SELECT id FROM sessions WHERE expires_at <= ?)",
            (now,),
        )
        db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (session_id, now, now, now + SESSION_LIFETIME_SECONDS, json.dumps(payload, ensure_ascii=False)),
        )


def get_session(session_id: str) -> dict | None:
    with database() as db:
        row = db.execute(
            "SELECT payload FROM sessions WHERE id = ? AND expires_at > ?",
            (session_id, int(time.time())),
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def update_session(session_id: str, payload: dict, changes: list[tuple[str, dict, dict]]) -> bool:
    now = int(time.time())
    changed_at = datetime.now(timezone.utc).isoformat()
    with database() as db:
        current = db.execute(
            "SELECT id FROM sessions WHERE id = ? AND expires_at > ?", (session_id, now)
        ).fetchone()
        if not current:
            return False
        db.execute(
            "UPDATE sessions SET payload = ?, updated_at = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), now, session_id),
        )
        db.executemany(
            "INSERT INTO review_events (session_id, changed_at, segment_id, before_json, after_json) VALUES (?, ?, ?, ?, ?)",
            [
                (session_id, changed_at, segment_id, json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False))
                for segment_id, before, after in changes
            ],
        )
    return True


def review_events(session_id: str) -> list[dict]:
    with database() as db:
        rows = db.execute(
            "SELECT changed_at, segment_id, before_json, after_json FROM review_events WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {
            "changed_at": row["changed_at"],
            "segment_id": row["segment_id"],
            "before": json.loads(row["before_json"]),
            "after": json.loads(row["after_json"]),
        }
        for row in rows
    ]


def consume_daily_quota(action: str, limit: int) -> bool:
    day = datetime.now(timezone.utc).date().isoformat()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM daily_usage WHERE day_utc < ?", (day,))
        row = db.execute(
            "SELECT used FROM daily_usage WHERE day_utc = ? AND action = ?", (day, action)
        ).fetchone()
        used = row["used"] if row else 0
        if used >= limit:
            return False
        db.execute(
            "INSERT INTO daily_usage (day_utc, action, used) VALUES (?, ?, 1) "
            "ON CONFLICT(day_utc, action) DO UPDATE SET used = used + 1",
            (day, action),
        )
    return True
