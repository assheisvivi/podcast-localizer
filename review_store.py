"""Review storage: local SQLite or a persistent PostgreSQL database."""

from __future__ import annotations

import os

import review_store_sqlite as sqlite_store


def backend() -> str:
    if os.environ.get("DATABASE_URL"):
        return "postgres"
    if os.environ.get("REQUIRE_PERSISTENT_STORE") == "1":
        raise RuntimeError("DATABASE_URL is required for persistent review storage.")
    return "sqlite"


def store():
    if backend() == "postgres":
        import review_store_postgres

        return review_store_postgres
    return sqlite_store


def ensure_ready() -> None:
    if backend() == "postgres":
        store().ensure_ready()


def create_session(session_id: str, payload: dict) -> None:
    store().create_session(session_id, payload)


def get_session(session_id: str) -> dict | None:
    return store().get_session(session_id)


def update_session(session_id: str, payload: dict, changes: list[tuple[str, dict, dict]]) -> bool:
    return store().update_session(session_id, payload, changes)


def review_events(session_id: str) -> list[dict]:
    return store().review_events(session_id)


def consume_daily_quota(action: str, limit: int) -> bool:
    return store().consume_daily_quota(action, limit)
