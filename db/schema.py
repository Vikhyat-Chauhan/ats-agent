"""
Database migration module for the ATS agent fine-tuning pipeline.

Usage:
    python -m db.schema --db runs.db
"""

import argparse
import sqlite3
from dataclasses import dataclass, field
from typing import List


@dataclass
class MigrationSummary:
    columns_added: List[str] = field(default_factory=list)
    columns_skipped: List[str] = field(default_factory=list)
    tables_created: List[str] = field(default_factory=list)
    tables_skipped: List[str] = field(default_factory=list)


_RUN_EVENTS_NEW_COLUMNS = [
    ("is_correct",       "INTEGER DEFAULT NULL"),
    ("corrected_value",  "TEXT DEFAULT NULL"),
    ("rejection_reason", "TEXT DEFAULT NULL"),
    ("resume_snapshot",  "TEXT DEFAULT NULL"),
]


def _existing_columns(conn: sqlite3.Connection, table: str) -> set:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


def _migrate_run_events_columns(conn: sqlite3.Connection, summary: MigrationSummary) -> None:
    if not _table_exists(conn, "run_events"):
        conn.execute("""
            CREATE TABLE run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TEXT,
                event_type TEXT,
                ats_platform TEXT,
                job_url TEXT,
                field_id TEXT,
                field_label TEXT,
                value_used TEXT,
                confidence REAL,
                source TEXT,
                error TEXT
            )
        """)
        summary.tables_created.append("run_events")

    existing = _existing_columns(conn, "run_events")
    for col_name, col_def in _RUN_EVENTS_NEW_COLUMNS:
        if col_name in existing:
            summary.columns_skipped.append(f"run_events.{col_name}")
        else:
            conn.execute(f"ALTER TABLE run_events ADD COLUMN {col_name} {col_def}")
            summary.columns_added.append(f"run_events.{col_name}")


def _create_shadow_eval(conn: sqlite3.Connection, summary: MigrationSummary) -> None:
    if _table_exists(conn, "shadow_eval"):
        summary.tables_skipped.append("shadow_eval")
        return
    conn.execute("""
        CREATE TABLE shadow_eval (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_label TEXT,
            platform TEXT,
            base_answer TEXT,
            ft_answer TEXT,
            human_chose TEXT,
            timestamp TEXT
        )
    """)
    summary.tables_created.append("shadow_eval")


def _create_model_registry(conn: sqlite3.Connection, summary: MigrationSummary) -> None:
    if _table_exists(conn, "model_registry"):
        summary.tables_skipped.append("model_registry")
        return
    conn.execute("""
        CREATE TABLE model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            last_trained_at TEXT,
            train_examples INTEGER,
            ft_win_pct REAL
        )
    """)
    summary.tables_created.append("model_registry")


def migrate(db_path: str) -> MigrationSummary:
    """Run all migrations against the database at db_path. Safe to call repeatedly."""
    summary = MigrationSummary()
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            _migrate_run_events_columns(conn, summary)
            _create_shadow_eval(conn, summary)
            _create_model_registry(conn, summary)
    finally:
        conn.close()
    return summary


def _print_summary(summary: MigrationSummary) -> None:
    if summary.tables_created:
        print(f"Tables created  : {', '.join(summary.tables_created)}")
    if summary.tables_skipped:
        print(f"Tables skipped  : {', '.join(summary.tables_skipped)} (already exist)")
    if summary.columns_added:
        print(f"Columns added   : {', '.join(summary.columns_added)}")
    if summary.columns_skipped:
        print(f"Columns skipped : {', '.join(summary.columns_skipped)} (already exist)")
    if not any([
        summary.tables_created, summary.tables_skipped,
        summary.columns_added, summary.columns_skipped,
    ]):
        print("Nothing to do — schema is already up to date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate ATS agent database schema.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file")
    args = parser.parse_args()

    summary = migrate(args.db)
    _print_summary(summary)
