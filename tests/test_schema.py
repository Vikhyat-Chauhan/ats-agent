import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from db.schema import migrate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def col_default(conn: sqlite3.Connection, table: str, col: str) -> str | None:
    for row in conn.execute(f"PRAGMA table_info({table})"):
        if row[1] == col:
            return row[4]  # dflt_value
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh_db(tmp_path: Path) -> Path:
    """Return path to a brand-new empty database."""
    return tmp_path / "test.db"


@pytest.fixture()
def seeded_db(tmp_path: Path) -> Path:
    """Return path to a DB that already has run_events with the original schema."""
    db_path = tmp_path / "seeded.db"
    conn = sqlite3.connect(db_path)
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
    conn.execute(
        "INSERT INTO run_events (run_id, event_type) VALUES (?, ?)",
        ("run-1", "field_filled"),
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# run_events column migration
# ---------------------------------------------------------------------------

class TestRunEventsColumns:
    NEW_COLUMNS = {"is_correct", "corrected_value", "rejection_reason", "resume_snapshot"}

    def test_adds_all_new_columns_to_fresh_db(self, fresh_db):
        migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        assert self.NEW_COLUMNS.issubset(columns(conn, "run_events"))
        conn.close()

    def test_adds_new_columns_to_existing_run_events(self, seeded_db):
        migrate(str(seeded_db))
        conn = sqlite3.connect(seeded_db)
        assert self.NEW_COLUMNS.issubset(columns(conn, "run_events"))
        conn.close()

    def test_preserves_existing_data_after_migration(self, seeded_db):
        migrate(str(seeded_db))
        conn = sqlite3.connect(seeded_db)
        row = conn.execute("SELECT run_id, event_type FROM run_events").fetchone()
        assert row == ("run-1", "field_filled")
        conn.close()

    def test_new_rows_default_to_null(self, seeded_db):
        migrate(str(seeded_db))
        conn = sqlite3.connect(seeded_db)
        conn.execute(
            "INSERT INTO run_events (run_id, event_type) VALUES (?, ?)",
            ("run-2", "submitted"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT is_correct, corrected_value, rejection_reason, resume_snapshot "
            "FROM run_events WHERE run_id='run-2'"
        ).fetchone()
        assert row == (None, None, None, None)
        conn.close()

    @pytest.mark.parametrize("col", NEW_COLUMNS)
    def test_column_default_is_null(self, fresh_db, col):
        # PRAGMA table_info returns the string 'NULL' for DEFAULT NULL columns
        migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        assert col_default(conn, "run_events", col) in (None, "NULL")
        conn.close()


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestTablesCreated:
    @pytest.mark.parametrize("table", ["shadow_eval", "model_registry"])
    def test_table_exists_after_migrate(self, fresh_db, table):
        migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        assert table in tables(conn)
        conn.close()

    def test_shadow_eval_columns(self, fresh_db):
        migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        expected = {"id", "field_label", "platform", "base_answer", "ft_answer", "human_chose", "timestamp"}
        assert columns(conn, "shadow_eval") == expected
        conn.close()

    def test_model_registry_columns(self, fresh_db):
        migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        expected = {"id", "model_id", "last_trained_at", "train_examples", "ft_win_pct"}
        assert columns(conn, "model_registry") == expected
        conn.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_double_migrate_fresh_db(self, fresh_db):
        migrate(str(fresh_db))
        migrate(str(fresh_db))  # must not raise

    def test_double_migrate_seeded_db(self, seeded_db):
        migrate(str(seeded_db))
        migrate(str(seeded_db))  # must not raise

    def test_schema_unchanged_after_second_run(self, fresh_db):
        migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        schema_first = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        }
        conn.close()

        migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        schema_second = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        }
        conn.close()

        assert schema_first == schema_second

    def test_ten_migrations_are_stable(self, fresh_db):
        for _ in range(10):
            migrate(str(fresh_db))
        conn = sqlite3.connect(fresh_db)
        assert "shadow_eval" in tables(conn)
        assert "model_registry" in tables(conn)
        conn.close()


# ---------------------------------------------------------------------------
# Summary return value
# ---------------------------------------------------------------------------

class TestMigrationSummary:
    def test_fresh_db_reports_all_created(self, fresh_db):
        summary = migrate(str(fresh_db))
        assert "run_events.is_correct" in summary.columns_added
        assert "run_events.corrected_value" in summary.columns_added
        assert "shadow_eval" in summary.tables_created
        assert "model_registry" in summary.tables_created
        assert not summary.columns_skipped
        # run_events itself may appear in tables_created when starting from scratch
        assert not summary.tables_skipped

    def test_second_run_reports_all_skipped(self, fresh_db):
        migrate(str(fresh_db))
        summary = migrate(str(fresh_db))
        assert not summary.columns_added
        assert not summary.tables_created
        assert "run_events.is_correct" in summary.columns_skipped
        assert "shadow_eval" in summary.tables_skipped
        assert "model_registry" in summary.tables_skipped

    def test_seeded_db_columns_added_tables_created(self, seeded_db):
        summary = migrate(str(seeded_db))
        assert set(summary.columns_added) == {
            "run_events.is_correct",
            "run_events.corrected_value",
            "run_events.rejection_reason",
            "run_events.resume_snapshot",
        }
        assert "shadow_eval" in summary.tables_created
        assert "model_registry" in summary.tables_created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "db.schema", *args],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

    def test_cli_exits_zero(self, fresh_db):
        result = self._run("--db", str(fresh_db))
        assert result.returncode == 0

    def test_cli_prints_tables_created(self, fresh_db):
        result = self._run("--db", str(fresh_db))
        assert "shadow_eval" in result.stdout
        assert "model_registry" in result.stdout

    def test_cli_prints_columns_added(self, fresh_db):
        result = self._run("--db", str(fresh_db))
        assert "is_correct" in result.stdout

    def test_cli_second_run_prints_skipped(self, fresh_db):
        self._run("--db", str(fresh_db))
        result = self._run("--db", str(fresh_db))
        assert "skipped" in result.stdout.lower()
        assert result.returncode == 0

    def test_cli_missing_db_flag_exits_nonzero(self):
        result = self._run()
        assert result.returncode != 0
