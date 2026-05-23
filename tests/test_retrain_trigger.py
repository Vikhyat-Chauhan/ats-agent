import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from db.schema import migrate
from pipeline.finetune_openai import FT_MODEL_ID_FILE
from pipeline.retrain_trigger import (
    DEFAULT_THRESHOLD,
    _count_new_examples,
    _last_trained_at,
    dry_run,
    retrain,
)

OLD_MODEL = "ft:gpt-4o-mini:job-agent:old001"
NEW_MODEL = "ft:gpt-4o-mini:job-agent:new002"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "runs.db"
    migrate(str(p))
    return p


def _seed_labeled(db: Path, n: int, since: str | None = None) -> None:
    conn = sqlite3.connect(db)
    for i in range(n):
        ts = since if since else "2026-01-01T00:00:00"
        conn.execute(
            """
            INSERT INTO run_events
                (run_id, event_type, ats_platform, field_label,
                 value_used, is_correct, resume_snapshot, timestamp)
            VALUES (?, 'field_filled', 'greenhouse', 'Email',
                    'x@x.com', 1, '{"name":"A"}', ?)
            """,
            (f"run-{i}", ts),
        )
    conn.commit()
    conn.close()


def _seed_model_registry(db: Path, model_id: str, last_trained_at: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO model_registry (model_id, last_trained_at, train_examples) VALUES (?,?,?)",
            (model_id, last_trained_at, 50),
        )


def _make_retrain_mocks(promoted: bool = True):
    """Return (mock_export, mock_finetune, mock_report) patch targets."""
    report = {
        "total_reviewed": 20, "ft_wins": 15, "base_wins": 4, "human_custom": 1,
        "ft_win_pct": 75.0 if promoted else 60.0,
        "ready_to_promote": promoted,
    }
    return report


# ---------------------------------------------------------------------------
# _last_trained_at
# ---------------------------------------------------------------------------

class TestLastTrainedAt:
    def test_returns_none_when_no_registry_row(self, db):
        assert _last_trained_at(str(db)) is None

    def test_returns_latest_timestamp(self, db):
        _seed_model_registry(db, OLD_MODEL, "2026-05-10T14:22:00")
        assert _last_trained_at(str(db)) == "2026-05-10T14:22:00"

    def test_returns_most_recent_when_multiple_rows(self, db):
        _seed_model_registry(db, OLD_MODEL, "2026-01-01T00:00:00")
        _seed_model_registry(db, NEW_MODEL, "2026-05-10T14:22:00")
        assert _last_trained_at(str(db)) == "2026-05-10T14:22:00"


# ---------------------------------------------------------------------------
# _count_new_examples
# ---------------------------------------------------------------------------

class TestCountNewExamples:
    def test_counts_all_when_no_since(self, db):
        _seed_labeled(db, 10)
        assert _count_new_examples(str(db), None) == 10

    def test_counts_only_after_since(self, db):
        _seed_labeled(db, 5, since="2026-01-01T00:00:00")
        _seed_labeled(db, 3, since="2026-06-01T00:00:00")
        assert _count_new_examples(str(db), "2026-05-01T00:00:00") == 3

    def test_returns_zero_when_no_rows(self, db):
        assert _count_new_examples(str(db), None) == 0

    def test_excludes_unlabeled_rows(self, db):
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO run_events (run_id, event_type, ats_platform, field_label, "
            "value_used, is_correct, resume_snapshot, timestamp) "
            "VALUES ('r1', 'field_filled', 'greenhouse', 'Email', 'x', NULL, '{\"a\":1}', '2026-01-01')"
        )
        conn.commit()
        conn.close()
        assert _count_new_examples(str(db), None) == 0


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_prints_example_count(self, db, capsys):
        _seed_labeled(db, 12)
        dry_run(str(db), 10)
        assert "12" in capsys.readouterr().out

    def test_prints_would_retrain_when_above_threshold(self, db, capsys):
        _seed_labeled(db, 12)
        dry_run(str(db), 10)
        assert "Would export" in capsys.readouterr().out

    def test_prints_no_retrain_when_below_threshold(self, db, capsys):
        _seed_labeled(db, 3)
        dry_run(str(db), 10)
        out = capsys.readouterr().out
        assert "Would export" not in out
        assert "3" in out

    def test_prints_since_timestamp_when_registry_exists(self, db, capsys):
        _seed_model_registry(db, OLD_MODEL, "2026-05-10T14:22:00")
        _seed_labeled(db, 5, since="2026-06-01T00:00:00")
        dry_run(str(db), 4)
        assert "2026-05-10T14:22:00" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# retrain — below threshold
# ---------------------------------------------------------------------------

class TestRetrainBelowThreshold:
    def test_returns_false(self, db):
        _seed_labeled(db, 3)
        result = retrain(str(db), threshold=10)
        assert result is False

    def test_prints_no_retrain_message(self, db, capsys):
        _seed_labeled(db, 3)
        retrain(str(db), threshold=10)
        assert "No retrain" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# retrain — promotion path
# ---------------------------------------------------------------------------

class TestRetrainPromotionPath:
    def _run(self, db, tmp_path, promoted: bool):
        report = _make_retrain_mocks(promoted=promoted)
        out_dir = str(tmp_path / "export")
        with (
            patch("pipeline.retrain_trigger.export") as mock_export,
            patch("pipeline.retrain_trigger.run_finetune", return_value=NEW_MODEL) as mock_ft,
            patch("pipeline.retrain_trigger.win_rate_report", return_value=report),
            patch("pipeline.retrain_trigger._update_ft_win_pct"),
        ):
            _seed_labeled(db, 110)
            result = retrain(str(db), threshold=100, out_dir=out_dir)
        return result, mock_export, mock_ft

    def test_returns_true_on_promotion(self, db, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result, _, _ = self._run(db, tmp_path, promoted=True)
        assert result is True

    def test_writes_ft_model_id_on_promotion(self, db, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._run(db, tmp_path, promoted=True)
        assert Path(FT_MODEL_ID_FILE).read_text() == NEW_MODEL

    def test_prints_promoted_message(self, db, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        self._run(db, tmp_path, promoted=True)
        assert "Promoted new model" in capsys.readouterr().out

    def test_returns_false_when_not_promoted(self, db, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result, _, _ = self._run(db, tmp_path, promoted=False)
        assert result is False

    def test_prints_keeping_current_when_not_promoted(self, db, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        self._run(db, tmp_path, promoted=False)
        assert "Keeping current" in capsys.readouterr().out

    def test_does_not_overwrite_model_id_when_not_promoted(self, db, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        old_id_file = tmp_path / FT_MODEL_ID_FILE
        old_id_file.write_text(OLD_MODEL)
        self._run(db, tmp_path, promoted=False)
        assert old_id_file.read_text() == OLD_MODEL

    def test_export_called_with_db_path(self, db, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _, mock_export, _ = self._run(db, tmp_path, promoted=True)
        args = mock_export.call_args[0]
        assert str(db) in args

    def test_finetune_called_after_export(self, db, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _, mock_export, mock_ft = self._run(db, tmp_path, promoted=True)
        assert mock_export.call_count == 1
        assert mock_ft.call_count == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pipeline.retrain_trigger", *args],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )

    def test_dry_run_exits_zero(self, db):
        result = self._run("--db", str(db), "--threshold", "5", "--dry-run")
        assert result.returncode == 0

    def test_dry_run_prints_dry_run_prefix(self, db):
        result = self._run("--db", str(db), "--threshold", "5", "--dry-run")
        assert "[DRY RUN]" in result.stdout

    def test_dry_run_shows_example_count(self, db, tmp_path):
        # seed via direct DB write so we don't need to call _seed_labeled here
        conn = sqlite3.connect(db)
        for i in range(12):
            conn.execute(
                "INSERT INTO run_events (run_id, event_type, ats_platform, field_label, "
                "value_used, is_correct, resume_snapshot, timestamp) "
                "VALUES (?, 'field_filled', 'greenhouse', 'Email', 'x', 1, '{\"a\":1}', '2026-01-01')",
                (f"r{i}",),
            )
        conn.commit()
        conn.close()
        result = self._run("--db", str(db), "--threshold", "5", "--dry-run")
        assert "12" in result.stdout

    def test_missing_db_arg_exits_nonzero(self):
        result = self._run("--dry-run")
        assert result.returncode != 0
