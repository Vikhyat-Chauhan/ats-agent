import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from db.schema import migrate
from pipeline.shadow_eval import (
    PROMOTE_THRESHOLD,
    read_model_id,
    shadow_predict,
    win_rate_report,
)

MODEL_ID = "ft:gpt-4o-mini:job-agent:test001"
RESUME   = '{"name": "Jane Doe", "email": "jane@example.com"}'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "runs.db"
    migrate(str(p))
    return p


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    p = tmp_path / ".ft_model_id"
    p.write_text(MODEL_ID)
    return p


def _seed_shadow(db: Path, rows: list[dict]) -> None:
    """Insert rows into shadow_eval. Each dict must have base_answer, ft_answer, human_chose, platform."""
    conn = sqlite3.connect(db)
    for r in rows:
        conn.execute(
            """
            INSERT INTO shadow_eval (field_label, platform, base_answer, ft_answer, human_chose, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (r.get("field_label", "Email"), r["platform"],
             r["base_answer"], r["ft_answer"], r["human_chose"]),
        )
    conn.commit()
    conn.close()


def _make_openai_client(answer: str = "Jane Doe") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    return client


# ---------------------------------------------------------------------------
# read_model_id
# ---------------------------------------------------------------------------

class TestReadModelId:
    def test_reads_model_id(self, model_file):
        assert read_model_id(str(model_file)) == MODEL_ID

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_model_id(str(tmp_path / "missing"))

    def test_raises_if_file_empty(self, tmp_path):
        p = tmp_path / ".ft_model_id"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_model_id(str(p))

    def test_strips_trailing_newline(self, tmp_path):
        p = tmp_path / ".ft_model_id"
        p.write_text(MODEL_ID + "\n")
        assert read_model_id(str(p)) == MODEL_ID


# ---------------------------------------------------------------------------
# shadow_predict
# ---------------------------------------------------------------------------

class TestShadowPredict:
    def test_returns_ft_answer(self, db):
        client = _make_openai_client("Jane Doe")
        ft_answer, _ = shadow_predict(
            "First Name", RESUME, "greenhouse", str(db),
            model_id=MODEL_ID, client=client,
        )
        assert ft_answer == "Jane Doe"

    def test_returns_shadow_id(self, db):
        client = _make_openai_client("Jane Doe")
        _, shadow_id = shadow_predict(
            "First Name", RESUME, "greenhouse", str(db),
            model_id=MODEL_ID, client=client,
        )
        assert isinstance(shadow_id, int)
        assert shadow_id > 0

    def test_inserts_row_into_shadow_eval(self, db):
        client = _make_openai_client("Jane Doe")
        shadow_predict(
            "First Name", RESUME, "greenhouse", str(db),
            model_id=MODEL_ID, client=client,
        )
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT ft_answer, platform, field_label FROM shadow_eval").fetchone()
        conn.close()
        assert row == ("Jane Doe", "greenhouse", "First Name")

    def test_base_answer_is_null_initially(self, db):
        client = _make_openai_client("Jane Doe")
        shadow_predict(
            "First Name", RESUME, "greenhouse", str(db),
            model_id=MODEL_ID, client=client,
        )
        conn = sqlite3.connect(db)
        base = conn.execute("SELECT base_answer FROM shadow_eval").fetchone()[0]
        conn.close()
        assert base is None

    def test_strips_whitespace_from_answer(self, db):
        client = _make_openai_client("  Jane Doe  ")
        ft_answer, _ = shadow_predict(
            "First Name", RESUME, "greenhouse", str(db),
            model_id=MODEL_ID, client=client,
        )
        assert ft_answer == "Jane Doe"

    def test_calls_api_with_correct_model(self, db):
        client = _make_openai_client()
        shadow_predict(
            "Email", RESUME, "workday", str(db),
            model_id=MODEL_ID, client=client,
        )
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["model"] == MODEL_ID

    def test_calls_api_with_temperature_zero(self, db):
        client = _make_openai_client()
        shadow_predict(
            "Email", RESUME, "workday", str(db),
            model_id=MODEL_ID, client=client,
        )
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0

    def test_user_message_contains_platform(self, db):
        client = _make_openai_client()
        shadow_predict(
            "Email", RESUME, "icims", str(db),
            model_id=MODEL_ID, client=client,
        )
        messages = client.chat.completions.create.call_args[1]["messages"]
        user_content = messages[1]["content"]
        assert "icims" in user_content

    def test_user_message_contains_field_label(self, db):
        client = _make_openai_client()
        shadow_predict(
            "Salary Expectation", RESUME, "lever", str(db),
            model_id=MODEL_ID, client=client,
        )
        messages = client.chat.completions.create.call_args[1]["messages"]
        user_content = messages[1]["content"]
        assert "Salary Expectation" in user_content

    def test_shadow_ids_are_unique_across_calls(self, db):
        client = _make_openai_client()
        _, id1 = shadow_predict("Email", RESUME, "lever", str(db), model_id=MODEL_ID, client=client)
        _, id2 = shadow_predict("Phone", RESUME, "lever", str(db), model_id=MODEL_ID, client=client)
        assert id1 != id2


# ---------------------------------------------------------------------------
# win_rate_report — stats
# ---------------------------------------------------------------------------

class TestWinRateReport:
    def _seed_ft_wins(self, db: Path, n_ft: int, n_base: int, n_custom: int,
                      platform: str = "greenhouse") -> None:
        rows = (
            [{"platform": platform, "base_answer": "base", "ft_answer": "ft",
              "human_chose": "ft"}] * n_ft
            + [{"platform": platform, "base_answer": "base", "ft_answer": "ft",
                "human_chose": "base"}] * n_base
            + [{"platform": platform, "base_answer": "base", "ft_answer": "ft",
                "human_chose": "custom_val"}] * n_custom
        )
        _seed_shadow(db, rows)

    def test_total_reviewed(self, db):
        self._seed_ft_wins(db, 38, 10, 4)
        report = win_rate_report(str(db))
        assert report["total_reviewed"] == 52

    def test_ft_wins_count(self, db):
        self._seed_ft_wins(db, 38, 10, 4)
        report = win_rate_report(str(db))
        assert report["ft_wins"] == 38

    def test_base_wins_count(self, db):
        self._seed_ft_wins(db, 38, 10, 4)
        report = win_rate_report(str(db))
        assert report["base_wins"] == 10

    def test_human_custom_count(self, db):
        self._seed_ft_wins(db, 38, 10, 4)
        report = win_rate_report(str(db))
        assert report["human_custom"] == 4

    def test_ft_win_pct(self, db):
        self._seed_ft_wins(db, 38, 10, 4)
        report = win_rate_report(str(db))
        assert report["ft_win_pct"] == round(38 / 52 * 100, 1)

    def test_ready_to_promote_true_at_threshold(self, db):
        # exactly 70%: 7 ft wins out of 10
        self._seed_ft_wins(db, 7, 2, 1)
        report = win_rate_report(str(db))
        assert report["ready_to_promote"] is True

    def test_ready_to_promote_false_below_threshold(self, db):
        self._seed_ft_wins(db, 6, 3, 1)  # 60%
        report = win_rate_report(str(db))
        assert report["ready_to_promote"] is False

    def test_empty_db_returns_zero_totals(self, db):
        report = win_rate_report(str(db))
        assert report["total_reviewed"] == 0
        assert report["ft_win_pct"] == 0.0
        assert report["ready_to_promote"] is False

    def test_unreviewed_rows_excluded(self, db):
        # Insert one reviewed + one unreviewed row
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO shadow_eval (platform, base_answer, ft_answer, human_chose) "
            "VALUES ('greenhouse', 'base', 'ft', 'ft')"
        )
        conn.execute(
            "INSERT INTO shadow_eval (platform, base_answer, ft_answer, human_chose) "
            "VALUES ('greenhouse', 'base', 'ft', NULL)"
        )
        conn.commit()
        conn.close()
        report = win_rate_report(str(db))
        assert report["total_reviewed"] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pipeline.shadow_eval", *args],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )

    def _seed(self, db: Path, n_ft: int, n_base: int, n_custom: int) -> None:
        rows = (
            [{"platform": "greenhouse", "base_answer": "b", "ft_answer": "f",
              "human_chose": "f"}] * n_ft
            + [{"platform": "greenhouse", "base_answer": "b", "ft_answer": "f",
                "human_chose": "b"}] * n_base
            + [{"platform": "greenhouse", "base_answer": "b", "ft_answer": "f",
                "human_chose": "custom"}] * n_custom
        )
        _seed_shadow(db, rows)

    def test_exits_zero_when_ready_to_promote(self, db):
        self._seed(db, 38, 10, 4)   # 73.1% → ready
        result = self._run("--report", "--db", str(db))
        assert result.returncode == 0

    def test_exits_one_when_not_ready(self, db):
        self._seed(db, 6, 3, 1)     # 60% → not ready
        result = self._run("--report", "--db", str(db))
        assert result.returncode == 1

    def test_prints_report_table(self, db):
        self._seed(db, 38, 10, 4)
        result = self._run("--report", "--db", str(db))
        assert "Shadow Evaluation Report" in result.stdout

    def test_prints_ft_wins(self, db):
        self._seed(db, 38, 10, 4)
        result = self._run("--report", "--db", str(db))
        assert "38" in result.stdout

    def test_prints_ready_yes(self, db):
        self._seed(db, 38, 10, 4)
        result = self._run("--report", "--db", str(db))
        assert "YES" in result.stdout

    def test_prints_ready_no(self, db):
        self._seed(db, 6, 3, 1)
        result = self._run("--report", "--db", str(db))
        assert "NO" in result.stdout

    def test_missing_db_flag_exits_nonzero(self):
        result = self._run("--report")
        assert result.returncode != 0

    def test_no_args_exits_nonzero(self):
        result = self._run()
        assert result.returncode != 0
