import sqlite3
from pathlib import Path

import pytest

from db.schema import migrate
from human_in_loop import (
    record_approval,
    record_correction,
    record_shadow_choice,
    review_field,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """Migrated DB with one run_events row and one shadow_eval row."""
    p = tmp_path / "test.db"
    migrate(str(p))
    conn = sqlite3.connect(p)
    conn.execute(
        "INSERT INTO run_events (id, run_id, field_label, value_used, confidence) "
        "VALUES (1, 'run-1', 'Salary expectation', '$120,000', 0.7)"
    )
    conn.execute(
        "INSERT INTO shadow_eval (id, field_label, platform, base_answer, ft_answer) "
        "VALUES (1, 'Salary expectation', 'greenhouse', '$120,000', '$130,000')"
    )
    conn.commit()
    conn.close()
    return p


def run_events_row(db: Path) -> dict:
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT is_correct, corrected_value, rejection_reason FROM run_events WHERE id=1"
    ).fetchone()
    conn.close()
    return {"is_correct": row[0], "corrected_value": row[1], "rejection_reason": row[2]}


def shadow_row(db: Path) -> dict:
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT human_chose FROM shadow_eval WHERE id=1"
    ).fetchone()
    conn.close()
    return {"human_chose": row[0]}


# ---------------------------------------------------------------------------
# record_approval
# ---------------------------------------------------------------------------

class TestRecordApproval:
    def test_sets_is_correct_1(self, db):
        record_approval(str(db), 1)
        assert run_events_row(db)["is_correct"] == 1

    def test_leaves_corrected_value_null(self, db):
        record_approval(str(db), 1)
        assert run_events_row(db)["corrected_value"] is None

    def test_idempotent(self, db):
        record_approval(str(db), 1)
        record_approval(str(db), 1)
        assert run_events_row(db)["is_correct"] == 1


# ---------------------------------------------------------------------------
# record_correction
# ---------------------------------------------------------------------------

class TestRecordCorrection:
    def test_sets_is_correct_0(self, db):
        record_correction(str(db), 1, "$125,000")
        assert run_events_row(db)["is_correct"] == 0

    def test_writes_corrected_value(self, db):
        record_correction(str(db), 1, "$125,000")
        assert run_events_row(db)["corrected_value"] == "$125,000"

    def test_writes_reason(self, db):
        record_correction(str(db), 1, "$125,000", reason="market rate")
        assert run_events_row(db)["rejection_reason"] == "market rate"

    def test_reason_none_when_omitted(self, db):
        record_correction(str(db), 1, "$125,000")
        assert run_events_row(db)["rejection_reason"] is None


# ---------------------------------------------------------------------------
# record_shadow_choice
# ---------------------------------------------------------------------------

class TestRecordShadowChoice:
    def test_writes_human_chose(self, db):
        record_shadow_choice(str(db), 1, "$130,000")
        assert shadow_row(db)["human_chose"] == "$130,000"

    def test_overwrites_previous_choice(self, db):
        record_shadow_choice(str(db), 1, "$120,000")
        record_shadow_choice(str(db), 1, "$130,000")
        assert shadow_row(db)["human_chose"] == "$130,000"


# ---------------------------------------------------------------------------
# review_field — two-way (no ft_answer)
# ---------------------------------------------------------------------------

class TestReviewFieldTwoWay:
    def test_enter_accepts_suggestion(self, db, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        result = review_field(1, "Salary", "$120,000", str(db))
        assert result == "$120,000"
        assert run_events_row(db)["is_correct"] == 1

    def test_y_accepts_suggestion(self, db, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        result = review_field(1, "Salary", "$120,000", str(db))
        assert result == "$120,000"
        assert run_events_row(db)["is_correct"] == 1

    def test_custom_value_sets_correction(self, db, monkeypatch):
        responses = iter(["$125,000", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = review_field(1, "Salary", "$120,000", str(db))
        assert result == "$125,000"
        row = run_events_row(db)
        assert row["is_correct"] == 0
        assert row["corrected_value"] == "$125,000"

    def test_custom_value_with_reason(self, db, monkeypatch):
        responses = iter(["$125,000", "market rate"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        review_field(1, "Salary", "$120,000", str(db))
        assert run_events_row(db)["rejection_reason"] == "market rate"

    def test_custom_value_empty_reason_stored_as_none(self, db, monkeypatch):
        responses = iter(["$125,000", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        review_field(1, "Salary", "$120,000", str(db))
        assert run_events_row(db)["rejection_reason"] is None


# ---------------------------------------------------------------------------
# review_field — three-way (with ft_answer)
# ---------------------------------------------------------------------------

class TestReviewFieldThreeWay:
    def _review(self, db, inputs):
        responses = iter(inputs)
        import builtins
        import unittest.mock as mock
        with mock.patch("builtins.input", side_effect=lambda _: next(responses)):
            return review_field(
                1, "Salary", "$120,000", str(db), ft_answer="$130,000"
            )

    def test_choice_a_returns_suggested(self, db, monkeypatch):
        responses = iter(["A"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = review_field(1, "Salary", "$120,000", str(db), ft_answer="$130,000")
        assert result == "$120,000"
        assert run_events_row(db)["is_correct"] == 1

    def test_empty_enter_acts_as_a(self, db, monkeypatch):
        responses = iter([""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = review_field(1, "Salary", "$120,000", str(db), ft_answer="$130,000")
        assert result == "$120,000"

    def test_choice_f_returns_ft_answer(self, db, monkeypatch):
        responses = iter(["F", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = review_field(1, "Salary", "$120,000", str(db), ft_answer="$130,000")
        assert result == "$130,000"
        row = run_events_row(db)
        assert row["is_correct"] == 0
        assert row["corrected_value"] == "$130,000"

    def test_choice_f_with_reason(self, db, monkeypatch):
        responses = iter(["F", "ft was better"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        review_field(1, "Salary", "$120,000", str(db), ft_answer="$130,000")
        assert run_events_row(db)["rejection_reason"] == "ft was better"

    def test_choice_c_returns_custom(self, db, monkeypatch):
        responses = iter(["C", "$135,000", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = review_field(1, "Salary", "$120,000", str(db), ft_answer="$130,000")
        assert result == "$135,000"
        row = run_events_row(db)
        assert row["is_correct"] == 0
        assert row["corrected_value"] == "$135,000"

    def test_invalid_choice_loops_until_valid(self, db, monkeypatch):
        responses = iter(["X", "Z", "A"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = review_field(1, "Salary", "$120,000", str(db), ft_answer="$130,000")
        assert result == "$120,000"

    def test_choice_c_empty_value_loops(self, db, monkeypatch):
        # First C attempt with empty value should re-prompt, second C succeeds
        responses = iter(["C", "", "C", "$135,000", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = review_field(1, "Salary", "$120,000", str(db), ft_answer="$130,000")
        assert result == "$135,000"
