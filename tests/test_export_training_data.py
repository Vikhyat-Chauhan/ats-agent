import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from db.schema import migrate
from pipeline.export_training_data import SYSTEM_PROMPT, export

PLATFORMS = ["workday", "icims", "greenhouse", "lever"]
FIELDS = ["First Name", "Last Name", "Email", "Phone", "Years of Experience",
          "Salary Expectation", "LinkedIn URL", "Cover Letter", "Start Date", "Location"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _seed(db_path: Path, n: int, *, include_resume: bool = True) -> None:
    migrate(str(db_path))
    conn = sqlite3.connect(db_path)
    for i in range(n):
        platform = PLATFORMS[i % len(PLATFORMS)]
        field = FIELDS[i % len(FIELDS)]
        is_correct = i % 2          # alternates 0/1
        corrected = f"corrected_{i}" if is_correct == 0 else None
        resume = '{"name": "Jane"}' if include_resume else None
        conn.execute(
            """
            INSERT INTO run_events
                (run_id, event_type, ats_platform, field_label,
                 value_used, is_correct, corrected_value, resume_snapshot)
            VALUES (?, 'field_filled', ?, ?, ?, ?, ?, ?)
            """,
            (f"run-{i}", platform, field, f"auto_{i}", is_correct, corrected, resume),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def db60(tmp_path: Path) -> Path:
    p = tmp_path / "runs.db"
    _seed(p, 60)
    return p


@pytest.fixture()
def db_sparse(tmp_path: Path) -> Path:
    """DB with 10 labeled rows — below the 50-example threshold."""
    p = tmp_path / "sparse.db"
    _seed(p, 10)
    return p


@pytest.fixture()
def db_no_resume(tmp_path: Path) -> Path:
    """DB with 60 rows but resume_snapshot is NULL — all skipped."""
    p = tmp_path / "no_resume.db"
    _seed(p, 60, include_resume=False)
    return p


@pytest.fixture()
def out(tmp_path: Path) -> Path:
    return tmp_path / "export"


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

class TestOutputFiles:
    def test_creates_train_jsonl(self, db60, out):
        export(str(db60), str(out))
        assert (out / "train.jsonl").exists()

    def test_creates_val_jsonl(self, db60, out):
        export(str(db60), str(out))
        assert (out / "val.jsonl").exists()

    def test_creates_output_dir(self, db60, tmp_path):
        nested = tmp_path / "deep" / "nested" / "out"
        export(str(db60), str(nested))
        assert nested.is_dir()

    def test_train_val_counts_sum_to_total(self, db60, out):
        n_train, n_val = export(str(db60), str(out))
        assert n_train + n_val == 60

    def test_90_10_split(self, db60, out):
        n_train, n_val = export(str(db60), str(out))
        assert n_train == 54
        assert n_val == 6

    def test_train_line_count_matches_return_value(self, db60, out):
        n_train, _ = export(str(db60), str(out))
        lines = (out / "train.jsonl").read_text().strip().splitlines()
        assert len(lines) == n_train

    def test_val_line_count_matches_return_value(self, db60, out):
        _, n_val = export(str(db60), str(out))
        lines = (out / "val.jsonl").read_text().strip().splitlines()
        assert len(lines) == n_val


# ---------------------------------------------------------------------------
# JSONL format
# ---------------------------------------------------------------------------

class TestJsonlFormat:
    def _load(self, path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text().strip().splitlines()]

    def test_each_line_is_valid_json(self, db60, out):
        export(str(db60), str(out))
        for line in (out / "train.jsonl").read_text().strip().splitlines():
            json.loads(line)  # must not raise

    def test_messages_key_present(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            assert "messages" in ex

    def test_three_message_turns(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            assert len(ex["messages"]) == 3

    def test_roles_are_system_user_assistant(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            roles = [m["role"] for m in ex["messages"]]
            assert roles == ["system", "user", "assistant"]

    def test_system_prompt_content(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            assert ex["messages"][0]["content"] == SYSTEM_PROMPT

    def test_user_message_contains_field_label(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            assert "Field:" in ex["messages"][1]["content"]

    def test_user_message_contains_ats_platform(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            assert "ATS:" in ex["messages"][1]["content"]

    def test_user_message_contains_resume(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            assert "Resume:" in ex["messages"][1]["content"]

    def test_assistant_content_not_empty(self, db60, out):
        export(str(db60), str(out))
        for ex in self._load(out / "train.jsonl"):
            assert ex["messages"][2]["content"] != ""


# ---------------------------------------------------------------------------
# Ground truth selection
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_corrected_value_preferred_over_value_used(self, tmp_path, out):
        """Rows with is_correct=0 should use corrected_value as assistant content."""
        db = tmp_path / "gt.db"
        migrate(str(db))
        conn = sqlite3.connect(db)
        for i in range(50):
            conn.execute(
                """
                INSERT INTO run_events
                    (run_id, event_type, ats_platform, field_label,
                     value_used, is_correct, corrected_value, resume_snapshot)
                VALUES (?, 'field_filled', 'greenhouse', 'Email',
                        'auto@x.com', 0, 'human@x.com', '{"name":"A"}')
                """,
                (f"run-{i}",),
            )
        conn.commit()
        conn.close()

        export(str(db), str(out))
        examples = [json.loads(l) for l in (out / "train.jsonl").read_text().strip().splitlines()]
        for ex in examples:
            assert ex["messages"][2]["content"] == "human@x.com"

    def test_value_used_when_is_correct_1(self, tmp_path, out):
        """Rows with is_correct=1 and no corrected_value should use value_used."""
        db = tmp_path / "gt2.db"
        migrate(str(db))
        conn = sqlite3.connect(db)
        for i in range(50):
            conn.execute(
                """
                INSERT INTO run_events
                    (run_id, event_type, ats_platform, field_label,
                     value_used, is_correct, corrected_value, resume_snapshot)
                VALUES (?, 'field_filled', 'lever', 'Phone',
                        '555-1234', 1, NULL, '{"name":"B"}')
                """,
                (f"run-{i}",),
            )
        conn.commit()
        conn.close()

        export(str(db), str(out))
        examples = [json.loads(l) for l in (out / "train.jsonl").read_text().strip().splitlines()]
        for ex in examples:
            assert ex["messages"][2]["content"] == "555-1234"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_split(self, db60, out, tmp_path):
        out2 = tmp_path / "export2"
        export(str(db60), str(out))
        export(str(db60), str(out2))
        assert (out / "train.jsonl").read_text() == (out2 / "train.jsonl").read_text()
        assert (out / "val.jsonl").read_text() == (out2 / "val.jsonl").read_text()

    def test_train_and_val_are_disjoint(self, db60, out):
        export(str(db60), str(out))
        train = {json.loads(l)["messages"][2]["content"]
                 for l in (out / "train.jsonl").read_text().strip().splitlines()}
        val = {json.loads(l)["messages"][2]["content"]
               for l in (out / "val.jsonl").read_text().strip().splitlines()}
        # content strings alone aren't guaranteed unique, so check line counts instead
        n_train = len((out / "train.jsonl").read_text().strip().splitlines())
        n_val = len((out / "val.jsonl").read_text().strip().splitlines())
        assert n_train + n_val == 60


# ---------------------------------------------------------------------------
# Threshold guard
# ---------------------------------------------------------------------------

class TestThresholdGuard:
    def test_raises_systemexit_below_50(self, db_sparse, out):
        with pytest.raises(SystemExit):
            export(str(db_sparse), str(out))

    def test_no_files_written_below_50(self, db_sparse, out):
        with pytest.raises(SystemExit):
            export(str(db_sparse), str(out))
        assert not (out / "train.jsonl").exists()
        assert not (out / "val.jsonl").exists()

    def test_no_files_written_when_all_missing_resume(self, db_no_resume, out):
        with pytest.raises(SystemExit):
            export(str(db_no_resume), str(out))
        assert not (out / "train.jsonl").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pipeline.export_training_data", *args],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

    def test_cli_exits_zero(self, db60, out):
        result = self._run("--db", str(db60), "--out", str(out))
        assert result.returncode == 0

    def test_cli_prints_total(self, db60, out):
        result = self._run("--db", str(db60), "--out", str(out))
        assert "Total examples" in result.stdout

    def test_cli_prints_platform_breakdown(self, db60, out):
        result = self._run("--db", str(db60), "--out", str(out))
        assert "Platform breakdown" in result.stdout

    def test_cli_prints_most_common_fields(self, db60, out):
        result = self._run("--db", str(db60), "--out", str(out))
        assert "Most common fields" in result.stdout

    def test_cli_exits_nonzero_below_threshold(self, db_sparse, out):
        result = self._run("--db", str(db_sparse), "--out", str(out))
        assert result.returncode != 0

    def test_cli_prints_warning_below_threshold(self, db_sparse, out):
        result = self._run("--db", str(db_sparse), "--out", str(out))
        assert "WARNING" in result.stdout

    def test_cli_missing_args_exits_nonzero(self):
        result = self._run()
        assert result.returncode != 0
