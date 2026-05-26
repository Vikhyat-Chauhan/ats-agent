import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from db.schema import migrate
from pipeline.finetune_openai import (
    BASE_MODEL,
    FT_MODEL_ID_FILE,
    N_EPOCHS,
    SUFFIX,
    dry_run,
    run_finetune,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a job application assistant."


def _make_example(i: int) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Field: Email\nResume:\n{{\"name\": \"A\"}}\nWhat value?"},
            {"role": "assistant", "content": f"user{i}@example.com"},
        ]
    }


def _write_jsonl(path: Path, n: int) -> Path:
    path.write_text("\n".join(json.dumps(_make_example(i)) for i in range(n)))
    return path


@pytest.fixture(autouse=True)
def _isolate_ft_model_id_file(tmp_path: Path, monkeypatch):
    """Redirect all writes to .ft_model_id into a temp directory so tests never touch the real file."""
    monkeypatch.setattr("pipeline.finetune_openai.FT_MODEL_ID_FILE", str(tmp_path / ".ft_model_id"))


@pytest.fixture()
def jsonl_files(tmp_path: Path):
    train = _write_jsonl(tmp_path / "train.jsonl", 54)
    val   = _write_jsonl(tmp_path / "val.jsonl",    6)
    return train, val


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "runs.db"
    migrate(str(p))
    return p


def _make_client(status: str = "succeeded", model_id: str = "ft:gpt-4o-mini:job-agent:abc123"):
    """Build a mock OpenAI client that returns a completed job on the first poll."""
    client = MagicMock()

    # file uploads
    client.files.create.side_effect = [
        SimpleNamespace(id="file-train-001"),
        SimpleNamespace(id="file-val-001"),
    ]

    # job creation
    client.fine_tuning.jobs.create.return_value = SimpleNamespace(
        id="ftjob-001", status="running"
    )

    # job retrieval — first call "running", second call terminal
    event = SimpleNamespace(id="evt-1", message="Step 1/10")
    events_page = SimpleNamespace(data=[event])

    running_job  = SimpleNamespace(id="ftjob-001", status="running",  fine_tuned_model=None, error=None)
    terminal_job = SimpleNamespace(id="ftjob-001", status=status,     fine_tuned_model=model_id, error=None)

    client.fine_tuning.jobs.retrieve.side_effect = [running_job, terminal_job]
    client.fine_tuning.jobs.list_events.return_value = events_page

    return client


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_prints_train_path_and_count(self, jsonl_files, capsys):
        train, val = jsonl_files
        dry_run(str(train), str(val))
        out = capsys.readouterr().out
        assert str(train) in out
        assert "54" in out

    def test_prints_val_path_and_count(self, jsonl_files, capsys):
        train, val = jsonl_files
        dry_run(str(train), str(val))
        out = capsys.readouterr().out
        assert str(val) in out
        assert "6" in out

    def test_prints_base_model(self, jsonl_files, capsys):
        train, val = jsonl_files
        dry_run(str(train), str(val))
        assert BASE_MODEL in capsys.readouterr().out

    def test_prints_epochs(self, jsonl_files, capsys):
        train, val = jsonl_files
        dry_run(str(train), str(val))
        assert str(N_EPOCHS) in capsys.readouterr().out

    def test_prints_ft_model_id_file(self, jsonl_files, capsys):
        train, val = jsonl_files
        dry_run(str(train), str(val))
        assert FT_MODEL_ID_FILE in capsys.readouterr().out

    def test_raises_on_invalid_jsonl(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not json\n")
        good = _write_jsonl(tmp_path / "val.jsonl", 6)
        with pytest.raises(ValueError, match="not valid JSON"):
            dry_run(str(bad), str(good))

    def test_raises_on_missing_messages_key(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"wrong_key": []}) + "\n")
        good = _write_jsonl(tmp_path / "val.jsonl", 6)
        with pytest.raises(ValueError, match="missing 'messages'"):
            dry_run(str(bad), str(good))


# ---------------------------------------------------------------------------
# run_finetune — file upload
# ---------------------------------------------------------------------------

class TestFileUpload:
    def test_uploads_train_file(self, jsonl_files, db, tmp_path):
        train, val = jsonl_files
        client = _make_client()
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        assert client.files.create.call_count == 2

    def test_train_uploaded_with_fine_tune_purpose(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client()
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        first_call_kwargs = client.files.create.call_args_list[0][1]
        assert first_call_kwargs["purpose"] == "fine-tune"

    def test_val_uploaded_with_fine_tune_purpose(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client()
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        second_call_kwargs = client.files.create.call_args_list[1][1]
        assert second_call_kwargs["purpose"] == "fine-tune"


# ---------------------------------------------------------------------------
# run_finetune — job creation
# ---------------------------------------------------------------------------

class TestJobCreation:
    def test_creates_job_with_correct_model(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client()
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        kwargs = client.fine_tuning.jobs.create.call_args[1]
        assert kwargs["model"] == BASE_MODEL

    def test_creates_job_with_correct_epochs(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client()
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        kwargs = client.fine_tuning.jobs.create.call_args[1]
        assert kwargs["hyperparameters"]["n_epochs"] == N_EPOCHS

    def test_creates_job_with_suffix(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client()
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        kwargs = client.fine_tuning.jobs.create.call_args[1]
        assert kwargs["suffix"] == SUFFIX

    def test_passes_uploaded_file_ids_to_job(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client()
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        kwargs = client.fine_tuning.jobs.create.call_args[1]
        assert kwargs["training_file"] == "file-train-001"
        assert kwargs["validation_file"] == "file-val-001"


# ---------------------------------------------------------------------------
# run_finetune — success path
# ---------------------------------------------------------------------------

class TestSuccessPath:
    MODEL_ID = "ft:gpt-4o-mini:job-agent:abc123"

    def test_returns_model_id(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client(model_id=self.MODEL_ID)
        with patch("pipeline.finetune_openai.time.sleep"):
            result = run_finetune(str(train), str(val), str(db), client=client)
        assert result == self.MODEL_ID

    def test_writes_model_id_to_file(self, jsonl_files, db):
        import pipeline.finetune_openai as ft_mod
        train, val = jsonl_files
        client = _make_client(model_id=self.MODEL_ID)
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        assert Path(ft_mod.FT_MODEL_ID_FILE).read_text() == self.MODEL_ID

    def test_writes_model_registry_row(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client(model_id=self.MODEL_ID)
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT model_id, train_examples FROM model_registry"
        ).fetchone()
        conn.close()
        assert row[0] == self.MODEL_ID
        assert row[1] == 54  # lines in train.jsonl

    def test_model_registry_last_trained_at_set(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client(model_id=self.MODEL_ID)
        with patch("pipeline.finetune_openai.time.sleep"):
            run_finetune(str(train), str(val), str(db), client=client)
        conn = sqlite3.connect(db)
        ts = conn.execute("SELECT last_trained_at FROM model_registry").fetchone()[0]
        conn.close()
        assert ts is not None


# ---------------------------------------------------------------------------
# run_finetune — failure path
# ---------------------------------------------------------------------------

class TestFailurePath:
    def test_raises_runtime_error_on_failed_job(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client(status="failed", model_id=None)
        # Replace side_effect list so we can attach an error object to the terminal job
        running_job  = SimpleNamespace(id="ftjob-001", status="running",  fine_tuned_model=None, error=None)
        terminal_job = SimpleNamespace(id="ftjob-001", status="failed",   fine_tuned_model=None,
                                       error=SimpleNamespace(message="Training diverged"))
        client.fine_tuning.jobs.retrieve.side_effect = [running_job, terminal_job]
        with patch("pipeline.finetune_openai.time.sleep"):
            with pytest.raises(RuntimeError, match="failed"):
                run_finetune(str(train), str(val), str(db), client=client)

    def test_no_model_registry_row_on_failure(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client(status="failed", model_id=None)
        with patch("pipeline.finetune_openai.time.sleep"):
            with pytest.raises(RuntimeError):
                run_finetune(str(train), str(val), str(db), client=client)
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0]
        conn.close()
        assert count == 0

    def test_raises_runtime_error_on_cancelled_job(self, jsonl_files, db):
        train, val = jsonl_files
        client = _make_client(status="cancelled", model_id=None)
        with patch("pipeline.finetune_openai.time.sleep"):
            with pytest.raises(RuntimeError, match="cancelled"):
                run_finetune(str(train), str(val), str(db), client=client)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pipeline.finetune_openai", *args],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )

    def test_dry_run_exits_zero(self, jsonl_files):
        train, val = jsonl_files
        result = self._run("--train", str(train), "--val", str(val),
                           "--db", "irrelevant.db", "--dry-run")
        assert result.returncode == 0

    def test_dry_run_prints_dry_run_prefix(self, jsonl_files):
        train, val = jsonl_files
        result = self._run("--train", str(train), "--val", str(val),
                           "--db", "irrelevant.db", "--dry-run")
        assert "[DRY RUN]" in result.stdout

    def test_dry_run_shows_train_count(self, jsonl_files):
        train, val = jsonl_files
        result = self._run("--train", str(train), "--val", str(val),
                           "--db", "irrelevant.db", "--dry-run")
        assert "54" in result.stdout

    def test_dry_run_shows_val_count(self, jsonl_files):
        train, val = jsonl_files
        result = self._run("--train", str(train), "--val", str(val),
                           "--db", "irrelevant.db", "--dry-run")
        assert "6" in result.stdout

    def test_dry_run_shows_model(self, jsonl_files):
        train, val = jsonl_files
        result = self._run("--train", str(train), "--val", str(val),
                           "--db", "irrelevant.db", "--dry-run")
        assert BASE_MODEL in result.stdout

    def test_missing_args_exits_nonzero(self):
        result = self._run()
        assert result.returncode != 0
