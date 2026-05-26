"""
Upload training data and run an OpenAI fine-tuning job.

Usage:
    python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db
    python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db --dry-run
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

BASE_MODEL = "gpt-4.1-mini-2025-04-14"
N_EPOCHS = 3
SUFFIX = "job-agent"
POLL_INTERVAL = 30
FT_MODEL_ID_FILE = ".ft_model_id"

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _validate_jsonl(path: Path) -> None:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {i} is not valid JSON: {exc}") from exc
        if "messages" not in obj:
            raise ValueError(f"{path}: line {i} missing 'messages' key")


def _upload_file(client: openai.OpenAI, path: Path) -> str:
    with open(path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    return response.id


def _write_model_registry(db_path: str, model_id: str, train_examples: int) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO model_registry (model_id, last_trained_at, train_examples) "
            "VALUES (?, datetime('now'), ?)",
            (model_id, train_examples),
        )


def _save_model_id(model_id: str) -> None:
    Path(FT_MODEL_ID_FILE).write_text(model_id)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(train_path: str, val_path: str) -> None:
    train = Path(train_path)
    val = Path(val_path)

    _validate_jsonl(train)
    _validate_jsonl(val)

    n_train = _count_jsonl(train)
    n_val = _count_jsonl(val)

    print(f"[DRY RUN] Would upload: {train_path} ({n_train} examples)")
    print(f"[DRY RUN] Would upload: {val_path} ({n_val} examples)")
    print(f"[DRY RUN] Would start fine-tune job on {BASE_MODEL} for {N_EPOCHS} epochs")
    print(f"[DRY RUN] Would write model ID to {FT_MODEL_ID_FILE} and model_registry")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_finetune(
    train_path: str,
    val_path: str,
    db_path: str,
    *,
    client: openai.OpenAI | None = None,
) -> str:
    """
    Upload files, start a fine-tune job, poll until done, persist results.
    Returns the fine_tuned_model ID on success.
    """
    if client is None:
        client = openai.OpenAI()

    train = Path(train_path)
    val = Path(val_path)

    print("Uploading training file…")
    train_file_id = _upload_file(client, train)
    print(f"  train file id: {train_file_id}")

    print("Uploading validation file…")
    val_file_id = _upload_file(client, val)
    print(f"  val   file id: {val_file_id}")

    print(f"Creating fine-tune job (base={BASE_MODEL}, epochs={N_EPOCHS})…")
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=BASE_MODEL,
        hyperparameters={"n_epochs": N_EPOCHS},
        suffix=SUFFIX,
    )
    job_id = job.id
    print(f"  job id: {job_id}")

    start = time.time()
    seen_event_ids: set[str] = set()

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>4}s] status: {job.status}")

        # Print any new events
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=20)
        for event in reversed(events.data):
            if event.id not in seen_event_ids:
                seen_event_ids.add(event.id)
                print(f"         event: {event.message}")

        if job.status in TERMINAL_STATES:
            break

        time.sleep(POLL_INTERVAL)

    if job.status == "succeeded":
        model_id = job.fine_tuned_model
        n_train = _count_jsonl(train)
        print(f"\nSuccess! Fine-tuned model: {model_id}")
        _write_model_registry(db_path, model_id, n_train)
        _save_model_id(model_id)
        print(f"Model ID written to {FT_MODEL_ID_FILE} and model_registry.")
        return model_id

    # Failed or cancelled
    error_msg = getattr(job, "error", None)
    detail = error_msg.message if error_msg else job.status
    raise RuntimeError(f"Fine-tune job {job_id} ended with status '{job.status}': {detail}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI fine-tuning job.")
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--val",   required=True, help="Path to val.jsonl")
    parser.add_argument("--db",    required=True, help="Path to SQLite database")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and print plan without submitting")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.train, args.val)
    else:
        run_finetune(args.train, args.val, args.db)
