"""
Export labeled run_events rows as OpenAI fine-tuning JSONL.

Usage:
    python -m pipeline.export_training_data --db runs.db --out data/
"""

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a job application assistant. Given a form field label and a resume JSON, "
    "return exactly the value to fill in that field. No explanation. No markdown. Just the value."
)

QUERY = """
    SELECT id, ats_platform, field_label, value_used, corrected_value,
           resume_snapshot, is_correct
      FROM run_events
     WHERE event_type = 'field_filled'
       AND is_correct IS NOT NULL
       AND resume_snapshot IS NOT NULL
       AND (corrected_value IS NOT NULL OR is_correct = 1)
"""

MIN_EXAMPLES = 50


def _fetch_rows(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(QUERY).fetchall()]
    finally:
        conn.close()


def _count_skipped(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            """
            SELECT COUNT(*) FROM run_events
             WHERE event_type = 'field_filled'
               AND is_correct IS NOT NULL
               AND resume_snapshot IS NULL
            """
        ).fetchone()[0]
    finally:
        conn.close()


def _to_example(row: dict) -> dict:
    ground_truth = row["corrected_value"] if row["corrected_value"] else row["value_used"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ATS: {row['ats_platform']}\n"
                    f"Field: {row['field_label']}\n"
                    f"Resume:\n{row['resume_snapshot']}\n"
                    "What value?"
                ),
            },
            {"role": "assistant", "content": ground_truth or ""},
        ]
    }


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _print_summary(
    rows: list[dict],
    n_train: int,
    n_val: int,
    n_skipped: int,
    top_n_fields: int = 5,
) -> None:
    platform_counts = Counter(r["ats_platform"] for r in rows)
    field_counts = Counter(r["field_label"] for r in rows).most_common(top_n_fields)

    print(f"Total examples : {len(rows)}")
    print(f"Train          : {n_train}  |  Val: {n_val}")
    print(f"Platform breakdown: {dict(platform_counts)}")
    print(f"Most common fields: {field_counts}")
    print(f"Skipped (missing resume_snapshot): {n_skipped}")


def export(db_path: str, out_dir: str) -> tuple[int, int]:
    """
    Run the full export. Returns (n_train, n_val).
    Raises SystemExit with code 1 if fewer than MIN_EXAMPLES labeled rows exist.
    """
    rows = _fetch_rows(db_path)
    n_skipped = _count_skipped(db_path)

    if len(rows) < MIN_EXAMPLES:
        print(
            f"WARNING: only {len(rows)} labeled examples found "
            f"(minimum {MIN_EXAMPLES} required). No files written."
        )
        raise SystemExit(1)

    rng = random.Random(42)
    rng.shuffle(rows)

    split = max(1, int(len(rows) * 0.9))
    train_rows, val_rows = rows[:split], rows[split:]

    examples_train = [_to_example(r) for r in train_rows]
    examples_val = [_to_example(r) for r in val_rows]

    out = Path(out_dir)
    _write_jsonl(out / "train.jsonl", examples_train)
    _write_jsonl(out / "val.jsonl", examples_val)

    _print_summary(rows, len(examples_train), len(examples_val), n_skipped)

    return len(examples_train), len(examples_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuning data from run_events.")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--out", required=True, help="Output directory for JSONL files")
    args = parser.parse_args()
    export(args.db, args.out)
