"""
Retrain trigger — designed to run as a cron job.

Counts new labeled examples since the last training run. If the count meets
the threshold, runs the full export → fine-tune → evaluate → promote loop.

Usage:
    python -m pipeline.retrain_trigger --db runs.db --threshold 100
    python -m pipeline.retrain_trigger --db runs.db --threshold 5 --dry-run
"""

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import openai

from pipeline.export_training_data import export
from pipeline.finetune_openai import FT_MODEL_ID_FILE, run_finetune
from pipeline.shadow_eval import win_rate_report

load_dotenv()

DEFAULT_THRESHOLD = 100
PROMOTE_WIN_PCT   = 70.0


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _last_trained_at(db_path: str) -> str | None:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT last_trained_at FROM model_registry ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _count_new_examples(db_path: str, since: str | None) -> int:
    conn = sqlite3.connect(db_path)
    if since:
        count = conn.execute(
            """
            SELECT COUNT(*) FROM run_events
             WHERE event_type = 'field_filled'
               AND is_correct IS NOT NULL
               AND resume_snapshot IS NOT NULL
               AND (corrected_value IS NOT NULL OR is_correct = 1)
               AND timestamp > ?
            """,
            (since,),
        ).fetchone()[0]
    else:
        count = conn.execute(
            """
            SELECT COUNT(*) FROM run_events
             WHERE event_type = 'field_filled'
               AND is_correct IS NOT NULL
               AND resume_snapshot IS NOT NULL
               AND (corrected_value IS NOT NULL OR is_correct = 1)
            """
        ).fetchone()[0]
    conn.close()
    return count


def _update_ft_win_pct(db_path: str, model_id: str, ft_win_pct: float) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE model_registry SET ft_win_pct = ? WHERE model_id = ?",
            (ft_win_pct, model_id),
        )


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(db_path: str, threshold: int) -> None:
    last = _last_trained_at(db_path)
    count = _count_new_examples(db_path, last)
    since_str = last if last else "the beginning of time"
    print(f"[DRY RUN] Found {count} new labeled examples since {since_str}")
    if count >= threshold:
        print("[DRY RUN] Would export data → fine-tune → evaluate → conditionally promote")
    else:
        print(
            f"[DRY RUN] {count} examples < threshold {threshold}. "
            "No retrain would be triggered."
        )


# ---------------------------------------------------------------------------
# Full retrain loop
# ---------------------------------------------------------------------------

def retrain(
    db_path: str,
    threshold: int,
    *,
    client: openai.OpenAI | None = None,
    out_dir: str | None = None,
) -> bool:
    """
    Run the full retrain loop. Returns True if a new model was promoted.
    """
    last = _last_trained_at(db_path)
    count = _count_new_examples(db_path, last)
    since_str = last if last else "the beginning of time"

    print(f"Found {count} new labeled examples since {since_str}")

    if count < threshold:
        print(f"{count} < threshold {threshold}. No retrain needed.")
        return False

    print(f"Threshold met ({count} >= {threshold}). Starting retrain pipeline…")

    # 1. Export
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="ats_retrain_")
        out_dir = tmp

    print(f"Exporting training data to {out_dir}…")
    export(db_path, out_dir)
    train_path = str(Path(out_dir) / "train.jsonl")
    val_path   = str(Path(out_dir) / "val.jsonl")

    # 2. Fine-tune
    print("Starting fine-tune job…")
    new_model_id = run_finetune(train_path, val_path, db_path, client=client)

    # 3. Evaluate
    print("Running shadow eval report…")
    report = win_rate_report(db_path)
    ft_win_pct = report["ft_win_pct"]
    _update_ft_win_pct(db_path, new_model_id, ft_win_pct)

    # 4. Promote or reject
    if report["ready_to_promote"]:
        Path(FT_MODEL_ID_FILE).write_text(new_model_id)
        print(f"Promoted new model: {new_model_id}  (ft_win_pct={ft_win_pct}%)")
        return True
    else:
        print(
            f"New model did not clear threshold. "
            f"ft_win_pct={ft_win_pct}% < {PROMOTE_WIN_PCT}%. "
            "Keeping current."
        )
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger model retrain if enough new data exists.")
    parser.add_argument("--db",        required=True, help="Path to SQLite database")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"Minimum new examples to trigger retrain (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print what would happen without running anything")
    parser.add_argument("--out",       default=None,
                        help="Output directory for exported JSONL (default: temp dir)")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.db, args.threshold)
    else:
        retrain(args.db, args.threshold, out_dir=args.out)
