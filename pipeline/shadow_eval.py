"""
Shadow evaluation: run the fine-tuned model alongside the base agent and track
which answer humans prefer.

Usage:
    python -m pipeline.shadow_eval --report --db runs.db
"""

import argparse
import sqlite3
from pathlib import Path

import openai

from pipeline.finetune_openai import BASE_MODEL, FT_MODEL_ID_FILE

SYSTEM_PROMPT = (
    "You are a job application assistant. Given a form field label and a resume JSON, "
    "return exactly the value to fill in that field. No explanation. No markdown. Just the value."
)

PROMOTE_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Model ID
# ---------------------------------------------------------------------------

def read_model_id(path: str = FT_MODEL_ID_FILE) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model ID file '{path}' not found. Run the fine-tune step first."
        )
    model_id = p.read_text().strip()
    if not model_id:
        raise ValueError(f"Model ID file '{path}' is empty.")
    return model_id


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def shadow_predict(
    field_label: str,
    resume_json: str,
    platform: str,
    db_path: str,
    *,
    model_id: str | None = None,
    client: openai.OpenAI | None = None,
) -> tuple[str, int]:
    """
    Call the fine-tuned model and record the result in shadow_eval.
    Returns (ft_answer, shadow_id).
    """
    if model_id is None:
        model_id = read_model_id()
    if client is None:
        client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ATS: {platform}\n"
                    f"Field: {field_label}\n"
                    f"Resume:\n{resume_json}\n"
                    "What value?"
                ),
            },
        ],
        temperature=0,
    )
    ft_answer = response.choices[0].message.content.strip()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO shadow_eval (field_label, platform, ft_answer, timestamp)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (field_label, platform, ft_answer),
        )
        shadow_id = cursor.lastrowid

    return ft_answer, shadow_id


# ---------------------------------------------------------------------------
# Win-rate report
# ---------------------------------------------------------------------------

def win_rate_report(db_path: str) -> dict:
    """
    Compute win-rate statistics from reviewed shadow_eval rows.
    Returns a dict and prints a formatted platform-breakdown table.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT platform, base_answer, ft_answer, human_chose
          FROM shadow_eval
         WHERE human_chose IS NOT NULL
        """
    ).fetchall()
    conn.close()

    total = len(rows)
    ft_wins = sum(1 for r in rows if r["human_chose"] == r["ft_answer"])
    base_wins = sum(1 for r in rows if r["human_chose"] == r["base_answer"])
    human_custom = total - ft_wins - base_wins
    ft_win_pct = (ft_wins / total * 100) if total > 0 else 0.0

    # Platform breakdown
    platforms: dict[str, dict] = {}
    for r in rows:
        p = r["platform"] or "unknown"
        if p not in platforms:
            platforms[p] = {"total": 0, "ft_wins": 0}
        platforms[p]["total"] += 1
        if r["human_chose"] == r["ft_answer"]:
            platforms[p]["ft_wins"] += 1

    _print_platform_table(platforms)

    return {
        "total_reviewed": total,
        "ft_wins": ft_wins,
        "base_wins": base_wins,
        "human_custom": human_custom,
        "ft_win_pct": round(ft_win_pct, 1),
        "ready_to_promote": ft_win_pct >= PROMOTE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{n / total * 100:.1f}%"


def _print_platform_table(platforms: dict) -> None:
    if not platforms:
        return
    print("\nPlatform breakdown:")
    header = f"  {'Platform':<12}  {'Reviewed':>8}  {'FT wins':>8}  {'FT win %':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for platform, stats in sorted(platforms.items()):
        t = stats["total"]
        w = stats["ft_wins"]
        print(f"  {platform:<12}  {t:>8}  {w:>8}  {_pct(w, t):>9}")


def _print_report_table(report: dict) -> None:
    total   = report["total_reviewed"]
    ft_w    = report["ft_wins"]
    base_w  = report["base_wins"]
    custom  = report["human_custom"]
    pct     = report["ft_win_pct"]
    promote = "✓ YES" if report["ready_to_promote"] else "✗ NO"

    rows = [
        ("Total reviewed",    str(total)),
        ("FT model wins",     f"{ft_w} ({_pct(ft_w, total)})"),
        ("Base model wins",   f"{base_w} ({_pct(base_w, total)})"),
        ("Human custom",      f"{custom} ({_pct(custom, total)})"),
        ("Ready to promote?", promote),
    ]

    label_w = max(len(r[0]) for r in rows) + 2
    value_w = max(len(r[1]) for r in rows) + 2
    total_w = label_w + value_w + 3  # borders + separator

    top    = "┌" + "─" * total_w + "┐"
    title  = "│" + " Shadow Evaluation Report".ljust(total_w) + "│"
    sep    = "├" + "─" * (label_w + 2) + "┬" + "─" * (value_w + 1) + "┤"
    bottom = "└" + "─" * (label_w + 2) + "┴" + "─" * (value_w + 1) + "┘"

    print(top)
    print(title)
    print(sep)
    for label, value in rows:
        print(f"│ {label:<{label_w}}│ {value:<{value_w}}│")
    print(bottom)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow evaluation reporting.")
    parser.add_argument("--report", action="store_true", help="Print win-rate report")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    args = parser.parse_args()

    if args.report:
        report = win_rate_report(args.db)
        _print_report_table(report)
        raise SystemExit(0 if report["ready_to_promote"] else 1)
    else:
        parser.print_help()
        raise SystemExit(1)
