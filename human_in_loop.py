"""
Human-in-the-loop review for low-confidence field fills.

All DB writes are parameterised on db_path — nothing is hardcoded.
"""

import sqlite3


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------

def record_approval(db_path: str, event_id: int) -> None:
    """Mark a run_events row as correct."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE run_events SET is_correct = 1 WHERE id = ?",
            (event_id,),
        )


def record_correction(
    db_path: str,
    event_id: int,
    corrected_value: str,
    reason: str | None = None,
) -> None:
    """Mark a run_events row as wrong and store the correction."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE run_events
               SET is_correct = 0,
                   corrected_value = ?,
                   rejection_reason = ?
             WHERE id = ?
            """,
            (corrected_value, reason, event_id),
        )


def record_shadow_choice(db_path: str, shadow_id: int, human_chose: str) -> None:
    """Write the human's choice back to a shadow_eval row."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE shadow_eval SET human_chose = ? WHERE id = ?",
            (human_chose, shadow_id),
        )


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def review_field(
    event_id: int,
    field_label: str,
    suggested_value: str,
    db_path: str,
    ft_answer: str | None = None,
) -> str:
    """
    Prompt the user to accept, override, or (when ft_answer is provided) choose
    between the auto-suggested and fine-tuned values.

    Returns the value that should be used for the field.
    """
    print(f"\n[REVIEW] {field_label}")

    if ft_answer is not None:
        # Three-way choice
        print(f"  [A] Auto-suggested:  {suggested_value}")
        print(f"  [F] Fine-tuned:      {ft_answer}")
        print("  [C] Type custom value")

        while True:
            choice = input("  Choice [A/F/C]: ").strip().upper()
            if choice in ("", "A"):
                record_approval(db_path, event_id)
                return suggested_value
            elif choice == "F":
                reason = input("  Reason? (press Enter to skip): ").strip() or None
                record_correction(db_path, event_id, ft_answer, reason)
                return ft_answer
            elif choice == "C":
                custom = input("  Enter value: ").strip()
                if not custom:
                    continue
                reason = input("  Reason? (press Enter to skip): ").strip() or None
                record_correction(db_path, event_id, custom, reason)
                return custom
            else:
                print("  Please enter A, F, or C.")

    else:
        # Two-way choice: accept or override
        print(f"  Suggested: {suggested_value}")
        response = input("  Press Enter to accept, or type a new value: ").strip()

        if response in ("", "y", "Y"):
            record_approval(db_path, event_id)
            return suggested_value
        else:
            reason = input("  Reason? (press Enter to skip): ").strip() or None
            record_correction(db_path, event_id, response, reason)
            return response
