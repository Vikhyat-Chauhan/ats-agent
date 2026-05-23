# ats-agent

An agentic browser automation tool that fills job application forms across **Workday, iCIMS, Greenhouse, and Lever** — using your tailored resume JSON as the data source.

---

## Changelog

### [Unreleased]

#### Added
- **`db/schema.py`** — idempotent SQLite migration module for the fine-tuning pipeline
  - Adds four review columns to `run_events`: `is_correct`, `corrected_value`, `rejection_reason`, `resume_snapshot`
  - Creates `shadow_eval` table for base-vs-fine-tuned answer comparison
  - Creates `model_registry` table for tracking trained model versions
  - `migrate(db_path)` is safe to call repeatedly — skips already-present columns and tables
  - CLI: `python -m db.schema --db runs.db` prints a summary of what changed
- **`tests/test_schema.py`** — 24-test pytest suite covering column migration, table creation, idempotency, summary return values, and CLI behaviour
- **`pytest.ini`** — pytest configured with `testpaths = tests`
- **`.gitignore`** — excludes `__pycache__`, `.venv`, `.env`, `logs/`, `models/`, `dist/`

---

## Fine-Tuning Pipeline

The `db/` module is the data layer for a fine-tuning loop that turns human corrections into training signal:

```
run_events (field fills)
    ↓ human reviews is_correct / corrected_value
    ↓
fine-tune job (model_registry tracks versions)
    ↓
shadow_eval (base vs. fine-tuned answer, human picks winner)
    ↓
ft_win_pct tracked in model_registry
```

### Setup

```bash
python -m db.schema --db runs.db
```

Re-running is safe — already-present columns and tables are skipped.

### Schema

```sql
-- Existing table, new review columns added by migration
ALTER TABLE run_events ADD COLUMN is_correct       INTEGER DEFAULT NULL;
ALTER TABLE run_events ADD COLUMN corrected_value  TEXT DEFAULT NULL;
ALTER TABLE run_events ADD COLUMN rejection_reason TEXT DEFAULT NULL;
ALTER TABLE run_events ADD COLUMN resume_snapshot  TEXT DEFAULT NULL;

CREATE TABLE shadow_eval (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    field_label TEXT, platform TEXT,
    base_answer TEXT, ft_answer TEXT,
    human_chose TEXT, timestamp TEXT
);

CREATE TABLE model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT, last_trained_at TEXT,
    train_examples INTEGER, ft_win_pct REAL
);
```

### Running Tests

```bash
python3 -m pytest -v
```
