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
- **`human_in_loop.py`** — human review module for low-confidence field fills
  - Two-way prompt (accept / override) when no fine-tuned answer is available
  - Three-way `[A/F/C]` menu when a `ft_answer` is passed — auto-suggested, fine-tuned, or custom
  - Writes `is_correct`, `corrected_value`, `rejection_reason` back to `run_events`
  - `record_shadow_choice()` writes the human's pick back to `shadow_eval`
  - All DB writes parameterised on `db_path` — nothing hardcoded
- **`tests/test_human_in_loop.py`** — 21-test pytest suite covering all DB helpers, both prompt modes, and edge cases (invalid choice loops, empty custom value loops)
- **`pipeline/export_training_data.py`** — exports labeled `run_events` rows as OpenAI fine-tuning JSONL
  - Filters for rows with `is_correct IS NOT NULL`, `resume_snapshot IS NOT NULL`, and a ground truth value
  - Ground truth: `corrected_value` when set, otherwise `value_used`
  - 90/10 train/val split, shuffled with `random.seed(42)` for reproducibility
  - Prints dataset summary: total, split counts, platform breakdown, most common fields, skipped count
  - Refuses to write files if fewer than 50 labeled examples exist (prints `WARNING` and exits 1)
  - CLI: `python -m pipeline.export_training_data --db runs.db --out data/`
- **`tests/test_export_training_data.py`** — 30-test pytest suite covering output files, JSONL format, ground truth selection, reproducibility, threshold guard, and CLI
- **`pipeline/finetune_openai.py`** — uploads JSONL files and runs an OpenAI fine-tuning job
  - Uploads `train.jsonl` and `val.jsonl` to the Files API (`purpose="fine-tune"`)
  - Creates a fine-tune job on `gpt-4o-mini-2024-07-18`, 3 epochs, suffix `job-agent`
  - Polls every 30s, printing status and new job events as they arrive
  - On success: writes model ID to `model_registry` and `.ft_model_id`
  - On failure/cancellation: raises `RuntimeError` with error detail
  - `--dry-run` validates files and prints the plan without touching the API
  - CLI: `python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db`
- **`tests/test_finetune_openai.py`** — 27-test pytest suite with fully mocked OpenAI client covering file upload, job creation, success/failure paths, DB writes, and CLI
- **`pipeline/shadow_eval.py`** — runs the fine-tuned model alongside the base agent and tracks human preference
  - `read_model_id()` — reads model ID from `.ft_model_id`, raises on missing/empty
  - `shadow_predict(field_label, resume_json, platform, db_path)` — calls fine-tuned model at `temperature=0`, inserts row into `shadow_eval`, returns `(ft_answer, shadow_id)`
  - `win_rate_report(db_path)` — queries reviewed rows, classifies ft-win/base-win/custom, prints platform breakdown table, returns stats dict
  - `--report` CLI exits 0 if `ft_win_pct ≥ 70%` (CI-gate ready), exits 1 otherwise
  - CLI: `python -m pipeline.shadow_eval --report --db runs.db`
- **`tests/test_shadow_eval.py`** — 31-test pytest suite covering model ID reading, prediction (DB insert, API args, whitespace stripping), win-rate stats, promote threshold, and CLI exit codes

---

## Fine-Tuning Pipeline

The `db/` module is the data layer for a fine-tuning loop that turns human corrections into training signal:

```
run_events (field fills)
    ↓ human_in_loop.review_field() — accept / correct / choose ft answer
    ↓ writes is_correct, corrected_value, rejection_reason
    ↓
pipeline/export_training_data.py — exports train.jsonl + val.jsonl
    ↓ 90/10 split, OpenAI chat format, ground truth = corrected_value ?? value_used
    ↓
pipeline/finetune_openai.py — uploads files, starts job, polls to completion
    ↓ writes model_id + train_examples to model_registry, saves .ft_model_id
    ↓
pipeline/shadow_eval.shadow_predict() — calls ft model, inserts shadow_eval row
    ↓ human reviews via human_in_loop [A/F/C], record_shadow_choice() writes human_chose
    ↓
pipeline/shadow_eval --report — win_rate_report(), exits 0 if ft_win_pct ≥ 70%
    ↓ ft_win_pct written to model_registry on promotion
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

### Human-in-the-Loop Review

```python
from human_in_loop import review_field

# Two-way: accept or type a correction
value = review_field(event_id=1, field_label="Salary expectation",
                     suggested_value="$120,000", db_path="runs.db")

# Three-way: also show a fine-tuned model answer
value = review_field(event_id=1, field_label="Salary expectation",
                     suggested_value="$120,000", db_path="runs.db",
                     ft_answer="$130,000")
```

### Exporting Training Data

```bash
python -m pipeline.export_training_data --db runs.db --out data/
# Total examples : 120
# Train          : 108  |  Val: 12
# Platform breakdown: {'workday': 40, 'greenhouse': 35, ...}
# Most common fields: [('Email', 20), ...]
# Skipped (missing resume_snapshot): 3
```

Outputs `data/train.jsonl` and `data/val.jsonl` in OpenAI fine-tuning format.

### Running a Fine-Tune Job

```bash
# Validate files without submitting
python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db --dry-run
# [DRY RUN] Would upload: data/train.jsonl (54 examples)
# [DRY RUN] Would upload: data/val.jsonl (6 examples)
# [DRY RUN] Would start fine-tune job on gpt-4o-mini-2024-07-18 for 3 epochs
# [DRY RUN] Would write model ID to .ft_model_id and model_registry

# Run for real (requires OPENAI_API_KEY)
python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db
```

### Shadow Evaluation Report

```bash
python -m pipeline.shadow_eval --report --db runs.db
# ┌─────────────────────────────────┐
# │ Shadow Evaluation Report        │
# ├──────────────────┬──────────────┤
# │ Total reviewed   │ 52           │
# │ FT model wins    │ 38 (73.1%)   │
# │ Base model wins  │ 10 (19.2%)   │
# │ Human custom     │ 4  (7.7%)    │
# │ Ready to promote?│ ✓ YES        │
# └──────────────────┴──────────────┘
# exits 0 if ft_win_pct ≥ 70%, exits 1 otherwise
```

### Running Tests

```bash
python3 -m pytest -v   # 133 tests
```
