# ats-agent

> An agentic browser-automation tool that fills job-application forms across **Workday, iCIMS, Greenhouse, and Lever** from your tailored résumé JSON — with a human-in-the-loop review step and a **self-improving fine-tuning loop** that learns from your corrections over time.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C)
![Browser Use](https://img.shields.io/badge/Browser_Use-automation-6E56CF)
![Playwright](https://img.shields.io/badge/Playwright-driver-2EAD33?logo=playwright&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-fine--tune-412991?logo=openai&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-175_tests-0A9EDC?logo=pytest&logoColor=white)

---

## What it does

- **Multi-ATS form filling** — drives **Workday, iCIMS, Greenhouse, and Lever** through Browser Use + Playwright, orchestrated as a LangGraph agent.
- **Résumé as the data source** — reads a structured résumé JSON and maps each form field to the right value.
- **Human-in-the-loop review** — low-confidence fills pause for an `accept / correct / choose` menu; every correction is logged as training signal instead of being thrown away.
- **Self-improving** — logged corrections export to an OpenAI fine-tuning job; a **shadow-evaluation gate** promotes the fine-tuned model *only* when it beats the base model on ≥ 70% of human-reviewed fields.
- **Cheap to run** — ~**$0.003 per application**, ~**$0.33 per 100 applications** (see cost table below).
- **Tested** — **175 pytest cases** across the migration, review, export, fine-tune, shadow-eval, and retrain modules.

---

## Self-improving fine-tuning loop

The form-filler is wrapped in a feedback loop that turns human corrections into a continuously-retrained model:

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
    ↓
field_mapper.py — auto-selects ft model if .ft_model_id exists, fires shadow in background
    ↓ pipeline/retrain_trigger.py checks for 100 new examples → reruns loop
```

---

## Quick start

```bash
# 1. Initialise the data layer (idempotent — safe to re-run)
python -m db.schema --db runs.db

# 2. Fill forms with human review on low-confidence fields (field_mapper / agent entrypoints)
#    Corrections are written back to runs.db as training signal.

# 3. Once you have ≥ 50 labeled examples, export and fine-tune
python -m pipeline.export_training_data --db runs.db --out data/
python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db

# 4. Check whether the fine-tuned model is good enough to promote
python -m pipeline.shadow_eval --report --db runs.db   # exits 0 if ft wins ≥ 70%
```

Requires `OPENAI_API_KEY` in `.env` (loaded via `python-dotenv`).

---

## Pipeline reference

### Human-in-the-loop review

```python
from human_in_loop import review_field

# Two-way: accept the suggestion or type a correction
value = review_field(event_id=1, field_label="Salary expectation",
                     suggested_value="$120,000", db_path="runs.db")

# Three-way: also show a fine-tuned model answer (A = auto / F = fine-tuned / C = custom)
value = review_field(event_id=1, field_label="Salary expectation",
                     suggested_value="$120,000", db_path="runs.db",
                     ft_answer="$130,000")
```

### Export training data

```bash
python -m pipeline.export_training_data --db runs.db --out data/
# Total examples : 120  | Train: 108  Val: 12
# Platform breakdown: {'workday': 40, 'greenhouse': 35, ...}
```

Outputs `data/train.jsonl` and `data/val.jsonl` in OpenAI fine-tuning format. Refuses to write if fewer than 50 labeled examples exist.

### Run a fine-tune job

```bash
# Validate without submitting
python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db --dry-run
# Run for real (gpt-4o-mini, 3 epochs)
python -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db
```

### Shadow-evaluation report

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

### Retrain trigger (cron)

```bash
python -m pipeline.retrain_trigger --db runs.db --threshold 100
# Add to crontab to run nightly:
# 0 2 * * * cd /path/to/ats-agent && python -m pipeline.retrain_trigger --db runs.db --threshold 100
```

### Data model

```sql
-- Review columns added to the existing run_events table by db/schema.py
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

---

## Estimated API costs

| Operation | Cost |
|---|---|
| Fine-tune training run (~60 examples) | ~$0.03 |
| Per field fill (ft model inference) | ~$0.0001 |
| Per job application (15 fields + shadow) | ~$0.003 |
| 100 applications / month | ~$0.33 |

---

## Running tests

```bash
python3 -m pytest -v   # 175 tests
```

---

## Changelog

### [Unreleased]

**Added**
- `db/schema.py` — idempotent SQLite migration module (adds four review columns to `run_events`; creates `shadow_eval` and `model_registry` tables; safe to re-run).
- `human_in_loop.py` — accept / correct / `[A/F/C]` review for low-confidence fills; writes labels back to `run_events` and `shadow_eval`.
- `pipeline/export_training_data.py` — exports labeled rows as OpenAI fine-tuning JSONL (90/10 split, seeded; refuses < 50 examples).
- `pipeline/finetune_openai.py` — uploads JSONL, runs a `gpt-4o-mini` fine-tune job, polls to completion, registers the model (`--dry-run` supported).
- `pipeline/shadow_eval.py` — runs the fine-tuned model alongside the base agent and reports win rate (`--report` exits 0 at ≥ 70%).
- `field_mapper.py` — auto-selects the fine-tuned model when present, fires shadow predictions in the background (`--force-base` to override).
- `pipeline/retrain_trigger.py` — cron-friendly loop: export → fine-tune → evaluate → conditionally promote.
- Test suites for every module above (**175 tests total**), plus `pytest.ini` and `.gitignore`.
