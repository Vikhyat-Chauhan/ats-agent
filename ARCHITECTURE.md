# ATS Agent — Architecture & Code Guide

## What this project does

This is a self-improving job application bot. It opens job listing URLs in a real browser, fills every application form field using an LLM, and records what it filled. Over time you label those fills as correct or wrong. When enough labeled data accumulates, the system fine-tunes a model on your corrections and promotes it to replace the previous one. The loop repeats indefinitely.

---

## High-level flow

```
Job URL
  │
  ▼
browser/agent.py   ← main orchestrator
  │  ├─ browser/navigator.py   detect & click Apply / Next / Submit buttons
  │  ├─ browser/ats.py         detect ATS platform, discover form fields
  │  ├─ field_mapper.py        call active LLM to get fill values
  │  ├─ browser/filler.py      write values into the DOM
  │  ├─ human_in_loop.py       prompt for review when confidence is low
  │  └─ db/schema.py           record every fill event to runs.db
  │
  └─ runs.db  (SQLite — ground truth for everything below)

Offline pipeline (run manually or on a schedule)
  │
  ├─ pipeline/export_training_data.py   turn labeled rows into JSONL
  ├─ pipeline/finetune_openai.py        upload JSONL, start OpenAI fine-tune job
  ├─ pipeline/shadow_eval.py            compare ft model vs base, track win rate
  └─ pipeline/retrain_trigger.py        cron-friendly wrapper for the full loop
```

---

## Component reference

### `browser/agent.py` — the orchestrator

The `ApplyAgent` class owns the full apply session. Call `agent.apply(url)`.

```
apply(url)
  1. page.goto(url)
  2. detect_platform(url, page)          → "greenhouse" | "lever" | "workday" | "unknown"
  3. click_apply(page)                   → navigates into the actual form
  4. detect_platform(new_url, page)      → update platform if we landed somewhere new
  5. loop (up to MAX_STEPS=20 pages):
       fields = find_fields(page, platform)
       for each field:
           _process_field(page, field, ...)
       click_next_or_submit(page)
```

`_process_field` is where the decision-making happens:

```
_process_field(field)
  1. If checkbox + consent label → skip LLM, use "yes", confidence 0.95
  2. Otherwise → map_field(label, resume, platform, ...)  ← LLM call
  3. score_confidence(label, field_type)                  ← rule-based table
  4. INSERT into run_events
  5. If confidence < threshold → review_field()           ← human prompt
  6. fill_field(page, field, value)                       ← DOM write
```

**Confidence table** (`_CONFIDENCE_TABLE`): maps field label patterns to a base score (0.60–0.95). Select fields get a −0.08 penalty. Fields below `confidence_threshold` (default 0.70) pause for human review.

**CLI flags:**
| Flag | Effect |
|------|--------|
| `--headful` | Show the browser window |
| `--slow` | 800 ms delay between fills |
| `--pause` | Stop and wait for Enter before each Next/Submit click |
| `--no-submit` | Fill but do not click the final Submit |
| `--threshold N` | Override confidence review threshold |
| `--force-base` | Ignore `.ft_model_id`, always use the base OpenAI model |

---

### `browser/ats.py` — platform detection and field discovery

**Platform detection** matches the page URL against regex patterns for six ATS platforms (Greenhouse, Lever, Workday, iCIMS, Ashby, SmartRecruiters). Falls back to scanning page HTML, then returns `"unknown"`.

**Field discovery** runs JavaScript on the page via `page.evaluate()`. There are four JS snippets:

| Platform | Strategy |
|----------|----------|
| Greenhouse | `.field`, `.form-group` wrappers |
| Lever | `.application-field` wrappers |
| Workday | `[data-automation-id]` wrappers |
| Generic | Four passes: `label[for]`, wrapping labels, `aria-label`/`aria-labelledby`, then `placeholder` as last resort |

All finders return the same shape (`label`, `id`, `name`, `tag`, `type`, `required`, `options`), which `_eval_fields` converts into `FormField` dataclasses.

**`FormField`** fields:
- `label` — human-readable field name (asterisks/whitespace stripped)
- `selector` — CSS selector used to target the input (uses `[id="..."]` syntax when the id contains `.`, `#`, `[`, etc. to avoid CSS parsing errors)
- `field_type` — `text | email | tel | url | number | date | select | textarea | checkbox`
- `options` — non-empty list only for `select` fields
- `field_id` — stable key (html id or name attribute)
- `required` — whether the field is marked required

---

### `browser/navigator.py` — dynamic button discovery

Instead of matching button text against a hardcoded list, every visible interactive element is scored using signals the developer put there intentionally.

**Scoring signals collected from the DOM:**
- `attr_apply / attr_next / attr_submit / attr_continue` — whether the keyword appears in `id`, `class`, `aria-label`, `data-testid`, `name`, or `href`
- `is_submit` — `type="submit"` (weak signal — multi-step forms use it for "Next" buttons too)
- `is_primary` — class name contains `primary`, `cta`, `main`, `hero`, `featured`, or `action`
- `rect_score` — element area × proximity to top of viewport (bigger + higher = more prominent)

`click_apply` weights: `attr_apply×5 + is_primary×2 + rect_score + text_score`

`click_next_or_submit` runs two separate scorers and picks the winner. The submit scorer hard-penalizes any button whose text matches `next|continue|proceed`.

---

### `field_mapper.py` — LLM field mapping

`map_field(field_label, resume_json, platform, db_path, ...)` returns `{"value": ..., "model_id": ..., "model_source": "ft"|"base"}`.

**Model selection** (checked in order):
1. If `.ft_model_id` exists and is non-empty → use fine-tuned model
   - If it's a filesystem path → local inference via HuggingFace + 4-bit quantization
   - If it's an OpenAI model ID → OpenAI API
2. Otherwise → base model via OpenAI API (`gpt-4.1-mini-2025-04-14`)
3. `force_base=True` skips step 1 entirely

**Type-aware prompts** — the LLM receives a type hint so it knows what format to return:
| `field_type` | Instruction sent to LLM |
|---|---|
| `checkbox` | Reply with exactly "yes" or "no" |
| `select` | You MUST reply with exactly one of: "opt1", "opt2"… |
| `date` | Reply in YYYY-MM-DD format |
| `number` | Reply with digits only |
| `textarea` | Reply with a concise paragraph |
| everything else | Reply with only the value to fill in |

Placeholder options ("Please Select", "-- Select --", "N/A") are stripped before the option list is sent to the LLM.

**Shadow eval** fires in a background thread when the active model is an OpenAI fine-tuned model (not local). It asks the base model the same question and stores both answers for later comparison.

---

### `browser/filler.py` — DOM writing

`fill_field(page, field, value)` dispatches by `field_type`:
- **select** → `snap_to_option` finds the best matching option (exact → case-insensitive → difflib fuzzy → first), then `page.select_option`
- **checkbox** → `page.check` or `page.uncheck` based on whether value is truthy ("yes", "true", "1", "checked", "on")
- **everything else** → `page.fill(selector, value)`

---

### `human_in_loop.py` — interactive review

Called when `confidence < threshold`. Prints the field name and suggested value and waits for terminal input.

**Two-way mode** (no shadow eval): press Enter to accept, or type a replacement.

**Three-way mode** (when shadow eval ran): choose `[A]` auto-suggested, `[F]` fine-tuned, or `[C]` custom. All choices write back to `run_events` (`is_correct`, `corrected_value`, `rejection_reason`).

---

### `db/schema.py` — database migrations

`migrate(db_path)` is safe to call repeatedly (additive only — never drops or modifies existing data). It creates or upgrades three tables:

**`run_events`** — one row per field fill
| Column | Purpose |
|--------|---------|
| `run_id` | Groups all fills from one `agent.apply()` call |
| `ats_platform` | Detected platform name |
| `job_url` | URL at time of fill |
| `field_id / field_label` | Identity of the form field |
| `value_used` | What the LLM returned |
| `confidence` | Score from the rule table |
| `source` | Always `"auto"` for LLM fills |
| `resume_snapshot` | Full resume JSON at time of fill (needed for training) |
| `is_correct` | NULL until reviewed; 1 = correct, 0 = wrong |
| `corrected_value` | Human's override (if any) |
| `rejection_reason` | Free-text reason for rejection |

**`shadow_eval`** — one row per background shadow prediction
| Column | Purpose |
|--------|---------|
| `ft_answer` | What the fine-tuned model said |
| `base_answer` | What the base model said (populated at review time) |
| `human_chose` | Which answer the human picked |

**`model_registry`** — one row per fine-tune job
| Column | Purpose |
|--------|---------|
| `model_id` | OpenAI fine-tuned model ID |
| `last_trained_at` | Timestamp of the training run |
| `train_examples` | Size of the training set |
| `ft_win_pct` | Win rate from shadow eval at promotion time |

---

### `pipeline/export_training_data.py` — labeled data → JSONL

Reads `run_events` rows where `is_correct IS NOT NULL` (i.e., human-reviewed). The ground truth value is `corrected_value` if the fill was wrong, or `value_used` if it was correct. Output is an OpenAI chat-format JSONL (system + user + assistant messages), split 90/10 train/val. Requires at least 50 examples; prints a platform and field breakdown.

---

### `pipeline/finetune_openai.py` — OpenAI fine-tune job

Uploads train/val JSONL to OpenAI Files API, starts a fine-tuning job on `gpt-4.1-mini-2025-04-14`, polls every 30 seconds until terminal state, then:
- Writes the new model ID to `.ft_model_id`
- Inserts a row into `model_registry`

`--dry-run` validates the JSONL files and prints what it would do without making any API calls.

---

### `pipeline/shadow_eval.py` — model comparison

`shadow_predict` is called fire-and-forget (in a background thread from `field_mapper.py`) when the active model is an OpenAI fine-tuned model. It calls the fine-tuned model on the same input and stores the result in `shadow_eval`.

`win_rate_report` reads all `shadow_eval` rows where a human made a choice and computes the fine-tuned model's win percentage, broken down by ATS platform. Returns `ready_to_promote: True` when `ft_win_pct >= 70%`.

---

### `pipeline/retrain_trigger.py` — automated retrain loop

Designed to run as a cron job. Full sequence:

```
1. Count new labeled rows since last training run
2. If count < threshold (default 100) → exit, no retrain
3. export_training_data  → train.jsonl + val.jsonl
4. finetune_openai       → new OpenAI model ID
5. shadow_eval.win_rate_report()
6. If ft_win_pct >= 70%  → write new model ID to .ft_model_id (promotes it)
   Else                  → keep current model
```

---

## Data flow summary

```
runs.db
  run_events.is_correct = NULL   ← just filled, not yet reviewed
  run_events.is_correct = 1      ← human approved
  run_events.is_correct = 0      ← human corrected
       │
       │  export_training_data.py
       ▼
  train.jsonl / val.jsonl
       │
       │  finetune_openai.py
       ▼
  OpenAI fine-tuned model
       │
       │  .ft_model_id (single line — the active model)
       ▼
  field_mapper.py uses it for all future fills
```

---

## The `.ft_model_id` file

This single file controls which model is active. Its contents determine behavior at startup:

| Content | Behavior |
|---------|----------|
| Empty / missing | Use base model via OpenAI API |
| OpenAI model ID (starts with `ft:`) | Use that fine-tuned model via OpenAI API |
| Filesystem path (`/path/to/model`) | Load that model locally with 4-bit quantization |

**Do not run the test suite without first noting that tests in `test_finetune_openai.py` now use an autouse fixture that redirects all writes to a temp path — they will not touch the real `.ft_model_id` file.**

---

## Running the agent

```bash
# Apply to a job (observe everything, don't actually submit)
python3 -m browser.agent \
  --url "https://boards.greenhouse.io/acme/jobs/12345" \
  --resume data/master_resume.json \
  --db runs.db \
  --headful --slow --pause --no-submit

# Force the base model (ignore .ft_model_id)
python3 -m browser.agent ... --force-base

# Initialize the database schema
python3 -m db.schema --db runs.db

# Export labeled data and start a fine-tune
python3 -m pipeline.export_training_data --db runs.db --out data/
python3 -m pipeline.finetune_openai --train data/train.jsonl --val data/val.jsonl --db runs.db

# Check shadow eval win rate
python3 -m pipeline.shadow_eval --report --db runs.db

# Run the full retrain loop (cron-style)
python3 -m pipeline.retrain_trigger --db runs.db --threshold 100
```

---

## Tests

226 tests across 7 test files. Run with:

```bash
python3 -m pytest
```

Key testing notes:
- `tests/test_finetune_openai.py` patches `FT_MODEL_ID_FILE` in every test via an autouse fixture — the real `.ft_model_id` is never touched.
- `tests/test_browser_agent.py` mocks Playwright (`page` is a `MagicMock`) and patches `_click_next_or_submit` to return `False` so the fill loop terminates after one step.
- `shadow_predict` is called in background threads during live runs; tests patch it out entirely.
