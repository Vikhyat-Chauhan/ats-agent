# ats-agent

An agentic browser automation tool that fills job application forms across **Workday, iCIMS, Greenhouse, and Lever** — using your tailored resume JSON as the data source.

> **Prerequisite:** Works best after `resume-tailor` has generated a tailored resume JSON for the application. Can also run standalone with your master resume.

---

## What It Does

1. Detects which ATS platform a job application URL uses
2. Dispatches to the correct platform-specific playbook
3. Fills all standard fields: personal info, work history, education, skills, cover letter
4. Pauses and prompts you for fields it cannot fill automatically (custom questions, captchas)
5. Logs what was filled, what was skipped, and any errors per application

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Browser automation | `browser-use` + `playwright` |
| Agent orchestration | `langgraph` |
| LLM (field mapping) | Claude API or local Qwen2.5-14B (GGUF via `llama-cpp-python`) |
| ATS detection | URL pattern matching + DOM heuristics |
| Data source | `tailored_resume.json` from `resume-tailor`, or `master_resume.json` |
| CLI | `typer` |

---

## Project Structure

```
ats-agent/
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
│
├── data/
│   └── master_resume.json        # Fallback if no tailored JSON provided
│
├── playbooks/                    # Per-platform DOM strategies
│   ├── workday.py
│   ├── icims.py
│   ├── greenhouse.py
│   └── lever.py
│
├── logs/                         # Git-ignored; one JSON log per application
│   └── amazon_sde2_2025-06-01.json
│
├── src/
│   └── ats_agent/
│       ├── __init__.py
│       ├── cli.py                # typer entrypoint: `ats fill`
│       ├── detector.py           # Identify ATS platform from URL/DOM
│       ├── router.py             # Dispatch to correct playbook
│       ├── field_mapper.py       # LLM maps resume fields → form fields
│       ├── human_in_loop.py      # Pause + prompt for unresolvable fields
│       ├── logger.py             # Per-run structured log
│       └── models.py             # Pydantic: ApplicationRun, FieldFill
│
└── tests/
    ├── test_detector.py
    └── test_field_mapper.py
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/vikhyat/ats-agent.git
cd ats-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — see .env.example section below
```

### 3. Test with a Greenhouse application (easiest ATS to start with)

```bash
ats fill --url "https://boards.greenhouse.io/company/jobs/12345" --resume ../resume-tailor/outputs/company_role.json
```

---

## Usage

```bash
# Basic fill (auto-detects ATS)
ats fill --url "https://..." --resume path/to/tailored_resume.json

# Use master resume as fallback
ats fill --url "https://..." --resume data/master_resume.json

# Dry run: open browser, detect platform, print fields found — don't fill
ats fill --url "https://..." --dry-run

# Headless mode (no visible browser window)
ats fill --url "https://..." --headless

# Use local LLM instead of Claude API for field mapping
ats fill --url "https://..." --llm local
```

---

## How the Agent Works

```
Job Application URL
       │
       ▼
   detector.py
   ┌──────────────────────────────────┐
   │ URL pattern match:               │
   │  myworkday.com  → Workday        │
   │  icims.com      → iCIMS          │
   │  greenhouse.io  → Greenhouse     │
   │  lever.co       → Lever          │
   │  (DOM fallback if URL ambiguous) │
   └──────────────────────────────────┘
       │
       ▼
   router.py  →  playbooks/workday.py (or icims, greenhouse, lever)
       │
       ▼
   field_mapper.py
   ┌──────────────────────────────────────────────────────────┐
   │ For each form field found on page:                        │
   │   LLM prompt: "Given this field label and my resume JSON, │
   │   what value should I fill? Return null if unsure."       │
   │                                                           │
   │ Structured output:                                        │
   │   { field_id: str, value: str | null, confidence: float } │
   └──────────────────────────────────────────────────────────┘
       │
       ├── confidence > 0.85  →  Auto-fill
       └── confidence ≤ 0.85  →  human_in_loop.py (pause + prompt)
       │
       ▼
   Browser fills form → logger.py writes run log
```

---

## ATS Playbooks

Each playbook handles the quirks of its platform.

### Greenhouse (easiest — start here)
- Clean, stable DOM with `data-field` attributes
- Standard fields: name, email, phone, LinkedIn, resume upload, cover letter
- Custom questions vary by company — handled by `field_mapper.py`

### Lever
- Similar to Greenhouse in simplicity
- Frequently uses iframes; playwright handles these natively

### iCIMS
- Older platform; more form pages (multi-step wizard)
- Playbook tracks `current_step` in agent state

### Workday (hardest — build last)
- Heavy React SPA; DOM is dynamically generated
- Requires waiting for specific network idle states
- Many companies require uploading resume PDF which auto-populates fields — handle this first, then correct mistakes
- Dropdown values must match Workday's internal options exactly

---

## Platform Detection (`detector.py`)

```python
ATS_URL_PATTERNS = {
    "workday":    [r"myworkday\.com", r"wd\d+\.myworkday\.com"],
    "icims":      [r"icims\.com", r"careers\..*icims"],
    "greenhouse": [r"boards\.greenhouse\.io", r"greenhouse\.io/jobs"],
    "lever":      [r"jobs\.lever\.co", r"lever\.co"],
}
```

If URL matching fails, the agent inspects the page DOM for known ATS fingerprints (meta tags, script srcs, form IDs).

---

## Human-in-the-Loop (`human_in_loop.py`)

When the agent cannot confidently fill a field, it pauses the browser session and prompts you in the terminal:

```
[PAUSE] Could not determine value for field: "Years of experience in Java"
  Resume data available: skills.languages = ["Python", "C++", "JavaScript", ...]
  Java not found in resume.

> Enter value (or press Enter to skip): 
```

You type the value, it fills the field and continues.

---

## Logging

Each run produces `logs/company_role_date.json`:

```json
{
  "run_id": "amazon_sde2_2025-06-01",
  "url": "https://...",
  "ats": "workday",
  "resume_used": "outputs/amazon_sde2_2025-06-01.json",
  "fields": [
    { "label": "First Name", "value": "Vikhyat", "method": "auto", "confidence": 0.99 },
    { "label": "Years of Java Experience", "value": "0", "method": "human", "confidence": null }
  ],
  "skipped": ["Custom essay question — filled manually"],
  "status": "submitted",
  "duration_seconds": 142
}
```

---

## Build Phases

### Phase 1 — Detector + Greenhouse playbook (start here)
- [ ] `detector.py`: URL pattern matching
- [ ] `playbooks/greenhouse.py`: fill standard fields
- [ ] `field_mapper.py`: Claude API maps resume → field values
- [ ] `human_in_loop.py`: terminal pause for low-confidence fields
- [ ] CLI: `ats fill --url --dry-run`

### Phase 2 — Lever + logging
- [ ] `playbooks/lever.py`
- [ ] `logger.py`: structured JSON run log
- [ ] `ats logs` command: list past runs and status

### Phase 3 — iCIMS
- [ ] `playbooks/icims.py`: multi-step wizard state tracking
- [ ] Handle resume upload flow

### Phase 4 — Workday
- [ ] `playbooks/workday.py`: React SPA handling, network idle waits
- [ ] Resume upload + auto-populate + correction loop
- [ ] Handle Workday-specific dropdown value matching

### Phase 5 — Local LLM option
- [ ] `--llm local` flag: route `field_mapper.py` to Qwen2.5-14B via llama-cpp
- [ ] Reduces API costs for high-volume applications

---

## `.env.example`

```env
ANTHROPIC_API_KEY=sk-ant-...
LLM_BACKEND=claude            # or: local
LOCAL_LLM_MODEL_PATH=models/qwen2.5-14b.gguf
BROWSER_HEADLESS=false        # false = see what's happening while you build
LOG_DIR=logs
CONFIDENCE_THRESHOLD=0.85     # Below this → human-in-the-loop
```

---

## `.gitignore`

```
.env
.venv/
logs/
__pycache__/
*.egg-info/
dist/
models/
data/master_resume.json
```

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "ats-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.25.0",
    "browser-use>=0.1.0",
    "playwright>=1.44.0",
    "langgraph>=0.2.0",
    "typer>=0.12.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
local-llm = ["llama-cpp-python>=0.2.0"]
dev = ["pytest", "pytest-asyncio", "ruff", "mypy"]

[project.scripts]
ats = "ats_agent.cli:app"
```

---

## Notes on Workday Specifically

Workday is the hardest ATS to automate reliably. Recommended approach:

1. Upload your resume PDF first — Workday will auto-parse and fill many fields
2. Agent then reviews each field and corrects parsing errors
3. Manually handle any remaining fields via human-in-the-loop
4. Do not attempt to fill Workday from scratch field-by-field — it's unreliable

This mirrors how the browser-use/playwright community handles Workday.
