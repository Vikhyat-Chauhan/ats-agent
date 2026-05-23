"""
Maps form field labels to values using the active LLM.

On startup:
  - If .ft_model_id exists → use fine-tuned model (model_source="ft")
  - Otherwise → fall back to base model (model_source="base")
  - Pass --force-base to always use the base model (for debugging)

Shadow eval runs fire-and-forget in a background thread so it accumulates
data after promotion without blocking the fill loop.
"""

import concurrent.futures
import logging
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from pipeline.finetune_openai import BASE_MODEL, FT_MODEL_ID_FILE
from pipeline.shadow_eval import shadow_predict

load_dotenv()

log = logging.getLogger(__name__)

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="shadow")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _load_ft_model_id() -> str | None:
    p = Path(FT_MODEL_ID_FILE)
    if p.exists():
        text = p.read_text().strip()
        return text if text else None
    return None


def resolve_model(force_base: bool = False) -> tuple[str, str]:
    """
    Return (model_id, model_source).
    model_source is "ft" or "base".
    """
    if not force_base:
        ft_id = _load_ft_model_id()
        if ft_id:
            log.info("Using fine-tuned model: %s", ft_id)
            return ft_id, "ft"

    log.info("Using base model: %s", BASE_MODEL)
    return BASE_MODEL, "base"


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a job application assistant. Given a form field label and a resume JSON, "
    "return exactly the value to fill in that field. No explanation. No markdown. Just the value."
)


def map_field(
    field_label: str,
    resume_json: str,
    platform: str,
    db_path: str,
    *,
    force_base: bool = False,
    client: openai.OpenAI | None = None,
) -> dict:
    """
    Call the active model to fill a form field.

    Returns:
        {
            "value":        str,
            "model_id":     str,
            "model_source": "ft" | "base",
        }
    """
    if client is None:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    model_id, model_source = resolve_model(force_base=force_base)

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
    value = response.choices[0].message.content.strip()

    # Fire-and-forget shadow prediction (only when running ft model)
    if model_source == "ft":
        _fire_shadow(field_label, resume_json, platform, db_path,
                     model_id=model_id, client=client)

    return {"value": value, "model_id": model_id, "model_source": model_source}


def _fire_shadow(
    field_label: str,
    resume_json: str,
    platform: str,
    db_path: str,
    model_id: str,
    client: openai.OpenAI,
) -> None:
    """Submit shadow_predict to the thread pool; log but never raise."""
    def _task():
        try:
            shadow_predict(
                field_label, resume_json, platform, db_path,
                model_id=model_id, client=client,
            )
        except Exception as exc:
            log.warning("shadow_predict failed (non-fatal): %s", exc)

    _thread_pool.submit(_task)
