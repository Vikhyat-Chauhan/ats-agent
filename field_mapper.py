"""
Maps form field labels to values using the active LLM.

On startup:
  - If .ft_model_id exists → use fine-tuned model (model_source="ft")
    - If the value is a local path → use local Llama via Unsloth
    - If the value is an OpenAI model ID → use OpenAI API
  - Otherwise → fall back to base model via OpenAI API (model_source="base")
  - Pass force_base=True to always use the base model (for debugging)

Shadow eval runs fire-and-forget in a background thread so it accumulates
data after promotion without blocking the fill loop.
"""

import concurrent.futures
import logging
import os
import re
import warnings
from pathlib import Path

# Set before any transformers/torch import so their startup code respects it
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GLOG_minloglevel", "3")

import openai
from dotenv import load_dotenv

# Belt-and-suspenders: also suppress via Python warnings + logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)  # transformers captureWarnings sink
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

from pipeline.finetune_openai import BASE_MODEL, FT_MODEL_ID_FILE
from pipeline.shadow_eval import shadow_predict

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="shadow")

# Cached local pipeline to avoid reloading the model on every call
_local_pipeline = None


SYSTEM_PROMPT = (
    "You are a job application assistant. Given a form field label and a resume JSON, "
    "return exactly the value to fill in that field. No explanation. No markdown. Just the value."
)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _load_ft_model_id() -> str | None:
    p = Path(FT_MODEL_ID_FILE)
    if p.exists():
        text = p.read_text().strip()
        return text if text else None
    return None


def _is_local_model(model_id: str) -> bool:
    """True if model_id looks like a filesystem path rather than an OpenAI model ID."""
    return model_id.startswith("/") or model_id.startswith("./") or Path(model_id).exists()


def resolve_model(force_base: bool = False) -> tuple[str, str]:
    """Return (model_id, model_source). model_source is 'ft' or 'base'."""
    if not force_base:
        ft_id = _load_ft_model_id()
        if ft_id:
            log.info("Using fine-tuned model: %s", ft_id)
            return ft_id, "ft"
    log.info("Using base model: %s", BASE_MODEL)
    return BASE_MODEL, "base"


# ---------------------------------------------------------------------------
# Local inference
# ---------------------------------------------------------------------------

def _get_local_pipeline(model_path: str):
    global _local_pipeline
    if _local_pipeline is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        log.info("Loading local model from %s …", model_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()
        _local_pipeline = (model, tokenizer)
    return _local_pipeline


def _infer_local(model_path: str, messages: list[dict]) -> str:
    import torch
    model, tokenizer = _get_local_pipeline(model_path)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------

def map_field(
    field_label: str,
    resume_json: str,
    platform: str,
    db_path: str,
    *,
    field_type: str = "text",
    options: list[str] | None = None,
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
    model_id, model_source = resolve_model(force_base=force_base)

    # Strip placeholder options before showing choices to the LLM
    _PLACEHOLDER = re.compile(
        r"^(please\s+select|select\s*(one|an?\s+option)?|--+\s*select\s*--+|choose\s+one|n/?a)$",
        re.I,
    )
    real_options = [o for o in (options or []) if not _PLACEHOLDER.match(o.strip())]

    # Build a type-aware instruction so the LLM knows the expected format
    if field_type == "checkbox":
        type_hint = 'Field type: checkbox. Reply with exactly "yes" or "no".'
    elif field_type == "select" and real_options:
        opts = ", ".join(f'"{o}"' for o in real_options[:20])
        type_hint = f"Field type: dropdown. You MUST reply with exactly one of: {opts}"
    elif field_type in ("date",):
        type_hint = "Field type: date. Reply in YYYY-MM-DD format."
    elif field_type in ("number",):
        type_hint = "Field type: number. Reply with digits only."
    elif field_type == "textarea":
        type_hint = "Field type: textarea (multi-line text). Reply with a concise paragraph."
    else:
        type_hint = "Reply with only the value to fill in — no explanation, no markdown."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"ATS: {platform}\n"
                f"Field: {field_label}\n"
                f"{type_hint}\n"
                f"Resume:\n{resume_json}\n"
                "Value:"
            ),
        },
    ]

    if model_source == "ft" and _is_local_model(model_id):
        value = _infer_local(model_id, messages)
    else:
        if client is None:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0,
        )
        value = response.choices[0].message.content.strip()

    # Fire-and-forget shadow prediction when ft model is active (OpenAI only)
    if model_source == "ft" and not _is_local_model(model_id):
        _fire_shadow(field_label, resume_json, platform, db_path,
                     model_id=model_id, client=client)

    return {"value": value, "model_id": model_id, "model_source": model_source}


def _fire_shadow(
    field_label: str,
    resume_json: str,
    platform: str,
    db_path: str,
    model_id: str,
    client: openai.OpenAI | None,
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
