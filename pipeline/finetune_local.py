"""
Local fine-tuning using Unsloth + LoRA on Llama 3.1 8B.

Reads train.jsonl / val.jsonl produced by export_training_data.py,
fine-tunes, saves the merged model to models/job-agent/, and writes
the local path to .ft_model_id.

Usage:
    python -m pipeline.finetune_local --train data/train.jsonl --val data/val.jsonl --db runs.db
    python -m pipeline.finetune_local --train data/train.jsonl --val data/val.jsonl --db runs.db --dry-run
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Must run before importing torch/unsloth so CUDA libs are on the path
_env = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env)
_ldpath_parts = [
    os.getenv("TORCH_LIB", ""),
    os.getenv("CUDA_LIB", ""),
    os.environ.get("LD_LIBRARY_PATH", ""),
]
os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in _ldpath_parts if p)

from pipeline.finetune_openai import FT_MODEL_ID_FILE

BASE_MODEL   = "unsloth/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR   = "models/job-agent"
MAX_SEQ_LEN  = 2048
LORA_RANK    = 16
BATCH_SIZE   = 2
GRAD_ACCUM   = 4          # effective batch = 8
N_EPOCHS     = 3
LEARNING_RATE = 2e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _validate_jsonl(path: Path) -> None:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {i} is not valid JSON: {exc}") from exc
        if "messages" not in obj:
            raise ValueError(f"{path}: line {i} missing 'messages' key")


def _write_model_registry(db_path: str, model_id: str, train_examples: int) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO model_registry (model_id, last_trained_at, train_examples) "
            "VALUES (?, ?, ?)",
            (model_id, datetime.now(timezone.utc).isoformat(), train_examples),
        )


def _save_model_id(model_id: str) -> None:
    Path(FT_MODEL_ID_FILE).write_text(model_id)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(train_path: str, val_path: str) -> None:
    train = Path(train_path)
    val   = Path(val_path)
    _validate_jsonl(train)
    _validate_jsonl(val)
    n_train = _count_jsonl(train)
    n_val   = _count_jsonl(val)
    print(f"[DRY RUN] Would load base model : {BASE_MODEL}")
    print(f"[DRY RUN] Would train on        : {train_path} ({n_train} examples)")
    print(f"[DRY RUN] Would validate on     : {val_path} ({n_val} examples)")
    print(f"[DRY RUN] LoRA rank={LORA_RANK}, epochs={N_EPOCHS}, lr={LEARNING_RATE}")
    print(f"[DRY RUN] Would save model to   : {OUTPUT_DIR}")
    print(f"[DRY RUN] Would write model path to {FT_MODEL_ID_FILE} and model_registry")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_finetune(train_path: str, val_path: str, db_path: str) -> str:
    """
    Fine-tune Llama 3.1 8B with LoRA via Unsloth.
    Returns the local model path (written to .ft_model_id).
    """
    # unsloth must be imported first to patch transformers/trl
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train = Path(train_path)
    val   = Path(val_path)

    # ---- Load model + tokenizer ----
    print(f"Loading {BASE_MODEL} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # ---- Apply LoRA ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ---- Build datasets ----
    def _load(path: Path) -> Dataset:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        texts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
            for r in rows
        ]
        return Dataset.from_dict({"text": texts})

    print("Preparing datasets…")
    train_ds = _load(train)
    val_ds   = _load(val)

    # ---- Train ----
    steps_per_epoch = max(1, len(train_ds) // (BATCH_SIZE * GRAD_ACCUM))
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=max(1, steps_per_epoch // 5),
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=True,
        packing_strategy="bfd",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    print(f"Training for {N_EPOCHS} epochs …")
    trainer.train()

    # ---- Save merged model ----
    print(f"Saving merged model to {OUTPUT_DIR} …")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")

    model_path = str(Path(OUTPUT_DIR).resolve())
    _write_model_registry(db_path, model_path, _count_jsonl(train))
    _save_model_id(model_path)
    print(f"\nDone! Model saved to: {model_path}")
    print(f"Model path written to {FT_MODEL_ID_FILE} and model_registry.")
    return model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B locally with Unsloth.")
    parser.add_argument("--train",   required=True, help="Path to train.jsonl")
    parser.add_argument("--val",     required=True, help="Path to val.jsonl")
    parser.add_argument("--db",      required=True, help="Path to SQLite database")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate files and print plan without training")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.train, args.val)
    else:
        run_finetune(args.train, args.val, args.db)
