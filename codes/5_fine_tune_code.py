import os
import json
import torch
from pathlib import Path
from collections import Counter
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel, is_bfloat16_supported

TASK_MODE = os.environ.get("TASK_MODE", "cls").lower()
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BASE_OUT = Path("icml/checkpoints_v4")

if TASK_MODE == "cls":
    DATA_DIR = Path("icml/firstpass_splits_cls")
    KEEP_TASKS = {"outcome_prediction", "revision_cycle_prediction"} 
    OUTPUT_DIR = BASE_OUT / "cls_adapter"
    MAX_SEQ_LENGTH = 12288
    BATCH_SIZE = 2
    GRAD_ACCUM = 8
    LR = 5e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 30
    LORA_RANK = 32
    LORA_ALPHA = 64
    SAVE_STEPS = 100
    EVAL_STEPS = 100
else:
    # Use teammate's SFT split
    DATA_DIR = Path("icml/firstpass_splits_sft")
    KEEP_TASKS = {"review_generation", "reviewer_updating"}
    OUTPUT_DIR = BASE_OUT / "sft_adapter"
    MAX_SEQ_LENGTH = 16384
    BATCH_SIZE = 1
    GRAD_ACCUM = 16
    LR = 5e-5
    NUM_EPOCHS = 1
    WARMUP_STEPS = 50
    LORA_RANK = 32
    LORA_ALPHA = 64
    SAVE_STEPS = 200
    EVAL_STEPS = 200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    print(f"[WARN] {DATA_DIR} not found, falling back to firstpass_splits_v3")
    DATA_DIR = Path("icml/firstpass_splits_v3")

print(f"""
  TASK_MODE : {TASK_MODE}
  DRY_RUN   : {DRY_RUN}
  MODEL     : {MODEL_NAME}
  DATA_DIR  : {DATA_DIR}
  OUTPUT    : {OUTPUT_DIR}
  LR        : {LR}
  LoRA rank : {LORA_RANK}
  Max seq   : {MAX_SEQ_LENGTH}
  Epochs    : {NUM_EPOCHS}
""")

print(f"Loading {MODEL_NAME} at BF16...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=True,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} ({trainable/total_params:.2%})")


SYS_PROMPT = {
    "outcome_prediction": (
        "You are a senior editor at Nature Communications. Based on the peer "
        "review dialogue provided, predict whether this manuscript will require "
        "a STANDARD revision cycle (2 rounds) or an EXTENDED revision cycle "
        "(3+ rounds). Consider the severity of reviewer concerns, number of "
        "outstanding issues, and the overall trajectory of the review. "
        "Answer with exactly one word: STANDARD or EXTENDED."
    ),
    "revision_cycle_prediction": (
        "You are a senior editor at Nature Communications. Based on the peer "
        "review dialogue provided, predict whether this manuscript will require "
        "a STANDARD revision cycle (2 rounds) or an EXTENDED revision cycle "
        "(3+ rounds). Consider the severity of reviewer concerns, number of "
        "outstanding issues, and the overall trajectory of the review. "
        "Answer with exactly one word: STANDARD or EXTENDED."
    ),
    "review_generation": (
        "You are a rigorous, constructive expert reviewer for Nature Communications. "
        "Write a detailed peer review covering significance, methodology, "
        "statistical rigor, and specific weaknesses."
    ),
    "reviewer_updating": (
        "You are a peer reviewer for Nature Communications writing a Round 2 review. "
        "Evaluate whether the authors satisfactorily addressed your Round 1 concerns."
    ),
}

VALID_CLS_LABELS = {"STANDARD", "EXTENDED", "MAJOR", "MINOR"}

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task") in KEEP_TASKS:
                data.append(r)
    return data


def format_record(record):
    task = record["task"]
    sys_prompt = SYS_PROMPT.get(task, "")
    target = record["target"]

    # Normalize CLS targets
    if task in {"outcome_prediction", "revision_cycle_prediction"}:
        target = target.strip().upper()
        if target not in VALID_CLS_LABELS:
            return None

    user_content = record["input"]

    # Task-aware char budget
    if task in {"outcome_prediction", "revision_cycle_prediction"}:
        max_input_chars = int(MAX_SEQ_LENGTH * 3.0) - 200
    else:
        max_input_chars = int(MAX_SEQ_LENGTH * 3.0) - len(target) - 500

    if max_input_chars < 2000:
        return None

    if len(user_content) > max_input_chars:
        head = int(max_input_chars * 0.55)
        tail = max_input_chars - head - 50
        user_content = user_content[:head] + "\n\n[... content truncated ...]\n\n" + user_content[-tail:]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


train_data = load_jsonl(DATA_DIR / "train.jsonl")
val_data = load_jsonl(DATA_DIR / "val.jsonl")
print(f"Loaded {len(train_data)} train, {len(val_data)} val examples")

if TASK_MODE == "cls":
    labels = Counter(r["target"] for r in train_data)
    print(f"Train label distribution: {labels}")
    label_srcs = Counter(r.get("label_src", "unknown") for r in train_data)
    print(f"Label sources: {label_srcs}")

train_formatted = [r for r in (format_record(x) for x in train_data) if r]
val_formatted = [r for r in (format_record(x) for x in val_data) if r]
print(f"After formatting: {len(train_formatted)} train, {len(val_formatted)} val")

train_ds = Dataset.from_list(train_formatted)
val_ds = Dataset.from_list(val_formatted)

if DRY_RUN:
    print("\n*** DRY RUN: slicing to 20 train / 10 val ***")
    train_ds = train_ds.select(range(min(20, len(train_ds))))
    val_ds = val_ds.select(range(min(10, len(val_ds))))

RESPONSE_PART = "<|im_start|>assistant\n"
INSTRUCTION_PART = "<|im_start|>user\n"

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    warmup_steps=WARMUP_STEPS,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    dataset_num_proc=2,
    packing=False,
    padding_free=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_args,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part=INSTRUCTION_PART,
    response_part=RESPONSE_PART,
)

print("\n=== Starting training (with response-only masking) ===")
result = trainer.train()

print(f"\n=== Training complete ===")
print(f"  Final train loss: {result.metrics.get('train_loss', 0):.4f}")

final_dir = OUTPUT_DIR / "final"
final_dir.mkdir(exist_ok=True)
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"Adapter saved to {final_dir}")

if TASK_MODE == "cls" and not DRY_RUN:
    print("\n=== Running test-set evaluation ===")
    FastLanguageModel.for_inference(model)

    # Load test data — accept both task names
    test_records = []
    with open(DATA_DIR / "test.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task") in KEEP_TASKS:
                test_records.append(r)
    print(f"Test examples: {len(test_records)}")

    from collections import defaultdict
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    y_true, y_pred = [], []
    per_domain = defaultdict(lambda: {"true": [], "pred": []})

    # Determine which label set we're using
    sample_labels = set(r["target"].strip().upper() for r in test_records[:50])
    if "MAJOR" in sample_labels or "MINOR" in sample_labels:
        label_set = ["MAJOR", "MINOR"]
        default_label = "MINOR"
    else:
        label_set = ["STANDARD", "EXTENDED"]
        default_label = "STANDARD"

    print(f"Label set detected: {label_set}")

    # Use the appropriate system prompt
    cls_task = test_records[0]["task"] if test_records else "outcome_prediction"
    sys_p = SYS_PROMPT.get(cls_task, SYS_PROMPT["outcome_prediction"])

    for i, rec in enumerate(test_records):
        true_label = rec["target"].strip().upper()

        user_content = rec["input"]
        max_chars = int(MAX_SEQ_LENGTH * 3.0) - 200
        if len(user_content) > max_chars:
            head = int(max_chars * 0.55)
            tail = max_chars - head - 50
            user_content = user_content[:head] + "\n\n[... truncated ...]\n\n" + user_content[-tail:]

        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_content},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        if inputs.shape[1] > MAX_SEQ_LENGTH - 10:
            inputs = inputs[:, :MAX_SEQ_LENGTH - 10]

        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip().upper()

        # Normalize prediction to valid label
        pred = default_label  # fallback
        for lbl in label_set:
            if lbl in generated:
                pred = lbl
                break

        y_true.append(true_label)
        y_pred.append(pred)
        domain = rec.get("domain", "unknown")
        per_domain[domain]["true"].append(true_label)
        per_domain[domain]["pred"].append(pred)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_records)} | acc: {accuracy_score(y_true, y_pred):.3f}")

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    majority_acc = Counter(y_true).most_common(1)[0][1] / len(y_true)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (with response-only masking)")
    print(f"{'='*60}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"F1 (macro):       {f1_macro:.4f}")
    print(f"Majority BL:      {majority_acc:.4f}")
    print(f"Improvement:      {acc - majority_acc:+.4f}")
    print("\n" + classification_report(y_true, y_pred, labels=label_set, digits=4))

    print("\nPer-domain:")
    for domain in sorted(per_domain.keys()):
        d = per_domain[domain]
        d_acc = accuracy_score(d["true"], d["pred"])
        print(f"  {domain}: {d_acc:.4f} (n={len(d['true'])})")

    eval_results = {
        "accuracy": round(acc, 4), "f1_macro": round(f1_macro, 4),
        "majority_baseline": round(majority_acc, 4),
        "label_set": label_set,
    }
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Eval saved to {OUTPUT_DIR / 'eval_results.json'}")