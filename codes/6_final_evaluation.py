import os
import json
import time
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional
from unsloth import FastLanguageModel
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, precision_recall_fscore_support,
)


def ensure_installed(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([
            "pip", "install", pip_name or package,
            "--break-system-packages", "-q",
        ])


ensure_installed("openai")
ensure_installed("statsmodels")
ensure_installed("rouge_score", "rouge-score")
ensure_installed("nltk")
ensure_installed("bert_score", "bert-score")

from openai import OpenAI
from statsmodels.stats.contingency_tables import mcnemar
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

CONFIG = {
    "test_jsonl":     "icml/firstpass_splits_cls/test.jsonl",
    "train_jsonl":    "icml/firstpass_splits_cls/train.jsonl",
    "sft_test_jsonl": "icml/firstpass_splits_sft/test.jsonl",

    "base_model":    "Qwen/Qwen2.5-7B-Instruct",
    "lora_masked":   "icml/checkpoints_v4/cls_adapter/final",
    "lora_no_mask":  "icml/checkpoints_v3/outcome_adapter/final",
    "sft_lora_path": "icml/checkpoints_v4/sft_adapter/final",

    "hf_token":          os.environ.get("HF_TOKEN", ""),
    "hf_base_url":       "https://router.huggingface.co/v1",
    "llama_model_id":    "meta-llama/Meta-Llama-3-8B-Instruct:novita",
    "deepseek_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:nscale",

    "gemini_api_key":   os.environ.get("GEMINI_API_KEY", ""),
    "gemini_model_cls": "gemini-3.1-flash-lite-preview",
    "gemini_model_gen": "gemini-3.1-flash-lite-preview",

    "max_seq_length":  12288,
    "max_new_tokens":  16,
    "few_shot_k":      5,
    "seed":            42,

    "max_new_tokens_local":  1500,
    "max_new_tokens_api":    1500,
    "max_input_chars_local": 24000,
    "max_input_chars_api":   16000,

    "llama_sleep_cls":    0.3,
    "deepseek_sleep_cls": 0.5,
    "gemini_sleep_cls":   2.0,   
    "llama_sleep_gen":    1.0,
    "deepseek_sleep_gen": 1.5,
    "gemini_sleep_gen":   3.0,

    "bertscore_model": "microsoft/deberta-xlarge-mnli",
    "bertscore_batch": 4,

    "n_bootstrap": 2000,
    "alpha":       0.05,

    "output_dir": "icml/eval_results",
}

SKIP_MODELS         = {m.strip().lower() for m in os.environ.get("SKIP_MODELS",         "").split(",") if m.strip()}
SKIP_BERTSCORE      = os.environ.get("SKIP_BERTSCORE",      "0") == "1"
SKIP_GENERATION     = os.environ.get("SKIP_GENERATION",     "0") == "1"
SKIP_CLASSIFICATION = os.environ.get("SKIP_CLASSIFICATION", "0") == "1"

DOMAINS = ["biology", "chemistry", "earth_science", "neuroscience", "physics"]

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

OUT     = Path(CONFIG["output_dir"])
GEN_OUT = OUT / "generation"
CKP_DIR = OUT / "checkpoints"

for d in [OUT / "tables", GEN_OUT / "tables", CKP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LABEL_SET      = ["STANDARD", "EXTENDED"]
MAJORITY_LABEL = "STANDARD"

ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def ckp_path(task: str, model_key: str) -> Path:
    return CKP_DIR / f"{task}__{model_key}.json"


def save_checkpoint(task: str, model_key: str, data: dict) -> None:
    p   = ckp_path(task, model_key)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(p)
    print(f"  [CKPT] Saved: {p.name}")


def load_checkpoint(task: str, model_key: str) -> Optional[dict]:
    p = ckp_path(task, model_key)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("y_pred", data.get("hypotheses", [])))
        print(f"  [RESUME] Loaded: {p.name}  ({n} items)")
        return data
    return None


def save_partial_checkpoint(task: str, model_key: str, data: dict) -> None:
    p   = ckp_path(task, f"{model_key}__partial")
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(p)


def load_partial_checkpoint(task: str, model_key: str) -> Optional[dict]:
    p = ckp_path(task, f"{model_key}__partial")
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("y_pred", data.get("hypotheses", [])))
        print(f"  [RESUME] Partial checkpoint: {p.name}  ({n} done)")
        return data
    return None


def clear_partial_checkpoint(task: str, model_key: str) -> None:
    p = ckp_path(task, f"{model_key}__partial")
    if p.exists():
        p.unlink()

CLS_CKPT_FREQ = 25
GEN_CKPT_FREQ = 10

CLS_TASK_NAMES = {"outcome_prediction", "revision_cycle_prediction"}

def load_jsonl(path, task_filter=None):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if task_filter is None or r.get("task") in task_filter:
                records.append(r)
    return records

def load_sft_test(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task") == "review_generation":
                records.append(r)
    return records

print("firstpass Combined Evaluation Pipeline")

print("\nLoading classification test data...")
cls_test_records = load_jsonl(CONFIG["test_jsonl"], task_filter=CLS_TASK_NAMES)
print(f"  {len(cls_test_records)} classification test examples")

sample_labels = {r["target"].strip().upper() for r in cls_test_records}
if "MAJOR" in sample_labels or "MINOR" in sample_labels:
    LABEL_SET      = ["MAJOR", "MINOR"]
    MAJORITY_LABEL = Counter(r["target"].upper() for r in cls_test_records).most_common(1)[0][0]
else:
    LABEL_SET      = ["STANDARD", "EXTENDED"]
    MAJORITY_LABEL = "STANDARD"

print(f"  Labels: {LABEL_SET}  |  Majority: {MAJORITY_LABEL}")
for lbl, cnt in sorted(Counter(r["target"].upper() for r in cls_test_records).items()):
    print(f"    {lbl}: {cnt} ({cnt / len(cls_test_records) * 100:.1f}%)")

print("\nLoading train data for few-shot pool...")
train_records = load_jsonl(CONFIG["train_jsonl"], task_filter=CLS_TASK_NAMES)
few_shot_pool: dict = defaultdict(list)
for r in train_records:
    few_shot_pool[r["target"].upper()].append(r)
print(f"  {len(train_records)} train records")

gen_test_records = []
gen_references   = []
ref_word_counts  = []
if not SKIP_GENERATION:
    print("\nLoading generation test data...")
    gen_test_records = load_sft_test(CONFIG["sft_test_jsonl"])
    if not gen_test_records:
        print(f"  [WARN] No review_generation records — SKIP_GENERATION set to True")
        SKIP_GENERATION = True
    else:
        gen_references  = [r["target"] for r in gen_test_records]
        ref_word_counts = [len(word_tokenize(ref)) for ref in gen_references]
        print(f"  {len(gen_test_records)} generation examples")
        print(f"  Avg reference length: {np.mean(ref_word_counts):.0f} words")
        for d, n in sorted(Counter(r.get("domain", "?") for r in gen_test_records).items()):
            print(f"    {d}: {n}")


def make_cls_sys_prompt() -> str:
    choices = " or ".join(LABEL_SET)
    return (
        "You are a senior editor at Nature Communications with deep expertise "
        "across biology, chemistry, earth science, neuroscience, and physics. "
        "Based on the peer review dialogue provided, predict the editorial outcome. "
        "Consider: severity of unresolved methodological concerns, number of "
        "outstanding reviewer requests, whether authors adequately addressed core "
        "issues, and the overall trajectory of the review dialogue. "
        f"Answer with exactly one word on the last line: {choices}."
    )

CLS_SYS_PROMPT = make_cls_sys_prompt()

GEN_SYS_PROMPT = (
    "You are assisting with a research study on automated scientific peer review. "
    "The following is an excerpt from a manuscript submitted to Nature Communications. "
    "As part of this NLP evaluation study, write the kind of detailed expert peer "
    "review that would appear in Nature Communications transparent peer review files. "
    "Your review must cover: "
    "(1) significance and novelty of the scientific contribution, "
    "(2) soundness of the methodology and experimental design, "
    "(3) quality and reproducibility of the results, "
    "(4) clarity and completeness of the reporting, "
    "(5) statistical rigor where applicable, "
    "(6) specific weaknesses that must be addressed before publication. "
    "Be specific — cite section names and claims where relevant. "
    "Write the full review text directly, without preamble."
)


def truncate_input(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.55)
    tail = max_chars - head - 80
    return text[:head] + "\n\n[... content truncated ...]\n\n" + text[-tail:]


def build_cls_user_message(record: dict, max_chars: int = 36000) -> str:
    return truncate_input(record["input"].strip(), max_chars)


def build_gen_prompt(record: dict, max_chars: int) -> str:
    return truncate_input(record["input"].strip(), max_chars)


def extract_label(generated_text: str) -> str:
    text  = generated_text.strip().upper()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for candidate in (lines[-1:] if lines else []) + lines:
        for lbl in LABEL_SET:
            if lbl in candidate:
                return lbl
    return MAJORITY_LABEL


def get_few_shot_examples(k: int = 5) -> list:
    examples  = []
    per_label = k // len(LABEL_SET)
    remainder = k % len(LABEL_SET)
    for i, lbl in enumerate(LABEL_SET):
        n    = per_label + (1 if i < remainder else 0)
        pool = few_shot_pool.get(lbl, [])
        examples.extend(random.sample(pool, min(n, len(pool))))
    random.shuffle(examples)
    return examples


def safe_gemini_text(response) -> str:
    """
    FIX 2: Safe text extraction from Gemini response.

    response.text raises ValueError when:
      - candidates list is empty (safety block)
      - finish_reason is SAFETY, RECITATION, or OTHER
      - content.parts is empty

    This function handles all those cases gracefully and returns
    an empty string instead of crashing, so the retry/fallback
    logic can handle it properly.
    """
    try:
        candidates = response.candidates
        if not candidates:
            return ""
        candidate = candidates[0]
        # Check finish reason — STOP is normal, anything else may be blocked
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_name   = getattr(finish_reason, "name", str(finish_reason)) if finish_reason else "UNKNOWN"
        if finish_name not in ("STOP", "MAX_TOKENS", "1", "2"):
            # 1=STOP, 2=MAX_TOKENS in protobuf enum
            print(f"    [GEMINI] finish_reason={finish_name} — empty text returned")
            return ""
        content = getattr(candidate, "content", None)
        if content is None:
            return ""
        parts = getattr(content, "parts", [])
        if not parts:
            return ""
        return parts[0].text or ""
    except (ValueError, AttributeError, IndexError) as e:
        print(f"    [GEMINI] safe_gemini_text fallback: {e}")
        return ""

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05):
    arr_t  = np.array(y_true)
    arr_p  = np.array(y_pred)
    n      = len(arr_t)
    scores = [metric_fn(arr_t[idx := np.random.randint(0, n, n)], arr_p[idx])
              for _ in range(n_boot)]
    point  = float(metric_fn(arr_t, arr_p))
    return point, float(np.percentile(scores, 100*alpha/2)), float(np.percentile(scores, 100*(1-alpha/2)))


def mcnemar_test(pred_a, pred_b, y_true):
    ca   = np.array([p == t for p, t in zip(pred_a, y_true)])
    cb   = np.array([p == t for p, t in zip(pred_b, y_true)])
    b    = int(np.sum( ca & ~cb))
    c    = int(np.sum(~ca &  cb))
    n11  = int(np.sum( ca &  cb))
    n00  = int(np.sum(~ca & ~cb))
    exact = (b + c) < 25
    try:
        res   = mcnemar(np.array([[n11, b], [c, n00]]), exact=exact, correction=(not exact))
        p_val = float(res.pvalue)
        stat  = float(res.statistic)
    except Exception:
        p_val = float("nan")
        stat  = float("nan")
    return {
        "n_comparison_only_correct": b,
        "n_ours_only_correct":       c,
        "statistic":                 round(stat,  4) if not np.isnan(stat)  else None,
        "p_value":                   round(p_val, 6) if not np.isnan(p_val) else None,
        "significant_at_0.05":       bool(p_val < 0.05) if not np.isnan(p_val) else None,
        "exact_test":                exact,
    }


def compute_all_cls_metrics(y_true, y_pred):
    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_cls  = {
        lbl: float(f1_score(y_true, y_pred, labels=[lbl], average="micro", zero_division=0))
        for lbl in LABEL_SET
    }
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_SET, average="macro", zero_division=0
    )
    _, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score)
    _, f1_lo,  f1_hi  = bootstrap_ci(
        y_true, y_pred,
        lambda t, p: f1_score(t, p, average="macro", zero_division=0),
    )
    return {
        "accuracy":         round(float(acc),        4),
        "accuracy_ci":      [round(acc_lo,4), round(acc_hi,4)],
        "f1_macro":         round(float(f1_macro),   4),
        "f1_macro_ci":      [round(f1_lo, 4), round(f1_hi, 4)],
        "f1_weighted":      round(float(f1_weighted),4),
        "precision_macro":  round(float(prec),       4),
        "recall_macro":     round(float(rec),        4),
        "f1_per_class":     {k: round(v,4) for k,v in f1_per_cls.items()},
        "n_correct":        int(sum(p==t for p,t in zip(y_pred,y_true))),
        "n_total":          len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_SET).tolist(),
    }


def compute_rouge_corpus(hypotheses, refs):
    r1, r2, rL = [], [], []
    for h, r in zip(hypotheses, refs):
        if not h.strip():
            r1.append(0.0); r2.append(0.0); rL.append(0.0)
            continue
        s = ROUGE.score(r, h)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {"rouge1": r1, "rouge2": r2, "rougeL": rL}


def bootstrap_ci_scores(scores, n_boot=2000, alpha=0.05):
    arr  = np.array(scores)
    n    = len(arr)
    boot = [np.mean(arr[np.random.randint(0,n,n)]) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.percentile(boot,100*alpha/2)), float(np.percentile(boot,100*(1-alpha/2)))


def compute_bertscore(hypotheses, refs):
    from bert_score import score as bsf
    print(f"    Computing BERTScore (batch={CONFIG['bertscore_batch']})...")
    _, _, F1 = bsf(
        hypotheses, refs,
        model_type            = CONFIG["bertscore_model"],
        batch_size            = CONFIG["bertscore_batch"],
        lang                  = "en",
        rescale_with_baseline = True,
        device                = "cuda" if torch.cuda.is_available() else "cpu",
        verbose               = False,
    )
    return F1.tolist()


def paired_bootstrap_test(scores_a, scores_b, n_boot=2000):
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    n     = len(arr_a)
    obs   = float(np.mean(arr_b) - np.mean(arr_a))
    c     = (arr_b - arr_a) - np.mean(arr_b - arr_a)
    boot  = np.array([float(np.mean(c[np.random.randint(0,n,n)])) for _ in range(n_boot)])
    p     = max(float(np.mean(np.abs(boot) >= abs(obs))), 1.0/n_boot)
    return {
        "observed_diff_mean":  round(obs, 6),
        "p_value":             round(p,   6),
        "n_bootstrap":         n_boot,
        "alternative":         "two-sided",
        "significant_at_0.05": bool(p < 0.05),
        "significant_at_0.01": bool(p < 0.01),
    }


def compute_length_stats(texts):
    lens = [len(word_tokenize(t)) for t in texts if t.strip()]
    if not lens:
        return {"mean":0,"median":0,"std":0,"min":0,"max":0}
    return {
        "mean":   round(float(np.mean(lens)),  1),
        "median": round(float(np.median(lens)),1),
        "std":    round(float(np.std(lens)),   1),
        "min":    int(np.min(lens)),
        "max":    int(np.max(lens)),
    }


def compute_vocab_diversity(texts):
    ttrs = []
    for t in texts:
        if not t.strip(): continue
        toks = word_tokenize(t.lower())
        if toks: ttrs.append(len(set(toks))/len(toks))
    return round(float(np.mean(ttrs)) if ttrs else 0.0, 4)


def compute_generation_metrics(hypotheses, refs, compute_bs=True):
    rs = compute_rouge_corpus(hypotheses, refs)
    r1_pt,r1_lo,r1_hi = bootstrap_ci_scores(rs["rouge1"], CONFIG["n_bootstrap"], CONFIG["alpha"])
    r2_pt,r2_lo,r2_hi = bootstrap_ci_scores(rs["rouge2"], CONFIG["n_bootstrap"], CONFIG["alpha"])
    rL_pt,rL_lo,rL_hi = bootstrap_ci_scores(rs["rougeL"], CONFIG["n_bootstrap"], CONFIG["alpha"])
    bs_scores = bs_mean = bs_lo = bs_hi = None
    if compute_bs and not SKIP_BERTSCORE:
        try:
            bs_scores           = compute_bertscore(hypotheses, refs)
            bs_mean,bs_lo,bs_hi = bootstrap_ci_scores(bs_scores, CONFIG["n_bootstrap"], CONFIG["alpha"])
            bs_mean,bs_lo,bs_hi = round(bs_mean,4), round(bs_lo,4), round(bs_hi,4)
        except Exception as e:
            print(f"    [WARN] BERTScore failed: {e}")
    return {
        "n_examples":        len(hypotheses),
        "n_empty":           sum(1 for h in hypotheses if not h.strip()),
        "rouge1":            round(r1_pt,4), "rouge1_ci": [round(r1_lo,4),round(r1_hi,4)],
        "rouge2":            round(r2_pt,4), "rouge2_ci": [round(r2_lo,4),round(r2_hi,4)],
        "rougeL":            round(rL_pt,4), "rougeL_ci": [round(rL_lo,4),round(rL_hi,4)],
        "bertscore_f1":      bs_mean,
        "bertscore_f1_ci":   [bs_lo,bs_hi] if bs_lo is not None else None,
        "length_stats":      compute_length_stats(hypotheses),
        "vocab_ttr":         compute_vocab_diversity(hypotheses),
        "_rouge1_scores":    rs["rouge1"],
        "_rouge2_scores":    rs["rouge2"],
        "_rougeL_scores":    rs["rougeL"],
        "_bertscore_scores": bs_scores,
    }


def compute_per_domain_cls(y_true, y_pred, records):
    dd = defaultdict(lambda: {"true":[],"pred":[]})
    for t, p, r in zip(y_true, y_pred, records):
        dd[r.get("domain","unknown")]["true"].append(t)
        dd[r.get("domain","unknown")]["pred"].append(p)
    results = {}
    for domain, data in sorted(dd.items()):
        dt, dp = data["true"], data["pred"]
        if not dt: continue
        acc = float(accuracy_score(dt,dp))
        f1  = float(f1_score(dt,dp,average="macro",zero_division=0))
        f1c = {lbl: float(f1_score(dt,dp,labels=[lbl],average="micro",zero_division=0)) for lbl in LABEL_SET}
        _,alo,ahi = bootstrap_ci(dt,dp,accuracy_score,n_boot=1000)
        results[domain] = {
            "n":len(dt), "accuracy":round(acc,4),
            "accuracy_ci":[round(alo,4),round(ahi,4)],
            "f1_macro":round(f1,4),
            "f1_per_class":{k:round(v,4) for k,v in f1c.items()},
            "label_dist":dict(Counter(dt)),
        }
    return results


def compute_per_domain_rouge(hypotheses, refs, records):
    dd = defaultdict(lambda: {"hyps":[],"refs":[]})
    for h, r, rec in zip(hypotheses, refs, records):
        dd[rec.get("domain","unknown")]["hyps"].append(h)
        dd[rec.get("domain","unknown")]["refs"].append(r)
    results = {}
    for domain, data in sorted(dd.items()):
        if not data["hyps"]: continue
        rs = compute_rouge_corpus(data["hyps"],data["refs"])
        r1_pt,r1_lo,r1_hi = bootstrap_ci_scores(rs["rouge1"],n_boot=1000)
        r2_pt,r2_lo,r2_hi = bootstrap_ci_scores(rs["rouge2"],n_boot=1000)
        rL_pt,rL_lo,rL_hi = bootstrap_ci_scores(rs["rougeL"],n_boot=1000)
        results[domain] = {
            "n":len(data["hyps"]),
            "rouge1":round(r1_pt,4),"rouge1_ci":[round(r1_lo,4),round(r1_hi,4)],
            "rouge2":round(r2_pt,4),"rouge2_ci":[round(r2_lo,4),round(r2_hi,4)],
            "rougeL":round(rL_pt,4),"rougeL_ci":[round(rL_lo,4),round(rL_hi,4)],
            "avg_len":compute_length_stats(data["hyps"])["mean"],
        }
    return results


def load_unsloth_model(model_name_or_path: str, max_seq_length: int = None):
    if max_seq_length is None:
        max_seq_length = CONFIG["max_seq_length"]
    print(f"  Loading via Unsloth: {model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name_or_path,
        max_seq_length = max_seq_length,
        dtype          = torch.bfloat16,
        load_in_4bit   = False,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_local_cls_inference(model, tokenizer, test_records,
                             few_shot_examples=None, model_label="model",
                             partial_ckp=None):
    max_input_chars = int(CONFIG["max_seq_length"] * 3.0) - 800
    if partial_ckp:
        y_true, y_pred    = partial_ckp["y_true"], partial_ckp["y_pred"]
        completed         = set(partial_ckp["completed_indices"])
    else:
        y_true = y_pred = []
        y_true, y_pred    = [], []
        completed         = set()

    for i, record in enumerate(test_records):
        if i in completed:
            continue
        true_label = record["target"].strip().upper()
        user_msg   = build_cls_user_message(record, max_input_chars)

        if few_shot_examples:
            per_ex = max_input_chars // (len(few_shot_examples) + 1)
            messages = [{"role":"system","content":CLS_SYS_PROMPT}]
            for ex in few_shot_examples:
                messages.append({"role":"user","content":build_cls_user_message(ex,per_ex)})
                messages.append({"role":"assistant","content":ex["target"].strip().upper()})
            messages.append({"role":"user","content":user_msg})
        else:
            messages = [
                {"role":"system","content":CLS_SYS_PROMPT},
                {"role":"user",  "content":user_msg},
            ]

        inputs = tokenizer.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=True,return_tensors="pt"
        ).to("cuda")
        if inputs.shape[1] > CONFIG["max_seq_length"] - 20:
            inputs = inputs[:, -(CONFIG["max_seq_length"]-20):]

        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens = CONFIG["max_new_tokens"],
                do_sample      = False,
                temperature    = None,
                top_p          = None,
                pad_token_id   = tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        y_true.append(true_label)
        y_pred.append(extract_label(generated))
        completed.add(i)

        if (i+1) % CLS_CKPT_FREQ == 0:
            save_partial_checkpoint("cls", model_label, {
                "y_true":y_true,"y_pred":y_pred,"completed_indices":sorted(completed)
            })
            print(f"    [{model_label}] {i+1}/{len(test_records)} | acc: {accuracy_score(y_true,y_pred):.3f}")

    return y_true, y_pred


def run_hf_cls_inference(model_id, test_records, model_label="hf",
                          sleep_secs=0.3, partial_ckp=None):
    client = OpenAI(base_url=CONFIG["hf_base_url"], api_key=CONFIG["hf_token"])
    if partial_ckp:
        y_true, y_pred = partial_ckp["y_true"], partial_ckp["y_pred"]
        completed      = set(partial_ckp["completed_indices"])
    else:
        y_true, y_pred = [], []
        completed      = set()

    for i, record in enumerate(test_records):
        if i in completed: continue
        true_label = record["target"].strip().upper()
        user_msg   = build_cls_user_message(record, max_chars=24000)
        messages   = [{"role":"system","content":CLS_SYS_PROMPT},
                      {"role":"user",  "content":user_msg}]
        generated  = ""
        for attempt in range(4):
            try:
                completion = client.chat.completions.create(
                    model=model_id, messages=messages,
                    max_tokens=CONFIG["max_new_tokens"], temperature=0.0,
                )
                generated = completion.choices[0].message.content or ""
                break
            except Exception as e:
                wait = 2**attempt
                if attempt == 3: print(f"    [WARN] {model_label} #{i} failed: {e}")
                else: time.sleep(wait)

        y_true.append(true_label)
        y_pred.append(extract_label(generated))
        completed.add(i)

        if (i+1) % CLS_CKPT_FREQ == 0:
            save_partial_checkpoint("cls", model_label, {
                "y_true":y_true,"y_pred":y_pred,"completed_indices":sorted(completed)
            })
            print(f"    [{model_label}] {i+1}/{len(test_records)} | acc: {accuracy_score(y_true,y_pred):.3f}")
        time.sleep(sleep_secs)

    return y_true, y_pred


def run_gemini_cls_inference(test_records, api_key, model_name,
                              model_label="gemini", partial_ckp=None):
    """
    FIX 1: max_input_chars raised from 8000 → 24000 so full review
    dialogues are passed to the model, not truncated ~25% excerpts.
    FIX 2: uses safe_gemini_text() for response extraction.
    """
    ensure_installed("google.generativeai", "google-generativeai")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gem_model  = genai.GenerativeModel(model_name)
    gen_config = genai.types.GenerationConfig(
        max_output_tokens = 128,   # enough for 1 word + any thinking prefix
        temperature       = 0.0,
    )

    max_input_chars = 24000

    if partial_ckp:
        y_true, y_pred = partial_ckp["y_true"], partial_ckp["y_pred"]
        completed      = set(partial_ckp["completed_indices"])
    else:
        y_true, y_pred = [], []
        completed      = set()

    for i, record in enumerate(test_records):
        if i in completed: continue
        true_label = record["target"].strip().upper()
        user_msg   = build_cls_user_message(record, max_chars=max_input_chars)
        prompt     = f"{CLS_SYS_PROMPT}\n\n{user_msg}"

        generated  = ""
        for attempt in range(4):
            try:
                response  = gem_model.generate_content(prompt, generation_config=gen_config)
                generated = safe_gemini_text(response)
                break
            except Exception as e:
                wait = 5*(attempt+1)
                if attempt == 3: print(f"    [WARN] Gemini cls #{i} failed: {e}")
                else:
                    print(f"    [RETRY {attempt+1}] Gemini cls #{i}: {e}")
                    time.sleep(wait)

        y_true.append(true_label)
        y_pred.append(extract_label(generated))
        completed.add(i)

        if (i+1) % CLS_CKPT_FREQ == 0:
            save_partial_checkpoint("cls", model_label, {
                "y_true":y_true,"y_pred":y_pred,"completed_indices":sorted(completed)
            })
            print(f"    [{model_label}] {i+1}/{len(test_records)} | acc: {accuracy_score(y_true,y_pred):.3f}")
        time.sleep(CONFIG["gemini_sleep_cls"])

    return y_true, y_pred


def run_local_gen_inference(model, tokenizer, test_records,
                             model_label="model", partial_ckp=None):
    max_chars = CONFIG["max_input_chars_local"]
    if partial_ckp:
        hypotheses = partial_ckp["hypotheses"]
        completed  = set(partial_ckp["completed_indices"])
    else:
        hypotheses, completed = [], set()

    for i, record in enumerate(test_records):
        if i in completed: continue
        messages = [{"role":"system","content":GEN_SYS_PROMPT},
                    {"role":"user",  "content":build_gen_prompt(record,max_chars)}]
        inputs   = tokenizer.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=True,return_tensors="pt"
        )
        keep = CONFIG["max_seq_length"] - CONFIG["max_new_tokens_local"] - 10
        if inputs.shape[1] > keep:
            inputs = inputs[:, -keep:]
        inputs = inputs.to(next(model.parameters()).device)

        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens     = CONFIG["max_new_tokens_local"],
                do_sample          = False,
                temperature        = None,
                top_p              = None,
                repetition_penalty = 1.1,
                pad_token_id       = tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs.shape[1]:],skip_special_tokens=True).strip()
        hypotheses.append(generated)
        completed.add(i)

        if (i+1) % GEN_CKPT_FREQ == 0:
            save_partial_checkpoint("gen", model_label, {
                "hypotheses":hypotheses,"completed_indices":sorted(completed)
            })
            done = [h for h in hypotheses if h]
            avg  = np.mean([len(word_tokenize(h)) for h in done]) if done else 0
            print(f"    [{model_label}] {i+1}/{len(test_records)} | avg len: {avg:.0f}w")

    return hypotheses


def run_hf_gen_inference(model_id, test_records, model_label="hf",
                          sleep_secs=1.0, partial_ckp=None):
    client    = OpenAI(base_url=CONFIG["hf_base_url"], api_key=CONFIG["hf_token"])
    max_chars = CONFIG["max_input_chars_api"]
    if partial_ckp:
        hypotheses = partial_ckp["hypotheses"]
        completed  = set(partial_ckp["completed_indices"])
    else:
        hypotheses, completed = [], set()

    for i, record in enumerate(test_records):
        if i in completed: continue
        messages  = [{"role":"system","content":GEN_SYS_PROMPT},
                     {"role":"user",  "content":build_gen_prompt(record,max_chars)}]
        generated = ""
        for attempt in range(4):
            try:
                comp      = client.chat.completions.create(
                    model=model_id,messages=messages,
                    max_tokens=CONFIG["max_new_tokens_api"],temperature=0.0,
                )
                generated = (comp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                wait = 2**attempt
                if attempt == 3: print(f"    [WARN] {model_label} gen #{i} failed: {e}")
                else: time.sleep(wait)

        hypotheses.append(generated)
        completed.add(i)

        if (i+1) % GEN_CKPT_FREQ == 0:
            save_partial_checkpoint("gen", model_label, {
                "hypotheses":hypotheses,"completed_indices":sorted(completed)
            })
            done = [h for h in hypotheses if h]
            avg  = np.mean([len(word_tokenize(h)) for h in done]) if done else 0
            print(f"    [{model_label}] {i+1}/{len(test_records)} | avg len: {avg:.0f}w")
        time.sleep(sleep_secs)

    return hypotheses


def run_gemini_gen_inference(test_records, api_key, model_name,
                              model_label="gemini", partial_ckp=None):
    """
    FIX 2: uses safe_gemini_text() for response extraction (was response.text
            which raises ValueError on safety blocks, producing empty outputs).
    FIX 3: GEN_SYS_PROMPT includes research framing to avoid safety refusals.
    """
    ensure_installed("google.generativeai", "google-generativeai")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gem_model  = genai.GenerativeModel(model_name)
    gen_config = genai.types.GenerationConfig(
        max_output_tokens = CONFIG["max_new_tokens_api"],
        temperature       = 0.0,
    )
    max_chars = CONFIG["max_input_chars_api"]

    if partial_ckp:
        hypotheses = partial_ckp["hypotheses"]
        completed  = set(partial_ckp["completed_indices"])
    else:
        hypotheses, completed = [], set()

    for i, record in enumerate(test_records):
        if i in completed: continue
        prompt    = f"{GEN_SYS_PROMPT}\n\n{build_gen_prompt(record,max_chars)}"
        generated = ""
        for attempt in range(4):
            try:
                response  = gem_model.generate_content(prompt, generation_config=gen_config)
                generated = safe_gemini_text(response)
                if not generated.strip():
                    print(f"    [GEMINI GEN] Empty output on record {i} — will record as empty")
                break
            except Exception as e:
                wait = 5*(attempt+1)
                if attempt == 3: print(f"    [WARN] Gemini gen #{i} failed: {e}")
                else:
                    print(f"    [RETRY {attempt+1}] Gemini gen #{i}: {e}")
                    time.sleep(wait)

        hypotheses.append(generated)
        completed.add(i)

        if (i+1) % GEN_CKPT_FREQ == 0:
            save_partial_checkpoint("gen", model_label, {
                "hypotheses":hypotheses,"completed_indices":sorted(completed)
            })
            non_empty = [h for h in hypotheses if h.strip()]
            avg = np.mean([len(word_tokenize(h)) for h in non_empty]) if non_empty else 0
            n_empty = len(hypotheses) - len(non_empty)
            print(f"    [{model_label}] {i+1}/{len(test_records)} | avg len: {avg:.0f}w | empty: {n_empty}")
        time.sleep(CONFIG["gemini_sleep_gen"])

    # Final summary
    n_empty = sum(1 for h in hypotheses if not h.strip())
    if n_empty > 0:
        print(f"  [GEMINI GEN] WARNING: {n_empty}/{len(hypotheses)} outputs were empty "
              f"(safety blocks or API errors). Consider checking raw_generations.json.")

    return hypotheses


cls_predictions: dict = {}
cls_metrics:     dict = {}
gen_generations: dict = {}
gen_metrics:     dict = {}


def register_cls(model_key, y_true, y_pred):
    if not y_true:
        print(f"  [SKIP] No cls predictions for {model_key}")
        return
    cls_predictions[model_key] = {"y_true":y_true,"y_pred":y_pred}
    cls_metrics[model_key]     = compute_all_cls_metrics(y_true,y_pred)
    mm  = cls_metrics[model_key]
    acc = mm["accuracy"]*100
    lo,hi = mm["accuracy_ci"][0]*100, mm["accuracy_ci"][1]*100
    print(f"  -> {model_key}: acc={acc:.1f}% [{lo:.1f}–{hi:.1f}]  F1={mm['f1_macro']*100:.1f}%")
    save_checkpoint("cls", model_key, {"y_true":y_true,"y_pred":y_pred})


def register_gen(model_key, hypotheses):
    if not hypotheses:
        print(f"  [SKIP] No generations for {model_key}")
        return
    gen_generations[model_key] = hypotheses
    print(f"  Computing ROUGE for {model_key}...")
    metrics = compute_generation_metrics(hypotheses, gen_references)
    gen_metrics[model_key] = metrics
    print(f"  -> {model_key}: R1={metrics['rouge1']:.3f}  R2={metrics['rouge2']:.3f}  "
          f"RL={metrics['rougeL']:.3f}  len={metrics['length_stats']['mean']:.0f}w  "
          f"empty={metrics['n_empty']}")
    save_checkpoint("gen", model_key, {"hypotheses":hypotheses})


def run_cls_with_resume(model_key, fn, *args, **kwargs):
    ckp = load_checkpoint("cls", model_key)
    if ckp:
        register_cls(model_key, ckp["y_true"], ckp["y_pred"])
        return
    partial = load_partial_checkpoint("cls", model_key)
    kwargs["partial_ckp"] = partial
    y_true, y_pred = fn(*args, **kwargs)
    clear_partial_checkpoint("cls", model_key)
    register_cls(model_key, y_true, y_pred)


def run_gen_with_resume(model_key, fn, *args, **kwargs):
    ckp = load_checkpoint("gen", model_key)
    if ckp:
        register_gen(model_key, ckp["hypotheses"])
        return
    partial = load_partial_checkpoint("gen", model_key)
    kwargs["partial_ckp"] = partial
    hypotheses = fn(*args, **kwargs)
    clear_partial_checkpoint("gen", model_key)
    register_gen(model_key, hypotheses)


CLS_MODEL_ORDER = [
    ("majority",          "Majority baseline"),
    ("qwen_zeroshot",     "Qwen2.5-7B-Instruct (zero-shot)"),
    ("qwen_fewshot",      "Qwen2.5-7B-Instruct (5-shot)"),
    ("llama_zeroshot",    "Llama-3-8B-Instruct (zero-shot)"),
    ("deepseek_zeroshot", "DeepSeek-R1-Distill-Qwen-7B (zero-shot)"),
    ("gemini_zeroshot",   "Gemini-3.1-flash-lite-preview (zero-shot)"),
    ("qwen_lora_nomask",  "Qwen2.5-7B + LoRA (no masking)"),
    ("qwen_lora_masked",  "\\textbf{Qwen2.5-7B + LoRA (ours)}"),
]

GEN_MODEL_ORDER = [
    ("qwen_zeroshot",     "Qwen2.5-7B-Instruct (zero-shot)"),
    ("llama_zeroshot",    "Llama-3-8B-Instruct (zero-shot)"),
    ("deepseek_zeroshot", "DeepSeek-R1-Distill-Qwen-7B (zero-shot)"),
    ("gemini_zeroshot",   "Gemini-3.1-flash-lite-preview (zero-shot)"),
    ("qwen_sft",          "\\textbf{Qwen2.5-7B + LoRA SFT (ours)}"),
]


def minority_label():
    return [l for l in LABEL_SET if l != MAJORITY_LABEL][0]


def _p_str(p, key, is_ours=False):
    if is_ours or p is None: return "---"
    if p < 0.001: return f"{p:.4f}$\\S$"
    if p < 0.01:  return f"{p:.4f}$\\ddagger$"
    if p < 0.05:  return f"{p:.4f}$\\dagger$"
    return f"{p:.4f}"


def build_main_table_latex(metrics, mc_tests):
    n   = list(metrics.values())[0]["n_total"] if metrics else "?"
    ml  = minority_label()
    ls  = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        f"\\caption{{Revision cycle prediction on \\textsc{{firstpass}} test set ($n={n}$). "
        "Accuracy and F1-macro with 95\\% bootstrap CIs. "
        f"F1-{ml.title()} = per-class F1 for the minority class. "
        "McNemar $p$ vs.\\ our model ($\\dagger$ p$<$0.05, $\\ddagger$ p$<$0.01, $\\S$ p$<$0.001).}}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        f"\\textbf{{Model}} & \\textbf{{Acc (\\%)}} & \\textbf{{F1-macro (\\%)}} "
        f"& \\textbf{{F1-{ml.title()} (\\%)}} & \\textbf{{McNemar $p$}} \\\\",
        "\\midrule",
    ]
    for key, name in CLS_MODEL_ORDER:
        if key not in metrics: continue
        if key == "qwen_lora_nomask": ls.append("\\midrule")
        m    = metrics[key]
        acc  = m["accuracy"]*100
        alo  = m["accuracy_ci"][0]*100
        ahi  = m["accuracy_ci"][1]*100
        f1   = m["f1_macro"]*100
        flo  = m["f1_macro_ci"][0]*100
        fhi  = m["f1_macro_ci"][1]*100
        f1m  = m["f1_per_class"].get(ml,0)*100
        p    = mc_tests.get(key,{}).get("p_value")
        ps   = _p_str(p, key, key=="qwen_lora_masked")
        if key == "majority":
            ls.append(f"  {name} & {acc:.1f} & --- & 0.0 & {ps} \\\\")
        else:
            ls.append(
                f"  {name} & {acc:.1f} [{alo:.1f}--{ahi:.1f}] "
                f"& {f1:.1f} [{flo:.1f}--{fhi:.1f}] & {f1m:.1f} & {ps} \\\\"
            )
    ls += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(ls)


def build_cls_domain_table_latex(per_domain):
    ml = minority_label()
    ls = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Per-domain classification results for our fine-tuned model. "
        "95\\% bootstrap CIs for accuracy.}",
        "\\label{tab:domain_results}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        f"\\textbf{{Domain}} & \\textbf{{$n$}} & \\textbf{{Acc (\\%)}} "
        f"& \\textbf{{F1-macro}} & \\textbf{{F1-{ml.title()}}} \\\\",
        "\\midrule",
    ]
    for domain in DOMAINS:
        if domain not in per_domain: continue
        dm  = per_domain[domain]
        acc = dm["accuracy"]*100
        lo  = dm["accuracy_ci"][0]*100
        hi  = dm["accuracy_ci"][1]*100
        f1  = dm["f1_macro"]*100
        f1m = dm["f1_per_class"].get(ml,0)*100
        ls.append(
            f"  {domain.replace('_',' ').title()} & {dm['n']} "
            f"& {acc:.1f} [{lo:.1f}--{hi:.1f}] & {f1:.1f} & {f1m:.1f} \\\\"
        )
    ls += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(ls)


def build_ablation_table_latex(metrics):
    maj_acc = metrics.get("majority",{}).get("accuracy",0)*100
    rows    = [
        ("qwen_zeroshot",    "No fine-tuning (zero-shot)"),
        ("qwen_lora_nomask", "LoRA without response masking"),
        ("qwen_lora_masked", "\\textbf{LoRA with response masking (ours)}"),
    ]
    ls = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Ablation: effect of response-only loss masking. "
        "All rows use Qwen2.5-7B-Instruct as the base model.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "\\textbf{Configuration} & \\textbf{Acc (\\%)} "
        "& \\textbf{F1-macro (\\%)} & \\textbf{$\\Delta$ vs majority} \\\\",
        "\\midrule",
    ]
    for key, name in rows:
        if key not in metrics:
            ls.append(f"  {name} & --- & --- & --- \\\\")
            continue
        m  = metrics[key]
        a  = m["accuracy"]*100
        f  = m["f1_macro"]*100
        d  = a - maj_acc
        ls.append(f"  {name} & {a:.1f} & {f:.1f} & {('+' if d>=0 else '')}{d:.1f} \\\\")
    ls += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(ls)


def build_generation_table_latex(gm, st, ref_avg_len):
    n    = list(gm.values())[0]["n_examples"] if gm else "?"
    has_bs = any(m.get("bertscore_f1") is not None for m in gm.values())
    cols = "lcccc" if not has_bs else "lccccc"
    hdr  = ("\\textbf{Model} & \\textbf{ROUGE-1} & \\textbf{ROUGE-2} "
            "& \\textbf{ROUGE-L}")
    if has_bs: hdr += " & \\textbf{BERTScore}"
    hdr += " & \\textbf{Avg Len} \\\\"

    ls = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{Peer review generation on firstpass test set ($n={n}$). "
        "ROUGE-1/2/L with 95\\% bootstrap CIs. "
        f"Human reviews average {ref_avg_len:.0f} words. "
        "Bootstrap $p$ vs.\\ our model "
        "($\\dagger$ p$<$0.05, $\\ddagger$ p$<$0.01, $\\S$ p$<$0.001). "
        "Gemini-3.1-flash-lite-preview is excluded from significance testing due to high "
        "rate of safety-blocked outputs (see \\S\\ref{sec:results}).}}",
        "\\label{tab:generation_results}",
        f"\\begin{{tabular}}{{{cols}}}", "\\toprule", hdr, "\\midrule",
    ]

    def fmt(val, ci):
        if val is None: return "---"
        lo, hi = ci or [0,0]
        return f"{val:.3f} [{lo:.3f}--{hi:.3f}]"

    for key, name in GEN_MODEL_ORDER:
        if key not in gm: continue
        if key == "qwen_sft": ls.append("\\midrule")
        m   = gm[key]
        p   = st.get(key,{}).get("rougeL_test",{}).get("p_value")
        ps  = _p_str(p, key, key=="qwen_sft" or key=="gemini_zeroshot")
        r1  = fmt(m["rouge1"],m["rouge1_ci"])
        r2  = fmt(m["rouge2"],m["rouge2_ci"])
        rL  = fmt(m["rougeL"],m["rougeL_ci"])
        ln  = f"{m['length_stats']['mean']:.0f}"
        row = f"  {name} & {r1} & {r2} & {rL}"
        if has_bs: row += f" & {fmt(m.get('bertscore_f1'),m.get('bertscore_f1_ci'))}"
        row += f" & {ln} \\\\ % p={ps}"
        ls.append(row)

    ls += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(ls)


def build_gen_domain_table_latex(per_domain):
    ls = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Per-domain generation results for our fine-tuned model. "
        "95\\% bootstrap CIs for ROUGE-L.}",
        "\\label{tab:gen_domain}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "\\textbf{Domain} & \\textbf{N} & \\textbf{ROUGE-1} "
        "& \\textbf{ROUGE-2} & \\textbf{ROUGE-L} \\\\",
        "\\midrule",
    ]
    for domain in DOMAINS:
        if domain not in per_domain: continue
        dm = per_domain[domain]
        ls.append(
            f"  {domain.replace('_',' ').title()} & {dm['n']} "
            f"& {dm['rouge1']:.3f} [{dm['rouge1_ci'][0]:.3f}--{dm['rouge1_ci'][1]:.3f}] "
            f"& {dm['rouge2']:.3f} [{dm['rouge2_ci'][0]:.3f}--{dm['rouge2_ci'][1]:.3f}] "
            f"& {dm['rougeL']:.3f} [{dm['rougeL_ci'][0]:.3f}--{dm['rougeL_ci'][1]:.3f}] \\\\"
        )
    ls += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(ls)


def build_cls_paper_numbers(metrics, mc_tests, per_domain):
    ml   = minority_label()
    ours = metrics.get("qwen_lora_masked",{})
    maj  = metrics.get("majority",{})
    ls   = [
        "="*65, "firstpass — CLASSIFICATION NUMBERS",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "="*65, "",
        f"Our model accuracy:     {ours.get('accuracy',0)*100:.1f}%",
        f"Our model F1-macro:     {ours.get('f1_macro',0)*100:.1f}%",
        f"Our model F1-{ml}:  {ours.get('f1_per_class',{}).get(ml,0)*100:.1f}%",
        f"Majority baseline:      {maj.get('accuracy',0)*100:.1f}%",
        f"Improvement over maj:   {(ours.get('accuracy',0)-maj.get('accuracy',0))*100:+.1f} pp",
        "",
    ]
    for key, name in CLS_MODEL_ORDER:
        if key not in metrics: continue
        m  = metrics[key]
        mc = mc_tests.get(key,{})
        p  = mc.get("p_value","N/A")
        ls.append(f"  {name:<45} acc={m['accuracy']*100:.1f}%  p={p}")
    ls += ["","Per-domain (our model):"]
    for domain in DOMAINS:
        if domain not in per_domain: continue
        dm = per_domain[domain]
        ls.append(f"  {domain:<16} acc={dm['accuracy']*100:.1f}%  n={dm['n']}")
    return "\n".join(ls)


def build_gen_paper_numbers(gm, st, per_domain, ref_avg_len):
    def fr(d,m):
        v=d.get(m,0); ci=d.get(f"{m}_ci",[0,0])
        return f"{v:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]"
    ours = gm.get("qwen_sft",{})
    ls   = [
        "="*65, "firstpass — GENERATION NUMBERS (copy-paste ready)",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "="*65, "",
        f"  Human reference avg length: {ref_avg_len:.0f} words",
        "", "── OUR MODEL (Qwen2.5-7B + LoRA SFT) ───────────────────",
        f"  ROUGE-1  : {fr(ours,'rouge1')}",
        f"  ROUGE-2  : {fr(ours,'rouge2')}",
        f"  ROUGE-L  : {fr(ours,'rougeL')}",
    ]
    if ours.get("bertscore_f1") is not None:
        bci = ours.get("bertscore_f1_ci",[0,0])
        ls.append(f"  BERTScore: {ours['bertscore_f1']:.3f} [{bci[0]:.3f}–{bci[1]:.3f}]")
    ls += [
        f"  Avg len  : {ours.get('length_stats',{}).get('mean','N/A')} words",
        f"  N empty  : {ours.get('n_empty',0)}",
        "", "── ALL BASELINES ─────────────────────────────────────────",
    ]
    for key, name in GEN_MODEL_ORDER:
        if key=="qwen_sft" or key not in gm: continue
        bm = gm[key]
        ls.append(
            f"  {name[:40]:<40}: R1={bm.get('rouge1',0):.3f}  R2={bm.get('rouge2',0):.3f}  "
            f"RL={bm.get('rougeL',0):.3f}  len={bm.get('length_stats',{}).get('mean','?')}  "
            f"empty={bm.get('n_empty',0)}"
        )
    ls += ["","── PAIRED BOOTSTRAP TESTS (ROUGE-L vs ours) ─────────────"]
    for key, _ in GEN_MODEL_ORDER:
        if key=="qwen_sft" or key not in st: continue
        rLt = st[key].get("rougeL_test",{})
        p   = rLt.get("p_value")
        sig = ("***" if p and p<0.001 else "**" if p and p<0.01 else "*" if p and p<0.05 else "ns")
        ls.append(f"  {key:<25}: p={p}  {sig}  (Δ ROUGE-L={rLt.get('observed_diff_mean',0):.4f})")
    ls += ["","── PER-DOMAIN ROUGE-L (ours) ────────────────────────────"]
    for domain in DOMAINS:
        if domain not in per_domain: continue
        dm     = per_domain[domain]
        lo, hi = dm["rougeL_ci"]
        ls.append(f"  {domain:<16}: RL={dm['rougeL']:.3f} [{lo:.3f}–{hi:.3f}]  "
                  f"R1={dm['rouge1']:.3f}  R2={dm['rouge2']:.3f}  n={dm['n']}")
    ls += ["","="*65]
    return "\n".join(ls)


if not SKIP_CLASSIFICATION:
    print("\n" + "═"*65)
    print("CLASSIFICATION EVALUATION (Table 1)")
    print("═"*65)

    # [CLS 1/8] Majority baseline
    print("\n[CLS 1/8] Majority baseline...")
    ckp = load_checkpoint("cls","majority")
    if ckp:
        register_cls("majority", ckp["y_true"], ckp["y_pred"])
    else:
        register_cls(
            "majority",
            [r["target"].upper() for r in cls_test_records],
            [MAJORITY_LABEL]*len(cls_test_records),
        )

    # [CLS 2+3] Qwen base: load once, run zero-shot + 5-shot
    cls2_done = load_checkpoint("cls","qwen_zeroshot") is not None
    cls3_done = load_checkpoint("cls","qwen_fewshot")  is not None

    if cls2_done and cls3_done:
        print("\n[CLS 2/8] Qwen zero-shot... [RESUME]")
        ckp = load_checkpoint("cls","qwen_zeroshot")
        register_cls("qwen_zeroshot", ckp["y_true"], ckp["y_pred"])
        print("\n[CLS 3/8] Qwen 5-shot... [RESUME]")
        ckp = load_checkpoint("cls","qwen_fewshot")
        register_cls("qwen_fewshot", ckp["y_true"], ckp["y_pred"])
    else:
        print("\n[CLS 2/8] Loading Qwen base for zero-shot + 5-shot...")
        qm, qt = load_unsloth_model(CONFIG["base_model"])

        if not cls2_done:
            run_cls_with_resume("qwen_zeroshot", run_local_cls_inference,
                                qm, qt, cls_test_records, model_label="qwen_zeroshot")
        else:
            ckp = load_checkpoint("cls","qwen_zeroshot")
            register_cls("qwen_zeroshot", ckp["y_true"], ckp["y_pred"])

        print("\n[CLS 3/8] Qwen 5-shot...")
        if not cls3_done:
            fs  = get_few_shot_examples(k=CONFIG["few_shot_k"])
            par = load_partial_checkpoint("cls","qwen_fewshot")
            yt, yp = run_local_cls_inference(
                qm, qt, cls_test_records,
                few_shot_examples=fs, model_label="qwen_fewshot",
                partial_ckp=par,
            )
            clear_partial_checkpoint("cls","qwen_fewshot")
            register_cls("qwen_fewshot", yt, yp)
        else:
            ckp = load_checkpoint("cls","qwen_fewshot")
            register_cls("qwen_fewshot", ckp["y_true"], ckp["y_pred"])

        del qm, qt
        torch.cuda.empty_cache()
        print("  [Qwen base freed]")

    # [CLS 4/8] LLaMA
    print("\n[CLS 4/8] Llama-3-8B-Instruct zero-shot...")
    if "llama" not in SKIP_MODELS and CONFIG["hf_token"]:
        run_cls_with_resume("llama_zeroshot", run_hf_cls_inference,
                            CONFIG["llama_model_id"], cls_test_records,
                            model_label="llama_zeroshot",
                            sleep_secs=CONFIG["llama_sleep_cls"])
    else:
        print("  SKIPPED")

    # [CLS 5/8] DeepSeek
    print("\n[CLS 5/8] DeepSeek zero-shot...")
    if "deepseek" not in SKIP_MODELS and CONFIG["hf_token"]:
        run_cls_with_resume("deepseek_zeroshot", run_hf_cls_inference,
                            CONFIG["deepseek_model_id"], cls_test_records,
                            model_label="deepseek_zeroshot",
                            sleep_secs=CONFIG["deepseek_sleep_cls"])
    else:
        print("  SKIPPED")

    # [CLS 6/8] Gemini
    print("\n[CLS 6/8] Gemini-3.1-flash-lite-preview zero-shot...")
    if "gemini" not in SKIP_MODELS and CONFIG["gemini_api_key"]:
        run_cls_with_resume("gemini_zeroshot", run_gemini_cls_inference,
                            cls_test_records, CONFIG["gemini_api_key"],
                            CONFIG["gemini_model_cls"], model_label="gemini_zeroshot")
    else:
        print("  SKIPPED")

    # [CLS 7/8] LoRA no masking
    print(f"\n[CLS 7/8] Qwen + LoRA (no masking)...")
    lnm = CONFIG["lora_no_mask"]
    if lnm and Path(lnm).exists():
        ckp = load_checkpoint("cls","qwen_lora_nomask")
        if ckp:
            register_cls("qwen_lora_nomask", ckp["y_true"], ckp["y_pred"])
        else:
            m_nom, t_nom = load_unsloth_model(lnm)
            run_cls_with_resume("qwen_lora_nomask", run_local_cls_inference,
                                m_nom, t_nom, cls_test_records, model_label="qwen_lora_nomask")
            del m_nom, t_nom; torch.cuda.empty_cache()
    else:
        print(f"  SKIPPED (path not found: {lnm})")

    # [CLS 8/8] LoRA masked (final)
    print(f"\n[CLS 8/8] Qwen + LoRA (with masking)...")
    lm = CONFIG["lora_masked"]
    if Path(lm).exists():
        ckp = load_checkpoint("cls","qwen_lora_masked")
        if ckp:
            register_cls("qwen_lora_masked", ckp["y_true"], ckp["y_pred"])
        else:
            m_fin, t_fin = load_unsloth_model(lm)
            run_cls_with_resume("qwen_lora_masked", run_local_cls_inference,
                                m_fin, t_fin, cls_test_records, model_label="qwen_lora_masked")
            del m_fin, t_fin; torch.cuda.empty_cache()
    else:
        print(f"  [ERROR] Not found: {lm}")

    # McNemar tests
    print("\n── McNemar tests ──")
    mcnemar_vs_ours = {}
    ours_p = cls_predictions.get("qwen_lora_masked",{})
    if ours_p:
        for key, preds in cls_predictions.items():
            if key == "qwen_lora_masked": continue
            if len(preds["y_pred"]) != len(ours_p["y_pred"]):
                print(f"  [WARN] {key}: size mismatch"); continue
            result = mcnemar_test(preds["y_pred"], ours_p["y_pred"], ours_p["y_true"])
            mcnemar_vs_ours[key] = result
            p   = result["p_value"]
            sig = ("***" if p and p<0.001 else "**" if p and p<0.01 else "*" if p and p<0.05 else "ns")
            print(f"  {key:<25}: p={p}  {sig}")

    # Per-domain
    print("\n── Per-domain (classification) ──")
    per_domain_cls = {}
    if "qwen_lora_masked" in cls_predictions:
        pd = cls_predictions["qwen_lora_masked"]
        per_domain_cls = compute_per_domain_cls(pd["y_true"],pd["y_pred"],cls_test_records)
        for d, dm in sorted(per_domain_cls.items()):
            print(f"  {d:<16}: acc={dm['accuracy']*100:.1f}%  n={dm['n']}")

    # Save
    print("\n── Saving classification outputs ──")
    with open(OUT/"per_model_metrics.json","w") as f: json.dump(cls_metrics,f,indent=2)
    with open(OUT/"per_domain_metrics.json","w") as f: json.dump(per_domain_cls,f,indent=2)
    with open(OUT/"statistical_tests.json","w") as f:
        json.dump({"mcnemar_vs_ours":mcnemar_vs_ours},f,indent=2)
    with open(OUT/"raw_predictions.json","w") as f: json.dump(cls_predictions,f,indent=2)
    with open(OUT/"tables"/"table_main.tex","w") as f:
        f.write(build_main_table_latex(cls_metrics,mcnemar_vs_ours))
    with open(OUT/"tables"/"table_domain.tex","w") as f:
        f.write(build_cls_domain_table_latex(per_domain_cls))
    with open(OUT/"tables"/"table_ablation.tex","w") as f:
        f.write(build_ablation_table_latex(cls_metrics))
    cls_nums = build_cls_paper_numbers(cls_metrics,mcnemar_vs_ours,per_domain_cls)
    with open(OUT/"paper_numbers.txt","w") as f: f.write(cls_nums)
    print("  ✓ per_model_metrics.json  per_domain_metrics.json  statistical_tests.json")
    print("  ✓ raw_predictions.json")
    print("  ✓ tables/table_main.tex  table_domain.tex  table_ablation.tex")
    print("  ✓ paper_numbers.txt")
    print("\n" + "="*65)
    print(cls_nums)
    print("="*65)

if not SKIP_GENERATION and gen_test_records:
    print("\n" + "═"*65)
    print("GENERATION EVALUATION (Table 2)")
    print(f"BERTScore: {'DISABLED' if SKIP_BERTSCORE else 'ENABLED'}")
    print("═"*65)

    # [GEN 1/5] Qwen zero-shot
    print("\n[GEN 1/5] Qwen2.5-7B zero-shot...")
    ckp = load_checkpoint("gen","qwen_zeroshot")
    if ckp:
        register_gen("qwen_zeroshot", ckp["hypotheses"])
    else:
        qm, qt = load_unsloth_model(CONFIG["base_model"])
        run_gen_with_resume("qwen_zeroshot", run_local_gen_inference,
                            qm, qt, gen_test_records, model_label="qwen_zeroshot")
        del qm, qt; torch.cuda.empty_cache(); print("  [Qwen freed]")

    # [GEN 2/5] LLaMA
    print("\n[GEN 2/5] Llama-3-8B-Instruct zero-shot...")
    if "llama" not in SKIP_MODELS and CONFIG["hf_token"]:
        ckp = load_checkpoint("gen","llama_zeroshot")
        if ckp:
            register_gen("llama_zeroshot", ckp["hypotheses"])
        else:
            run_gen_with_resume("llama_zeroshot", run_hf_gen_inference,
                                CONFIG["llama_model_id"], gen_test_records,
                                model_label="llama_zeroshot",
                                sleep_secs=CONFIG["llama_sleep_gen"])
    else:
        print("  SKIPPED")

    # [GEN 3/5] DeepSeek
    print("\n[GEN 3/5] DeepSeek zero-shot...")
    if "deepseek" not in SKIP_MODELS and CONFIG["hf_token"]:
        ckp = load_checkpoint("gen","deepseek_zeroshot")
        if ckp:
            register_gen("deepseek_zeroshot", ckp["hypotheses"])
        else:
            run_gen_with_resume("deepseek_zeroshot", run_hf_gen_inference,
                                CONFIG["deepseek_model_id"], gen_test_records,
                                model_label="deepseek_zeroshot",
                                sleep_secs=CONFIG["deepseek_sleep_gen"])
    else:
        print("  SKIPPED")

    # [GEN 4/5] Gemini
    print("\n[GEN 4/5] Gemini-3.1-flash-lite-preview zero-shot...")
    if "gemini" not in SKIP_MODELS and CONFIG["gemini_api_key"]:
        ckp = load_checkpoint("gen","gemini_zeroshot")
        if ckp:
            register_gen("gemini_zeroshot", ckp["hypotheses"])
        else:
            run_gen_with_resume("gemini_zeroshot", run_gemini_gen_inference,
                                gen_test_records, CONFIG["gemini_api_key"],
                                CONFIG["gemini_model_gen"], model_label="gemini_zeroshot")
    else:
        print("  SKIPPED")

    # [GEN 5/5] Qwen SFT (ours)
    print(f"\n[GEN 5/5] Qwen + LoRA SFT (ours)...")
    sft_path = CONFIG["sft_lora_path"]
    if Path(sft_path).exists():
        ckp = load_checkpoint("gen","qwen_sft")
        if ckp:
            register_gen("qwen_sft", ckp["hypotheses"])
        else:
            sm, st_tok = load_unsloth_model(sft_path)
            run_gen_with_resume("qwen_sft", run_local_gen_inference,
                                sm, st_tok, gen_test_records, model_label="qwen_sft")
            del sm, st_tok; torch.cuda.empty_cache(); print("  [SFT freed]")
    else:
        print(f"  [ERROR] Not found: {sft_path}")

    # Statistical tests
    print("\n── Paired bootstrap tests (vs qwen_sft) ──")
    gen_stat_tests = {}
    ours_hyps      = gen_generations.get("qwen_sft",[])
    if not ours_hyps:
        print("  [WARN] qwen_sft empty — tests skipped")
    else:
        ours_gm = gen_metrics.get("qwen_sft",{})
        for key in gen_generations:
            if key == "qwen_sft": continue
            cm = gen_metrics.get(key,{})
            r1a,r2a,rLa = cm.get("_rouge1_scores",[]),cm.get("_rouge2_scores",[]),cm.get("_rougeL_scores",[])
            r1b,r2b,rLb = ours_gm.get("_rouge1_scores",[]),ours_gm.get("_rouge2_scores",[]),ours_gm.get("_rougeL_scores",[])
            if len(rLa) != len(rLb):
                print(f"  [WARN] {key}: length mismatch"); continue
            tests = {
                "rouge1_test": paired_bootstrap_test(r1a,r1b,CONFIG["n_bootstrap"]),
                "rouge2_test": paired_bootstrap_test(r2a,r2b,CONFIG["n_bootstrap"]),
                "rougeL_test": paired_bootstrap_test(rLa,rLb,CONFIG["n_bootstrap"]),
            }
            bsa  = cm.get("_bertscore_scores")
            bsb  = ours_gm.get("_bertscore_scores")
            if bsa and bsb and len(bsa)==len(bsb):
                tests["bertscore_test"] = paired_bootstrap_test(bsa,bsb,CONFIG["n_bootstrap"])
            gen_stat_tests[key] = tests
            p  = tests["rougeL_test"]["p_value"]
            d  = tests["rougeL_test"]["observed_diff_mean"]
            sig= ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
            print(f"  {key:<25}: RL p={p:.4f} {sig}  (Δ={d:+.4f})")

    # Per-domain
    print("\n── Per-domain ROUGE (qwen_sft) ──")
    per_domain_gen = {}
    if "qwen_sft" in gen_generations:
        per_domain_gen = compute_per_domain_rouge(
            gen_generations["qwen_sft"], gen_references, gen_test_records
        )
        for d, dm in sorted(per_domain_gen.items()):
            print(f"  {d:<16}: RL={dm['rougeL']:.3f}  n={dm['n']}")

    # Save
    def clean(m): return {k:v for k,v in m.items() if not k.startswith("_")}
    print("\n── Saving generation outputs ──")
    with open(GEN_OUT/"per_model_rouge.json","w") as f:
        json.dump({k:clean(v) for k,v in gen_metrics.items()},f,indent=2)
    with open(GEN_OUT/"per_domain_rouge.json","w") as f:
        json.dump(per_domain_gen,f,indent=2)
    with open(GEN_OUT/"statistical_tests.json","w") as f:
        json.dump({"tests":gen_stat_tests,"note":"paired bootstrap, n_boot=2000"},f,indent=2)
    raw_out = [
        {"article_id":r.get("article_id",str(i)), "domain":r.get("domain","?"),
         "reference":gen_references[i],
         "generated":{k:v[i] for k,v in gen_generations.items() if i<len(v)}}
        for i, r in enumerate(gen_test_records)
    ]
    with open(GEN_OUT/"raw_generations.json","w",encoding="utf-8") as f:
        json.dump(raw_out,f,indent=2,ensure_ascii=False)

    ref_avg_len = float(np.mean(ref_word_counts))
    with open(GEN_OUT/"tables"/"table_generation.tex","w") as f:
        f.write(build_generation_table_latex(gen_metrics,gen_stat_tests,ref_avg_len))
    with open(GEN_OUT/"tables"/"table_gen_domain.tex","w") as f:
        f.write(build_gen_domain_table_latex(per_domain_gen))

    gen_nums = build_gen_paper_numbers(gen_metrics,gen_stat_tests,per_domain_gen,ref_avg_len)
    with open(GEN_OUT/"paper_numbers_generation.txt","w") as f: f.write(gen_nums)

    print("  ✓ per_model_rouge.json  per_domain_rouge.json  statistical_tests.json")
    print("  ✓ raw_generations.json")
    print("  ✓ tables/table_generation.tex  table_gen_domain.tex")
    print("  ✓ paper_numbers_generation.txt")
    print(gen_nums)
    print(f"\nGeneration outputs saved to: {GEN_OUT}")

print("\n" + "="*65)
print("DONE.")
print("="*65)
