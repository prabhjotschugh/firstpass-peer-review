import json
import random
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path("firstpass_records")
OUT_CLS  = Path("firstpass_splits_cls")
OUT_SFT  = Path("firstpass_splits_sft")
OUT_CLS.mkdir(parents=True, exist_ok=True)
OUT_SFT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["biology", "chemistry", "earth_science", "neuroscience", "physics"]
RANDOM_SEED = 42

MIN_R1_CHARS = 600       
MIN_GEN_TARGET = 300      
MAX_GEN_TARGET = 40000    
MIN_UPDATE_TARGET = 300   
MAX_UPDATE_TARGET = 20000
MAX_INPUT_CHARS = 52000   


def build_paper_text(pdata):
    content = pdata.get("paper_content", {})
    parts = []
    for key, header in [
        ("abstract",     "ABSTRACT"),
        ("introduction", "INTRODUCTION"),
        ("methods",      "METHODS"),
        ("results",      "RESULTS"),
    ]:
        val = content.get(key, "").strip()
        if val:
            parts.append(f"{header}:\n{val}")
    return "\n\n".join(parts)


def reviewer_block(comments_dict):
    return "\n\n".join(
        f"=== {k} ===\n{v.strip()}"
        for k, v in comments_dict.items()
        if v and v.strip()
    )


def derive_label(pdata):
    """
    Returns (label, source).
    source = 'heuristic' (from n_rounds) or 'drop' (unusable).
    Note: editorial_decision field is empty in 97.7% of papers,
    so we don't rely on it.
    """
    n = pdata.get("n_rounds", 0)
    if n >= 3:
        return ("EXTENDED", "heuristic")
    if n == 2:
        return ("STANDARD", "heuristic")
    return (None, "drop")


def truncate_input(text, max_chars):
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.55)
    tail = max_chars - head - 80
    return (
        text[:head]
        + "\n\n[... content truncated for length ...]\n\n"
        + text[-tail:]
    )


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def process_paper(pdata, domain):
    """
    Returns dict with up to 3 records: 'generation', 'updating', 'outcome'.
    Returns empty dict if paper fails quality checks.
    """
    article_id = pdata.get("article_id", pdata.get("folder", "unknown"))
    rounds = pdata.get("peer_review_rounds", [])

    if not rounds:
        return {}

    # Round 1 data
    round_1 = rounds[0]
    r1_dict = round_1.get("reviewer_comments", {})
    r1_text = reviewer_block(r1_dict)
    author_resp = round_1.get("author_response", "").strip()

    if len(r1_text) < MIN_R1_CHARS:
        return {}  # low-quality review

    paper_text = build_paper_text(pdata)
    label, label_src = derive_label(pdata)

    meta = {
        "article_id": article_id,
        "domain": domain,
        "n_rounds": pdata.get("n_rounds", len(rounds)),
        "label_src": label_src,
    }

    out = {}

    # ── Task 1: review_generation ──
    if MIN_GEN_TARGET <= len(r1_text) <= MAX_GEN_TARGET:
        inp = truncate_input(
            f"Critique the following scientific paper:\n\n{paper_text}",
            MAX_INPUT_CHARS
        )
        out["generation"] = {
            "task": "review_generation",
            "input": inp,
            "target": r1_text[:MAX_GEN_TARGET],
            **meta,
        }

    # ── Task 2: reviewer_updating ──
    if len(rounds) > 1:
        round_2 = rounds[1]
        r2_dict = round_2.get("reviewer_comments", {})
        r2_text = reviewer_block(r2_dict)
        if MIN_UPDATE_TARGET <= len(r2_text) <= MAX_UPDATE_TARGET:
            inp = truncate_input(
                f"Paper:\n{paper_text}\n\n"
                f"Round 1 Reviews:\n{r1_text}\n\n"
                f"Author Response:\n{author_resp}\n\n"
                f"Based on the author's response, write the Round 2 review.",
                MAX_INPUT_CHARS
            )
            out["updating"] = {
                "task": "reviewer_updating",
                "input": inp,
                "target": r2_text[:MAX_UPDATE_TARGET],
                **meta,
            }

    # ── Task 3: outcome_prediction (THE KEY IMPROVEMENT) ──
    # Include Round 2 reviews when available — this gives the model
    # much more signal about whether the paper is heading toward
    # STANDARD vs EXTENDED resolution.
    if label is not None:
        inp_parts = [
            f"Paper:\n{paper_text}",
            f"Round 1 Reviews:\n{r1_text}",
        ]
        if author_resp:
            inp_parts.append(f"Author Response:\n{author_resp}")

        # KEY: Include Round 2 reviewer comments if they exist
        if len(rounds) > 1:
            r2_dict = rounds[1].get("reviewer_comments", {})
            r2_text = reviewer_block(r2_dict)
            if r2_text and len(r2_text) > 100:
                inp_parts.append(f"Round 2 Reviews:\n{r2_text}")

        raw_input = "\n\n".join(inp_parts)
        inp = truncate_input(raw_input, MAX_INPUT_CHARS)

        prompt_suffix = (
            "Will this manuscript require a standard (2-round) or extended "
            "(3+ round) revision cycle? Answer with exactly one word: "
            "STANDARD or EXTENDED."
        )

        out["outcome"] = {
            "task": "outcome_prediction",
            "input": f"{inp}\n\n{prompt_suffix}",
            "target": label,
            "label_src": label_src,
            **{k: v for k, v in meta.items() if k != "label_src"},
        }

    return out

def stratified_split(papers, train_r=0.8, val_r=0.1):
    """
    Stratify by (domain, label) to ensure balanced classes in all splits.
    """
    buckets = defaultdict(list)
    for p in papers:
        key = (p["domain"], p.get("label", "unknown"))
        buckets[key].append(p)

    train, val, test = [], [], []
    for key, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        n_tr = max(1, int(n * train_r))
        n_va = max(1, int(n * val_r))
        train.extend(group[:n_tr])
        val.extend(group[n_tr:n_tr + n_va])
        test.extend(group[n_tr + n_va:])

    return train, val, test

def main():
    random.seed(RANDOM_SEED)

    all_papers = []
    stats = Counter()

    for domain in DOMAINS:
        domain_path = DATA_DIR / domain
        if not domain_path.exists():
            print(f"[WARN] Missing: {domain_path}")
            continue

        paper_files = sorted(domain_path.glob("*.json"))
        print(f"[{domain}] {len(paper_files)} files")

        for fp in paper_files:
            stats["total"] += 1
            try:
                with open(fp, encoding="utf-8") as f:
                    pdata = json.load(f)
            except Exception as e:
                stats["parse_error"] += 1
                continue

            records = process_paper(pdata, domain)
            if not records:
                stats["dropped"] += 1
                continue

            label = records.get("outcome", {}).get("target", "unknown")
            all_papers.append({
                "domain": domain,
                "article_id": pdata.get("article_id", fp.stem),
                "label": label,
                "records": records,
            })
            stats["kept"] += 1
            for task_key in records:
                stats[f"task_{task_key}"] += 1

    print(f"\n=== COLLECTION SUMMARY ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Stratified split
    train_papers, val_papers, test_papers = stratified_split(all_papers)
    print(f"\nPaper-level split: {len(train_papers)} train / {len(val_papers)} val / {len(test_papers)} test")

    # Write to separate directories
    splits = {"train": train_papers, "val": val_papers, "test": test_papers}

    sft_records = {"train": [], "val": [], "test": []}
    cls_records = {"train": [], "val": [], "test": []}
    task_counts = Counter()
    label_counts = Counter()

    for split_name, papers in splits.items():
        for p in papers:
            for task_key, rec in p["records"].items():
                if task_key == "outcome":
                    cls_records[split_name].append(rec)
                    label_counts[(split_name, rec["target"])] += 1
                else:
                    sft_records[split_name].append(rec)
                task_counts[(split_name, rec["task"])] += 1

    for split_name in ["train", "val", "test"]:
        write_jsonl(cls_records[split_name], OUT_CLS / f"{split_name}.jsonl")
        write_jsonl(sft_records[split_name], OUT_SFT / f"{split_name}.jsonl")

    # Print diagnostics
    print("\n=== EXAMPLES BY (split, task) ===")
    for (sp, task), cnt in sorted(task_counts.items()):
        print(f"  {sp:6} | {task:25} | {cnt}")

    print("\n=== OUTCOME LABEL BALANCE ===")
    for (sp, lbl), cnt in sorted(label_counts.items()):
        total_in_split = sum(v for (s, l), v in label_counts.items() if s == sp)
        pct = cnt / total_in_split * 100 if total_in_split > 0 else 0
        print(f"  {sp:6} | {lbl:10} | {cnt:5} ({pct:.1f}%)")

    print(f"\nCLS data -> {OUT_CLS}")
    print(f"SFT data -> {OUT_SFT}")
    print("Done.")


if __name__ == "__main__":
    main()