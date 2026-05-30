import os
import re
import sys
import json
import time
import logging
import requests
import pdfplumber
import shutil
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

N_PER_DOMAIN   = 750    
ACTIVE_DOMAINS = ["physics"]  

META_API_KEY = os.getenv("SPRINGER_OA_API_KEY", "")
OA_API_KEY   = os.getenv("SPRINGER_OA_API_KEY", "")


NCOMMS_ISSN = "2041-1723"
DATE_FROM   = "2023-01-01"
DATE_TO     = "2026-12-31"

DISCOVERY_API_URL   = "https://api.springernature.com/openaccess/json"
DISCOVERY_API_KEY   = OA_API_KEY
DISCOVERY_PAGE_SIZE = 10

DOMAIN_SUBJECTS = {
    "biology": [
        "biological sciences", "cell biology", "molecular biology",
        "genetics", "genomics", "biochemistry", "developmental biology",
        "microbiology", "evolutionary biology", "structural biology",
        "biology", "life sciences", "biophysics",
    ],
    "chemistry": [
        "chemistry", "organic chemistry", "inorganic chemistry",
        "chemical synthesis", "catalysis", "electrochemistry",
        "materials chemistry", "polymer chemistry", "photocatalysis",
        "chemical engineering",
    ],
    "neuroscience": [
        "neuroscience", "neurology", "neurobiology",
        "cognitive neuroscience", "brain", "neural",
    ],
        "physics": [
        "physics", "quantum physics", "condensed matter",
        "quantum mechanics", "optics", "photonics",
        "astrophysics", "particle physics", "quantum computing",
        "materials science", "nanotechnology",
    ],
    "earth_science": [
        "earth science", "geology", "geophysics", "oceanography",
        "atmospheric science", "climate", "hydrology",
        "volcanology", "seismology", "paleontology",
    ],
}

# Domain prefix for folder naming
DOMAIN_PREFIX = {
    "biology":     "Bio",
    "chemistry":   "Chem",
    "neuroscience": "Neuro",
    "physics":      "Phy",
    "earth_science": "EarthSci",
}

OUTPUT_DIR       = Path("firstpass_data")
MIN_REVIEW_WORDS = 100
REQUEST_DELAY    = 1.5
MAX_RETRIES      = 3


_fmt      = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
_file_h   = logging.FileHandler("firstpass_download.log", encoding="utf-8")
_file_h.setFormatter(_fmt)
_stream_h = logging.StreamHandler(stream=sys.stdout)
_stream_h.setFormatter(_fmt)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

log = logging.getLogger("firstpass")
log.setLevel(logging.INFO)
log.handlers = []
log.addHandler(_file_h)
log.addHandler(_stream_h)
log.propagate = False


SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})


def _get(url, params=None, stream=False, timeout=30, headers=None):
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            r = SESSION.get(
                url, params=params, stream=stream,
                timeout=timeout, headers=headers,
            )
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limited. Sleeping {wait}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(5 * (attempt + 1))


def _flatten_abstract(raw):
    """OA API sometimes returns abstract as dict with 'p' key."""
    if isinstance(raw, dict):
        return raw.get("p", "") or raw.get("text", "") or ""
    return raw or ""


def fetch_article_metadata(domain, n):
    results   = []
    seen_dois = set()
    start     = 1
    MAX_PAGES = 2000

    log.info(f"[{domain}] Scanning NComms OA records (need {n}) ...")

    for page_num in range(MAX_PAGES):
        if len(results) >= n:
            break

        subjects = DOMAIN_SUBJECTS[domain]
        subject_query = " OR ".join(f'"{s}"' for s in subjects[:5])
        params = {
            "q":       f'issn:{NCOMMS_ISSN} ({subject_query})',
            "api_key": DISCOVERY_API_KEY,
            "s":       start,
            "p":       DISCOVERY_PAGE_SIZE,
            "sort":    "date",
            "order":   "asc",
        }

        try:
            r    = _get(DISCOVERY_API_URL, params=params,
                        headers={"Accept": "application/json"})
            data = r.json()
        except Exception as e:
            log.error(f"[{domain}] OA API error at s={start}: {e}")
            break

        records = data.get("records", [])
        if not records:
            log.info(f"[{domain}] No more records at s={start}.")
            break

        total_avail = int((data.get("result") or [{}])[0].get("total", 0))
        matched_this_page = 0

        for rec in records:
            doi = rec.get("doi", "")
            if "s41467" not in doi:
                continue
            if doi in seen_dois:
                continue

            pub_date = (
                rec.get("publicationDate", "")
                or rec.get("onlineDate", "")
                or rec.get("coverDate", "")
            )
            if pub_date:
                year = pub_date[:4]
                if not ("2023" <= year <= "2026"):
                    continue

            seen_dois.add(doi)
            results.append({
                "doi":      doi,
                "title":    rec.get("title", ""),
                "abstract": _flatten_abstract(rec.get("abstract", "")),
                "pub_date": pub_date,
                "domain":   domain,
            })
            matched_this_page += 1
            if len(results) >= n:
                break

        log.info(
            f"[{domain}] page={page_num+1:4d} s={start:6d} | "
            f"page_matched={matched_this_page} | "
            f"total_collected={len(results)} | api_total={total_avail}"
        )

        if len(records) < DISCOVERY_PAGE_SIZE:
            log.info(f"[{domain}] Reached end of OA API results.")
            break

        start += DISCOVERY_PAGE_SIZE

    log.info(f"[{domain}] Metadata done: {len(results)} articles found.")
    return results[:n]

def art_id(doi):
    return doi.split("10.1038/")[-1]

def safe_stem(doi):
    return art_id(doi).replace("/", "_").replace(":", "_")

def abs_url(href):
    return href if href.startswith("http") else "https://www.nature.com" + href


def find_peer_review_url(doi):
    url = f"https://www.nature.com/articles/{art_id(doi)}"
    try:
        r = _get(url, timeout=30)
    except Exception as e:
        log.warning(f"  Article page load failed for {doi}: {e}")
        return None, None

    soup = BeautifulSoup(r.text, "lxml")

    # Also grab paper PDF URL while we have the page
    paper_pdf_url = None
    for a in soup.find_all("a", href=True):
        label = a.get_text(strip=True).lower()
        href  = a["href"]
        if "download pdf" in label or ("pdf" in label and "article" in label):
            paper_pdf_url = abs_url(href)
            break

    # Try Springer PDF direct URL as fallback
    if not paper_pdf_url:
        cand = f"https://www.nature.com/articles/{art_id(doi)}.pdf"
        try:
            head = SESSION.head(cand, timeout=10, allow_redirects=True)
            if head.status_code == 200 and "pdf" in head.headers.get("Content-Type", "").lower():
                paper_pdf_url = cand
        except Exception:
            pass

    # Find peer review PDF
    peer_review_url = None
    for a in soup.find_all("a", href=True):
        label = a.get_text(strip=True).lower()
        href  = a["href"]
        if "peer review" in label and href.lower().endswith(".pdf"):
            peer_review_url = abs_url(href)
            break

    if not peer_review_url:
        for a in soup.find_all("a", href=True):
            h = a["href"].lower()
            if ("peer_review" in h or "peerreview" in h) and h.endswith(".pdf"):
                peer_review_url = abs_url(a["href"])
                break

    if not peer_review_url:
        for script in soup.find_all("script"):
            text = script.string or ""
            hits = re.findall(
                r'https?://[^\s"\'<>]+(?:peer.review|MOESM1)[^\s"\'<>]*\.pdf',
                text, re.IGNORECASE,
            )
            if hits:
                peer_review_url = hits[0]
                break

    if not peer_review_url:
        a_id = art_id(doi)
        slug = a_id.replace("-", "_").replace(".", "_")
        cand = (
            f"https://static-content.springer.com/esm/"
            f"art%3A10.1038%2F{a_id}/MediaObjects/{slug}_MOESM1_ESM.pdf"
        )
        try:
            head = SESSION.head(cand, timeout=10, allow_redirects=True)
            if head.status_code == 200 and "pdf" in head.headers.get("Content-Type", "").lower():
                peer_review_url = cand
        except Exception:
            pass

    if not peer_review_url:
        log.warning(f"  Could not find peer review PDF for {doi}")

    return peer_review_url, paper_pdf_url


def download_pdf(url, dest, label="PDF"):
    if dest.exists() and dest.stat().st_size > 1000:
        log.info(f"  {label} already on disk: {dest.name}")
        return True
    try:
        r  = _get(url, stream=True)
        ct = r.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and "octet" not in ct.lower():
            log.warning(f"  Expected PDF, got '{ct}' from {url}")
            return False
        dest.write_bytes(r.content)
        log.info(f"  Downloaded {label}: {dest.name} ({dest.stat().st_size} bytes)")
        return True
    except Exception as e:
        log.warning(f"  {label} download failed: {e}")
        return False

RE_REVIEWER = re.compile(r"^Reviewer\s+#?\d+", re.IGNORECASE)
RE_RESPONSE = re.compile(
    r"^(Response|Author.?s?.?\s*(Reply|Response)|Point.by.Point|Rebuttal)",
    re.IGNORECASE,
)
RE_DECISION = re.compile(
    r"^(Decision\s+Letter|Editorial\s+(Decision|Comments)|Editor.?s?\s+(Letter|Note))",
    re.IGNORECASE,
)
RE_ROUND = re.compile(r"^(Round|Revision)\s*\d+", re.IGNORECASE)


def pdf_to_text(path):
    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
        return "\n".join(pages)
    except Exception as e:
        log.error(f"pdfplumber error on {path}: {e}")
        return ""


def parse_peer_review(text):
    sections  = []
    cur_label = "preamble"
    cur_lines = []

    for line in text.split("\n"):
        s = line.strip()
        if RE_DECISION.match(s) and len(s) < 80:
            sections.append((cur_label, "\n".join(cur_lines)))
            cur_label, cur_lines = "decision", [line]
        elif RE_ROUND.match(s) and len(s) < 60:
            sections.append((cur_label, "\n".join(cur_lines)))
            cur_label, cur_lines = f"round_{s}", [line]
        elif RE_REVIEWER.match(s) and len(s) < 80:
            sections.append((cur_label, "\n".join(cur_lines)))
            num = re.search(r"\d+", s)
            cur_label = f"reviewer_{num.group(0) if num else '1'}"
            cur_lines = [line]
        elif RE_RESPONSE.match(s) and len(s) < 80:
            sections.append((cur_label, "\n".join(cur_lines)))
            cur_label, cur_lines = "author_response", [line]
        else:
            cur_lines.append(line)

    sections.append((cur_label, "\n".join(cur_lines)))

    rounds        = []
    cur_round     = {"round": 1, "reviewer_comments": {}, "author_response": ""}
    decision_text = ""
    resp_count    = 0

    for label, content in sections:
        content = content.strip()
        if not content:
            continue
        if label == "decision":
            decision_text = content
        elif label.startswith("reviewer_"):
            num = label.split("_")[-1]
            cur_round["reviewer_comments"][f"reviewer_{num}"] = content
        elif label == "author_response":
            cur_round["author_response"] = content
            resp_count += 1
            rounds.append(cur_round)
            cur_round = {
                "round": resp_count + 1,
                "reviewer_comments": {},
                "author_response": "",
            }

    if cur_round["reviewer_comments"]:
        rounds.append(cur_round)

    total_rev_words = sum(
        len(t.split())
        for rnd in rounds
        for t in rnd["reviewer_comments"].values()
    )

    return {
        "rounds":               rounds,
        "decision_letter":      decision_text,
        "n_rounds":             len(rounds),
        "total_reviewer_words": total_rev_words,
    }


def infer_outcome(pr):
    txt = pr.get("decision_letter", "").lower()
    for kw in ["accept without", "accepted as is", "pleased to accept",
               "happy to accept", "accept in its current"]:
        if kw in txt:
            return "accept"
    for kw in ["major revision", "substantial revision", "new experiments",
               "additional experiments", "major concerns", "significant revision"]:
        if kw in txt:
            return "major_revision"
    for kw in ["minor revision", "minor concerns", "minor changes",
               "small changes", "acceptable with minor"]:
        if kw in txt:
            return "minor_revision"
    return "major_revision" if pr.get("n_rounds", 1) >= 2 else "unknown"


TARGET_SECTION_KEYS = {
    "abstract":              "abstract",
    "introduction":          "introduction",
    "methods":               "methods",
    "materials and methods": "methods",
    "experimental":          "methods",
    "experimental section":  "methods",
    "results":               "results",
    "results and discussion": "results",
}


def fetch_fulltext_html(doi, fallback_abstract=""):
    url = f"https://www.nature.com/articles/{art_id(doi)}"
    try:
        r = _get(url, timeout=30)
    except Exception as e:
        log.warning(f"  HTML fetch failed for {doi}: {e}")
        return {"abstract": fallback_abstract}

    soup = BeautifulSoup(r.text, "lxml")
    out  = {}

    abs_div = (
        soup.find("div", id=re.compile(r"Abs\d+-content", re.I))
        or soup.find("section", id=re.compile(r"abstract", re.I))
        or soup.find("div", class_=re.compile(r"abstract", re.I))
    )
    if abs_div:
        out["abstract"] = abs_div.get_text(separator=" ", strip=True)
    elif fallback_abstract:
        out["abstract"] = fallback_abstract

    body = (
        soup.find("div", class_=re.compile(r"c-article-body", re.I))
        or soup.find("div", id=re.compile(r"article-body", re.I))
        or soup.find("article")
    )

    if body:
        for section in body.find_all(
            ["section", "div"],
            class_=re.compile(r"c-article-section$|article-section", re.I),
            recursive=False,
        ):
            heading_tag = section.find(["h2", "h3"])
            if not heading_tag:
                continue
            heading = heading_tag.get_text(strip=True).lower()

            matched_key = None
            for pattern, norm_key in TARGET_SECTION_KEYS.items():
                if pattern in heading:
                    matched_key = norm_key
                    break
            if matched_key is None:
                continue

            paras = []
            for elem in section.find_all(
                ["p", "div"],
                class_=re.compile(r"c-article-section__content|article-text", re.I),
            ):
                txt = elem.get_text(separator=" ", strip=True)
                if txt:
                    paras.append(txt)

            if not paras:
                for p in section.find_all("p"):
                    txt = p.get_text(separator=" ", strip=True)
                    if txt:
                        paras.append(txt)

            if paras and matched_key not in out:
                out[matched_key] = " ".join(paras)

    return out


def load_skip(domain_dir):
    p = domain_dir / "skip_cache.json"
    return set(json.loads(p.read_text())) if p.exists() else set()

def save_skip(domain_dir, skip_set):
    p = domain_dir / "skip_cache.json"
    p.write_text(json.dumps(sorted(skip_set), indent=2))

def load_counter(domain_dir):
    p = domain_dir / "counter.json"
    return json.loads(p.read_text())["count"] if p.exists() else 0

def save_counter(domain_dir, count):
    p = domain_dir / "counter.json"
    p.write_text(json.dumps({"count": count}))


def process_domain(domain, n):
    domain_dir = OUTPUT_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Clean up empty/incomplete folders and fix numbering
    prefix = DOMAIN_PREFIX[domain]
    all_folders = sorted(domain_dir.glob(f"{prefix}_*"),
                        key=lambda p: int(p.name.split("_")[-1]))
    valid_folders = []
    for folder in all_folders:
        has_json = any(folder.glob("*.json"))
        if not has_json:
            shutil.rmtree(folder)
            log.info(f"  Deleted incomplete folder: {folder.name}")
        else:
            valid_folders.append(folder)

    # Renumber remaining folders sequentially
    for i, folder in enumerate(valid_folders, start=1):
        new_name = domain_dir / f"{prefix}_{i}"
        if folder != new_name:
            folder.rename(new_name)
            log.info(f"  Renamed {folder.name} → {new_name.name}")
            # Update the JSON inside to reflect new folder name
            for json_file in new_name.glob("*.json"):
                data = json.loads(json_file.read_text())
                data["folder"] = new_name.name
                json_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    articles = fetch_article_metadata(domain, n*5)

    if not articles:
        log.error(f"[{domain}] No articles found.")
        return []

    skip_set = load_skip(domain_dir)
    prefix   = DOMAIN_PREFIX[domain]
    counter  = len(list(domain_dir.glob(f"{prefix}_*")))

    api_calls = {"html": 0, "scrape": 0}
    skipped   = {k: 0 for k in [
        "already_done", "permanent_skip", "no_pdf_url",
        "pdf_fail", "parse_fail", "too_few_words", "too_few_rounds",
    ]}
    processed = []

    log.info(f"[{domain}] {len(articles)} candidates to process (target: {n}) ...")

    for art in tqdm(articles, desc=f"[{domain}]"):
        if len(processed) >= n:
            break
        doi  = art["doi"]
        stem = safe_stem(doi)

        # Check if already processed (look for any folder with this stem)
        existing = list(OUTPUT_DIR.glob(f"*/*/{stem}.json"))
        if existing:
            skipped["already_done"] += 1
            skip_set.add(doi)
            save_skip(domain_dir, skip_set)
            log.info(f"  Already exists in {existing[0].parent.parent.name}/{existing[0].parent.name} — skip")
            continue

        if doi in skip_set:
            skipped["permanent_skip"] += 1
            continue

        # --- find PDFs ---
        api_calls["scrape"] += 1
        pr_url, paper_url = find_peer_review_url(doi)

        if not pr_url:
            skipped["no_pdf_url"] += 1
            skip_set.add(doi)
            save_skip(domain_dir, skip_set)
            continue

        # --- create per-paper folder ---
        counter += 1
        paper_folder = domain_dir / f"{prefix}_{counter}"
        paper_folder.mkdir(exist_ok=True)

        pr_pdf_path    = paper_folder / f"{stem}_peer_review.pdf"
        paper_pdf_path = paper_folder / f"{stem}_paper.pdf"

        if not download_pdf(pr_url, pr_pdf_path, label="PeerReview PDF"):
            skipped["pdf_fail"] += 1
            counter -= 1          # don't advance counter for failed paper
            paper_folder.rmdir()
            continue

        if paper_url:
            download_pdf(paper_url, paper_pdf_path, label="Paper PDF")
        else:
            log.warning(f"  No paper PDF URL found for {doi}")

        # --- parse peer review ---
        text = pdf_to_text(pr_pdf_path)
        if not text:
            skipped["parse_fail"] += 1
            skip_set.add(doi)
            save_skip(domain_dir, skip_set)
            continue

        pr = parse_peer_review(text)

        if pr["total_reviewer_words"] < MIN_REVIEW_WORDS:
            skipped["too_few_words"] += 1
            skip_set.add(doi)
            save_skip(domain_dir, skip_set)
            shutil.rmtree(paper_folder)
            counter -= 1
            log.warning(f"  Too few reviewer words ({pr['total_reviewer_words']}) — skip")
            continue

        if pr["n_rounds"] < 2:
            skipped["too_few_rounds"] += 1
            skip_set.add(doi)
            save_skip(domain_dir, skip_set)
            shutil.rmtree(paper_folder)
            counter -= 1
            log.warning(f"  Only {pr['n_rounds']} round(s) — skip")
            continue

        # --- fetch full text HTML ---
        api_calls["html"] += 1
        sections = fetch_fulltext_html(doi, fallback_abstract=art.get("abstract", ""))
        outcome  = infer_outcome(pr)

        record = {
            "doi":        doi,
            "article_id": art_id(doi),
            "folder":     paper_folder.name,
            "domain":     domain,
            "pub_date":   art.get("pub_date", ""),
            "title":      art.get("title", ""),
            "paper_content": {
                "abstract":     sections.get("abstract",     art.get("abstract", "")),
                "introduction": sections.get("introduction", ""),
                "methods":      sections.get("methods",      ""),
                "results":      sections.get("results",      ""),
            },
            "peer_review":          pr["rounds"],
            "decision_letter":      pr["decision_letter"],
            "n_rounds":             pr["n_rounds"],
            "total_reviewer_words": pr["total_reviewer_words"],
            "outcome_label":        outcome,
            "peer_review_pdf":      str(pr_pdf_path),
            "paper_pdf":            str(paper_pdf_path) if paper_url else "",
        }

        rec_path = paper_folder / f"{stem}.json"
        rec_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        processed.append(record)

        log.info(
            f"  [OK] {paper_folder.name} | {art_id(doi)} | "
            f"rounds={pr['n_rounds']} | rev_words={pr['total_reviewer_words']} | "
            f"outcome={outcome} | paper_pdf={'yes' if paper_url else 'NO'}"
        )
        log.info(f"  Target reached: {len(processed)}/{n}")

    log.info(
        f"\n[{domain}] SESSION SUMMARY\n"
        f"  Processed (incl. resumed) : {len(processed)}\n"
        f"  HTML fetch calls          : {api_calls['html']}\n"
        f"  Nature.com scrapes        : {api_calls['scrape']}\n"
        f"  Skipped                   : {json.dumps(skipped)}\n"
        f"  Permanent skip set size   : {len(skip_set)}"
    )
    return processed


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_records = {}

    for domain in ACTIVE_DOMAINS:
        log.info(f"\n{'='*55}\nDOMAIN: {domain.upper()}\n{'='*55}")
        all_records[domain] = process_domain(domain, N_PER_DOMAIN)

    manifest = []
    for domain, records in all_records.items():
        for r in records:
            manifest.append({
                "doi":                  r["doi"],
                "folder":               r.get("folder", ""),
                "domain":               domain,
                "outcome_label":        r["outcome_label"],
                "n_rounds":             r["n_rounds"],
                "total_reviewer_words": r["total_reviewer_words"],
                "has_paper_pdf":        bool(r.get("paper_pdf", "")),
                "record_path": str(
                    OUTPUT_DIR / domain / r.get("folder", "") / f"{safe_stem(r['doi'])}.json"
                ),
            })

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n" + "=" * 55)
    print("DONE")
    for domain in ACTIVE_DOMAINS:
        recs   = all_records.get(domain, [])
        labels = [r["outcome_label"] for r in recs]
        pdfs   = sum(1 for r in recs if r.get("paper_pdf"))
        print(f"\n{domain.upper()}: {len(recs)} records | paper PDFs: {pdfs}/{len(recs)}")
        for lbl in ["major_revision", "minor_revision", "accept", "unknown"]:
            print(f"  {lbl}: {labels.count(lbl)}")
    print(f"\nManifest: {manifest_path}")
    print("=" * 55)

    print("\nFolder structure:")
    for domain in ACTIVE_DOMAINS:
        for r in all_records.get(domain, []):
            folder = OUTPUT_DIR / domain / r.get("folder", "")
            files  = [f.name for f in folder.iterdir()] if folder.exists() else []
            print(f"  {folder}/ → {files}")


if __name__ == "__main__":
    main()