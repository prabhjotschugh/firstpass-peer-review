import os
import re
import json
import time
import sys
import logging
import hashlib
import threading
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdfplumber
import requests


GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")  
GEMINI_MODEL     = "gemini-3.1-flash-lite-preview"   
DATA_DIR         = "firstpass_data"    
OUT_DIR          = "firstpass_records" 
ACTIVE_DOMAINS   = ["neuroscience", "physics", "chemistry", "biology", "earth_science"]  # set to [] to process all domains

MAX_WORKERS      = 10  
MAX_RETRIES      = 5     
BASE_RETRY_WAIT  = 10    
REQUEST_DELAY    = 0.5   
MAX_PAPER_CHARS  = 180_000  
MAX_REVIEW_CHARS = 280_000  
MIN_REVIEWER_WORDS = 150   


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("firstpass_parser.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("firstpass")

def pdf_to_text(path: str) -> str:
    """Extract plain text from a PDF using pdfplumber. Returns '' on failure."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
    except Exception as exc:
        log.warning(f"[PDF] extraction failed for {path}: {exc}")
    return "\n".join(pages)

def _raw_to_json(raw: str):
    """
    6-layer JSON recovery pipeline.
    Returns a parsed dict/list or raises ValueError if all layers fail.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from Gemini")

    text = raw.strip()

    # Layer 1: strip markdown fences (```json ... ``` or ``` ... ```)
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    # Layer 2: naive parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Layer 3: fix unescaped backslashes (LaTeX, Windows paths)
    # Replace \ not followed by a valid JSON escape char
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Layer 4: strip trailing commas before } or ]
    no_trail = re.sub(r',\s*([}\]])', r'\1', fixed)
    try:
        return json.loads(no_trail)
    except json.JSONDecodeError:
        pass

    # Layer 5: extract the outermost {...} or [...] and try that substring
    m = re.search(r'(\{.*\}|\[.*\])', no_trail, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Layer 6: json_repair (handles truncated / severely malformed JSON)
    try:
        from json_repair import repair_json
        repaired = repair_json(no_trail)
        return json.loads(repaired)
    except Exception as exc:
        raise ValueError(f"All JSON recovery layers failed. Last error: {exc}\n"
                         f"Raw snippet: {raw[:300]}")


def call_gemini(prompt: str, api_key: str, max_output_tokens: int = 65_000) -> dict:
    """
    Call Gemini with exponential-backoff retry.
    Returns a parsed dict on success, raises RuntimeError on total failure.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        wait = BASE_RETRY_WAIT * (2 ** attempt)
        try:
            resp = requests.post(url, json=payload, timeout=360)

            # Rate-limit
            if resp.status_code == 429:
                log.warning(f"[Gemini] 429 rate-limited (attempt {attempt+1}), sleeping {wait}s")
                time.sleep(wait)
                continue

            # Transient server errors
            if resp.status_code in (500, 502, 503, 504):
                log.warning(f"[Gemini] {resp.status_code} server error (attempt {attempt+1}), sleeping {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Safety block
            candidates = data.get("candidates", [])
            if not candidates:
                reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                raise RuntimeError(f"Gemini blocked prompt: {reason}")

            finish = candidates[0].get("finishReason", "")
            if finish == "SAFETY":
                raise RuntimeError("Gemini SAFETY finish reason")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini returned no content parts")

            raw_text = parts[0].get("text", "")
            result = _raw_to_json(raw_text)
            return result

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            log.warning(f"[Gemini] network error (attempt {attempt+1}): {exc}, sleeping {wait}s")
            time.sleep(wait)

        except (ValueError, RuntimeError) as exc:
            # JSON parse failure or API logic error — retry makes sense
            last_error = exc
            log.warning(f"[Gemini] parse/logic error (attempt {attempt+1}): {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

        except Exception as exc:
            last_error = exc
            log.warning(f"[Gemini] unexpected error (attempt {attempt+1}): {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts. Last: {last_error}")


PAPER_PROMPT = """\
You are a precise scientific document parser. Your ONLY job is to extract four \
specific sections from a Nature Communications paper and return them as a JSON object.

══════════════════════════════════════════════════════════
CRITICAL RULES — MUST FOLLOW ALL OF THEM
══════════════════════════════════════════════════════════

1. EXTRACT VERBATIM — copy text exactly as it appears. Never paraphrase, summarize, \
shorten, or rewrite. Every sentence and every number must be preserved exactly.

2. FOUR TARGET SECTIONS:
   • "abstract"      — the paper's abstract (usually before the introduction)
   • "introduction"  — the Introduction section
   • "results"       — the Results section (or the Results portion of \
"Results and Discussion")
   • "methods"       — any section labelled "Methods", "Materials and Methods", \
"Experimental", "Experimental Section", "Experimental Procedures", \
or "Online Methods"

3. EXCLUDE from all fields (do NOT copy these into any field):
   • The section heading itself (do not start "abstract" with "Abstract")
   • Reference lists / bibliography
   • Figure and table captions
   • Author affiliations and correspondence lines
   • Acknowledgements, Funding, Competing interests
   • Supplementary information sections
   • Page headers and footers
   • Discussion sections (unless it is purely Results content)

4. If a section is genuinely absent from the text, return an empty string "".

5. Preserve paragraph breaks as \\n\\n between paragraphs.

6. For equations / LaTeX: escape all backslashes as \\\\ (e.g. \\\\sigma, not \\sigma).

7. Do NOT truncate any section. If a section is 3000 words long, copy all 3000 words.

8. Return ONLY the JSON object below — no preamble, no explanation, no markdown fences.

══════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT (return exactly this structure):
══════════════════════════════════════════════════════════
{
  "abstract": "<complete verbatim abstract text, or empty string>",
  "introduction": "<complete verbatim introduction text, or empty string>",
  "results": "<complete verbatim results text, or empty string>",
  "methods": "<complete verbatim methods text, or empty string>"
}

══════════════════════════════════════════════════════════
PAPER TEXT TO PARSE:
══════════════════════════════════════════════════════════
"""


def parse_paper(paper_pdf_path: str, api_key: str) -> dict:
    """
    Use Gemini to extract abstract/introduction/results/methods from a paper PDF.
    Always returns a dict with all four keys (empty strings for missing sections).
    """
    EMPTY = {"abstract": "", "introduction": "", "results": "", "methods": ""}
    keys  = list(EMPTY.keys())

    raw = pdf_to_text(paper_pdf_path)
    if not raw or len(raw.split()) < 80:
        log.warning(f"[PAPER] Too little text extracted from {paper_pdf_path}")
        return EMPTY

    prompt = PAPER_PROMPT + raw[:MAX_PAPER_CHARS]

    try:
        parsed = call_gemini(prompt, api_key, max_output_tokens=32_000)
    except RuntimeError as exc:
        log.error(f"[PAPER] Gemini failed: {exc}")
        return EMPTY

    # Normalise output: ensure all keys exist, coerce to string, strip
    sections = {}
    for key in keys:
        val = parsed.get(key, "")
        if not isinstance(val, str):
            val = str(val) if val else ""
        sections[key] = val.strip()

    found = [k for k in keys if len(sections[k].split()) > 20]
    log.info(f"[PAPER] Sections extracted: {found} from {Path(paper_pdf_path).name}")
    return sections


# Pre-processing: detect CC license block and label any attached reviewer report
_RE_CC_START = re.compile(
    r"Open\s+Access\s+This\s+Peer\s+Review\s+File\s+is\s+licensed",
    re.IGNORECASE,
)
_RE_CC_END = re.compile(
    r"visit\s+https?://creativecommons\.org/licenses/by/4\.0/",
    re.IGNORECASE,
)
_RE_REVIEWER_CONTENT = re.compile(
    r"(major\s+concern|minor\s+comment|the\s+authors?\s+(should|need|must)|"
    r"the\s+manuscript|the\s+paper|figure\s+\d|I\s+(have|would|suggest)|"
    r"this\s+study|this\s+work|the\s+results|the\s+method)",
    re.IGNORECASE,
)


def _preprocess_review_text(raw: str) -> tuple[str, bool]:
    """
    Strip the Creative Commons license block.
    If substantial reviewer text follows it (an attached reviewer report),
    label it clearly so Gemini can attribute it correctly.
    Returns (processed_text, had_attachment_flag).
    """
    cc_match = _RE_CC_START.search(raw)
    if not cc_match:
        return raw, False

    cc_start = cc_match.start()
    cc_end_m = _RE_CC_END.search(raw, cc_start)
    cc_end   = cc_end_m.end() if cc_end_m else min(cc_start + 600, len(raw))

    after_cc = raw[cc_end:].strip()
    had_attachment = (
        len(after_cc.split()) > 120
        and bool(_RE_REVIEWER_CONTENT.search(after_cc))
    )

    main_text = raw[:cc_start].rstrip()

    if had_attachment:
        attachment_block = (
            "\n\n" + "=" * 70 + "\n"
            "ATTACHED REVIEWER REPORT — IMPORTANT\n"
            "The text below is a complete reviewer report submitted as a PDF\n"
            "attachment. It appears after the Creative Commons license block.\n"
            "Assign it to whichever reviewer wrote 'see pdf' or had no inline text.\n"
            + "=" * 70 + "\n\n"
            + after_cc
        )
        return main_text + attachment_block, True

    return main_text, False


REVIEW_PROMPT = """\
You are an expert at parsing Nature Communications transparent peer-review PDF files \
into structured JSON for machine-learning training data.
Accuracy and completeness are CRITICAL — this data will be used to train scientific AI models.

══════════════════════════════════════════════════════════════════
BACKGROUND — HOW NATURE COMMUNICATIONS PEER REVIEW WORKS
══════════════════════════════════════════════════════════════════
• A paper goes through 1–3 revision rounds.
• Each round: referees submit reports → authors submit a point-by-point rebuttal → \
editors decide whether to ask for another round or accept.
• The PDF you will parse contains all rounds concatenated, plus an editorial \
decision letter.

══════════════════════════════════════════════════════════════════
WHAT YOU MUST EXTRACT
══════════════════════════════════════════════════════════════════

OUTPUT SCHEMA:
{
  "rounds": [
    {
      "round": <integer starting at 1>,
      "reviewer_comments": {
        "reviewer_1": "<complete verbatim text of Reviewer 1's report>",
        "reviewer_2": "<complete verbatim text of Reviewer 2's report>",
        ... (as many reviewers as exist)
      },
      "author_response": "<complete verbatim text of authors' point-by-point \
rebuttal to this round, or empty string if absent>"
    },
    ... (one entry per revision round)
  ],
  "decision_letter": "<editorial decision letter text, or empty string>"
}

══════════════════════════════════════════════════════════════════
PARSING RULES — MUST FOLLOW ALL
══════════════════════════════════════════════════════════════════

RULE 1 — VERBATIM PRESERVATION
Copy every word of every reviewer comment and every author response exactly as written. \
Do NOT summarize, shorten, paraphrase, or omit ANY content. \
If a reviewer writes 2000 words, you must copy all 2000 words.

RULE 2 — ROUND BOUNDARIES
A new round of REVIEWER COMMENTS starts when you see headers like:
  "Reviewer #1 (Remarks to the Author):", "Reviewer 1:", "Referee 1:", etc.
An AUTHOR RESPONSE section starts when you see:
  "Response to Reviewers", "Authors' Response", "Point-by-Point Response",
  "Rebuttal Letter", or similar.
The response to Round N goes into rounds[N-1].author_response.
Round N+1 reviewer comments come AFTER that response.

RULE 3 — DO NOT CONFUSE RE-QUOTED HEADERS WITH NEW ROUNDS
Author rebuttal letters often QUOTE reviewer comments by repeating the reviewer \
header, e.g. "Reviewer #1:" followed by the original comment, then "Response:" \
followed by the author's reply. These quoted reviewer headers are NOT new rounds — \
they are inside the author_response field. Keep them there.

RULE 4 — REVIEWER_COMMENTS MUST BE PURE REVIEWER TEXT
Strip out any "Response:", "Author Response:", "Authors' Reply:", or "Our response:" \
paragraphs from the reviewer_comments fields. Those belong only in author_response.

RULE 5 — ATTACHED REVIEWER REPORTS
If a reviewer's inline text says only "see pdf" or "[Editorial Note: See attachment]", \
their full report appears later in the document (often after a marker like \
"ATTACHED REVIEWER REPORT"). Find that text and use it as that reviewer's comments. \
Do NOT return "[Review submitted as PDF attachment]" if the actual text exists anywhere \
in the document.
If the attached text truly cannot be found anywhere, use exactly:
  "[Review submitted as PDF attachment - text not found in document]"

RULE 6 — WHAT TO EXCLUDE from all fields:
  • Creative Commons license boilerplate
  • Manuscript text that leaked into the PDF after the rebuttal (numbered lines \
of abstract, introduction, methods, etc.)
  • PDF page headers and footers
  • Springer Nature submission system boilerplate

RULE 7 — ESCAPE SEQUENCES
For mathematical notation / LaTeX inside string values, escape all backslashes as \\\\ \
(write \\\\alpha, not \\alpha, so the JSON is valid).

RULE 8 — OUTPUT FORMAT
Return ONLY the JSON object. No preamble, no explanation, no markdown fences.
The JSON must be fully valid and parseable.

══════════════════════════════════════════════════════════════════
PEER REVIEW TEXT TO PARSE:
══════════════════════════════════════════════════════════════════
"""

ATTACHMENT_REPAIR_PROMPT = """\
A peer-review PDF was parsed, but one reviewer's report could not be found inline — \
their entry was marked as "[Review submitted as PDF attachment - text not found in document]".

The text below is extracted from the end of the peer-review PDF and likely contains \
the full reviewer report that was submitted as an attachment.

Extract the COMPLETE reviewer report verbatim — every point, every comment, every question.
Return ONLY valid JSON (no markdown, no explanation):

{"reviewer_report": "<complete verbatim reviewer report text>"}

TEXT FROM END OF PDF:
"""

ATTACHMENT_PLACEHOLDER = "[Review submitted as PDF attachment - text not found in document]"


def _normalize_reviewer_key(raw_key: str) -> str:
    """Convert any reviewer key variant to 'reviewer_N'."""
    key = raw_key.lower().strip().replace(" ", "_").replace("#", "").replace(".", "")
    if not key.startswith("reviewer_"):
        digits = re.findall(r'\d+', key)
        key = f"reviewer_{digits[0]}" if digits else f"reviewer_{key}"
    return key


def _normalize_rounds(parsed) -> list:
    # Gemini sometimes returns a bare list instead of {"rounds": [...]}
    if isinstance(parsed, list):
        # Gemini wrapped the dict in a list — unwrap it
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
            rounds_raw = parsed.get("rounds", [])
        else:
            rounds_raw = parsed
    elif isinstance(parsed, dict):
        rounds_raw = parsed.get("rounds", [])
    else:
        return []

    clean_rounds = []
    for rnd in rounds_raw:
        if not isinstance(rnd, dict):
            continue
        rc_raw = rnd.get("reviewer_comments", {})
        if not isinstance(rc_raw, dict):
            continue

        reviewer_comments = {}
        for k, v in rc_raw.items():
            norm_key = _normalize_reviewer_key(str(k))
            text = (str(v) if v else "").strip()
            if text and len(text.split()) >= 3:
                reviewer_comments[norm_key] = text

        if not reviewer_comments:
            continue

        ar = rnd.get("author_response", "") or ""
        clean_rounds.append({
            "round": len(clean_rounds) + 1,
            "reviewer_comments": reviewer_comments,
            "author_response": str(ar).strip(),
        })

    return clean_rounds


_RE_CC_STRIP  = re.compile(r"Open\s+Access\s+This\s+Peer\s+Review.*", re.IGNORECASE | re.DOTALL)
_RE_REBUTTAL  = re.compile(
    r"\n\s*(Response|Author['\u2019s]*\s*(Response|Reply|Rebuttal)|Our\s+response)\s*:[^\n]*(\n(?!\n*Reviewer\s).*)*",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Strip CC license remnants and any leaked manuscript lines."""
    text = _RE_CC_STRIP.sub("", text).strip()
    # Strip leaked numbered manuscript lines (e.g. "  42  Introduction\n  43  The ...")
    text = re.sub(r'\n\s*\d{1,4}\s{1,6}[A-Z].*', '', text)
    return text.strip()


def _try_repair_attachment(raw_pdf_text: str, api_key: str) -> str | None:
    """
    Extract an attached reviewer report from the tail of the PDF text
    using a focused Gemini call.
    """
    cc = _RE_CC_START.search(raw_pdf_text)
    if cc:
        end_m = _RE_CC_END.search(raw_pdf_text, cc.start())
        tail_start = end_m.end() if end_m else cc.start() + 600
    else:
        tail_start = int(len(raw_pdf_text) * 0.65)

    tail = raw_pdf_text[tail_start:].strip()
    if len(tail.split()) < 60:
        return None

    try:
        result = call_gemini(
            ATTACHMENT_REPAIR_PROMPT + tail[:80_000],
            api_key,
            max_output_tokens=16_000,
        )
        report = (result.get("reviewer_report", "") or "").strip()
        if len(report.split()) > 60:
            return report
    except Exception as exc:
        log.warning(f"[REVIEW] Attachment repair call failed: {exc}")
    return None


def parse_review(review_pdf_path: str, api_key: str) -> dict:
    """
    Parse a Nature Communications peer-review PDF into structured rounds.
    Returns a dict with keys: rounds, decision_letter, n_rounds,
    total_reviewer_words, review_chars, had_attachment, error.
    """
    FAIL = {
        "rounds": [], "decision_letter": "", "n_rounds": 0,
        "total_reviewer_words": 0, "review_chars": 0,
        "had_attachment": False, "error": "",
    }

    raw = pdf_to_text(review_pdf_path)
    if not raw or len(raw.split()) < 80:
        FAIL["error"] = "PDF extraction returned too little text"
        return FAIL

    review_chars = len(raw)
    processed, had_attachment = _preprocess_review_text(raw)
    if had_attachment:
        log.info(f"[REVIEW] Attached reviewer report detected in {Path(review_pdf_path).name}")

    try:
        parsed = call_gemini(
            REVIEW_PROMPT + processed[:MAX_REVIEW_CHARS],
            api_key,
            max_output_tokens=65_000,
        )
    except RuntimeError as exc:
        FAIL["error"] = str(exc)
        return FAIL

    rounds   = _normalize_rounds(parsed)
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    decision = _clean_text(str(parsed.get("decision_letter", "") or "")) if isinstance(parsed, dict) else ""
    log.info(f"[REVIEW DEBUG] n_rounds={len(rounds)}, words_per_round={[sum(len(t.split()) for t in r['reviewer_comments'].values()) for r in rounds]}, raw_parsed_type={type(parsed).__name__}, raw_keys={list(parsed.keys()) if isinstance(parsed, dict) else 'LIST'}")

    # ── Post-processing: clean each field ────────────────────
    for rnd in rounds:
        for key in list(rnd["reviewer_comments"].keys()):
            txt = _clean_text(rnd["reviewer_comments"][key])
            # Strip any author rebuttal text that leaked into reviewer fields
            txt = _RE_REBUTTAL.sub("", txt).strip()
            if txt and len(txt.split()) >= 3:
                rnd["reviewer_comments"][key] = txt
            else:
                del rnd["reviewer_comments"][key]

        rnd["author_response"] = _clean_text(rnd["author_response"])

    # Remove rounds that lost all reviewers after cleaning
    rounds = [r for r in rounds if r["reviewer_comments"]]
    for i, r in enumerate(rounds):
        r["round"] = i + 1

    # ── Repair unresolved attachment placeholders ─────────────
    for rnd in rounds:
        for key, val in list(rnd["reviewer_comments"].items()):
            if ATTACHMENT_PLACEHOLDER in val:
                log.info(f"[REVIEW] Attempting attachment repair for {key}...")
                repaired = _try_repair_attachment(raw, api_key)
                if repaired:
                    rnd["reviewer_comments"][key] = _RE_REBUTTAL.sub("", repaired).strip()
                    log.info(f"[REVIEW] Attachment repaired for {key}")

    total_reviewer_words = sum(
        len(t.split())
        for r in rounds
        for t in r["reviewer_comments"].values()
    )

    return {
        "rounds":               rounds,
        "decision_letter":      decision,
        "n_rounds":             len(rounds),
        "total_reviewer_words": total_reviewer_words,
        "review_chars":         review_chars,
        "had_attachment":       had_attachment,
        "error":                "",
    }

_MAJOR_KW = [
    "major revision", "substantial revision", "new experiments",
    "additional experiments", "fundamental concern", "significant revision",
    "major concern", "not sufficient", "cannot be accepted",
    "re-calculate", "thoroughly revise",
]
_MINOR_KW = [
    "minor revision", "minor concern", "minor change",
    "acceptable with minor", "minor modification",
    "small change", "clarification", "stylistic",
]


def infer_outcome(decision: str, n_rounds: int, rounds: list) -> str:
    if not rounds:
        return "unknown"

    # 3+ round papers are almost always initially major revision
    if n_rounds >= 3:
        return "major_revision"

    # Search decision letter + first-round reviewer text
    r1_text = " ".join(rounds[0]["reviewer_comments"].values()).lower()
    ctx = (decision + " " + r1_text).lower()

    if any(kw in ctx for kw in _MAJOR_KW):
        return "major_revision"
    if any(kw in ctx for kw in _MINOR_KW):
        return "minor_revision"

    # Default for 2-round papers
    return "minor_revision" if n_rounds == 2 else "accept"

def _build_paper_context(sections: dict) -> str:
    parts = []
    for key, header in [
        ("abstract",     "## Abstract"),
        ("introduction", "## Introduction"),
        ("results",      "## Results"),
        ("methods",      "## Methods"),
    ]:
        text = (sections.get(key) or "").strip()
        if text:
            parts.append(f"{header}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def _format_reviewer_comments(rc: dict) -> str:
    lines = []
    for key in sorted(rc.keys()):
        val = (rc[key] or "").strip()
        if val and ATTACHMENT_PLACEHOLDER not in val:
            lines.append(f"### {key.replace('_', ' ').title()}\n\n{val}")
    return "\n\n".join(lines)


_PROMPT_REVIEW_GEN = (
    "You are an expert peer reviewer for Nature Communications, one of the world's "
    "leading multidisciplinary journals.\n\n"
    "Read the following paper carefully and write a thorough, technically rigorous "
    "peer review. Identify the most important scientific concerns regarding: "
    "methodology, statistical analysis, data quality, claims vs. evidence, "
    "reproducibility, and clarity. Organise your review with clear sections.\n\n"
    "{paper_ctx}"
)

_PROMPT_REVIEWER_UPD = (
    "You are an expert peer reviewer for Nature Communications.\n\n"
    "You previously reviewed this paper and raised several concerns. The authors "
    "have now submitted a revised manuscript along with a detailed point-by-point "
    "response. Your task is to write your updated review after carefully reading:\n"
    "  1. The revised paper\n"
    "  2. Your original Round 1 review\n"
    "  3. The authors' point-by-point response\n\n"
    "In your updated review, explicitly state which concerns have been adequately "
    "addressed and which remain outstanding. Be specific.\n\n"
    "## Paper\n\n{paper_ctx}\n\n"
    "## Your Original Review (Round 1)\n\n{r1_comments}\n\n"
    "## Authors' Response to Round 1\n\n{r1_response}"
)

_PROMPT_OUTCOME_PRED = (
    "You are a senior editor at Nature Communications.\n\n"
    "Read the paper and the peer-review interaction below. Based on the scientific "
    "concerns raised by the reviewers and the authors' point-by-point response, "
    "predict the editorial outcome.\n\n"
    "Respond with EXACTLY one of these labels on the first line:\n"
    "  major_revision\n"
    "  minor_revision\n"
    "  accept\n\n"
    "Then provide a 2–3 sentence rationale explaining your prediction.\n\n"
    "## Paper\n\n{paper_ctx}\n\n"
    "## Round 1 — Reviewer Comments\n\n{r1_comments}\n\n"
    "## Round 1 — Authors' Response\n\n{r1_response}"
)


def build_training_examples(sections: dict, rounds: list,
                             outcome: str, decision: str) -> dict:
    """
    Build up to 3 training examples from one paper's full peer-review dialogue.
    Returns a dict with keys: review_generation, reviewer_updating, outcome_prediction.
    Each value is {"input": str, "target": str}.
    """
    examples = {}
    if not rounds:
        return examples

    paper_ctx = _build_paper_context(sections)
    if not paper_ctx.strip():
        # Paper content is empty — review_generation example would be useless
        log.warning("[EXAMPLES] Paper context is empty; skipping review_generation")

    r1          = rounds[0]
    r1_comments = _format_reviewer_comments(r1["reviewer_comments"])
    r1_response = (r1.get("author_response") or "").strip()

    if paper_ctx.strip() and r1_comments.strip() and len(r1_comments.split()) > 30:
        examples["review_generation"] = {
            "input":  _PROMPT_REVIEW_GEN.format(paper_ctx=paper_ctx),
            "target": r1_comments,
        }

    if len(rounds) >= 2:
        r2_comments = _format_reviewer_comments(rounds[1]["reviewer_comments"])
        if (
            r2_comments.strip()
            and r1_response
            and len(r1_response.split()) > 30
            and paper_ctx.strip()
        ):
            examples["reviewer_updating"] = {
                "input": _PROMPT_REVIEWER_UPD.format(
                    paper_ctx=paper_ctx,
                    r1_comments=r1_comments,
                    r1_response=r1_response,
                ),
                "target": r2_comments,
            }
            
    if (
        len(rounds) >= 2
        and r1_comments.strip()
        and r1_response
        and len(r1_response.split()) > 30
        and paper_ctx.strip()
    ):
        target = f"Outcome: {outcome}"
        if decision:
            target += f"\n\nRationale: {decision.strip()[:600]}"

        examples["outcome_prediction"] = {
            "input": _PROMPT_OUTCOME_PRED.format(
                paper_ctx=paper_ctx,
                r1_comments=r1_comments,
                r1_response=r1_response,
            ),
            "target": target,
        }

    return examples


def build_record(paper_pdf: str, review_pdf: str, meta: dict,
                 domain: str, api_key: str) -> dict:
    """
    Parse both PDFs and assemble the full firstpass record.
    Two sequential Gemini calls: one for paper, one for review.
    """
    doi        = meta.get("doi", "")
    article_id = doi.split("10.1038/")[-1] if "10.1038/" in doi else doi

    log.info(f"  [PAPER ] Parsing {Path(paper_pdf).name}")
    sections = parse_paper(paper_pdf, api_key)
    time.sleep(REQUEST_DELAY)   # be polite between calls

    log.info(f"  [REVIEW] Parsing {Path(review_pdf).name}")
    pr = parse_review(review_pdf, api_key)
    time.sleep(REQUEST_DELAY)

    rounds   = pr["rounds"]
    decision = pr["decision_letter"]
    outcome  = infer_outcome(decision, pr["n_rounds"], rounds)
    examples = build_training_examples(sections, rounds, outcome, decision)

    rounds_summary = [
        {
            "round":          r["round"],
            "n_reviewers":    len(r["reviewer_comments"]),
            "reviewer_words": sum(len(t.split()) for t in r["reviewer_comments"].values()),
            "response_words": len(r["author_response"].split()) if r.get("author_response") else 0,
        }
        for r in rounds
    ]
    sections_found = [k for k in ("abstract", "introduction", "results", "methods")
                      if len((sections.get(k) or "").split()) > 20]

    return {
        "doi":               doi,
        "article_id":        article_id,
        "folder":            meta.get("folder", ""),
        "domain":            domain,
        "pub_date":          meta.get("pub_date", ""),
        "title":             meta.get("title", ""),
        "paper_content":     {k: sections.get(k, "") for k in
                              ("abstract", "introduction", "results", "methods")},
        "peer_review_rounds": rounds,
        "decision_letter":   decision,
        "n_rounds":          pr["n_rounds"],
        "total_reviewer_words": pr["total_reviewer_words"],
        "outcome_label":     outcome,
        "training_examples": examples,
        "_stats": {
            "paper_chars":          sum(len(v) for v in sections.values()),
            "review_chars":         pr["review_chars"],
            "sections_found":       sections_found,
            "n_training_examples":  len(examples),
            "rounds_summary":       rounds_summary,
            "had_attachment":       pr["had_attachment"],
            "review_parse_error":   pr["error"],
        },
    }


def record_is_complete(out_file: Path) -> bool:
    """
    Return True if out_file exists and contains a fully valid, complete record.
    A record is complete when:
      • JSON parses cleanly
      • n_rounds >= 1
      • total_reviewer_words >= MIN_REVIEWER_WORDS
      • paper_content has at least 2 non-empty sections with >20 words each
      • No ATTACHMENT_PLACEHOLDER remaining in reviewer comments
    """
    if not out_file.exists():
        return False
    try:
        rec = json.loads(out_file.read_text(encoding="utf-8"))
    except Exception:
        return False  # corrupt JSON → redo

    if rec.get("n_rounds", 0) < 1:
        return False
    if rec.get("total_reviewer_words", 0) < MIN_REVIEWER_WORDS:
        return False

    pc = rec.get("paper_content", {})
    good_sections = sum(
        1 for v in pc.values()
        if isinstance(v, str) and len(v.split()) > 20
    )
    if good_sections < 2:
        return False

    for rnd in rec.get("peer_review_rounds", []):
        for v in rnd.get("reviewer_comments", {}).values():
            if ATTACHMENT_PLACEHOLDER in v:
                return False

    return True


def load_meta(folder: Path) -> dict:
    """Load the metadata JSON that the download script saved in each paper folder."""
    for jf in folder.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            # Must look like a metadata file (has doi key)
            if "doi" in data:
                data["folder"] = folder.name
                return data
        except Exception:
            pass
    return {"folder": folder.name}


# Global lock + shared state for the failed.jsonl log
_failed_lock = threading.Lock()


def process_folder(pf: Path, odir: Path, domain: str,
                   api_key: str, failed_log: Path,
                   counters: dict, counters_lock: threading.Lock) -> dict:
    """
    Process one paper folder. Called from a ThreadPoolExecutor worker.
    Returns a status dict.
    """
    fn     = pf.name
    out_f  = odir / f"{fn}.json"
    tmp_f  = odir / f"{fn}.tmp.{os.getpid()}.{threading.get_ident()}.json"

    if record_is_complete(out_f):
        with counters_lock:
            counters["skipped"] += 1
        return {"status": "skipped", "folder": fn}

    pr_pdfs    = sorted(pf.glob("*peer_review*.pdf"))
    paper_pdfs = sorted(pf.glob("*_paper.pdf"))

    if not pr_pdfs:
        log.warning(f"  [SKIP] {fn}: no peer-review PDF found")
        with counters_lock:
            counters["no_pdf"] += 1
        return {"status": "no_pdf", "folder": fn}

    if not paper_pdfs:
        log.warning(f"  [SKIP] {fn}: no paper PDF found")
        with counters_lock:
            counters["no_pdf"] += 1
        return {"status": "no_pdf", "folder": fn}

    meta = load_meta(pf)

    try:
        rec = build_record(
            paper_pdf  = str(paper_pdfs[0]),
            review_pdf = str(pr_pdfs[0]),
            meta       = meta,
            domain     = domain,
            api_key    = api_key,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        log.error(f"  [ERROR] {fn}: {exc}\n{tb}")
        with _failed_lock:
            with open(failed_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "folder": fn, "domain": domain,
                    "error": str(exc), "traceback": tb,
                }, ensure_ascii=False) + "\n")
        with counters_lock:
            counters["errors"] += 1
        return {"status": "error", "folder": fn, "error": str(exc)}

    # ── Quality gate ──────────────────────────────────────────
    if rec["total_reviewer_words"] < MIN_REVIEWER_WORDS:
        reason = (f"total_reviewer_words={rec['total_reviewer_words']} "
                  f"< {MIN_REVIEWER_WORDS}")
        log.warning(f"  [QUAL ] {fn}: {reason}")
        with counters_lock:
            counters["low_quality"] += 1
        return {"status": "low_quality", "folder": fn, "reason": reason}

    # ── Atomic write (tmp → rename) ───────────────────────────
    try:
        tmp_f.write_text(
            json.dumps(rec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_f.replace(out_f)   # atomic on POSIX; near-atomic on Windows
    except Exception as exc:
        log.error(f"  [WRITE] {fn}: {exc}")
        try:
            tmp_f.unlink(missing_ok=True)
        except Exception:
            pass
        with counters_lock:
            counters["errors"] += 1
        return {"status": "write_error", "folder": fn, "error": str(exc)}

    # ── Update counters ───────────────────────────────────────
    exs = rec["training_examples"]
    with counters_lock:
        counters["processed"] += 1
        if rec["_stats"].get("had_attachment"):
            counters["attachments"] += 1
        if "review_generation"  in exs: counters["ex_rg"]  += 1
        if "reviewer_updating"  in exs: counters["ex_ru"]  += 1
        if "outcome_prediction" in exs: counters["ex_op"]  += 1
        lbl = rec["outcome_label"]
        counters.setdefault("outcomes", {})
        counters["outcomes"][lbl] = counters["outcomes"].get(lbl, 0) + 1

    # ── Pretty log line ───────────────────────────────────────
    sf  = rec["_stats"]["sections_found"]
    rs  = rec["_stats"]["rounds_summary"]
    rw  = [r["reviewer_words"] for r in rs]
    aw  = [r["response_words"] for r in rs]
    ex_tags = " ".join([
        "RG✓" if "review_generation"  in exs else "rg✗",
        "RU✓" if "reviewer_updating"  in exs else "ru✗",
        "OP✓" if "outcome_prediction" in exs else "op✗",
    ])
    att = " 📎" if rec["_stats"]["had_attachment"] else ""
    log.info(
        f"  ✓ {fn}  rnds={rec['n_rounds']}  "
        f"rev_words={rw}  resp_words={aw}  "
        f"secs={sf}  {ex_tags}  {rec['outcome_label']}{att}"
    )

    return {"status": "ok", "folder": fn, "rec": rec}


def run():
    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        log.error("GEMINI_API_KEY is not set. Edit the CONFIG block or set the env var.")
        sys.exit(1)

    data_root  = Path(DATA_DIR)
    out_root   = Path(OUT_DIR)
    failed_log = out_root / "failed.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. GATHER ALL TASKS FROM ALL DOMAINS
    all_tasks = []
    for domain in ACTIVE_DOMAINS:
        ddir = data_root / domain
        if not ddir.exists():
            log.warning(f"Domain directory not found, skipping: {ddir}")
            continue

        odir = out_root / domain
        odir.mkdir(exist_ok=True)

        folders = sorted([f for f in ddir.iterdir() if f.is_dir()])
        for f in folders:
            all_tasks.append((f, odir, domain))

    if not all_tasks:
        log.error("No folders found to process in any domain.")
        return

    log.info(f"🚀 Starting parallel processing for {len(all_tasks)} folders across {ACTIVE_DOMAINS}...")

    # 2. SHARED COUNTERS (Thread-safe)
    grand = {
        "processed": 0, "skipped": 0, "errors": 0,
        "no_pdf": 0, "low_quality": 0, "attachments": 0,
        "ex_rg": 0, "ex_ru": 0, "ex_op": 0,
        "outcomes": {},
    }
    grand_lock = threading.Lock()

    # 3. SINGLE THREAD POOL FOR ALL DOMAINS
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                process_folder,
                pf, odir, domain, api_key,
                failed_log, grand, grand_lock,
            ): pf
            for pf, odir, domain in all_tasks
        }

        for future in as_completed(future_map):
            pf = future_map[future]
            try:
                # result is captured but logging is handled inside the worker
                future.result()
            except KeyboardInterrupt:
                log.info("\nInterrupted by user. Re-run to resume.")
                sys.exit(0)
            except Exception as exc:
                log.error(f"Future error for {pf.name}: {exc}")
                with grand_lock:
                    grand["errors"] += 1

    # ── Grand summary ─────────────────────────────────────────
    p = grand["processed"] or 1
    log.info(f"  ALL DOMAINS — FINAL SUMMARY")
    log.info(f"  Processed        : {grand['processed']}")
    log.info(f"  Skipped (done)   : {grand['skipped']}")
    log.info(f"  Errors           : {grand['errors']}")
    log.info(f"  No PDF           : {grand['no_pdf']}")
    log.info(f"  Low quality      : {grand['low_quality']}")
    log.info(f"  Attachments      : {grand['attachments']}")
    log.info(f"  ───────────────────────────────────────────")
    log.info(f"  review_generation  : {grand['ex_rg']}")
    log.info(f"  reviewer_updating  : {grand['ex_ru']}  ← key training signal")
    log.info(f"  outcome_prediction : {grand['ex_op']}")
    log.info(f"  RU coverage        : {100*grand['ex_ru']/p:.1f}%")
    log.info(f"  ───────────────────────────────────────────")
    log.info(f"  Outcome distribution: {grand['outcomes']}")
    if grand["errors"]:
        log.info(f"  Failed folders logged to: {failed_log}")
    log.info(f"{'=' * 65}")

    # ── Write manifest ────────────────────────────────────────
    manifest = []
    for domain in ACTIVE_DOMAINS:
        odir = out_root / domain
        if not odir.exists():
            continue
        for jf in sorted(odir.glob("*.json")):
            if jf.name == "failed.jsonl":
                continue
            try:
                rec = json.loads(jf.read_text(encoding="utf-8"))
                manifest.append({
                    "domain":               domain,
                    "folder":               rec.get("folder", jf.stem),
                    "doi":                  rec.get("doi", ""),
                    "outcome_label":        rec.get("outcome_label", ""),
                    "n_rounds":             rec.get("n_rounds", 0),
                    "total_reviewer_words": rec.get("total_reviewer_words", 0),
                    "sections_found":       rec.get("_stats", {}).get("sections_found", []),
                    "n_training_examples":  rec.get("_stats", {}).get("n_training_examples", 0),
                    "training_example_types": list(rec.get("training_examples", {}).keys()),
                    "record_path":          str(jf),
                })
            except Exception:
                pass

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"  Manifest written: {manifest_path}  ({len(manifest)} records)")


if __name__ == "__main__":
    run()