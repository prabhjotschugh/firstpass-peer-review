# 🎯 FirstPass: Grounding AI Scientific Judgment in Multi-Round Editorial Outcomes

Official repository for the ICML 2026 AI 4 Science Workshop papers:
* **Research Track**: "FirstPass: Grounding AI Scientific Judgment in Multi-Round Editorial Outcomes"
* **Dataset Competition Track**: "FIRSTPASS: A Multi-Domain, Multi-Round Peer Review Dataset Grounded in Real Editorial Outcomes"

**Authors:** Prabhjot Singh, Somnath Luitel, Manmeet Singh, and Josh Durkee

---

## 📖 Overview

Peer review is dialogue, not monologue. Existing AI systems fail because they train exclusively on CS/ML venues, ignore multi-round iterations, and evaluate on stylistic mimicry rather than real editorial outcomes.

**FirstPass** is the first system grounded in **3,668 complete multi-round peer-review dialogues** from *Nature Communications* across **5 scientific domains** (biology, chemistry, neuroscience, physics, earth science). Our work is presented across two tracks at ICML 2026:
1. **Research Track**: Modeling scientific judgment as multi-round editorial outcome prediction.
2. **Dataset Competition Track**: Curating and auditing the largest multi-domain, multi-round scientific peer-review dataset.

---

## 📊 Key Findings

Based on **318 test examples** from *Nature Communications* papers with 100% verified content integrity:

* **Revision-Cycle Prediction**
  * FirstPass achieves **80.5% accuracy** and **F1-macro 78.2%** on predicting editorial outcomes (*Standard* = 2-round vs. *Extended* = 3+ rounds).
  * Outperforms Gemini-3.1-flash-lite-preview zero-shot by **10.4 pp** (McNemar $p < 0.001$).
  * Per-domain consistency: 76.9%–83.8% across all five disciplines; 6.9 pp spread confirms cross-domain generalization.

* **The Masking Finding (18.5 pp swing)**
  * Without response-only loss masking: **62.0%** accuracy (below 65.4% majority baseline).
  * With masking: **80.5%** accuracy.
  * **Insight:** For long-input/short-output classification (10,000+ input tokens, 1-word label), standard full-sequence loss training drowns the classification signal. Masking is **not an optimization—it's an architectural prerequisite.**

* **Review Generation**
  * FirstPass generates reviews averaging **1,187 words** (closer to human references at 2,155 words than any baseline).
  * ROUGE-L: **0.154** with statistically significant gains over Qwen ($\Delta = +0.018$), DeepSeek ($\Delta = +0.039$), all $p < 0.001$.

* **Cross-Domain Robustness**
  * Biology: 83.8% | Chemistry: 77.6% | Neuroscience: 81.5% | Physics: 81.8% | Earth Science: 76.9%

---

## 📁 Repository Structure

```text
.
├── codes/
│   ├── 1_download_files.py           # Springer API queries, scrape Nature Communications
│   ├── 2_clean_and_extract_data.py   # PDF parsing, Gemini extraction with 6-layer JSON recovery
│   ├── 3_audit_dataset.py            # Integrity verification, statistics, domain distribution
│   ├── 4_data_split.py               # Stratified 80/10/10 split by domain + label
│   ├── 5_fine_tune_code.py           # LoRA training with response-only loss masking
│   └── 6_final_evaluation.py         # McNemar tests, per-domain breakdown, baseline comparisons
├── raw_data/
│   ├── manifest.json
│   ├── failed.jsonl
│   └── {biology,chemistry,neuroscience,physics,earth_science}/
├── results/
│   ├── per_domain_metrics.json
│   ├── per_model_metrics.json
│   ├── raw_predictions.json
│   ├── statistical_tests.json
│   ├── checkpoints/
│   ├── generation/
│   └── tables/
└── README.md
```

---

## 📥 Download Pre-trained Models & Dataset

**Model & Adapters:** [🤗 Hugging Face Hub](https://huggingface.co/Prabhjotschugh/FirstPass-Models)

**Dataset:** [🤗 Hugging Face Hub](https://huggingface.co/datasets/Prabhjotschugh/firstpass-peer-review)

---

## ⚙️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/prabhjotschugh/firstpass-peer-review.git
   cd firstpass-peer-review
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers==4.36.0 peft unsloth pdfplumber beautifulsoup4 requests tqdm datasets trl huggingface_hub
   pip install google-generativeai springer-api-client statsmodels scikit-learn rouge-score
   ```

4. Configure API keys:
   ```bash
   export SPRINGER_OA_API_KEY="your-api-key-here"
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```

---

## 🚀 Running the Pipeline

All scripts run sequentially. Execute in your local Python environment:

* **Data Preparation:** `python codes/1_download_files.py`
* **Cleaning & Extraction:** `python codes/2_clean_and_extract_data.py`
* **Audit & Statistics:** `python codes/3_audit_dataset.py`
* **Train/Val/Test Split:** `python codes/4_data_split.py`
* **Fine-tune LoRA Adapters:** `python codes/5_fine_tune_code.py` (set `TASK_MODE=cls` or `sft`)
* **Evaluation & Baselines:** `python codes/6_final_evaluation.py`

---

## 📊 Dataset Details

| **Domain** | **Count** | **Train** | **Val** | **Test** | **Standard %** | **Extended %** |
|---|---|---|---|---|---|---|
| Biology | 741 | 593 | 74 | 74 | 63.2 | 36.8 |
| Chemistry | 744 | 595 | 75 | 74 | 62.8 | 37.2 |
| Neuroscience | 739 | 591 | 74 | 74 | 64.6 | 35.4 |
| Physics | 727 | 582 | 73 | 72 | 66.7 | 33.3 |
| Earth Science | 717 | 573 | 72 | 72 | 71.2 | 28.8 |
| **Total** | **3,668** | **2,934** | **368** | **366** | **65.4** | **34.6** |

**Quality:** 100% automated integrity verification (0 hollow files); 1.6% manual audit (60 papers) confirms 2.1% field-level error rate (formatting only—no label corruption).

---

## �📧 Contact

**Corresponding Author:** Prabhjot Singh (prabhjot.singh@utexas.edu)  
**Affiliations:** UT Austin, RediMinds Inc.

