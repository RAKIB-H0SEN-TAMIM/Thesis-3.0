# bangla-audit: How Clean Are Bangla NLP Benchmarks?

Label-error + train–test leakage audit of SentNoB, BLP-2023 Task 2, VITD and BLP-2025 Task 1A, with LLM-assisted detection. Target: BLP 2027 @ NAACL (15 Feb 2027).

## Folder map

```
data/raw/            4 datasets as downloaded (see docs/data_card.md for source + commit)
src/common.py        loaders (same columns for every dataset), normalizer, Wilson CI, script tag
src/leakage.py       RQ2: exact + MinHash near-dup scan  (CPU, ~2 min)        -> results/leakage/
src/cl_flag.py       RQ3: cleanlab confident learning (TF-IDF on CPU, or BanglaBERT probs) -> results/cl/
src/bert_cv.py       BanglaBERT/XLM-R baseline + 5-fold grouped OOF probs (GPU)
src/llm_flag.py      RQ3: 4-bit LLM ensemble, option-letter logits (GPU)      -> results/llm/
src/hybrid.py        hybrid ranking + capped annotation pool (400 flagged + 200 random, source hidden)
src/killtest.py      Week-1 kill test: make / score / final
notebooks/01_*.ipynb Kaggle: BanglaBERT baseline + OOF + cleanlab
notebooks/02_*.ipynb Kaggle: LLM flagging (Week 1 = 50-item smoke test only)
notebooks/03_*.ipynb Kaggle: LaBSE embedding near-dup (Week 2)
annotation/guideline_v0.md    guideline + Bangla error taxonomy (E1–E12)
annotation/killtest/          2 blind sheets (A, B) + hidden key
prereg/analysis_plan.md       commit this BEFORE looking at kill-test step-2 results
docs/data_card.md             sources, licenses, leakage v0 numbers + caveats
```

## Quick start (laptop, CPU)

```bash
pip install -r requirements.txt
python src/leakage.py                          # leakage tables
python src/cl_flag.py --dataset sentnob        # TF-IDF + cleanlab candidates
python src/killtest.py score                   # after both blind sheets are filled
python src/killtest.py final                   # after adjudication sheet is filled
```

## Pipeline order (Weeks 1–5)
1. `leakage.py` → hand-check `results/leakage/manual_check_100.csv`
2. `bert_cv.py oof` → `cl_flag.py --probs ...` (per dataset)
3. `llm_flag.py` per model, then `llm_flag.py --ensemble`
4. `hybrid.py --dataset X --exclude annotation/killtest/killtest_KEY_do_not_open_until_step2.csv` → annotation pool

## Before going public
Do not push `data/raw/`, `results/leakage/pairs.csv` (contains text) or any `*_KEY_*` file to a public repo. Release IDs + corrected labels only.

## Status
- Week 1 code written and run on CPU: leakage scan (all 4), TF-IDF cleanlab (SentNoB, VITD), kill-test sheets (tested end to end with fake labels, then reset).
- GPU scripts (`bert_cv.py`, `llm_flag.py`, notebooks) are written but **not yet run**: first Kaggle run may need small fixes.
