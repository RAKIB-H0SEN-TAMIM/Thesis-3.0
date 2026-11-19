# Data card (Week 1, v0, 19 Nov 2026)

| Dataset | Source used here | Commit | Train / Dev / Test | Labels | License |
|---|---|---|---|---|---|
| SentNoB | Unofficial GitHub mirror `SwapnanilHalder/SentNoB-Dataset` (official copy is on Kaggle `cryptexcode/sentnob-sentiment-analysis-in-noisy-bangla-texts`, which this environment could not reach). 3 independent mirrors had byte-identical Test.csv (md5 `83f3cf2e…`). | `a32ef29` | 12,575 / 1,567 / 1,586 | 0 Neutral, 1 Positive, 2 Negative | CC BY-NC-SA 4.0 (stated in BLP-2023 Task 2 README) |
| BLP-2023 Task 2 (sentiment) | `github.com/blp-workshop/blp_task2` | `2ece02f` | 35,266 / 3,934 / 6,707 | Positive / Negative / Neutral | CC BY-NC-SA 4.0 |
| VITD (BLP-2023 Task 1) | `github.com/blp-workshop/blp_task1` | `0f92f63` | 2,700 / 1,330 / 2,016 | 0 Non-Violence, 1 Passive, 2 Direct | **No license file in repo**: check the paper / ask organizers |
| BLP-2025 Task 1A (hate type) | `github.com/AridHasan/blp25_task1` | `3f40548` | 35,522 / 2,512 / 10,200 | None, Abusive, Political Hate, Profane, Religious Hate, Sexism | CC BY-NC-SA 4.0 |
| BD-SHS (optional) | not downloaded (repo `naurosromim/BD-SHS` not public at that path) | | | | |

Notes:
- BLP-2025 uses the literal label `None`; read it with `keep_default_na=False` (src/common.py does).
- BLP-2023 Task 2 is MUBASE + SentNoB, so SentNoB ↔ BLP-2023 overlap is by construction. The SentNoB test set appears inside BLP-2023 **dev** (and partly train), which matters for anyone who trains on BLP-2023 and evaluates on SentNoB.
- Script profile of test sets (Bangla / mixed / romanized): SentNoB 1586/0/0, BLP-2023 T2 5605/1094/8, VITD 2016/0/0, BLP-2025 10200/0/0. So code-mixed analysis (fix b) needs BLP-2023 T2 + annotator `code_mixed`/`dialect` tags, not script alone.

## Leakage scan v0 (src/leakage.py, exact + MinHash Jaccard ≥ 0.8, texts ≥ 15 chars for near-dup)

Test items that have a duplicate in the **same dataset's train split**:

| Dataset | Test n | Exact | Near | Any | % of test | Of these, label-conflicting |
|---|---|---|---|---|---|---|
| SentNoB | 1,586 | 430 | 6 | 435 | **27.4%** | 46 |
| BLP-2023 T2 | 6,707 | 256 | 21 | 272 | 4.1% | 60 |
| VITD | 2,016 | 4 | 7 | 11 | 0.5% | 0 |
| BLP-2025 T1A | 10,200 | 1 | 14 | 15 | 0.15% | 3 |

Also: SentNoB train has 3,187 rows that are part of an exact-duplicate group (25%).

**Caveats before anyone quotes these numbers:** (1) confirm with the official Kaggle SentNoB files; (2) the "exact" key strips punctuation, emoji and digits, so hand-check ~50 SentNoB pairs to confirm they really are the same comment; (3) near-dup threshold is not yet validated (Week 2, `results/leakage/manual_check_100.csv`); (4) search whether any SentNoB follow-up paper already reported this duplication.

Full outputs: `results/leakage/summary.csv`, `pairs.csv`, `label_conflicts.csv`, `manual_check_100.csv`.
