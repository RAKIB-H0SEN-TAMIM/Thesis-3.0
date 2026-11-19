# Pre-registered analysis plan (lightweight)

Commit this file to GitHub **before** any human annotation result is looked at
(target: Sun 29 Nov 2026). The commit timestamp is the pre-registration.
Later changes go in a new dated section at the bottom, never by editing above.

Title (working): *How Clean Are Bangla NLP Benchmarks? Auditing Label Errors and Train–Test Leakage with LLM-Assisted Detection*

## Datasets and splits
SentNoB (3-class sentiment), BLP-2023 Task 2 (3-class sentiment), VITD / BLP-2023 Task 1 (3-class violence), BLP-2025 Task 1A (6-class hate type). Official splits as distributed (see docs/data_card.md for exact source commits). BD-SHS optional.

## Research questions
- **RQ1 Prevalence:** what % of test labels are wrong, per dataset and per class?
- **RQ2 Leakage:** how much exact / near-duplicate overlap exists (train–test, within-test, cross-dataset), and are scores inflated on overlapping items?
- **RQ3 Detector reliability:** precision@k and estimated recall of confident learning, LLM ensemble and hybrid detectors in Bangla; does LLM detection degrade on code-mixed / romanized / dialectal text?
- **RQ4 Ranking stability:** do model rankings change on corrected test labels?

## Hypotheses
- H1: estimated test error rate ≥ 3% in at least 3 of 4 datasets; highest in Neutral (sentiment) and implicit hate.
- H2: at least one benchmark has non-trivial train–test duplicate overlap, and models score higher on overlapping items.
- H3: LLM-ensemble precision in Bangla is lower than the precision reported for English by Nahum et al. (2025), and lower than CL on subjective classes; LLM agreement with humans is lower on items tagged code-mixed/dialect than on standard Bangla items.
- H4: at equal review budget, hybrid ranking has higher precision@k than the best single detector.
- H5: on corrected labels, at least one pair of top models swaps rank (Kendall τ < 1).

## Kill test (Week 1)
200 uniformly random SentNoB test items, seed 20261123, two-step blind protocol (annotation/guideline_v0.md).
Decision rule: if the adjudicated "Wrong" rate is **< 3%**, the paper's headline moves to leakage + ambiguity, and label-error prevalence becomes a secondary result. Report the Wilson 95% CI either way (n=200 gives roughly ±2.5 points, so this is a go/no-go signal, not the final estimate).

## Annotation protocol
- Per dataset: ~400 flagged (hybrid top-200 ∪ CL-only top-100 ∪ LLM-only top-100) + ~200 uniform random control, shuffled; detector source hidden from annotators.
- Every item labeled by both authors independently; ≥1 external native-speaker annotator labels a random subset of ≥150 items per dataset blind.
- Agreement: Cohen's κ (authors), Fleiss' κ with the external annotator; reported before and after adjudication.
- Verdicts: Correct / Wrong / Ambiguous / Unintelligible; Wrong only when gold is indefensible.

## Estimation
- Prevalence: from the random control sample only (never from flagged items), Wilson 95% CI; class-wise rates.
- Detector precision@k (k=50,100) on flagged items; recall estimated as the share of errors in the random sample that each detector ranks in its top-k / flags.
- RQ2: % test items with a duplicate (exact = same normalized key; near = char-5-gram Jaccard ≥ 0.8, threshold validated on 100 hand-checked pairs; embedding = LaBSE cosine, threshold set by hand check). Accuracy on overlapping vs non-overlapping items, paired bootstrap.
- RQ4: 9 models (TF-IDF+LR, BiLSTM, mBERT, XLM-R-base, BanglaBERT-base/large, MuRIL, zero-shot open LLM, few-shot API LLM), 3 seeds for fine-tuned models; macro-F1 on original vs corrected vs ambiguous-removed test; Kendall τ, rank flips, 1,000-sample paired bootstrap.

## Release
Item IDs + corrected labels + verdict + taxonomy code only (no raw text re-hosting), code, guideline. All four datasets are CC BY-NC-SA 4.0 (SentNoB via BLP-2023 Task 2 README; confirm the Kaggle original).

## Known limitations (stated up front)
Two author-annotators (mitigated by external annotator), only test subsets verified, possible LLM contamination with public test sets.
