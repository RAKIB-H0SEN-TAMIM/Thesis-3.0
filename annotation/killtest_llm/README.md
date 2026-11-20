# Kill test: Claude (LLM) pre-annotation. NOT a human annotation

This folder holds labels that Claude (an LLM) produced on 20 Nov 2026 for the 200 kill-test items. **They are not human labels and must never be copied into the human blind sheets** (`annotation/killtest/killtest_step1_blind_annotator_{A,B}.xlsx`). The paper's error rate has to come from the two students' blind labels plus the external annotator.

How the labels were made:
1. Step 1 (blind): Claude read the 200 texts without the gold label and gave Positive/Negative/Neutral + confidence 1–3.
2. Step 2: the labels were compared with gold. For each disagreement Claude gave a verdict using guideline_v0 rules ("Wrong" only when the gold label is indefensible, otherwise "Ambiguous").

## Provisional result (LLM-only)

| | count | rate | Wilson 95% CI |
|---|---|---|---|
| Claude agrees with gold | 137 / 200 | 68.5% | |
| Claude verdict **Wrong** | 27 | **13.5%** | 9.4–18.9% |
| Wrong + Ambiguous | 63 | 31.5% | 25.5–38.2% |

- Of the 27 Wrong items, 17 have gold = Positive (e.g. questions, sad or angry comments labeled Positive) and 10 have gold = Neutral (clear insults or complaints).
- Taxonomy of Wrong: E1 clear slip 15, E9 question/request 5, E5 neutral boundary 4, E2 sarcasm 2, E3 dialect 1.
- The label mapping (0 Neutral, 1 Positive, 2 Negative) was cross-checked against the same 1,586 texts in the BLP-2023 Task 2 release: 1,533 match, and **53 SentNoB test items carry a different label in BLP-2023 Task 2**. That is a second, independent noise signal.

## What this means for the kill-test decision
Even if humans find only a quarter of what Claude found (≈3.4%), the rate is still above the 3% threshold. So the provisional read is **keep the label-error headline**. This is not the decision: the decision is made after both students fill the blind sheets (`python src/killtest.py score` → adjudicate → `final`). Use this file afterwards as a ready-made LLM–human agreement data point (RQ3).
