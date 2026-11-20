# Baseline reproduce (Week 1 gate)

| Run | Dataset | Model | Setup | Test acc | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|---|
| 20 Nov 2026 | SentNoB (official split) | csebuetnlp/banglabert | CPU, seed 13, 3 epochs, lr 2e-5, bs 32, max_len 128, NFC only (csebuetnlp normalizer did not install on Python 3.13) | 0.7535 | **0.7136** | 0.7450 |

Reported in the BanglaBERT paper for SentNoB: ~72.9 macro-F1 (check the paper's table). Our 71.4 is within ~1.5 points on a single seed without the official normalizer, so the Week-1 "close to reported" gate passes. Still to do on Kaggle: seeds 42 and 7, with the normalizer, and report mean ± std.
