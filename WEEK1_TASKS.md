# Week 1 (23–29 Nov 2026): মানুষের হাতে যা করতে হবে

Claude যা করে দিয়েছে: dataset download, repo structure, leakage scan (run করা), cleanlab + LLM + hybrid code, Kaggle notebook, kill-test sheet, guideline v0, pre-registration draft, data card। নিচের কাজগুলো Claude করতে পারে না।

## Researcher A (Data & Annotation lead), ~14 ঘণ্টা
- [ ] Kaggle থেকে official SentNoB নামিয়ে md5 মেলাও: `md5sum Test.csv` = `83f3cf2e8a2eb0a85aa374b0756baf77` হলে mirror ঠিক আছে (1 ঘণ্টা)
- [ ] SentNoB paper-এর annotation section পড়ে `annotation/guideline_v0.md`-এর definition মিলাও; Northcutt 2021, Rohan et al. 2025 পড়ো (4 ঘণ্টা)
- [ ] VITD-এর license খোঁজো (paper / organizer-কে email) (30 মিনিট)
- [ ] `results/leakage/pairs.csv` থেকে SentNoB train_test exact pair 50টা হাতে দেখো: সত্যিই একই comment কিনা (1 ঘণ্টা)
- [ ] Google Scholar/ACL Anthology-তে "SentNoB duplicate" খোঁজো: কেউ আগে 27% duplication report করেছে কিনা (30 মিনিট)
- [ ] **Kill test step 1:** `annotation/killtest/killtest_step1_blind_annotator_A.xlsx` পূরণ (~40 মিনিট, ২ session)
- [ ] External blind annotator খোঁজা শুরু (অন্য department-এর একজন native speaker)

## Researcher B (Modeling & Detection lead), ~14 ঘণ্টা
- [ ] Kaggle: repo-টা private Kaggle Dataset `bangla-audit` হিসেবে upload, অথবা GitHub থেকে clone
- [ ] `notebooks/01_...` চালাও: BanglaBERT baseline, seed 13/42/7; reported score-এর সাথে তুলনা (~2 GPU-hour)
- [ ] একই notebook-এ SentNoB 5-fold OOF + cleanlab (~40 মিনিট GPU)
- [ ] `notebooks/02_...` smoke test: Qwen3-8B 4-bit, 50 item; Gemma-র HF license accept
- [ ] GPU script-এ কোনো error হলে Claude-কে error message পাঠাও
- [ ] **Kill test step 1:** `..._annotator_B.xlsx` পূরণ (~40 মিনিট)

## দুজনে মিলে
- [ ] Private GitHub repo বানিয়ে সব commit; `prereg/analysis_plan.md` **step 2-এর আগে** commit (timestamp = pre-registration)
- [ ] দুজনের sheet শেষ হলে `python src/killtest.py score` → `killtest_step2_adjudication.xlsx` একসাথে পূরণ → `python src/killtest.py final`
- [ ] **রবিবার 29 Nov: kill-test decision** (error ≥ 3% → label-error headline রাখো; < 3% → leakage + ambiguity headline)
- [ ] Supervisor-কে leakage v0 আর kill-test result দেখাও
