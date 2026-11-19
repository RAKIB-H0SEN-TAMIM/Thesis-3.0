"""Week-1 kill test: 200 random SentNoB TEST items.

Protocol (two steps, so annotators are not anchored by the gold label):
  Step 1  BLIND: each annotator labels the 200 items in their own sheet,
          WITHOUT seeing the gold label or each other's answers.
  Step 2  ADJUDICATE: `score` builds a sheet with only the items where at least
          one annotator disagreed with gold. Now gold is shown; both annotators
          decide together: Correct / Wrong / Ambiguous / Unintelligible, plus
          the error-taxonomy code.
  Result  error rate = Wrong / 200, Wilson 95% CI. Decision rule (pre-registered):
          point estimate < 3%  ->  shift headline to leakage + ambiguity.

Usage:
    python src/killtest.py make                # creates annotation/killtest/*.xlsx
    python src/killtest.py score               # after both blind sheets are filled
    python src/killtest.py final               # after adjudication sheet is filled
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load, script_profile, wilson_ci  # noqa: E402

KT = ROOT / "annotation" / "killtest"
SEED = 20261123  # fixed and pre-registered: do not change after sampling
N = 200
LABELS = ["Positive", "Negative", "Neutral"]
BLIND_CHOICES = LABELS + ["Can't decide"]
VERDICTS = ["Correct", "Wrong", "Ambiguous", "Unintelligible"]
TAXONOMY = [
    "E1 clear slip", "E2 sarcasm/irony", "E3 dialect word", "E4 code-mixed/romanized",
    "E5 neutral boundary", "E6 mixed sentiment", "E7 target confusion",
    "E8 religious/formulaic phrase", "E9 question/request", "E10 needs outside context",
    "E11 duplicate label conflict", "E12 other",
]


def _write_xlsx(df: pd.DataFrame, path: Path, dropdowns: dict[str, list[str]], widths: dict[str, int],
                readme: list[str]):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "items"
    cols = list(df.columns)
    ws.append(cols)
    for r in df.itertuples(index=False):
        ws.append(list(r))
    head = PatternFill("solid", fgColor="DDE7F0")
    edit = PatternFill("solid", fgColor="FFF7D6")
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j)
        cell.font = Font(bold=True)
        cell.fill = head
        ws.column_dimensions[get_column_letter(j)].width = widths.get(c, 14)
        if c in dropdowns or c.startswith("note") or c.startswith("correct_label"):
            for i in range(2, len(df) + 2):
                ws.cell(row=i, column=j).fill = edit
    for i in range(2, len(df) + 2):
        ws.cell(row=i, column=cols.index("text") + 1).alignment = Alignment(wrap_text=True, vertical="top")
    for c, opts in dropdowns.items():
        j = cols.index(c) + 1
        L = get_column_letter(j)
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{L}2:{L}{len(df) + 1}")
    ws.freeze_panes = "C2"
    rd = wb.create_sheet("README")
    for line in readme:
        rd.append([line])
    rd.column_dimensions["A"].width = 120
    wb.save(path)


def make():
    KT.mkdir(parents=True, exist_ok=True)
    d = load("sentnob")
    test = d[d.split == "test"].reset_index(drop=True)
    s = test.sample(N, random_state=SEED).reset_index(drop=True)
    s["item_no"] = np.arange(1, N + 1)
    s["script"] = s.text.map(script_profile)

    # leakage context for later analysis (NOT shown to annotators)
    pairs = ROOT / "results" / "leakage" / "pairs.csv"
    if pairs.exists():
        P = pd.read_csv(pairs, usecols=["relation", "uid_a", "uid_b"])
        tt = P[P.relation == "train_test"]
        dup = set(tt.uid_a) | set(tt.uid_b)
        s["has_train_duplicate"] = s.uid.isin(dup)
    s[["item_no", "uid", "label"] + [c for c in ["has_train_duplicate"] if c in s]].rename(
        columns={"label": "gold"}).to_csv(KT / "killtest_KEY_do_not_open_until_step2.csv", index=False)

    blind = s[["item_no", "text"]].copy()
    blind["your_label"] = ""
    blind["confidence (1-3)"] = ""
    blind["code_mixed_or_romanized"] = ""
    blind["dialect"] = ""
    blind["sarcasm"] = ""
    blind["note"] = ""
    readme = [
        "SentNoB kill test - STEP 1 (blind). Label each comment's sentiment yourself.",
        "Do NOT look at the gold label, the KEY file, or your partner's sheet. Do not discuss items.",
        "your_label: Positive / Negative / Neutral, or Can't decide if truly impossible.",
        "confidence: 1 = guess, 2 = fairly sure, 3 = certain.",
        "code_mixed_or_romanized / dialect / sarcasm: put y if present, else leave blank.",
        "Follow annotation/guideline_v0.md. Work in 2-3 sessions of ~25 minutes; ~10 s per item.",
        "Save the file with the same name when done.",
    ]
    widths = {"item_no": 8, "text": 80, "your_label": 14, "note": 30}
    dd = {"your_label": BLIND_CHOICES, "confidence (1-3)": ["1", "2", "3"],
          "code_mixed_or_romanized": ["y"], "dialect": ["y"], "sarcasm": ["y"]}
    for who in ["A", "B"]:
        _write_xlsx(blind, KT / f"killtest_step1_blind_annotator_{who}.xlsx", dd, widths,
                    [f"Annotator {who}"] + readme)
    print(f"wrote 2 blind sheets + key for {N} items (seed {SEED}); "
          f"script mix: {s.script.value_counts().to_dict()}")


def _kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b)


def score():
    key = pd.read_csv(KT / "killtest_KEY_do_not_open_until_step2.csv")
    A = pd.read_excel(KT / "killtest_step1_blind_annotator_A.xlsx")
    B = pd.read_excel(KT / "killtest_step1_blind_annotator_B.xlsx")
    for n, x in [("A", A), ("B", B)]:
        miss = x.your_label.isna().sum()
        assert miss == 0, f"annotator {n} has {miss} empty labels"
    m = key.merge(A[["item_no", "text", "your_label", "confidence (1-3)"]].rename(
        columns={"your_label": "A", "confidence (1-3)": "A_conf"}), on="item_no")
    m = m.merge(B[["item_no", "your_label", "confidence (1-3)"]].rename(
        columns={"your_label": "B", "confidence (1-3)": "B_conf"}), on="item_no")
    k_ab = _kappa(m.A, m.B)
    print(f"Cohen's kappa A vs B: {k_ab:.3f}")
    print(f"A agrees with gold: {(m.A == m.gold).mean():.3f};  B agrees with gold: {(m.B == m.gold).mean():.3f}")
    print(f"both disagree with gold AND agree with each other: {((m.A == m.B) & (m.A != m.gold)).sum()}")
    adj = m[(m.A != m.gold) | (m.B != m.gold)].copy()
    adj["verdict"] = ""
    adj["correct_label"] = ""
    adj["taxonomy"] = ""
    adj["note"] = ""
    cols = ["item_no", "text", "gold", "A", "B", "A_conf", "B_conf", "verdict", "correct_label", "taxonomy", "note"]
    _write_xlsx(adj[cols], KT / "killtest_step2_adjudication.xlsx",
                {"verdict": VERDICTS, "correct_label": LABELS, "taxonomy": TAXONOMY},
                {"text": 70, "taxonomy": 26, "note": 30},
                ["STEP 2: gold is now visible. Discuss each item together and fill verdict.",
                 "Wrong = gold label is clearly incorrect (fill correct_label + taxonomy).",
                 "Ambiguous = two labels are both defensible (fill taxonomy, e.g. E5/E6/E10).",
                 "Unintelligible = cannot understand the text.",
                 "Correct = gold is fine (you made the mistake in step 1)."])
    m.to_csv(KT / "killtest_step1_merged.csv", index=False)
    pd.DataFrame([{"kappa_AB": k_ab, "n": len(m), "n_to_adjudicate": len(adj)}]).to_csv(
        KT / "killtest_step1_stats.csv", index=False)
    print(f"{len(adj)} items need adjudication -> {KT / 'killtest_step2_adjudication.xlsx'}")


def final():
    adj = pd.read_excel(KT / "killtest_step2_adjudication.xlsx")
    assert adj.verdict.notna().all(), "fill every verdict first"
    key = pd.read_csv(KT / "killtest_KEY_do_not_open_until_step2.csv")
    n = len(key)
    wrong = int((adj.verdict == "Wrong").sum())
    amb = int((adj.verdict == "Ambiguous").sum())
    p, lo, hi = wilson_ci(wrong, n)
    pa, loa, hia = wilson_ci(wrong + amb, n)
    print(f"Label error rate: {100*p:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f}%), {wrong}/{n}")
    print(f"Error + ambiguous: {100*pa:.1f}%  (95% CI {100*loa:.1f}-{100*hia:.1f}%)")
    print("Taxonomy:", adj[adj.verdict.isin(['Wrong', 'Ambiguous'])].taxonomy.value_counts().to_dict())
    if "has_train_duplicate" in key:
        e = adj[adj.verdict == "Wrong"].item_no
        dup = key.set_index("item_no").has_train_duplicate
        print(f"errors among items with a train duplicate: {dup.loc[e].sum()} / {dup.sum()}")
    decision = ("KEEP label-error headline" if p >= 0.03 else
                "SHIFT headline to leakage + ambiguity (pre-registered rule: error < 3%)")
    print("DECISION:", decision)
    pd.DataFrame([{"n": n, "wrong": wrong, "ambiguous": amb, "error_rate": p, "ci_lo": lo, "ci_hi": hi,
                   "decision": decision}]).to_csv(KT / "killtest_RESULT.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "score", "final"])
    {"make": make, "score": score, "final": final}[ap.parse_args().step]()
