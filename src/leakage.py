"""RQ2: exact + near-duplicate leakage scan.

Usage (from repo root):
    python src/leakage.py                      # all 4 datasets
    python src/leakage.py --datasets sentnob blp23_task2
    python src/leakage.py --jaccard 0.8 --num-perm 128

Outputs in results/leakage/:
    summary.csv           one row per (dataset, relation) with counts and %
    pairs.csv             every duplicate pair found (uids, labels, jaccard, kind)
    label_conflicts.csv   duplicate pairs whose gold labels differ (direct noise evidence)
    manual_check_100.csv  100 near-dup pairs stratified by jaccard, for the
                          Week-2 manual threshold check (fill `is_dup` column)

Embedding-based near-dup (LaBSE / BanglaBERT cosine) is in
notebooks/02_embedding_dups_kaggle.ipynb because it needs a GPU.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, ROOT, dedup_key, load, normalize  # noqa: E402


def shingles(text: str, k: int = 5) -> set[str]:
    t = normalize(text, use_banglabert=False).lower()
    if len(t) < k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def relation(r1, r2) -> str | None:
    """Classify a pair. Only pairs that touch a TEST item matter for leakage."""
    s1, s2 = (r1.dataset, r1.split), (r2.dataset, r2.split)
    if r1.dataset == r2.dataset:
        splits = {r1.split, r2.split}
        if splits == {"test"}:
            return "within_test"
        if "test" in splits and splits & {"train", "dev"}:
            return "train_test" if "train" in splits else "dev_test"
        if splits == {"train"}:
            return "within_train"
        return None
    # cross-dataset: only care if one side is test
    if "test" in (r1.split, r2.split):
        return "cross_dataset_test"
    return "cross_dataset_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--min-chars", type=int, default=15,
                    help="skip near-dup for very short texts (exact still counted)")
    ap.add_argument("--out", default=str(ROOT / "results" / "leakage"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([load(n) for n in args.datasets], ignore_index=True)
    df["key"] = df["text"].map(dedup_key)
    print(f"loaded {len(df):,} rows from {args.datasets}")

    pairs = {}  # (i, j) -> (kind, jaccard)

    # ---- exact duplicates (same normalized key)
    for _, idx in df.groupby("key").indices.items():
        if len(idx) < 2:
            continue
        idx = sorted(idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs[(idx[a], idx[b])] = ("exact", 1.0)
    print(f"exact pairs: {len(pairs):,}")

    # ---- near duplicates (MinHash LSH)
    lsh = MinHashLSH(threshold=args.jaccard, num_perm=args.num_perm)
    sh, mh = {}, {}
    seen_key = {}
    for i, t in enumerate(df["text"]):
        if len(t) < args.min_chars:
            continue
        k = df.at[i, "key"]
        if k in seen_key:  # exact dups already handled; index one representative
            continue
        seen_key[k] = i
        s = shingles(t)
        m = MinHash(num_perm=args.num_perm)
        for g in s:
            m.update(g.encode("utf-8"))
        sh[i], mh[i] = s, m
        lsh.insert(str(i), m)
    key_members = df.groupby("key").indices
    n_near = 0
    for i, m in mh.items():
        for j in map(int, lsh.query(m)):
            if j <= i:
                continue
            jac = jaccard(sh[i], sh[j])
            if jac < args.jaccard:
                continue
            # expand to all exact-duplicate members of both representatives
            for a in key_members[df.at[i, "key"]]:
                for b in key_members[df.at[j, "key"]]:
                    p = (min(a, b), max(a, b))
                    if p not in pairs:
                        pairs[p] = ("near", round(jac, 3))
                        n_near += 1
    print(f"near pairs (jaccard >= {args.jaccard}): {n_near:,}")

    # ---- build pair table
    rows = []
    for (i, j), (kind, jac) in pairs.items():
        r1, r2 = df.iloc[i], df.iloc[j]
        rel = relation(r1, r2)
        if rel is None:
            continue
        rows.append({
            "relation": rel, "kind": kind, "jaccard": jac,
            "uid_a": r1.uid, "uid_b": r2.uid,
            "label_a": r1.label, "label_b": r2.label,
            "label_conflict": r1.label != r2.label if r1.dataset == r2.dataset or
                              _same_label_space(r1.dataset, r2.dataset) else None,
            "text_a": r1.text, "text_b": r2.text,
        })
    P = pd.DataFrame(rows)
    P.to_csv(out / "pairs.csv", index=False)

    # ---- summary: % of each test set involved in each relation
    summ = []
    test = df[df.split == "test"]
    for ds in args.datasets:
        tuids = set(test[test.dataset == ds].uid)
        n_test = len(tuids)
        for rel in ["train_test", "dev_test", "within_test", "cross_dataset_test"]:
            for kind in ["exact", "near", "any"]:
                sub = P[(P.relation == rel)] if len(P) else P
                if kind != "any" and len(sub):
                    sub = sub[sub.kind == kind]
                hit = set()
                if len(sub):
                    hit = (set(sub.uid_a) | set(sub.uid_b)) & tuids
                conf = 0
                if len(sub):
                    cs = sub[(sub.label_conflict == True) &  # noqa: E712
                             (sub.uid_a.isin(tuids) | sub.uid_b.isin(tuids))]
                    conf = len((set(cs.uid_a) | set(cs.uid_b)) & tuids)
                summ.append({"dataset": ds, "relation": rel, "kind": kind,
                             "n_test": n_test, "test_items_hit": len(hit),
                             "pct_test": round(100 * len(hit) / max(1, n_test), 2),
                             "test_items_label_conflict": conf})
    S = pd.DataFrame(summ)
    S.to_csv(out / "summary.csv", index=False)
    print(S[S.kind == "any"].to_string(index=False))

    if len(P):
        P[P.label_conflict == True].to_csv(out / "label_conflicts.csv", index=False)  # noqa: E712
        near = P[P.kind == "near"].copy()
        if len(near):
            near["bin"] = pd.cut(near.jaccard, [0, 0.85, 0.9, 0.95, 1.0], include_lowest=True)
            per = max(1, 100 // near["bin"].nunique())
            sample = pd.concat([g.sample(min(len(g), per), random_state=13)
                                for _, g in near.groupby("bin", observed=True)])
            sample = sample.drop(columns="bin")
            sample["is_dup (y/n)"] = ""
            sample["note"] = ""
            sample.head(100).to_csv(out / "manual_check_100.csv", index=False)


def _same_label_space(a: str, b: str) -> bool:
    sentiment = {"sentnob", "blp23_task2"}
    return a in sentiment and b in sentiment


if __name__ == "__main__":
    main()
