"""Hybrid ranking (our method) + annotation-pool sampler.

score = mean( rank_pct(1 - CL label quality), rank_pct(LLM disagreement) ) + dup_bonus
dup_bonus = 0.25 if the item is in a label-conflicting duplicate pair (results/leakage).

Pool per dataset (cross-check fix 3, capped): top ~400 flagged (union of hybrid /
CL-only / LLM-only tops) + ~200 uniform random control, shuffled, with the source
HIDDEN from annotators (source is kept in a separate key file).

Usage:
    python src/hybrid.py --dataset sentnob --cl-tag banglabert
    python src/hybrid.py --dataset sentnob --cl-tag tfidf --no-llm   # before LLM runs exist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load, script_profile  # noqa: E402


def pct_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, method="average")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cl-tag", default="banglabert")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--n-hybrid", type=int, default=200)
    ap.add_argument("--n-cl", type=int, default=100)
    ap.add_argument("--n-llm", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--exclude", default=None,
                    help="csv with uid column to exclude (e.g. the kill-test 200 already annotated)")
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    ds = a.dataset

    cl = pd.read_csv(ROOT / "results" / "cl" / f"{ds}_{a.cl_tag}_flags.csv", keep_default_na=False)
    t = cl[["uid", "label", "cl_quality", "cl_issue", "cl_pred"]].copy()
    t["cl_score"] = 1 - t.cl_quality
    if not a.no_llm:
        llm = pd.read_csv(ROOT / "results" / "llm" / f"{ds}_llm_flags.csv", keep_default_na=False)
        t = t.merge(llm[["uid", "llm_score", "llm_pred", "llm_flag"]], on="uid", how="left")
        t["hybrid"] = (pct_rank(t.cl_score) + pct_rank(t.llm_score)) / 2
    else:
        t["llm_score"] = np.nan
        t["hybrid"] = pct_rank(t.cl_score)

    conf_path = ROOT / "results" / "leakage" / "label_conflicts.csv"
    conflict_uids = set()
    if conf_path.exists():
        c = pd.read_csv(conf_path, usecols=["uid_a", "uid_b"])
        conflict_uids = set(c.uid_a) | set(c.uid_b)
    t["dup_conflict"] = t.uid.isin(conflict_uids)
    t["hybrid"] += 0.25 * t.dup_conflict
    t = t.sort_values("hybrid", ascending=False).reset_index(drop=True)

    excl = set()
    if a.exclude:
        excl = set(pd.read_csv(a.exclude).uid)
    cand = t[~t.uid.isin(excl)]

    rng = np.random.default_rng(a.seed)
    chosen: dict[str, str] = {}

    def take(frame, n, src):
        k = 0
        for u in frame.uid:
            if k >= n:
                break
            if u not in chosen:
                chosen[u] = src
                k += 1

    # random control FIRST, so it is a true uniform sample of the test set
    rand_uids = rng.choice(cand.uid.values, size=min(a.n_random, len(cand)), replace=False)
    for u in rand_uids:
        chosen[u] = "random"
    take(cand, a.n_hybrid, "hybrid")
    take(cand.sort_values("cl_score", ascending=False), a.n_cl, "cl")
    if not a.no_llm:
        take(cand.sort_values("llm_score", ascending=False), a.n_llm, "llm")

    texts = load(ds).set_index("uid").text
    pool = pd.DataFrame({"uid": list(chosen), "source": list(chosen.values())})
    pool = pool.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    pool["item_no"] = np.arange(1, len(pool) + 1)
    pool["text"] = pool.uid.map(texts)
    pool["script"] = pool.text.map(script_profile)

    out = ROOT / "annotation" / "pools"
    out.mkdir(parents=True, exist_ok=True)
    pool[["item_no", "uid", "source"]].to_csv(out / f"{ds}_pool_KEY_do_not_share.csv", index=False)
    pool[["item_no", "uid", "text", "script"]].to_csv(out / f"{ds}_pool_for_annotators.csv", index=False)
    t.to_csv(ROOT / "results" / f"{ds}_hybrid_ranking.csv", index=False)
    print(pool.source.value_counts().to_string())
    print(f"pool size {len(pool)} -> annotation/pools/{ds}_pool_for_annotators.csv")


if __name__ == "__main__":
    main()
