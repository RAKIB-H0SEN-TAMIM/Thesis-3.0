"""RQ3 detector 1: confident learning (cleanlab) on out-of-sample probabilities.

Two backbones:
  --model tfidf      char 2-5gram TF-IDF + LogisticRegression. CPU, ~1 min.
                     Use it as the cheap baseline and for quick sanity checks.
  --probs FILE.npy   probabilities produced elsewhere (e.g. BanglaBERT 5-fold in
                     notebooks/01_banglabert_cv_kaggle.ipynb). Row order must match
                     the pooled frame written to results/cl/<ds>_pool.csv.

Important: folds are grouped by the exact-duplicate key, so a text and its
duplicate never sit in train and held-out at the same time (otherwise the
model "remembers" the label and CL misses label conflicts).

Usage:
    python src/cl_flag.py --dataset sentnob --model tfidf
    python src/cl_flag.py --dataset sentnob --probs results/cl/sentnob_banglabert_probs.npy --tag banglabert
Output: results/cl/<ds>_<tag>_flags.csv (all TEST items, ranked by label quality)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, dedup_key, load, normalize  # noqa: E402


def pooled(ds: str) -> pd.DataFrame:
    d = load(ds)
    d = d[d.split.isin(["train", "dev", "test"])].reset_index(drop=True)
    d["key"] = d.text.map(dedup_key)
    return d


def tfidf_oof(d: pd.DataFrame, classes: list[str], k: int = 5, seed: int = 13) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    X_text = d.text.map(lambda t: normalize(t, use_banglabert=False))
    y = d.label.map({c: i for i, c in enumerate(classes)}).values
    probs = np.zeros((len(d), len(classes)))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(cv.split(X_text, y, groups=d.key)):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                              sublinear_tf=True, max_features=300_000)
        Xtr = vec.fit_transform(X_text.iloc[tr])
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        probs[te] = clf.predict_proba(vec.transform(X_text.iloc[te]))
        print(f"fold {f}: acc={np.mean(probs[te].argmax(1) == y[te]):.3f}")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="tfidf")
    ap.add_argument("--probs", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model
    out = ROOT / "results" / "cl"
    out.mkdir(parents=True, exist_ok=True)

    d = pooled(args.dataset)
    classes = sorted(d.label.unique())
    d.drop(columns=["text"]).to_csv(out / f"{args.dataset}_pool.csv", index=False)
    if args.probs:
        probs = np.load(args.probs)
        assert probs.shape == (len(d), len(classes)), "row order / class order mismatch"
    else:
        probs = tfidf_oof(d, classes)
        np.save(out / f"{args.dataset}_{tag}_probs.npy", probs)

    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    y = d.label.map({c: i for i, c in enumerate(classes)}).values
    d["cl_issue"] = find_label_issues(y, probs, filter_by="prune_by_noise_rate")
    d["cl_quality"] = get_label_quality_scores(y, probs, method="self_confidence")
    d["cl_pred"] = [classes[i] for i in probs.argmax(1)]
    d["cl_pred_conf"] = probs.max(1)
    for i, c in enumerate(classes):
        d[f"p_{c}"] = probs[:, i].round(4)

    t = d[d.split == "test"].sort_values("cl_quality").reset_index(drop=True)
    t["cl_rank"] = np.arange(1, len(t) + 1)
    t.to_csv(out / f"{args.dataset}_{tag}_flags.csv", index=False)
    acc = (d.cl_pred == d.label).mean()
    print(f"{args.dataset}/{tag}: OOF acc={acc:.3f}; test flagged by CL: "
          f"{t.cl_issue.sum()} / {len(t)} ({100*t.cl_issue.mean():.1f}%)")
    print("  NOTE: CL flags are CANDIDATES only. Error rate comes from the human-"
          "annotated random sample, never from this number.")


if __name__ == "__main__":
    main()
