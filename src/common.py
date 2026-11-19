"""Shared helpers: dataset loading + Bangla text normalization.

Every loader returns a DataFrame with the same columns:
    dataset, split, uid, text, label
`uid` is stable (dataset:split:row-or-official-id) so that we can release
IDs + corrected labels only (license fix), never the raw text.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Label maps taken from each dataset's official README / paper.
SENTNOB_LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}
VITD_LABELS = {0: "Non-Violence", 1: "Passive Violence", 2: "Direct Violence"}

DATASETS = ["sentnob", "blp23_task2", "vitd", "blp25_task1a"]


# ---------------------------------------------------------------- normalize
_ZW = re.compile(r"[​‌‍﻿]")  # zero-width chars
_WS = re.compile(r"\s+")

try:  # csebuetnlp normalizer (same one BanglaBERT used). Optional.
    from normalizer import normalize as _bb_normalize  # type: ignore
except Exception:  # pragma: no cover
    _bb_normalize = None


def normalize(text: str, use_banglabert: bool = True) -> str:
    """NFC + (optional) BanglaBERT normalizer + whitespace cleanup.

    Emoji are KEPT on purpose (hate/sentiment signal).
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = unicodedata.normalize("NFC", text)
    if use_banglabert and _bb_normalize is not None:
        t = _bb_normalize(t, unicode_norm="NFKC", url_replacement="<URL>")
    t = _ZW.sub("", t)
    t = _WS.sub(" ", t).strip()
    return t


def dedup_key(text: str) -> str:
    """Aggressive key for exact-duplicate detection: normalized, lowercased,
    punctuation/emoji/digits stripped. Two texts with the same key are
    treated as exact duplicates."""
    t = normalize(text, use_banglabert=False).lower()
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] in ("L", "M") or ch == " ")
    t = _WS.sub(" ", t).strip()
    return hashlib.md5(t.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- loaders
def _read_tsv(p: Path) -> pd.DataFrame:
    # quoting=3 (QUOTE_NONE): BLP tsv files contain stray quotes.
    # keep_default_na=False: BLP-2025 uses the literal label "None".
    return pd.read_csv(p, sep="\t", quoting=3, keep_default_na=False, dtype=str)


def load_sentnob() -> pd.DataFrame:
    parts = []
    for split, fn in [("train", "sentnob_Train.csv"), ("dev", "sentnob_Val.csv"), ("test", "sentnob_Test.csv")]:
        d = pd.read_csv(RAW / "sentnob" / fn)
        d = d.rename(columns={"Data": "text", "Label": "label"})
        d["label"] = d["label"].map(SENTNOB_LABELS)
        d["uid"] = [f"sentnob:{split}:{i}" for i in range(len(d))]
        d["split"] = split
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["dataset"] = "sentnob"
    return out[["dataset", "split", "uid", "text", "label"]]


def load_blp23_task2() -> pd.DataFrame:
    files = {
        "train": "blp23_sentiment_train.tsv",
        "dev": "blp23_sentiment_dev.tsv",
        "test": "blp23_sentiment_test_with_label.tsv",
    }
    parts = []
    for split, fn in files.items():
        d = _read_tsv(RAW / "blp23_task2" / fn)
        d["uid"] = "blp23_task2:" + split + ":" + d["id"].astype(str)
        d["split"] = split
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["dataset"] = "blp23_task2"
    return out[["dataset", "split", "uid", "text", "label"]]


def load_vitd() -> pd.DataFrame:
    parts = []
    for split in ["train", "dev", "test"]:
        d = pd.read_csv(RAW / "vitd" / f"{split}.csv")
        d["label"] = d["label"].map(VITD_LABELS)
        d["uid"] = [f"vitd:{split}:{i}" for i in range(len(d))]
        d["split"] = split
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["dataset"] = "vitd"
    return out[["dataset", "split", "uid", "text", "label"]]


def load_blp25_task1a() -> pd.DataFrame:
    files = {
        "train": "blp25_hatespeech_subtask_1A_train.tsv",
        "dev": "blp25_hatespeech_subtask_1A_dev.tsv",
        "test": "blp25_hatespeech_subtask_1A_test_with_labels.tsv",
    }
    parts = []
    for split, fn in files.items():
        d = _read_tsv(RAW / "blp25_task1a" / fn)
        d["uid"] = "blp25_task1a:" + split + ":" + d["id"].astype(str)
        d["split"] = split
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["dataset"] = "blp25_task1a"
    return out[["dataset", "split", "uid", "text", "label"]]


LOADERS = {
    "sentnob": load_sentnob,
    "blp23_task2": load_blp23_task2,
    "vitd": load_vitd,
    "blp25_task1a": load_blp25_task1a,
}


def load(name: str) -> pd.DataFrame:
    d = LOADERS[name]()
    d["text"] = d["text"].fillna("").astype(str)
    d["label"] = d["label"].astype(str)
    return d


def load_all() -> pd.DataFrame:
    return pd.concat([load(n) for n in DATASETS], ignore_index=True)


# ---------------------------------------------------------------- stats
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate + Wilson 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def script_profile(text: str) -> str:
    """Rough script tag used for the code-mixed / romanized breakdown
    (cross-check fix b). bangla | romanized | mixed | other."""
    bn = sum(1 for c in text if "ঀ" <= c <= "৿")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    tot = bn + lat
    if tot == 0:
        return "other"
    if bn / tot >= 0.9:
        return "bangla"
    if lat / tot >= 0.9:
        return "romanized"
    return "mixed"
