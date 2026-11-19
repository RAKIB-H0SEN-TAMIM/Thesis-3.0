"""RQ3 detector 2: LLM ensemble label-error flagging (Nahum et al., EMNLP 2025 style).

Runs on Kaggle (T4 x2) with 4-bit open-weight models. Each (model, prompt)
pair gives a probability over the label options, read from the logits of
the option letters (A/B/C...), not from generated text. Ensemble = mean.
An item is flagged when ensemble argmax != gold and confidence >= tau.

Usage on Kaggle (see notebooks/03_llm_flag_kaggle.ipynb):
    python src/llm_flag.py --dataset sentnob --model Qwen/Qwen3-8B --prompts bn en en_def
    python src/llm_flag.py --dataset sentnob --ensemble     # combine all runs
Outputs:
    results/llm/<ds>__<model>__<prompt>.csv   per-run probabilities (test split)
    results/llm/<ds>_llm_flags.csv            ensemble + flag, ranked
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load  # noqa: E402

# Label definitions copied/paraphrased from each dataset's official guideline.
# Students: check these against the original papers in Week 1 and edit.
LABEL_DEFS = {
    "sentnob": {
        "Positive": "the comment expresses a positive opinion, praise, support or happiness",
        "Negative": "the comment expresses a negative opinion, criticism, anger, insult or sadness",
        "Neutral": "the comment is factual, a question or request, or has no clear sentiment",
    },
    "blp23_task2": {
        "Positive": "the text expresses positive sentiment",
        "Negative": "the text expresses negative sentiment",
        "Neutral": "the text has no clear positive or negative sentiment",
    },
    "vitd": {
        "Direct Violence": "explicit threats or calls for killing, rape, vandalism, deportation, or forcing people to abandon religion/culture",
        "Passive Violence": "derogatory or abusive language or slang targeting people/communities, or justification of violence",
        "Non-Violence": "no violence; e.g. discussion of social rights or general conversation",
    },
    "blp25_task1a": {
        "None": "not hateful or abusive",
        "Abusive": "insulting or abusive language not tied to a protected group",
        "Political Hate": "hate targeting political parties, politicians or political groups",
        "Profane": "vulgar or obscene language",
        "Religious Hate": "hate targeting a religion or religious group",
        "Sexism": "hate or discrimination based on gender",
    },
}

TASK_NAME = {
    "sentnob": ("sentiment", "অনুভূতি (sentiment)"),
    "blp23_task2": ("sentiment", "অনুভূতি (sentiment)"),
    "vitd": ("violence-inciting content", "সহিংসতা উসকানি"),
    "blp25_task1a": ("hate speech type", "ঘৃণা বক্তব্যের ধরন"),
}

LETTERS = "ABCDEFGH"


def build_prompt(ds: str, text: str, labels: list[str], style: str) -> str:
    en_task, bn_task = TASK_NAME[ds]
    opts = "\n".join(f"{LETTERS[i]}. {l}" for i, l in enumerate(labels))
    if style == "bn":
        return (f"নিচের বাংলা মন্তব্যটির {bn_task} কোনটি? শুধু একটি অক্ষর (A, B, C...) দিয়ে উত্তর দাও।\n\n"
                f"মন্তব্য: {text}\n\nবিকল্প:\n{opts}\n\nউত্তর:")
    if style == "en":
        return (f"Classify the {en_task} of the following Bangla social-media comment. "
                f"Answer with a single letter only.\n\nComment: {text}\n\nOptions:\n{opts}\n\nAnswer:")
    if style == "en_def":
        defs = "\n".join(f"{LETTERS[i]}. {l}: {LABEL_DEFS[ds][l]}" for i, l in enumerate(labels))
        return (f"You are an expert Bangla annotator. Label the {en_task} of the comment using the "
                f"annotation guideline below. The comment may be code-mixed, romanized or dialectal. "
                f"Answer with a single letter only.\n\nGuideline:\n{defs}\n\nComment: {text}\n\nAnswer:")
    raise ValueError(style)


def run(ds: str, model_id: str, styles: list[str], batch_size: int, limit: int | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    out = ROOT / "results" / "llm"
    out.mkdir(parents=True, exist_ok=True)
    d = load(ds)
    d = d[d.split == "test"].reset_index(drop=True)
    if limit:
        d = d.head(limit)
    labels = list(LABEL_DEFS[ds].keys())

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_quant_type="nf4"),
    ).eval()
    letter_ids = [tok.encode(LETTERS[i], add_special_tokens=False)[-1] for i in range(len(labels))]

    for style in styles:
        probs = []
        for s in range(0, len(d), batch_size):
            chunk = d.text.iloc[s:s + batch_size].tolist()
            msgs = [[{"role": "user", "content": build_prompt(ds, t, labels, style)}] for t in chunk]
            kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
                       for m in msgs]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            p = torch.softmax(logits[:, letter_ids].float(), dim=-1).cpu().numpy()
            probs.append(p)
            if s % (batch_size * 20) == 0:
                print(f"{style}: {s}/{len(d)}")
        probs = np.concatenate(probs)
        r = d[["uid", "label"]].copy()
        for i, l in enumerate(labels):
            r[f"p_{l}"] = probs[:, i].round(4)
        name = f"{ds}__{model_id.replace('/', '_')}__{style}.csv"
        r.to_csv(out / name, index=False)
        acc = (np.array(labels)[probs.argmax(1)] == r.label.values).mean()
        print(f"saved {name}  acc vs gold = {acc:.3f}")


def ensemble(ds: str, tau: float):
    out = ROOT / "results" / "llm"
    labels = list(LABEL_DEFS[ds].keys())
    runs = sorted(glob.glob(str(out / f"{ds}__*.csv")))
    assert runs, "no LLM runs found"
    P = None
    for f in runs:
        r = pd.read_csv(f, keep_default_na=False)
        m = r[[f"p_{l}" for l in labels]].values
        P = m if P is None else P + m
        base = r[["uid", "label"]]
    P /= len(runs)
    e = base.copy()
    for i, l in enumerate(labels):
        e[f"llm_p_{l}"] = P[:, i].round(4)
    e["llm_pred"] = np.array(labels)[P.argmax(1)]
    e["llm_conf"] = P.max(1)
    gold_idx = e.label.map({l: i for i, l in enumerate(labels)}).values
    e["llm_p_gold"] = P[np.arange(len(P)), gold_idx]
    e["llm_flag"] = (e.llm_pred != e.label) & (e.llm_conf >= tau)
    # disagreement score used for ranking: confidence of the non-gold argmax
    e["llm_score"] = np.where(e.llm_pred != e.label, e.llm_conf, 0) + (1 - e.llm_p_gold) * 1e-3
    e = e.sort_values("llm_score", ascending=False).reset_index(drop=True)
    e["llm_rank"] = np.arange(1, len(e) + 1)
    e["n_runs"] = len(runs)
    e.to_csv(out / f"{ds}_llm_flags.csv", index=False)
    print(f"{ds}: {len(runs)} runs, flagged {e.llm_flag.sum()} / {len(e)} at tau={tau}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--prompts", nargs="+", default=["bn", "en", "en_def"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N test items")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--tau", type=float, default=0.6)
    a = ap.parse_args()
    if a.ensemble:
        ensemble(a.dataset, a.tau)
    else:
        run(a.dataset, a.model, a.prompts, a.batch_size, a.limit)
