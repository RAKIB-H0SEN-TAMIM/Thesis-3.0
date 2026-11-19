"""BanglaBERT / XLM-R fine-tuning: (1) baseline reproduce, (2) 5-fold OOF probs for cleanlab.

GPU script (Kaggle T4). Plain PyTorch loop so it does not break across
transformers versions.

    python src/bert_cv.py baseline --dataset sentnob                       # train->test score
    python src/bert_cv.py oof --dataset sentnob --tag banglabert            # writes results/cl/sentnob_banglabert_probs.npy
    python src/cl_flag.py --dataset sentnob --probs results/cl/sentnob_banglabert_probs.npy --tag banglabert
    python src/bert_cv.py oof --dataset sentnob --model xlm-roberta-base --tag xlmr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cl_flag import pooled  # noqa: E402  (same row order as cl_flag.py)
from common import ROOT, normalize  # noqa: E402


def train_predict(tr_text, tr_y, te_text, n_cls, model_id, epochs, lr, bs, max_len, seed):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=n_cls).to(dev)

    def enc(texts):
        return tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")

    idx = np.arange(len(tr_text))
    steps = epochs * int(np.ceil(len(idx) / bs))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    model.train()
    for ep in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            x = enc(tr_text[b]).to(dev)
            y = torch.tensor(tr_y[b]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(**x, labels=y).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
        print(f"  epoch {ep + 1}/{epochs} done, last loss {loss.item():.3f}")
    model.eval()
    out = []
    for s in DataLoader(list(te_text), batch_size=64):
        x = enc(np.array(s)).to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(torch.softmax(model(**x).logits.float(), -1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "oof"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="csebuetnlp/banglabert")
    ap.add_argument("--tag", default="banglabert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    d = pooled(a.dataset)
    classes = sorted(d.label.unique())
    y = d.label.map({c: i for i, c in enumerate(classes)}).values
    X = d.text.map(normalize).values  # BanglaBERT normalizer if installed
    kw = dict(n_cls=len(classes), model_id=a.model, epochs=a.epochs, lr=a.lr, bs=a.bs,
              max_len=a.max_len, seed=a.seed)

    if a.mode == "baseline":
        tr, te = (d.split == "train").values, (d.split == "test").values
        p = train_predict(X[tr], y[tr], X[te], **kw)
        pred = p.argmax(1)
        print(f"{a.dataset} {a.model} TEST: acc={accuracy_score(y[te], pred):.4f} "
              f"macroF1={f1_score(y[te], pred, average='macro'):.4f} "
              f"weightedF1={f1_score(y[te], pred, average='weighted'):.4f} "
              f"microF1={f1_score(y[te], pred, average='micro'):.4f}")
        return

    probs = np.zeros((len(d), len(classes)))
    cv = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    for f, (tr, te) in enumerate(cv.split(X, y, groups=d.key)):
        print(f"fold {f + 1}/{a.folds}: train {len(tr)}, held-out {len(te)}")
        probs[te] = train_predict(X[tr], y[tr], X[te], **kw)
        print(f"  fold acc {np.mean(probs[te].argmax(1) == y[te]):.3f}")
    out = ROOT / "results" / "cl"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{a.dataset}_{a.tag}_probs.npy", probs)
    print(f"saved results/cl/{a.dataset}_{a.tag}_probs.npy  OOF acc={np.mean(probs.argmax(1) == y):.3f}")


if __name__ == "__main__":
    main()
