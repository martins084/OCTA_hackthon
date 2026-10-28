"""Self-supervised pre-training (masked patch reconstruction) + fine-tuning for trip danger, on GPU.

Data: processed/grab_dev_seq.npz (Grab dev trips only; blind 20% never exported), 1 Hz, 1800 s, 6 channels.
Compares, with identical 5-fold CV: (a) Transformer from scratch, (b) same Transformer pre-trained with
masked reconstruction on all dev sequences (no labels), then fine-tuned. Saves out-of-fold predictions.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SRC = next(Path("/kaggle/input").rglob("grab_dev_seq.npz"))
OUT = Path("/kaggle/working")
DEV = "cuda"
PATCH, D, HEADS, LAYERS = 10, 96, 4, 4
PRE_EPOCHS, FT_EPOCHS, BS = 30, 12, 128
torch.backends.cudnn.benchmark = True

z = np.load(SRC)
X, y, bid = z["X"].astype(np.float32), z["y"].astype(np.int64), z["bookingID"]
present = (np.abs(X).sum(-1, keepdims=True) > 0).astype(np.float32)          # [n, T, 1]
mu = (X * present).sum((0, 1)) / present.sum((0, 1))
sd = np.sqrt((((X - mu) * present) ** 2).sum((0, 1)) / present.sum((0, 1))) + 1e-6
Xn = np.concatenate([(X - mu) / sd * present, present], -1)                   # 6 channels + presence
n, T, C = Xn.shape
NP = T // PATCH
print(f"{n} trips, {y.mean():.3f} positive, T={T}, C={C}, patches={NP}", flush=True)
Xt = torch.tensor(Xn).view(n, NP, PATCH * C)                                   # patch tokens
yt = torch.tensor(y)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(PATCH * C, D)
        self.pos = nn.Parameter(torch.randn(1, NP, D) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, D))
        layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, LAYERS)
        self.norm = nn.LayerNorm(D)

    def forward(self, x, mask=None):
        h = self.inp(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_tok.expand_as(h), h)
        return self.norm(self.tf(h + self.pos))


class Recon(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc, self.dec = enc, nn.Linear(D, PATCH * C)

    def forward(self, x, mask):
        return self.dec(self.enc(x, mask))


class Classifier(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Dropout(0.2), nn.Linear(D, 1))

    def forward(self, x):
        valid = (x[..., C - 1::C].sum(-1) > 0).float()                           # patches with data
        h = self.enc(x)
        mean = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        mx = (h - 1e4 * (1 - valid[..., None])).amax(1)
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


def pretrain(idx):
    enc = Encoder().to(DEV)
    model = Recon(enc).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.05)
    steps = PRE_EPOCHS * (len(idx) // BS + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    for ep in range(PRE_EPOCHS):
        perm = idx[torch.randperm(len(idx))]
        tot = 0
        for i in range(0, len(perm), BS):
            xb = Xt[perm[i:i + BS]].to(DEV, non_blocking=True)
            valid = xb[..., C - 1::C].sum(-1) > 0
            mask = (torch.rand(xb.shape[:2], device=DEV) < 0.4) & valid
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(xb, mask)
            loss = ((pred.float() - xb) ** 2).mean(-1)[mask].mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == PRE_EPOCHS - 1:
            print(f"  pretrain ep {ep} recon loss {tot / (len(perm) / BS):.4f}", flush=True)
    return enc.state_dict()


def finetune(tr, te, init=None, seed=0):
    torch.manual_seed(seed)
    enc = Encoder()
    if init is not None:
        enc.load_state_dict(init)
    model = Classifier(enc).to(DEV)
    head_lr, enc_lr = 1e-3, (3e-4 if init is not None else 1e-3)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": enc_lr},
                             {"params": model.head.parameters(), "lr": head_lr}], weight_decay=0.05)
    steps = FT_EPOCHS * (len(tr) // BS + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [enc_lr, head_lr], total_steps=steps)
    pos_w = torch.tensor((1 - y[tr].mean()) / y[tr].mean(), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(FT_EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            xb, yb = Xt[b].to(DEV), yt[b].float().to(DEV)
            xb = xb * (1 + 0.05 * torch.randn(xb.shape[0], 1, xb.shape[2], device=DEV))   # gain jitter
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(xb).float(), yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 512):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(model(Xt[te[i:i + 512]].to(DEV)).float()).cpu())
    return torch.cat(out).numpy()


t0 = time.time()
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y))
oof = {"scratch": np.zeros(n), "pretrained": np.zeros(n)}
for k, (tr, te) in enumerate(folds):
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)
    # Pre-train on this fold's training trips only (labels unused), so the test fold stays unseen.
    init = pretrain(tr_t)
    oof["scratch"][te] = finetune(tr_t, te_t, None, seed=k)
    oof["pretrained"][te] = finetune(tr_t, te_t, init, seed=k)
    print(f"fold {k}: scratch {roc_auc_score(y[te], oof['scratch'][te]):.4f} "
          f"pretrained {roc_auc_score(y[te], oof['pretrained'][te]):.4f}  ({time.time() - t0:.0f}s)", flush=True)

res = {k: float(roc_auc_score(y, v)) for k, v in oof.items()}
print(json.dumps(res, indent=2))
np.savez(OUT / "ssl_oof.npz", bookingID=bid, y=y, **oof)
(OUT / "ssl_metrics.json").write_text(json.dumps(res, indent=2))
