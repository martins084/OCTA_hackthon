"""1D-CNN driving-style model on UAH dev sequences, leave-one-driver-out, on Kaggle GPU.

Reads processed/uah_dev_seq.npz (dev drivers only; produced by src/export_sequences.py).
Writes oof probabilities + metrics to /kaggle/working for comparison with the GBM.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = next(Path("/kaggle/input").rglob("uah_dev_seq.npz"))
OUT = Path("/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, SEEDS, CROP = 40, 3, 540

z = np.load(SRC)
X, y, driver, trip = z["X"], z["y"], z["driver"], z["trip"]
classes = list(z["classes"])
print(DEV, X.shape, dict(zip(*np.unique(driver, return_counts=True))))


class Net(nn.Module):
    def __init__(self, c_in, n_cls):
        super().__init__()
        def block(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.f = nn.Sequential(block(c_in, 32, 7, 2), block(32, 64, 5, 2), block(64, 64, 5, 1),
                               block(64, 96, 3, 2), block(96, 96, 3, 1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(96 * 2, n_cls))

    def forward(self, x):                       # x: [B, T, C]
        h = self.f(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


def augment(xb):
    b, t, c = xb.shape
    start = torch.randint(0, t - CROP + 1, (1,)).item()
    xb = xb[:, start:start + CROP]
    xb = xb * (1 + 0.1 * torch.randn(b, 1, c, device=xb.device))       # per-channel gain
    return xb + 0.05 * torch.randn_like(xb)                               # sensor noise


def center(xb):
    s = (xb.shape[1] - CROP) // 2
    return xb[:, s:s + CROP]


def run_fold(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[2]).mean(0); sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
    Xt = torch.tensor((X[tr] - mu) / sd, device=DEV); yt = torch.tensor(y[tr], device=DEV)
    Xe = torch.tensor((X[te] - mu) / sd, device=DEV)
    w = torch.tensor(len(tr) / (len(classes) * np.bincount(y[tr], minlength=len(classes))), dtype=torch.float32, device=DEV)
    net = Net(X.shape[2], len(classes)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * (len(tr) // 64 + 1))
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(tr), device=DEV)
        for i in range(0, len(tr), 64):
            idx = perm[i:i + 64]
            loss = lossf(net(augment(Xt[idx])), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(center(Xe)), 1).cpu().numpy()


oof = np.zeros((len(X), len(classes)))
for d in sorted(set(driver)):
    tr, te = np.where(driver != d)[0], np.where(driver == d)[0]
    oof[te] = np.mean([run_fold(tr, te, s) for s in range(SEEDS)], axis=0)
    print(d, "window acc", (oof[te].argmax(1) == y[te]).mean().round(3))

from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
df = pd.DataFrame(oof, columns=classes).assign(y=y, trip=trip)
t = df.groupby("trip").agg({**{c: "mean" for c in classes}, "y": "first"})
t["pred"] = t[classes].to_numpy().argmax(1)
m = dict(window_macro_f1=float(f1_score(y, oof.argmax(1), average="macro")),
         trip_accuracy=float((t.pred == t.y).mean()),
         trip_auc_risky=float(roc_auc_score(t.y != classes.index("NORMAL"), 1 - t["NORMAL"])))
print(json.dumps(m, indent=2))
print(pd.crosstab(t.y.map(dict(enumerate(classes))), t.pred.map(dict(enumerate(classes)))))
np.save(OUT / "cnn_oof.npy", oof)
(OUT / "cnn_metrics.json").write_text(json.dumps(m, indent=2))
