"""Final self-supervised Transformer: pre-train on all Grab dev sequences (no labels), fine-tune on all dev
labels with 3 seeds, freeze to models/trip_ssl/ for src/blind_eval.py. Blind sequences are never touched.

Usage: python scripts/gpu/train_ssl_final.py [--patch 5 --d 128 --pre-epochs 40 --ft-epochs 12]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ssl_model import Classifier, Encoder, Recon, prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--patch", type=int, default=5)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--pre-epochs", type=int, default=40)
ap.add_argument("--ft-epochs", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
DEV, BS = "cuda", 128
OUT = ROOT / "models/trip_ssl"
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(ROOT / "data/processed/grab_dev_seq.npz")
Xn, mu, sd = prepare(z["X"].astype(np.float32))
y = z["y"].astype(np.int64)
n, T, C = Xn.shape
NP = T // args.patch
Xt = torch.tensor(Xn).view(n, NP, args.patch * C).to(DEV)
yt = torch.tensor(y, device=DEV).float()
del Xn
print(f"{n} dev trips on {torch.cuda.get_device_name(0)}", flush=True)


def make_enc():
    return Encoder(C, NP, args.patch, args.d, 4, args.layers)


# 1) pre-train once on all dev sequences (labels unused)
torch.manual_seed(0)
enc = make_enc()
rec = Recon(enc, C, args.patch, args.d).to(DEV)
opt = torch.optim.AdamW(rec.parameters(), 1e-3, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.pre_epochs * (n // BS + 1))
for ep in range(args.pre_epochs):
    perm, tot = torch.randperm(n, device=DEV), 0.0
    for i in range(0, n, BS):
        xb = Xt[perm[i:i + BS]]
        valid = xb[..., C - 1::C].sum(-1) > 0
        mask = (torch.rand(xb.shape[:2], device=DEV) < 0.4) & valid
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = rec(xb, mask)
        loss = ((pred.float() - xb) ** 2).mean(-1)[mask].mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(rec.parameters(), 1.0); opt.step(); sched.step()
        tot += loss.item()
    if ep % 5 == 0 or ep == args.pre_epochs - 1:
        print(f"pretrain ep {ep} loss {tot / (n / BS):.4f}", flush=True)
init = enc.state_dict()

# 2) fine-tune on all dev labels, several seeds
pr = yt.mean()
lossf = nn.BCEWithLogitsLoss(pos_weight=(1 - pr) / pr)
files = []
for seed in range(args.seeds):
    torch.manual_seed(100 + seed)
    e = make_enc()
    e.load_state_dict(init)
    m = Classifier(e, C, args.d).to(DEV)
    opt = torch.optim.AdamW([{"params": m.enc.parameters(), "lr": 3e-4}, {"params": m.head.parameters(), "lr": 1e-3}],
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [3e-4, 1e-3], total_steps=args.ft_epochs * (n // BS + 1))
    for ep in range(args.ft_epochs):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BS):
            b = perm[i:i + BS]
            xb = Xt[b] * (1 + 0.05 * torch.randn(len(b), 1, Xt.shape[2], device=DEV))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(m(xb).float(), yt[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    f = OUT / f"classifier_seed{seed}.pt"
    torch.save(m.state_dict(), f)
    files.append(f.name)
    print(f"seed {seed} saved", flush=True)

sha = hashlib.sha256(b"".join((OUT / f).read_bytes() for f in files)).hexdigest()[:12]
(OUT / "meta.json").write_text(json.dumps(dict(config=vars(args), files=files, mu=mu.tolist(), sd=sd.tolist(),
                                               channels=7, n_patch=NP, model_sha=sha), indent=2))
print(f"frozen -> {OUT} (sha {sha})")
