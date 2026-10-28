"""Patch Transformer for 1 Hz trip sequences (shared by training and blind evaluation).

Same architecture as scripts/gpu/train_ssl.py: 6 orientation-invariant channels + presence mask, cut into
PATCH-second tokens, Transformer encoder, mean+max pooling head. Pre-trained by masked-patch reconstruction.
"""
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, c, n_patch, patch, d, heads=4, layers=4):
        super().__init__()
        self.inp = nn.Linear(patch * c, d)
        self.pos = nn.Parameter(torch.randn(1, n_patch, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        h = self.inp(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_tok.expand_as(h), h)
        return self.norm(self.tf(h + self.pos))


class Recon(nn.Module):
    def __init__(self, enc, c, patch, d):
        super().__init__()
        self.enc, self.dec = enc, nn.Linear(d, patch * c)

    def forward(self, x, mask):
        return self.dec(self.enc(x, mask))


class Classifier(nn.Module):
    def __init__(self, enc, c, d):
        super().__init__()
        self.enc, self.c = enc, c
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.2), nn.Linear(d, 1))

    def forward(self, x):
        valid = (x[..., self.c - 1::self.c].sum(-1) > 0).float()
        h = self.enc(x)
        mean = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        mx = (h - 1e4 * (1 - valid[..., None])).amax(1)
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


def prepare(X: np.ndarray, mu=None, sd=None):
    """[n, T, 6] raw channels -> normalised + presence channel; returns (Xn, mu, sd)."""
    present = (np.abs(X).sum(-1, keepdims=True) > 0).astype(np.float32)
    if mu is None:
        mu = (X * present).sum((0, 1)) / present.sum((0, 1))
        sd = np.sqrt((((X - mu) * present) ** 2).sum((0, 1)) / present.sum((0, 1))) + 1e-6
    Xn = np.concatenate([(X - mu) / sd * present, present], -1).astype(np.float32)
    return Xn, mu, sd


@torch.no_grad()
def predict(models: list, X: np.ndarray, mu, sd, patch: int, device="cuda", bs=512) -> np.ndarray:
    Xn, _, _ = prepare(X, mu, sd)
    n, T, C = Xn.shape
    Xt = torch.tensor(Xn).view(n, T // patch, patch * C)
    out = np.zeros(n)
    for m in models:
        m.eval().to(device)
        ps = []
        for i in range(0, n, bs):
            with torch.autocast(device, dtype=torch.bfloat16):
                ps.append(torch.sigmoid(m(Xt[i:i + bs].to(device)).float()).cpu().numpy())
        out += np.concatenate(ps) / len(models)
    return out
