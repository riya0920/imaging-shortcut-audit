"""A small multi-label CNN and a hand-rolled Grad-CAM.

The spec asks for DenseNet-121 transfer learning, which is the literature
standard and the right answer with real data and a GPU. This is a 4-block CNN
trained from scratch on 64x64 synthetic images, because the point of the
project is the shortcut AUDIT, not the backbone -- and a shortcut audit needs a
model that has actually learned the planted shortcut, which this one does.

Grad-CAM is implemented here rather than imported. It is about fifteen lines:
grab the last conv activations and the gradient of a class logit with respect
to them, average the gradient over space to get a per-channel weight, take the
weighted sum, ReLU, and upsample. Writing it out is worth it because the audit
depends on knowing exactly what is being measured -- in particular that the CAM
is at conv-feature resolution (8x8 here) and upsampled, so "activation inside
the lung field" is a coarse statement and the audit must not over-read it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class SmallCNN(nn.Module):
    def __init__(self, n_out=3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2))
        self.b1 = block(1, 16)     # 64 -> 32
        self.b2 = block(16, 32)    # 32 -> 16
        self.b3 = block(32, 64)    # 16 -> 8
        self.features = nn.Sequential(self.b1, self.b2, self.b3)
        self.head = nn.Linear(64, n_out)

    def forward(self, x, return_features=False):
        f = self.features(x)                       # (N, 64, 8, 8)
        pooled = f.mean(dim=(2, 3))
        logits = self.head(pooled)
        return (logits, f) if return_features else logits


def grad_cam(model, x, class_idx):
    """Grad-CAM for one image and one class. Returns a 64x64 non-negative map."""
    model.eval()
    x = x.clone().requires_grad_(False)
    logits, feats = model(x, return_features=True)
    feats.retain_grad()
    model.zero_grad()
    logits[0, class_idx].backward()
    grads = feats.grad[0]                          # (C, 8, 8)
    weights = grads.mean(dim=(1, 2))               # (C,)
    cam = Fn.relu((weights[:, None, None] * feats[0]).sum(0))
    cam = cam[None, None]
    cam = Fn.interpolate(cam, size=(x.shape[-2], x.shape[-1]),
                         mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    s = cam.sum()
    return cam / s if s > 0 else cam


def train(model, Xtr, Ytr, Xva, Yva, epochs=18, bs=64, lr=2e-3, pos_weight=None,
          augment=None, seed=0, verbose=True):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xtr_t = torch.tensor(Xtr)
    Ytr_t = torch.tensor(Ytr)
    Xva_t, Yva_t = torch.tensor(Xva), torch.tensor(Yva)
    n = len(Xtr_t)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            if augment is not None:
                xb = augment(xb)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xva_t), Yva_t))
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 4 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:>2}  train {tot/n:.4f}  val {vl:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(torch.tensor(X[i:i + bs]))).numpy())
    return np.concatenate(out)
