"""Train, evaluate at operating points, and audit for shortcut learning.

The centrepiece is the shortcut audit, and the design principle is that
FINDING a shortcut is a better outcome than claiming none exist. Three
confounds were planted with known strengths; the audit's job is to recover
them from the model's behaviour without being told.

Three independent lines of evidence, because a heatmap alone proves nothing:

  1. SPATIAL   -- what fraction of Grad-CAM activation falls inside the true
                  lung mask, and what fraction inside the laterality-marker
                  box? Compared against the area baseline, since a uniform map
                  would put `area` of its mass anywhere by construction.
  2. CAUSAL    -- occlude the confound at test time and re-measure AUROC. If
                  performance falls, the model was using it. This is the one
                  that actually settles the argument.
  3. COUNTERFACTUAL -- score images that differ ONLY in the confound. Any
                  change in the mean prediction is attributable to the
                  confound alone.

Then a mitigation (occlusion augmentation), re-audited honestly including what
it costs.

Run:  python train_audit.py [--split patient|image]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

import synth
from synth import CONFOUND_STRENGTH, IMG, MARKER_BOX, PATHOLOGIES
import model as M

OUT = "out"
DOC = "docs/SHORTCUT_AUDIT.md"


# ---------------------------------------------------------------------------
def split_indices(groups, mode="patient", seed=3, fracs=(0.65, 0.15, 0.20)):
    rng = np.random.default_rng(seed)
    if mode == "patient":
        pats = np.unique(groups)
        rng.shuffle(pats)
        n1 = int(len(pats) * fracs[0])
        n2 = n1 + int(len(pats) * fracs[1])
        sets = [set(pats[:n1]), set(pats[n1:n2]), set(pats[n2:])]
        return [np.where([g in s for g in groups])[0] for s in sets]
    idx = np.arange(len(groups))
    rng.shuffle(idx)
    n1 = int(len(idx) * fracs[0])
    n2 = n1 + int(len(idx) * fracs[1])
    return [idx[:n1], idx[n1:n2], idx[n2:]]


def operating_points(y, p):
    """Radiology is deployed at thresholds, not at AUCs, and the threshold
    depends on the job: triage wants sensitivity, confirmation wants
    specificity. Report both."""
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    # Among all points MEETING the constraint, take the best on the other axis.
    # argmin(|spec-0.90|) is wrong: a near-perfect classifier has a ROC point at
    # spec=1.0, tpr=0.0 (the trivial "predict nothing" corner), which is 0.10
    # from the target and wins the argmin -- reporting 0% sensitivity for a
    # model with AUROC 1.000. That is what the first version of this did.
    ok = np.where(spec >= 0.90)[0]
    i_sens = int(ok[np.argmax(tpr[ok])]) if len(ok) else int(np.argmax(spec))
    ok2 = np.where(tpr >= 0.90)[0]
    i_spec = int(ok2[np.argmax(spec[ok2])]) if len(ok2) else int(np.argmax(tpr))
    return {
        "sens_at_90_spec": float(tpr[i_sens]),
        "threshold_at_90_spec": float(thr[i_sens]),
        "spec_at_90_sens": float(spec[i_spec]),
        "threshold_at_90_sens": float(thr[i_spec]),
    }


def occlude_marker(X):
    """Replace the marker region with the image's own border-region median."""
    X = X.copy()
    r0, r1, c0, c1 = MARKER_BOX
    for i in range(len(X)):
        fill = np.median(X[i, 0, r0:r1, :c0])
        X[i, 0, r0:r1, c0:c1] = fill
    return X


def occlude_border(X):
    X = X.copy()
    w = synth.BORDER_WIDTH
    for i in range(len(X)):
        inner = X[i, 0, w + 1:-(w + 1), w + 1:-(w + 1)]
        fill = np.median(inner)
        X[i, 0, :w, :] = fill
        X[i, 0, -w:, :] = fill
        X[i, 0, :, :w] = fill
        X[i, 0, :, -w:] = fill
    return X


def augment_marker_occlusion(xb, p=0.5):
    """Training-time mitigation: randomly blank the marker box so the model
    cannot rely on it. Cheap, and it does not require relabelling anything."""
    xb = xb.clone()
    r0, r1, c0, c1 = MARKER_BOX
    mask = torch.rand(len(xb)) < p
    if mask.any():
        fill = xb[mask][:, :, r0:r1, :c0].median()
        xb[mask, :, r0:r1, c0:c1] = fill
    return xb


def cam_audit(net, X, Y, M_masks, preds, n=120, seed=5):
    """Spatial evidence: where does the activation land?"""
    rng = np.random.default_rng(seed)
    marker_area = ((MARKER_BOX[1] - MARKER_BOX[0])
                   * (MARKER_BOX[3] - MARKER_BOX[2])) / (IMG * IMG)
    rows = []
    for ci, path in enumerate(PATHOLOGIES):
        pos = np.where((Y[:, ci] == 1) & (preds[:, ci] > 0.5))[0]
        if len(pos) == 0:
            # fall back to the highest-scoring true positives, so a pathology
            # the model is diffident about still gets audited rather than
            # silently dropping out of the table
            truth = np.where(Y[:, ci] == 1)[0]
            if len(truth) == 0:
                continue
            pos = truth[np.argsort(-preds[truth, ci])[:n]]
        pick = rng.choice(pos, size=min(n, len(pos)), replace=False)
        in_lung, in_marker = [], []
        for i in pick:
            cam = M.grad_cam(net, torch.tensor(X[i:i + 1]), ci)
            tot = cam.sum()
            if tot <= 0:
                continue
            in_lung.append(float(cam[M_masks[i]].sum() / tot))
            r0, r1, c0, c1 = MARKER_BOX
            in_marker.append(float(cam[r0:r1, c0:c1].sum() / tot))
        lung_area = float(M_masks[pick].mean())
        rows.append({
            "pathology": path, "n_sampled": len(in_lung),
            "in_lung_fraction": float(np.mean(in_lung)),
            "lung_area_fraction": lung_area,
            "in_lung_over_baseline": float(np.mean(in_lung)) / lung_area,
            "in_marker_fraction": float(np.mean(in_marker)),
            "marker_area_fraction": marker_area,
            "in_marker_over_baseline": float(np.mean(in_marker)) / marker_area,
        })
    return rows


def counterfactual(net, seed=17, n=400):
    """Render matched pairs differing ONLY in the marker, and score both."""
    with_marker, without = [], []
    for i in range(n):
        labels = {p: False for p in PATHOLOGIES}
        arng = np.random.default_rng(2000 + i)
        an = {
            "lung_r": IMG * 0.47, "lung_c_left": IMG * 0.31,
            "lung_c_right": IMG * 0.69, "lung_rr": IMG * 0.235,
            "lung_cc": IMG * 0.145, "heart_w": IMG * 0.11,
            "rib_phase": float(arng.random() * 6.28), "opacity_side": 0,
        }
        img, _, _ = synth.make_study(np.random.default_rng(3000 + i), an, labels)
        clean = img.copy()
        marked = img.copy()
        r0, r1, c0, c1 = MARKER_BOX
        marked[r0:r1, c0:c1] += 0.05
        marked[r0:r1, c0:c0 + 2] += 0.55
        marked[r0:r0 + 2, c0:c1 - 1] += 0.55
        marked[r0 + 3:r0 + 5, c0:c1 - 1] += 0.55
        marked[r0 + 5:r1, c1 - 3:c1 - 1] += 0.55
        without.append(clean)
        with_marker.append(np.clip(marked, 0, 1))
    A = np.stack(without)[:, None]
    B = np.stack(with_marker)[:, None]
    pa = M.predict(net, A.astype(np.float32))
    pb = M.predict(net, B.astype(np.float32))
    oi = PATHOLOGIES.index("opacity")
    return float(pa[:, oi].mean()), float(pb[:, oi].mean())


# ---------------------------------------------------------------------------
def main(split_mode="patient", n_patients=900, epochs=18):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print(f"generating {n_patients} patients...")
    studies = synth.build_dataset(n_patients)
    X, Y, Ytrue, Masks, G, C = synth.as_arrays(studies)
    print(f"  {len(X):,} studies from {len(np.unique(G)):,} patients")
    for i, p in enumerate(PATHOLOGIES):
        print(f"  prevalence {p:<14} {Y[:, i].mean():.1%}")

    tr, va, te = split_indices(G, split_mode)
    overlap = len(set(G[tr]) & set(G[te]))
    print(f"\nsplit={split_mode}  train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    print(f"  patients appearing in BOTH train and test: {overlap}")
    if split_mode == "patient":
        assert overlap == 0, "patient-level split must be disjoint"

    pw = torch.tensor((1 - Y[tr].mean(0)) / Y[tr].mean(0), dtype=torch.float32)
    print(f"  positive weights {pw.numpy().round(2)}")

    print("\ntraining baseline...")
    net = M.SmallCNN(len(PATHOLOGIES))
    M.train(net, X[tr], Y[tr], X[va], Y[va], epochs=epochs, pos_weight=pw)
    p_te = M.predict(net, X[te])

    print("\nPER-PATHOLOGY PERFORMANCE (test)")
    print("  labels carry {:.0%} simulated mining error, so the AUROC"
          .format(synth.LABEL_NOISE))
    print("  column is the honest, reportable number. The vs-true column --")
    print("  unavailable on any real dataset -- is the ceiling the labels")
    print("  impose: the gap is what the model is punished for that is not")
    print("  its fault, and it is why beating a published AUROC should worry")
    print("  you before it pleases you.")
    print()
    print(f"  {'pathology':<14}{'prev':>7}{'AUROC':>8}{'vs true':>9}"
          f"{'sens@90spec':>13}{'spec@90sens':>13}{'Brier':>8}")
    perf = {}
    for i, p in enumerate(PATHOLOGIES):
        auc = roc_auc_score(Y[te][:, i], p_te[:, i])
        auc_true = roc_auc_score(Ytrue[te][:, i], p_te[:, i])
        op = operating_points(Y[te][:, i], p_te[:, i])
        br = brier_score_loss(Y[te][:, i], p_te[:, i])
        perf[p] = {"auroc": float(auc), "auroc_vs_true_label": float(auc_true),
                   "brier": float(br), **op}
        print(f"  {p:<14}{Y[te][:, i].mean():>7.1%}{auc:>8.3f}{auc_true:>9.3f}"
              f"{op['sens_at_90_spec']:>13.1%}{op['spec_at_90_sens']:>13.1%}{br:>8.3f}")

    # ---- audit ----------------------------------------------------------
    print("\n" + "=" * 76)
    print("SHORTCUT AUDIT")
    print("=" * 76)
    print("\n[spatial] Grad-CAM mass inside the true lung mask and the marker box")
    print("  'over baseline' divides by the region's area fraction: 1.0 means the")
    print("  activation is no more concentrated there than chance would put it.")
    cams = cam_audit(net, X[te], Y[te], Masks[te], p_te)
    print(f"  {'pathology':<14}{'in-lung':>9}{'x base':>8}{'in-marker':>11}{'x base':>9}")
    for r in cams:
        print(f"  {r['pathology']:<14}{r['in_lung_fraction']:>9.1%}"
              f"{r['in_lung_over_baseline']:>8.2f}{r['in_marker_fraction']:>11.1%}"
              f"{r['in_marker_over_baseline']:>9.2f}")

    print("\n[causal] occlude a confound at test time, re-measure AUROC")
    print(f"  {'pathology':<14}{'baseline':>10}{'no marker':>11}{'delta':>8}"
          f"{'no border':>11}{'delta':>8}")
    p_nomark = M.predict(net, occlude_marker(X[te]))
    p_noborder = M.predict(net, occlude_border(X[te]))
    causal = {}
    for i, p in enumerate(PATHOLOGIES):
        a0 = roc_auc_score(Y[te][:, i], p_te[:, i])
        a1 = roc_auc_score(Y[te][:, i], p_nomark[:, i])
        a2 = roc_auc_score(Y[te][:, i], p_noborder[:, i])
        causal[p] = {"baseline": a0, "marker_occluded": a1, "border_occluded": a2}
        print(f"  {p:<14}{a0:>10.3f}{a1:>11.3f}{a1-a0:>+8.3f}{a2:>11.3f}{a2-a0:>+8.3f}")

    print("\n[counterfactual] identical images, marker added; mean P(opacity)")
    cf_no, cf_yes = counterfactual(net)
    print(f"  without marker {cf_no:.3f}   with marker {cf_yes:.3f}   "
          f"delta {cf_yes-cf_no:+.3f}")

    # ---- mitigation ------------------------------------------------------
    print("\n" + "=" * 76)
    print("MITIGATION: marker-occlusion augmentation, then re-audit")
    print("=" * 76)
    net2 = M.SmallCNN(len(PATHOLOGIES))
    M.train(net2, X[tr], Y[tr], X[va], Y[va], epochs=epochs, pos_weight=pw,
            augment=augment_marker_occlusion, seed=1)
    p2 = M.predict(net2, X[te])
    p2_nomark = M.predict(net2, occlude_marker(X[te]))
    cf2_no, cf2_yes = counterfactual(net2)
    print(f"  {'pathology':<14}{'AUROC before':>14}{'AUROC after':>13}{'delta':>8}")
    mitig = {}
    for i, p in enumerate(PATHOLOGIES):
        a0 = roc_auc_score(Y[te][:, i], p_te[:, i])
        a1 = roc_auc_score(Y[te][:, i], p2[:, i])
        mitig[p] = {"before": a0, "after": a1}
        print(f"  {p:<14}{a0:>14.3f}{a1:>13.3f}{a1-a0:>+8.3f}")
    oi = PATHOLOGIES.index("opacity")
    rel_before = (roc_auc_score(Y[te][:, oi], p_te[:, oi])
                  - roc_auc_score(Y[te][:, oi], p_nomark[:, oi]))
    rel_after = (roc_auc_score(Y[te][:, oi], p2[:, oi])
                 - roc_auc_score(Y[te][:, oi], p2_nomark[:, oi]))
    print(f"\n  reliance on the marker (AUROC lost when it is occluded):")
    print(f"    before mitigation {rel_before:+.3f}   after {rel_after:+.3f}")
    print(f"  counterfactual marker effect on P(opacity):")
    print(f"    before {cf_yes-cf_no:+.3f}   after {cf2_yes-cf2_no:+.3f}")

    payload = {
        "planted": {k: {"target": v[0], "p_given_positive": v[1],
                        "p_given_negative": v[2]}
                    for k, v in CONFOUND_STRENGTH.items()},
        "split": split_mode, "n_studies": int(len(X)),
        "n_patients": int(len(np.unique(G))), "train_test_patient_overlap": overlap,
        "performance": perf, "cam_audit": cams, "causal": causal,
        "counterfactual": {"without_marker": cf_no, "with_marker": cf_yes,
                           "after_mitigation": {"without": cf2_no, "with": cf2_yes}},
        "mitigation": mitig,
        "marker_reliance": {"before": rel_before, "after": rel_after},
    }
    with open(f"{OUT}/audit_{split_mode}.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {OUT}/audit_{split_mode}.json")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="patient", choices=["patient", "image"])
    ap.add_argument("--patients", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=18)
    a = ap.parse_args()
    main(a.split, a.patients, a.epochs)
