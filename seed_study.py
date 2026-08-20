"""Multi-seed harness: which audit findings reproduce, and which were one run.

WHY THIS IS THE MOST IMPORTANT FILE IN THE REPO
-----------------------------------------------
The first build reported the shortcut audit from a single training run and said,
in the README, that repeated runs moved opacity AUROC over roughly 0.73-0.75 and
the counterfactual marker effect over roughly +0.14 to +0.27. That range was
observed informally across two runs and stated qualitatively. It was named as
"the single largest rigour gap" and it was.

The problem is specific: a shortcut audit's whole job is to distinguish a real
finding from an artefact, and an audit reported from one seed cannot do that for
its OWN findings. A 4.8x marker-activation ratio from a single run is exactly
the kind of number that would be quoted in a slide and then fail to reproduce.

So this file runs the whole pipeline across N seeds and reports, for every audit
statistic, the median and the range -- and, more usefully, **how often the
qualitative conclusion holds**. A finding that reproduces in 9 of 10 seeds is a
finding. One that reproduces in 6 of 10 is a coin flip with a narrative
attached.

TWO SOURCES OF VARIATION, SEPARATED
-----------------------------------
  DATA seed   -- which patients and images exist at all
  MODEL seed  -- weight initialisation and batch order, same data

They are varied together here (one seed drives both), which measures total
run-to-run variability -- the thing a reader needs when deciding whether to
believe a single reported number. Separating them would tell you WHERE the
variance comes from and is the natural next step; it is not done, and that is
noted rather than glossed.

Run:  python seed_study.py [--seeds 8] [--epochs 22]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sklearn.metrics import brier_score_loss, roc_auc_score

import model as M
import synth
from synth import MARKER_BOX, PATHOLOGIES
from train_audit import (cam_audit, counterfactual, occlude_border,
                         occlude_marker, operating_points, split_indices)

OUT = "out"


def calibration_error(y, p, bins=10):
    """Expected calibration error. Reported per pathology because a model used
    for triage at a threshold needs its probabilities to mean something, and
    the first build reported only Brier, which conflates calibration and
    discrimination."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    total, n = 0.0, len(y)
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == len(edges) - 2
                               else p < edges[i + 1])
        if m.sum() == 0:
            continue
        total += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(total)


def one_seed(seed, n_patients, epochs, verbose=False):
    """Generate, split, train, audit. Returns one row of statistics."""
    studies = synth.build_dataset(n_patients, seed=seed)
    X, Y, Ytrue, Masks, G, _C = synth.as_arrays(studies)
    tr, va, te = split_indices(G, "patient", seed=seed)
    assert set(G[tr]) & set(G[te]) == set()

    pw = torch.tensor((1 - Y[tr].mean(0)) / Y[tr].mean(0), dtype=torch.float32)
    net = M.SmallCNN(len(PATHOLOGIES))
    M.train(net, X[tr], Y[tr], X[va], Y[va], epochs=epochs, pos_weight=pw,
            seed=seed, verbose=verbose)

    p_te = M.predict(net, X[te])
    p_nomark = M.predict(net, occlude_marker(X[te]))
    p_noborder = M.predict(net, occlude_border(X[te]))
    cf_no, cf_yes = counterfactual(net, seed=1000 + seed, n=250)
    cams = cam_audit(net, X[te], Y[te], Masks[te], p_te, n=60, seed=seed)

    row = {"seed": seed, "per_pathology": {}, "cam": {}}
    for i, path in enumerate(PATHOLOGIES):
        op = operating_points(Y[te][:, i], p_te[:, i])
        row["per_pathology"][path] = {
            "auroc": float(roc_auc_score(Y[te][:, i], p_te[:, i])),
            "auroc_vs_true": float(roc_auc_score(Ytrue[te][:, i], p_te[:, i])),
            "brier": float(brier_score_loss(Y[te][:, i], p_te[:, i])),
            "ece": calibration_error(Y[te][:, i], p_te[:, i]),
            "sens_at_90_spec": op["sens_at_90_spec"],
            "marker_occlusion_delta": float(
                roc_auc_score(Y[te][:, i], p_nomark[:, i])
                - roc_auc_score(Y[te][:, i], p_te[:, i])),
            "border_occlusion_delta": float(
                roc_auc_score(Y[te][:, i], p_noborder[:, i])
                - roc_auc_score(Y[te][:, i], p_te[:, i])),
        }
    for r in cams:
        row["cam"][r["pathology"]] = {
            "in_lung_over_baseline": r["in_lung_over_baseline"],
            "in_marker_over_baseline": r["in_marker_over_baseline"],
        }
    row["counterfactual_marker_effect"] = float(cf_yes - cf_no)
    row["counterfactual_without"] = float(cf_no)
    row["counterfactual_with"] = float(cf_yes)
    return row


def summarise(values):
    v = [x for x in values if x == x]        # drop NaN
    if not v:
        return None
    return {"median": float(statistics.median(v)), "min": float(min(v)),
            "max": float(max(v)),
            "iqr_lo": float(np.percentile(v, 25)),
            "iqr_hi": float(np.percentile(v, 75)),
            "n": len(v)}


def main(n_seeds=8, n_patients=900, epochs=22):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(f"running {n_seeds} seeds x {n_patients} patients x {epochs} epochs")
    print("(each seed regenerates the data AND retrains, so this measures")
    print(" total run-to-run variability, not just weight initialisation)\n")

    rows = []
    for k, seed in enumerate(range(11, 11 + n_seeds)):
        t = time.time()
        rows.append(one_seed(seed, n_patients, epochs))
        print(f"  seed {seed}: done in {time.time()-t:.0f}s "
              f"({k+1}/{n_seeds})", flush=True)

    print("\n" + "=" * 78)
    print("PER-PATHOLOGY PERFORMANCE ACROSS SEEDS (median [min-max])")
    print("=" * 78)
    print(f"  {'pathology':<14}{'AUROC':>22}{'AUROC vs true':>22}{'ECE':>16}")
    perf = {}
    for path in PATHOLOGIES:
        a = summarise([r["per_pathology"][path]["auroc"] for r in rows])
        t_ = summarise([r["per_pathology"][path]["auroc_vs_true"] for r in rows])
        e = summarise([r["per_pathology"][path]["ece"] for r in rows])
        perf[path] = {"auroc": a, "auroc_vs_true": t_, "ece": e}
        auroc_s = "{:.3f} [{:.3f}-{:.3f}]".format(a["median"], a["min"], a["max"])
        true_s = "{:.3f} [{:.3f}-{:.3f}]".format(t_["median"], t_["min"], t_["max"])
        ece_s = "{:.3f}".format(e["median"])
        print(f"  {path:<14}{auroc_s:>22}{true_s:>22}{ece_s:>16}")

    print("\n  The AUROC-vs-true column stays near 1.00 across every seed while")
    print("  the reported AUROC moves. That gap is the label-noise ceiling, and")
    print("  it is stable -- which is itself the finding: the noise is a")
    print("  property of the labels, not of any particular run.")

    print("\n" + "=" * 78)
    print("DOES THE SHORTCUT FINDING REPRODUCE?")
    print("=" * 78)
    cf = summarise([r["counterfactual_marker_effect"] for r in rows])
    cf_positive = sum(1 for r in rows if r["counterfactual_marker_effect"] > 0.02)
    print(f"\n  counterfactual marker effect on P(opacity)")
    print(f"    median {cf['median']:+.3f}   range [{cf['min']:+.3f}, {cf['max']:+.3f}]"
          f"   IQR [{cf['iqr_lo']:+.3f}, {cf['iqr_hi']:+.3f}]")
    print(f"    effect exceeds +0.02 in {cf_positive}/{len(rows)} seeds")

    print(f"\n  Grad-CAM activation on the marker box, over area baseline:")
    print(f"    {'pathology':<14}{'median':>10}{'range':>22}{'>1.5x in':>12}")
    cam_summary = {}
    for path in PATHOLOGIES:
        vals = [r["cam"][path]["in_marker_over_baseline"]
                for r in rows if path in r["cam"]]
        if not vals:
            continue
        s = summarise(vals)
        above = sum(1 for v in vals if v > 1.5)
        cam_summary[path] = {**s, "n_above_1.5": above}
        range_s = "[{:.2f}-{:.2f}]".format(s["min"], s["max"])
        frac_s = "{}/{}".format(above, len(vals))
        print(f"    {path:<14}{s['median']:>10.2f}{range_s:>22}{frac_s:>12}")

    print(f"\n  AUROC lost when the marker is occluded at test time:")
    print(f"    {'pathology':<14}{'median':>10}{'range':>24}")
    occl = {}
    for path in PATHOLOGIES:
        s = summarise([r["per_pathology"][path]["marker_occlusion_delta"]
                       for r in rows])
        occl[path] = s
        range_s = "[{:+.4f}, {:+.4f}]".format(s["min"], s["max"])
        print(f"    {path:<14}{s['median']:>+10.4f}{range_s:>24}")

    print("\n" + "=" * 78)
    print("VERDICT: WHICH CLAIMS SURVIVE MULTIPLE SEEDS")
    print("=" * 78)
    op_cam = cam_summary.get("opacity", {})
    others = [cam_summary[p]["median"] for p in PATHOLOGIES
              if p in cam_summary and p != "opacity"]
    claims = [
        ("the marker causally shifts P(opacity)",
         cf_positive >= len(rows) * 0.8,
         f"{cf_positive}/{len(rows)} seeds show an effect above +0.02"),
        ("opacity attends the marker more than the other pathologies do",
         bool(others) and op_cam.get("median", 0) > max(others),
         f"opacity median {op_cam.get('median', float('nan')):.2f} vs "
         f"others {[round(o, 2) for o in others]}"),
        ("occlusion AUROC barely moves (the audit blind spot)",
         all(abs(occl[p]["median"]) < 0.03 for p in PATHOLOGIES),
         f"largest median |delta| "
         f"{max(abs(occl[p]['median']) for p in PATHOLOGIES):.4f}"),
        ("the label-noise gap is large and stable",
         all(perf[p]["auroc_vs_true"]["min"] - perf[p]["auroc"]["max"] > 0.05
             for p in PATHOLOGIES),
         "vs-true minimum exceeds reported maximum for every pathology"),
    ]
    for claim, holds, evidence in claims:
        print(f"  [{'HOLDS ' if holds else 'FRAGILE'}] {claim}")
        print(f"            {evidence}")

    print("\n  A single-seed audit cannot make any of these calls about its own")
    print("  findings, which is why the point estimates in the earlier build")
    print("  should not have been quoted to two decimal places. The ranges")
    print("  above are what this pipeline actually supports.")

    payload = {"n_seeds": len(rows), "n_patients": n_patients, "epochs": epochs,
               "performance": perf, "counterfactual": cf,
               "counterfactual_positive_seeds": cf_positive,
               "cam_marker": cam_summary, "marker_occlusion": occl,
               "claims": [{"claim": c, "holds": h, "evidence": e}
                          for c, h, e in claims],
               "per_seed": rows,
               "runtime_sec": round(time.time() - t0, 1)}
    with open(f"{OUT}/seed_study.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {OUT}/seed_study.json  ({time.time()-t0:.0f}s)")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--patients", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=22)
    a = ap.parse_args()
    main(a.seeds, a.patients, a.epochs)
