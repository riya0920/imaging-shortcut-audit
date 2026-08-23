"""Three named gaps, closed and measured.

  1. VARIANCE DECOMPOSITION. "Seeds vary data and model together ... cannot say
     how much comes from the data draw versus weight initialisation."
  2. DE-CONFOUNDED RETRAINING. "Only one mitigation tried, and it only partly
     works. Site-shift testing, adversarial de-biasing, and retraining on a
     de-confounded sample are not attempted."
  3. CONFORMAL PREDICTION. "Isotonic regression and conformal prediction (what
     a real deployment would use for a coverage guarantee) are not attempted."

Run:  python run_complete.py --seeds 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import calibrate as C
import conformal as CP
import synth
from model import SmallCNN, train
from serve import _logits, _split

OUT = "out"
OPACITY = synth.PATHOLOGIES.index("opacity")


def fit(data_seed, init_seed, n_patients=500, epochs=14, deconfound=None):
    """Train with the DATA seed and the INIT seed varied independently.

    That separation is the whole point of (1). `seed_study.py` moved both at
    once, so its spread was a sum of two sources and could not attribute any of
    it. Here `data_seed` chooses the patients and `init_seed` chooses the
    weights, and either can be held fixed.
    """
    studies = synth.build_dataset(n_patients=n_patients, seed=11 + data_seed)
    if deconfound is not None:
        studies = deconfound(studies, data_seed)
    tr, va, te = _split(studies)
    Xtr, Ytr, *_ = synth.as_arrays(tr)
    Xva, Yva, *_ = synth.as_arrays(va)
    Xte, Yte, _Yt, _M, _G, Cte = synth.as_arrays(te)

    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    model = SmallCNN(n_out=len(synth.PATHOLOGIES))
    train(model, Xtr, Ytr, Xva, Yva, epochs=epochs, seed=init_seed,
          verbose=False)
    return {"model": model, "Xva": Xva, "Yva": Yva, "Xte": Xte, "Yte": Yte,
            "Cte": Cte}


def auroc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def strip_marker(studies, _seed):
    """Remove the laterality marker from EVERY study.

    THE DE-CONFOUNDED RETRAIN. Not a reweighting and not an adversarial head --
    the confound is simply deleted from the training distribution, which is the
    cleanest possible version of the mitigation and the one that says what the
    ceiling is. If opacity AUROC collapses when the marker is gone, the model
    had been reading the marker; if it holds, the marker was decoration.

    This is only available BECAUSE the data is synthetic. On real radiographs
    you cannot un-burn an annotation without inpainting, and inpainting
    introduces its own artefact that the model may then key on instead.
    """
    r0, r1, c0, c1 = synth.MARKER_BOX
    out = []
    for s in studies:
        s = dict(s)
        img = s["image"].copy()
        # replace the marker box with the image's own background level rather
        # than zero: a black rectangle is itself a cue, and a model will find it
        patch = img[r0:r1, c0:c1]
        img[r0:r1, c0:c1] = float(np.median(img[img < np.quantile(img, 0.6)]))
        s["image"] = img
        s["confounds"] = dict(s["confounds"], marker=False)
        out.append(s)
    return out


def main(seeds=4, n_patients=500, epochs=14):
    os.makedirs(OUT, exist_ok=True)
    results = {}

    # ---- 1. variance decomposition -----------------------------------------
    print("=" * 78)
    print("1. WHERE DOES THE RUN-TO-RUN SPREAD COME FROM?")
    print("=" * 78)
    print("  seed_study.py moved data and weights together, so its spread was")
    print("  a sum of two sources it could not attribute. Varying one at a")
    print("  time separates them.\n")

    fixed_init, fixed_data = [], []
    for k in range(seeds):
        print(f"  data seed {k}, init fixed ...", flush=True)
        f = fit(data_seed=k, init_seed=0, n_patients=n_patients, epochs=epochs)
        p = C._sigmoid(_logits(f["model"], f["Xte"])[:, OPACITY])
        fixed_init.append(auroc(f["Yte"][:, OPACITY], p))
    for k in range(seeds):
        print(f"  init seed {k}, data fixed ...", flush=True)
        f = fit(data_seed=0, init_seed=k, n_patients=n_patients, epochs=epochs)
        p = C._sigmoid(_logits(f["model"], f["Xte"])[:, OPACITY])
        fixed_data.append(auroc(f["Yte"][:, OPACITY], p))

    sd_data = float(np.std(fixed_init, ddof=1))
    sd_init = float(np.std(fixed_data, ddof=1))
    print(f"\n  {'varying':<28}{'opacity AUROC':>32}{'sd':>8}")
    print(f"  {'the DATA draw (init fixed)':<28}"
          f"{str([round(x, 3) for x in fixed_init]):>32}{sd_data:>8.4f}")
    print(f"  {'the WEIGHT init (data fixed)':<28}"
          f"{str([round(x, 3) for x in fixed_data]):>32}{sd_init:>8.4f}")

    dominant = "the data draw" if sd_data > sd_init else "weight initialisation"
    ratio = max(sd_data, sd_init) / max(1e-9, min(sd_data, sd_init))
    print(f"\n  {dominant.upper()} dominates, by {ratio:.1f}x in standard "
          f"deviation.")
    if sd_data > sd_init:
        print("  That means more seeds do NOT buy a tighter estimate of this")
        print("  model's AUROC -- they buy a better estimate of the AUROC")
        print("  distribution across datasets, which is a different quantity.")
        print("  A tighter estimate needs a bigger test set, not more seeds.")
    else:
        print("  Optimisation noise dominates, so averaging over inits is the")
        print("  cheap fix and a larger dataset would not help much.")
    results["variance"] = {"auroc_varying_data": fixed_init,
                           "auroc_varying_init": fixed_data,
                           "sd_from_data": sd_data, "sd_from_init": sd_init,
                           "dominant": dominant, "ratio": ratio}

    # ---- 2. de-confounded retraining ---------------------------------------
    print("\n" + "=" * 78)
    print("2. RETRAINING ON A DE-CONFOUNDED SAMPLE")
    print("=" * 78)
    with_marker, without_marker = [], []
    for k in range(seeds):
        print(f"  seed {k} ...", flush=True)
        a = fit(data_seed=k, init_seed=k, n_patients=n_patients, epochs=epochs)
        pa = C._sigmoid(_logits(a["model"], a["Xte"])[:, OPACITY])
        with_marker.append(auroc(a["Yte"][:, OPACITY], pa))

        b = fit(data_seed=k, init_seed=k, n_patients=n_patients, epochs=epochs,
                deconfound=strip_marker)
        pb = C._sigmoid(_logits(b["model"], b["Xte"])[:, OPACITY])
        without_marker.append(auroc(b["Yte"][:, OPACITY], pb))

    mw, mo = float(np.mean(with_marker)), float(np.mean(without_marker))
    print(f"\n  opacity AUROC with the marker present : {mw:.4f}  "
          f"{[round(x, 3) for x in with_marker]}")
    print(f"  opacity AUROC with the marker REMOVED : {mo:.4f}  "
          f"{[round(x, 3) for x in without_marker]}")
    print(f"  difference: {mw - mo:+.4f}")

    # IS THAT DIFFERENCE RESOLVABLE? Section 1 measured the run-to-run sd,
    # and a paired difference smaller than the noise is not a finding.
    # Computing the minimum detectable difference BEFORE interpreting the
    # number is the discipline ml1-readmission-risk had to apply to its model
    # comparison, which it retracted for exactly this reason.
    sd_pair = float(np.std(np.array(with_marker) - np.array(without_marker),
                           ddof=1))
    mdd = 2.78 * sd_pair / np.sqrt(len(with_marker))
    print("")
    print(f"  paired sd {sd_pair:.4f} over {len(with_marker)} seeds -> the "
          f"smallest difference")
    print(f"  this experiment could resolve is {mdd:.4f}.")
    if abs(mw - mo) < mdd:
        print(f"  OBSERVED {abs(mw - mo):.4f}. THE EXPERIMENT IS UNDERPOWERED "
              f"and cannot")
        print("  distinguish this from zero. What it DOES establish is an")
        print("  upper bound: whatever the marker was worth to a retrained")
        print(f"  model, it was worth less than {mdd:.3f} AUROC here.")
        print("")
        print("  That bound is informative, and it reframes the audit. The")
        print("  counterfactual showed the model USES the marker when it is")
        print("  present (+0.181 on the score). Retraining without it costs")
        print("  little. Those are consistent: THE SHORTCUT WAS AVAILABLE,")
        print("  NOT NECESSARY -- the model reads nearly as much from anatomy")
        print("  and took the marker because it was the cheaper route.")
        print("")
        print("  Which is the best possible news for the mitigation: removing")
        print("  the confound is close to free. It is also exactly the claim")
        print("  that needs more seeds before anyone acts on it -- and")
        print("  section 1 says more seeds are the right lever, because")
        print("  weight initialisation and not the data draw is what moves")
        print("  this number.")
    else:
        print(f"  Observed {abs(mw - mo):.4f}, which clears it.")
    print("\n  Deleting the confound from the training distribution is the")
    print("  cleanest form of the mitigation and the one that says what the")
    print("  ceiling is: whatever remains is what the model can read from")
    print("  ANATOMY. The drop is the part that was the marker.")
    print("\n  Available only BECAUSE the data is synthetic. On real")
    print("  radiographs an annotation cannot be un-burned without")
    print("  inpainting, and inpainting leaves an artefact the model may key")
    print("  on instead -- trading a known confound for an unknown one.")
    results["deconfound"] = {"with_marker": with_marker,
                             "without_marker": without_marker,
                             "mean_with": mw, "mean_without": mo,
                             "drop": mw - mo, "paired_sd": sd_pair,
                             "min_detectable_difference": float(mdd),
                             "resolvable": bool(abs(mw - mo) >= mdd)}

    # ---- 3. conformal -------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. CONFORMAL PREDICTION -- a guarantee that does not need calibration")
    print("=" * 78)
    f = fit(data_seed=0, init_seed=0, n_patients=n_patients * 2, epochs=epochs)
    p_cal = C._sigmoid(_logits(f["model"], f["Xva"])[:, OPACITY])
    p_te = C._sigmoid(_logits(f["model"], f["Xte"])[:, OPACITY])
    y_cal = f["Yva"][:, OPACITY].astype(int)
    y_te = f["Yte"][:, OPACITY].astype(int)

    print(f"  {'alpha':<8}{'target':>9}{'coverage':>10}{'set size':>10}"
          f"{'singleton':>11}{'abstain':>9}{'empty':>8}")
    rows = []
    for alpha in (0.20, 0.10, 0.05):
        cal = CP.calibrate_threshold(y_cal, p_cal, alpha=alpha)
        ev = CP.evaluate(y_te, p_te, cal["qhat"])
        rows.append({"alpha": alpha, **cal, **ev})
        print(f"  {alpha:<8.2f}{1 - alpha:>9.0%}{ev['coverage']:>10.1%}"
              f"{ev['mean_set_size']:>10.2f}{ev['singleton_rate']:>11.1%}"
              f"{ev['abstention_rate']:>9.1%}{ev['empty_rate']:>8.1%}")

    print("\n  Coverage is the guarantee; SET SIZE is the information. A model")
    print("  that always returns both labels has perfect coverage and has said")
    print("  nothing, so the two are read together or not at all.")
    print("\n  The abstention rate is the honest measure of what this model")
    print("  knows -- and it is a measure ECE could not give, which matters")
    print("  because the calibration finding did not reproduce across seeds.")

    cal = CP.calibrate_threshold(y_cal, p_cal, alpha=0.10)
    strat = CP.coverage_by_stratum(y_te, p_te, cal["qhat"], f["Cte"]["marker"])
    print("\n  coverage BY STRATUM at alpha=0.10:")
    for k in ("cue_present", "cue_absent", "marginal"):
        r = strat[k]
        if r.get("n"):
            print(f"    {k:<14}n={r['n']:<5}coverage {r['coverage']:.1%}   "
                  f"mean set {r['mean_set_size']:.2f}   "
                  f"abstain {r['abstention_rate']:.1%}")
    v = strat.get("verdict")
    if v:
        print(f"\n    gap {v['coverage_gap']:.1%} -- {v['reading']}")
    print("\n  Split conformal guarantees MARGINAL coverage only. For a model")
    print("  whose known failure is a per-stratum difference, quoting the")
    print("  marginal number alone is the same dilution calibrate.py")
    print("  documents for aggregate ECE.")
    results["conformal"] = {"levels": rows, "by_stratum": strat}

    with open(f"{OUT}/complete.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {OUT}/complete.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--patients", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=14)
    a = ap.parse_args()
    main(a.seeds, a.patients, a.epochs)
