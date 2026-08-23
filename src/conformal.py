"""Conformal prediction: a coverage guarantee that does not need calibration.

WHY THIS AND NOT MORE CALIBRATION
----------------------------------
`calibration_study.py` ended with a negative result: the stratified calibration
finding held in only 4 of 5 seeds, and appeared just as often for the pathology
with no shortcut. Per-stratum ECE at n≈120 is too noisy to settle the question,
and adding a sixth seed does not fix an estimator that is noisy by construction.

Conformal prediction answers a different and more useful question. Instead of
"is this probability right", it gives a SET of labels with a guarantee:

    P(true label in the predicted set) >= 1 - alpha

The guarantee is DISTRIBUTION-FREE and FINITE-SAMPLE. It needs no assumption
that the model is calibrated, well-specified, or even good -- only that the
calibration and test data are exchangeable. A badly miscalibrated model still
gets valid coverage; it just pays for it with wider sets. That property is
exactly what a model with a known shortcut needs, because the one thing it does
not have is a trustworthy probability.

WHAT THE SET WIDTH MEANS HERE
------------------------------
For a binary head the prediction set is one of four things, and the useless one
is informative:

    {positive}          confident it is present
    {negative}          confident it is absent
    {positive,negative} THE MODEL DOES NOT KNOW. Not a probability near 0.5 --
                        an explicit abstention with a coverage guarantee behind
                        it.
    {}                  the empty set: NEITHER label is plausible at this
                        confidence. It means this image looks unlike anything
                        in the calibration set, which for a shortcut model is
                        the most interesting output it can produce.

A miscalibrated model does not lose coverage. It produces more two-label sets.
So the ABSTENTION RATE becomes the honest measure of how much this model
actually knows, and it is a measure the ECE could not give.

THE ASSUMPTION THAT CAN BREAK, AND DOES
----------------------------------------
Exchangeability. Split-conformal guarantees coverage when calibration and test
data are drawn from the same distribution. A shortcut model deployed at a site
whose marker convention differs violates that immediately, and the guarantee is
void -- not degraded, void. `coverage_by_stratum()` measures coverage
separately on cue-present and cue-absent studies for exactly that reason: if
those two differ, the marginal guarantee is being met by averaging over a
subgroup it fails on, which is the same dilution `calibrate.py` documents for
ECE.

WHAT THIS IS NOT
----------------
Split conformal only. No cross-conformal, no jackknife+, no CQR, no Mondrian
(class-conditional) conformal, and no adaptive prediction sets. No conditional
coverage of any kind -- the guarantee is marginal, which is a much weaker
statement than most readers assume.
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(y_cal, p_cal, alpha=0.1):
    """Split-conformal threshold on the nonconformity score s = 1 - p_true.

    The quantile index uses the finite-sample correction ceil((n+1)(1-alpha))/n
    rather than plain (1-alpha). Without it, coverage is only asymptotic, and at
    the sample sizes here that is the difference between a guarantee and a
    hope.
    """
    y_cal = np.asarray(y_cal, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    # nonconformity of the TRUE label
    scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    n = len(scores)
    if n == 0:
        return {"qhat": 1.0, "n": 0, "alpha": alpha,
                "note": "no calibration data; every set is both labels"}
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        # Cannot achieve this alpha with this n. Say so rather than clamping
        # and reporting a guarantee that does not hold.
        return {"qhat": 1.0, "n": n, "alpha": alpha, "achievable": False,
                "note": (f"alpha={alpha} needs at least {int(np.ceil(1/alpha))-1} "
                         f"calibration points; with n={n} the smallest "
                         f"achievable alpha is {1/(n+1):.3f}")}
    return {"qhat": float(np.quantile(scores, level, method="higher")),
            "n": n, "alpha": alpha, "achievable": True,
            "effective_level": float(level)}


def predict_set(p, qhat):
    """The conformal set for one score. Returns a tuple of admitted labels."""
    labels = []
    if 1.0 - p <= qhat:
        labels.append(1)
    if p <= qhat:
        labels.append(0)
    return tuple(labels)


def predict_sets(p, qhat):
    return [predict_set(float(x), qhat) for x in np.asarray(p, dtype=float)]


def evaluate(y, p, qhat):
    """Coverage and set-size distribution.

    Coverage is the guarantee; SET SIZE is the information. A model that always
    returns both labels has perfect coverage and has said nothing, so the two
    have to be read together and never separately.
    """
    y = np.asarray(y, dtype=int)
    sets = predict_sets(p, qhat)
    covered = np.array([int(t) in s for t, s in zip(y, sets)])
    sizes = np.array([len(s) for s in sets])
    return {
        "n": int(len(y)),
        "coverage": float(covered.mean()) if len(y) else float("nan"),
        "mean_set_size": float(sizes.mean()) if len(y) else float("nan"),
        "singleton_rate": float((sizes == 1).mean()) if len(y) else float("nan"),
        "abstention_rate": float((sizes == 2).mean()) if len(y) else float("nan"),
        "empty_rate": float((sizes == 0).mean()) if len(y) else float("nan"),
    }


def coverage_by_stratum(y, p, qhat, cue_present):
    """Coverage on cue-present and cue-absent studies separately.

    THE CHECK THAT MATTERS FOR A SHORTCUT MODEL. Split conformal guarantees
    MARGINAL coverage -- averaged over the whole distribution. It says nothing
    about any subgroup, and a marginal 90% can be 97% on one stratum and 78% on
    another. For a model whose known failure mode is exactly a per-stratum
    difference, reporting only the marginal number would be the same dilution
    `calibrate.py` documents for aggregate ECE.
    """
    y = np.asarray(y, dtype=int)
    cue = np.asarray(cue_present, dtype=bool)
    out = {}
    for name, mask in (("cue_present", cue), ("cue_absent", ~cue)):
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        out[name] = evaluate(y[mask], np.asarray(p)[mask], qhat)
    out["marginal"] = evaluate(y, p, qhat)
    both = [out[k] for k in ("cue_present", "cue_absent") if out[k].get("n")]
    if len(both) == 2:
        gap = abs(both[0]["coverage"] - both[1]["coverage"])
        out["verdict"] = {
            "coverage_gap": gap,
            "reading": (
                "the marginal guarantee is being met by averaging over a "
                "stratum it fails on; conditional coverage does not hold and "
                "the marginal number should not be quoted alone"
                if gap > 0.05 else
                "coverage is similar across strata, so the marginal guarantee "
                "is not hiding a subgroup here")}
    return out
