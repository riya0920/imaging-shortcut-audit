"""Calibration, and the reason calibration is not reassurance.

WHY THIS FILE EXISTS
--------------------
The first build reported Brier score and stopped, which is the standard way to
gesture at calibration without doing any. Brier is a proper scoring rule that
mixes discrimination and calibration together, so a model can improve its Brier
by getting better at ranking while its probabilities stay wrong.

WHAT IS MEASURED
----------------
RELIABILITY, by binning predictions and comparing mean predicted to observed
    frequency. Equal-COUNT bins rather than equal-width, because chest-imaging
    predictions pile up near zero and equal-width bins leave the top deciles
    holding four cases each.

ECE, the count-weighted mean gap between the two. Reported with its bin count
    attached, because ECE is not comparable across binning schemes and quoting
    it bare is how it gets misused.

TEMPERATURE SCALING, one parameter per pathology, fitted on a validation split
    by minimising NLL. One parameter because there is not enough validation
    data here for anything richer, and a Platt fit with two parameters can
    silently re-rank in the low-prevalence tail.

THE POINT OF THE FILE
---------------------
Calibration is measured SEPARATELY on the confounded and unconfounded strata --
studies where the shortcut cue is present, and studies where it is absent.

THE MECHANISM IS DILUTION, NOT CANCELLATION, and getting that right matters.
The intuitive story -- "one stratum over-predicts, the other under-predicts,
and the aggregate averages them to zero" -- is FALSE for ECE, which takes an
absolute value inside each bin and so cannot cancel opposite-direction errors.
A test written on the intuitive story failed, correctly, and this paragraph is
the corrected version.

What actually happens is that the aggregate ECE is close to the count-weighted
mean of the strata. A small badly-calibrated stratum is diluted by a large
well-behaved one, and the single reported number sits between them -- lower
than the worst, and therefore reassuring about a subgroup it is not describing.
Reporting only that converts a documented shortcut into a clean bill of health,
which is worse than not calibrating at all: it launders the failure the audit
found.

WHAT THE SEED STUDY DID TO THIS CLAIM
------------------------------------
The first write-up of this file quoted one training run: opacity aggregate ECE
0.127 against a worst stratum of 0.198, plus an "internal consistency check" --
that the effusion head, which the audit could NOT confirm a cue dependency for,
showed strata calibrated within 0.008 of each other.

`calibration_study.py` ran five seeds. Both claims came apart.

    OPACITY   (audit: shortcut CONFIRMED)          EFFUSION  (no dependency)
    seed  ratio  verdict                           seed  ratio  verdict
    0     1.24   no split                          0     1.02   no split
    1     2.83   HIDES                             1     1.09   no split
    2     2.91   HIDES                             2     2.23   HIDES
    3     1.86   HIDES                             3     2.02   HIDES
    4     3.30   HIDES                             4     2.31   HIDES

    aggregate understates the worst stratum:  opacity 4/5, effusion 3/5

Two things follow, and neither is what the single run suggested.

FIRST, the opacity result does not clear this project's own bar. `seed_study.py`
holds findings to 5 of 5, and this is 4 of 5. Reported as not reproducing.

SECOND, and worse for the interpretation: THE SPLIT IS NOT SPECIFIC TO THE
SHORTCUT. Effusion splits in 3 of 5 seeds despite the audit finding no cue
dependency there, and the two ratio ranges ([1.24, 3.30] and [1.02, 2.31])
overlap across most of their length. The 0.008 agreement quoted from one run
was a draw, not a property.

The likely explanation is sample size rather than anything about shortcuts:
these strata hold roughly 120 and 245 studies across 6 bins, so a per-stratum
ECE is a noisy statistic, and the maximum of two noisy statistics exceeds their
pooled value most of the time whether or not a real effect exists. The
temperature is unstable for the same reason -- across seeds it ranges [0.944,
2.044] for opacity, crossing 1.0, so even the DIRECTION of the post-hoc
correction is not stable.

WHAT SURVIVES, STATED AT THE STRENGTH THE EVIDENCE SUPPORTS:
stratifying a calibration report is worth doing, because the aggregate
understates the worst stratum in 7 of 10 pathology-seed combinations here, and
a number that reassures about a subgroup it is not describing is the failure
mode this whole project is about. But at this sample size a stratum split is a
PROMPT TO LOOK, not evidence of a shortcut. It cannot carry the audit's
conclusion, and this file no longer claims that it can.

WHAT THIS IS NOT
----------------
Not isotonic regression (needs more validation data than exists here, and
overfits the tails at this sample size). Not conformal prediction, which is
what a real deployment should use for a coverage guarantee. Not Platt scaling.
Not calibration drift over time -- there is no time axis in this generator.
"""

from __future__ import annotations

import numpy as np


def equal_count_bins(p, n_bins=10):
    """Bin edges holding roughly equal counts.

    Equal-width bins are the default everywhere and are wrong for this data:
    predictions concentrate near zero, so the upper bins hold a handful of
    cases and the reliability curve becomes mostly noise drawn as signal.
    """
    p = np.asarray(p, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p, qs))
    if len(edges) < 2:
        edges = np.array([p.min(), p.min() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def reliability(y, p, n_bins=10):
    """Per-bin predicted vs observed. The table a calibration plot is drawn from."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = equal_count_bins(p, n_bins)
    idx = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "n": int(m.sum()),
                     "predicted": float(p[m].mean()),
                     "observed": float(y[m].mean()),
                     "gap": float(y[m].mean() - p[m].mean())})
    return rows


def ece(y, p, n_bins=10):
    """Expected calibration error, count-weighted.

    Returned with n_bins so it is never quoted bare -- ECE is not comparable
    across binning schemes, and the number alone invites exactly that.
    """
    rows = reliability(y, p, n_bins)
    n = sum(r["n"] for r in rows)
    if not n:
        return {"ece": float("nan"), "n_bins": n_bins, "n": 0}
    value = sum(r["n"] * abs(r["gap"]) for r in rows) / n
    return {"ece": float(value), "n_bins": n_bins, "n": int(n),
            "max_gap": float(max(abs(r["gap"]) for r in rows))}


def brier_decomposition(y, p, n_bins=10):
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    The reason to decompose rather than report Brier alone: a model can lower
    its Brier by getting better at RANKING (resolution) while its probabilities
    stay just as wrong (reliability). Those are different problems with
    different fixes, and the aggregate hides which one is moving.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    base = float(y.mean())
    rows = reliability(y, p, n_bins)
    n = len(y)
    rel = sum(r["n"] * (r["predicted"] - r["observed"]) ** 2 for r in rows) / n
    res = sum(r["n"] * (r["observed"] - base) ** 2 for r in rows) / n
    unc = base * (1 - base)
    return {"brier": float(np.mean((p - y) ** 2)),
            "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc),
            "recomposed": float(rel - res + unc)}


# ---------------------------------------------------------------------------
# temperature scaling
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_temperature(y, logits, lo=0.05, hi=20.0, iters=200):
    """One temperature per pathology, by golden-section search on NLL.

    Golden section rather than gradient descent because the objective is
    one-dimensional and unimodal in T, and a closed search cannot diverge or
    need a learning rate. 200 iterations is far more than needed and costs
    nothing.

    Fitted on logits, NOT on probabilities. Scaling probabilities directly is
    a different and worse transform -- it does not preserve the ranking the
    same way and has no interpretation as a softening of the decision function.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(logits, dtype=float)

    def nll(T):
        p = np.clip(_sigmoid(z / T), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
        if b - a < 1e-6:
            break
    T = (a + b) / 2
    return {"temperature": float(T), "nll_before": nll(1.0),
            "nll_after": nll(T),
            "direction": ("softening overconfident scores" if T > 1.05 else
                          "sharpening underconfident scores" if T < 0.95 else
                          "already calibrated (T is approximately 1)")}


def apply_temperature(logits, T):
    return _sigmoid(np.asarray(logits, dtype=float) / T)


# ---------------------------------------------------------------------------
# the part that matters
# ---------------------------------------------------------------------------

def stratified_calibration(y, p, cue_present, n_bins=6):
    """Calibration on the confounded and unconfounded strata separately.

    THE ARGUMENT FOR THIS FUNCTION. A model that keys on a shortcut cue can be
    well calibrated in aggregate -- the cue appears in deployment at the rate it
    appeared in training, so the average comes out right -- while being badly
    calibrated on one of the strata. An aggregate curve averages the two and
    draws a straight line, which converts a documented shortcut into a clean
    bill of health.

    The function does NOT assume which stratum will be worse, because on this
    model it was the one I did not expect: see the module docstring. Predicting
    the direction and then only checking for it is how a stratified analysis
    becomes a confirmation exercise.

    Fewer bins than the aggregate call, because splitting the sample in two and
    then asking for ten bins of it produces bins of four cases and a reliability
    curve that is mostly sampling noise.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    cue = np.asarray(cue_present, dtype=bool)
    out = {}
    for name, mask in (("cue_present", cue), ("cue_absent", ~cue)):
        if mask.sum() < 2 * n_bins:
            out[name] = {"n": int(mask.sum()),
                         "note": "too few studies to bin honestly"}
            continue
        out[name] = {"n": int(mask.sum()),
                     "prevalence": float(y[mask].mean()),
                     "mean_predicted": float(p[mask].mean()),
                     **ece(y[mask], p[mask], n_bins),
                     "reliability_table": reliability(y[mask], p[mask], n_bins)}
    agg = ece(y, p, n_bins)
    out["aggregate"] = {"n": int(len(y)), **agg}

    both = [out[k] for k in ("cue_present", "cue_absent") if "ece" in out[k]]
    if len(both) == 2:
        worse = max(both, key=lambda d: d["ece"])
        gap = worse["ece"] - agg["ece"]
        # BOTH a ratio and an absolute floor. A ratio alone is unstable when
        # the denominator is near zero: on well-calibrated random data the two
        # strata land at, say, 0.008 and 0.011, the ratio clears 1.25, and the
        # function confidently reports a hidden stratum that is sampling noise.
        # A test caught exactly that.
        hiding = worse["ece"] > 1.25 * agg["ece"] and gap > 0.02
        out["verdict"] = {
            "aggregate_ece": agg["ece"],
            "worst_stratum_ece": worse["ece"],
            "absolute_gap": float(gap),
            "ratio": (worse["ece"] / agg["ece"]) if agg["ece"] else float("nan"),
            "reading": (
                "the aggregate number is optimistic relative to the worst "
                "stratum; calibrating on the mix hides where the model is wrong"
                if hiding else
                "the strata are calibrated similarly; the aggregate is not "
                "hiding a stratum here")}
    return out
