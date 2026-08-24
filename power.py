"""What would it actually cost to resolve the de-confounding result?

THE CLAIM THIS EXISTS TO CHECK
-------------------------------
The gap list said, for a long time:

    "The de-confounding result is an upper bound, not a measurement. Three
     seeds resolve differences above 0.094 AUROC and the observed difference
     is 0.004. More seeds would close this and were not run."

The last sentence is the interesting one, because it is a claim about
FEASIBILITY that nobody had checked. "More seeds would close this" reads like
"we could not be bothered". This module works out what it would take.

THE ARITHMETIC
--------------
The study's own minimum detectable difference is

    MDD = k * sd_paired / sqrt(n)

so resolving an effect of size `d` needs

    n = (k * sd_paired / d)^2

With the measured `sd_paired = 0.0585` and the observed `d = 0.0039`, that is
roughly **1,750 seeds** -- each of which trains TWO models, at several minutes
apiece.

So the honest statement is not "more seeds would close this". It is:

    Closing it is a multi-WEEK compute job on this machine. More seeds tighten
    the bound; they do not close the question, and no realistic number of them
    turns this into a measurement.

WHY THE OBVIOUS SHORTCUTS DO NOT WORK
--------------------------------------
* A bigger TEST set would cut test-sampling noise -- but section 1 of the study
  already established that WEIGHT INITIALISATION, not the data draw, dominates
  the spread (sd 0.097 vs 0.037). The noise is in the weights, not the test
  split.
* Averaging several inits per condition reduces that noise by sqrt(k), and
  costs exactly k times more compute. It is the same trade in a different
  shape, not a way out of it.
* The paired design already removes what pairing can remove: both arms share
  the data seed AND the init seed, and the residual sd of 0.0585 against an
  unpaired init sd of 0.097 is what that bought.

WHAT THAT MEANS FOR THE AUDIT
------------------------------
Nothing about the conclusion changes -- the upper bound stands and is
informative. What changes is that the gap list no longer implies a cheap fix is
sitting there untaken. It is a genuine limit of running this on one CPU, and it
is now stated as one.

Run:  python power.py [--seconds-per-fit 515]
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(ROOT, "out", "complete.json")

# The multiplier the study uses in its own MDD formula. Kept identical on
# purpose: a power calculation that quietly uses a different constant from the
# experiment it describes is not describing that experiment.
K = 2.78


def seeds_needed(effect, sd_paired, k=K):
    """Seeds required for the MDD to fall to `effect`."""
    if effect <= 0:
        return float("inf")
    return (k * sd_paired / effect) ** 2


def mdd_at(n, sd_paired, k=K):
    return k * sd_paired / (n ** 0.5)


def load_measured(path=RESULT):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("deconfound")


def report(sd_paired, effect, n_current, seconds_per_fit):
    """Rows for a range of seed counts, plus the number that would resolve."""
    rows = []
    for n in (n_current, 10, 25, 50, 100, 250, 1000):
        if n < n_current:
            continue
        hours = n * 2 * seconds_per_fit / 3600.0
        rows.append({"seeds": n, "mdd": mdd_at(n, sd_paired),
                     "hours": hours,
                     "resolves": mdd_at(n, sd_paired) <= effect})
    need = seeds_needed(effect, sd_paired)
    return rows, need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds-per-fit", type=float, default=515.0,
                    help="measured wall time for one fit at the study's "
                         "settings (500 patients, 14 epochs)")
    args = ap.parse_args()

    measured = load_measured()
    if measured:
        sd = float(measured["paired_sd"])
        effect = abs(float(measured["drop"]))
        n_current = len(measured["with_marker"])
    else:
        sd, effect, n_current = 0.0585, 0.0039, 3
        print("(no out/complete.json -- using the last recorded values)")

    rows, need = report(sd, effect, n_current, args.seconds_per_fit)

    print("=" * 72)
    print("  measured paired sd       %.4f" % sd)
    print("  observed difference      %.4f" % effect)
    print("  seeds run                %d" % n_current)
    print("  seconds per fit          %.0f  (two fits per seed)" %
          args.seconds_per_fit)
    print("=" * 72)
    print("  %8s %10s %14s %12s" % ("seeds", "MDD", "compute", "resolves?"))
    for row in rows:
        print("  %8d %10.4f %11.1f h %12s"
              % (row["seeds"], row["mdd"], row["hours"],
                 "yes" if row["resolves"] else "no"))
    print("=" * 72)
    days = need * 2 * args.seconds_per_fit / 86400.0
    print("  seeds needed to resolve %.4f : %.0f" % (effect, need))
    print("  which is %.1f DAYS of compute on this machine." % days)
    print()
    print("  So \"more seeds would close this\" was wrong. More seeds tighten")
    print("  the bound; no realistic number turns it into a measurement.")

    doc = os.path.join(ROOT, "docs")
    os.makedirs(doc, exist_ok=True)
    path = os.path.join(doc, "DECONFOUND_POWER.md")
    table = "\n".join(
        "| %d | %.4f | %.1f h | %s |"
        % (r["seeds"], r["mdd"], r["hours"], "yes" if r["resolves"] else "no")
        for r in rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("""# What it would cost to resolve the de-confounding result

The gap list used to say:

> The de-confounding result is an upper bound, not a measurement. Three seeds
> resolve differences above 0.094 AUROC and the observed difference is 0.004.
> **More seeds would close this and were not run.**

That last sentence is a claim about **feasibility**, and nobody had checked it.
It reads like "we could not be bothered". Here is what it would actually take.

## The arithmetic

The study's own minimum detectable difference is `MDD = %.2f * sd / sqrt(n)`,
so resolving an effect of size *d* needs `n = (%.2f * sd / d)^2`.

| | |
|---|---|
| measured paired sd | **%.4f** |
| observed difference | **%.4f** |
| seeds run | %d |
| seconds per fit (two per seed) | %.0f |

| seeds | MDD | compute | resolves? |
|---|---|---|---|
%s

**Seeds needed to resolve %.4f: about %.0f — roughly %.0f days of compute on
this machine.**

## So the claim was wrong

Not "more seeds would close this". The honest version:

> More seeds **tighten the bound**. No realistic number of them turns this into
> a measurement, because the effect is more than an order of magnitude below
> the run-to-run noise.

## Why the obvious shortcuts do not work

- **A bigger test set** would cut test-sampling noise — but section 1 of the
  study already established that **weight initialisation dominates the data
  draw** (sd 0.097 vs 0.037). The noise is in the weights, not the split.
- **Averaging k inits per condition** reduces that noise by `sqrt(k)` and costs
  exactly `k` times more compute. Same trade, different shape.
- **The paired design already did what pairing can do**: both arms share the
  data seed *and* the init seed, and a residual paired sd of %.4f against an
  unpaired init sd of 0.097 is what that bought.

## What does not change

The conclusion stands. The upper bound is informative, and combined with the
+0.181 counterfactual it still says **the shortcut was available, not
necessary** — the model reads nearly as much from anatomy and took the marker
because it was the cheaper route.

What changes is that the gap list no longer implies a cheap fix is sitting
there untaken. This is a real limit of running the experiment on one CPU, and
it is now stated as one rather than deferred.
""" % (K, K, sd, effect, n_current, args.seconds_per_fit, table,
       effect, need, days, sd))
    print()
    print("wrote", path)


if __name__ == "__main__":
    main()
