# What it would cost to resolve the de-confounding result

The gap list used to say:

> The de-confounding result is an upper bound, not a measurement. Three seeds
> resolve differences above 0.094 AUROC and the observed difference is 0.004.
> **More seeds would close this and were not run.**

That last sentence is a claim about **feasibility**, and nobody had checked it.
It reads like "we could not be bothered". Here is what it would actually take.

## The arithmetic

The study's own minimum detectable difference is `MDD = 2.78 * sd / sqrt(n)`,
so resolving an effect of size *d* needs `n = (2.78 * sd / d)^2`.

| | |
|---|---|
| measured paired sd | **0.0585** |
| observed difference | **0.0039** |
| seeds run | 3 |
| seconds per fit (two per seed) | 175 |

| seeds | MDD | compute | resolves? |
|---|---|---|---|
| 3 | 0.0939 | 0.3 h | no |
| 10 | 0.0514 | 1.0 h | no |
| 25 | 0.0325 | 2.4 h | no |
| 50 | 0.0230 | 4.9 h | no |
| 100 | 0.0163 | 9.7 h | no |
| 250 | 0.0103 | 24.3 h | no |
| 1000 | 0.0051 | 97.3 h | no |

**Seeds needed to resolve 0.0039: about 1751 — roughly 7 days of compute on
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
  data seed *and* the init seed, and a residual paired sd of 0.0585 against an
  unpaired init sd of 0.097 is what that bought.

## What does not change

The conclusion stands. The upper bound is informative, and combined with the
+0.181 counterfactual it still says **the shortcut was available, not
necessary** — the model reads nearly as much from anatomy and took the marker
because it was the cheaper route.

What changes is that the gap list no longer implies a cheap fix is sitting
there untaken. This is a real limit of running the experiment on one CPU, and
it is now stated as one rather than deferred.
