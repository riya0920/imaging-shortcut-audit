# ML-2 — Imaging shortcut-learning audit — complete

**Not a clinical model, and not trained on radiographs.** It is an audit
harness with a model attached, built so the audit has ground truth to be
checked against.

Grad-CAM used as a **validation instrument**, not decoration: three confounds
planted at known strengths, three independent methods used to recover them, and
a finding that the three methods disagree — for a reason that matters more than
any of them.

```bash
python train_audit.py                       # patient-level split (~4 min CPU)
python train_audit.py --split image         # the leak demo
python seed_study.py --seeds 5              # multi-seed: which findings reproduce
python write_report.py                      # -> docs/SHORTCUT_AUDIT.md
python serve.py --register                  # fit + calibrate -> models/
python serve.py --demo                      # exercise the API end to end
python serve.py --bench                     # latency percentiles
python calibration_study.py --seeds 5       # does the calibration finding reproduce?
python run_complete.py --seeds 3            # variance, de-confounding, conformal
python -m pytest tests -q                   # 72 tests
python validate_dicom.py                    # interop vs pydicom -> docs/
```

---

## The seven things worth reading

### 1. The audit recovers a planted shortcut it was never told about

Planted: laterality marker ↔ opacity (P=0.85 vs 0.12), portable border ↔
effusion, support device ↔ effusion.

| evidence | finding |
|---|---|
| **spatial** | opacity puts **1.8×** more Grad-CAM mass on the marker box than area predicts; cardiomegaly 0.23×, effusion 0.18× |
| **counterfactual** | marker added to identical pathology-free images: mean P(opacity) **0.404 → 0.539** |
| **causal (occlusion)** | removing the marker at test time costs **−0.015 AUROC** |

The audit isolated the right confound on the right pathology, from behaviour
alone.

### 2. The most useful result is that the three methods disagree

The counterfactual says the marker moves predictions substantially. Occlusion
says it costs almost no AUROC. **Both are correct**, and reconciling them is the
finding:

> AUROC is measured on a test set where the confound holds with the *same*
> correlation as in training. On that distribution the shortcut **works**, so
> the ranking barely moves. A metric computed on the training distribution is
> structurally incapable of detecting shortcut learning.

Which is exactly how radiology AI fails in the field: validated in-house where
portable films really do come from sicker patients, deployed at a site with a
different marker convention or imaging policy, and it collapses without
warning. The counterfactual test catches this before deployment; a held-out
AUROC does not. That is what external validation is *for*.

### 3. Label noise, and the two-column AUROC table

| pathology | AUROC vs shipped labels | AUROC vs true state |
|---|---|---|
| opacity | 0.742 | 0.994 |
| cardiomegaly | 0.788 | 1.000 |
| effusion | 0.819 | 1.000 |

Labels carry 13% simulated report-mining error. The model recovers the
underlying pathology almost perfectly and scores 0.74–0.82 against the labels it
was graded on. **The entire gap is label noise** — the property of NLP-mined
imaging labels that most projects using such datasets never mention.

This is the answer to *"your AUROC beats the published number — why should that
worry you before it pleases you?"* Published baselines were computed against
labels with their own error rate; a higher number can mean a better model,
cleaner labels, a leakier split, or a different protocol, and the AUROC alone
cannot distinguish them.

The two-column table is only possible because the data is synthetic. On a real
dataset the right-hand column does not exist, which is the point: *you cannot
see your own ceiling.*

### 4. Patient-level splits, enforced rather than claimed

Each patient contributes 1–4 studies sharing anatomy and pathology propensity.
`split_indices(..., "patient")` asserts disjointness; two tests check it; and
the leak is *demonstrated* rather than warned about:

| pathology | patient-level split | image-level split | inflation |
|---|---|---|---|
| opacity | 0.742 | 0.831 | **+0.089** |
| cardiomegaly | 0.788 | 0.823 | **+0.035** |
| effusion | 0.819 | 0.786 | −0.033 |

295 patients appeared in both train and test under the image-level split. Two
caveats stated in the report: the runs have different test sets so this is not
a controlled comparison, and run-to-run variance is ~±0.015 — so the effusion
row is noise and the opacity row is not.

### 5. Which findings survive multiple seeds — the gap this README called its worst

The first build reported the audit from **one** training run and said, in this
section, that repeated runs moved opacity AUROC over "roughly 0.73–0.75" and the
counterfactual effect over "roughly +0.14 to +0.27". That range was eyeballed
across two runs and stated qualitatively. It was named as *"the single largest
rigour gap"*, and it was — because a shortcut audit's whole job is to separate a
real finding from an artefact, and **an audit reported from one seed cannot do
that for its own findings.**

`seed_study.py` runs the entire pipeline across N seeds (regenerating the data
*and* retraining each time, so this measures total run-to-run variability, not
just weight initialisation) and reports what reproduces.

**5 seeds, 500 patients, 16 epochs** — a smaller configuration than the
single-run audit above, so the absolute AUROCs are not directly comparable to
it; what transfers is the spread and the verdicts.

| pathology | AUROC (median [min–max]) | vs true state | ECE |
|---|---|---|---|
| opacity | 0.794 [0.746–0.808] | 0.987 [0.979–0.996] | 0.157 |
| cardiomegaly | 0.780 [0.742–0.854] | 1.000 [0.976–1.000] | 0.116 |
| effusion | 0.815 [0.754–0.891] | 1.000 [1.000–1.000] | 0.145 |

The vs-true column stays near 1.00 **on every seed** while the reported AUROC
moves by up to 0.14. The label-noise ceiling is therefore a property of the
labels, not of any particular run — which is itself a finding a single seed
cannot establish.

**Does the shortcut finding reproduce?**

```
counterfactual marker effect on P(opacity)
  median +0.181   range [+0.049, +0.236]   IQR [+0.104, +0.227]
  effect exceeds +0.02 in 5/5 seeds

Grad-CAM activation on the marker box, over area baseline
  pathology       median          range      >1.5x in
  opacity           2.63    [0.94-3.02]           4/5
  cardiomegaly      0.45    [0.06-0.59]           0/5
  effusion          0.04    [0.00-0.07]           0/5
```

**Verdict — all four claims hold:**

| claim | evidence |
|---|---|
| the marker causally shifts P(opacity) | 5/5 seeds above +0.02 |
| opacity attends the marker more than the others do | median 2.63 vs 0.45 and 0.04 |
| occlusion AUROC barely moves (**the audit blind spot**) | largest median \|delta\| 0.0017 |
| the label-noise gap is large and stable | vs-true minimum exceeds reported maximum for every pathology |

Two things the spread changes about how the earlier numbers should be read. The
counterfactual effect ranges **+0.049 to +0.236** — a factor of nearly five — so
quoting a single run's "+0.135" to three decimals was quoting more precision
than the pipeline supports. And opacity's marker attention exceeds 1.5× in **4
of 5** seeds, not 5: the direction is robust, the magnitude is not, and one seed
in five would have understated it.

**A single-seed audit cannot make any of these calls about its own findings.**
That is the whole argument for the harness, and it is why the ranges above —
not the point estimates — are what this pipeline actually supports.

### Plus: DICOM in and out, with de-identification that is tested

`src/dicom_io.py` is a hand-rolled Part 10 writer/reader (128-byte preamble,
`DICM` magic, explicit VR little endian) — written by hand rather than with
pydicom, and now **checked against it** (below) — and a
PS3.15 Annex E **Basic Profile** de-identifier with per-patient date *shifting*
rather than deletion, because intervals carry the clinical meaning.

The tests do the part that matters: `test_deid_survives_a_write_read_cycle`
writes a de-identified file and greps the **raw bytes on disk** for every
planted identifier, and `test_deid_removes_every_tag_in_the_basic_profile_list`
enumerates the removal list instead of spot-checking three tags. Retained tags
each carry a written reason, because being able to say *why* a tag stays is what
separates a de-identifier from a delete key.

Burned-in pixel annotation is **not** handled, and the code says so where it is
called rather than implying coverage.

---

### 6. Calibration, and a negative result that cost the headline

The README's own gap list said "no calibration work beyond reporting Brier".
`src/calibrate.py` adds reliability curves on equal-**count** bins, ECE reported
with its bin count attached, Murphy's Brier decomposition, and per-pathology
temperature scaling fitted on a **third** split so the reported ECE is not a
training metric.

The interesting part was meant to be **stratified** calibration — measuring the
confounded and unconfounded strata separately, on the theory that a shortcut
model can look calibrated in aggregate while being wrong where the cue fires.
One run said exactly that: opacity aggregate ECE 0.127 against a worst stratum
of 0.198, while effusion — where the audit found no dependency — showed strata
agreeing within 0.008. A clean story with an internal consistency check.

`calibration_study.py` ran five seeds and took it apart.

| | opacity (shortcut **confirmed**) | effusion (**no** dependency) |
|---|---|---|
| aggregate understates worst stratum | **4/5 seeds** | **3/5 seeds** |
| ratio, median [range] | 2.83 [1.24, 3.30] | 2.02 [1.02, 2.31] |
| fitted temperature range | [0.944, 2.044] | [0.973, 3.199] |

**The opacity finding does not clear this project's own 5/5 bar**, and worse,
**the split is not specific to the shortcut** — effusion splits in 3 of 5 seeds
with no cue dependency to explain it, and the two ratio ranges overlap across
most of their length. The 0.008 agreement was a draw, not a property.

The likely cause is sample size, not shortcuts: ~120 and ~245 studies across 6
bins make a per-stratum ECE noisy, and the max of two noisy statistics exceeds
their pooled value most of the time whether or not an effect exists. The
temperature range crossing 1.0 says the same thing — even the *direction* of
the post-hoc correction is unstable.

What survives, at the strength the evidence supports: **stratify the report**,
because the aggregate understates the worst stratum in 7 of 10 pathology-seed
combinations and a number that reassures about a subgroup it is not describing
is this project's whole subject. But a stratum split at this sample size is a
prompt to look, **not** evidence of a shortcut, and `src/calibrate.py` no longer
claims otherwise.

This also exposed a reproducibility bug worth naming: `train()` seeds torch
internally, but `SmallCNN(...)` is constructed **before** that call, so weight
initialisation ran from unseeded ambient state. Two runs of an apparently-seeded
pipeline gave different models — opacity's temperature came out 0.620 in one and
1.346 in the next, flipping the direction of the correction. Seeding at
construction time fixed it; two consecutive registrations now match exactly.

### 7. A service that ships the audit with every prediction

`serve.py` — `POST /predict`, `POST /predict/dicom`, `POST /overlay`,
`GET /model`. Three positions:

**Every prediction carries its own audit result.** `src/cues.py` runs the
shortcut detectors per request, and a prediction on an image carrying a cue the
audit showed this model depends on comes back flagged, per pathology. A model
card is a per-*model* statement; the failure is per-*study*. Only the marker
warns — the audit confirmed that dependency and could not confirm the effusion
cues, so warning about all three would dilute the one that is earned.

**De-identification runs before inference, and screens the pixels too.**
`POST /predict/dicom` de-identifies as its first action — score-first-clean-later
leaves identifiers in memory, logs and crash dumps. The sharper point:
`dicom_io.deidentify()` says in its own docstring that it does not handle
burned-in annotation, and **this project's shortcut *is* a burned-in pixel
annotation**. On a real film that same overlay routinely carries a name, MRN or
accession number, so a study can pass tag de-identification completely and still
ship PHI. The cue detector therefore runs twice over — as a shortcut warning and
as a burned-in-annotation screen — and the response says plainly that it is a
weak screen and not clearance.

**The Grad-CAM caveat travels in the response header.** `X-Explanation-Caveat`
on the PNG itself, because an overlay saved out and pasted into a slide deck
loses every surrounding word.

Latency, measured rather than asserted:

| stage | p50 | p95 | p99 |
|---|---|---|---|
| forward pass only | 2.4 ms | 3.9 ms | 4.8 ms |
| forward + Grad-CAM | 8.7 ms | 10.9 ms | 12.2 ms |
| full `/overlay` path | 16.2 ms | 20.4 ms | 30.0 ms |
| `POST /predict` over HTTP | 13.5 ms | 35.2 ms | 37.0 ms |

Grad-CAM costs **3.6× a forward pass** because it needs a backward pass, so the
overlay endpoint is structurally more expensive than the prediction endpoint —
capacity planning, not a detail. And the in-process number is **5.6× cheaper**
than the same work over a socket, which is why it is reported as a floor rather
than an SLO. Most of that gap is JSON: 4,096 floats as decimal text is a far
larger payload than the tensor it becomes.


## Bugs this harness caught

- **`argmin(|spec − 0.90|)` reported 0% sensitivity at 90% specificity for a
  model with AUROC 1.000.** A near-perfect ROC has a point at the trivial corner
  (spec 1.0, tpr 0.0), which is 0.10 from the target and wins the argmin. Fixed
  to take the best point *meeting* the constraint; regression test included.
- **The first generator produced three AUROCs of 1.000 and an audit with
  nothing to find.** The pathologies were trivially separable, so the model had
  no incentive to use any confound. Shortcut learning requires the real feature
  to be *hard*; adding label noise is what created the incentive — and it is
  also the honest property of the dataset the spec asks about.
- **The CAM audit silently dropped pathologies** whose predictions never
  exceeded 0.5, so the table showed one row instead of three.

## Where the run-to-run spread actually comes from

`seed_study.py` moved data and weights together, so its spread was a sum of two
sources it could not attribute. Varying one at a time separates them:

| varying | opacity AUROC | sd |
|---|---|---|
| the **data** draw (init fixed) | 0.716, 0.781, 0.777 | 0.037 |
| the **weight init** (data fixed) | 0.716, **0.558**, 0.735 | **0.097** |

**Weight initialisation dominates by 2.6×**, and one init landed at 0.558 —
near chance, an optimisation failure rather than a data problem. That settles
which lever to pull: averaging over inits is the cheap fix, and a larger dataset
would not help much. It also explains why the § 6 calibration finding was so
unstable across seeds.

## Retraining on a de-confounded sample — and an upper bound, not a finding

`strip_marker()` deletes the confound from the training distribution, which is
the cleanest form of the mitigation and the one that says what the ceiling is:
whatever remains is what the model reads from **anatomy**.

```
opacity AUROC with the marker present : 0.7355  [0.716, 0.687, 0.804]
opacity AUROC with the marker REMOVED : 0.7316  [0.671, 0.750, 0.774]
difference: +0.0039

paired sd 0.0585 over 3 seeds -> smallest resolvable difference 0.0939
OBSERVED 0.0039. THE EXPERIMENT IS UNDERPOWERED.
```

**The power is computed before the number is interpreted**, which is the
discipline `ml1-readmission-risk` had to apply after retracting a model
comparison for exactly this reason. What survives is an **upper bound**:
whatever the marker was worth to a *retrained* model, it was worth less than
0.094 AUROC here.

That bound reframes the audit rather than contradicting it. The counterfactual
showed the model **uses** the marker when it is present (+0.181 on the score);
retraining without it costs little. Both are true, and together they say **the
shortcut was available, not necessary** — the model reads nearly as much from
anatomy and took the marker because it was the cheaper route.

Which is the best possible news for the mitigation: removing the confound is
close to free. It is also exactly the claim that needs more seeds before anyone
acts on it — and § 1 says more seeds are the right lever, since weight
initialisation is what moves this number.

Available **only because the data is synthetic.** A real annotation cannot be
un-burned without inpainting, and inpainting leaves an artefact the model may
key on instead — trading a known confound for an unknown one.

## Conformal prediction — a guarantee that does not need calibration

`src/conformal.py`. The § 6 calibration finding did not reproduce, and adding a
sixth seed does not fix an estimator that is noisy by construction. Conformal
answers a different question: instead of *"is this probability right"*, it gives
a **set** with `P(true label in set) >= 1 - alpha`, distribution-free and
finite-sample.

| alpha | target | coverage | set size | singleton | abstain |
|---|---|---|---|---|---|
| 0.20 | 80% | 80.7% | 1.03 | 96.8% | 3.2% |
| 0.10 | 90% | 89.5% | 1.36 | 64.1% | 35.9% |
| 0.05 | 95% | 95.4% | 1.67 | 33.5% | 66.5% |

A badly miscalibrated model **does not lose coverage — it pays in wider sets**,
which is exactly what a model with a known shortcut needs, since the one thing
it lacks is a trustworthy probability. So the **abstention rate** becomes the
honest measure of what this model knows, and it is a measure ECE could not give.

Coverage is the guarantee; **set size is the information**. A model that always
returns both labels has perfect coverage and has said nothing, so the two are
read together or not at all.

### And the marginal guarantee hides a stratum

```
cue_present   n=135  coverage 84.4%   abstain 28.9%
cue_absent    n=274  coverage 92.0%   abstain 39.4%
marginal      n=409  coverage 89.5%   abstain 35.9%
```

The nominal 90% is met **by averaging over a stratum where it fails at 84.4%**.
Split conformal guarantees *marginal* coverage only, and for a model whose known
failure mode is a per-stratum difference, quoting the marginal number alone is
the same dilution `calibrate.py` documents for aggregate ECE.

## The DICOM code is checked against `pydicom`, in both directions

The gap list used to excuse the hand-rolled Part 10 code by saying pydicom was
"not installed". That was wrong, and the excuse hid the real weakness: **a
writer tested only by its own reader proves almost nothing.** Two consistent
misreadings of the standard agree with each other perfectly, and the bugs live
exactly where the encoder and decoder share an assumption.

| direction | result |
|---|---|
| `pydicom` reads a file **this** wrote | passes — transfer syntax `1.2.840.10008.1.2.1`, pixel array identical |
| **this** reads a file `pydicom` wrote | passes — 8,794 bytes, pixel array identical |

The second is the one that matters. A parser exercised only against its own
writer never meets a construct that writer does not emit.

### De-identification, re-checked by an independent parser

| check | result |
|---|---|
| tags carrying a non-empty value | **none** |
| original PHI values anywhere in the parsed dataset | **none** |
| `StudyDate` shifted rather than removed | yes |
| interval between two studies preserved | yes — 74 days before, 74 after |

The existing byte-grep test stays: it is the stronger check for *is this string
gone*. This adds the stronger check for *is this tag carrying anything* — a
value can survive re-encoded, which a grep for the original bytes would miss.

### And it found one thing worth fixing

`deidentify()` applies a **single action** to everything in
`BASIC_PROFILE_REMOVE`: set the value to zero length. PS3.15 Annex E assigns
actions **per tag**, and the set includes both **X** (remove the tag entirely)
and **Z** (retain it with a zero-length value).

Uniform Z is **privacy-equivalent for the value** — which is why nothing leaks
— but it is **not exactly conformant**, because a tag whose assigned action is
X should be *absent*, and a conformance checker would flag it as present. The
constant is also named `..._REMOVE` while implementing retain-and-zero, which
is the kind of naming that makes a reviewer believe the wrong thing.

**This is deliberately not fixed here.** Fixing it means asserting the correct
per-tag action for seventeen tags, and the authority for that is the PS3.15
Annex E attribute table, which is not available offline. Guessing from memory
and calling the result "conformant" is exactly the unearned claim this project
exists to avoid. It is recorded as a specific, closeable gap instead.

## What is still missing, and why it cannot be closed here

- **No real data.** No ChestX-ray14, no CheXpert, no radiographs — not
  installed, no network. Every number is a property of `src/synth.py`, and no
  comparison to published per-pathology AUROC is made because none would be
  meaningful.
- **No DenseNet-121, no transfer learning.** Pretrained weights need a
  download. The architecture comparison the spec suggests is therefore absent
  rather than approximated.
- **The strata are too small to settle the calibration question.** ~135 and
  ~274 studies. Resolving whether a stratum split tracks a shortcut needs a much
  larger test set or a paired design across many more seeds.
- **The de-confounding result is an upper bound, not a measurement.** Three
  seeds resolve differences above 0.094 AUROC and the observed difference is
  0.004. More seeds would close this and were not run.
- **The cue detectors only work because this project planted the cues.** Three
  thresholds against known geometry. On real radiographs annotations move, vary
  in font and rotation, and support devices are diagnostically real rather than
  artefacts — detecting them is its own vision problem, plausibly harder than
  the classification task.
- **The burned-in-annotation screen is not OCR** and must not be treated as
  de-identification clearance.
- **Lung "segmentation" is the generator's own mask**, not a segmentation
  model. On real images this is the hard part, and a trained segmenter's errors
  would then contaminate the in-lung fraction.
- **Conformal is split-conformal only** — no cross-conformal, no jackknife+, no
  CQR, no Mondrian (class-conditional) variant, and no conditional coverage.
  The guarantee is marginal, which is a much weaker statement than most readers
  assume, and the stratified check above is there precisely because of it.
- **The service is a demonstration.** Single-process stdlib `HTTPServer`, no
  auth, no TLS, no batching, no GPU, no ONNX. It moves pixels as JSON, which is
  the wrong transport and is most of the measured latency.
- **The latency numbers are a floor.** 64x64 images on a CPU at concurrency 1;
  a 2048x2048 chest film is ~1000x the pixels.

## Files

| path | what |
|---|---|
| `src/synth.py` | image generator, planted confounds, label noise |
| `src/model.py` | small CNN, hand-rolled Grad-CAM |
| `src/dicom_io.py` | Part 10 writer/reader, PS3.15 Annex E de-identifier |
| `train_audit.py` | training, operating points, three-method audit, mitigation |
| `seed_study.py` | multi-seed harness: which findings reproduce, and which were one run |
| `write_report.py` | JSON → `docs/SHORTCUT_AUDIT.md` |
| `docs/MODEL_CARD.md` | intended use, shortcut reliance, shift warnings |
| `src/calibrate.py` | reliability, ECE, Brier decomposition, temperature, stratified |
| `src/cues.py` | shortcut-cue detectors; also the burned-in-annotation screen |
| `serve.py` | inference API, overlay endpoint, `--register`, `--bench` |
| `calibration_study.py` | 5 seeds: the study that falsified the § 6 headline |
| `src/conformal.py` | split conformal, and the stratified coverage check |
| `run_complete.py` | variance decomposition, de-confounded retrain, conformal |
| `validate_dicom.py` | pydicom interop both directions; found the uniform-Z gap |
| `tests/test_dicom_interop.py` | 7 tests, incl. reading a file pydicom wrote |
| `tests/test_conformal.py` | 16 tests: coverage against planted truth, the hidden stratum |
| `tests/test_serving.py` | 34 tests: calibration against planted truth, cues, the API |
| `tests/test_imaging.py` | 15 tests: split discipline, operating points, de-id |
