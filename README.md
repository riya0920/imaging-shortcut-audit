# ML-2 — Imaging shortcut-learning audit (~50% build)

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
python -m pytest tests -q                   # 15 tests
```

---

## The five things worth reading

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

### Plus: DICOM in and out, with de-identification that is tested

`src/dicom_io.py` is a hand-rolled Part 10 writer/reader (128-byte preamble,
`DICM` magic, explicit VR little endian) — pydicom is not installed — and a
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

## What is missing (the other 80%)

- **No real data.** No ChestX-ray14, no CheXpert, no radiographs. Every number
  is a property of `src/synth.py`. No comparison to published per-pathology
  AUROC is made, because none would be meaningful.
- **No DenseNet-121, no transfer learning, no modern-architecture comparison.**
- **Seeds vary data and model together.** `seed_study.py` measures total
  run-to-run variability but cannot say how much comes from the data draw versus
  weight initialisation. Separating them is the natural next step and is not
  done.
- **5 seeds is few.** Enough to say a finding reproduces 5/5 or 4/5; not enough
  for a real confidence interval on the effect size.
- **No inference API, no heatmap overlay endpoint, no latency measurement.**
  The spec asks for all three.
- **Lung "segmentation" is the generator's own mask**, not a segmentation
  model. On real images this is the hard part and would need a trained
  segmenter, whose errors would then contaminate the in-lung fraction.
- **Only one mitigation tried**, and it only partly works. Site-shift testing,
  adversarial de-biasing, and retraining on a de-confounded sample are not
  attempted.
- **No calibration work** beyond reporting Brier.
- **DICOM support is a demonstration**, not an implementation: no sequences, no
  compressed transfer syntaxes, no conformance statement.

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
| `tests/test_imaging.py` | 15 tests: split discipline, operating points, de-id |
