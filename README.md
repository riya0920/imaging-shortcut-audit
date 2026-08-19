# ML-2 — Imaging shortcut-learning audit (first 20%)

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
python write_report.py                      # -> docs/SHORTCUT_AUDIT.md
python -m pytest tests -q                   # 15 tests
```

---

## The four things worth reading

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
- **No multi-seed harness.** Single-run point estimates with variance stated
  qualitatively; the right build reports intervals over seeds. This is the
  single largest rigour gap.
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
| `write_report.py` | JSON → `docs/SHORTCUT_AUDIT.md` |
| `docs/MODEL_CARD.md` | intended use, shortcut reliance, shift warnings |
| `tests/test_imaging.py` | 15 tests: split discipline, operating points, de-id |
