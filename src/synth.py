"""Synthetic chest-radiograph-like images with PLANTED confounds.

WHY SYNTHETIC, AND WHAT IT BUYS
------------------------------
The spec asks for NIH ChestX-ray14 or CheXpert. Neither is downloadable in this
offline build. Substituting a generator is a real loss -- no genuine anatomy, no
real label noise, no scanner variation, and any AUROC below is a property of
this file rather than of radiology.

But it buys the one thing the shortcut audit actually needs and that no public
dataset provides: **ground truth about the confound**. On ChestX-ray14 you can
show a heatmap sitting on a laterality marker and argue about it. Here the
marker's pixel coordinates, its exact correlation with the label, and the true
lung-field mask are all known, so "what fraction of positive activation falls
inside the lungs" is a measured quantity with a correct answer rather than an
impression. The audit is therefore verifiable in a way it could not be on real
data, which is the trade being made.

THE PLANTED CONFOUNDS
---------------------
Each mirrors a documented real-world failure:

  laterality marker  -- a bright glyph in the image corner, correlated with
                        OPACITY. Models learning acquisition metadata instead
                        of anatomy is the classic finding.
  portable border    -- a bright frame, marking portable/bedside acquisition,
                        correlated with EFFUSION. Portable films come from
                        sicker patients, so the view itself predicts the label.
  support device     -- a thin bright line across a lung, correlated with
                        EFFUSION. This is the pneumothorax/chest-drain story:
                        the model finds the treatment, not the disease.

Correlations are set per-confound and recorded in CONFOUND_STRENGTH so the
audit can compare what the model learned against what was planted.

PATIENT STRUCTURE
-----------------
Each patient contributes 1-4 studies sharing anatomy and pathology tendency.
This is what makes an image-level split leak, and the leak is demonstrated
rather than asserted (see `train_audit.py --split image`).
"""

from __future__ import annotations

import numpy as np

IMG = 64
PATHOLOGIES = ["opacity", "cardiomegaly", "effusion"]

# P(confound present | pathology present), P(confound present | absent).
# Confounds are drawn from the TRUE label, because in reality the acquisition
# circumstance is caused by the patient's actual condition, not by what the
# report-mining pipeline recorded about it.
CONFOUND_STRENGTH = {
    "marker":  ("opacity",     0.85, 0.12),
    "border":  ("effusion",    0.78, 0.20),
    "device":  ("effusion",    0.72, 0.08),
}

# LABEL NOISE -- the property of ChestX-ray14 that most projects using it do
# not mention. Its labels were mined from free-text reports by an NLP pipeline
# with published, non-trivial error rates, so the ceiling on any model trained
# against them is set by the labels, not the images.
#
# It is also what makes the shortcut audit meaningful. If the rendered
# pathology were a perfect predictor of the shipped label, a CNN would learn it
# and ignore every confound -- which is what the first version of this
# generator did, producing three AUROCs of 1.000 and an audit with nothing to
# find. With noisy labels the image is an imperfect view of the truth, the
# confound is a SECOND imperfect view, and combining them genuinely beats the
# image alone. That is exactly the incentive structure that produces shortcut
# learning in the wild.
LABEL_NOISE = 0.13

# where each confound lives, so the audit can measure activation inside it
MARKER_BOX = (3, 11, IMG - 12, IMG - 4)     # (r0, r1, c0, c1) top-right glyph
BORDER_WIDTH = 2


def _ellipse(shape, cr, cc, rr, cc_rad, angle=0.0):
    r, c = np.ogrid[: shape[0], : shape[1]]
    dr, dc = r - cr, c - cc
    if angle:
        ca, sa = np.cos(angle), np.sin(angle)
        dr, dc = dr * ca + dc * sa, -dr * sa + dc * ca
    return (dr / rr) ** 2 + (dc / cc_rad) ** 2 <= 1.0


def make_study(rng, patient_anatomy, labels):
    """Render one study. Returns (image, lung_mask, confounds_present)."""
    a = patient_anatomy
    img = rng.normal(0.10, 0.02, (IMG, IMG)).astype(np.float32)

    # thorax soft tissue
    body = _ellipse((IMG, IMG), IMG * 0.52, IMG * 0.5, IMG * 0.46, IMG * 0.40)
    img[body] += 0.34

    # lung fields (air = dark). These masks are the ground truth the audit uses.
    lung_l = _ellipse((IMG, IMG), a["lung_r"], a["lung_c_left"],
                      a["lung_rr"], a["lung_cc"], -0.12)
    lung_r = _ellipse((IMG, IMG), a["lung_r"], a["lung_c_right"],
                      a["lung_rr"], a["lung_cc"], 0.12)
    lungs = lung_l | lung_r
    img[lungs] -= 0.26

    # heart; cardiomegaly widens it
    heart_w = a["heart_w"] * (1.26 if labels["cardiomegaly"] else 1.0)
    heart = _ellipse((IMG, IMG), IMG * 0.60, IMG * 0.455, IMG * 0.15, heart_w)
    img[heart] += 0.30

    # ribs
    rows = np.arange(IMG).reshape(-1, 1)
    img[lungs] += (0.035 * np.sin(rows * 0.9 + a["rib_phase"])
                   * np.ones((1, IMG)))[lungs]

    # ---- pathologies -----------------------------------------------------
    if labels["opacity"]:
        side = lung_l if a["opacity_side"] == 0 else lung_r
        cr = a["lung_r"] + rng.integers(-6, 7)
        cc = (a["lung_c_left"] if a["opacity_side"] == 0
              else a["lung_c_right"]) + rng.integers(-4, 5)
        blob = _ellipse((IMG, IMG), cr, cc, rng.integers(4, 8), rng.integers(4, 8))
        img[blob & side] += 0.15

    if labels["effusion"]:
        # blunting of the costophrenic angle: bright wedge at a lung base
        base = np.zeros((IMG, IMG), bool)
        r0 = int(a["lung_r"] + a["lung_rr"] * 0.55)
        base[r0:r0 + 9, :] = True
        img[base & lungs] += 0.13

    # ---- confounds -------------------------------------------------------
    present = {}
    for name, (target, p_pos, p_neg) in CONFOUND_STRENGTH.items():
        p = p_pos if labels[target] else p_neg
        present[name] = bool(rng.random() < p)

    if present["marker"]:
        r0, r1, c0, c1 = MARKER_BOX
        img[r0:r1, c0:c1] += 0.05
        img[r0:r1, c0:c0 + 2] += 0.55           # stem of an "R"
        img[r0:r0 + 2, c0:c1 - 1] += 0.55       # top bar
        img[r0 + 3:r0 + 5, c0:c1 - 1] += 0.55   # middle bar
        img[r0 + 5:r1, c1 - 3:c1 - 1] += 0.55   # leg

    if present["border"]:
        w = BORDER_WIDTH
        img[:w, :] += 0.45
        img[-w:, :] += 0.45
        img[:, :w] += 0.45
        img[:, -w:] += 0.45

    if present["device"]:
        for t in np.linspace(0, 1, 90):
            r = int(a["lung_r"] - a["lung_rr"] * 0.7 + t * a["lung_rr"] * 1.5)
            c = int(a["lung_c_right"] + 6 - t * 10)
            if 0 <= r < IMG and 0 <= c < IMG:
                img[r, max(0, c - 1):c + 1] += 0.50

    # Noise is set so the pathology signals sit near, not far above, it. That
    # is deliberate: shortcut learning happens when the real feature is HARD
    # and the confound is easy. A generator whose pathologies are trivially
    # separable produces a model with no incentive to cheat, and an audit with
    # nothing to find.
    img += rng.normal(0, 0.030, (IMG, IMG))
    return np.clip(img, 0, 1).astype(np.float32), lungs, present


def make_patient(rng, pid):
    """Anatomy and pathology tendency are patient-level, studies are not."""
    anatomy = {
        "lung_r": IMG * 0.47 + rng.integers(-2, 3),
        "lung_c_left": IMG * 0.31 + rng.integers(-2, 3),
        "lung_c_right": IMG * 0.69 + rng.integers(-2, 3),
        "lung_rr": IMG * 0.235 + rng.integers(-2, 3),
        "lung_cc": IMG * 0.145 + rng.integers(-1, 2),
        "heart_w": IMG * 0.11,
        "rib_phase": float(rng.random() * 6.28),
        "opacity_side": int(rng.integers(0, 2)),
    }
    # patient-level propensity: this is what leaks across an image-level split
    prop = {p: float(np.clip(rng.beta(1.2, 4.5), 0.02, 0.95)) for p in PATHOLOGIES}
    n_studies = int(rng.integers(1, 5))

    studies = []
    for k in range(n_studies):
        # true state of the patient: drives the image AND the confounds
        truth = {p: bool(rng.random() < prop[p]) for p in PATHOLOGIES}
        img, lungs, conf = make_study(rng, anatomy, truth)
        # what the report-mining pipeline recorded, which is what ships
        observed = {p: (not v if rng.random() < LABEL_NOISE else v)
                    for p, v in truth.items()}
        studies.append({
            "patient_id": f"P{pid:05d}", "study_id": f"P{pid:05d}_S{k}",
            "image": img, "lung_mask": lungs, "labels": observed,
            "labels_true": truth, "confounds": conf,
        })
    return studies


def build_dataset(n_patients=900, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_patients):
        out.extend(make_patient(rng, i))
    return out


def as_arrays(studies):
    X = np.stack([s["image"] for s in studies])[:, None, :, :]
    Y = np.array([[float(s["labels"][p]) for p in PATHOLOGIES] for s in studies],
                 dtype=np.float32)
    Ytrue = np.array([[float(s["labels_true"][p]) for p in PATHOLOGIES]
                      for s in studies], dtype=np.float32)
    M = np.stack([s["lung_mask"] for s in studies])
    G = np.array([s["patient_id"] for s in studies])
    C = {k: np.array([s["confounds"][k] for s in studies])
         for k in CONFOUND_STRENGTH}
    return X, Y, Ytrue, M, G, C
