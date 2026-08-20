"""Detect the shortcut cues at inference time.

WHY A SERVICE SHOULD DETECT ITS OWN SHORTCUTS
---------------------------------------------
The audit established that this model keys on the laterality marker rather than
on lung findings. The usual response is to write that in a model card and serve
the model anyway. But the audit result is per-MODEL, and the failure is
per-STUDY: a prediction on an image with the marker present is a different kind
of claim from a prediction on an image without it, and the service knows which
one it is holding.

So the cue detectors run on every request, and any prediction made on an image
carrying a cue the model was shown to depend on comes back flagged. The
alternative -- a static disclaimer in a document the integrating team read once
-- puts the caveat in the wrong place and at the wrong time.

WHAT MAKES THIS EASY HERE AND HARD IN REALITY
---------------------------------------------
These detectors are trivial because this project PLANTED the cues and knows
their exact geometry: the marker is a glyph in a fixed box, the border is a
fixed-width frame, the device is a bright curve. Three thresholds and the job
is done.

On real radiographs none of that holds. Burned-in annotations move, vary in
font and size, and are sometimes rotated; portable-film borders vary by
manufacturer and by processing; support devices genuinely appear in the lung
fields and are diagnostically relevant rather than being artefacts. Detecting
them is itself a vision problem, plausibly harder than the classification task,
and a detector with its own false-negative rate cannot be the safety control
for a model whose failure mode it is meant to catch.

That limitation is the honest content of this file. What generalises is the
DESIGN -- attach the audit finding to the individual prediction rather than to
the model as a whole -- not this implementation of it.
"""

from __future__ import annotations

import numpy as np

from synth import BORDER_WIDTH, IMG, MARKER_BOX

# Thresholds chosen against the generator's rendering constants: the marker
# adds +0.55 over a ~0.44 background, the border +0.45, the device +0.50. They
# are calibrated to THIS generator and would not survive contact with a real
# PACS. Stated rather than buried, because a threshold tuned on the data it is
# evaluated on is exactly the kind of thing that gets quoted as a detection rate.
MARKER_MIN_BRIGHT_FRACTION = 0.12
BORDER_MIN_MEAN_EXCESS = 0.12
DEVICE_MIN_BRIGHT_PIXELS = 25
BRIGHT = 0.80


def detect_marker(img):
    """Laterality marker: a bright glyph in the top-right box."""
    r0, r1, c0, c1 = MARKER_BOX
    patch = img[r0:r1, c0:c1]
    frac = float((patch > BRIGHT).mean())
    return {"present": frac >= MARKER_MIN_BRIGHT_FRACTION,
            "bright_fraction": frac,
            "threshold": MARKER_MIN_BRIGHT_FRACTION}


def detect_border(img):
    """Portable-film border: a bright frame of fixed width."""
    w = BORDER_WIDTH
    frame = np.concatenate([img[:w, :].ravel(), img[-w:, :].ravel(),
                            img[:, :w].ravel(), img[:, -w:].ravel()])
    interior = img[w + 2:-(w + 2), w + 2:-(w + 2)]
    excess = float(frame.mean() - interior.mean())
    return {"present": excess >= BORDER_MIN_MEAN_EXCESS,
            "mean_excess": excess, "threshold": BORDER_MIN_MEAN_EXCESS}


def detect_device(img):
    """Support device: a bright near-vertical line in the right lung field.

    Deliberately scoped to the region the generator draws it in. A whole-image
    bright-pixel count would also fire on the marker and on the border, and a
    cue detector that cannot tell its cues apart is not usable for attaching a
    per-cue warning to a prediction.
    """
    band = img[int(IMG * 0.20):int(IMG * 0.80), int(IMG * 0.55):int(IMG * 0.85)]
    n = int((band > BRIGHT).sum())
    return {"present": n >= DEVICE_MIN_BRIGHT_PIXELS, "bright_pixels": n,
            "threshold": DEVICE_MIN_BRIGHT_PIXELS}


# Which pathology each cue was confounded WITH, from synth.CONFOUND_STRENGTH.
# A cue only matters for the pathology it is correlated with; flagging every
# prediction because some cue is present would make the warning worthless.
CUE_TARGET = {"marker": "opacity", "border": "effusion", "device": "effusion"}


def detect_all(img):
    return {"marker": detect_marker(img), "border": detect_border(img),
            "device": detect_device(img)}


def warnings_for(cues, audited_dependencies):
    """Per-pathology warnings, given which dependencies the audit CONFIRMED.

    `audited_dependencies` is the set of cue names the shortcut audit actually
    demonstrated the model relies on -- not every cue the generator planted.
    The audit found the marker dependency real and the effusion cues weak, so
    warning about all three would dilute the one warning that is earned.
    """
    out = {}
    for cue, info in cues.items():
        if not info["present"] or cue not in audited_dependencies:
            continue
        target = CUE_TARGET[cue]
        out.setdefault(target, []).append(
            f"a '{cue}' cue is present in this image, and the shortcut audit "
            f"showed this model's '{target}' prediction depends on it. Treat "
            f"this score as reflecting the cue at least as much as the anatomy")
    return out
