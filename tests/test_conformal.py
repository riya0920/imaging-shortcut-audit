"""Tests for conformal prediction.

Coverage is tested against PLANTED truth: construct data whose true label
distribution is known, calibrate, and check the empirical coverage lands on the
target. A coverage guarantee tested only against a model's own output tests
nothing, because there is no reference to be wrong about.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import conformal as CP


def _draw(p, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(len(p)) < p).astype(int)


def _calibrated(n=4000, seed=0):
    """Scores that ARE the probabilities, so the model is perfectly calibrated
    by construction and coverage should land on the nominal level."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    return _draw(p, seed + 1), p


# --------------------------------------------------------------------------
# the guarantee
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_coverage_lands_on_the_nominal_level(alpha):
    y_cal, p_cal = _calibrated(seed=0)
    y_te, p_te = _calibrated(seed=10)
    q = CP.calibrate_threshold(y_cal, p_cal, alpha=alpha)
    ev = CP.evaluate(y_te, p_te, q["qhat"])
    assert ev["coverage"] == pytest.approx(1 - alpha, abs=0.03)


def test_a_badly_miscalibrated_model_still_gets_coverage():
    """THE PROPERTY THAT MATTERS FOR THIS PROJECT. Conformal needs no
    assumption that the model is calibrated, well-specified, or even good --
    only exchangeability. A miscalibrated model pays with WIDER SETS, not with
    lost coverage, which is exactly what a model with a known shortcut needs
    since the one thing it lacks is a trustworthy probability."""
    y_cal, p_cal = _calibrated(seed=1)
    y_te, p_te = _calibrated(seed=2)
    # ruin the calibration without changing the ranking
    bad_cal = np.clip(p_cal ** 3, 1e-6, 1 - 1e-6)
    bad_te = np.clip(p_te ** 3, 1e-6, 1 - 1e-6)
    q = CP.calibrate_threshold(y_cal, bad_cal, alpha=0.10)
    ev = CP.evaluate(y_te, bad_te, q["qhat"])
    assert ev["coverage"] == pytest.approx(0.90, abs=0.04)


def test_a_worse_model_pays_in_set_size_not_coverage():
    y_cal, p_cal = _calibrated(seed=3)
    y_te, p_te = _calibrated(seed=4)
    rng = np.random.default_rng(9)
    noisy_cal = np.clip(p_cal + rng.normal(0, 0.35, len(p_cal)), 0.01, 0.99)
    noisy_te = np.clip(p_te + rng.normal(0, 0.35, len(p_te)), 0.01, 0.99)

    good = CP.evaluate(y_te, p_te,
                       CP.calibrate_threshold(y_cal, p_cal, 0.10)["qhat"])
    poor = CP.evaluate(y_te, noisy_te,
                       CP.calibrate_threshold(y_cal, noisy_cal, 0.10)["qhat"])
    assert poor["mean_set_size"] > good["mean_set_size"]
    assert poor["coverage"] > 0.85


def test_a_tighter_alpha_widens_the_sets():
    y_cal, p_cal = _calibrated(seed=5)
    y_te, p_te = _calibrated(seed=6)
    loose = CP.evaluate(y_te, p_te,
                        CP.calibrate_threshold(y_cal, p_cal, 0.20)["qhat"])
    tight = CP.evaluate(y_te, p_te,
                        CP.calibrate_threshold(y_cal, p_cal, 0.02)["qhat"])
    assert tight["mean_set_size"] > loose["mean_set_size"]
    assert tight["coverage"] > loose["coverage"]


# --------------------------------------------------------------------------
# the finite-sample correction
# --------------------------------------------------------------------------

def test_an_unachievable_alpha_is_declared_rather_than_clamped():
    """At n=5 the smallest achievable alpha is 1/6. Clamping and reporting a
    95% guarantee that does not hold is worse than refusing."""
    out = CP.calibrate_threshold([1, 0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2, 0.7],
                                 alpha=0.01)
    assert out.get("achievable") is False
    assert "smallest achievable" in out["note"]


def test_the_quantile_uses_the_finite_sample_correction():
    """Plain (1-alpha) gives only asymptotic coverage; at these sample sizes
    that is the difference between a guarantee and a hope."""
    out = CP.calibrate_threshold([1] * 100, [0.9] * 100, alpha=0.10)
    assert out["effective_level"] > 0.90


def test_no_calibration_data_yields_the_widest_possible_set():
    out = CP.calibrate_threshold([], [], alpha=0.10)
    assert out["qhat"] == 1.0
    assert CP.predict_set(0.5, out["qhat"]) == (1, 0)


# --------------------------------------------------------------------------
# what a set means
# --------------------------------------------------------------------------

def test_a_confident_score_gives_a_singleton():
    assert CP.predict_set(0.99, 0.2) == (1,)
    assert CP.predict_set(0.01, 0.2) == (0,)


def test_an_uncertain_score_gives_both_labels():
    """Not a probability near 0.5 -- an explicit abstention with a coverage
    guarantee behind it."""
    assert set(CP.predict_set(0.5, 0.6)) == {0, 1}


def test_an_empty_set_is_possible_and_is_the_interesting_output():
    """Neither label is plausible at this confidence: the image looks unlike
    anything in the calibration set, which for a shortcut model is the most
    informative thing it can say."""
    assert CP.predict_set(0.5, 0.1) == ()


def test_the_abstention_rate_is_reported_alongside_coverage():
    """Coverage is the guarantee; SET SIZE is the information. A model that
    always returns both labels has perfect coverage and has said nothing."""
    y_te, p_te = _calibrated(seed=7)
    ev = CP.evaluate(y_te, p_te, 0.9)
    assert ev["coverage"] == pytest.approx(1.0, abs=0.01)
    # scores are uniform on (0.02, 0.98); at qhat=0.9 both labels are admitted
    # for p in [0.10, 0.90], which is ~83% of them
    assert ev["abstention_rate"] > 0.75       # ...and says nothing


# --------------------------------------------------------------------------
# the stratified check
# --------------------------------------------------------------------------

def test_the_marginal_guarantee_can_hide_a_failing_stratum():
    """Split conformal guarantees MARGINAL coverage only. For a model whose
    known failure is a per-stratum difference, quoting the marginal number
    alone is the same dilution calibrate.py documents for aggregate ECE."""
    # CONTINUOUS scores. A fixture with two discrete score values puts qhat on
    # one of them, every set then contains both labels, coverage is 1.0
    # everywhere and the gap is exactly zero -- a degenerate pass that would
    # have looked like the property failing to hold.
    n = 3000
    rng = np.random.default_rng(11)
    cue = np.zeros(2 * n, dtype=bool)
    cue[:n] = True
    base = rng.uniform(0.05, 0.95, 2 * n)
    y = _draw(base, 12)
    # cue-present scores are pushed toward the extremes: same ranking, badly
    # overconfident, which is what a shortcut model does where the cue fires
    p_said = base.copy()
    p_said[:n] = np.clip(base[:n] ** 0.25, 0.01, 0.99)
    q = CP.calibrate_threshold(y, p_said, alpha=0.10)
    out = CP.coverage_by_stratum(y, p_said, q["qhat"], cue)
    gap = abs(out["cue_present"]["coverage"] - out["cue_absent"]["coverage"])
    assert gap > 0.05
    assert "does not hold" in out["verdict"]["reading"]


def test_similar_strata_are_reported_as_such():
    y, p = _calibrated(seed=8)
    cue = np.random.default_rng(13).random(len(y)) < 0.4
    q = CP.calibrate_threshold(y, p, alpha=0.10)
    out = CP.coverage_by_stratum(y, p, q["qhat"], cue)
    assert "not hiding a subgroup" in out["verdict"]["reading"]


def test_the_marginal_is_reported_beside_the_strata():
    y, p = _calibrated(seed=14)
    cue = np.random.default_rng(15).random(len(y)) < 0.5
    out = CP.coverage_by_stratum(y, p, 0.5, cue)
    assert out["marginal"]["n"] == len(y)
    assert out["cue_present"]["n"] + out["cue_absent"]["n"] == len(y)
