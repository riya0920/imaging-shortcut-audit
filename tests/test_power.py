"""What it would cost to resolve the de-confounding result.

The gap list said "more seeds would close this and were not run", which reads
like "we could not be bothered". It is a claim about FEASIBILITY, and it was
wrong: the effect is more than an order of magnitude below the run-to-run
noise, so closing it is a multi-week compute job.

These tests need no training and no reference library.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import power


# The values the study actually measured, kept here so the tests describe the
# real experiment rather than a convenient one.
SD = 0.0585
EFFECT = 0.0039


def test_the_mdd_formula_matches_the_study():
    """The calculator must use the SAME constant as the experiment it
    describes. A power calculation that quietly picks a different multiplier
    is not describing that experiment."""
    assert power.K == 2.78
    assert power.mdd_at(3, SD) == pytest.approx(2.78 * SD / 3 ** 0.5)


def test_three_seeds_reproduce_the_published_mdd():
    """0.094 is what the study reported. If this drifts, one of the two is
    wrong."""
    assert power.mdd_at(3, SD) == pytest.approx(0.0939, abs=0.001)


def test_resolving_the_observed_effect_needs_over_a_thousand_seeds():
    """THE CORRECTION. 'More seeds would close this' implied a cheap fix was
    sitting there untaken."""
    need = power.seeds_needed(EFFECT, SD)
    assert need > 1000


def test_that_is_weeks_of_compute_not_an_afternoon(monkeypatch):
    """Stated in wall-clock, because 1,750 seeds does not sound like anything
    until it is converted into days."""
    need = power.seeds_needed(EFFECT, SD)
    seconds_per_fit = 515.0                 # measured, two fits per seed
    days = need * 2 * seconds_per_fit / 86400.0
    assert days > 7


def test_more_seeds_do_tighten_the_bound(_=None):
    """The claim was not that seeds are useless -- it was that they do not
    close the question. The bound genuinely improves."""
    assert power.mdd_at(100, SD) < power.mdd_at(25, SD) < power.mdd_at(3, SD)


def test_the_bound_shrinks_as_the_square_root(_=None):
    """Which is exactly why it is hopeless here: cutting the MDD by 24x needs
    576x the compute."""
    assert power.mdd_at(4, SD) == pytest.approx(power.mdd_at(1, SD) / 2)
    assert power.mdd_at(100, SD) == pytest.approx(power.mdd_at(1, SD) / 10)


def test_a_feasible_seed_count_still_does_not_resolve_it():
    """Even 100 seeds -- already a many-hour job -- leaves the MDD about FOUR
    times the effect.

    A first draft of this test asserted "an order of magnitude above" and
    failed: the real ratio at 100 seeds is 4.2x, not >10x. Worth leaving the
    correction visible, because the overclaim was in the direction of making
    the limitation sound more dramatic than it is.
    """
    ratio = power.mdd_at(100, SD) / EFFECT
    assert 3.0 < ratio < 6.0
    assert power.mdd_at(100, SD) > EFFECT


def test_zero_effect_is_never_resolvable():
    assert power.seeds_needed(0.0, SD) == float("inf")


def test_the_report_flags_which_rows_resolve():
    rows, need = power.report(SD, EFFECT, 3, 515.0)
    assert rows[0]["seeds"] == 3
    assert not any(r["resolves"] for r in rows), (
        "no feasible seed count should resolve a 0.0039 effect at sd 0.0585")
    assert need > 1000


def test_it_uses_the_measured_result_when_one_exists():
    """The calculator reads `out/complete.json` rather than hard-coding the
    numbers, so it cannot drift away from the run it describes."""
    measured = power.load_measured()
    if measured is None:
        pytest.skip("no out/complete.json -- run run_complete.py")
    assert 0.0 < float(measured["paired_sd"]) < 0.5
    assert len(measured["with_marker"]) == len(measured["without_marker"])
