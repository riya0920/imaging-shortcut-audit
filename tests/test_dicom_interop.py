"""Interoperability of the hand-rolled DICOM code against `pydicom`.

SKIPS when `pydicom` is absent. `src/dicom_io.py` does not depend on it -- the
reference is used to AUDIT the implementation, never to provide it.

The point of these tests is the one thing the existing suite structurally
cannot do. A hand-rolled writer checked only by its own reader proves almost
nothing: two consistent misreadings of the standard agree with each other
perfectly, and the bugs live exactly where the encoder and decoder share an
assumption.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dicom_io as D

pytest.importorskip("pydicom", reason="interoperability audit only")

import validate_dicom as V                                    # noqa: E402


@pytest.fixture(scope="module")
def tmp():
    return tempfile.mkdtemp(prefix="dicom-interop-")


def test_pydicom_can_read_what_we_write(tmp):
    """If an independent implementation cannot read it, it is not Part 10 --
    it is a private format that happens to start with DICM."""
    r = V.mine_to_pydicom(tmp)
    assert r["transfer_syntax"] == "1.2.840.10008.1.2.1"
    assert r["patient_id"] == "MRN-99"
    assert r["rows_cols"] == (64, 64)


def test_pixels_survive_the_trip_out(tmp):
    """Metadata agreeing while pixels are mangled is the failure that a
    tag-only comparison would miss entirely."""
    assert V.mine_to_pydicom(tmp)["pixels_match"]


def test_we_can_read_what_pydicom_writes(tmp):
    """THE TEST THAT COULD NOT EXIST BEFORE.

    A parser exercised only against its own writer never meets a construct that
    writer does not emit. This is the direction that catches "I never encode it
    that way, so I never learned to decode it".
    """
    r = V.pydicom_to_mine(tmp)
    assert r["patient_id"] == "MRN-99"
    assert r["study_date"] == "20240612"
    assert r["pixels_match"]


def test_no_phi_value_survives_de_identification(tmp):
    """Verified by an independent parser rather than by a byte grep.

    The grep in `test_deid_survives_a_write_read_cycle` is the stronger check
    for 'is this string gone'. This is the stronger check for 'is this tag
    carrying anything' -- a value can survive re-encoded, and a grep for the
    original bytes would not notice.
    """
    r = V.deid_verified_by_pydicom(tmp)
    assert r["nonempty_tags"] == []
    assert r["value_survivors"] == []
    assert r["patient_id_pseudonymised"]


def test_removed_tags_are_present_but_zero_length(tmp):
    """DOCUMENTS A KNOWN NON-CONFORMANCE, so it cannot be mistaken for a leak.

    `deidentify()` applies ONE action to everything in `BASIC_PROFILE_REMOVE`:
    zero-length value, i.e. PS3.15 action Z. The standard assigns actions PER
    TAG and the set includes both X (remove the tag) and Z (retain it empty).

    Uniform Z is privacy-equivalent for the value -- which is why the test
    above passes -- but a tag whose assigned action is X should be ABSENT, and
    a strict conformance checker would flag it. Asserted here so the behaviour
    is pinned and visible rather than discovered by a reviewer.
    """
    r = V.deid_verified_by_pydicom(tmp)
    assert "PatientName" in r["zeroed_tags"]
    assert "AccessionNumber" in r["zeroed_tags"]
    # and the list is named REMOVE while implementing retain-and-zero
    assert D.BASIC_PROFILE_REMOVE, "the naming is misleading; see docs/"


def test_dates_are_shifted_and_the_interval_is_preserved():
    """The whole argument for shifting rather than deleting: the gap between
    two studies carries clinical meaning, the calendar anchor does not."""
    r = V.date_intervals_preserved()
    assert r["preserved"]
    assert r["before"] == 74


def test_retained_tags_are_the_documented_ones(tmp):
    """Every retained tag has a written reason in `RETAINED`. Being able to say
    WHY a tag stays is what separates a de-identifier from a delete key."""
    r = V.deid_verified_by_pydicom(tmp)
    for kw in r["retained_tags"]:
        assert kw in D.RETAINED and D.RETAINED[kw]
