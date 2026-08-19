"""Tests for the split discipline, the DICOM handling, and the de-identifier.

The model's accuracy is not tested, because it is a property of a generator I
wrote. What is tested is everything that would make the accuracy a lie: patient
leakage across the split, a metric that reports the wrong operating point, and
a de-identifier that leaves PHI in the header.
"""

import datetime
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import dicom_io
import synth
from train_audit import operating_points, split_indices


# ---------------------------------------------------------------------------
# Split discipline
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    studies = synth.build_dataset(n_patients=120, seed=2)
    return synth.as_arrays(studies)


def test_patient_level_split_shares_no_patient(data):
    _X, _Y, _Yt, _M, G, _C = data
    tr, va, te = split_indices(G, "patient")
    assert set(G[tr]) & set(G[te]) == set()
    assert set(G[tr]) & set(G[va]) == set()
    assert set(G[va]) & set(G[te]) == set()
    assert len(tr) + len(va) + len(te) == len(G)


def test_image_level_split_does_share_patients(data):
    """The failure mode, demonstrated rather than described. Same patient in
    train and test means the model can recognise the anatomy instead of the
    pathology, and every reported metric is inflated by an unknown amount."""
    _X, _Y, _Yt, _M, G, _C = data
    tr, _va, te = split_indices(G, "image")
    shared = set(G[tr]) & set(G[te])
    assert len(shared) > 0, "if this ever passes, the generator stopped "\
                            "giving patients multiple studies"


def test_multiple_studies_per_patient_exist(data):
    _X, _Y, _Yt, _M, G, _C = data
    _uniq, counts = np.unique(G, return_counts=True)
    assert counts.max() > 1


# ---------------------------------------------------------------------------
# The confounds are actually planted at the requested strength
# ---------------------------------------------------------------------------
def test_planted_confound_correlations_match_specification():
    studies = synth.build_dataset(n_patients=800, seed=4)
    _X, _Y, Ytrue, _M, _G, C = synth.as_arrays(studies)
    for name, (target, p_pos, p_neg) in synth.CONFOUND_STRENGTH.items():
        ti = synth.PATHOLOGIES.index(target)
        pos = C[name][Ytrue[:, ti] == 1].mean()
        neg = C[name][Ytrue[:, ti] == 0].mean()
        assert abs(pos - p_pos) < 0.06, f"{name}: P(confound|pos)={pos:.3f}"
        assert abs(neg - p_neg) < 0.06, f"{name}: P(confound|neg)={neg:.3f}"


def test_confounds_track_the_true_label_not_the_noisy_one():
    """Confounds come from the patient's real state, because in reality the
    acquisition circumstance is caused by the illness, not by what the
    report-mining pipeline wrote down."""
    studies = synth.build_dataset(n_patients=600, seed=6)
    _X, Y, Ytrue, _M, _G, C = synth.as_arrays(studies)
    oi = synth.PATHOLOGIES.index("opacity")
    gap_true = C["marker"][Ytrue[:, oi] == 1].mean() - C["marker"][Ytrue[:, oi] == 0].mean()
    gap_obs = C["marker"][Y[:, oi] == 1].mean() - C["marker"][Y[:, oi] == 0].mean()
    assert gap_true > gap_obs, "confound must correlate more with truth than "\
                               "with the noisy label"


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------
def test_operating_points_handle_a_perfect_classifier():
    """Regression test. argmin(|spec - 0.90|) picks the trivial ROC corner
    (spec=1.0, tpr=0.0) for a perfect classifier and reports 0% sensitivity
    at 90% specificity for a model with AUROC 1.000."""
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    op = operating_points(y, p)
    assert op["sens_at_90_spec"] == 1.0
    assert op["spec_at_90_sens"] == 1.0


def test_operating_points_constraints_are_met_not_approximated():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.35 + rng.normal(0.4, 0.2, 500), 0, 1)
    op = operating_points(y, p)
    pred = p >= op["threshold_at_90_spec"]
    spec = ((pred == 0) & (y == 0)).sum() / (y == 0).sum()
    assert spec >= 0.90 - 1e-9


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_meta():
    return {
        "PatientName": "DOE^JANE^^^", "PatientID": "MRN0042857",
        "PatientBirthDate": "19560312", "PatientSex": "F", "PatientAge": "094Y",
        "PatientAddress": "17 Elm St, Springfield", "PatientTelephoneNumbers":
        "555-0142", "AccessionNumber": "ACC99812", "InstitutionName":
        "Springfield General", "InstitutionAddress": "1 Hospital Way",
        "ReferringPhysicianName": "SMITH^ALAN", "PerformingPhysicianName":
        "OKONKWO^N", "OperatorsName": "TECH01", "StationName": "CR-ROOM-3",
        "StudyDescription": "CHEST PA AND LATERAL", "PhysiciansOfRecord":
        "SMITH^ALAN", "DeviceSerialNumber": "SN-77120", "StudyID": "ST-9001",
        "PatientComments": "call daughter Mary on 555-0199",
        "OtherPatientIDs": "OLDMRN-113", "StudyDate": "20240612",
        "SeriesDate": "20240612", "StudyTime": "141233",
        "StudyInstanceUID": "1.2.840.113619.2.55.3.1", "SeriesInstanceUID":
        "1.2.840.113619.2.55.3.2", "SOPInstanceUID": "1.2.840.113619.2.55.3.3",
        "PatientPosition": "AP",
    }


def test_dicom_roundtrip_preserves_pixels_and_tags(tmp_path, sample_meta):
    rng = np.random.default_rng(1)
    px = (rng.random((64, 64)) * 4095).astype(np.uint16)
    path = tmp_path / "t.dcm"
    dicom_io.write(str(path), px, sample_meta)
    with open(path, "rb") as fh:
        assert fh.read(132)[128:] == b"DICM"
    ds, back = dicom_io.read(str(path))
    assert back.shape == (64, 64)
    assert np.array_equal(back, px)
    assert ds["PatientID"] == "MRN0042857"
    assert ds["Rows"] == 64 and ds["Columns"] == 64


def test_deid_removes_every_tag_in_the_basic_profile_list(tmp_path, sample_meta):
    """Enumerates the list rather than spot-checking three tags. A de-identifier
    that handles PatientName and forgets PatientComments has not de-identified
    anything."""
    clean = dicom_io.deidentify(sample_meta)
    for kw in dicom_io.BASIC_PROFILE_REMOVE:
        assert clean.get(kw, "") == "", f"{kw} survived de-identification"
    assert clean["PatientIdentityRemoved"] == "YES"
    assert "DeidentificationMethod" in clean


def test_deid_survives_a_write_read_cycle(tmp_path, sample_meta):
    """The de-identified header must be clean ON DISK, not just in memory."""
    px = np.zeros((8, 8), dtype=np.uint16)
    path = tmp_path / "clean.dcm"
    dicom_io.write(str(path), px, dicom_io.deidentify(sample_meta))
    raw = open(path, "rb").read()
    for secret in [b"DOE^JANE", b"MRN0042857", b"Springfield", b"555-0142",
                   b"SMITH^ALAN", b"ACC99812", b"555-0199", b"OLDMRN-113"]:
        assert secret not in raw, f"{secret!r} still present in the file bytes"


def test_patient_id_pseudonym_is_stable_and_not_reversible(sample_meta):
    a = dicom_io.deidentify(sample_meta)
    b = dicom_io.deidentify(dict(sample_meta, StudyDate="20240901"))
    assert a["PatientID"] == b["PatientID"], "same patient must map to same id"
    assert "MRN0042857" not in a["PatientID"]
    other = dicom_io.deidentify(dict(sample_meta, PatientID="MRN0000001"))
    assert other["PatientID"] != a["PatientID"]


def test_dates_are_shifted_consistently_so_intervals_survive(sample_meta):
    """Date DELETION would destroy every interval-dependent analysis. The
    profile used here shifts by a per-patient offset instead."""
    s1 = dicom_io.deidentify(dict(sample_meta, StudyDate="20240612"))
    s2 = dicom_io.deidentify(dict(sample_meta, StudyDate="20240712"))
    d1 = datetime.datetime.strptime(s1["StudyDate"], "%Y%m%d")
    d2 = datetime.datetime.strptime(s2["StudyDate"], "%Y%m%d")
    assert (d2 - d1).days == 30
    assert s1["StudyDate"] != "20240612", "dates must actually move"


def test_ages_over_89_are_aggregated(sample_meta):
    assert dicom_io.deidentify(sample_meta)["PatientAge"] == "090Y+"
    young = dicom_io.deidentify(dict(sample_meta, PatientAge="064Y"))
    assert young["PatientAge"] == "064Y"


def test_uids_are_remapped_but_stay_internally_consistent(sample_meta):
    a = dicom_io.deidentify(sample_meta)
    b = dicom_io.deidentify(sample_meta)
    assert a["StudyInstanceUID"] != sample_meta["StudyInstanceUID"]
    assert a["StudyInstanceUID"] == b["StudyInstanceUID"]
    assert a["StudyInstanceUID"].startswith("2.25.")


def test_retained_tags_are_documented():
    """Every tag that survives must have a stated reason. Being able to say why
    a tag stays is what separates a de-identifier from a delete key."""
    for kw in dicom_io.RETAINED:
        assert dicom_io.RETAINED[kw], f"{kw} retained without a reason"
    assert set(dicom_io.RETAINED) & set(dicom_io.BASIC_PROFILE_REMOVE) == set()
