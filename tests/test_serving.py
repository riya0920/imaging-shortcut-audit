"""Tests for calibration, cue detection, and the inference API.

The calibration tests are built around planted ground truth, the same way the
rest of the project is: construct a distribution whose true calibration is
known, then check the estimator recovers it. A calibration metric tested only
against a model's output tests nothing -- there is no reference to be wrong
about.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from http.server import HTTPServer

import calibrate as C
import cues as Q
import synth


# --------------------------------------------------------------------------
# calibration, against planted truth
# --------------------------------------------------------------------------

def _draw(p, seed=0):
    """Outcomes drawn from p, so the data is calibrated BY CONSTRUCTION."""
    rng = np.random.default_rng(seed)
    return (rng.random(len(p)) < p).astype(float)


def test_perfectly_calibrated_data_has_near_zero_ece():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    assert C.ece(_draw(p, 2), p)["ece"] < 0.02


def test_ece_grows_when_predictions_are_shifted_off_truth():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.75, 20000)
    y = _draw(p, 4)
    good = C.ece(y, p)["ece"]
    bad = C.ece(y, np.clip(p + 0.20, 0, 1))["ece"]
    assert bad > good
    assert bad == pytest.approx(0.20, abs=0.03)   # the size of the shift


def test_ece_is_reported_with_its_bin_count():
    """ECE is not comparable across binning schemes. Returning it bare is how
    it gets quoted against a number computed a different way."""
    out = C.ece(np.array([0.0, 1.0] * 50), np.linspace(0.01, 0.99, 100))
    assert out["n_bins"] == 10 and out["n"] == 100


def test_equal_count_bins_beat_equal_width_on_skewed_scores():
    """The reason for equal-count binning: predictions pile up near zero, and
    equal-width bins leave the top deciles holding a handful of cases."""
    rng = np.random.default_rng(5)
    p = rng.beta(1.2, 12.0, 4000)                 # heavily right-skewed
    rows = C.reliability(_draw(p, 6), p, n_bins=10)
    counts = [r["n"] for r in rows]
    assert min(counts) > 0.5 * (len(p) / 10)      # no near-empty bin
    # an equal-width scheme on the same data would not manage that
    width_counts, _ = np.histogram(p, bins=np.linspace(0, 1, 11))
    assert min(width_counts) < min(counts)


def test_reliability_gap_sign_says_which_direction_the_model_is_wrong():
    p = np.full(2000, 0.30)
    y = np.zeros(2000)
    y[:800] = 1                                    # observed 40% vs predicted 30%
    rows = C.reliability(y, p, n_bins=4)
    assert all(r["gap"] > 0 for r in rows)         # under-predicting


def test_brier_decomposition_recomposes_to_brier():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 8000)
    d = C.brier_decomposition(_draw(p, 8), p)
    assert d["recomposed"] == pytest.approx(d["brier"], abs=0.01)


def test_brier_can_improve_while_calibration_does_not():
    """The argument for decomposing rather than quoting Brier: a model can
    lower Brier by ranking better while its probabilities stay just as wrong."""
    rng = np.random.default_rng(9)
    n = 8000
    y = (rng.random(n) < 0.3).astype(float)
    flat = np.full(n, 0.3)                          # calibrated, no resolution
    sharp = np.clip(0.3 + 0.25 * (2 * y - 1) + rng.normal(0, 0.05, n), 0.01, 0.99)
    d_flat, d_sharp = (C.brier_decomposition(y, flat),
                       C.brier_decomposition(y, sharp))
    assert d_sharp["brier"] < d_flat["brier"]       # better Brier
    assert d_sharp["resolution"] > d_flat["resolution"]   # ...from resolution


def test_temperature_recovers_a_planted_distortion():
    """Plant the miscalibration, then recover it. Logits divided by 2 produce
    underconfident scores, so the fitted temperature should be about 2."""
    rng = np.random.default_rng(11)
    z = rng.normal(0, 2.5, 20000)
    y = (rng.random(20000) < C._sigmoid(z)).astype(float)
    fit = C.fit_temperature(y, z * 2.0)
    assert fit["temperature"] == pytest.approx(2.0, rel=0.10)
    assert fit["nll_after"] < fit["nll_before"]


def test_temperature_of_already_calibrated_logits_is_about_one():
    rng = np.random.default_rng(12)
    z = rng.normal(0, 2.0, 20000)
    y = (rng.random(20000) < C._sigmoid(z)).astype(float)
    fit = C.fit_temperature(y, z)
    assert fit["temperature"] == pytest.approx(1.0, rel=0.12)
    assert "already calibrated" in fit["direction"]


def test_temperature_scaling_never_changes_the_ranking():
    """A one-parameter temperature is monotone in the logit, so it cannot
    re-rank. This is the property that makes it safe to apply after the fact,
    and the reason a two-parameter Platt fit was not used."""
    z = np.linspace(-6, 6, 500)
    for T in (0.4, 1.0, 3.7):
        p = C.apply_temperature(z, T)
        assert np.all(np.diff(p) > 0)


def test_ece_does_not_cancel_opposite_direction_errors():
    """A positive property, pinned because I assumed the opposite and was wrong.

    The intuitive story about aggregate calibration is that one stratum
    over-predicting and another under-predicting average out to a clean number.
    That is false for ECE, which takes an absolute value inside each bin. Both
    strata here are wrong by 0.20 in opposite directions and the aggregate
    reports 0.20, not 0."""
    n = 4000
    p = np.concatenate([np.full(n, 0.70), np.full(n, 0.30)])
    y = np.concatenate([_draw(np.full(n, 0.50), 13),
                        _draw(np.full(n, 0.50), 14)])
    assert C.ece(y, p, n_bins=4)["ece"] == pytest.approx(0.20, abs=0.02)


def test_the_aggregate_hides_a_bad_stratum_by_diluting_it():
    """The mechanism that DOES operate: a small badly-calibrated stratum
    averaged against a large well-behaved one. The aggregate lands between
    them, below the worst, and reads as reassurance about a subgroup it is not
    describing.

    Planted to match the shape of the real finding: a small cue-present stratum
    over-predicted by ~0.20, a large cue-absent stratum calibrated."""
    small, large = 400, 3600
    cue = np.zeros(small + large, dtype=bool)
    cue[:small] = True
    p = np.concatenate([np.full(small, 0.70), np.full(large, 0.30)])
    y = np.concatenate([_draw(np.full(small, 0.50), 13),
                        _draw(np.full(large, 0.30), 14)])
    out = C.stratified_calibration(y, p, cue, n_bins=4)
    assert out["cue_present"]["ece"] > 0.15
    assert out["cue_absent"]["ece"] < 0.05
    # the aggregate sits strictly between the two, nearer the large stratum
    assert out["cue_absent"]["ece"] < out["aggregate"]["ece"]         < out["cue_present"]["ece"]
    assert out["aggregate"]["ece"] < 0.5 * out["cue_present"]["ece"]
    assert "hides" in out["verdict"]["reading"]


def test_aggregate_ece_is_about_the_count_weighted_mean_of_the_strata():
    """Names the dilution arithmetic explicitly, so the claim in the module
    docstring is checked rather than asserted."""
    small, large = 400, 3600
    cue = np.zeros(small + large, dtype=bool)
    cue[:small] = True
    p = np.concatenate([np.full(small, 0.70), np.full(large, 0.30)])
    y = np.concatenate([_draw(np.full(small, 0.50), 13),
                        _draw(np.full(large, 0.30), 14)])
    out = C.stratified_calibration(y, p, cue, n_bins=4)
    expected = ((small * out["cue_present"]["ece"]
                 + large * out["cue_absent"]["ece"]) / (small + large))
    assert out["aggregate"]["ece"] == pytest.approx(expected, abs=0.03)


def test_stratified_verdict_does_not_fire_on_noise():
    """Regression. The verdict was a bare ratio test, so on well-calibrated
    data where both strata land near 0.01 the ratio cleared 1.25 and it
    confidently reported a hidden stratum that was sampling noise. It now needs
    an absolute gap as well."""
    rng = np.random.default_rng(15)
    p = rng.uniform(0.05, 0.9, 6000)
    cue = rng.random(6000) < 0.4
    out = C.stratified_calibration(_draw(p, 16), p, cue, n_bins=5)
    assert out["verdict"]["absolute_gap"] < 0.02
    assert "not hiding a stratum" in out["verdict"]["reading"]


def test_stratified_calibration_refuses_to_bin_a_tiny_stratum():
    """Splitting a small sample and then asking for bins of it produces bins of
    four cases and a reliability curve that is sampling noise drawn as signal."""
    p = np.full(100, 0.4)
    cue = np.zeros(100, dtype=bool)
    cue[:5] = True
    out = C.stratified_calibration(_draw(p, 17), p, cue, n_bins=6)
    assert "too few" in out["cue_present"]["note"]


# --------------------------------------------------------------------------
# cue detection
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def studies():
    return synth.build_dataset(n_patients=60, seed=21)


def test_marker_detector_agrees_with_the_generator(studies):
    """Scored against the generator's own record of what it drew, which is the
    only reason a threshold detector like this can be trusted at all here."""
    truth = np.array([s["confounds"]["marker"] for s in studies])
    got = np.array([Q.detect_marker(s["image"])["present"] for s in studies])
    assert (truth == got).mean() > 0.98


def test_border_detector_agrees_with_the_generator(studies):
    truth = np.array([s["confounds"]["border"] for s in studies])
    got = np.array([Q.detect_border(s["image"])["present"] for s in studies])
    assert (truth == got).mean() > 0.95


def test_device_detector_does_not_fire_on_the_marker_or_the_border(studies):
    """A cue detector that cannot tell its cues apart cannot attach a per-cue
    warning to a prediction, which is the only thing it is for."""
    only_marker = [s for s in studies if s["confounds"]["marker"]
                   and not s["confounds"]["device"]]
    assert only_marker
    fired = [Q.detect_device(s["image"])["present"] for s in only_marker]
    assert np.mean(fired) < 0.10


def test_warnings_fire_only_for_audited_dependencies():
    """The audit confirmed the marker dependency and could not confirm the
    effusion cues. Warning about all three would dilute the one that is earned."""
    cues = {"marker": {"present": True}, "border": {"present": True},
            "device": {"present": True}}
    warn = Q.warnings_for(cues, {"marker"})
    assert "opacity" in warn and "effusion" not in warn


def test_no_warning_when_the_cue_is_absent():
    cues = {"marker": {"present": False}, "border": {"present": False},
            "device": {"present": False}}
    assert Q.warnings_for(cues, {"marker"}) == {}


# --------------------------------------------------------------------------
# the API
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api():
    import serve as S
    if not os.path.exists(os.path.join(ROOT, S.MODEL_DIR,
                                       f"{S.NAME}-v1.json")):
        pytest.skip("no registered model; run `python serve.py --register`")
    S.warm(1)
    httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", S
    httpd.shutdown()
    httpd.server_close()


def _post(base, path, payload, raw=False):
    data = payload if raw else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_and_model_routes(api):
    base, _S = api
    assert _get(base, "/health")[1]["status"] == "ok"
    status, m = _get(base, "/model")
    assert status == 200
    assert m["audited_dependencies"] == ["marker"]
    assert "NOT a diagnostic device" in m["intended_use"]


def test_wrong_sized_image_is_refused(api):
    base, _S = api
    code, _h, b = _post(base, "/predict", {"pixels": [[0.0] * 8] * 8})
    assert code == 400
    assert "4096" in json.loads(b)["error"]


def test_missing_image_field_is_refused(api):
    base, _S = api
    code, _h, b = _post(base, "/predict", {"nothing": 1})
    assert code == 400
    assert "pixels" in json.loads(b)["error"]


def test_prediction_on_a_marker_image_carries_a_shortcut_warning(api, studies):
    base, _S = api
    s = next(x for x in studies if x["confounds"]["marker"])
    _c, _h, b = _post(base, "/predict", {"pixels": s["image"].tolist()})
    body = json.loads(b)
    op = next(p for p in body["predictions"] if p["pathology"] == "opacity")
    assert body["cues_detected"]["marker"] is True
    assert op["shortcut_warnings"]
    assert "shortcut audit" in op["shortcut_warnings"][0]


def test_prediction_without_the_cue_carries_no_warning(api, studies):
    base, _S = api
    s = next(x for x in studies if not x["confounds"]["marker"])
    _c, _h, b = _post(base, "/predict", {"pixels": s["image"].tolist()})
    body = json.loads(b)
    op = next(p for p in body["predictions"] if p["pathology"] == "opacity")
    assert op["shortcut_warnings"] == []


def test_cardiomegaly_is_never_warned_about(api, studies):
    """No cue was planted for it and the audit found no dependency. A warning
    there would be noise."""
    base, _S = api
    s = next(x for x in studies if x["confounds"]["marker"])
    _c, _h, b = _post(base, "/predict", {"pixels": s["image"].tolist()})
    card = next(p for p in json.loads(b)["predictions"]
                if p["pathology"] == "cardiomegaly")
    assert card["shortcut_warnings"] == []


def test_both_calibrated_and_uncalibrated_scores_are_returned(api, studies):
    """The temperature is disclosed alongside both numbers, so a reader can
    see what the post-hoc adjustment did rather than taking it on trust."""
    base, _S = api
    _c, _h, b = _post(base, "/predict",
                      {"pixels": studies[0]["image"].tolist()})
    for p in json.loads(b)["predictions"]:
        assert 0.0 <= p["probability"] <= 1.0
        assert 0.0 <= p["probability_uncalibrated"] <= 1.0
        assert p["temperature"] > 0


def test_every_prediction_carries_intended_use(api, studies):
    base, _S = api
    _c, _h, b = _post(base, "/predict",
                      {"pixels": studies[0]["image"].tolist()})
    assert "NOT a diagnostic device" in json.loads(b)["intended_use"]


def test_overlay_returns_a_png_with_the_caveat_in_a_header(api, studies):
    """The caveat travels with the artefact: a PNG saved out of this response
    and pasted into a slide loses every surrounding word."""
    base, _S = api
    s = next(x for x in studies if x["confounds"]["marker"])
    code, hdr, png = _post(base, "/overlay", {"pixels": s["image"].tolist()})
    assert code == 200
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    caveat = hdr["X-Explanation-Caveat"]
    assert "not an explanation" in caveat
    assert "SHORTCUT CUE DETECTED" in caveat


def test_overlay_of_a_clean_image_omits_the_shortcut_line(api, studies):
    base, _S = api
    s = next(x for x in studies if not any(x["confounds"].values()))
    code, hdr, _png = _post(base, "/overlay", {"pixels": s["image"].tolist()})
    assert code == 200
    assert "SHORTCUT CUE DETECTED" not in hdr["X-Explanation-Caveat"]


def test_dicom_path_deidentifies_before_scoring(api, studies, tmp_path):
    """The order is the control: identifiers must not survive into the scoring
    step, its memory or its logs."""
    import dicom_io
    base, _S = api
    s = next(x for x in studies if x["confounds"]["marker"])
    px = (s["image"] * 65535).astype(np.uint16)
    path = str(tmp_path / "t.dcm")
    dicom_io.write(path, px, {
        "PatientName": "DOE^JANE", "PatientID": "MRN-77", "StudyDate": "20240311",
        "PatientBirthDate": "19631104", "AccessionNumber": "ACC-1",
        "InstitutionName": "St Elsewhere", "Modality": "CR",
        "StudyInstanceUID": "1.2.3", "SeriesInstanceUID": "1.2.4",
        "SOPInstanceUID": "1.2.5", "Rows": px.shape[0], "Columns": px.shape[1]})
    with open(path, "rb") as fh:
        code, _h, b = _post(base, "/predict/dicom", fh.read(), raw=True)
    assert code == 200
    d = json.loads(b)["deidentification"]
    assert d["applied_before_inference"] is True
    assert "PatientName" in d["tags_blanked"]
    assert "PatientID" in d["tags_pseudonymised_or_shifted"]
    assert "Modality" in d["tags_retained_unchanged"]
    # nothing identifying survives anywhere in the response
    assert "DOE^JANE" not in b.decode()
    assert "MRN-77" not in b.decode()


def test_dicom_response_flags_burned_in_pixel_annotation(api, studies, tmp_path):
    """Tag de-identification cannot see pixels, and this project's entire
    subject is a burned-in pixel annotation. On a real film that same overlay
    routinely carries a name or an MRN."""
    import dicom_io
    base, _S = api
    s = next(x for x in studies if x["confounds"]["marker"])
    px = (s["image"] * 65535).astype(np.uint16)
    path = str(tmp_path / "m.dcm")
    dicom_io.write(path, px, {"PatientID": "X", "Modality": "CR",
                              "Rows": px.shape[0], "Columns": px.shape[1]})
    with open(path, "rb") as fh:
        _c, _h, b = _post(base, "/predict/dicom", fh.read(), raw=True)
    pd_ = json.loads(b)["deidentification"]["pixel_data"]
    assert "marker" in pd_["burned_in_candidates"]
    assert "NOT de-identified" in pd_["status"]
    assert "must not be treated as clearance" in pd_["screen_quality"]


def test_non_dicom_body_is_a_400_not_a_500(api):
    base, _S = api
    code, _h, b = _post(base, "/predict/dicom", b"this is not a DICOM", raw=True)
    assert code == 400
    assert "DICOM" in json.loads(b)["error"]


def test_unknown_route_lists_the_real_ones(api):
    base, _S = api
    code, body = _get(base, "/diagnose")
    assert code == 404
    assert "POST /predict" in body["routes"]
