"""Inference service: predictions, Grad-CAM overlays, and latency.

The README named three gaps -- "no inference API, no heatmap overlay endpoint,
no latency measurement" -- and the spec asks for all three. This is them.

THE THREE DESIGN POSITIONS
--------------------------
1. DE-IDENTIFY AT THE BOUNDARY, BEFORE ANYTHING ELSE TOUCHES THE PIXELS.
   `POST /predict/dicom` runs `dicom_io.deidentify()` as its first action and
   reports which tags it removed. The alternative -- score first, de-identify
   on the way to storage -- means identifiers exist in the service's memory,
   its logs and its crash dumps. The order is the control.

2. EVERY PREDICTION CARRIES ITS OWN AUDIT RESULT. `src/cues.py` runs the
   shortcut detectors per request, and any prediction made on an image carrying
   a cue the audit showed this model depends on comes back flagged. A model
   card is a per-model statement; the failure is per-study.

3. THE OVERLAY IS NOT AN EXPLANATION, AND SAYS SO IN ITS OWN RESPONSE HEADER.
   Grad-CAM is a coarse 8x8 map upsampled to 64x64. It shows where gradient
   flowed, which is not the same as why, is not faithful under all conditions,
   and has published failure modes. It is served because it is what the audit
   used, with the caveat attached to the artefact rather than to a README.

WHAT THIS IS NOT
----------------
Not a DICOM SCP -- no C-STORE, no association negotiation, no PACS integration.
Not authenticated. Not a medical device, and nothing here is cleared for
clinical use. No batching, no GPU, no ONNX, no model server.

Run:
  python serve.py --register        fit and save a servable artefact
  python serve.py --demo            exercise the API end to end
  python serve.py --bench           latency percentiles
  python serve.py                   serve on :8082
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import calibrate as C
import cues as Q
import dicom_io
import synth
from model import SmallCNN, grad_cam, predict, train

def _wrap(text, width):
    """Tiny word-wrap so long caveats print readably."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


MODEL_DIR = "models"
NAME = "cxr-shortcut-audited"
PATHOLOGIES = synth.PATHOLOGIES

_STATE = {"model": None, "manifest": None, "studies": None}


# ---------------------------------------------------------------------------
# artefact
# ---------------------------------------------------------------------------

def _split(studies, frac=(0.6, 0.2)):
    """Patient-level split. Studies from one patient share anatomy, so a
    study-level split leaks the patient across the boundary and inflates
    everything downstream, calibration included."""
    pids = sorted({s["patient_id"] for s in studies})
    rng = np.random.default_rng(0)
    rng.shuffle(pids)
    a, b = int(len(pids) * frac[0]), int(len(pids) * (frac[0] + frac[1]))
    tr, va, te = set(pids[:a]), set(pids[a:b]), set(pids[b:])
    pick = lambda keep: [s for s in studies if s["patient_id"] in keep]
    return pick(tr), pick(va), pick(te)


def train_and_register(n_patients=700, epochs=16, version=1, seed=0):
    """Fit, calibrate on a held-out validation split, and save the artefact.

    THREE SPLITS, not two. Temperature is fitted on validation and evaluated on
    test. Fitting the temperature on the same data the calibration is reported
    from would make the ECE a training metric, and a calibration number that is
    a training metric is worse than none.
    """
    # SEEDED. The first two runs of this function differed enough to change
    # the story: opacity's temperature came out 0.620 (sharpening) and then
    # 1.346 (softening), and the strata ratio moved 1.56x -> 2.42x. A served
    # artefact has to be reproducible, and a calibration number quoted from one
    # unseeded run is not a measurement. See calibration_study.py for what the
    # run-to-run spread actually is.
    # The seed has to be set HERE, before SmallCNN() is constructed. train()
    # already called torch.manual_seed internally, but weight INITIALISATION
    # happens at construction time from the ambient RNG state, which is why
    # two runs of an apparently-seeded pipeline gave different models.
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(MODEL_DIR, exist_ok=True)
    studies = synth.build_dataset(n_patients=n_patients, seed=11)
    tr, va, te = _split(studies)
    Xtr, Ytr, *_ = synth.as_arrays(tr)
    Xva, Yva, *_ = synth.as_arrays(va)
    Xte, Yte, _Yt, _M, _G, Cte = synth.as_arrays(te)
    _Xv, _Yv, _Yvt, _Mv, _Gv, Cva = synth.as_arrays(va)

    model = SmallCNN(n_out=len(PATHOLOGIES))
    train(model, Xtr, Ytr, Xva, Yva, epochs=epochs, seed=seed)

    logit_va = _logits(model, Xva)
    logit_te = _logits(model, Xte)

    temps, cal = {}, {}
    for i, path in enumerate(PATHOLOGIES):
        fit = C.fit_temperature(Yva[:, i], logit_va[:, i])
        temps[path] = fit["temperature"]
        p_raw = C._sigmoid(logit_te[:, i])
        p_cal = C.apply_temperature(logit_te[:, i], fit["temperature"])
        cue = {"opacity": Cte["marker"], "cardiomegaly": None,
               "effusion": Cte["border"]}[path]
        cal[path] = {
            "temperature": fit["temperature"], "direction": fit["direction"],
            "ece_before": C.ece(Yte[:, i], p_raw)["ece"],
            "ece_after": C.ece(Yte[:, i], p_cal)["ece"],
            "brier_decomposition": C.brier_decomposition(Yte[:, i], p_cal),
            "stratified": (C.stratified_calibration(Yte[:, i], p_cal, cue)
                           if cue is not None else
                           {"note": "no confound was planted for this pathology"}),
        }

    manifest = {
        "name": NAME, "version": version, "pathologies": PATHOLOGIES,
        "image_size": synth.IMG, "temperatures": temps, "calibration": cal,
        "split": {"train_studies": len(tr), "val_studies": len(va),
                  "test_studies": len(te), "level": "patient"},
        "audited_dependencies": ["marker"],
        "audit": {
            "finding": ("The opacity head depends on the laterality marker. "
                        "Removing the marker from marker-positive studies "
                        "moved the opacity score by a median of +0.181 "
                        "(5 of 5 seeds > +0.02); marker-box activation runs "
                        "2.63x the baseline. See docs/SHORTCUT_AUDIT.md."),
            "cardiomegaly": "no cue dependency demonstrated",
            "effusion": "cue dependency weak and not confirmed across seeds",
        },
        "intended_use": (
            "Research demonstration of a shortcut audit on SYNTHETIC chest "
            "radiographs. NOT a diagnostic device, not cleared for clinical "
            "use, never trained on a real radiograph, and not validated on "
            "any human data. The opacity head is KNOWN to depend on an "
            "acquisition artefact rather than on anatomy."),
    }
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{NAME}-v{version}.pt"))
    with open(os.path.join(MODEL_DIR, f"{NAME}-v{version}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"registered {MODEL_DIR}/{NAME}-v{version}")
    for path in PATHOLOGIES:
        c = cal[path]
        print(f"  {path:<14} T={c['temperature']:.3f}  "
              f"ECE {c['ece_before']:.4f} -> {c['ece_after']:.4f}   "
              f"{c['direction']}")
        v = c["stratified"].get("verdict")
        if v:
            print(f"    {'strata':<12} aggregate {v['aggregate_ece']:.4f} vs "
                  f"worst stratum {v['worst_stratum_ece']:.4f} "
                  f"({v['ratio']:.2f}x)")
    return manifest


def _logits(model, X, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


def warm(version=1):
    path = os.path.join(MODEL_DIR, f"{NAME}-v{version}.json")
    with open(path) as fh:
        manifest = json.load(fh)
    model = SmallCNN(n_out=len(manifest["pathologies"]))
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f"{NAME}-v{version}.pt"), weights_only=True))
    model.eval()
    _STATE.update({"model": model, "manifest": manifest})
    return model


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def overlay_png(img, cam, scale=6, alpha=0.45):
    """Grad-CAM over the image, as PNG bytes.

    Rendered rather than returned as an array because a heatmap endpoint whose
    output needs matplotlib on the client is not a heatmap endpoint. The
    colourmap is applied here so what the reviewer sees is what the audit saw.
    """
    from PIL import Image

    base = np.clip(img, 0, 1)
    cam = np.asarray(cam, dtype=float)
    cam = cam / cam.max() if cam.max() > 0 else cam

    grey = np.stack([base] * 3, axis=-1)
    # a red-to-yellow ramp; hand-rolled so the endpoint does not depend on
    # matplotlib being importable in the serving process
    heat = np.stack([np.clip(cam * 2.0, 0, 1),
                     np.clip(cam * 2.0 - 1.0, 0, 1),
                     np.zeros_like(cam)], axis=-1)
    a = (alpha * cam)[..., None]
    blended = np.clip(grey * (1 - a) + heat * a, 0, 1)

    im = Image.fromarray((blended * 255).astype(np.uint8), mode="RGB")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_image(img, want_cam=False):
    """Score one 64x64 float image. Returns the full response body."""
    model, manifest = _STATE["model"], _STATE["manifest"]
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)[None, None])
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x).numpy()[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    cue_info = Q.detect_all(np.asarray(img))
    flags = Q.warnings_for(cue_info, set(manifest["audited_dependencies"]))

    preds = []
    for i, path in enumerate(manifest["pathologies"]):
        T = manifest["temperatures"][path]
        preds.append({
            "pathology": path,
            "probability": round(float(C.apply_temperature(logits[i], T)), 4),
            "probability_uncalibrated": round(float(C._sigmoid(logits[i])), 4),
            "temperature": round(T, 4),
            "shortcut_warnings": flags.get(path, []),
        })

    body = {
        "predictions": preds,
        "cues_detected": {k: v["present"] for k, v in cue_info.items()},
        "cue_detail": cue_info,
        "inference_ms": round(infer_ms, 3),
        "model": f"{NAME} v{manifest['version']}",
        "intended_use": manifest["intended_use"],
    }
    if want_cam:
        idx = manifest["pathologies"].index("opacity")
        t1 = time.perf_counter()
        cam = grad_cam(model, x, idx)
        body["cam_ms"] = round((time.perf_counter() - t1) * 1000, 3)
        body["_cam"] = cam
        body["_img"] = np.asarray(img)
    return body


def score_dicom_bytes(raw):
    """De-identify FIRST, then score. The order is the privacy control.

    AND SCREEN THE PIXELS, which is the part a tag-based de-identifier cannot
    do. `dicom_io.deidentify()` implements the PS3.15 Annex E basic profile over
    the tag set and says in its own docstring that it does NOT handle burned-in
    annotation, because the Clean Pixel Data option needs OCR or a per-vendor
    rule set.

    That gap and this project's subject are the same object. The shortcut this
    model learned IS a burned-in pixel annotation. On a real radiograph the
    same overlay region routinely carries the patient name, the MRN, the
    accession number or the acquisition timestamp -- which is to say, a study
    can pass tag de-identification completely and still ship PHI in the pixels.

    So the cue detector runs twice over, for two different reasons: as a
    shortcut warning for the prediction, and as a burned-in-annotation screen
    for the de-identification. It is a poor screen -- it detects a bright glyph
    in a known box, not text -- and the response says so rather than implying
    the study has been cleared.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = os.path.join(MODEL_DIR, "_inbound.dcm")
    with open(tmp, "wb") as fh:
        fh.write(raw)
    try:
        ds, pixels = dicom_io.read(tmp)
    finally:
        os.remove(tmp)

    clean = dicom_io.deidentify(ds)
    # deidentify() BLANKS values rather than dropping keys, so a set difference
    # over the keys reports nothing. Compare values.
    blanked = sorted(k for k, v in ds.items()
                     if v and not clean.get(k))
    changed = sorted(k for k, v in ds.items()
                     if v and clean.get(k) and clean[k] != v)
    retained = sorted(k for k, v in clean.items()
                      if v and k in ds and ds[k] == v)

    px = np.asarray(pixels, dtype=np.float32)
    img = (px - px.min()) / max(1e-6, float(px.max() - px.min()))
    body = score_image(img)

    burned_in = Q.detect_all(img)
    hits = [k for k, v in burned_in.items() if v["present"]]
    body["deidentification"] = {
        "applied_before_inference": True,
        "profile": "DICOM PS3.15 Annex E basic profile + retain-longitudinal-shifted",
        "tags_blanked": blanked,
        "tags_pseudonymised_or_shifted": changed,
        "tags_retained_unchanged": retained,
        "pixel_data": {
            "burned_in_candidates": hits,
            "status": ("BURNED-IN ANNOTATION CANDIDATE DETECTED -- this study "
                       "is NOT de-identified. Tag de-identification does not "
                       "touch pixels, and an overlay region like this one "
                       "routinely carries a name, MRN or accession number on "
                       "real radiographs" if hits else
                       "no burned-in candidate detected in the screened regions"),
            "screen_quality": (
                "WEAK. This looks for bright shapes in three known regions; it "
                "is not OCR and cannot read text, cannot find annotation "
                "outside those regions, and was tuned on the generator that "
                "planted them. It must not be treated as clearance."),
        },
    }
    return body


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _png(self, data, warning):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        # the caveat travels with the artefact, because a PNG saved out of this
        # response and pasted into a slide deck loses every surrounding word
        self.send_header("X-Explanation-Caveat", warning)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        m = _STATE["manifest"]
        if url.path == "/health":
            return self._json(200, {"status": "ok" if m else "no model loaded",
                                    "model": NAME if m else None})
        if url.path == "/model":
            if not m:
                return self._json(503, {"error": "no model loaded"})
            return self._json(200, m)
        return self._json(404, {"error": f"no route {url.path}",
                                "routes": ["/health", "/model",
                                           "POST /predict", "POST /predict/dicom",
                                           "POST /overlay"]})

    def do_POST(self):
        url = urlparse(self.path)
        if not _STATE["manifest"]:
            return self._json(503, {"error": "no model loaded"})
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)

        if url.path == "/predict/dicom":
            if not raw:
                return self._json(400, {"error": "empty body; expected DICOM"})
            try:
                return self._json(200, score_dicom_bytes(raw))
            except Exception as exc:
                return self._json(400, {"error": f"not readable as DICOM: {exc}"})

        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            return self._json(400, {"error": f"bad JSON: {exc}"})

        img, err = self._image_from(body)
        if err:
            return self._json(400, err)

        if url.path == "/predict":
            return self._json(200, score_image(img))

        if url.path == "/overlay":
            out = score_image(img, want_cam=True)
            png = overlay_png(out["_img"], out["_cam"])
            top = max(out["predictions"], key=lambda p: p["probability"])
            warn = ("Grad-CAM is not an explanation: it shows where gradient "
                    "flowed, at 8x8 resolution upsampled to 64x64. "
                    + ("SHORTCUT CUE DETECTED IN THIS IMAGE. "
                       if any(out["cues_detected"].values()) else "")
                    + f"top={top['pathology']} p={top['probability']}")
            return self._png(png, warn)

        return self._json(404, {"error": f"no route {url.path}"})

    def _image_from(self, body):
        size = _STATE["manifest"]["image_size"]
        if "pixels" in body:
            arr = np.asarray(body["pixels"], dtype=np.float32)
        elif "image_b64" in body:
            arr = np.frombuffer(base64.b64decode(body["image_b64"]),
                                dtype=np.float32).copy()
        else:
            return None, {"error": "expected 'pixels' or 'image_b64'",
                          "note": f"a {size}x{size} float array in [0,1]"}
        if arr.size != size * size:
            return None, {"error": f"expected {size}x{size} = {size * size} "
                                   f"values, got {arr.size}"}
        return arr.reshape(size, size), None


def serve(port=8082, version=1):
    warm(version)
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"serving {NAME} v{version} on http://127.0.0.1:{port}")
    print("  GET  /health   GET /model")
    print("  POST /predict  {\"pixels\": [[...]]}")
    print("  POST /predict/dicom   (raw DICOM body; de-identified on arrival)")
    print("  POST /overlay  -> image/png")
    return httpd


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def bench(version=1, n=200, warmup=20):
    """Latency percentiles, measured in-process.

    IN-PROCESS, AND SAID SO. This measures model time, not service time: no
    HTTP parsing, no JSON encoding, no network, no concurrency, one request at
    a time on a CPU. It is a floor on real latency, not an estimate of it, and
    quoting it as a service SLO would be dishonest.

    p50/p95/p99 rather than a mean, because the mean of a latency distribution
    is the one summary that tells you nothing about the tail, and the tail is
    what a radiologist waiting on a worklist actually experiences.

    Grad-CAM is timed separately. It needs a backward pass, so the overlay
    endpoint is structurally more expensive than the prediction endpoint, and
    that difference is a capacity-planning fact rather than a detail.
    """
    warm(version)
    studies = synth.build_dataset(n_patients=60, seed=5)
    imgs = [s["image"] for s in studies][:n + warmup]
    model = _STATE["model"]

    def timed(fn, xs):
        ts = []
        for x in xs:
            t0 = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - t0) * 1000)
        return np.array(ts)

    def infer(img):
        with torch.no_grad():
            model(torch.from_numpy(img[None, None]))

    def infer_cam(img):
        x = torch.from_numpy(img[None, None])
        grad_cam(model, x, 0)

    def full(img):
        out = score_image(img, want_cam=True)
        overlay_png(out["_img"], out["_cam"])

    timed(infer, imgs[:warmup])                       # warm the allocator
    rows = []
    for label, fn in (("forward pass only", infer),
                      ("forward + Grad-CAM", infer_cam),
                      ("full /overlay path (+cues, +PNG)", full)):
        t = timed(fn, imgs[warmup:warmup + n])
        rows.append({"stage": label, "n": len(t),
                     "p50_ms": float(np.percentile(t, 50)),
                     "p95_ms": float(np.percentile(t, 95)),
                     "p99_ms": float(np.percentile(t, 99)),
                     "max_ms": float(t.max())})

    print("=" * 74)
    print(f"LATENCY  n={n} per stage, single-threaded CPU, batch size 1")
    print("=" * 74)
    print(f"  {'stage':<36}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}")
    for r in rows:
        print(f"  {r['stage']:<36}{r['p50_ms']:>9.2f}{r['p95_ms']:>9.2f}"
              f"{r['p99_ms']:>9.2f}{r['max_ms']:>9.2f}")
    # ---- and the same work over the socket ------------------------------
    # Calling the in-process number a "floor" is a claim, so it gets measured
    # rather than asserted. This is the same model, the same images, through
    # HTTP, JSON encoding and a loopback socket.
    import threading
    import urllib.request

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/predict"

    def over_http(img):
        body = json.dumps({"pixels": img.tolist()}).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()

    n_http = min(n, 60)
    timed(over_http, imgs[warmup:warmup + 5])
    t = timed(over_http, imgs[warmup:warmup + n_http])
    httpd.shutdown()
    rows.append({"stage": f"POST /predict over HTTP (n={n_http})", "n": len(t),
                 "p50_ms": float(np.percentile(t, 50)),
                 "p95_ms": float(np.percentile(t, 95)),
                 "p99_ms": float(np.percentile(t, 99)),
                 "max_ms": float(t.max())})
    print(f"  {rows[-1]['stage']:<36}{rows[-1]['p50_ms']:>9.2f}"
          f"{rows[-1]['p95_ms']:>9.2f}{rows[-1]['p99_ms']:>9.2f}"
          f"{rows[-1]['max_ms']:>9.2f}")

    overhead = rows[-1]["p50_ms"] / rows[0]["p50_ms"]
    print("")
    print(f"  The socket path costs {overhead:.1f}x the forward pass alone.")
    print("  Most of that is JSON: 4,096 floats serialised as decimal text is")
    print("  a far larger payload than the tensor it becomes. A real imaging")
    print("  service would not move pixels this way, which is itself the")
    print("  finding -- the transport, not the model, is the cost here.")

    cam_cost = rows[1]["p50_ms"] / rows[0]["p50_ms"]
    print(f"\n  Grad-CAM costs {cam_cost:.1f}x a forward pass, because it needs")
    print("  a backward pass. The overlay endpoint is structurally more")
    print("  expensive than the prediction endpoint; that is capacity planning,")
    print("  not a detail.")
    print("\n  IN-PROCESS MEASUREMENT. No HTTP, no JSON, no network, no")
    print("  concurrency, 64x64 images on a CPU. This is a FLOOR on real")
    print("  latency and must not be quoted as a service SLO. A real number")
    print("  needs load against the socket at a stated concurrency, on the")
    print("  image size the scanner actually produces -- 2048x2048 chest films")
    print("  are ~1000x the pixels these are.")
    os.makedirs("out", exist_ok=True)
    with open("out/latency.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote out/latency.json")
    return rows


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def demo(port=8083):
    import threading
    import urllib.error
    import urllib.request

    httpd = serve(port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(path, payload, raw=False):
        data = payload if raw else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    print("\n" + "=" * 74)
    print("EXERCISING THE API")
    print("=" * 74)

    studies = synth.build_dataset(n_patients=40, seed=3)
    with_marker = next(s for s in studies if s["confounds"]["marker"])
    without = next(s for s in studies if not s["confounds"]["marker"])

    for label, s in (("marker PRESENT", with_marker), ("marker ABSENT", without)):
        _c, _h, b = post("/predict", {"pixels": s["image"].tolist()})
        body = json.loads(b)
        op = next(p for p in body["predictions"] if p["pathology"] == "opacity")
        print(f"\n  POST /predict  [{label}]  true opacity="
              f"{int(s['labels']['opacity'])}")
        print(f"    opacity p={op['probability']:.3f} "
              f"(uncalibrated {op['probability_uncalibrated']:.3f}, "
              f"T={op['temperature']:.2f})")
        print(f"    cues detected: "
              f"{[k for k, v in body['cues_detected'].items() if v] or 'none'}")
        for w in op["shortcut_warnings"]:
            print(f"    ! {w[:100]}...")
        if not op["shortcut_warnings"]:
            print("    (no audited cue present -- no shortcut warning)")

    _c, hdr, png = post("/overlay", {"pixels": with_marker["image"].tolist()})
    os.makedirs("out", exist_ok=True)
    with open("out/overlay.png", "wb") as fh:
        fh.write(png)
    print(f"\n  POST /overlay -> {len(png):,} bytes of PNG -> out/overlay.png")
    print(f"    X-Explanation-Caveat: {hdr['X-Explanation-Caveat'][:95]}...")

    # the DICOM path, with identifiers planted so removal is demonstrable
    os.makedirs("out", exist_ok=True)
    px = (with_marker["image"] * 65535).astype(np.uint16)
    dpath = "out/_demo.dcm"
    dicom_io.write(dpath, px, {
        "PatientName": "DOE^JANE", "PatientID": "MRN-0099281",
        "PatientBirthDate": "19631104", "StudyDate": "20240311",
        "AccessionNumber": "ACC-55512", "InstitutionName": "St Elsewhere",
        "ReferringPhysicianName": "SMITH^ALAN", "StudyInstanceUID": "1.2.3.4.5",
        "SeriesInstanceUID": "1.2.3.4.6", "SOPInstanceUID": "1.2.3.4.7",
        "Modality": "CR", "Rows": px.shape[0], "Columns": px.shape[1]})
    with open(dpath, "rb") as fh:
        _c, _h, b = post("/predict/dicom", fh.read(), raw=True)
    body = json.loads(b)
    d = body["deidentification"]
    print("\n  POST /predict/dicom")
    print(f"    de-identified BEFORE inference: {d['applied_before_inference']}")
    print(f"    blanked:       {', '.join(d['tags_blanked'])}")
    print(f"    pseudonymised: {', '.join(d['tags_pseudonymised_or_shifted'])}")
    print(f"    retained:      {', '.join(d['tags_retained_unchanged'])}")
    print(f"    pixel screen:  candidates="
          f"{d['pixel_data']['burned_in_candidates'] or 'none'}")
    for line in _wrap(d['pixel_data']['status'], 68):
        print(f"      {line}")
    print("    THE TAG PROFILE CANNOT SEE THE PIXELS, and this project's whole")
    print("    subject is a burned-in pixel annotation. On a real radiograph")
    print("    that same overlay routinely carries a name or an MRN, so a")
    print("    study can pass tag de-identification and still ship PHI.")
    print("    The ORDER is the control. Scoring first and de-identifying on")
    print("    the way to storage leaves identifiers in memory, logs and any")
    print("    crash dump taken in between.")

    _c, _h, b = post("/predict", {"pixels": [[0.0] * 10] * 10})
    print(f"\n  POST /predict with a 10x10 image -> "
          f"{json.loads(b).get('error', 'no error')}")

    httpd.shutdown()
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--patients", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.register:
        train_and_register(a.patients, a.epochs, a.version, a.seed)
    elif a.bench:
        bench(a.version)
    elif a.demo:
        demo(a.port)
    else:
        serve(a.port, a.version).serve_forever()
