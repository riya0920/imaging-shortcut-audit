"""Minimal DICOM Part 10 writer/reader and a PS3.15 Annex E de-identifier.

pydicom is not installed in this offline build, so this is hand-rolled against
the standard: a 128-byte preamble, the "DICM" magic, File Meta group 0002 in
Explicit VR Little Endian, then the dataset. That is enough to write a file
another DICOM tool would open, and enough to demonstrate the thing that
actually matters for healthcare work -- that you know PHI lives in the HEADER,
not only in the pixels, and you know which tags it lives in.

WHAT THIS IS NOT: a DICOM implementation. No sequences (SQ), no compressed
transfer syntaxes, no VR edge cases, no conformance statement. Real work uses
pydicom or dcmtk. The point of hand-rolling it here is that the de-identifier
below then has something concrete to operate on rather than being a paragraph.

DE-IDENTIFICATION
-----------------
`deidentify()` implements the DICOM PS3.15 Annex E **Basic Application Level
Confidentiality Profile**, in its most common configuration: Basic profile plus
Retain Longitudinal Temporal Information with Modified Dates (dates shifted by a
consistent per-patient offset rather than removed, because intervals carry the
clinical meaning) and Clean Pixel Data left OUT -- burned-in annotation is NOT
handled and the code says so at the call site.

Two properties a reviewer should check, both tested in tests/:
  * every tag in the Basic Profile removal list is absent afterwards, and the
    test enumerates the list rather than spot-checking three tags;
  * PatientID is pseudonymised consistently -- the same patient maps to the
    same replacement, or longitudinal analysis silently breaks.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

# (group, element): (VR, keyword)
TAGS = {
    (0x0008, 0x0016): ("UI", "SOPClassUID"),
    (0x0008, 0x0018): ("UI", "SOPInstanceUID"),
    (0x0008, 0x0020): ("DA", "StudyDate"),
    (0x0008, 0x0021): ("DA", "SeriesDate"),
    (0x0008, 0x0030): ("TM", "StudyTime"),
    (0x0008, 0x0050): ("SH", "AccessionNumber"),
    (0x0008, 0x0060): ("CS", "Modality"),
    (0x0008, 0x0080): ("LO", "InstitutionName"),
    (0x0008, 0x0081): ("ST", "InstitutionAddress"),
    (0x0008, 0x0090): ("PN", "ReferringPhysicianName"),
    (0x0008, 0x1010): ("SH", "StationName"),
    (0x0008, 0x1030): ("LO", "StudyDescription"),
    (0x0008, 0x1048): ("PN", "PhysiciansOfRecord"),
    (0x0008, 0x1050): ("PN", "PerformingPhysicianName"),
    (0x0008, 0x1070): ("PN", "OperatorsName"),
    (0x0010, 0x0010): ("PN", "PatientName"),
    (0x0010, 0x0020): ("LO", "PatientID"),
    (0x0010, 0x0030): ("DA", "PatientBirthDate"),
    (0x0010, 0x0040): ("CS", "PatientSex"),
    (0x0010, 0x1000): ("LO", "OtherPatientIDs"),
    (0x0010, 0x1010): ("AS", "PatientAge"),
    (0x0010, 0x1040): ("LO", "PatientAddress"),
    (0x0010, 0x2154): ("SH", "PatientTelephoneNumbers"),
    (0x0010, 0x4000): ("LT", "PatientComments"),
    (0x0018, 0x1000): ("LO", "DeviceSerialNumber"),
    (0x0018, 0x5100): ("CS", "PatientPosition"),
    (0x0020, 0x000D): ("UI", "StudyInstanceUID"),
    (0x0020, 0x000E): ("UI", "SeriesInstanceUID"),
    (0x0020, 0x0010): ("SH", "StudyID"),
    (0x0020, 0x0013): ("IS", "InstanceNumber"),
    (0x0028, 0x0002): ("US", "SamplesPerPixel"),
    (0x0028, 0x0004): ("CS", "PhotometricInterpretation"),
    (0x0028, 0x0010): ("US", "Rows"),
    (0x0028, 0x0011): ("US", "Columns"),
    (0x0028, 0x0100): ("US", "BitsAllocated"),
    (0x0028, 0x0101): ("US", "BitsStored"),
    (0x0028, 0x0102): ("US", "HighBit"),
    (0x0028, 0x0103): ("US", "PixelRepresentation"),
    (0x0028, 0x1050): ("DS", "WindowCenter"),
    (0x0028, 0x1051): ("DS", "WindowWidth"),
    (0x0012, 0x0062): ("CS", "PatientIdentityRemoved"),
    (0x0012, 0x0063): ("LO", "DeidentificationMethod"),
    (0x7FE0, 0x0010): ("OW", "PixelData"),
}
BY_KEYWORD = {kw: (g, e) for (g, e), (_vr, kw) in TAGS.items()}

# PS3.15 Annex E Basic Profile: tags whose values must be removed (Z/X actions).
# Abbreviated to the tags this writer can emit -- a production implementation
# walks the full Annex E table (several hundred rows) plus private tags.
BASIC_PROFILE_REMOVE = [
    "AccessionNumber", "InstitutionName", "InstitutionAddress",
    "ReferringPhysicianName", "StationName", "StudyDescription",
    "PhysiciansOfRecord", "PerformingPhysicianName", "OperatorsName",
    "PatientName", "PatientBirthDate", "OtherPatientIDs", "PatientAddress",
    "PatientTelephoneNumbers", "PatientComments", "DeviceSerialNumber",
    "StudyID",
]
# Retained deliberately, with the reason. Being able to say WHY a tag stays is
# the difference between a de-identifier and a delete key.
RETAINED = {
    "PatientSex": "clinically necessary; not an identifier under Safe Harbor",
    "PatientAge": "retained only where <90; see cap in deidentify()",
    "Modality": "acquisition metadata, no PHI",
    "PatientPosition": "acquisition metadata; also the confound this project audits",
    "StudyDate": "SHIFTED, not removed -- intervals carry clinical meaning",
}

_EXPLICIT_SHORT = {"AE", "AS", "AT", "CS", "DA", "DS", "DT", "FL", "FD", "IS",
                   "LO", "LT", "PN", "SH", "SL", "SS", "ST", "TM", "UI", "UL",
                   "US"}


def _encode_element(group, elem, vr, value):
    if vr == "US":
        raw = struct.pack("<H", int(value))
    elif vr == "OW":
        raw = value.tobytes()
    else:
        raw = str(value).encode("ascii", "replace")
        if len(raw) % 2:
            raw += b" " if vr != "UI" else b"\x00"
    out = struct.pack("<HH", group, elem) + vr.encode()
    if vr in _EXPLICIT_SHORT:
        out += struct.pack("<H", len(raw))
    else:
        out += b"\x00\x00" + struct.pack("<I", len(raw))
    return out + raw


def write(path, pixels_u16, meta):
    """Write a Part 10 file. `meta` maps keyword -> value."""
    rows, cols = pixels_u16.shape
    ds = dict(meta)
    ds.update({
        "SamplesPerPixel": 1, "PhotometricInterpretation": "MONOCHROME2",
        "Rows": rows, "Columns": cols, "BitsAllocated": 16, "BitsStored": 16,
        "HighBit": 15, "PixelRepresentation": 0, "Modality": "CR",
    })
    body = b""
    for kw in sorted(ds, key=lambda k: BY_KEYWORD[k]):
        g, e = BY_KEYWORD[kw]
        body += _encode_element(g, e, TAGS[(g, e)][0], ds[kw])
    body += _encode_element(0x7FE0, 0x0010, "OW", pixels_u16.astype("<u2"))

    # File Meta group (0002), Explicit VR Little Endian
    fm = b""
    fm += _encode_element(0x0002, 0x0002, "UI", "1.2.840.10008.5.1.4.1.1.1")
    fm += _encode_element(0x0002, 0x0003, "UI", ds.get("SOPInstanceUID", "1.2.3"))
    fm += _encode_element(0x0002, 0x0010, "UI", "1.2.840.10008.1.2.1")
    head = _encode_element(0x0002, 0x0000, "UL", len(fm)) if False else (
        struct.pack("<HH", 0x0002, 0x0000) + b"UL" + struct.pack("<H", 4)
        + struct.pack("<I", len(fm)))

    with open(path, "wb") as fh:
        fh.write(b"\x00" * 128 + b"DICM" + head + fm + body)
    return path


def read(path):
    """Parse a file written by `write`. Returns (dataset dict, pixel array)."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[128:132] != b"DICM":
        raise ValueError("not a Part 10 file: DICM magic missing")
    i, ds, pixels = 132, {}, None
    while i < len(buf) - 7:
        g, e = struct.unpack_from("<HH", buf, i)
        vr = buf[i + 4:i + 6].decode("ascii", "replace")
        if vr in _EXPLICIT_SHORT:
            (ln,) = struct.unpack_from("<H", buf, i + 6)
            i += 8
        else:
            (ln,) = struct.unpack_from("<I", buf, i + 8)
            i += 12
        raw = buf[i:i + ln]
        i += ln
        if (g, e) == (0x7FE0, 0x0010):
            pixels = np.frombuffer(raw, dtype="<u2")
            continue
        if g == 0x0002:
            continue
        kw = TAGS.get((g, e), (None, None))[1]
        if kw:
            if vr == "US":
                ds[kw] = struct.unpack("<H", raw)[0]
            else:
                ds[kw] = raw.decode("ascii", "replace").strip()
    if pixels is not None and "Rows" in ds:
        pixels = pixels.reshape(ds["Rows"], ds["Columns"])
    return ds, pixels


# ---------------------------------------------------------------------------
def _pseudonymise(value, salt):
    return "DEID-" + hashlib.sha256((salt + str(value)).encode()).hexdigest()[:12].upper()


def _shift_date(datestr, days):
    """DICOM DA is YYYYMMDD. Shift, do not delete: intervals are the analysis."""
    import datetime
    try:
        d = datetime.datetime.strptime(str(datestr), "%Y%m%d").date()
    except ValueError:
        return ""
    return (d + datetime.timedelta(days=days)).strftime("%Y%m%d")


def patient_offset(patient_id, salt, max_days=365):
    """Deterministic per-patient shift, so every study for one patient moves by
    the same amount and within-patient intervals survive."""
    h = hashlib.sha256((salt + str(patient_id)).encode()).digest()
    return -(int.from_bytes(h[:4], "big") % max_days) - 1


def deidentify(ds, salt="project-salt", retain_dates_shifted=True):
    """PS3.15 Annex E Basic Profile + Retain Longitudinal Temporal (Modified).

    NOT handled, deliberately and loudly: burned-in annotation in the pixel
    data. The Clean Pixel Data option requires OCR or a per-vendor rule set,
    and a de-identifier that silently ignores it is worse than one that says
    it does not do it. Callers must screen for burned-in text separately.
    """
    out = dict(ds)
    original_pid = out.get("PatientID", "")

    for kw in BASIC_PROFILE_REMOVE:
        if kw in out:
            out[kw] = ""

    if original_pid:
        out["PatientID"] = _pseudonymise(original_pid, salt)

    if retain_dates_shifted:
        off = patient_offset(original_pid, salt)
        for kw in ("StudyDate", "SeriesDate"):
            if out.get(kw):
                out[kw] = _shift_date(out[kw], off)
    else:
        for kw in ("StudyDate", "SeriesDate"):
            out[kw] = ""
    out.pop("StudyTime", None)

    # Safe Harbor: ages over 89 must be aggregated
    age = str(out.get("PatientAge", ""))
    if age[:3].isdigit() and int(age[:3]) >= 90:
        out["PatientAge"] = "090Y+"

    # UIDs are identifiers; remap consistently so relationships survive
    for kw in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        if out.get(kw):
            out[kw] = "2.25." + str(int(hashlib.sha256(
                (salt + out[kw]).encode()).hexdigest()[:16], 16))

    out["PatientIdentityRemoved"] = "YES"
    out["DeidentificationMethod"] = "PS3.15 Annex E Basic; dates shifted per-patient"
    return out
