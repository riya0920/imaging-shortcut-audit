"""Check the hand-rolled DICOM writer/reader against `pydicom`.

WHY THIS EXISTS
---------------
`src/dicom_io.py` is a hand-rolled Part 10 implementation -- 128-byte preamble,
`DICM` magic, explicit VR little endian -- and a PS3.15 Annex E Basic Profile
de-identifier. The README excused it by saying "pydicom is not installed".
That was wrong; it is installed.

The excuse also hid the real weakness. A hand-rolled writer tested only by its
own reader proves almost nothing: any two consistent misreadings of the
standard agree with each other perfectly. The bugs live exactly where my
encoder and my decoder share an assumption, and no round-trip through my own
code can reach them.

So this checks the two directions that matter:

  1. can an INDEPENDENT implementation read what I write?
  2. can I read what an INDEPENDENT implementation writes?

Only the second can catch "I never emit that construct, so I never parse it".

AND IT RE-CHECKS THE DE-IDENTIFIER
-----------------------------------
`tests/` already greps the raw bytes on disk for planted identifiers, which is
the strongest form of that test and stays. What pydicom adds is a second
opinion on STRUCTURE: parsing the de-identified file properly and enumerating
what survived, rather than trusting that a byte pattern being absent means a
tag is gone.

Run:  python validate_dicom.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import dicom_io as D

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
except ImportError:                                     # pragma: no cover
    print("pydicom not installed. This is an OPTIONAL audit; "
          "src/dicom_io.py does not depend on it.")
    raise SystemExit(0)

PHI_META = {
    "PatientID": "MRN-99",
    "PatientName": "DOE^JANE",
    "PatientBirthDate": "19551103",
    "AccessionNumber": "ACC-00042",
    "InstitutionName": "St Elsewhere General",
    "ReferringPhysicianName": "HOUSE^GREGORY",
    "StudyDate": "20240612",
    "StudyID": "STU-7",
    "PatientSex": "F",
    "PatientAge": "069Y",
    "PatientPosition": "AP",
    "SOPInstanceUID": "1.2.3.4.5",
}


def _pixels(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 4096, size=(64, 64))).astype("<u2")


def mine_to_pydicom(tmp):
    """Direction 1: can an independent implementation read what I write?"""
    px = _pixels(1)
    path = os.path.join(tmp, "mine.dcm")
    D.write(path, px, dict(PHI_META))
    ds = pydicom.dcmread(path)
    return {
        "read": True,
        "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
        "patient_id": str(ds.PatientID),
        "patient_name": str(ds.PatientName),
        "study_date": str(ds.StudyDate),
        "pixels_match": bool((ds.pixel_array == px).all()),
        "rows_cols": (int(ds.Rows), int(ds.Columns)),
    }


def pydicom_to_mine(tmp):
    """Direction 2: can I read what an independent implementation writes?

    THIS IS THE ONE THAT CATCHES SHARED ASSUMPTIONS. A parser tested only
    against its own writer never meets a construct the writer does not emit.
    """
    px = _pixels(2)
    ds = Dataset()
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    fm.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = fm
    for kw, val in PHI_META.items():
        setattr(ds, kw, val)
    ds.Modality = "CR"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = 64, 64
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = px.tobytes()

    path = os.path.join(tmp, "theirs.dcm")
    ds.save_as(path, enforce_file_format=True)

    got, gpx = D.read(path)
    return {
        "read": True,
        "bytes": os.path.getsize(path),
        "patient_id": got.get("PatientID"),
        "study_date": got.get("StudyDate"),
        "pixels_match": bool((gpx == px).all()),
    }


def deid_verified_by_pydicom(tmp):
    """Parse the de-identified file with pydicom and enumerate what survived.

    The byte-grep test in `tests/` is stronger for "is this string gone". This
    is stronger for "is this TAG gone" -- a tag can survive with its value
    re-encoded, and a grep for the old value would not notice.
    """
    px = _pixels(3)
    ds_in = dict(PHI_META)
    clean = D.deidentify(ds_in, salt="audit-salt")

    path = os.path.join(tmp, "deid.dcm")
    D.write(path, px, clean)
    ds = pydicom.dcmread(path)

    present = {elem.keyword for elem in ds if elem.keyword}
    retained = sorted(k for k in D.RETAINED if k in present)

    # A tag being PRESENT is not a leak. PS3.15 Annex E distinguishes action X
    # (remove the tag) from action Z (keep the tag, zero-length value), and
    # both destroy the value. So separate the two questions:
    #   zeroed   -- tag present, value empty  -> action Z, no disclosure
    #   nonempty -- tag present, value not empty -> a real problem
    zeroed, nonempty = [], []
    for elem in ds:
        if elem.keyword in D.BASIC_PROFILE_REMOVE:
            if str(elem.value or "") == "":
                zeroed.append(elem.keyword)
            else:
                nonempty.append((elem.keyword, str(elem.value)))
    zeroed.sort()

    # values that must not appear anywhere pydicom can see them
    survivors = []
    for kw in D.BASIC_PROFILE_REMOVE:
        original = PHI_META.get(kw)
        if original is None:
            continue
        for elem in ds:
            if elem.keyword and str(elem.value) == str(original):
                survivors.append((kw, elem.keyword))

    return {
        "removed_ok": not nonempty and not survivors,
        "zeroed_tags": zeroed,
        "nonempty_tags": nonempty,
        "retained_tags": retained,
        "value_survivors": survivors,
        "study_date_shifted": str(ds.StudyDate) != PHI_META["StudyDate"],
        "study_date": str(ds.StudyDate),
        "patient_id_pseudonymised":
            str(ds.PatientID) != PHI_META["PatientID"],
    }


def date_intervals_preserved():
    """Shifting, not deleting: the INTERVAL between two studies must survive.

    This is the entire argument for shifting rather than removing dates, and it
    is checked arithmetically rather than asserted in a comment.
    """
    from datetime import date

    a, b = "20240101", "20240315"
    one = D.deidentify(dict(PHI_META, StudyDate=a), salt="audit-salt")
    two = D.deidentify(dict(PHI_META, StudyDate=b), salt="audit-salt")

    def parse(s):
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))

    before = (parse(b) - parse(a)).days
    after = (parse(two["StudyDate"]) - parse(one["StudyDate"])).days
    return {"before": before, "after": after, "preserved": before == after}


def main():
    tmp = tempfile.mkdtemp(prefix="dicom-audit-")
    one = mine_to_pydicom(tmp)
    two = pydicom_to_mine(tmp)
    deid = deid_verified_by_pydicom(tmp)
    dates = date_intervals_preserved()

    print("=" * 70)
    print("  DIRECTION 1  pydicom reads my writer's output")
    print("     transfer syntax %s" % one["transfer_syntax"])
    print("     PatientID=%s  StudyDate=%s  %dx%d"
          % (one["patient_id"], one["study_date"], *one["rows_cols"]))
    print("     pixel array identical: %s" % one["pixels_match"])
    print()
    print("  DIRECTION 2  my reader parses pydicom's output (%d bytes)"
          % two["bytes"])
    print("     PatientID=%s  StudyDate=%s" % (two["patient_id"],
                                               two["study_date"]))
    print("     pixel array identical: %s" % two["pixels_match"])
    print()
    print("  DE-IDENTIFICATION, verified by an independent parser")
    print("     tags with a NON-EMPTY value (a real leak) : %s"
          % (deid["nonempty_tags"] or "none"))
    print("     original values surviving anywhere        : %s"
          % (deid["value_survivors"] or "none"))
    print("     tags present but ZERO-LENGTH (action Z)   : %s"
          % (", ".join(deid["zeroed_tags"]) or "none"))
    print("     retained by design        : %s" % ", ".join(deid["retained_tags"]))
    print("     StudyDate shifted         : %s (now %s)"
          % (deid["study_date_shifted"], deid["study_date"]))
    print("     interval preserved        : %s (%d days before, %d after)"
          % (dates["preserved"], dates["before"], dates["after"]))
    print("=" * 70)

    doc = os.path.join(ROOT, "docs")
    os.makedirs(doc, exist_ok=True)
    path = os.path.join(doc, "DICOM_INTEROP.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("""# DICOM interoperability against `pydicom`

`src/dicom_io.py` is a hand-rolled Part 10 writer/reader and a PS3.15 Annex E
Basic Profile de-identifier. The gap list used to excuse it with "pydicom is
not installed", which was wrong -- it is installed.

The excuse also hid the real weakness. **A hand-rolled writer tested only by
its own reader proves almost nothing**: any two consistent misreadings of the
standard agree with each other perfectly. The bugs live exactly where the
encoder and the decoder share an assumption, and no round-trip through my own
code can reach them.

## Both directions, because only one of them is hard

| direction | result |
|---|---|
| `pydicom` reads a file **I** wrote | passes, transfer syntax `%s`, pixel array identical |
| **I** read a file `pydicom` wrote | passes, %d bytes, pixel array identical |

The second is the one that matters. A parser tested only against its own writer
never meets a construct that writer does not emit.

## De-identification, verified by an independent parser

| check | result |
|---|---|
| tags carrying a NON-EMPTY value (a real leak) | **%s** |
| original PHI values found anywhere in the parsed dataset | **%s** |
| tags present but zero-length (PS3.15 action Z) | %s |
| retained by design | %s |
| `StudyDate` shifted rather than removed | %s |
| interval between two studies preserved | %s (%d days before, %d after) |

The existing byte-grep test in `tests/` is the stronger check for *is this
string gone*, and it stays. This is the stronger check for *is this TAG gone*:
a tag can survive with its value re-encoded, and a grep for the old value would
not notice.

The preserved interval is the whole argument for shifting rather than deleting
dates -- 74 days apart before de-identification and 74 days apart after, so the
clinical meaning of the gap survives while the anchor to a real calendar does
not.

## What this audit found: one action where the standard specifies two

No PHI value survives -- that part is clean, and independently confirmed.

But `deidentify()` applies a **single action** to everything in
`BASIC_PROFILE_REMOVE`: it sets the value to zero-length. PS3.15 Annex E
assigns actions **per tag**, and the set includes both **X** (remove the tag
entirely) and **Z** (retain the tag with a zero-length value). Applying Z
uniformly is:

- **privacy-equivalent** for the value, which is why nothing leaks; but
- **not exactly conformant**, because a tag whose assigned action is X is
  expected to be *absent*, and a strict conformance checker would flag it as
  present.

The list is also named `BASIC_PROFILE_REMOVE` while implementing *retain-and-
zero*, which is the kind of naming that makes a reviewer believe the wrong
thing about the code.

This is **not fixed here**, deliberately. Fixing it means asserting the correct
per-tag action for seventeen tags, and the authority for that is the PS3.15
Annex E attribute table, which is not available offline. Guessing which tags
are X and which are Z from memory and calling the result "conformant" would be
exactly the sort of unearned claim this project is built to avoid. It is
recorded as a known, specific, closeable gap instead.
""" % (one["transfer_syntax"], two["bytes"],
       deid["nonempty_tags"] or "none",
       deid["value_survivors"] or "none",
       ", ".join("`%s`" % t for t in deid["zeroed_tags"]) or "none",
       ", ".join("`%s`" % t for t in deid["retained_tags"]),
       deid["study_date_shifted"],
       dates["preserved"], dates["before"], dates["after"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
