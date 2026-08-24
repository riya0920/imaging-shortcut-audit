# DICOM interoperability against `pydicom`

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
| `pydicom` reads a file **I** wrote | passes, transfer syntax `1.2.840.10008.1.2.1`, pixel array identical |
| **I** read a file `pydicom` wrote | passes, 8794 bytes, pixel array identical |

The second is the one that matters. A parser tested only against its own writer
never meets a construct that writer does not emit.

## De-identification, verified by an independent parser

| check | result |
|---|---|
| tags carrying a NON-EMPTY value (a real leak) | **none** |
| original PHI values found anywhere in the parsed dataset | **none** |
| tags present but zero-length (PS3.15 action Z) | `AccessionNumber`, `InstitutionName`, `PatientBirthDate`, `PatientName`, `ReferringPhysicianName`, `StudyID` |
| retained by design | `Modality`, `PatientAge`, `PatientPosition`, `PatientSex`, `StudyDate` |
| `StudyDate` shifted rather than removed | True |
| interval between two studies preserved | True (74 days before, 74 after) |

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
