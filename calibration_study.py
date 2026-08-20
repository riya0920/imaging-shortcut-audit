"""Does the calibration finding reproduce, or was it one lucky run?

WHY THIS EXISTS
---------------
The stratified calibration result was written up from a single training run:
opacity aggregate ECE 0.127 against a worst-stratum 0.198, a ratio of 1.56x.
Re-running the same function produced 0.111 against 0.269, a ratio of 2.42x --
and the fitted temperature flipped direction, from 0.620 (sharpening
underconfident scores) to 1.346 (softening overconfident ones).

Two runs, two different stories about what post-hoc calibration is doing. That
is not a finding, it is a draw from a distribution, and quoting either number
as "the" calibration of this model would be exactly the kind of single-run
result this project exists to be sceptical of.

`seed_study.py` already establishes the pattern for the audit claims. This does
the same for the calibration claim, and asks the only question worth asking:

    does the ordering hold every time?

The magnitude will move. The claim being tested is the DIRECTION -- that the
aggregate ECE understates the worst stratum, for the pathology the audit found
a shortcut dependency in, and not for the one it did not.

Run:  python calibration_study.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import calibrate as C
import synth
from model import SmallCNN, train
from serve import _logits, _split


def one_seed(seed, n_patients=700, epochs=16):
    torch.manual_seed(seed)
    np.random.seed(seed)
    studies = synth.build_dataset(n_patients=n_patients, seed=11 + seed)
    tr, va, te = _split(studies)
    Xtr, Ytr, *_ = synth.as_arrays(tr)
    Xva, Yva, *_ = synth.as_arrays(va)
    Xte, Yte, _Yt, _M, _G, Cte = synth.as_arrays(te)

    model = SmallCNN(n_out=len(synth.PATHOLOGIES))
    train(model, Xtr, Ytr, Xva, Yva, epochs=epochs, seed=seed, verbose=False)
    zva, zte = _logits(model, Xva), _logits(model, Xte)

    out = {"seed": seed}
    for i, path in enumerate(synth.PATHOLOGIES):
        cue = {"opacity": Cte["marker"], "cardiomegaly": None,
               "effusion": Cte["border"]}[path]
        fit = C.fit_temperature(Yva[:, i], zva[:, i])
        p = C.apply_temperature(zte[:, i], fit["temperature"])
        row = {"temperature": fit["temperature"],
               "ece_before": C.ece(Yte[:, i], C._sigmoid(zte[:, i]))["ece"],
               "ece_after": C.ece(Yte[:, i], p)["ece"]}
        if cue is not None:
            st = C.stratified_calibration(Yte[:, i], p, cue, n_bins=6)
            row.update({
                "aggregate_ece": st["aggregate"]["ece"],
                "cue_present_ece": st["cue_present"].get("ece"),
                "cue_absent_ece": st["cue_absent"].get("ece"),
                "worst_ece": st["verdict"]["worst_stratum_ece"],
                "ratio": st["verdict"]["ratio"],
                "absolute_gap": st["verdict"]["absolute_gap"],
                "hides": "hides" in st["verdict"]["reading"],
                "worse_stratum": ("cue_present"
                                  if st["cue_present"].get("ece", 0)
                                  >= st["cue_absent"].get("ece", 0)
                                  else "cue_absent"),
            })
        out[path] = row
    return out


def main(seeds=5, n_patients=700, epochs=16):
    rows = []
    for s in range(seeds):
        print(f"  seed {s} ...", flush=True)
        rows.append(one_seed(s, n_patients, epochs))

    print("\n" + "=" * 78)
    print(f"CALIBRATION ACROSS {seeds} SEEDS")
    print("=" * 78)

    for path in ("opacity", "effusion"):
        rs = [r[path] for r in rows if "aggregate_ece" in r[path]]
        if not rs:
            continue
        print(f"\n{path.upper()}"
              f"   (audit: {'shortcut CONFIRMED' if path == 'opacity' else 'no dependency confirmed'})")
        print(f"  {'seed':<6}{'T':>7}{'ECE all':>10}{'cue+':>9}{'cue-':>9}"
              f"{'ratio':>8}{'gap':>8}  verdict")
        for r, row in zip(rows, rs):
            print(f"  {r['seed']:<6}{row['temperature']:>7.3f}"
                  f"{row['aggregate_ece']:>10.4f}{row['cue_present_ece']:>9.4f}"
                  f"{row['cue_absent_ece']:>9.4f}{row['ratio']:>8.2f}"
                  f"{row['absolute_gap']:>8.4f}  "
                  f"{'HIDES' if row['hides'] else 'no split'}")
        ratios = [r["ratio"] for r in rs]
        hides = sum(r["hides"] for r in rs)
        worse = [r["worse_stratum"] for r in rs]
        temps = [r["temperature"] for r in rs]
        print(f"  ratio  median {np.median(ratios):.2f}  "
              f"range [{min(ratios):.2f}, {max(ratios):.2f}]")
        print(f"  aggregate understates the worst stratum in {hides}/{len(rs)} seeds")
        print(f"  worse stratum: "
              f"cue_present {worse.count('cue_present')}/{len(rs)}, "
              f"cue_absent {worse.count('cue_absent')}/{len(rs)}")
        print(f"  temperature range [{min(temps):.3f}, {max(temps):.3f}]"
              + ("  <- crosses 1.0, so the DIRECTION of the post-hoc "
                 "correction is not stable"
                 if min(temps) < 1.0 < max(temps) else ""))

    print("\n" + "-" * 78)
    print("READING THIS")
    print("-" * 78)
    print("  The magnitudes move a lot run to run, so no single ECE from this")
    print("  project should be quoted as 'the' calibration of the model. What")
    print("  is worth reporting is whether the ORDERING reproduces, and")
    print("  whether it reproduces only where the audit found a dependency.")
    print("  Anything that holds in 3 of 5 seeds is a coin flip with extra")
    print("  steps; the bar is 5/5, or it is reported as not reproducing.")

    os.makedirs("out", exist_ok=True)
    with open("out/calibration_study.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print("\nwrote out/calibration_study.json")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--patients", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=16)
    a = ap.parse_args()
    main(a.seeds, a.patients, a.epochs)
