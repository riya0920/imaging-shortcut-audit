"""Re-run ONLY the de-confounding arm, with as many seeds as you can afford.

`run_complete.py --seeds N` re-runs all three studies. This runs just the arm
whose bound is seed-limited, so the compute goes where it buys something.

`power.py` says what each seed count is worth: 3 seeds bound the effect at
0.094 AUROC, 25 at 0.033, and resolving the observed 0.004 would need ~1,750
seeds and about a week. So this TIGHTENS the bound; it does not close it, and
the output says so.

Run:  python more_seeds.py --seeds 25
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import calibrate as C
import run_complete as RC
from run_complete import OPACITY, _logits, auroc, strip_marker

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=25)
ap.add_argument("--n-patients", type=int, default=500)
ap.add_argument("--epochs", type=int, default=14)
a = ap.parse_args()

with_m, without_m = [], []
t0 = time.time()
for k in range(a.seeds):
    x = RC.fit(data_seed=k, init_seed=k, n_patients=a.n_patients,
               epochs=a.epochs)
    with_m.append(auroc(x["Yte"][:, OPACITY],
                        C._sigmoid(_logits(x["model"], x["Xte"])[:, OPACITY])))
    y = RC.fit(data_seed=k, init_seed=k, n_patients=a.n_patients,
               epochs=a.epochs, deconfound=strip_marker)
    without_m.append(auroc(y["Yte"][:, OPACITY],
                           C._sigmoid(_logits(y["model"], y["Xte"])[:, OPACITY])))
    print("  seed %d/%d  with=%.4f without=%.4f  (%.0fs elapsed)"
          % (k + 1, a.seeds, with_m[-1], without_m[-1], time.time() - t0),
          flush=True)

mw, mo = float(np.mean(with_m)), float(np.mean(without_m))
sd = float(np.std(np.array(with_m) - np.array(without_m), ddof=1))
mdd = 2.78 * sd / np.sqrt(len(with_m))
out = {"seeds": a.seeds, "with_marker": with_m, "without_marker": without_m,
       "mean_with": mw, "mean_without": mo, "drop": mw - mo,
       "paired_sd": sd, "min_detectable_difference": float(mdd),
       "resolvable": bool(abs(mw - mo) >= mdd),
       "seconds": time.time() - t0}
os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
with open(os.path.join(ROOT, "out", "deconfound_seeds.json"), "w",
          encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print()
print("SEEDS %d   drop %+.4f   paired sd %.4f   MDD %.4f   resolvable %s"
      % (a.seeds, mw - mo, sd, mdd, out["resolvable"]))
print("DONE")
