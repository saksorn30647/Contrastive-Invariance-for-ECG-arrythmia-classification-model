"""Zero-shot vs INCART-supervised ceiling, scored on the SAME 8 records.

Answers the co-advisor's first comment ("prove zero-shot works, not just
improves") by bracketing the transferred model between two absolute anchors:

  FLOOR    a majority-class (always-N) predictor, pure arithmetic.
  CEILING  the reverse-direction runs, which trained on 67 INCART records and
           validated on the 8 held out. That is supervision ON the target
           domain, patient-disjoint.

The comparison is only fair if both are scored on the same beats, so the
MIT-BIH-trained checkpoints are re-scored on exactly those 8 validation
records (I08 I18 I19 I29 I30 I33 I53 I60) rather than on all 75. Predictions
come from `runs/_analysis/preds_{tag}_seed{seed}.npy`
(`code/incart_predict.py`), so this is a file read, not a training run.

READ THE PER-CLASS CEILING WITH ITS RECORD COUNT. The 8-record validation set
is NOT equally informative across classes:

    class   val beats   records   effective records
    N          14,058         8        7.33
    V           4,216         7        5.24
    S             592         2        1.01   <-- I33 is 100%
    F              56         1        1.00   <-- I18 is 100%
    Q               2         2        2.00

So the N and V ceilings are meaningful, while the S and F "ceilings" are each
measured on a single patient and inherit exactly the single-patient pathology
this project documents everywhere else. They are reported, but they are one
draw of a coin, not a class-level capability.

Run: python code/incart_ceiling.py         (use PYTHONIOENCODING=utf-8)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset2D import LABEL_MAP
from utils.splits import INCART_VAL_RECORDS

CLASSES = ["N", "S", "V", "F", "Q"]
SEEDS = [42, 43, 44]
FWD = {"a": "focal_sampler", "b": "pretrained_focal_sampler",
       "c": "focal_inv04_sampler", "d": "pretrained_inv04_sampler"}
REV = {"a": "reverse_focal_sampler", "b": "reverse_pretrained_focal_sampler",
       "c": "reverse_inv04_sampler", "d": "reverse_pretrained_inv04_sampler"}


def prf(y_true, y_pred, n=5):
    P, R, F = np.zeros(n), np.zeros(n), np.zeros(n)
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        P[c] = tp / (tp + fp) if tp + fp else 0.0
        R[c] = tp / (tp + fn) if tp + fn else 0.0
        F[c] = 2 * P[c] * R[c] / (P[c] + R[c]) if P[c] + R[c] else 0.0
    return P, R, F


def main():
    rec = np.array([str(r).strip() for r in np.load("data/records.npy", allow_pickle=True)])  # INCART record ids (75 records / 175,811 beats). NOT root records.npy, which is MIT-BIH.
    y = np.array([LABEL_MAP[str(l).strip()] for l in np.load("data/y_incart_cwt.npy")])
    sub = np.isin(rec, INCART_VAL_RECORDS)
    y8 = y[sub]
    print(f"8-record subset: {sub.sum():,} beats of {len(y):,}")
    counts = np.bincount(y8, minlength=5)
    print("  support: " + "  ".join(f"{c}={counts[i]:,}" for i, c in enumerate(CLASSES)))
    print()

    res = {"support_8rec": counts.tolist(), "n_beats_8rec": int(sub.sum())}

    # ---- FLOOR: always-N, on both the 8-record subset and all 75 ----------
    for name, yy in (("all75", y), ("rec8", y8)):
        _, _, F = prf(yy, np.zeros_like(yy))
        res[f"floor_{name}"] = dict(per_class=F.tolist(),
                                    macro5=float(F[:5].mean()),
                                    macro4=float(F[:4].mean()),
                                    macro3=float(F[:3].mean()))
    print("FLOOR (always predict N)")
    for name in ("all75", "rec8"):
        f = res[f"floor_{name}"]
        print(f"  {name:<6} 5c={f['macro5']:.4f}  4c={f['macro4']:.4f}  3c={f['macro3']:.4f}")
    print()

    # ---- ZERO-SHOT on the same 8 records ----------------------------------
    print("ZERO-SHOT (MIT-BIH-trained), re-scored on the 8 INCART val records, n=3")
    zs = {}
    for k, tag in FWD.items():
        Ps, Rs, Fs = [], [], []
        missing = []
        for s in SEEDS:
            p = f"runs/_analysis/preds_{tag}_seed{s}.npy"
            if not os.path.exists(p):
                missing.append(s)
                continue
            pred = np.load(p).astype(int)[sub]
            P, R, F = prf(y8, pred)
            Ps.append(P); Rs.append(R); Fs.append(F)
        if not Fs:
            print(f"  {k}: no predictions (missing seeds {missing}) - skipped")
            continue
        P, R, F = np.mean(Ps, 0), np.mean(Rs, 0), np.mean(Fs, 0)
        zs[k] = dict(P=P.tolist(), R=R.tolist(), F=F.tolist(),
                     macro5=float(F[:5].mean()), macro4=float(F[:4].mean()),
                     macro3=float(F[:3].mean()), n_seeds=len(Fs))
        print(f"  {k}: " + "  ".join(f"{c}={F[i]:.4f}" for i, c in enumerate(CLASSES))
              + f" | 3c={F[:3].mean():.4f}  (n={len(Fs)})")
    res["zeroshot_rec8"] = zs
    print()

    # ---- CEILING: INCART-supervised, same 8 records ------------------------
    print("CEILING (INCART-supervised, reverse runs), same 8 records, n=3")
    ce = {}
    for k, tag in REV.items():
        Ps, Rs, Fs = [], [], []
        for s in SEEDS:
            p = f"runs/{tag}_seed{s}/metrics.json"
            if not os.path.exists(p):
                continue
            m = json.load(open(p))["mitbih_val"]["macro_checkpoint"]
            Ps.append(m["per_class_precision"]); Rs.append(m["per_class_recall"])
            Fs.append(m["per_class_f1"])
        if not Fs:
            continue
        P, R, F = np.mean(Ps, 0), np.mean(Rs, 0), np.mean(Fs, 0)
        ce[k] = dict(P=P.tolist(), R=R.tolist(), F=F.tolist(),
                     macro5=float(F[:5].mean()), macro4=float(F[:4].mean()),
                     macro3=float(F[:3].mean()), n_seeds=len(Fs))
        print(f"  {k}: " + "  ".join(f"{c}={F[i]:.4f}" for i, c in enumerate(CLASSES))
              + f" | 3c={F[:3].mean():.4f}  (n={len(Fs)})")
    res["ceiling_rec8"] = ce
    print()

    # ---- the bracket -------------------------------------------------------
    print("ZERO-SHOT AS A FRACTION OF THE SUPERVISED CEILING (same 8 records)")
    print(f"  {'':<4}{'class':<7}{'floor':>9}{'zero-shot':>11}{'ceiling':>10}{'% of ceiling':>14}")
    frac = {}
    for k in FWD:
        if k not in zs or k not in ce:
            continue
        row = {}
        for i, c in enumerate(CLASSES):
            z, cl = zs[k]["F"][i], ce[k]["F"][i]
            fl = res["floor_rec8"]["per_class"][i]
            pct = 100.0 * z / cl if cl > 0 else float("nan")
            row[c] = pct
            if k == "a":
                print(f"  {k:<4}{c:<7}{fl:>9.4f}{z:>11.4f}{cl:>10.4f}{pct:>13.1f}%")
        for sub_name, kk in (("3c", "macro3"), ("4c", "macro4"), ("5c", "macro5")):
            z, cl = zs[k][kk], ce[k][kk]
            row[sub_name] = 100.0 * z / cl if cl > 0 else float("nan")
        frac[k] = row
    print()
    print(f"  {'model':<8}" + "".join(f"{c:>9}" for c in CLASSES) +
          f"{'3c':>9}{'4c':>9}{'5c':>9}   (zero-shot as % of ceiling)")
    for k, row in frac.items():
        print(f"  {k:<8}" + "".join(f"{row[c]:>8.1f}%" for c in CLASSES) +
              f"{row['3c']:>8.1f}%{row['4c']:>8.1f}%{row['5c']:>8.1f}%")
    res["pct_of_ceiling"] = frac

    os.makedirs("runs/_analysis", exist_ok=True)
    json.dump(res, open("runs/_analysis/incart_ceiling.json", "w"), indent=1)
    print("\nwrote runs/_analysis/incart_ceiling.json")


if __name__ == "__main__":
    main()
