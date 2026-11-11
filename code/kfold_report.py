"""Fold-wise summary of the 5-fold CV arms, in the form the literature reports.

LMUEBCNet (Xie & Lin 2023) reports random 10-fold CV over pooled beats. This
script produces the equivalent presentation for this project's arms: per-fold
values plus mean +/- std ACROSS FOLDS, which is what makes the comparison
like-for-like instead of like-for-similar.

Two things it deliberately prints alongside the numbers:

  1. The SINGLE-DRAW random-split arm, for reference. 5-fold fold 0 and the
     `randsplit_*` arm are the same protocol sampled differently, so they should
     agree; a large disagreement would mean the fold construction is wrong.

  2. The EPOCH-CAP CONFOUND. Under a leaky pooled-beat split the validation
     metric keeps improving, so early stopping rarely fires and runs hit the
     30-epoch cap (CLAUDE.md 8-24-26 and 9-1-26). Capped runs have NOT converged,
     so their scores are lower bounds. This script counts how many folds capped.

NOTE on what the std means here. Across FOLDS it mixes two sources: genuine
partition variance, and the run-to-run nondeterminism of bug #8. It is NOT a
seed-variance estimate - the training seed is held fixed at 42 and the FOLD is
the varying factor. Do not compare this std against the seed-wise stds elsewhere
in the project without saying which is which.

Run: python code/kfold_report.py [--tag kfold_focal_sampler] [--n-folds 5]
     (use PYTHONIOENCODING=utf-8)
"""
import argparse
import glob
import json
import os

import numpy as np

NAMES = ["N", "S", "V", "F", "Q"]
MAX_EPOCHS = 30          # the cap set in dual_protocol_training.py


def load_fold(tag, fold, seed):
    p = f"runs/{tag}_fold{fold}_seed{seed}/metrics.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def subsets(block):
    f = np.asarray(block["macro_checkpoint"]["per_class_f1"], float)
    return {"5c": f[:5].mean(), "4c": f[:4].mean(), "3c": f[:3].mean(),
            "N": f[0], "S": f[1], "V": f[2], "F": f[3], "Q": f[4]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="kfold_focal_sampler")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference", default="randsplit_focal_sampler",
                    help="Single-draw arm of the same protocol, for the "
                         "consistency check. Pass '' to skip.")
    args = ap.parse_args()

    runs = {k: load_fold(args.tag, k, args.seed) for k in range(args.n_folds)}
    have = [k for k, v in runs.items() if v is not None]
    missing = [k for k, v in runs.items() if v is None]

    print("=" * 94)
    print(f"5-FOLD CV REPORT - {args.tag}, seed {args.seed}")
    print("=" * 94)
    if missing:
        print(f"INCOMPLETE: folds present {have}, MISSING {missing}. "
              f"Numbers below are over {len(have)} fold(s) only.\n")
    if not have:
        raise SystemExit("no folds found")

    cfg = runs[have[0]]["config"]
    print(f"protocol={cfg.get('protocol')}  n_folds={cfg.get('n_folds')}  "
          f"lambda_inv={cfg['lambda_inv']}  sampler={cfg.get('sampler')}  "
          f"init={'pretrained' if cfg.get('pretrained_init') else 'random'}")
    print()

    for blk, label in (("mitbih_val", "MIT-BIH validation fold (in-domain)"),
                       ("incart", "INCART zero-shot (cross-dataset)")):
        print("-" * 94)
        print(label)
        print("-" * 94)
        hdr = f"  {'fold':<6}" + "".join(f"{k:>9}" for k in
                                         ("5c", "4c", "3c", "N", "S", "V", "F", "Q"))
        print(hdr + f"{'epochs':>9}{'best':>6}")
        vals = {k: [] for k in ("5c", "4c", "3c", "N", "S", "V", "F", "Q")}
        capped = 0
        for k in have:
            m = runs[k]
            d = subsets(m[blk])
            ep = m["epochs_run"]
            be = m["mitbih_val"]["macro_checkpoint"]["epoch"]
            capped += (ep >= MAX_EPOCHS)
            row = f"  {k:<6}" + "".join(f"{d[c]:>9.4f}" for c in vals)
            print(row + f"{ep:>9}{be:>6}")
            for c in vals:
                vals[c].append(d[c])
        print("  " + "-" * 90)
        mean = "  MEAN  " + "".join(f"{np.mean(vals[c]):>9.4f}" for c in vals)
        std = "  STD   " + "".join(f"{np.std(vals[c], ddof=1):>9.4f}" for c in vals) \
            if len(have) > 1 else "  STD     (n=1)"
        print(mean)
        print(std)
        print()

    # ── epoch-cap confound ──────────────────────────────────────────────────
    eps = [runs[k]["epochs_run"] for k in have]
    n_capped = sum(e >= MAX_EPOCHS for e in eps)
    print("-" * 94)
    print("EPOCH-CAP CONFOUND")
    print("-" * 94)
    print(f"  epochs run per fold: {eps}")
    print(f"  folds that hit the {MAX_EPOCHS}-epoch cap: {n_capped}/{len(have)}")
    if n_capped:
        print("  -> Under a leaky pooled-beat split the val metric keeps improving, so")
        print("     early stopping rarely fires. Capped runs have NOT converged and their")
        print("     scores are LOWER BOUNDS. Disclose this exactly as on 8-24-26 / 9-1-26.")
    print()

    # ── consistency check against the single-draw arm ───────────────────────
    if args.reference:
        ref = sorted(glob.glob(f"runs/{args.reference}_seed*/metrics.json"))
        if ref:
            print("-" * 94)
            print(f"CONSISTENCY CHECK vs the single-draw arm ({args.reference}, "
                  f"n={len(ref)} seeds)")
            print("-" * 94)
            print("  Same protocol, sampled differently - these should agree. A large")
            print("  disagreement would indicate the fold construction is wrong.")
            print(f"  {'metric':<20}{'k-fold mean':>14}{'single-draw':>16}{'diff':>10}")
            for blk, keys in (("mitbih_val", ("3c",)), ("incart", ("3c",))):
                r = [subsets(json.load(open(p))[blk]) for p in ref]
                k = [subsets(runs[i][blk]) for i in have]
                for key in keys:
                    rm = float(np.mean([x[key] for x in r]))
                    km = float(np.mean([x[key] for x in k]))
                    name = ("MIT " if blk == "mitbih_val" else "INCART ") + key
                    print(f"  {name:<20}{km:>14.4f}{rm:>16.4f}{km - rm:>+10.4f}")
            print()

    os.makedirs("runs/_analysis", exist_ok=True)
    out = f"runs/_analysis/{args.tag}_kfold.json"
    with open(out, "w") as fh:
        json.dump({str(k): {b: subsets(runs[k][b])
                            for b in ("mitbih_val", "incart")} for k in have},
                  fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
