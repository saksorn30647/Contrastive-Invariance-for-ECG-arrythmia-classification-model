"""Paired FOLD-WISE comparisons between the 5-fold CV arms (a, b, c, d).

This is the k-fold analogue of four_model_ttests.py, but the pairing unit is the
FOLD, not the seed. Under this protocol the training seed is held FIXED at 42 and
the PARTITION is what varies, which changes what the numbers mean:

  * The spread reported here is PARTITION variance. It is NOT the seed-wise std
    quoted elsewhere in this project (CLAUDE.md 9-12-26) and the two must never
    be placed side by side without saying which is which. For reference, arm a's
    INCART 3-class spread is +/-0.0183 fold-wise against +/-0.0619 seed-wise
    inter-patient - the fold-wise figure understates real uncertainty.

  * Pairing is on the SAME PARTITION, which is the only valid way to compare
    these arms. Comparing an arm at k folds against another at 5 is the
    unequal-n error of CLAUDE.md 8-30-26, so this script REFUSES to compare arms
    with different fold coverage unless --allow-unequal is passed, and then it
    restricts to the intersection and says so loudly.

Per-fold differences are always printed, not just the mean. That is deliberate:
on 9-12-26 arm c's entire INCART effect turned out to be fold 2 (+0.0653 against
~+0.005 elsewhere), and on 9-14-26 arm b did the same thing on the same fold
(+0.0614). A mean that hides one dominant partition is not a result.

Run: python code/kfold_ttests.py            (use PYTHONIOENCODING=utf-8)
     python code/kfold_ttests.py --arms a b
"""
import argparse
import csv
import itertools
import json
import os

import numpy as np
from scipy import stats

ARMS = {
    "a": ("baseline", "kfold_focal_sampler"),
    "b": ("contrastive+FT", "kfold_pretrained_focal_sampler"),
    "c": ("invariance", "kfold_inv04_sampler"),
    "d": ("contrastive+inv", "kfold_pretrained_inv04_sampler"),
}
SUBSETS = [("5-class", 5), ("4-class", 4), ("3-class", 3)]
PERCLASS = [("F1(S)", 1), ("F1(F)", 3)]
DATASETS = [("MIT", "mitbih_val"), ("INCART", "incart")]
MAX_EPOCHS = 30


def load(tag, fold, seed):
    p = f"runs/{tag}_fold{fold}_seed{seed}/metrics.json"
    return json.load(open(p)) if os.path.exists(p) else None


def macro(run, block, k):
    f = np.asarray(run[block]["macro_checkpoint"]["per_class_f1"], float)
    return f[:k].mean()


def perclass(run, block, i):
    return float(np.asarray(run[block]["macro_checkpoint"]["per_class_f1"], float)[i])


def value(run, block, kind, k):
    return macro(run, block, k) if kind == "macro" else perclass(run, block, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["a", "b", "c"], choices=list(ARMS))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unequal", action="store_true",
                    help="Compare on the INTERSECTION of available folds when "
                         "arms have different coverage. Off by default because "
                         "unequal-n comparison is a documented past error.")
    args = ap.parse_args()

    runs, coverage = {}, {}
    for key in args.arms:
        _, tag = ARMS[key]
        got = {k: load(tag, k, args.seed) for k in range(args.n_folds)}
        runs[key] = got
        coverage[key] = sorted(k for k, v in got.items() if v is not None)
        if not coverage[key]:
            raise SystemExit(f"arm {key} ({tag}) has no completed folds")

    print("=" * 96)
    print(f"5-FOLD CV PAIRED FOLD-WISE COMPARISON  (seed fixed at {args.seed}; "
          f"the FOLD is the varying factor)")
    print("=" * 96)
    for key in args.arms:
        name, _ = ARMS[key]
        eps = [runs[key][k]["epochs_run"] for k in coverage[key]]
        capped = sum(e >= MAX_EPOCHS for e in eps)
        print(f"  {key}. {name:<16} folds {coverage[key]}  epochs {eps}  "
              f"capped {capped}/{len(eps)}")
    print()

    common = sorted(set.intersection(*[set(coverage[k]) for k in args.arms]))
    incomplete = [k for k in args.arms if len(coverage[k]) < args.n_folds]
    if incomplete:
        print("!" * 96)
        print(f"  INCOMPLETE ARM(S): {', '.join(incomplete)} have fewer than "
              f"{args.n_folds} folds.")
        print(f"  Comparing on the {len(common)} shared fold(s) {common} ONLY.")
        print("  An arm at n<5 must NOT be quoted against an arm at n=5 - that is")
        print("  the unequal-n error of CLAUDE.md 8-30-26. These are PROVISIONAL.")
        print("!" * 96)
        if not args.allow_unequal:
            print("\n  Refusing to print comparisons. Re-run with --allow-unequal "
                  "if you\n  understand these are provisional, or finish the arm.\n")
            return
        print()
    if len(common) < 2:
        raise SystemExit("need at least 2 shared folds to compare")

    # ---- EPOCH-CAP ASYMMETRY ------------------------------------------------
    # Arms that cap on every fold have in-domain scores that are LOWER BOUNDS;
    # arms that early-stop mostly do not. A between-arm in-domain difference is
    # then confounded by the stopping rule, not only by the intervention.
    caps = {k: sum(runs[k][f]["epochs_run"] >= MAX_EPOCHS for f in common)
            for k in args.arms}
    if len(set(caps.values())) > 1:
        print("-" * 96)
        print("EPOCH-CAP ASYMMETRY - read before quoting any IN-DOMAIN difference")
        print("-" * 96)
        for k in args.arms:
            print(f"  arm {k}: capped on {caps[k]}/{len(common)} shared folds")
        print("  Capped runs have NOT converged, so their in-domain scores are")
        print("  LOWER BOUNDS. An arm that early-stops is favoured in-domain by")
        print("  the STOPPING RULE as well as by the intervention. Disclose both.")
        print()

    rows = []
    for x, y in itertools.combinations(args.arms, 2):
        print("=" * 96)
        print(f"  {y} - {x}   ({ARMS[y][0]} minus {ARMS[x][0]})   "
              f"n={len(common)} folds")
        print("=" * 96)
        for dname, block in DATASETS:
            metrics = [(f"{dname} {s}", ("macro", k)) for s, k in SUBSETS]
            metrics += [(f"{dname} {s}", ("per", i)) for s, i in PERCLASS]
            for label, (kind, k) in metrics:
                a = np.array([value(runs[x][f], block, kind, k) for f in common])
                b = np.array([value(runs[y][f], block, kind, k) for f in common])
                d = b - a
                wins = int((d > 0).sum())
                if len(common) > 1 and np.std(d) > 0:
                    t, p = stats.ttest_rel(b, a)
                else:
                    t, p = float("nan"), float("nan")
                star = " *" if p == p and p < 0.05 else ""
                per = "  ".join(f"{v:+.4f}" for v in d)
                print(f"  {label:<16}{d.mean():+9.4f}  p={p:7.4f}{star:<2} "
                      f"{wins}/{len(common)}   per-fold: {per}")
                # A mean dominated by one fold is one observation, not n.
                med = float(np.median(np.abs(d)))
                if len(common) >= 3 and med > 0 and np.abs(d).max() > 3 * med:
                    j = int(np.argmax(np.abs(d)))
                    tot = float(np.abs(d).sum())
                    share = abs(d[j]) / tot if tot > 0 else float("nan")
                    print(f"  {'':16}  ^ DOMINATED BY FOLD {common[j]} "
                          f"({share:.0%} of total absolute effect) - treat as one "
                          f"observation, not {len(common)}")
                rows.append(dict(contrast=f"{y}-{x}", metric=label,
                                 mean_diff=round(float(d.mean()), 6),
                                 p=None if p != p else round(float(p), 6),
                                 wins=f"{wins}/{len(common)}",
                                 folds=str(common),
                                 per_fold=";".join(f"{v:+.4f}" for v in d)))
        print()

    os.makedirs("runs/_analysis", exist_ok=True)
    out = "runs/_analysis/kfold_ttests.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} comparisons)")
    print("\nREMINDER: the spread across folds is PARTITION variance (seed fixed "
          "at 42).\nDo not quote it beside the seed-wise stds elsewhere in this "
          "project.")


if __name__ == "__main__":
    main()
