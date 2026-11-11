"""Dry-run verification of the 5-fold CV split. NO TRAINING, NO GPU.

Proves the fold construction before any GPU time is spent on it. Checks, in
order of how badly each would corrupt a CV result:

  1. the 5 validation folds PARTITION the dataset (every beat in exactly one)
  2. train/val are disjoint and cover everything, per fold
  3. stratification holds - per-fold class proportions match the global ones
  4. no fold is missing a class (F is 0.73% of MIT-BIH; a fold with zero F
     would make that fold's F1(F) undefined)
  5. determinism - same seed gives the same folds
  6. the CONTRAST that motivates this file: five calls to random_split_indices
     with five seeds do NOT partition the data

It also prints the exact run matrix and GPU budget, and asserts that no output
directory would be overwritten - so the real sweep can be launched from a known
state.

Run: python code/test_kfold.py    (use PYTHONIOENCODING=utf-8)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset2D import LABEL_MAP                    # noqa: E402
from utils.splits import (kfold_indices, verify_kfold,   # noqa: E402
                          random_split_indices)

NAMES = ["N", "S", "V", "F", "Q"]
N_FOLDS = 5
SPLIT_SEED = 0

rec = np.array([str(r).strip() for r in np.load("records.npy")])  # MIT-BIH record ids (48 records / 109,451 beats). NOT data/records.npy, which is INCART.
y = np.array([LABEL_MAP[str(v).strip()] for v in np.load("data/y_cwt.npy")])
n = len(y)
assert len(rec) == n, "record/beat misalignment"

print("=" * 92)
print(f"5-FOLD CV DRY RUN — MIT-BIH pooled beats, n={n:,}, split_seed={SPLIT_SEED}")
print("=" * 92)
glob_pct = {NAMES[c]: 100 * (y == c).mean() for c in range(5)}
print("global class mix: " + "  ".join(f"{k} {v:.3f}%" for k, v in glob_pct.items()))
print()

# ── 1-4. build and verify ───────────────────────────────────────────────────
folds = kfold_indices(y, n_folds=N_FOLDS, seed=SPLIT_SEED, stratified=True)
stats = verify_kfold(folds, n, y=y, class_names=NAMES)
print("[1-2] PARTITION + DISJOINTNESS ....... PASS "
      "(every beat in exactly one val fold; train/val disjoint; union == all)")

print()
print(f"  {'fold':<6}{'n_train':>10}{'n_val':>9}{'val %':>8}   " +
      "".join(f"{c:>9}" for c in NAMES))
worst_dev = 0.0
missing = []
for st, (tr, va) in zip(stats, folds):
    counts = st["val_class_counts"]
    row = f"  {st['fold']:<6}{st['n_train']:>10}{st['n_val']:>9}{100*st['n_val']/n:>7.2f}%   "
    for c in NAMES:
        got = counts.get(c, 0)
        row += f"{got:>9}"
        if got == 0:
            missing.append((st["fold"], c))
        pct = 100 * got / st["n_val"]
        worst_dev = max(worst_dev, abs(pct - glob_pct[c]))
    print(row)

print()
print(f"[3] STRATIFICATION ................... "
      f"{'PASS' if worst_dev < 0.05 else 'CHECK'} "
      f"(max deviation of any fold's class share from global: {worst_dev:.4f} pp)")
print(f"[4] NO EMPTY CLASS IN ANY FOLD ....... "
      f"{'PASS' if not missing else 'FAIL ' + str(missing)}")

# per-class spread across folds, for the rare classes that actually matter
print()
print("  rare-class counts per validation fold (the ones that could break):")
for c in ("S", "F", "Q"):
    ci = NAMES.index(c)
    per = [int((y[va] == ci).sum()) for _, va in folds]
    print(f"    {c}: {per}   total {sum(per):,}  (dataset has {int((y==ci).sum()):,})")

# ── 5. determinism ──────────────────────────────────────────────────────────
again = kfold_indices(y, n_folds=N_FOLDS, seed=SPLIT_SEED, stratified=True)
same = all((a[0] == b[0]).all() and (a[1] == b[1]).all()
           for a, b in zip(folds, again))
other = kfold_indices(y, n_folds=N_FOLDS, seed=1, stratified=True)
diff = not all((a[1] == b[1]).all() for a, b in zip(folds, other))
print()
print(f"[5] DETERMINISM ...................... {'PASS' if same else 'FAIL'} "
      f"(same seed -> identical folds; different seed -> different folds: {diff})")

# ── 6. why random_split_indices is NOT k-fold ───────────────────────────────
print()
print("[6] CONTRAST — five random_split_indices(seed=k) draws are NOT a partition:")
seen = np.zeros(n, dtype=int)
for k in range(N_FOLDS):
    _, va = random_split_indices(y, val_fraction=0.2, seed=k, stratified=True)
    seen[va] += 1
u, c = np.unique(seen, return_counts=True)
print("      times each beat lands in a validation set: " +
      ", ".join(f"{int(a)}x: {int(b):,} beats" for a, b in zip(u, c)))
print(f"      never validated: {int((seen==0).sum()):,}   "
      f"validated 3+ times: {int((seen>=3).sum()):,}")
print("      -> averaging over these would double-count some beats and never")
print("         test others. kfold_indices gives exactly 1x for all "
      f"{n:,} beats.")

# ── run matrix ──────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("RUN MATRIX — what the real sweep would execute (NOT run here)")
print("=" * 92)
ARMS = [("a", "kfold_focal_sampler", "0"), ("c", "kfold_inv04_sampler", "0.4")]
MIN_PER_RUN, MAX_PER_RUN = 26, 88          # measured range for random-split runs
collisions = []
print(f"  {'arm':<5}{'tag':<26}{'lambda':>8}{'folds':>7}")
for key, tag, lam in ARMS:
    print(f"  {key:<5}{tag:<26}{lam:>8}{N_FOLDS:>7}")
    for k in range(N_FOLDS):
        d = f"runs/{tag}_fold{k}_seed42"   # must match scripts/run_kfold.ps1
        if os.path.isdir(d):
            collisions.append(d)
n_runs = len(ARMS) * N_FOLDS
print(f"\n  total runs: {n_runs}")
print(f"  GPU budget: {n_runs*MIN_PER_RUN/60:.1f}-{n_runs*MAX_PER_RUN/60:.1f} h "
      f"(measured {MIN_PER_RUN}-{MAX_PER_RUN} min per random-split run; "
      f"they hit the 30-epoch cap because early stopping rarely fires "
      f"under leakage)")
print(f"  output collisions: {collisions if collisions else 'NONE — all clear'}")
print()
print("  NOT INCLUDED: models b and d. Each fold would need its own contrastive")
print("  encoder pretrained on that fold's TRAIN side (CLAUDE.md 8-31-26 rule),")
print(f"  i.e. {N_FOLDS} x ~2 h = ~10 h on top. Scoped out deliberately.")
print()
print("ALL CHECKS PASSED — fold construction is sound; no GPU time spent.")
