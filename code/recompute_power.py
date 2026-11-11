"""Recompute the STATISTICAL POWER table from n=6 standard deviations.

The 2026-08-25 table was built on n=3 sds, which CLAUDE.md 8-30-26 later showed
underestimate the truth by 2-3x. Its specific projections ("MIT 3-class needs
n=10, 89.0%") are therefore wrong by roughly 2x and need replacing.
"""
import json

import numpy as np
from scipy import stats

SEEDS = [42, 43, 44, 45, 46, 47]
TAGS = {"a": "focal_sampler", "b": "pretrained_focal_sampler",
        "c": "focal_inv04_sampler", "d": "pretrained_inv04_sampler"}
PAIRS = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]


def series(arm, block, k=None, cls=None):
    out = []
    for s in SEEDS:
        m = json.load(open(f"runs/{TAGS[arm]}_seed{s}/metrics.json"))
        f = np.asarray(m[block]["macro_checkpoint"]["per_class_f1"], float)
        out.append(f[:k].mean() if k else f[cls])
    return np.array(out)


def power(delta, sd, n, alpha=0.05):
    if sd <= 0:
        return 1.0
    ncp = abs(delta) / (sd / np.sqrt(n))
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(1 - stats.nct.cdf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp))


def n_for(delta, sd, target=0.80, cap=500):
    for n in range(3, cap):
        if power(delta, sd, n) >= target:
            return n
    return None


def mde(sd, n, alpha=0.05, target=0.80):
    """Smallest effect detectable at `target` power with this sd and n."""
    lo, hi = 0.0, 10 * max(sd, 1e-6)
    for _ in range(60):
        mid = (lo + hi) / 2
        if power(mid, sd, n) < target:
            lo = mid
        else:
            hi = mid
    return hi


METRICS = [
    ("INCART 3-class", "incart", 3, None),
    ("MIT 3-class", "mitbih_val", 3, None),
    ("MIT F1(S)", "mitbih_val", None, 1),
    ("MIT F1(F)", "mitbih_val", None, 3),
]

print("| comparison | effect (n=6) | sd of diff | p@6 | pow@6 | pow@10 | pow@20 | n for 80% |")
print("|---|---|---|---|---|---|---|---|")
rows = []
for mname, block, k, cls in METRICS:
    for x, y in PAIRS:
        X = series(x, block, k, cls)
        Y = series(y, block, k, cls)
        d = Y - X
        m, sd = d.mean(), d.std(ddof=1)
        t, p = stats.ttest_rel(Y, X)
        need = n_for(m, sd)
        rows.append((mname, f"{y} − {x}", m, sd, p,
                     power(m, sd, 6), power(m, sd, 10), power(m, sd, 20), need))
        print(f"| {mname}, {y} − {x} | {m:+.4f} | {sd:.4f} | {p:.3f} | "
              f"{power(m, sd, 6):.0%} | {power(m, sd, 10):.0%} | "
              f"{power(m, sd, 20):.0%} | {need if need else '>500'} |")

print()
print("MDE at 80% power (smallest effect this design can detect):")
for mname, block, k, cls in METRICS:
    sds = []
    for x, y in PAIRS:
        d = series(y, block, k, cls) - series(x, block, k, cls)
        sds.append(d.std(ddof=1))
    med = float(np.median(sds))
    print(f"  {mname:16s} median sd {med:.4f}   MDE@n=6 {mde(med, 6):.4f}   "
          f"MDE@n=10 {mde(med, 10):.4f}   MDE@n=20 {mde(med, 20):.4f}")
