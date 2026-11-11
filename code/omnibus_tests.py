"""Omnibus ("group") tests across models a/b/c/d, plus multiple-comparison
correction of the pairwise table.

This is what the co-advisor asked for. The standard name for "is there ANY
difference among the k models" on paired data is:

  * FRIEDMAN TEST      - non-parametric, ranks models within each seed. This is
                         the ML-standard omnibus (Demsar 2006, "Statistical
                         Comparisons of Classifiers over Multiple Data Sets").
  * REPEATED-MEASURES  - the parametric equivalent. Assumes normality and
    ANOVA                sphericity; with n=10 and visibly skewed arms
                         (see arm a's 0.4081-0.5840 range) Friedman is safer.

Both are reported. If the omnibus is NOT significant, the pairwise results
should be treated as exploratory regardless of their individual p-values.

WHY THIS MATTERS HERE: four_model_ttests.py runs 36 paired t-tests with NO
multiple-comparison correction. At alpha=0.05 that is ~1.8 false positives
expected by chance. Ten cells currently read "significant". This script applies
Holm-Bonferroni so the writeup can state which survive.

Run:
  python code/omnibus_tests.py                     # inter-patient, n=10
  python code/omnibus_tests.py --protocol random   # random 8:2, n=3
  python code/omnibus_tests.py --protocol reverse  # reverse, n=3
"""
import argparse
import csv
import itertools
import json
import os

import numpy as np
from scipy import stats

# Reuse the protocol table so arms/seeds never drift between scripts.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_model_ttests import PROTOCOLS, SUBSETS  # noqa: E402

# Studentized range q_alpha / sqrt(2), alpha=0.05, infinite df, by k
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}


def load_matrix(cfg, block, k):
    """(n_seeds x n_models) matrix of macro-F1 over the first k classes."""
    rows = []
    for s in cfg["seeds"]:
        row = []
        for _, _, tag in cfg["arms"]:
            m = json.load(open(f"runs/{tag}_seed{s}/metrics.json"))
            f = np.asarray(m[block]["macro_checkpoint"]["per_class_f1"], float)
            row.append(f[:k].mean())
        rows.append(row)
    return np.asarray(rows)


def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p-values in input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj


def rm_anova(X):
    """Repeated-measures ANOVA F and p. X is (n_subjects x k_conditions)."""
    n, k = X.shape
    grand = X.mean()
    ss_cond = n * ((X.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_tot = ((X - grand) ** 2).sum()
    ss_err = ss_tot - ss_cond - ss_subj
    df_cond, df_err = k - 1, (n - 1) * (k - 1)
    if ss_err <= 0 or df_err <= 0:
        return np.nan, np.nan
    F = (ss_cond / df_cond) / (ss_err / df_err)
    return F, float(1 - stats.f.cdf(F, df_cond, df_err))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="inter_patient")
    args = ap.parse_args()
    cfg = PROTOCOLS[args.protocol]
    arms = [a[0] for a in cfg["arms"]]
    n = len(cfg["seeds"])
    k = len(arms)

    datasets = ([("INCARTval", "mitbih_val"), ("MITtarget", "incart")]
                if args.protocol == "reverse"
                else [("MIT", "mitbih_val"), ("INCART", "incart")])

    print("=" * 100)
    print(f"OMNIBUS TESTS  |  {cfg['label']}  |  k={k} models, n={n} seeds")
    print("=" * 100)
    print("H0: all four models have the same expected macro-F1.\n")

    rows = []
    for dname, block in datasets:
        for sname, nc in SUBSETS:
            X = load_matrix(cfg, block, nc)          # (n x k)
            chi2, p_fr = stats.friedmanchisquare(*[X[:, i] for i in range(k)])
            F, p_an = rm_anova(X)
            ranks = np.apply_along_axis(stats.rankdata, 1, -X)   # 1 = best
            avg_rank = ranks.mean(axis=0)
            cd = (NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
                  if k in NEMENYI_Q05 else np.nan)

            flag = "SIGNIFICANT" if p_fr < 0.05 else ("trend" if p_fr < 0.10 else "n.s.")
            print(f"{dname} {sname}")
            print(f"  Friedman   chi2={chi2:6.3f}  p={p_fr:.4f}   {flag}")
            print(f"  RM-ANOVA   F   ={F:6.3f}  p={p_an:.4f}")
            print("  mean rank  " + "  ".join(f"{a}={r:.2f}" for a, r in zip(arms, avg_rank))
                  + f"   (1=best, Nemenyi CD={cd:.2f})")
            best = int(np.argmin(avg_rank))
            sep = [arms[i] for i in range(k)
                   if i != best and abs(avg_rank[i] - avg_rank[best]) > cd]
            print(f"  Nemenyi    best={arms[best]}; separated from: "
                  f"{', '.join(sep) if sep else 'NONE'}")
            rows.append(dict(dataset=dname, subset=sname, chi2=round(float(chi2), 3),
                             p_friedman=round(float(p_fr), 4),
                             F=round(float(F), 3), p_anova=round(float(p_an), 4),
                             **{f"rank_{a}": round(float(r), 2)
                                for a, r in zip(arms, avg_rank)}))
            print()

    # ── Holm correction of the pairwise table ───────────────────────────────
    print("=" * 100)
    print("HOLM-BONFERRONI CORRECTION of the pairwise t-tests")
    print("=" * 100)
    print("Family = the 6 model pairs within each dataset x subset cell.")
    print("(Correcting across all 36 is over-conservative: the 5/4/3-class")
    print(" subsets are nested, not independent.)\n")

    print(f"  {'dataset':<10}{'subset':<9}{'pair':<8}{'diff':>10}{'p raw':>9}"
          f"{'p Holm':>9}  verdict")
    survivors, killed = 0, 0
    out = []
    for dname, block in datasets:
        for sname, nc in SUBSETS:
            X = load_matrix(cfg, block, nc)
            pairs, diffs, praw = [], [], []
            for i, j in itertools.combinations(range(k), 2):
                d = X[:, j] - X[:, i]
                t, p = stats.ttest_rel(X[:, j], X[:, i])
                pairs.append(f"{arms[j]} - {arms[i]}")
                diffs.append(d.mean())
                praw.append(p)
            padj = holm(praw)
            for pr, df, p0, p1 in zip(pairs, diffs, praw, padj):
                if p0 < 0.05:
                    v = "SURVIVES" if p1 < 0.05 else "*** KILLED by correction ***"
                    survivors += p1 < 0.05
                    killed += p1 >= 0.05
                    print(f"  {dname:<10}{sname:<9}{pr:<8}{df:>+10.4f}"
                          f"{p0:>9.4f}{p1:>9.4f}  {v}")
                out.append(dict(dataset=dname, subset=sname, pair=pr,
                                mean_diff=round(float(df), 4),
                                p_raw=round(float(p0), 4),
                                p_holm=round(float(p1), 4),
                                sig_raw=p0 < 0.05, sig_holm=p1 < 0.05))
    print(f"\n  of the raw-significant cells: {survivors} survive Holm, "
          f"{killed} do not.")

    os.makedirs("runs/_analysis", exist_ok=True)
    stem = cfg["out"]
    with open(f"runs/_analysis/{stem}_omnibus.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f"runs/_analysis/{stem}_holm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"\nwrote runs/_analysis/{stem}_omnibus.csv  ({len(rows)} rows)")
    print(f"wrote runs/_analysis/{stem}_holm.csv     ({len(out)} rows)")


if __name__ == "__main__":
    main()
