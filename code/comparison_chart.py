"""Full four-model comparison chart -> Markdown.

Every pairwise contrast, every class subset, both datasets, for every protocol
that has runs on disk, with the multiple-comparison corrections spelled out and
the FAMILY each correction assumes stated next to it. Regenerate with:

    python code/comparison_chart.py          (use PYTHONIOENCODING=utf-8)

Design decisions that are not arbitrary:

* **Paired tests only.** Every arm is run at the same seeds (or the same folds),
  so the contrast is within-seed. Comparing arm means with an unpaired test
  throws away the pairing and is much less powerful.
* **Three corrections, because the answer depends on the family.** Raw p, Holm
  over all 6 pairs, and max-T over the 3 vs-baseline contrasts. CLAUDE.md
  9-11-26: the study's question is "does the proposed method improve on the
  baseline", so {b-a, c-a, d-a} is the defensible family; the 6-pair table is
  the omnibus companion. The same c-a contrast reads p=0.035 or p=0.108
  depending only on which is declared.
* **max-T is reported only where it can resolve.** A sign-flip permutation
  test's floor is 2/2^n, so at n=5 folds the smallest achievable p is 0.0625 —
  it cannot clear 0.05 for ANY data. The script suppresses it below n=6 and
  says why, rather than printing a misleading 0.0625.
* **Sign test alongside the t-test.** They disagree informatively: INCART c-b
  is 9/10 seeds positive (sign p=0.021) but only +0.0108 (t p=0.444).
"""

import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_baseline_tests import maxt_permutation

OUTDIR = "results/reports"
NAME = {"a": "baseline", "b": "contrastive+FT", "c": "invariance", "d": "contrastive+inv"}
SUBSETS = [("5-class", 5), ("4-class", 4), ("3-class", 3)]

PROTOCOLS = {
    "inter_patient": dict(
        label="Inter-patient (record-disjoint 38/10) — THE HEADLINE PROTOCOL",
        unit="seed", units=list(range(42, 52)),
        src="MIT-BIH val (in-domain)", tgt="INCART (zero-shot)",
        tags={"a": "focal_sampler", "b": "pretrained_focal_sampler",
              "c": "focal_inv04_sampler", "d": "pretrained_inv04_sampler"},
        path=lambda t, u: f"runs/{t}_seed{u}/metrics.json"),
    "kfold": dict(
        label="5-fold CV, pooled-beat 8:2 (seed fixed at 42, FOLD varies)",
        unit="fold", units=list(range(5)),
        src="MIT-BIH val (in-domain)", tgt="INCART (zero-shot)",
        tags={"a": "kfold_focal_sampler", "b": "kfold_pretrained_focal_sampler",
              "c": "kfold_inv04_sampler", "d": "kfold_pretrained_inv04_sampler"},
        path=lambda t, u: f"runs/{t}_fold{u}_seed42/metrics.json"),
    "random": dict(
        label="Random 8:2 single draw (split_seed 0, stratified)",
        unit="seed", units=[42, 43, 44],
        src="MIT-BIH val (in-domain)", tgt="INCART (zero-shot)",
        tags={"a": "randsplit_focal_sampler", "b": "randsplit_pretrained_focal_sampler",
              "c": "randsplit_inv04_sampler", "d": "randsplit_pretrained_inv04_sampler"},
        path=lambda t, u: f"runs/{t}_seed{u}/metrics.json"),
    "reverse": dict(
        label="REVERSE direction: train INCART -> test MIT-BIH",
        unit="seed", units=[42, 43, 44],
        src="INCART val (in-domain)", tgt="MIT-BIH (zero-shot)",
        tags={"a": "reverse_focal_sampler", "b": "reverse_pretrained_focal_sampler",
              "c": "reverse_inv04_sampler", "d": "reverse_pretrained_inv04_sampler"},
        path=lambda t, u: f"runs/{t}_seed{u}/metrics.json"),
}
# metrics.json always calls the two blocks mitbih_val / incart; under the
# reverse protocol those are INCART-val and MIT-BIH-target respectively.
BLOCKS = [("src", "mitbih_val"), ("tgt", "incart")]


def vals(cfg, arm, block, k):
    out = []
    for u in cfg["units"]:
        p = cfg["path"](cfg["tags"][arm], u)
        if not os.path.exists(p):
            return None
        m = json.load(open(p))
        out.append(np.asarray(m[block]["macro_checkpoint"]["per_class_f1"], float)[:k].mean())
    return np.array(out)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    out = np.empty(len(ps))
    prev = 0.0
    for r, j in enumerate(order):
        out[j] = max(prev, min(1.0, ps[j] * (len(ps) - r)))
        prev = out[j]
    return out


def paired_row(x, y):
    """Return the full descriptive + inferential bundle for one contrast."""
    d = y - x
    n = len(d)
    t, p = stats.ttest_rel(y, x)
    wins = int((d > 0).sum())
    k = max(wins, n - wins)
    sign_p = min(1.0, 2 * stats.binom.sf(k - 1, n, 0.5))
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan            # paired Cohen's dz
    se = sd / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    lo, hi = d.mean() - crit * se, d.mean() + crit * se
    return dict(mean=d.mean(), sd=sd, t=t, p=p, wins=wins, n=n,
                sign_p=sign_p, dz=dz, lo=lo, hi=hi)


def star(p):
    if p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    L = []
    A = L.append

    A("# Four-model comparison chart")
    A("")
    A("Generated by `python code/comparison_chart.py` from `runs/*/metrics.json`.")
    A("Every contrast is **paired** within seed (or within fold): all arms are run")
    A("at identical seeds, so the pairing is real and discarding it would cost power.")
    A("")
    A("| model | initialisation | λ_inv |")
    A("|---|---|---|")
    A("| **a.** baseline | random | 0.0 |")
    A("| **b.** contrastive+FT | contrastive-pretrained | 0.0 |")
    A("| **c.** invariance | random | 0.4 |")
    A("| **d.** contrastive+inv | contrastive-pretrained | 0.4 |")
    A("")
    A("Significance marks on **raw** p: `*` <0.05, `**` <0.01, `***` <0.001. "
      "Read the corrected columns before believing any of them.")
    A("")
    A("**`dz`** is paired Cohen's *d* (mean difference ÷ sd of the differences); "
      "**CI** is the 95% confidence interval on the mean paired difference. "
      "A CI that excludes 0 is equivalent to raw p<0.05 — it is shown because the "
      "*width* says how well the effect is pinned down, which a p-value does not.")
    A("")

    # ================= headline protocol, full detail =======================
    for pname in ("inter_patient", "kfold", "random", "reverse"):
        cfg = PROTOCOLS[pname]
        # availability
        probe = vals(cfg, "a", "mitbih_val", 3)
        if probe is None:
            continue
        n = len(cfg["units"])
        A("\n---\n")
        A(f"## {cfg['label']}")
        A("")
        A(f"n = **{n} {cfg['unit']}s** ({cfg['units'][0]}–{cfg['units'][-1]}). "
          f"In-domain block = {cfg['src']}; target block = {cfg['tgt']}.")
        if pname == "kfold":
            A("")
            A("> The spread here is **partition variance** — the training seed is fixed")
            A("> at 42 and the fold is what varies. It is systematically smaller than a")
            A("> seed-wise spread and the two must not be quoted side by side.")
        A("")

        # ---- arm means ----
        A("### Arm means (macro-F1, mean ± sd)")
        A("")
        A("| dataset | subset | a. baseline | b. contr.+FT | c. invariance | d. contr.+inv |")
        A("|---|---|---|---|---|---|")
        for side, block in BLOCKS:
            dsname = cfg["src"] if side == "src" else cfg["tgt"]
            for sname, k in SUBSETS:
                cells = []
                for arm in "abcd":
                    v = vals(cfg, arm, block, k)
                    cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if v is not None else "—")
                A(f"| {dsname} | {sname} | " + " | ".join(cells) + " |")
        A("")

        # ---- full pairwise ----
        for side, block in BLOCKS:
            dsname = cfg["src"] if side == "src" else cfg["tgt"]
            A(f"### All six pairwise contrasts — {dsname}")
            A("")
            for sname, k in SUBSETS:
                V = {arm: vals(cfg, arm, block, k) for arm in "abcd"}
                if any(v is None for v in V.values()):
                    continue
                pairs = list(itertools.combinations("abcd", 2))
                rows = [paired_row(V[x], V[y]) for x, y in pairs]
                raw = [r["p"] for r in rows]
                h6 = holm(raw)
                A(f"**{sname}**")
                A("")
                A("| contrast | mean Δ | 95% CI | dz | t | p raw | Holm(6 pairs) | wins | sign p |")
                A("|---|---|---|---|---|---|---|---|---|")
                for (x, y), r, h in zip(pairs, rows, h6):
                    A(f"| **{y} − {x}** | {r['mean']:+.4f} | "
                      f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['dz']:+.2f} | {r['t']:+.2f} | "
                      f"{r['p']:.4f}{star(r['p'])} | {h:.4f}{'  ✓' if h < 0.05 else ''} | "
                      f"{r['wins']}/{r['n']} | {r['sign_p']:.3f} |")
                A("")

            # ---- vs-baseline family ----
            A(f"#### vs-baseline family {{b−a, c−a, d−a}} — {dsname}")
            A("")
            A("The family that matches the study's stated question. Holm(3) is the")
            A("conservative bound; max-T sign-flip permutation preserves the positive")
            A("correlation induced by the shared control arm and is the paired analogue")
            A("of Dunnett's test.")
            A("")
            floor = 2 / (2 ** n)
            if floor >= 0.05:
                A(f"> **max-T is not usable at n={n}.** A sign-flip test's floor is")
                A(f"> 2/2^{n} = **{floor:.4f}**, because |t| is invariant under a global")
                A("> sign flip so the observed pattern always ties its own mirror. It")
                A("> cannot return p<0.05 for *any* data at this n, so only Holm is shown.")
                A("")
            A("| subset | contrast | mean Δ | p raw | Holm(3) | " +
              ("max-T |" if floor < 0.05 else "") )
            A("|---|---|---|---|---|" + ("---|" if floor < 0.05 else ""))
            for sname, k in SUBSETS:
                V = {arm: vals(cfg, arm, block, k) for arm in "abcd"}
                if any(v is None for v in V.values()):
                    continue
                raw3 = [stats.ttest_rel(V[t], V["a"]).pvalue for t in "bcd"]
                h3 = holm(raw3)
                if floor < 0.05:
                    D = np.column_stack([V[t] - V["a"] for t in "bcd"])
                    mt, _, _ = maxt_permutation(D)
                else:
                    mt = [None] * 3
                for i, t in enumerate("bcd"):
                    d = (V[t] - V["a"]).mean()
                    cell = (f" {mt[i]:.4f}{'  ✓' if mt[i] < 0.05 else ''} |"
                            if floor < 0.05 else "")
                    A(f"| {sname} | **{t} − a** | {d:+.4f} | {raw3[i]:.4f} | "
                      f"{h3[i]:.4f}{'  ✓' if h3[i] < 0.05 else ''} |{cell}")
            A("")

            # ---- omnibus ----
            if n >= 6:
                A(f"#### Omnibus (Friedman) — {dsname}")
                A("")
                A("| subset | chi² | p | mean ranks (1 = best) |")
                A("|---|---|---|---|")
                for sname, k in SUBSETS:
                    V = {arm: vals(cfg, arm, block, k) for arm in "abcd"}
                    chi, pv = stats.friedmanchisquare(*[V[a] for a in "abcd"])
                    M = np.column_stack([V[a] for a in "abcd"])
                    ranks = np.apply_along_axis(
                        lambda r: stats.rankdata(-r), 1, M).mean(axis=0)
                    rk = ", ".join(f"{a} {r:.2f}" for a, r in
                                   sorted(zip("abcd", ranks), key=lambda z: z[1]))
                    A(f"| {sname} | {chi:.2f} | {pv:.4f}{star(pv)} | {rk} |")
                A("")
            else:
                A(f"#### Omnibus — not run at n={n}")
                A("")
                A(f"Friedman/Nemenyi is powerless here: the Nemenyi critical difference")
                A(f"at k=4, n={n} exceeds the span four mean ranks can occupy. Report")
                A("these arms descriptively.")
                A("")

    # ================= summary ==============================================
    A("\n---\n")
    A("## What is actually significant")
    A("")
    cfg = PROTOCOLS["inter_patient"]
    hits = []
    for side, block in BLOCKS:
        dsname = cfg["src"] if side == "src" else cfg["tgt"]
        for sname, k in SUBSETS:
            V = {arm: vals(cfg, arm, block, k) for arm in "abcd"}
            pairs = list(itertools.combinations("abcd", 2))
            raw = [stats.ttest_rel(V[y], V[x]).pvalue for x, y in pairs]
            h6 = holm(raw)
            for (x, y), r, h in zip(pairs, raw, h6):
                if r < 0.05:
                    hits.append((dsname, sname, f"{y} − {x}",
                                 (V[y] - V[x]).mean(), r, h))
    A("Raw p<0.05 at n=10 inter-patient, across all 6 pairs × 2 datasets × 3 subsets")
    A(f"(**{len(hits)} of 36 cells**):")
    A("")
    A("| dataset | subset | contrast | mean Δ | p raw | Holm(6) |")
    A("|---|---|---|---|---|---|")
    for dsname, sname, pair, d, r, h in hits:
        A(f"| {dsname} | {sname} | **{pair}** | {d:+.4f} | {r:.4f} | "
          f"{h:.4f}{'  ✓' if h < 0.05 else '  ✗'} |")
    A("")
    A(SUMMARY_PROSE)

    path = os.path.join(OUTDIR, "model_comparison.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {path}  ({len(L)} lines)")


SUMMARY_PROSE = """
### How to state these three effects

**1. In-domain: `c − b` and `d − c` (robust).** Both survive Holm over all six
pairs — the strictest family — and both replicate across 5-, 4- and 3-class.
The defensible sentence is *"invariance training costs in-domain performance and
contrastive pretraining recovers it."* It is **not** a vs-baseline claim: every
`× − a` in-domain is n.s., and the means run d > b > **a** > c, i.e. the
baseline sits *between* the arms. Never write "contrastive pretraining wins
in-domain" without naming the opponent.

**2. Cross-dataset: `c − a` (family-dependent).** +0.0499, raw p=0.018,
**max-T p=0.035 under the 3-contrast vs-baseline family**, but p=0.108 under the
6-pair family. Same data, same contrast. The 3-contrast family is the defensible
one because it matches the stated objective and INCART 3-class was pre-specified
as the primary endpoint on 2026-08-11, before these runs existed — so **declare
the family before quoting the correction.**

**3. What is NOT shown.** Cross-dataset, the three interventions beat the
baseline by a *similar* margin (b−a +0.0391, c−a +0.0499, d−a +0.0431); what
separates c's p-value is the denominator, not the numerator. The direct
comparisons are flat (c−b p=0.444, d−b p=0.655, d−c p=0.642). Reading the gap
between a significant and a non-significant result as a ranking is the error
this project made at n=6 and retracted at n=10.

**`b − a` and `d − a` are underpowered, not null.** Per the recomputed power
table, `b − a` needs n≈22 on this variance. Report them as such.

**One informative disagreement.** INCART `c − b` is **9/10 seeds positive
(sign-test p=0.021)** while its paired t reads p=0.444 — c beats b nearly every
seed, but by only +0.0108. For a ranking that sign consistency is better
evidence than the p-values against a; it is still not a significant paired
difference and must not be reported as one.
"""


if __name__ == "__main__":
    main()
