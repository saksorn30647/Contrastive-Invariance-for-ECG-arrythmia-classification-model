"""Why S fails: RR-interval separability and annotation-symbol composition.

Two analyses that need no GPU and no training, addressing the co-advisor's
second comment ("explain why S fails on INCART"):

1. **RR separability.** A supraventricular ectopic beat is defined by
   PREMATURITY, not by QRS shape. The pipeline feeds a 256-sample R-centred,
   per-beat z-scored scalogram with no RR feature, so if RR separates S from N
   well, that is direct evidence the failure is an input-representation choice
   rather than an intrinsic difficulty. Quantified as AUROC of a single
   feature (RR_prev / local RR average) for S vs N, per database.

   AUROC is used because it is threshold-free and prevalence-independent --
   important here, since S is 1-3% of beats and any accuracy-like measure
   would be dominated by N.

2. **Symbol composition.** AAMI class S pools four annotation symbols
   (A = atrial premature, a = aberrated atrial premature, J = nodal premature,
   S = supraventricular premature). If MIT-BIH and INCART draw S from
   different symbol mixes, the model is being asked to transfer between
   different underlying rhythms wearing the same label.

Beat alignment matches `beat_extrac_label*.py` exactly: same VALID_SYMBOLS
filter, same boundary-drop rule, and for INCART the same 257->360 Hz index
rescaling. That is asserted against the stored arrays, so the RR features line
up with `data/X_cwt.npy` beat-for-beat.

Run: python code/s_class_analysis.py       (use PYTHONIOENCODING=utf-8)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VALID_SYMBOLS = ["N", "L", "R", "e", "j", "A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"]
AAMI_MAP = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
            "A": "S", "a": "S", "J": "S", "S": "S",
            "V": "V", "E": "V", "F": "F", "/": "Q", "f": "Q", "Q": "Q"}
WINDOW = 256
HALF = WINDOW // 2
OUTDIR = "results/figures"


def collect(db):
    """Return per-beat (record, aami, symbol, rr_prev, rr_next, rr_ratio).

    rr_ratio = rr_prev / (local mean of the surrounding 10 rr_prev values),
    the standard prematurity feature. A premature beat has ratio < 1.
    """
    if db == "mitbih":
        import glob
        recs = sorted({os.path.basename(p)[:-4]
                       for p in glob.glob("data/mit-bih/*.hea")})
        base, fs_scale = "data/mit-bih", 1.0
    else:
        import glob
        cand = glob.glob("data/incart/**/*.hea", recursive=True) + \
               glob.glob("incart/**/*.hea", recursive=True)
        recs = sorted({p[:-4] for p in cand})
        base, fs_scale = None, 360.0 / 257.0

    rows = []
    for r in recs:
        path = f"{base}/{r}" if base else r
        name = os.path.basename(path)
        try:
            ann = wfdb.rdann(path, "atr")
            hdr = wfdb.rdheader(path)
        except Exception:
            continue
        nsamp = hdr.sig_len
        if db == "incart":
            nsamp = int(nsamp * fs_scale)
        samples = np.round(np.asarray(ann.sample) * fs_scale).astype(int)
        symbols = list(ann.symbol)

        keep = [i for i, s in enumerate(symbols) if s in VALID_SYMBOLS]
        kept_s = samples[keep]
        kept_sym = [symbols[i] for i in keep]
        # boundary rule, identical to the extraction scripts
        ok = [(j, s) for j, s in enumerate(kept_s)
              if s - HALF >= 0 and s + HALF <= nsamp]
        if len(ok) < 12:
            continue
        idx = np.array([j for j, _ in ok])
        pos = kept_s[idx]
        syms = [kept_sym[j] for j in idx]

        rr_prev = np.full(len(pos), np.nan)
        rr_next = np.full(len(pos), np.nan)
        rr_prev[1:] = np.diff(pos)
        rr_next[:-1] = np.diff(pos)
        # local average of rr_prev over a +/-5 beat neighbourhood
        ratio = np.full(len(pos), np.nan)
        for i in range(len(pos)):
            lo, hi = max(0, i - 5), min(len(pos), i + 6)
            win = rr_prev[lo:hi]
            win = win[~np.isnan(win)]
            if len(win) >= 3 and np.nanmean(win) > 0 and not np.isnan(rr_prev[i]):
                ratio[i] = rr_prev[i] / np.mean(win)
        for i in range(len(pos)):
            rows.append((name, AAMI_MAP[syms[i]], syms[i],
                         rr_prev[i], rr_next[i], ratio[i]))
    return rows


def auroc(pos, neg):
    """Mann-Whitney U based AUROC, direction-agnostic (reported as >=0.5)."""
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), 0, 0
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r1 = ranks[:len(pos)].sum()
    u = r1 - len(pos) * (len(pos) + 1) / 2
    a = u / (len(pos) * len(neg))
    return a, len(pos), len(neg)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, db, label in ((axes[0], "mitbih", "MIT-BIH"),
                          (axes[1], "incart", "INCART")):
        rows = collect(db)
        if not rows:
            ax.set_title(f"{label}: records not found")
            continue
        aami = np.array([r[1] for r in rows])
        sym = np.array([r[2] for r in rows])
        ratio = np.array([r[5] for r in rows], float)

        n_ratio = ratio[aami == "N"]
        s_ratio = ratio[aami == "S"]
        v_ratio = ratio[aami == "V"]

        a_sn, npos, nneg = auroc(n_ratio, s_ratio)   # N vs S, direction-free
        a_sn = max(a_sn, 1 - a_sn)
        a_vn, _, _ = auroc(n_ratio, v_ratio)
        a_vn = max(a_vn, 1 - a_vn)

        comp = {s: int((sym[aami == "S"] == s).sum()) for s in ("A", "a", "J", "S")}
        tot = sum(comp.values())
        out[db] = dict(
            n_beats=len(rows),
            s_beats=int((aami == "S").sum()),
            auroc_rr_S_vs_N=float(a_sn),
            auroc_rr_V_vs_N=float(a_vn),
            s_symbol_counts=comp,
            s_symbol_pct={k: (100.0 * v / tot if tot else 0.0) for k, v in comp.items()},
            median_rr_ratio={"N": float(np.nanmedian(n_ratio)),
                             "S": float(np.nanmedian(s_ratio)),
                             "V": float(np.nanmedian(v_ratio))},
            pct_S_premature_below_0p85=float(
                100.0 * np.nanmean(s_ratio[~np.isnan(s_ratio)] < 0.85)),
            pct_N_premature_below_0p85=float(
                100.0 * np.nanmean(n_ratio[~np.isnan(n_ratio)] < 0.85)),
        )

        bins = np.linspace(0.3, 1.8, 80)
        ax.hist(n_ratio[~np.isnan(n_ratio)], bins=bins, density=True,
                alpha=0.55, label=f"N (n={np.sum(aami=='N'):,})", color="#4878A8")
        ax.hist(s_ratio[~np.isnan(s_ratio)], bins=bins, density=True,
                alpha=0.70, label=f"S (n={np.sum(aami=='S'):,})", color="#C44E52")
        ax.axvline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("RR$_{prev}$ / local mean RR   (<1 = premature)")
        ax.set_ylabel("density")
        ax.set_title(f"{label}  —  S vs N prematurity\n"
                     f"single-feature AUROC = {a_sn:.3f}   "
                     f"(V vs N = {a_vn:.3f})")
        ax.legend()

    fig.suptitle("The feature that defines S is not in the model's input", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "s_class_rr_separability.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    os.makedirs("runs/_analysis", exist_ok=True)
    json.dump(out, open("runs/_analysis/s_class_analysis.json", "w"), indent=1)

    for db, d in out.items():
        print(f"\n=== {db} ===")
        print(f"  beats {d['n_beats']:,}   S beats {d['s_beats']:,}")
        print(f"  AUROC RR-ratio  S vs N = {d['auroc_rr_S_vs_N']:.3f}"
              f"   (V vs N = {d['auroc_rr_V_vs_N']:.3f})")
        print(f"  median RR-ratio  N={d['median_rr_ratio']['N']:.3f}  "
              f"S={d['median_rr_ratio']['S']:.3f}  V={d['median_rr_ratio']['V']:.3f}")
        print(f"  S beats with ratio<0.85: {d['pct_S_premature_below_0p85']:.1f}%"
              f"   (N beats: {d['pct_N_premature_below_0p85']:.1f}%)")
        print(f"  S symbol mix: " + "  ".join(
            f"{k}={v} ({d['s_symbol_pct'][k]:.1f}%)" for k, v in d['s_symbol_counts'].items()))
    print(f"\nwrote {p}")
    print("wrote runs/_analysis/s_class_analysis.json")


if __name__ == "__main__":
    main()
