"""All paired t-tests for the four-model comparison, across every class subset.

For each dataset (MIT-BIH val, INCART zero-shot) and each subset (5/4/3-class
macro-F1), runs the paired t-test over the same 6 seeds for all 6 model pairs.
Flags where the 4- and 5-class trends DIFFER from the 3-class one, since Q (and
to a lesser extent F) is structurally unlearnable cross-dataset and can flip or
mute a comparison that the 3-class view shows clearly.

Writes runs/_analysis/four_model_ttests.csv (all comparisons) and
runs/_analysis/four_model_significant.csv (p < 0.05 only).

Run: python code/four_model_ttests.py
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

# Arm tags differ per protocol, but the four MODELS are the same four models:
# only init (random vs contrastive-pretrained) and lambda_inv (0 vs 0.4) vary.
# Under --protocol random, b and d load pretrained_encoder_randsplit.pth, whose
# pretraining set is the random split's TRAIN side (see CLAUDE.md 8-31-26).
PROTOCOLS = {
    "inter_patient": {
        "label": "inter-patient (record-disjoint 38/10)",
        "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        "arms": [("a", "baseline", "focal_sampler"),
                 ("b", "contrastive+FT", "pretrained_focal_sampler"),
                 ("c", "invariance", "focal_inv04_sampler"),
                 ("d", "contrastive+inv", "pretrained_inv04_sampler")],
        "out": "four_model",
    },
    # REVERSE: source and target are swapped. metrics.json keeps the field
    # names mitbih_val/incart, but here mitbih_val is the INCART VALIDATION
    # block and incart is the MIT-BIH zero-shot TARGET. The dataset labels are
    # relabelled at print time below.
    "reverse": {
        "label": "REVERSE INCART->MIT-BIH (INCART record-disjoint 67/8)",
        "seeds": [42, 43, 44],
        "arms": [("a", "baseline", "reverse_focal_sampler"),
                 ("b", "contrastive+FT", "reverse_pretrained_focal_sampler"),
                 ("c", "invariance", "reverse_inv04_sampler"),
                 ("d", "contrastive+inv", "reverse_pretrained_inv04_sampler")],
        "out": "four_model_reverse",
    },
    "random": {
        "label": "random 8:2 pooled-beat (split_seed=0, stratified)",
        "seeds": [42, 43, 44],
        "arms": [("a", "baseline", "randsplit_focal_sampler"),
                 ("b", "contrastive+FT", "randsplit_pretrained_focal_sampler"),
                 ("c", "invariance", "randsplit_inv04_sampler"),
                 ("d", "contrastive+inv", "randsplit_pretrained_inv04_sampler")],
        "out": "four_model_randsplit",
    },
}
ARMS = PROTOCOLS["inter_patient"]["arms"]
SEEDS = PROTOCOLS["inter_patient"]["seeds"]
SUBSETS = [("5-class", 5), ("4-class", 4), ("3-class", 3)]
DATASETS = [("MIT", "mitbih_val"), ("INCART", "incart")]


def load(tag, seed):
    return json.load(open(f"runs/{tag}_seed{seed}/metrics.json"))


def macro(run, block, k):
    f = np.asarray(run[block]["macro_checkpoint"]["per_class_f1"], float)
    return f[:k].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="inter_patient",
                    help="Which split protocol's arms to compare. Arms from "
                         "DIFFERENT protocols must never be pooled or compared "
                         "as if they measured the same thing.")
    args = ap.parse_args()
    cfg = PROTOCOLS[args.protocol]

    global DATASETS
    if args.protocol == "reverse":
        DATASETS = [("INCARTval", "mitbih_val"), ("MITtarget", "incart")]

    global ARMS, SEEDS
    ARMS, SEEDS = cfg["arms"], cfg["seeds"]

    missing = [f"{tag}_seed{s}" for _, _, tag in ARMS for s in SEEDS
               if not os.path.exists(f"runs/{tag}_seed{s}/metrics.json")]
    if missing:
        raise SystemExit(
            f"{len(missing)} run(s) missing for --protocol {args.protocol}:\n  "
            + "\n  ".join(missing)
            + "\nComparing arms at unequal n is the error that produced a "
              "retracted finding on 8-30-26; refusing to report a partial table.")

    runs = {a[0]: [load(a[2], s) for s in SEEDS] for a in ARMS}
    rows = []
    for dname, block in DATASETS:
        for sname, k in SUBSETS:
            for i in range(len(ARMS)):
                for j in range(i + 1, len(ARMS)):
                    xi, xj = ARMS[i][0], ARMS[j][0]
                    x = np.array([macro(r, block, k) for r in runs[xi]])
                    y = np.array([macro(r, block, k) for r in runs[xj]])
                    d = y - x
                    sd = d.std(ddof=1)
                    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else np.nan
                    p = 2 * (1 - stats.t.cdf(abs(t), len(d) - 1)) if sd > 0 else np.nan
                    rows.append({
                        "dataset": dname, "subset": sname,
                        "pair": f"{xj} - {xi}",
                        "mean_A": round(float(x.mean()), 4),
                        "mean_B": round(float(y.mean()), 4),
                        "mean_diff": round(float(d.mean()), 4),
                        "sd_diff": round(float(sd), 4),
                        "t": round(float(t), 3),
                        "p": round(float(p), 4),
                        "wins": f"{int((d > 0).sum())}/{len(d)}",
                        "sig": "YES" if p < 0.05 else ("trend" if p < 0.10 else ""),
                    })

    # ── full table ────────────────────────────────────────────────────────────
    print("=" * 104)
    print(f"ALL PAIRED t-TESTS  |  four-model comparison, n={len(SEEDS)} seeds "
          f"({SEEDS[0]}-{SEEDS[-1]}), {cfg['label']}")
    print("=" * 104)
    for dname, _ in DATASETS:
        for sname, _ in SUBSETS:
            sel = [r for r in rows if r["dataset"] == dname and r["subset"] == sname]
            print(f"\n{dname} macro-F1 {sname}")
            print(f"  {'pair':<9}{'A mean':>9}{'B mean':>9}{'diff':>10}{'sd':>9}"
                  f"{'t':>8}{'p':>9}{'wins':>7}  sig")
            for r in sel:
                print(f"  {r['pair']:<9}{r['mean_A']:>9.4f}{r['mean_B']:>9.4f}"
                      f"{r['mean_diff']:>+10.4f}{r['sd_diff']:>9.4f}{r['t']:>8.2f}"
                      f"{r['p']:>9.4f}{r['wins']:>7}  {r['sig']}")

    # ── does the 4/5-class view disagree with 3-class? ────────────────────────
    print("\n" + "=" * 104)
    print("TREND CONSISTENCY — does the 4-/5-class view tell the same story as 3-class?")
    print("=" * 104)
    print(f"  {'dataset':<8}{'pair':<9}{'5-class':>22}{'4-class':>22}{'3-class':>22}  note")
    for dname, _ in DATASETS:
        for i in range(len(ARMS)):
            for j in range(i + 1, len(ARMS)):
                pair = f"{ARMS[j][0]} - {ARMS[i][0]}"
                cells = {}
                for sname, _ in SUBSETS:
                    r = next(x for x in rows if x["dataset"] == dname
                             and x["subset"] == sname and x["pair"] == pair)
                    cells[sname] = r
                signs = {s: np.sign(c["mean_diff"]) for s, c in cells.items()}
                sigs = {s: c["p"] < 0.05 for s, c in cells.items()}
                note = []
                if len(set(signs.values())) > 1:
                    note.append("SIGN FLIPS")
                if len(set(sigs.values())) > 1:
                    note.append("significance differs")
                fmt = lambda c: f"{c['mean_diff']:+.4f} p={c['p']:.3f}{'*' if c['p'] < 0.05 else ''}"
                print(f"  {dname:<8}{pair:<9}"
                      f"{fmt(cells['5-class']):>22}{fmt(cells['4-class']):>22}"
                      f"{fmt(cells['3-class']):>22}  {', '.join(note)}")

    # ── significant only ─────────────────────────────────────────────────────
    sig = [r for r in rows if r["p"] < 0.05]
    print("\n" + "=" * 104)
    print(f"SIGNIFICANT PAIRS (p < 0.05):  {len(sig)} of {len(rows)} comparisons")
    print("=" * 104)
    print(f"  {'dataset':<8}{'subset':<9}{'pair':<9}{'A':>9}{'B':>9}{'diff':>10}"
          f"{'t':>8}{'p':>9}{'wins':>7}")
    for r in sorted(sig, key=lambda r: r["p"]):
        print(f"  {r['dataset']:<8}{r['subset']:<9}{r['pair']:<9}{r['mean_A']:>9.4f}"
              f"{r['mean_B']:>9.4f}{r['mean_diff']:>+10.4f}{r['t']:>8.2f}"
              f"{r['p']:>9.4f}{r['wins']:>7}")

    os.makedirs("runs/_analysis", exist_ok=True)
    stem = cfg["out"]
    for path, data in ((f"runs/_analysis/{stem}_ttests.csv", rows),
                       (f"runs/_analysis/{stem}_significant.csv", sig)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"\nwrote {path}  ({len(data)} rows)")


if __name__ == "__main__":
    main()
