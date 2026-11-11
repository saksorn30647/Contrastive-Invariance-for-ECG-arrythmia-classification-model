"""Print the four-model comparison table at n=6, matched config.

a. baseline                     = focal_sampler              (random init,     lambda_inv=0)
b. contrastive pretrained + FT  = pretrained_focal_sampler   (pretrained init, lambda_inv=0)
c. invariance training          = focal_inv04_sampler        (random init,     lambda_inv=0.4)
d. contrastive + invariance     = pretrained_inv04_sampler   (pretrained init, lambda_inv=0.4)

All four share: focal gamma=2.0, no class weights, MinorityGuaranteedBatchSampler
(F floor=1), batch 64, lr 1e-4, wd 1e-5, seeds 42-47, 5-class macro checkpoint,
inter-patient 38/10 split. ONLY init and lambda_inv differ.

Run: python code/four_model_table.py [--per-seed]
"""
import argparse
import json

import numpy as np
from scipy import stats

ARMS = [("a. baseline", "focal_sampler"),
        ("b. contrastive+FT", "pretrained_focal_sampler"),
        ("c. invariance", "focal_inv04_sampler"),
        ("d. contrastive+inv", "pretrained_inv04_sampler")]
SEEDS = [42, 43, 44, 45, 46, 47]


def L(t, s):
    return json.load(open(f"runs/{t}_seed{s}/metrics.json"))


def sub(f, k):
    f = np.asarray(f, float)
    return {"m5": f.mean(), "m4": f[:4].mean(), "m3": f[:3].mean()}[k]


V = lambda r: r["mitbih_val"]["macro_checkpoint"]
I = lambda r: r["incart"]["macro_checkpoint"]
METRICS = [
    ("MIT-BIH val", None),
    ("  macro-F1 5-class (N/S/V/F/Q)", lambda r: sub(V(r)["per_class_f1"], "m5")),
    ("  macro-F1 4-class (N/S/V/F)", lambda r: sub(V(r)["per_class_f1"], "m4")),
    ("  macro-F1 3-class (N/S/V)", lambda r: sub(V(r)["per_class_f1"], "m3")),
    ("  F1(N)", lambda r: V(r)["per_class_f1"][0]),
    ("  F1(S)", lambda r: V(r)["per_class_f1"][1]),
    ("  F1(V)", lambda r: V(r)["per_class_f1"][2]),
    ("  F1(F)", lambda r: V(r)["per_class_f1"][3]),
    ("  F1(Q)", lambda r: V(r)["per_class_f1"][4]),
    ("  recall(N)", lambda r: V(r)["per_class_recall"][0]),
    ("INCART zero-shot", None),
    ("  macro-F1 5-class (N/S/V/F/Q)", lambda r: sub(I(r)["per_class_f1"], "m5")),
    ("  macro-F1 4-class (N/S/V/F)", lambda r: sub(I(r)["per_class_f1"], "m4")),
    ("  macro-F1 3-class (N/S/V)", lambda r: sub(I(r)["per_class_f1"], "m3")),
    ("  F1(N)", lambda r: I(r)["per_class_f1"][0]),
    ("  F1(S)", lambda r: I(r)["per_class_f1"][1]),
    ("  F1(V)", lambda r: I(r)["per_class_f1"][2]),
    ("  F1(F)", lambda r: I(r)["per_class_f1"][3]),
    ("Other", None),
    ("  epochs to best ckpt", lambda r: float(r["epochs_run"])),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", action="store_true")
    args = ap.parse_args()

    runs = {t: [L(t, s) for s in SEEDS] for _, t in ARMS}

    print("=" * 100)
    print("FOUR-MODEL COMPARISON  |  MIT-BIH inter-patient (38/10)  |  n=6 seeds (42-47)")
    print("matched config: focal g=2.0, no class weights, sampler F-floor=1, batch 64, lr 1e-4")
    print("only INIT (random vs contrastive-pretrained) and LAMBDA_INV (0 vs 0.4) differ")
    print("=" * 100)
    print(f"{'metric':<32}" + "".join(f"{n:>17}" for n, _ in ARMS))
    print("-" * 100)
    for name, fn in METRICS:
        if fn is None:
            print(f"\n{name}")
            continue
        vals = [[fn(r) for r in runs[t]] for _, t in ARMS]
        means = [np.mean(v) for v in vals]
        best = int(np.argmax(means))
        line = f"{name:<32}"
        for i, v in enumerate(vals):
            mark = "*" if i == best else " "
            line += f"{np.mean(v):>10.4f}±{np.std(v, ddof=1):.4f}{mark}"
        print(line)
    print("-" * 100)
    print("* = best mean.  ± is std over 6 seeds (NOT a confidence interval;")
    print("  for INCART use record-level bootstrap CIs, ~±0.033 on 3-class).")

    print("\n" + "=" * 100)
    print("PAIRED SIGNIFICANCE (paired t over the same 6 seeds)")
    print("=" * 100)
    keys = {"INCART 3-class": lambda r: sub(I(r)["per_class_f1"], "m3"),
            "MIT 3-class": lambda r: sub(V(r)["per_class_f1"], "m3")}
    for mn, fn in keys.items():
        print(f"\n{mn}")
        print(f"  {'pair':<12}{'mean diff':>11}{'t':>8}{'p':>9}{'wins':>7}   verdict")
        for i in range(4):
            for j in range(i + 1, 4):
                x = np.array([fn(r) for r in runs[ARMS[i][1]]])
                y = np.array([fn(r) for r in runs[ARMS[j][1]]])
                d = y - x
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                p = 2 * (1 - stats.t.cdf(abs(t), len(d) - 1))
                tag = "SIGNIFICANT" if p < 0.05 else ("trend" if p < 0.10 else "ns")
                nm = f"{ARMS[j][0][:1]} - {ARMS[i][0][:1]}"
                print(f"  {nm:<12}{d.mean():>+11.4f}{t:>8.2f}{p:>9.4f}"
                      f"{int((d > 0).sum()):>5}/6   {tag}")

    if args.per_seed:
        print("\n" + "=" * 100)
        print("PER-SEED INCART 3-class")
        print("=" * 100)
        print(f"  {'model':<20}" + "".join(f"{s:>9}" for s in SEEDS))
        for n, t in ARMS:
            v = [sub(I(r)["per_class_f1"], "m3") for r in runs[t]]
            print(f"  {n:<20}" + "".join(f"{x:>9.4f}" for x in v))


if __name__ == "__main__":
    main()
