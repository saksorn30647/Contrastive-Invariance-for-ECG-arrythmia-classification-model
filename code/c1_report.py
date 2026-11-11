"""C1 session report: random-split arms vs their inter-patient counterparts.

Everything is recomputed from metrics.json rather than transcribed, per
CLAUDE.md's maintenance protocol ("hand-transcription is how numeric drift gets
into this file").
"""
import json

import numpy as np
from scipy import stats

SEEDS = [42, 43, 44]
KEYS = ["MIT 5c", "MIT 4c", "MIT 3c", "MIT F1(S)", "MIT F1(F)", "MIT rec(N)",
        "INCART 5c", "INCART 4c", "INCART 3c", "INCART F1(F)"]


def load(tag, seed):
    m = json.load(open(f"runs/{tag}_seed{seed}/metrics.json"))
    o = {}
    for ds, key in (("MIT", "mitbih_val"), ("INCART", "incart")):
        b = m[key]["macro_checkpoint"]
        f1 = b["per_class_f1"]
        # Runs predating the 5/4/3-class reporting have no "macro_subsets"
        # block; recompute from per_class_f1 so old and new arms are comparable.
        ms = b.get("macro_subsets") or {
            "macro5_NSVFQ": float(np.mean(f1[:5])),
            "macro4_NSVF": float(np.mean(f1[:4])),
            "macro3_NSV": float(np.mean(f1[:3])),
        }
        o[f"{ds} 5c"] = ms["macro5_NSVFQ"]
        o[f"{ds} 4c"] = ms["macro4_NSVF"]
        o[f"{ds} 3c"] = ms["macro3_NSV"]
        o[f"{ds} F1(S)"] = f1[1]
        o[f"{ds} F1(F)"] = f1[3]
        o[f"{ds} rec(N)"] = b["per_class_recall"][0]
    o["epochs"] = m.get("epochs_run")
    o["best_epoch"] = m["mitbih_val"]["macro_checkpoint"]["epoch"]
    return o


def agg(tag, seeds=SEEDS):
    d = {s: load(tag, s) for s in seeds}
    return {k: np.array([d[s][k] for s in seeds]) for k in list(KEYS) + ["epochs", "best_epoch"]}


arms = {
    "a_inter": "focal_sampler",
    "c_inter": "focal_inv04_sampler",
    "a_rand": "randsplit_focal_sampler",
    "c_rand": "randsplit_inv04_sampler",
}
D = {k: agg(v) for k, v in arms.items()}

print("=" * 92)
print("C1 — RANDOM SPLIT vs INTER-PATIENT, models a and c, n=3 (seeds 42/43/44)")
print("=" * 92)
hdr = f"{'metric':13s} {'a inter':>17s} {'a random':>17s} | {'c inter':>17s} {'c random':>17s}"
print(hdr)
print("-" * 92)
for k in KEYS:
    row = f"{k:13s}"
    for arm in ("a_inter", "a_rand", "c_inter", "c_rand"):
        v = D[arm][k]
        row += f" {v.mean():7.4f}±{v.std(ddof=1):.4f}"
        if arm == "a_rand":
            row += " |"
    print(row)
print()
print("epochs run     a inter", list(D['a_inter']['epochs'].astype(int)),
      " a random", list(D['a_rand']['epochs'].astype(int)))
print("               c inter", list(D['c_inter']['epochs'].astype(int)),
      " c random", list(D['c_rand']['epochs'].astype(int)))

# ── the two questions C1 exists to answer ────────────────────────────────────
print()
print("=" * 92)
print("Q1. Does the lambda_inv effect (c - a) survive the protocol change?")
print("=" * 92)
print(f"{'metric':13s} {'inter-patient':>26s} {'random 8:2':>26s}")
print("-" * 92)
for k in ["MIT 3c", "MIT F1(F)", "INCART 5c", "INCART 4c", "INCART 3c"]:
    out = f"{k:13s}"
    for proto in ("inter", "rand"):
        a, c = D[f"a_{proto}"][k], D[f"c_{proto}"][k]
        d = c - a
        t, p = stats.ttest_rel(c, a)
        out += f"  {d.mean():+8.4f} {int((d > 0).sum())}/3 p={p:.3f}"
    print(out)

print()
print("=" * 92)
print("Q2. The protocol gap (random - inter), same seeds, same script")
print("=" * 92)
print(f"{'metric':13s} {'a: inter -> random':>34s} {'c: inter -> random':>34s}")
print("-" * 92)
for k in KEYS:
    out = f"{k:13s}"
    for arm in ("a", "c"):
        i, r = D[f"{arm}_inter"][k], D[f"{arm}_rand"][k]
        d = r - i
        t, p = stats.ttest_rel(r, i)
        out += f"   {i.mean():6.4f} -> {r.mean():6.4f}  {d.mean():+7.4f} p={p:.3f}"
    print(out)

print()
print("RATIO of in-domain jump to cross-dataset jump (the control argument):")
for arm in ("a", "c"):
    for indom, cross in (("MIT 4c", "INCART 3c"), ("MIT 3c", "INCART 3c")):
        di = (D[f"{arm}_rand"][indom] - D[f"{arm}_inter"][indom]).mean()
        dc = (D[f"{arm}_rand"][cross] - D[f"{arm}_inter"][cross]).mean()
        print(f"  {arm}: {indom} {di:+.4f} vs {cross} {dc:+.4f}  ->  {di / dc:.1f}x")

# ── b at n=1 ─────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("b (contrastive pretrained, random-split encoder) — n=1, seed 42 only")
print("=" * 92)
b = load("randsplit_pretrained_focal_sampler", 42)
for k in KEYS:
    print(f"  {k:13s} {b[k]:.4f}")
print(f"  epochs {b['epochs']}  best_epoch {b['best_epoch']}")
