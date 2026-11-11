"""
Aggregate metrics.json across seeded runs.

The point of this script is to answer one question the individual run logs
cannot: is an effect real, or is it seed luck? This project has documented
run-to-run swings large enough to invert conclusions (S recall 0.04 vs 0.60
across *identical* configs), so single-run numbers are treated as noisy
samples and only mean +/- std across seeds is quoted.

It also re-derives the "balanced" checkpoint under a minimum-epoch floor
(--min-epoch). The live selection rule picks the epoch with the best
worst-class F1 with no epoch floor, which has repeatedly landed on epoch 2-5
— early enough that the model's predictions are still diffuse enough to hit
every class, which trivially wins that metric. Because each run saves its
full per-epoch history, that hypothesis can be tested here without spending
any GPU time re-running anything.

Usage (from project root):
    python code/aggregate_runs.py
    python code/aggregate_runs.py --runs-dir runs --min-epoch 5
"""

import argparse
import glob
import json
import os

import numpy as np

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Reference points from CLAUDE.md, for context only. Round 1 = focal loss with
# NO sampler and NO fixed seed (two runs). These are not directly comparable to
# seeded runs — they're shown so the sweep's numbers can be read against the
# baseline they're meant to improve on.
ROUND1_REFERENCE = {
    "run1": {"mitbih_macro_f1": 0.5084, "f1_S": 0.00, "f1_F": 0.00,
             "incart_macro_f1": 0.3416, "balanced_worst_f1": 0.0366},
    "run2": {"mitbih_macro_f1": 0.4677, "f1_S": 0.00, "f1_F": 0.00,
             "incart_macro_f1": 0.3358, "balanced_worst_f1": 0.1775},
}


def load_runs(runs_dir, tag=None):
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        with open(path) as fh:
            data = json.load(fh)
        if tag is not None and data["config"].get("tag") != tag:
            continue
        data["_path"] = path
        data["_name"] = os.path.basename(os.path.dirname(path))
        runs.append(data)
    runs.sort(key=lambda r: (r["config"].get("tag", ""), r["config"]["seed"]))
    return runs


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def macro_subsets(block):
    """Macro-F1 over class subsets for a checkpoint block.

    Uses the stored `macro_subsets` when present, and otherwise derives it from
    `per_class_f1` so runs recorded before that field existed still aggregate.
    Q is source-domain-only (6 INCART beats) and F is single-patient under this
    split, so the 5-class figure carries two structurally-limited classes.
    """
    if block is None:
        return None
    if block.get("macro_subsets"):
        return block["macro_subsets"]
    f = block.get("per_class_f1")
    if not f:
        return None
    return {
        "macro5_NSVFQ": float(np.mean(f)),
        "macro4_NSVF": float(np.mean(f[:4])),
        "macro3_NSV": float(np.mean(f[:3])),
    }


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def summarize(label, values, nd=4):
    m, s = mean_std(values)
    if m is None:
        return f"  {label:<34} —"
    per_seed = ", ".join(fmt(v, nd) for v in values)
    return f"  {label:<34} {fmt(m, nd)} +/- {fmt(s, nd)}   [{per_seed}]"


def reselect_balanced(history, min_epoch):
    """Re-derive the balanced checkpoint restricted to epochs >= min_epoch.

    Mirrors the live rule (maximize the worst class's F1) but with an epoch
    floor, to test whether the very early epochs the live rule keeps choosing
    are a selection artifact rather than a genuinely better model.
    """
    eligible = [h for h in history if h["epoch"] >= min_epoch]
    if not eligible:
        return None
    return max(eligible, key=lambda h: min(h["per_class_f1"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--min-epoch", type=int, default=5,
                    help="Epoch floor for the post-hoc balanced re-selection.")
    ap.add_argument("--tag", default=None,
                    help="Only aggregate runs with this config tag. Use it to keep "
                         "separate experiment arms apart — averaging across arms "
                         "mixes conditions and the config-consistency check below "
                         "will warn about it.")
    args = ap.parse_args()

    runs = load_runs(args.runs_dir, tag=args.tag)
    if not runs:
        where = f"{args.runs_dir}/*/"
        extra = f" with tag={args.tag!r}" if args.tag else ""
        print(f"No metrics.json found under {where}{extra} — nothing to aggregate.")
        return

    print("=" * 100)
    print(f"Aggregating {len(runs)} run(s) from {args.runs_dir}/")
    print("=" * 100)

    # ── Config consistency ────────────────────────────────────────────────────
    # Averaging across runs is only meaningful if they share a config. Any field
    # that differs (other than the seed) is surfaced loudly rather than silently
    # folded into a mean.
    keys = [k for k in runs[0]["config"] if k != "seed"]
    differing = {k: sorted({json.dumps(r["config"].get(k)) for r in runs})
                 for k in keys}
    differing = {k: v for k, v in differing.items() if len(v) > 1}
    if differing:
        print("\n!! WARNING — runs do NOT share an identical config. "
              "Aggregating across them mixes conditions:")
        for k, v in differing.items():
            print(f"     {k}: {v}")
    else:
        c = runs[0]["config"]
        print(f"\nShared config: cls_loss={c['cls_loss']}(gamma={c['focal_gamma']}), "
              f"class_weights={c['use_class_weights']}, lambda_inv={c['lambda_inv']}, "
              f"batch={c['batch_size']}, lr={c['lr']}, min_per_class={c['min_per_class']}, "
              f"cudnn_deterministic={c.get('cudnn_deterministic')}")
    print(f"Seeds: {[r['config']['seed'] for r in runs]}")

    # ── Sampler liveness ──────────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("EPOCH-1 F EXPOSURE — confirms the sampler was active, or that it was correctly absent")
    print("-" * 100)
    for r in runs:
        d = r.get("sampler_diag")
        if not d:
            print(f"  seed {r['config']['seed']:<4} (no diagnostic recorded)")
            continue
        pct = 100.0 * d["zero_F_batches"] / d["batches"]
        # Expectation depends on whether a floor was requested. With a floor,
        # zero-F batches must be 0; without one, F reverts to its natural ~0.5%
        # frequency and ~72.7% of batches contain no F at all (Poisson).
        has_floor = bool(r["config"].get("min_per_class"))
        if has_floor:
            flag = "" if d["zero_F_batches"] == 0 else "   <-- UNEXPECTED, sampler floor not honoured"
        else:
            flag = "   (no sampler — natural exposure, ~72.7% expected)"
        print(f"  seed {r['config']['seed']:<4} batches={d['batches']:<6} "
              f"zero-F batches={d['zero_F_batches']:<4} ({pct:.1f}%)   "
              f"avg F/batch={d['avg_F_per_batch']:.3f}{flag}")

    # ── Per-run detail ────────────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("PER-RUN DETAIL")
    print("-" * 100)
    # MIT-BIH and INCART macro-F1 are each shown at all three class subsets.
    # 5c is what selection acts on; 4c keeps F but drops Q (the view to read F
    # on); 3c is the cross-dataset headline.
    hdr = (f"  {'seed':<6}{'ep':<5}{'MIT 5c':<8}{'MIT 4c':<8}{'MIT 3c':<8}"
           f"{'F1(F)':<8}{'rec(F)':<8}{'F1(S)':<8}{'rec(N)':<8}"
           f"{'INC 5c':<8}{'INC 4c':<8}{'INC 3c':<8}{'INC F1(F)':<10}{'3c drop':<8}")
    for policy, vkey in (("MACRO-F1 CHECKPOINT", "macro_checkpoint"),
                         ("BALANCED CHECKPOINT", "balanced_checkpoint")):
        print(f"\n  {policy}")
        print(hdr)
        for r in runs:
            v = r["mitbih_val"][vkey]
            inc = r["incart"][vkey]
            if v is None:
                print(f"  {r['config']['seed']:<6}(not recorded)")
                continue
            vm = macro_subsets(v)
            im = macro_subsets(inc)
            f1f = v["per_class_f1"][IDX["F"]]
            recf = v["per_class_recall"][IDX["F"]]
            f1s = v["per_class_f1"][IDX["S"]]
            recn = v["per_class_recall"][IDX["N"]]
            inc_f1f = inc["per_class_f1"][IDX["F"]] if inc else None
            drop3 = (vm["macro3_NSV"] - im["macro3_NSV"]) if im else None
            print(f"  {r['config']['seed']:<6}{v['epoch']:<5}"
                  f"{vm['macro5_NSVFQ']:<8.4f}{vm['macro4_NSVF']:<8.4f}{vm['macro3_NSV']:<8.4f}"
                  f"{f1f:<8.4f}{recf:<8.4f}{f1s:<8.4f}{recn:<8.4f}"
                  + (f"{im['macro5_NSVFQ']:<8.4f}{im['macro4_NSVF']:<8.4f}"
                     f"{im['macro3_NSV']:<8.4f}" if im else f"{'—':<8}{'—':<8}{'—':<8}")
                  + f"{fmt(inc_f1f):<10}{fmt(drop3):<8}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    for policy, vkey in (("MACRO-F1 CHECKPOINT", "macro_checkpoint"),
                         ("BALANCED CHECKPOINT", "balanced_checkpoint")):
        print("\n" + "-" * 100)
        print(f"MEAN +/- STD ACROSS SEEDS — {policy}")
        print("-" * 100)
        vs = [r["mitbih_val"][vkey] for r in runs]
        incs = [r["incart"][vkey] for r in runs]
        get = lambda v, k, c: v[k][IDX[c]] if v else None

        print(summarize("MIT-BIH macro-F1", [v["macro_f1"] if v else None for v in vs]))
        print(summarize("MIT-BIH F1(F)", [get(v, "per_class_f1", "F") for v in vs]))
        print(summarize("MIT-BIH recall(F)", [get(v, "per_class_recall", "F") for v in vs]))
        print(summarize("MIT-BIH F1(S)", [get(v, "per_class_f1", "S") for v in vs]))
        print(summarize("MIT-BIH recall(S)", [get(v, "per_class_recall", "S") for v in vs]))
        print(summarize("MIT-BIH recall(N)", [get(v, "per_class_recall", "N") for v in vs]))
        print(summarize("INCART macro-F1", [i["macro_f1"] if i else None for i in incs]))
        print(summarize("INCART F1(F)", [get(i, "per_class_f1", "F") for i in incs]))
        print(summarize("selected epoch", [float(v["epoch"]) if v else None for v in vs], nd=1))
        drops = [(v["macro_f1"] - i["macro_f1"]) if (v and i) else None
                 for v, i in zip(vs, incs)]
        print(summarize("domain-shift drop", drops))

    # ── Class-subset view ─────────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("CLASS-SUBSET MACRO-F1 — macro-F1 checkpoint")
    print("(Checkpoints are still SELECTED on 5-class MIT-BIH val macro-F1. These are "
          "reporting views.\n Q has 6 INCART beats and is source-domain-only; F is "
          "single-patient under this split.)")
    print("-" * 100)
    print(f"  {'seed':<6}{'MIT 5cls':<11}{'MIT 4cls':<11}{'MIT 3cls':<11}"
          f"{'INC 5cls':<11}{'INC 4cls':<11}{'INC 3cls':<11}{'3cls drop':<11}")
    agg = {k: {"mit": [], "inc": [], "drop": []}
           for k in ("macro5_NSVFQ", "macro4_NSVF", "macro3_NSV")}
    for r in runs:
        vm = macro_subsets(r["mitbih_val"]["macro_checkpoint"])
        im = macro_subsets(r["incart"]["macro_checkpoint"])
        if vm is None or im is None:
            print(f"  {r['config']['seed']:<6}(insufficient per-class data)")
            continue
        for k in agg:
            agg[k]["mit"].append(vm[k])
            agg[k]["inc"].append(im[k])
            agg[k]["drop"].append(vm[k] - im[k])
        print(f"  {r['config']['seed']:<6}"
              f"{vm['macro5_NSVFQ']:<11.4f}{vm['macro4_NSVF']:<11.4f}{vm['macro3_NSV']:<11.4f}"
              f"{im['macro5_NSVFQ']:<11.4f}{im['macro4_NSVF']:<11.4f}{im['macro3_NSV']:<11.4f}"
              f"{vm['macro3_NSV'] - im['macro3_NSV']:<11.4f}")
    print()
    for k, name in (("macro5_NSVFQ", "5-class N/S/V/F/Q"),
                    ("macro4_NSVF", "4-class N/S/V/F"),
                    ("macro3_NSV", "3-class N/S/V")):
        if not agg[k]["inc"]:
            continue
        print(summarize(f"{name}  MIT-BIH val", agg[k]["mit"]))
        print(summarize(f"{name}  INCART", agg[k]["inc"]))
        print(summarize(f"{name}  drop", agg[k]["drop"]))
        print()

    # ── Post-hoc balanced re-selection ────────────────────────────────────────
    print("\n" + "-" * 100)
    print(f"POST-HOC: balanced checkpoint re-selected with epoch >= {args.min_epoch}")
    print("(re-derived from saved history — no re-training. Tests whether the live "
          "rule's very early picks are a selection artifact.)")
    print("-" * 100)
    print(f"  {'seed':<6}{'live ep':<10}{'live worst':<13}"
          f"{'reselect ep':<14}{'reselect worst':<16}{'reselect F1(F)':<16}{'reselect macroF1':<16}")
    re_worst, re_f1f = [], []
    for r in runs:
        live = r["mitbih_val"]["balanced_checkpoint"]
        h = reselect_balanced(r["history"], args.min_epoch)
        if h is None:
            print(f"  {r['config']['seed']:<6}(no epochs >= {args.min_epoch})")
            continue
        w = min(h["per_class_f1"])
        re_worst.append(w)
        re_f1f.append(h["per_class_f1"][IDX["F"]])
        print(f"  {r['config']['seed']:<6}"
              f"{(live['epoch'] if live else '—'):<10}"
              f"{fmt(live['min_class_f1']) if live else '—':<13}"
              f"{h['epoch']:<14}{w:<16.4f}"
              f"{h['per_class_f1'][IDX['F']]:<16.4f}{h['val_macro_f1']:<16.4f}")
    if re_worst:
        print()
        print(summarize(f"re-selected worst-class F1", re_worst))
        print(summarize(f"re-selected F1(F)", re_f1f))

    # ── Round-1 reference ─────────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("REFERENCE — round 1 (focal, NO sampler, NO fixed seed) from CLAUDE.md")
    print("-" * 100)
    for name, m in ROUND1_REFERENCE.items():
        print(f"  {name}: MIT-BIH macro-F1={m['mitbih_macro_f1']:.4f}  "
              f"F1(S)={m['f1_S']:.2f}  F1(F)={m['f1_F']:.2f}  "
              f"INCART macro-F1={m['incart_macro_f1']:.4f}  "
              f"balanced worst-F1={m['balanced_worst_f1']:.4f}")
    print("\n  Note: round-1 runs were unseeded and used no sampler. Treat as a rough "
          "baseline,\n  not a matched control — no matched no-sampler seeded arm was run.")
    print()


if __name__ == "__main__":
    main()
