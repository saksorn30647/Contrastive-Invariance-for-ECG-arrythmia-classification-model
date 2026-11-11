"""Per-run checkpoint summary — print the headline numbers as soon as one seed
finishes, instead of waiting for a whole sweep to end.

Sweeps in this project take 5-14 h. Without this you get "finished in 47 min"
and nothing else until every seed is done, which means a broken or obviously
bad arm burns the whole session before anyone notices.

Prints, for the seed just finished: config provenance (so a mislabelled run is
caught immediately), the sampler diagnostic, MIT 5/4/3-class + F1(S)/F1(F),
INCART 5/4/3-class, and the RUNNING mean +- std over every seed of that tag
completed so far.

Usage (called by the runner after each seed):
  python code/run_summary.py --tag focal_sampler --seed 45
"""

import argparse
import glob
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, help="Seed just finished; omit to summarise all.")
    p.add_argument("--checkpoint", default="macro_checkpoint",
                   choices=["macro_checkpoint", "balanced_checkpoint"])
    return p.parse_args()


def load(tag, seed):
    path = os.path.join("runs", f"{tag}_seed{seed}", "metrics.json")
    return json.load(open(path)) if os.path.exists(path) else None


def sub(f, k):
    f = np.asarray(f, dtype=float)
    return {"m5": f.mean(), "m4": f[:4].mean(), "m3": f[:3].mean()}[k]


METRICS = [
    ("MIT 5-class", lambda r, c: sub(r["mitbih_val"][c]["per_class_f1"], "m5")),
    ("MIT 4-class", lambda r, c: sub(r["mitbih_val"][c]["per_class_f1"], "m4")),
    ("MIT 3-class", lambda r, c: sub(r["mitbih_val"][c]["per_class_f1"], "m3")),
    ("MIT F1(S)", lambda r, c: r["mitbih_val"][c]["per_class_f1"][1]),
    ("MIT F1(F)", lambda r, c: r["mitbih_val"][c]["per_class_f1"][3]),
    ("MIT rec(N)", lambda r, c: r["mitbih_val"][c]["per_class_recall"][0]),
    ("INCART 5-class", lambda r, c: sub(r["incart"][c]["per_class_f1"], "m5")),
    ("INCART 4-class", lambda r, c: sub(r["incart"][c]["per_class_f1"], "m4")),
    ("INCART 3-class", lambda r, c: sub(r["incart"][c]["per_class_f1"], "m3")),
]


def main():
    args = parse_args()
    ck = args.checkpoint

    seeds = []
    for d in sorted(glob.glob(os.path.join("runs", f"{args.tag}_seed*"))):
        base = os.path.basename(d)
        if os.path.exists(os.path.join(d, "metrics.json")):
            try:
                seeds.append(int(base.rsplit("seed", 1)[1]))
            except ValueError:
                pass
    seeds.sort()

    print("=" * 78)
    print(f"  CHECKPOINT SUMMARY - tag={args.tag}"
          + (f", seed {args.seed} just finished" if args.seed else ""))
    print("=" * 78)

    if args.seed is not None:
        r = load(args.tag, args.seed)
        if r is None:
            print(f"  !! no metrics.json for seed {args.seed} - run did not complete")
            print("=" * 78)
            return
        c = r["config"]
        print(f"  config : loss={c.get('cls_loss')}/g{c.get('focal_gamma')} "
              f"weights={c.get('use_class_weights')} lambda_inv={c.get('lambda_inv')} "
              f"inv_type={c.get('inv_type')}")
        print(f"           pretrained={c.get('pretrained_init')} "
              f"protocol={c.get('protocol', 'inter_patient')} "
              f"batch={c.get('batch_size')} lr={c.get('lr')}")
        sd = r.get("sampler_diag")
        if sd:
            print(f"  sampler: {sd['zero_F_batches']} zero-F / {sd['batches']} batches, "
                  f"{sd['avg_F_per_batch']:.3f} F per batch")
        else:
            print("  sampler: NONE")
        print(f"  epochs : {r['epochs_run']}")
        print("-" * 78)
        for name, fn in METRICS:
            print(f"  {name:<16}{fn(r, ck):.4f}")
        print("-" * 78)

    print(f"  RUNNING AGGREGATE over {len(seeds)} seed(s): {seeds}")
    if seeds:
        runs = [load(args.tag, s) for s in seeds]
        print(f"  {'metric':<16}{'mean':>9}{'std':>9}   per-seed")
        for name, fn in METRICS:
            v = np.array([fn(r, ck) for r in runs])
            sd_ = v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"  {name:<16}{v.mean():>9.4f}{sd_:>9.4f}   "
                  + ", ".join(f"{x:.4f}" for x in v))
    print("=" * 78)


if __name__ == "__main__":
    main()
