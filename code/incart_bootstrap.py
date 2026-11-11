"""Bootstrap confidence intervals for INCART metrics, beat-level and record-level.

Why two flavours. The naive bootstrap resamples individual beats as if they were
independent. They are not: beats from one patient share morphology, so a
patient contributes far less independent information than its beat count
suggests. This project's entire thesis is that patient structure dominates, so
the honest interval is a CLUSTER bootstrap that resamples whole records. The
beat-level interval is reported alongside only to show how much it understates
uncertainty.

Implementation note: both bootstraps are computed from per-record 5x5 confusion
matrices rather than by resampling 175,811 rows. Summing sampled records'
matrices is exactly equivalent to resampling their beats, and runs in
milliseconds instead of minutes.

Usage:
  python code/incart_bootstrap.py --tags focal_inv04_sampler --seeds 42 43 44
  python code/incart_bootstrap.py --tags focal_sampler focal_inv04_sampler \
      --seeds 42 43 44 --paired      # paired CI on the difference between arms
"""

import argparse
import os

import numpy as np

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
LABEL_MAP = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    p.add_argument("--paired", action="store_true",
                   help="Report a paired CI on the difference between the first "
                        "two tags, resampling the same records for both arms.")
    return p.parse_args()


def per_record_cms(y, preds, rec_ids, uniq):
    """Stack of (n_records, 5, 5) confusion matrices, rows=true, cols=pred."""
    cms = np.zeros((len(uniq), 5, 5), dtype=np.int64)
    index = {r: i for i, r in enumerate(uniq)}
    ridx = np.array([index[r] for r in rec_ids])
    np.add.at(cms, (ridx, y, preds), 1)
    return cms


def metrics_from_cm(cm):
    """per-class F1 plus 5/4/3-class macros from a 5x5 confusion matrix."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(5), where=denom > 0)
    return f1, f1.mean(), f1[:4].mean(), f1[:3].mean()


def ci(vals, lo=2.5, hi=97.5):
    return np.percentile(vals, lo), np.percentile(vals, hi)


def boot_indices(rng, n_rec, n_boot):
    return rng.integers(0, n_rec, size=(n_boot, n_rec))


def main():
    args = parse_args()
    y = np.array([LABEL_MAP[v] for v in np.load("data/y_incart_cwt.npy")])
    rec_ids = np.array([str(r).strip() for r in np.load("data/records.npy")])  # INCART record ids (75 records / 175,811 beats). NOT root records.npy, which is MIT-BIH.
    assert len(y) == len(rec_ids)
    uniq = sorted(set(rec_ids.tolist()))
    n_rec = len(uniq)
    rng = np.random.default_rng(args.seed)

    print(f"INCART: {len(y)} beats, {n_rec} records, {args.n_boot} bootstrap draws\n")

    store = {}
    for tag in args.tags:
        for seed in args.seeds:
            path = f"runs/_analysis/preds_{tag}_seed{seed}.npy"
            if not os.path.exists(path):
                print(f"SKIP {tag} seed {seed}: {path} missing")
                continue
            preds = np.load(path).astype(np.int64)
            store[(tag, seed)] = per_record_cms(y, preds, rec_ids, uniq)

    # Shared resampling indices, so arms are compared on identical record draws.
    idx_rec = boot_indices(rng, n_rec, args.n_boot)

    for (tag, seed), cms in store.items():
        total = cms.sum(0)
        f1, m5, m4, m3 = metrics_from_cm(total)

        # record-level (cluster) bootstrap
        boots = np.array([metrics_from_cm(cms[i].sum(0))[0] for i in idx_rec])
        macro3_b = boots[:, :3].mean(1)
        macro4_b = boots[:, :4].mean(1)

        # beat-level bootstrap: multinomial over the 25 cells of the pooled CM
        flat = total.ravel().astype(float)
        beat_draws = rng.multinomial(int(total.sum()), flat / flat.sum(),
                                     size=min(args.n_boot, 2000))
        beat_f1 = np.array([metrics_from_cm(d.reshape(5, 5))[0] for d in beat_draws])

        print(f"-- {tag} seed {seed} " + "-" * (60 - len(tag)))
        print(f"{'metric':<14}{'point':>9}{'record-level 95% CI':>26}{'width':>9}"
              f"{'beat-level 95% CI':>24}{'width':>9}")
        rows = [("3-class", m3, macro3_b, beat_f1[:, :3].mean(1)),
                ("4-class", m4, macro4_b, beat_f1[:, :4].mean(1))]
        for i, n in enumerate(CLASS_NAMES):
            rows.append((f"F1({n})", f1[i], boots[:, i], beat_f1[:, i]))
        for name, pt, rb, bb in rows:
            rlo, rhi = ci(rb)
            blo, bhi = ci(bb)
            print(f"{name:<14}{pt:>9.4f}   [{rlo:.4f}, {rhi:.4f}]{rhi-rlo:>9.4f}"
                  f"   [{blo:.4f}, {bhi:.4f}]{bhi-blo:>9.4f}")
        print()

    if args.paired and len(args.tags) >= 2:
        a_tag, b_tag = args.tags[0], args.tags[1]
        print("=" * 96)
        print(f"PAIRED record-level bootstrap on the DIFFERENCE: {b_tag} - {a_tag}")
        print("Same resampled records for both arms, so this is the CI on the")
        print("difference, not the (much wider) overlap of two separate CIs.")
        print("=" * 96)
        print(f"{'seed':<7}{'metric':<11}{'diff':>9}{'95% CI':>24}{'P(diff>0)':>12}")
        for seed in args.seeds:
            ka, kb = (a_tag, seed), (b_tag, seed)
            if ka not in store or kb not in store:
                continue
            ca, cb = store[ka], store[kb]
            for label, sl in (("3-class", slice(0, 3)), ("4-class", slice(0, 4)),
                              ("F1(F)", None)):
                da = []
                for i in idx_rec:
                    fa = metrics_from_cm(ca[i].sum(0))[0]
                    fb = metrics_from_cm(cb[i].sum(0))[0]
                    da.append((fb[3] - fa[3]) if sl is None
                              else (fb[sl].mean() - fa[sl].mean()))
                da = np.array(da)
                lo, hi = ci(da)
                print(f"{seed:<7}{label:<11}{da.mean():>+9.4f}   "
                      f"[{lo:+.4f}, {hi:+.4f}]{100*(da>0).mean():>11.1f}%")
        print()


if __name__ == "__main__":
    main()
