"""Post-hoc prior / logit correction for the MIT-BIH -> INCART label shift.

Motivation (label-compatibility audit, 2026-08-11):
  * The AAMI maps are identical across both extraction scripts, INCART F uses the
    same 'F' annotation symbol, and INCART F scalograms are statistically
    indistinguishable from MIT-BIH F ones (mean 0.837 vs 0.830, std 0.728 vs
    0.743). So p(x | y) is similar across domains.
  * But p(y) is very different: F is 0.733% of MIT-BIH beats and 0.125% of
    INCART beats (5.9x rarer); Q is 7.35% vs 0.003%; S 2.54% vs 1.11%;
    V 6.61% vs 11.38%.

That is textbook **label shift**, and it predicts exactly the observed failure:
a model calibrated at the source priors over-predicts F and Q on the target by
roughly the prior ratio. Under label shift the Bayes-optimal fix needs no
retraining — adjust the logits by log(target_prior / source_prior).

Two target-prior regimes are reported, and the distinction matters:

  ORACLE     — uses INCART's TRUE class priors. This is test-set information and
               is NOT a deployable method. Reported only as an upper bound: it
               answers "is prior correction worth pursuing at all?"
  ESTIMATED  — estimates the target priors from the model's own softmax outputs
               with the Saerens-Latinne-Decaestecker EM procedure. Uses no test
               labels, so it is a legitimate claimable method.

Run from the project root (after a sweep has produced checkpoints):
    python code/prior_correction.py --tag focal_sampler
    python code/prior_correction.py --tag pretrained_focal_sampler --checkpoint macro
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.baseline2D import BaselineCNN  # noqa: E402
from utils.dataset2D import ECGDataset  # noqa: E402

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
EPS = 1e-12


def collect_logits(ckpt_path, loader, device, use_amp):
    """One INCART pass, keeping logits rather than argmax predictions."""
    model = BaselineCNN().to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    logits_all, labels_all = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(signals)
            logits_all.append(out.float().cpu())
            labels_all.append(labels)
    return torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sld_em(probs, source_prior, max_iter=1000, tol=1e-8):
    """Saerens-Latinne-Decaestecker EM estimate of the target prior.

    Iteratively re-weights the source-calibrated posteriors by the current
    target-prior estimate and re-averages. Uses no target labels.
    """
    src = np.asarray(source_prior, dtype=np.float64)
    tgt = src.copy()
    for _ in range(max_iter):
        w = tgt / np.maximum(src, EPS)
        num = probs * w
        adj = num / np.maximum(num.sum(axis=1, keepdims=True), EPS)
        new = adj.mean(axis=0)
        new = new / new.sum()
        if np.max(np.abs(new - tgt)) < tol:
            tgt = new
            break
        tgt = new
    return tgt


def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(5)), average=None, zero_division=0)
    return dict(
        per_class_f1=f,
        per_class_recall=r,
        per_class_precision=p,
        macro5=float(np.mean(f)),
        macro4=float(np.mean(f[:4])),
        macro3=float(np.mean(f[:3])),
        n_pred=np.bincount(y_pred, minlength=5),
    )


def report(name, s, y_true):
    n_true = np.bincount(y_true, minlength=5)
    print(f"\n  --- {name} ---")
    print(f"    {'cls':<5}{'F1':>9}{'recall':>9}{'prec':>9}"
          f"{'#pred':>10}{'#true':>9}{'ratio':>9}")
    for i, c in enumerate(CLASS_NAMES):
        ratio = s["n_pred"][i] / n_true[i] if n_true[i] else float("nan")
        rs = f"{ratio:8.2f}x" if ratio == ratio else "        -"
        print(f"    {c:<5}{s['per_class_f1'][i]:>9.4f}{s['per_class_recall'][i]:>9.4f}"
              f"{s['per_class_precision'][i]:>9.4f}{s['n_pred'][i]:>10,}{n_true[i]:>9,}{rs:>9}")
    print(f"    macro-F1  5-class {s['macro5']:.4f} | 4-class {s['macro4']:.4f} "
          f"| 3-class {s['macro3']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--tag", default=None, help="Only process runs with this config tag.")
    ap.add_argument("--checkpoint", choices=["macro", "balanced"], default="macro")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    incart = ECGDataset("data/X_incart_cwt.dat", "data/y_incart_cwt.npy",
                        shape=(175811, 64, 256), memmap=True)
    loader = DataLoader(incart, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    ckpt_file = "best_macro.pth" if args.checkpoint == "macro" else "best_balanced.pth"
    summary = []

    for mpath in sorted(glob.glob(os.path.join(args.runs_dir, "*", "metrics.json"))):
        m = json.load(open(mpath))
        if args.tag and m["config"].get("tag") != args.tag:
            continue
        run_dir = os.path.dirname(mpath)
        ckpt = os.path.join(run_dir, ckpt_file)
        if not os.path.exists(ckpt):
            print(f"  (skip {os.path.basename(run_dir)} — no {ckpt_file})")
            continue

        seed = m["config"]["seed"]
        print("\n" + "=" * 86)
        print(f"{os.path.basename(run_dir)}  ({args.checkpoint} checkpoint)")
        print("=" * 86)

        counts = np.asarray(m["train_class_counts"], dtype=np.float64)
        source_prior = counts / counts.sum()

        logits, y_true = collect_logits(ckpt, loader, device, use_amp)
        probs = softmax(logits)

        true_prior = np.bincount(y_true, minlength=5).astype(np.float64)
        true_prior /= true_prior.sum()
        est_prior = sld_em(probs, source_prior)

        print(f"  source prior (MIT-BIH train): "
              f"{dict(zip(CLASS_NAMES, np.round(source_prior, 5)))}")
        print(f"  target prior TRUE  (INCART) : "
              f"{dict(zip(CLASS_NAMES, np.round(true_prior, 5)))}")
        print(f"  target prior SLD-EM estimate: "
              f"{dict(zip(CLASS_NAMES, np.round(est_prior, 5)))}")

        base = scores(y_true, logits.argmax(1))
        report("UNCORRECTED (as reported today)", base, y_true)

        results = {"uncorrected": base}
        for name, tgt in (("ORACLE (true target prior — NOT deployable, upper bound)",
                           true_prior),
                          ("ESTIMATED (SLD-EM, no test labels — claimable)", est_prior)):
            shift = np.log(np.maximum(tgt, EPS)) - np.log(np.maximum(source_prior, EPS))
            s = scores(y_true, (logits + shift).argmax(1))
            report(name, s, y_true)
            results["oracle" if name.startswith("ORACLE") else "estimated"] = s

        summary.append((seed, results))

    if not summary:
        print("\nNo matching runs found.")
        return

    print("\n" + "=" * 86)
    print(f"SUMMARY — {args.tag or 'all runs'} ({args.checkpoint} ckpt), "
          f"INCART macro-F1 mean +/- std over {len(summary)} seed(s)")
    print("=" * 86)
    print(f"  {'variant':<14}{'5-class':>18}{'4-class':>18}{'3-class':>18}")
    for key, lab in (("uncorrected", "uncorrected"),
                     ("oracle", "oracle"),
                     ("estimated", "estimated")):
        cells = []
        for k in ("macro5", "macro4", "macro3"):
            v = [r[key][k] for _, r in summary]
            sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
            cells.append(f"{np.mean(v):.4f} +/- {sd:.4f}")
        print(f"  {lab:<14}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}")
    print("\n  Oracle uses INCART's true priors and is an UPPER BOUND, not a method.")
    print("  Quote the ESTIMATED row as the deployable result.")


if __name__ == "__main__":
    main()
