"""Recompute per-beat predictions from SAVED CHECKPOINTS and build confusion
matrices for all four models under all three protocols.

metrics.json stores only aggregate per-class precision/recall/F1 — a full 5x5
confusion matrix cannot be reconstructed from those (they pin the diagonal and
the row/column sums, but not the off-diagonal structure). So this script runs
the forward pass again from `best_macro.pth`, which is also why the numbers are
traceable to a checkpoint rather than hand-copied.

THREE PROTOCOLS, and the k-fold one is not the same shape as the others:

  mitbih   MIT-BIH inter-patient VALIDATION fold (23,371 beats), seed 42.
  incart   INCART zero-shot, the full 175,811 beats, seed 42, from the SAME
           inter-patient checkpoint.
  kfold    5-fold CV, seed 42. Under this protocol every beat is validated
           EXACTLY ONCE (verified by `verify_kfold`), so the five per-fold
           confusion matrices SUM to a complete 109,451-beat matrix with no
           double-counting and no beat omitted. That is the honest CV
           confusion matrix; it is NOT an average of five overlapping ones.

NOTE on AMP: the original in-training evaluation used autocast, this does not.
CLAUDE.md 8-25-26 measured the resulting per-class F1 difference at <=6e-4,
which is why recomputed aggregates may differ from metrics.json in the 4th
decimal. Any larger discrepancy is a real problem and this script prints it.

Run: python code/confusion_matrices.py            (use PYTHONIOENCODING=utf-8)
     python code/confusion_matrices.py --protocols mitbih kfold
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.baseline2D import BaselineCNN
from utils.dataset2D import ECGDataset, LABEL_MAP
from utils.splits import inter_patient_indices, kfold_indices

CLASSES = ["N", "S", "V", "F", "Q"]
SEED = 42
N_FOLDS = 5
SPLIT_SEED = 0

# model key -> (display name, inter-patient tag, k-fold tag stem)
ARMS = [
    ("a", "baseline",        "focal_sampler",            "kfold_focal_sampler"),
    ("b", "contrastive+FT",  "pretrained_focal_sampler", "kfold_pretrained_focal_sampler"),
    ("c", "invariance",      "focal_inv04_sampler",      "kfold_inv04_sampler"),
    ("d", "contrastive+inv", "pretrained_inv04_sampler", "kfold_pretrained_inv04_sampler"),
]

OUT_JSON = "runs/_analysis/confusion_matrices.json"


def load_model(ckpt, device):
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def predict(model, dataset, device, indices=None, batch_size=256):
    """Return (y_true, y_pred) as int arrays over `indices` (or all of dataset)."""
    ds = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    preds = np.empty(len(ds), dtype=np.int64)
    trues = np.empty(len(ds), dtype=np.int64)
    i = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device, non_blocking=True))
            p = out.argmax(1).cpu().numpy()
            preds[i:i + len(p)] = p
            trues[i:i + len(p)] = y.numpy()
            i += len(p)
    assert i == len(ds), f"covered {i} of {len(ds)}"
    return trues, preds


def confusion(y_true, y_pred, n=5):
    """Rows = actual, cols = predicted (the orientation code/cm.py expects)."""
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def prf_from_cm(cm):
    """Per-class precision/recall/F1 derived FROM the confusion matrix.

    Deriving them from the matrix rather than re-running sklearn is a
    consistency check in itself: if these disagree with metrics.json by more
    than the AMP tolerance, either the matrix or the stored metric is wrong.
    """
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)   # column sums
    act_pos = cm.sum(axis=1).astype(float)    # row sums
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        rec = np.where(act_pos > 0, tp / act_pos, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, act_pos.astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocols", nargs="+",
                    default=["mitbih", "incart", "kfold"],
                    choices=["mitbih", "incart", "kfold"])
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs("runs/_analysis", exist_ok=True)

    results = {}
    if os.path.exists(OUT_JSON):
        results = json.load(open(OUT_JSON))

    # ---- shared data -------------------------------------------------------
    mit = None
    incart = None

    if "mitbih" in args.protocols or "kfold" in args.protocols:
        print("loading MIT-BIH scalograms...")
        mit = ECGDataset("data/X_cwt.npy", "data/y_cwt.npy")
        records = np.load("records.npy", allow_pickle=True)  # MIT-BIH record ids (48 records / 109,451 beats). NOT data/records.npy, which is INCART.
        records = np.array([str(r).strip() for r in records])
        assert len(records) == len(mit.y), "records/labels misaligned (bug #7)"

    # ---- 1. MIT-BIH inter-patient validation fold --------------------------
    if "mitbih" in args.protocols:
        tr_idx, va_idx = inter_patient_indices(records)
        print(f"\n=== MIT-BIH inter-patient val: {len(va_idx)} beats ===")
        for key, name, tag, _ in ARMS:
            ckpt = f"runs/{tag}_seed{SEED}/best_macro.pth"
            model = load_model(ckpt, device)
            yt, yp = predict(model, mit, device, va_idx, args.batch_size)
            cm = confusion(yt, yp)
            prec, rec, f1, sup = prf_from_cm(cm)
            results[f"mitbih|{key}"] = dict(
                model=key, model_name=name, protocol="mitbih", tag=tag, seed=SEED,
                cm=cm.tolist(), support=sup.tolist(),
                precision=prec.tolist(), recall=rec.tolist(), f1=f1.tolist(),
                macro_f1_5c=float(f1[:5].mean()), macro_f1_4c=float(f1[:4].mean()),
                macro_f1_3c=float(f1[:3].mean()), n=int(cm.sum()),
                accuracy=float(np.diag(cm).sum() / cm.sum()),
            )
            # cross-check against the stored metric
            stored = json.load(open(f"runs/{tag}_seed{SEED}/metrics.json"))
            s_f1 = np.asarray(stored["mitbih_val"]["macro_checkpoint"]["per_class_f1"], float)
            print(f"  {key} {name:<16} n={cm.sum():>6}  macroF1(5c)={f1.mean():.4f}"
                  f"  max|diff vs metrics.json|={np.abs(f1 - s_f1).max():.2e}")

    # ---- 2. INCART zero-shot ----------------------------------------------
    if "incart" in args.protocols:
        print("\nloading INCART scalograms (5.76 GB memmap)...")
        incart = ECGDataset("data/X_incart_cwt.dat", "data/y_incart_cwt.npy",
                            shape=(175811, 64, 256), memmap=True)
        print(f"=== INCART zero-shot: {len(incart)} beats ===")
        for key, name, tag, _ in ARMS:
            ckpt = f"runs/{tag}_seed{SEED}/best_macro.pth"
            model = load_model(ckpt, device)
            yt, yp = predict(model, incart, device, None, args.batch_size)
            cm = confusion(yt, yp)
            prec, rec, f1, sup = prf_from_cm(cm)
            results[f"incart|{key}"] = dict(
                model=key, model_name=name, protocol="incart", tag=tag, seed=SEED,
                cm=cm.tolist(), support=sup.tolist(),
                precision=prec.tolist(), recall=rec.tolist(), f1=f1.tolist(),
                macro_f1_5c=float(f1[:5].mean()), macro_f1_4c=float(f1[:4].mean()),
                macro_f1_3c=float(f1[:3].mean()), n=int(cm.sum()),
                accuracy=float(np.diag(cm).sum() / cm.sum()),
            )
            stored = json.load(open(f"runs/{tag}_seed{SEED}/metrics.json"))
            s_f1 = np.asarray(stored["incart"]["macro_checkpoint"]["per_class_f1"], float)
            print(f"  {key} {name:<16} n={cm.sum():>7}  macroF1(3c)={f1[:3].mean():.4f}"
                  f"  max|diff vs metrics.json|={np.abs(f1 - s_f1).max():.2e}")
            json.dump(results, open(OUT_JSON, "w"), indent=1)

    # ---- 3. 5-fold CV, pooled over folds ----------------------------------
    if "kfold" in args.protocols:
        folds = kfold_indices(mit.y, n_folds=N_FOLDS, seed=SPLIT_SEED, stratified=True)
        total_val = sum(len(v) for _, v in folds)
        print(f"\n=== 5-fold CV pooled MIT-BIH val: {total_val} beats "
              f"(= every beat exactly once) ===")
        assert total_val == len(mit.y), "folds do not cover the dataset exactly once"
        for key, name, _, ktag in ARMS:
            cm = np.zeros((5, 5), dtype=np.int64)
            per_fold = []
            for f in range(N_FOLDS):
                ckpt = f"runs/{ktag}_fold{f}_seed{SEED}/best_macro.pth"
                model = load_model(ckpt, device)
                _, va_idx = folds[f]
                yt, yp = predict(model, mit, device, va_idx, args.batch_size)
                cm_f = confusion(yt, yp)
                per_fold.append(cm_f.tolist())
                cm += cm_f
            prec, rec, f1, sup = prf_from_cm(cm)
            results[f"kfold|{key}"] = dict(
                model=key, model_name=name, protocol="kfold", tag=ktag, seed=SEED,
                cm=cm.tolist(), cm_per_fold=per_fold, support=sup.tolist(),
                precision=prec.tolist(), recall=rec.tolist(), f1=f1.tolist(),
                macro_f1_5c=float(f1[:5].mean()), macro_f1_4c=float(f1[:4].mean()),
                macro_f1_3c=float(f1[:3].mean()), n=int(cm.sum()),
                accuracy=float(np.diag(cm).sum() / cm.sum()),
            )
            print(f"  {key} {name:<16} n={cm.sum():>6}  macroF1(5c)={f1.mean():.4f}"
                  f"  acc={np.diag(cm).sum() / cm.sum():.4f}")
            json.dump(results, open(OUT_JSON, "w"), indent=1)

    json.dump(results, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_JSON}  ({len(results)} matrices)")


if __name__ == "__main__":
    main()
