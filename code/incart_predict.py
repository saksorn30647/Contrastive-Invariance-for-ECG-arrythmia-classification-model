"""Dump per-beat INCART predictions for a saved checkpoint.

Needed because metrics.json stores only aggregate per-class metrics. Bootstrap
CIs, per-record error analysis, and confusion structure all require sample-level
predictions, and re-deriving them is a ~2-4 min forward pass.

Run: python code/incart_predict.py --tag focal_inv04_sampler --seeds 42 43 44
Writes runs/_analysis/preds_{tag}_seed{seed}.npy (int8 predicted class per beat).
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.baseline2D import BaselineCNN
from utils.dataset2D import ECGDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--checkpoint", default="best_macro.pth")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ECGDataset("data/X_incart_cwt.dat", "data/y_incart_cwt.npy",
                    shape=(175811, 64, 256), memmap=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    os.makedirs("runs/_analysis", exist_ok=True)

    for seed in args.seeds:
        ckpt = os.path.join("runs", f"{args.tag}_seed{seed}", args.checkpoint)
        if not os.path.exists(ckpt):
            print(f"SKIP seed {seed}: {ckpt} not found")
            continue
        model = BaselineCNN().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        preds = np.empty(len(ds), dtype=np.int8)
        i = 0
        with torch.no_grad():
            for x, _ in loader:
                out = model(x.to(device, non_blocking=True))
                p = out.argmax(1).cpu().numpy().astype(np.int8)
                preds[i:i + len(p)] = p
                i += len(p)
        assert i == len(ds), f"covered {i} of {len(ds)}"

        out_path = f"runs/_analysis/preds_{args.tag}_seed{seed}.npy"
        np.save(out_path, preds)
        acc = float((preds == ds.y).mean())
        print(f"seed {seed}: acc {acc:.4f}  pred counts "
              f"{dict(zip('NSVFQ', np.bincount(preds, minlength=5).tolist()))}  -> {out_path}")


if __name__ == "__main__":
    main()
