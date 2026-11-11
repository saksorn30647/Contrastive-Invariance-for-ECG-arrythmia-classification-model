import argparse
import hashlib
import json
import os
import random

import matplotlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils.cwt import GPUCWT
from utils.datasetContrastivePretrain import ECGDataset, ECGTwoViewTransform, Compose, RandomGaussianNoise, RandomAmplitudeScale, RandomLinearTimeShift, RandomTimeShift, RandomBaselineWander
from models.baseline2D import BaselineCNN
from utils.dataset2D import LABEL_MAP
from utils.splits import build_split, format_leakage_report
from utils.NTXentLoss import NTXentLoss


# ── Projection head ───────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head as used in SimCLR.
    Maps encoder output (128-d) → projection space (128-d by default).
    This head is ONLY used during pretraining and discarded afterwards.
    The encoder backbone weights are what get saved and reused.
    """
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ── Encoder wrapper ───────────────────────────────────────────────────────────
class ContrastiveEncoder(nn.Module):
    """
    Wraps BaselineCNN as a feature encoder by removing its classifier layer.
    Accepts 1D raw signals, applies real-time CWT on the GPU, and routes to 2D CNN.
    """
    def __init__(self, backbone: BaselineCNN, proj_head: ProjectionHead,
                 legacy_cwt=False):
        super().__init__()
        self.backbone   = backbone
        self.proj_head  = proj_head
        
        # Real-time CWT on GPU layer initialization
        self.cwt        = GPUCWT(scales=np.arange(1, 65), wavelet='morl',
                                 legacy=legacy_cwt)

    def forward(self, x):
        # x input from batch: (B, 1, L)
        
        # 1. Compute CWT on GPU: (B, 1, L) -> (B, 64, L)
        x = self.cwt(x)
        
        # 2. Add channel dimension to create a valid 2D image matrix: (B, 1, 64, L)
        x = x.unsqueeze(1)
        
        # 3. Route to your 2D Baseline VGG Feature Blocks
        x = self.backbone.features(x)
        x = self.backbone.gap(x)
        x = x.view(x.size(0), -1)          # (B, 128)

        # Project
        z = self.proj_head(x)               # (B, out_dim)

        # L2 normalise — required for NT-Xent cosine similarity
        z = nn.functional.normalize(z, dim=1)

        return z


def parse_args():
    p = argparse.ArgumentParser(
        description="NT-Xent contrastive pretraining of the BaselineCNN encoder."
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Global seed. The original encoder "
                        "(pretrained_encoder_1D_aug.pth) was trained WITHOUT "
                        "seeding and cannot be reproduced.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Larger batch = more NT-Xent negatives.")
    # NOTE: default is deliberately NOT pretrained_encoder_1D_aug.pth. Models b
    # and d of the four-model comparison were trained from that file; silently
    # overwriting it would make six completed runs unreproducible.
    p.add_argument("--save-path", type=str, default="model_saved/pretrained_encoder_cwtfix.pth",
                   help="Where to save the encoder backbone weights.")
    p.add_argument("--legacy-cwt", action="store_true",
                   help="Use the pre-2026-08-31 buggy GPUCWT (no kernel flip, "
                        "replicate padding). ONLY for reproducing the original "
                        "encoder; it does not match data/X_cwt.npy.")
    # The encoder's pretraining set MUST be a subset of the training set of the
    # split it will later be fine-tuned under. Measured: 78.4% of random-split
    # val beats lie inside the inter-patient encoder's pretraining set (F:
    # 55.0%), versus 0.0% for inter-patient. Getting this wrong adds sample
    # leakage on top of whatever the arm is meant to measure.
    # Which database supplies the pretraining beats. 'incart' exists for the
    # reverse-direction experiment, where INCART is the SOURCE and MIT-BIH is
    # the untouched zero-shot target; an encoder that had seen MIT-BIH would
    # contaminate that target.
    p.add_argument("--dataset", choices=["mitbih", "incart"], default="mitbih",
                   help="Database to pretrain on. Must match the fine-tuning "
                        "run's TRAINING source.")
    p.add_argument("--fold", type=int, default=None,
                   help="Fold index for --protocol kfold. The encoder is "
                        "pretrained on THAT FOLD's train side only.")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of CV folds; must match the fine-tuning runs.")
    p.add_argument("--protocol", choices=["inter_patient", "random", "kfold"],
                   default="inter_patient",
                   help="Which split's TRAIN side to pretrain on. Must match the "
                        "--protocol of the fine-tuning runs that use this encoder.")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="random protocol only; must match the fine-tuning runs.")
    p.add_argument("--split-seed", type=int, default=0,
                   help="random protocol only; must match the fine-tuning runs. "
                        "Kept separate from --seed on purpose.")
    p.add_argument("--unstratified", action="store_true",
                   help="random protocol only; must match the fine-tuning runs.")
    p.add_argument("--show-plot", action="store_true")
    return p.parse_args()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # See CLAUDE.md bug #6: cuDNN autotuning is a second source of
    # nondeterminism that seeding alone does not cover. (Bug #8: these flags
    # are necessary but still not sufficient — runs remain only mostly
    # reproducible.)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = parse_args()
    set_global_seed(args.seed)
    if not args.show_plot:
        matplotlib.use("Agg")   # plt.show() blocks and would stall a sweep

    # ── Device ────────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Using device: {device}")

    # ── Dataset — 80% MIT-BIH train split with two-view transform ────────────
    print(f"Loading {args.dataset.upper()} dataset for contrastive pretraining...")

    
    contrastive_transform = Compose([
    RandomGaussianNoise(std=0.01, p=0.5),
    RandomAmplitudeScale((0.9, 1.1), p=0.5),
    RandomLinearTimeShift(max_shift=10, p=0.5),
    RandomBaselineWander(p=0.5)
])

    two_view_transform = ECGTwoViewTransform(
    transform=contrastive_transform
    )

    if args.dataset == "incart":
        # Reverse direction. data/y_incart.npy is byte-identical to
        # data/y_incart_cwt.npy (verified), so beat ordering matches the
        # scalogram array the fine-tuning run uses.
        x_path, y_path = "data/X_incart.npy", "data/y_incart.npy"
        rec_path, lab_path = "data/records.npy", "data/y_incart_cwt.npy"  # INCART record ids (75 records / 175,811 beats). NOT root records.npy, which is MIT-BIH.
    else:
        x_path, y_path = "data/X.npy", "data/y.npy"
        rec_path, lab_path = "records.npy", "data/y_cwt.npy"  # MIT-BIH record ids (48 records / 109,451 beats). NOT data/records.npy, which is INCART.

    full_dataset = ECGDataset(
        x_path=x_path,
        y_path=y_path,
        two_view_transform=two_view_transform        # returns ((v1, v2), label)
    )

    records = np.load(rec_path)

    # Labels MUST be mapped to ints before build_split. random_split_indices
    # stratifies over np.unique(y): on the raw <U1 chars that iterates
    # F,N,Q,S,V, on ints it iterates N,S,V,F,Q -> the RNG is consumed in a
    # different order and the resulting split is DIFFERENT. Using
    # dataset2D.LABEL_MAP here reproduces ECGDataset.y exactly, which is what
    # dual_protocol_training.py passes.
    y_int = np.array([LABEL_MAP[str(l).strip()] for l in np.load(lab_path)])
    assert len(records) == len(y_int) == len(full_dataset), (
        f"length mismatch: records {len(records)}, labels {len(y_int)}, "
        f"dataset {len(full_dataset)} - record/beat alignment is broken")

    if args.dataset == "incart":
        if args.protocol != "inter_patient":
            raise SystemExit("--dataset incart supports only "
                             "--protocol inter_patient (the 67/8 record split).")
        from utils.splits import incart_inter_patient_indices, split_leakage_report
        train_idx, val_idx = incart_inter_patient_indices(records)
        leakage = split_leakage_report(
            np.array([str(r).strip() for r in records]), y_int, train_idx, val_idx)
    else:
        if args.protocol == "kfold" and args.fold is None:
            raise SystemExit("--protocol kfold requires --fold")
        train_idx, val_idx, leakage = build_split(
            args.protocol,
            records,
            y_int,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            stratified=not args.unstratified,
            fold=args.fold,
            n_folds=args.n_folds,
        )
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    # The whole point of the sidecar. If this ever fails, the encoder would be
    # pretrained on beats that the fine-tuning run validates on.
    assert len(np.intersect1d(train_idx, val_idx)) == 0, \
        "train/val index overlap - encoder would see validation beats"

    train_dataset = Subset(full_dataset, train_idx.tolist())

    print(format_leakage_report(
        leakage, title=f"PRETRAINING SPLIT AUDIT ({args.protocol})"))
    print(f"Pretraining samples : {len(train_dataset)} "
          f"(protocol={args.protocol}, {len(np.unique(records[train_idx]))} "
          f"records represented)")
    print(f"HELD OUT from pretraining: {len(val_idx)} beats")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,   # larger batch = more negatives = better NT-Xent
        shuffle=True,
        num_workers=0,             # Windows safe
        pin_memory=True,
        drop_last=True             # NT-Xent requires consistent batch size
    )
    batch = next(iter(train_loader))
    (view_1, view_2), labels = batch

    print(view_1.shape)
    print(view_2.shape)
    
    # ── Model ─────────────────────────────────────────────────────────────────
    backbone   = BaselineCNN(num_classes=5)   # classifier head unused here
    proj_head  = ProjectionHead(in_dim=128, hidden_dim=128, out_dim=128)
    model      = ContrastiveEncoder(backbone, proj_head,
                                    legacy_cwt=args.legacy_cwt).to(device)

    print(f"Encoder parameters : "
          f"{sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"Proj head parameters: "
          f"{sum(p.numel() for p in model.proj_head.parameters()):,}")

    # ── Loss and optimizer ────────────────────────────────────────────────────
    criterion = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,   # matches epochs
        eta_min=1e-6
    )

    # ── Pretraining loop ──────────────────────────────────────────────────────
    epochs       = args.epochs
    loss_list    = []
    best_loss    = float('inf')
    save_path    = args.save_path

    if os.path.exists(save_path):
        raise SystemExit(
            f"REFUSING TO OVERWRITE {save_path!r}. Existing encoders are "
            f"depended on by completed runs; pass --save-path with a new name.")

    print(f"\n  seed        : {args.seed} (cudnn.deterministic=True)")
    print(f"  CWT         : {'LEGACY (buggy, pre-2026-08-31)' if args.legacy_cwt else 'FIXED - matches compute_cwt/pywt to ~1e-4'}")
    print(f"  save path   : {save_path}")

    print(f"\n── Contrastive pretraining · {epochs} epochs ────────────────────────────")

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for (view_1, view_2), _ in train_loader:
            # Labels are intentionally ignored — NT-Xent uses pairing only
            view_1 = view_1.to(device, non_blocking=True)   # (B, 1, L)
            view_2 = view_2.to(device, non_blocking=True)   # (B, 1, L)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                z1 = model(view_1)    # (B, 128) normalised
                z2 = model(view_2)    # (B, 128) normalised
    
                # Force float32 computation for NT-Xent to prevent float16 exponential overflow
                loss = criterion(z1.float(), z2.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Calculate average epoch loss and append to list so matplotlib stays happy
        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Save checkpoint whenever loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save backbone weights only — projection head is discarded after pretraining
            torch.save(
                model.backbone.state_dict(),
                save_path
            )

        print(f"Epoch [{epoch+1:02d}/{epochs}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Best: {best_loss:.4f}  "
              f"LR: {current_lr:.2e}")

    # ── Save final loss curve ─────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), loss_list, marker='o', color='steelblue', markersize=3)
    plt.title('NT-Xent Contrastive Pretraining Loss (MIT-BIH 80%)')
    plt.xlabel('Epoch')
    plt.ylabel('NT-Xent Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.splitext(save_path)[0] + "_loss.png", dpi=150)
    if args.show_plot:
        plt.show()

    # Split provenance sidecar. dual_protocol_training.py refuses to load an
    # encoder whose recorded split does not match the run's own split.
    meta = {
        "protocol": args.protocol,
        "fold": (args.fold if args.protocol == "kfold" else None),
        "n_folds": (args.n_folds if args.protocol == "kfold" else None),
        "source_dataset": ("INCART" if args.dataset == "incart" else "MIT-BIH"),
        "val_fraction": (args.val_fraction if args.protocol == "random" else None),
        # kfold uses split_seed as the FOLD-CONSTRUCTION seed (build_split ->
        # kfold_indices(seed=split_seed)), so it is load-bearing provenance:
        # a different split_seed is a different partition of the same size.
        # Omitting it here made dual_protocol_training.py's fold-aware guard
        # compare 0 against None and refuse a CORRECT encoder (2026-09-14).
        "split_seed": (args.split_seed
                       if args.protocol in ("random", "kfold") else None),
        "stratified": (not args.unstratified if args.protocol == "random" else None),
        "n_train": int(len(train_idx)),
        "n_held_out": int(len(val_idx)),
        "train_idx_sha256": hashlib.sha256(
            np.sort(train_idx).astype(np.int64).tobytes()).hexdigest(),
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "best_ntxent_loss": float(best_loss),
        "legacy_cwt": bool(args.legacy_cwt),
        "source_script": "contrastive_pretrain.py",
    }
    meta_path = os.path.splitext(save_path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Split provenance   : {meta_path}")

    print(f"\nPretraining complete.")
    print(f"Best loss          : {best_loss:.4f}")
    print(f"Encoder saved to   : {save_path}")
    print(f"Loss curve saved to: pretrain_loss_2d.png")
    print(f"\nNext step: load '{save_path}' into BaselineCNN.features "
          f"for supervised fine-tuning.")


if __name__ == '__main__':
    main()