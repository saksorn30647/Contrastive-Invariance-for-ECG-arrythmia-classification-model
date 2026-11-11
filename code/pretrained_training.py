"""Focal / sampler / invariance training starting from the CONTRASTIVE-PRETRAINED encoder.

This is a deliberate copy of `invariance_training.py`, not an import of it.
`invariance_training.py` is the random-init control arm for the final write-up and must
stay byte-identical; this project has a documented history of refactors introducing silent
bugs, so the control script is kept unable to break.

The only behavioural difference is weight initialisation: `features` is loaded from a
contrastive-pretrained checkpoint instead of being randomly initialised. Checkpoint
selection, the loss, the sampler, and the metrics.json schema are all unchanged, so runs
from this file are directly comparable to runs from the control.

Usage (from project root):
    python code/pretrained_training.py --seed 42 --tag pretrained_focal_sampler
    python code/pretrained_training.py --seed 42 --lambda-inv 0.4 --tag pretrained_full
    python code/pretrained_training.py --seed 42 --random-init --tag sanity_control
"""

import argparse
import json
import os
import random

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.dataset2D import ECGDataset
from models.baseline2D import BaselineCNN
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from utils.invariance import InvarianceLoss, apply_cwt_perturbations
from utils.samplers import MinorityGuaranteedBatchSampler

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# Default seed. Overridable with --seed so a sweep doesn't need a file edit
# per run — run-to-run variance was large enough (see CLAUDE.md "Known
# instability") to make prior comparisons unreliable, so seeds are swept.
DEFAULT_SEED = 42


DEFAULT_PRETRAINED = "pretrained_encoder_1D_aug.pth"


def parse_args():
    p = argparse.ArgumentParser(
        description="Focal loss + sampler training from a contrastive-pretrained encoder."
    )
    p.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED,
                   help="Contrastive-pretrained checkpoint to initialise `features` from. "
                        "Only the encoder is loaded; the classifier stays random.")
    p.add_argument("--random-init", action="store_true",
                   help="Skip pretrained loading — sanity control proving the flag has an "
                        "effect. Reproduces invariance_training.py's initialisation.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Global seed. Also seeds the batch sampler.")
    p.add_argument("--epochs", type=int, default=30,
                   help="Max epochs. Lower values are useful for smoke tests.")
    p.add_argument("--tag", type=str, default="focal_sampler",
                   help="Run label; output lands in runs/{tag}_seed{seed}/.")
    p.add_argument("--show-plot", action="store_true",
                   help="Open the plot window. Off by default — plt.show() "
                        "blocks and would stall a sequential sweep.")
    p.add_argument("--lambda-inv", type=float, default=0.0,
                   help="Invariance/consistency loss weight. 0.0 skips the "
                        "perturbed forward pass entirely (genuine ablation, "
                        "not a zeroed term). 0.4 is this project's tested value.")
    p.add_argument("--no-sampler", action="store_true",
                   help="Disable MinorityGuaranteedBatchSampler and fall back to "
                        "plain shuffled sampling, restoring F's natural ~0.5%% "
                        "frequency (and its ~72.7%% zero-F batch rate).")
    return p.parse_args()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Without these, cuDNN autotunes its conv algorithms and some of the
    # choices are non-deterministic — so seeding alone does NOT pin a GPU run,
    # and a seed sweep measures nothing. benchmark=False also stops the
    # autotuner from picking different algorithms on different runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subset_macros(per_class_f1):
    """Macro-F1 over class subsets, for REPORTING only.

    Why this exists: Q is a source-domain-only class. MIT-BIH val has 1,804 Q
    beats (paced record 217) and the model genuinely learns them — observed
    F1(Q) 0.39-0.89. INCART contains **6 Q beats total**, so Q's F1 there is
    ~0 by construction and the 5-class INCART macro is mechanically depressed
    by a dead class (measured ratio: 5-class = 0.800 x 4-class across all runs).
    F is additionally a single-patient class under this split.

    Reporting N/S/V and N/S/V/F alongside the 5-class figure makes that
    arithmetic explicit instead of burying it in one number.

    NOTE: this does NOT touch checkpoint selection, which stays on 5-class
    macro-F1 over MIT-BIH val. Dropping Q from *selection* would discard a
    class the model actually learns in-domain, and dropping F was measured to
    make selection prefer epochs where F collapses to zero.
    """
    f = [float(v) for v in per_class_f1]
    return {
        "macro5_NSVFQ": float(np.mean(f)),
        "macro4_NSVF": float(np.mean(f[:4])),
        "macro3_NSV": float(np.mean(f[:3])),
    }


def print_subset_macros(label, per_class_f1, indent="  "):
    s = subset_macros(per_class_f1)
    print(f"{indent}{label}:  "
          f"5-class (N/S/V/F/Q) {s['macro5_NSVFQ']:.4f}  |  "
          f"4-class (N/S/V/F) {s['macro4_NSVF']:.4f}  |  "
          f"3-class (N/S/V) {s['macro3_NSV']:.4f}")
    return s


def forward_with_features(model, x):
    """
    Run the encoder + classifier manually so we get both logits and the
    pooled embedding back, without requiring model.forward() itself to
    change its return signature (which would break every other script that
    calls model(x) expecting a single logits tensor — the val/test loops
    below, plus your other training scripts).
    """
    feats = model.features(x)
    pooled = model.gap(feats).flatten(1)
    logits = model.classifier(pooled)
    return logits, pooled


def main():

    args = parse_args()
    SEED = args.seed

    # Pick the backend BEFORE pyplot is imported. Agg is headless, so a
    # sequential sweep never stalls on an open figure window.
    if not args.show_plot:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    set_global_seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    # One directory per run, so seeds never overwrite each other's
    # checkpoints/plots the way the previous single-run layout did.
    run_dir = os.path.join("runs", f"{args.tag}_seed{SEED}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run outputs -> {run_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── MIT-BIH: interpatient split ───────────────────────────────────────────
    print("Loading MIT-BIH CWT dataset...")
    mitbih_dataset = ECGDataset(
        "data/X_cwt.npy",
        "data/y_cwt.npy"
    )
    records = np.load("records.npy")

    train_records = [
        "100", "101", "102", "104", "105", "106", "107", "108", "109", "112",
        "113", "114", "116", "117", "118", "119", "123", "124", "200", "202",
        "203", "205", "207", "210", "213", "214", "215", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234"
    ]
    val_records = [
        "103", "111", "115", "121", "122", "201", "208", "209", "212", "217"
    ]

    train_idx = [i for i, r in enumerate(records) if r in train_records]
    val_idx = [i for i, r in enumerate(records) if r in val_records]

    # Computed from the actual split rather than hardcoded, so this stays
    # correct if train_records/val_records ever change.
    y_train = mitbih_dataset.y[train_idx]

    train_dataset = Subset(mitbih_dataset, train_idx)
    val_dataset = Subset(mitbih_dataset, val_idx)

    print(f"Train samples : {len(train_dataset)} from {len(train_records)} records")
    print(f"Val samples   : {len(val_dataset)} from {len(val_records)} records")

    # ── Minority-guaranteed sampling ──────────────────────────────────────────
    # Focal loss (round 1) reweights examples that appear in a batch but does
    # nothing about whether they appear at all — at batch=64, class F (label 3)
    # is expected to be absent from ~72.5% of batches (Poisson exposure math,
    # see CLAUDE.md). This sampler floors F's exposure to >=1 per batch without
    # touching the relative frequency of N/V/Q, unlike the WeightedRandomSampler
    # approach ruled out earlier in the project. Start with F only, floor=1 —
    # the first agreed test — before adding S or raising the floor.
    MIN_PER_CLASS = None if args.no_sampler else {3: 1}  # class index 3 = F

    # ── Loaders — num_workers=0 for Windows ───────────────────────────────────
    # batch_sampler is mutually exclusive with batch_size/shuffle in PyTorch's
    # DataLoader API, so the two branches are built separately.
    if MIN_PER_CLASS is None:
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            num_workers=0, pin_memory=True, generator=g
        )
    else:
        train_batch_sampler = MinorityGuaranteedBatchSampler(
            y_train, batch_size=64, min_per_class=MIN_PER_CLASS, seed=SEED
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler,
            num_workers=0, pin_memory=True, generator=g
        )
    val_loader_mitbih = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ── Class-balancing config ────────────────────────────────────────────────
    # ALPHA_SMOOTH controls the per-class weight exponent, as before.
    # Set USE_CLASS_WEIGHTS = False to drop them entirely — worth testing with
    # focal loss, since focal already rebalances implicitly (via the
    # (1-p_t)^gamma term) and stacking both can over-correct toward minority
    # classes, which is roughly what the alpha=0.15 runs looked like.
    ALPHA_SMOOTH = 0.5
    USE_CLASS_WEIGHTS = False   # see note above; True reproduces the CE baseline

    class_counts = np.bincount(y_train)  # N, S, V, F, Q — from the real split
    max_samples = np.max(class_counts)
    smoothed_weights = (max_samples / class_counts) ** ALPHA_SMOOTH
    if USE_CLASS_WEIGHTS:
        class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)
        print(f"Smoothed class weights (alpha={ALPHA_SMOOTH}):", smoothed_weights)
    else:
        class_weights_tensor = None
        print("Class weights DISABLED — relying on focal loss rebalancing only")

    # ── INCART: load and verify before training starts ────────────────────────
    print("\nLoading INCART CWT data...")

    dat_path = "data/X_incart_cwt.dat"
    dtype = np.float16
    h, w = 64, 256                          # frequency bins × time steps
    bytes_each = np.dtype(dtype).itemsize * h * w  # bytes per sample
    file_bytes = os.path.getsize(dat_path)
    n_samples = file_bytes // bytes_each
    print(f"Detected {n_samples} samples in {dat_path} "
          f"({file_bytes / 1e9:.2f} GB)")

    incart_dataset = ECGDataset(
        "data/X_incart_cwt.dat",
        "data/y_incart_cwt.npy",
        shape=(175811, 64, 256),
        memmap=True
    )
    incart_loader = DataLoader(
        incart_dataset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True
    )

    y_incart = np.array(incart_dataset.y)
    print(f"INCART test samples: {len(incart_dataset)}")
    print(f"Class distribution (INCART): "
          f"{dict(zip(*np.unique(y_incart, return_counts=True)))}")

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    # Ablation switch: set to 0.0 to isolate whether the invariance/
    # consistency mechanism itself is hurting minority classes, independent
    # of the augmentation (already tested and mostly ruled out) or sampling
    # sparsity (still queued). At 0.0, InvarianceLoss skips the term
    # entirely and the code below skips building the perturbed branch too
    # — so this is a genuine "plain class-weighted CE only" baseline, not
    # just a zeroed-out term with wasted compute behind it.
    LAMBDA_INV = args.lambda_inv

    # ── Classification loss: 'ce' or 'focal' ─────────────────────────────────
    # Keep LAMBDA_INV = 0.0 for the first focal runs so this is a clean
    # single-variable comparison against the existing lambda=0 CE baselines
    # (MIT-BIH macro-F1 0.5270 / 0.5578 / 0.4575). Turn invariance back on
    # only once the focal-vs-CE question is settled.
    CLS_LOSS_TYPE = 'focal'   # 'ce' reproduces the previous baseline exactly
    FOCAL_GAMMA = 2.0         # 0.0 == weighted CE; 2.0 is the standard default

    model = BaselineCNN().to(device)

    # ── Encoder initialisation ────────────────────────────────────────────────
    # The ONLY behavioural difference from the random-init control arm.
    #
    # The loader in multistep_scec_finetune.py filters keys by name+shape and
    # silently keeps whatever survives — so if the architecture ever drifts it
    # loads ZERO layers, prints a cheerful message, and trains from random init
    # with no error. That is precisely the silent-failure class already in
    # CLAUDE.md's bug list, so here a partial match is a hard failure.
    pretrained_loaded = 0
    pretrained_path = None
    if args.random_init:
        print("\nEncoder init: RANDOM (--random-init) — matches the control arm.")
    else:
        pretrained_path = args.pretrained
        print(f"\nEncoder init: loading contrastive-pretrained weights from "
              f"{pretrained_path}")
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_path} "
                f"(run from the project root, paths are root-relative)")

        pretrained_dict = torch.load(
            pretrained_path, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        feature_keys = [k for k in model_dict if k.startswith("features")]
        matched = {
            k: v for k, v in pretrained_dict.items()
            if k.startswith("features") and k in model_dict
            and v.shape == model_dict[k].shape
        }
        if len(matched) != len(feature_keys):
            missing = sorted(set(feature_keys) - set(matched))
            raise RuntimeError(
                f"Pretrained load matched {len(matched)}/{len(feature_keys)} feature "
                f"params from {pretrained_path} — refusing to train from a partially "
                f"or silently random init. First unmatched: {missing[:5]}"
            )

        model_dict.update(matched)
        model.load_state_dict(model_dict)
        pretrained_loaded = len(matched)
        print(f"  loaded {pretrained_loaded}/{len(feature_keys)} encoder params "
              f"(classifier left at random init)")

    criterion = InvarianceLoss(
        cls_weight=class_weights_tensor,
        lambda_inv=LAMBDA_INV,
        inv_type='prediction',
        cls_loss_type=CLS_LOSS_TYPE,
        focal_gamma=FOCAL_GAMMA,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Provenance banner — printed into every log so a saved run can always be
    # traced back to the exact config that produced it.
    print("\n── Run config ────────────────────────────────────────────────")
    print(f"  seed          : {SEED} (cudnn.deterministic=True, benchmark=False)")
    print(f"  run_dir       : {run_dir}")
    print(f"  cls_loss      : {CLS_LOSS_TYPE}" +
          (f" (gamma={FOCAL_GAMMA})" if CLS_LOSS_TYPE == 'focal' else ""))
    print(f"  class weights : {'alpha=' + str(ALPHA_SMOOTH) if USE_CLASS_WEIGHTS else 'DISABLED'}")
    print(f"  lambda_inv    : {LAMBDA_INV}" +
          ("  (invariance/consistency branch ACTIVE — perturbed forward pass runs)"
           if LAMBDA_INV > 0 else "  (DISABLED — perturbed pass skipped entirely)"))
    if LAMBDA_INV > 0:
        print(f"  inv_type      : prediction (KL, clean branch detached)")
        print(f"  perturbations : gain [0.8,1.2] + noise 0.05 + time-mask "
              f"(p=0.3, width=3%)")
    print(f"  batch / lr    : 64 / 1e-4")
    print(f"  weight decay  : 1e-5")
    print(f"  optimizer     : Adam | scheduler: ReduceLROnPlateau(max, 0.5, patience=4)")
    print(f"  early stop    : patience=7, min_epochs=10, max_epochs={args.epochs}")
    print(f"  sampler       : " + ("DISABLED — plain shuffled sampling "
                                   "(F back to natural ~0.50%/batch)"
                                   if MIN_PER_CLASS is None
                                   else f"MinorityGuaranteedBatchSampler {MIN_PER_CLASS}"))
    print(f"  model init    : " + ("random (BaselineCNN) — --random-init sanity control"
                                   if args.random_init else
                                   f"PRETRAINED encoder from {pretrained_path} "
                                   f"({pretrained_loaded} params)"))
    print(f"  train / val   : {len(train_dataset)} / {len(val_dataset)} samples")
    print(f"  train counts  : {dict(zip(CLASS_NAMES, class_counts.tolist()))}")
    print("──────────────────────────────────────────────────────────────")

    # LR reduction on plateau — halves LR after 4 epochs with no val-F1
    # improvement, floors at 1e-6.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs = args.epochs
    best_f1 = 0.0
    best_epoch = -1
    best_path = os.path.join(run_dir, "best_macro.pth")

    # Second, independent checkpoint tracked by the WORST-performing class's
    # F1, not the macro average. Macro-F1 doesn't penalize a class hitting
    # zero hard enough to stop that checkpoint from "winning" — e.g. a run
    # where N/Q are excellent and F is completely abandoned (F1=0) can still
    # beat a more balanced checkpoint on macro-F1 alone. This tracker keeps
    # a separate best model that's never allowed to have quietly dropped a
    # class, so you always have it as an option even when the macro-F1
    # "winner" turns out to be a class-collapse case.
    best_min_class_f1 = -1.0
    best_min_class_epoch = -1
    best_min_class_path = os.path.join(run_dir, "best_balanced.pth")
    best_min_class_preds = []
    best_min_class_labels = []
    best_min_class_per_class = None

    # Early stopping — separate from the LR scheduler above. The scheduler
    # reacts to a 4-epoch plateau by shrinking the LR; this reacts to a
    # longer, sustained plateau (7 epochs, even after LR cuts) by stopping
    # the run outright — so a run doesn't ride 20+ epochs of overfitting
    # past its real peak the way the previous one did (best epoch 9 of 30,
    # val F1 oscillating and never recovering afterward).
    early_stop_patience = 7

    # Floor on how early stopping can trigger. Without this, a noisy early
    # spike (epoch 2 in one run, epoch 3 in the last one) becomes "best,"
    # and 7 patience-epochs later the run stops having barely trained —
    # this doesn't fix noisy checkpoint selection, it just guarantees the
    # run gets a fair number of epochs before that mechanism can end it.
    min_epochs = 10
    epochs_no_improve = 0

    train_loss_list = []
    val_acc_list = []
    val_f1_list = []
    best_preds = []
    best_labels = []

    # Full per-epoch record (loss, acc, macro-F1, per-class P/R/F1). Saved to
    # metrics.json so alternative checkpoint-selection rules can be re-derived
    # POST-HOC without re-running — e.g. "best worst-class F1 among epochs >= 5",
    # which is how the balanced checkpoint's habit of landing on epoch 2 gets
    # tested. The live selection rules below are deliberately left unchanged so
    # this run stays comparable to the earlier seed-42 run.
    history = []
    sampler_diag = None

    print("\n── Phase 1: Train on MIT-BIH, validate on MIT-BIH (interpatient) ───────")

    for epoch in range(epochs):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss, running_cls, running_inv = 0.0, 0.0, 0.0

        # Epoch-1 only: confirm from the run's own log that the sampler is
        # actually live. This project has a documented history of mechanisms
        # silently doing nothing (see CLAUDE.md bug #1), and an inactive
        # sampler would look exactly like "the sampler didn't help".
        diag_epoch = (epoch == 0)
        diag_class_totals = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        diag_zero_f_batches = 0
        diag_batches = 0

        for clean_images, targets in train_loader:
            clean_images, targets = clean_images.to(device), targets.to(device)

            if diag_epoch:
                counts = np.bincount(targets.cpu().numpy(),
                                     minlength=len(CLASS_NAMES))
                diag_class_totals += counts
                diag_batches += 1
                if counts[3] == 0:
                    diag_zero_f_batches += 1

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, features = forward_with_features(model, clean_images)

                if LAMBDA_INV > 0:
                    perturbed_images = apply_cwt_perturbations(clean_images, device)
                    logits_perturbed, features_perturbed = forward_with_features(model, perturbed_images)
                else:
                    logits_perturbed, features_perturbed = None, None

                loss, loss_cls, loss_inv = criterion(
                    logits=logits,
                    logits_perturbed=logits_perturbed,
                    targets=targets,
                    features=features,
                    features_perturbed=features_perturbed,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_cls += loss_cls.item()
            running_inv += loss_inv.item()

        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches
        avg_cls = running_cls / n_batches
        avg_inv = running_inv / n_batches
        train_loss_list.append(avg_loss)

        if diag_epoch:
            sampler_diag = {
                "batches": diag_batches,
                "zero_F_batches": int(diag_zero_f_batches),
                "avg_F_per_batch": float(diag_class_totals[3] / diag_batches),
                "class_totals": diag_class_totals.tolist(),
            }
            tag_ = "sampler check" if MIN_PER_CLASS is not None else "exposure check (NO sampler)"
            print(f"  [{tag_}, epoch 1] batches={diag_batches} | "
                  f"zero-F batches={diag_zero_f_batches} "
                  f"({100 * diag_zero_f_batches / diag_batches:.1f}%) | "
                  f"avg F/batch={sampler_diag['avg_F_per_batch']:.3f} | "
                  f"class totals={dict(zip(CLASS_NAMES, diag_class_totals.tolist()))}")

        # ── Validate on MIT-BIH held-out records ───────────────────────────
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for signals, labels in val_loader_mitbih:
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(signals)  # single logits tensor — matches model.forward()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        # Precision/recall/F1 in one call — recall was previously never
        # captured per epoch, only in the final printed report, which is why
        # run-log recall figures had to be read back off the console.
        per_class_p, per_class_r, per_class_f1, per_class_sup = \
            precision_recall_fscore_support(
                all_labels, all_preds,
                labels=list(range(len(CLASS_NAMES))),
                average=None, zero_division=0,
            )
        # Checkpoint-selection metric: 5-class macro-F1 over MIT-BIH val.
        # DELIBERATELY all five classes — see subset_macros() for why the
        # 3-/4-class views are reporting-only and must not be wired in here.
        f1 = float(np.mean(per_class_f1))  # macro-F1, computed from the same per-class array
        min_class_f1 = float(np.min(per_class_f1))
        worst_class = CLASS_NAMES[int(np.argmin(per_class_f1))]

        val_acc_list.append(acc)
        val_f1_list.append(f1)

        # Per-epoch subset macros. The 4-class view (N/S/V/F) is the one to
        # watch F on: it keeps F in the average but drops Q, whose in-domain
        # F1 is large and swings independently, masking F's contribution in
        # the 5-class number. Reporting only — selection is unchanged above.
        ep_subsets = subset_macros(per_class_f1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "train_loss_cls": float(avg_cls),
            "train_loss_inv": float(avg_inv),
            "val_acc": float(acc),
            "val_macro_f1": f1,
            "val_macro_subsets": ep_subsets,
            "val_min_class_f1": min_class_f1,
            "val_worst_class": worst_class,
            "per_class_f1": [float(v) for v in per_class_f1],
            "per_class_recall": [float(v) for v in per_class_r],
            "per_class_precision": [float(v) for v in per_class_p],
            "per_class_support": [int(v) for v in per_class_sup],
        })

        # Step the LR scheduler off val F1 (before the early-stop check, so
        # a plateau always gets a chance at an LR cut first).
        scheduler.step(f1)
        current_lr = optimizer.param_groups[0]['lr']

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_epoch = epoch + 1
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
            improve_marker = "  ↑ Best F1 updated — model saved"
        else:
            epochs_no_improve += 1
            improve_marker = ""

        # Independent of the macro-F1 tracker above: save this epoch as the
        # new "balanced" checkpoint if it raises the worst class's F1,
        # regardless of what macro-F1 does. This is what catches the run-2
        # failure mode — a checkpoint with excellent N/Q and an abandoned F
        # will never win here even if it wins on macro-F1.
        if min_class_f1 > best_min_class_f1 + 1e-4:
            best_min_class_f1 = min_class_f1
            best_min_class_epoch = epoch + 1
            best_min_class_preds = all_preds.copy()
            best_min_class_labels = all_labels.copy()
            best_min_class_per_class = per_class_f1.copy()
            torch.save(model.state_dict(), best_min_class_path)
            balanced_marker = f"  ↑ Best BALANCED (worst class {worst_class}={min_class_f1:.4f}) — model saved"
        else:
            balanced_marker = ""

        # Macro-F1 shown at all three class subsets. The 5-class figure is the
        # one selection acts on (marked *); 4-class is where F's contribution
        # is legible without Q dominating; 3-class is the cross-dataset headline.
        print(f"Epoch [{epoch+1:02d}/{epochs}]  "
              f"Loss: {avg_loss:.4f} (CLS: {avg_cls:.4f} | INV: {avg_inv:.4f}) | "
              f"Val Acc: {acc*100:.2f}% | "
              f"F1 5c*: {ep_subsets['macro5_NSVFQ']:.4f} "
              f"4c: {ep_subsets['macro4_NSVF']:.4f} "
              f"3c: {ep_subsets['macro3_NSV']:.4f} | "
              f"F1(F): {per_class_f1[3]:.4f} F1(S): {per_class_f1[1]:.4f} | "
              f"LR: {current_lr:.2e}{improve_marker}{balanced_marker}")

        if epochs_no_improve >= early_stop_patience and (epoch + 1) >= min_epochs:
            print(f"\nNo improvement for {early_stop_patience} epochs "
                  f"(best F1={best_f1:.4f}) — stopping early at epoch {epoch+1}.")
            break

    # ── MIT-BIH val report from BEST epoch (macro-F1 checkpoint) ─────────────
    print(f"\n── MIT-BIH Validation Report "
          f"(best epoch, F1={best_f1:.4f}) ─────────────────────")
    print(classification_report(
        best_labels, best_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    ))
    print("Confusion Matrix (MIT-BIH val, best epoch):")
    print(confusion_matrix(best_labels, best_preds))
    val_macro_subsets = None
    if best_epoch > 0:
        print()
        val_macro_subsets = print_subset_macros(
            "MIT-BIH val, macro-F1 ckpt", history[best_epoch - 1]["per_class_f1"])

    # ── MIT-BIH val report from the BALANCED checkpoint (min-class-F1) ───────
    # This is the model to look at if the macro-F1 "winner" above turned out
    # to have abandoned a class (check its per-class breakdown for a 0.00).
    print(f"\n── MIT-BIH Validation Report — BALANCED checkpoint "
          f"(worst-class F1={best_min_class_f1:.4f}) ─────────────")
    val_balanced_macro_subsets = None
    if best_min_class_preds:
        print(classification_report(
            best_min_class_labels, best_min_class_preds,
            target_names=CLASS_NAMES,
            zero_division=0
        ))
        print("Confusion Matrix (MIT-BIH val, balanced checkpoint):")
        print(confusion_matrix(best_min_class_labels, best_min_class_preds))
        print()
        val_balanced_macro_subsets = print_subset_macros(
            "MIT-BIH val, balanced ckpt",
            history[best_min_class_epoch - 1]["per_class_f1"])
    else:
        print("(No balanced checkpoint was saved — this shouldn't happen "
              "unless the run stopped after 0 epochs.)")

    # ── Phase 2: Load checkpoints → test on INCART ───────────────────────────
    def eval_incart(ckpt_path, label):
        """Zero-shot INCART pass for one checkpoint. Returns a metrics dict."""
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        preds, labels_all = [], []
        with torch.no_grad():
            for signals, labels in incart_loader:
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(signals)

                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        acc_ = accuracy_score(labels_all, preds)
        f1_ = f1_score(labels_all, preds, average='macro', zero_division=0)
        p_, r_, f_, s_ = precision_recall_fscore_support(
            labels_all, preds, labels=list(range(len(CLASS_NAMES))),
            average=None, zero_division=0,
        )

        print(f"\n── INCART Test Report — {label} checkpoint ─────────────────────")
        print(f"INCART Test Acc: {acc_*100:.2f}% | Macro-F1: {f1_:.4f}")
        print(classification_report(
            labels_all, preds, target_names=CLASS_NAMES, zero_division=0))
        print(f"Confusion Matrix (INCART, {label} checkpoint):")
        print(confusion_matrix(labels_all, preds))
        print()
        subsets = print_subset_macros(f"INCART, {label} ckpt", f_)
        print(f"    (INCART Q support = {int(s_[4])} beats — Q is source-domain-only, "
              f"so the 5-class figure carries a structurally dead class.)")

        return {
            "acc": float(acc_),
            "macro_f1": float(f1_),
            "macro_subsets": subsets,
            "per_class_f1": [float(v) for v in f_],
            "per_class_recall": [float(v) for v in r_],
            "per_class_precision": [float(v) for v in p_],
            "per_class_support": [int(v) for v in s_],
        }

    print(f"\n── Phase 2: Load checkpoints → test on INCART "
          f"(macro ckpt Val F1={best_f1:.4f}) ──────────────")

    incart_macro = eval_incart(best_path, "macro-F1")

    # The balanced checkpoint's cross-dataset behaviour was previously never
    # measured, even though it's the only checkpoint where F isn't dead. This
    # is added measurement only — it changes nothing about training.
    incart_balanced = None
    if best_min_class_preds:
        incart_balanced = eval_incart(best_min_class_path, "balanced")

    incart_f1 = incart_macro["macro_f1"]

    # ── Domain shift summary ──────────────────────────────────────────────────
    print("\n── Domain Shift Summary ─────────────────────────────────────────────────")
    print(f"  MIT-BIH Val  Macro-F1 (macro ckpt)   : {best_f1:.4f}")
    print(f"  INCART Test  Macro-F1 (macro ckpt)   : {incart_f1:.4f}")
    print(f"  Drop         (delta)                 : {best_f1 - incart_f1:.4f}")
    if incart_balanced is not None:
        print(f"  MIT-BIH Val  Macro-F1 (balanced ckpt) : "
              f"{float(np.mean(best_min_class_per_class)):.4f}")
        print(f"  INCART Test  Macro-F1 (balanced ckpt) : "
              f"{incart_balanced['macro_f1']:.4f}")

    # Same comparison restricted to the classes that are actually learnable
    # cross-dataset. The 5-class drop mixes in Q (6 INCART beats) and F
    # (single-patient), so it overstates the domain gap.
    if val_macro_subsets is not None:
        im = incart_macro["macro_subsets"]
        print("\n  By class subset (macro ckpt) — MIT-BIH val → INCART, drop:")
        for key, name in (("macro5_NSVFQ", "5-class N/S/V/F/Q"),
                          ("macro4_NSVF", "4-class N/S/V/F  "),
                          ("macro3_NSV", "3-class N/S/V    ")):
            v, i = val_macro_subsets[key], im[key]
            print(f"    {name} : {v:.4f} → {i:.4f}   drop {v - i:+.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    actual_epochs = len(train_loss_list)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(range(1, actual_epochs + 1), train_loss_list,
                 marker='o', color='steelblue')
    axes[0].set_title('Training Loss (CLS + Invariance) over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(range(1, actual_epochs + 1), val_acc_list,
                 marker='o', color='green')
    axes[1].set_title('MIT-BIH Val Accuracy over Epochs (interpatient)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)

    axes[2].plot(range(1, actual_epochs + 1), val_f1_list,
                 marker='o', color='orange')
    axes[2].axhline(
        y=incart_f1, color='red', linestyle='--',
        label=f'INCART Test F1 = {incart_f1:.4f}'
    )
    axes[2].set_title('MIT-BIH Val Macro-F1 vs INCART Test F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "curves.png")
    plt.savefig(plot_path, dpi=150)
    if args.show_plot:
        plt.show()
    plt.close(fig)

    # ── metrics.json — the machine-readable record of this run ───────────────
    # Everything the run log needs, so results stop being hand-transcribed off
    # the console. `history` also lets checkpoint-selection rules be re-derived
    # after the fact without re-running.
    val_best = {
        "epoch": best_epoch,
        "macro_f1": float(best_f1),
    }
    if best_epoch > 0:
        h = history[best_epoch - 1]
        val_best.update({
            "macro_subsets": val_macro_subsets,
            "per_class_f1": h["per_class_f1"],
            "per_class_recall": h["per_class_recall"],
            "per_class_precision": h["per_class_precision"],
        })

    val_balanced = None
    if best_min_class_epoch > 0:
        h = history[best_min_class_epoch - 1]
        val_balanced = {
            "epoch": best_min_class_epoch,
            "macro_f1": h["val_macro_f1"],
            "macro_subsets": val_balanced_macro_subsets,
            "min_class_f1": float(best_min_class_f1),
            "worst_class": h["val_worst_class"],
            "per_class_f1": h["per_class_f1"],
            "per_class_recall": h["per_class_recall"],
            "per_class_precision": h["per_class_precision"],
        }

    metrics = {
        "config": {
            "seed": SEED,
            "tag": args.tag,
            "cls_loss": CLS_LOSS_TYPE,
            "focal_gamma": FOCAL_GAMMA,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "alpha_smooth": ALPHA_SMOOTH,
            "lambda_inv": LAMBDA_INV,
            "batch_size": 64,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "max_epochs": epochs,
            "min_per_class": (None if MIN_PER_CLASS is None
                              else {str(k): v for k, v in MIN_PER_CLASS.items()}),
            "sampler": "none" if MIN_PER_CLASS is None else "MinorityGuaranteedBatchSampler",
            "early_stop_patience": early_stop_patience,
            "min_epochs": min_epochs,
            "cudnn_deterministic": True,
            "pretrained_init": not args.random_init,
            "pretrained_path": pretrained_path,
            "pretrained_layers_loaded": pretrained_loaded,
            "source_script": "pretrained_training.py",
            # Recorded explicitly so a future reader never has to infer it:
            # checkpoints are chosen on 5-class macro-F1 / 5-class worst-class
            # F1 over MIT-BIH val. The 3-/4-class figures are reporting only.
            "selection_metric_macro": "5class_NSVFQ_mitbih_val",
            "selection_metric_balanced": "5class_min_class_f1_mitbih_val",
        },
        "class_names": CLASS_NAMES,
        "train_class_counts": class_counts.tolist(),
        "sampler_diag": sampler_diag,
        "epochs_run": len(history),
        "history": history,
        "mitbih_val": {
            "macro_checkpoint": val_best,
            "balanced_checkpoint": val_balanced,
        },
        "incart": {
            "macro_checkpoint": incart_macro,
            "balanced_checkpoint": incart_balanced,
        },
        "domain_shift_drop": float(best_f1 - incart_f1),
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nBest MIT-BIH Val Macro-F1 : {best_f1:.4f} (epoch {best_epoch})")
    print(f"INCART Test Macro-F1      : {incart_f1:.4f}")
    print(f"Domain shift drop         : {best_f1 - incart_f1:.4f}")
    print(f"Plot saved to    {plot_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()