"""Computational resource analysis of the four compared models.

THE HEADLINE RESULT IS STRUCTURAL, NOT MEASURED: models a, b, c and d are all
the SAME `BaselineCNN`. They differ only in how the weights were obtained
(random vs contrastive-pretrained init) and in the training objective (lambda_inv
0 vs 0.4). **Inference cost is therefore identical across all four** — same
parameters, same MACs, same latency, same memory. What differs is TRAINING cost.

That is the deployment-relevant framing even though Jetson deployment is out of
scope (CLAUDE.md, 2026-08-11): the accuracy gained from invariance training is
free at inference time and costs ~1.6x at training time.

Reported:
  1. parameter count, per-module
  2. MACs / FLOPs by forward hook (not hand-arithmetic)
  3. model size on disk and in memory, at fp32 / fp16 / int8
  4. CWT preprocessing cost - part of the real inference pipeline, usually
     omitted from papers, and non-trivial here
  5. latency + throughput, GPU and CPU, batch 1 / 64 / 256
  6. peak GPU memory
  7. measured TRAINING cost per arm, from the run logs
  8. a hard check that all four checkpoints really do share one architecture

Run: python code/compute_analysis.py   (use PYTHONIOENCODING=utf-8)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.baseline2D import BaselineCNN  # noqa: E402

IN_SHAPE = (1, 64, 256)          # (C, freq bins, time steps) - one CWT scalogram
CLASS_TAGS = {
    "a. baseline": "focal_sampler",
    "b. contrastive+FT": "pretrained_focal_sampler",
    "c. invariance": "focal_inv04_sampler",
    "d. contrastive+inv": "pretrained_inv04_sampler",
}


# ── 1-2. params and MACs ────────────────────────────────────────────────────
def count_macs(model, in_shape):
    """MACs via forward hooks. One MAC = one multiply-accumulate = 2 FLOPs."""
    macs = {}

    def conv_hook(name):
        def hook(mod, inp, out):
            # out: (B, C_out, H, W); each output element costs C_in/groups * k_h * k_w MACs
            k = mod.kernel_size[0] * mod.kernel_size[1]
            cin = mod.in_channels // mod.groups
            macs[name] = int(out.numel() / out.shape[0] * cin * k)
        return hook

    def lin_hook(name):
        def hook(mod, inp, out):
            macs[name] = int(mod.in_features * mod.out_features)
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook(name)))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook(name)))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *in_shape))
    for h in handles:
        h.remove()
    return macs


def cwt_macs(n_scales=64, filt_len=1025, sig_len=256):
    """GPUCWT is 64 conv1d filters of length 1025 over a 256-sample beat.

    This runs BEFORE the CNN on every beat in a real deployment, so leaving it
    out understates end-to-end inference cost. Most papers omit it.
    """
    return n_scales * filt_len * sig_len


# ── 5. latency ──────────────────────────────────────────────────────────────
def sm_clock():
    """Current SM clock in MHz, or None. Used to prove the GPU was not idle."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def bench(model, device, batch, warmup_s=4.0, n_iter=50):
    """Latency benchmark with TIME-BASED warmup.

    A laptop GPU idles in P8 at ~210 MHz of a 2100 MHz max. A fixed count of
    warmup iterations on a 1 MB model finishes before the clocks ramp, so the
    first measured numbers are ~10x too slow and scale nonsensically with batch
    size (measured 2026-09-11: batch 1 -> 1.5 ms but batch 64 -> 189 ms, which
    is not physical). Warm up by WALL CLOCK until the clocks boost, and record
    the achieved clock alongside the result so the figure is auditable.
    """
    model = model.to(device).eval()
    x = torch.randn(batch, *IN_SHAPE, device=device)
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        t_end = time.perf_counter() + warmup_s
        while time.perf_counter() < t_end:
            model(x)
        sync()
        clk = sm_clock() if device.type == "cuda" else None
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(x)
            sync()
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000.0                      # ms
    return dict(median_ms=float(np.median(ts)),
                p95_ms=float(np.percentile(ts, 95)),
                per_beat_ms=float(np.median(ts) / batch),
                beats_per_s=float(batch / (np.median(ts) / 1000.0)),
                sm_mhz=clk)


def main():
    dev_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN()

    print("=" * 90)
    print("COMPUTATIONAL RESOURCE ANALYSIS — BaselineCNN")
    print("=" * 90)
    print(f"input per beat : {IN_SHAPE}  (CWT scalogram, 64 scales x 256 samples)")
    print(f"GPU            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"torch          : {torch.__version__}\n")

    # ── 8. architecture identity check ──────────────────────────────────────
    print("-" * 90)
    print("ARCHITECTURE IDENTITY CHECK — are a/b/c/d really the same network?")
    print("-" * 90)
    ref = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    all_same = True
    for label, tag in CLASS_TAGS.items():
        paths = sorted(glob.glob(f"runs/{tag}_seed*/best_macro.pth"))
        if not paths:
            print(f"  {label:22s} no checkpoint found"); all_same = False; continue
        sd = torch.load(paths[0], map_location="cpu", weights_only=True)
        shapes = {k: tuple(v.shape) for k, v in sd.items()}
        same = shapes == ref
        n = sum(v.numel() for v in sd.values())
        print(f"  {label:22s} {len(paths):2d} ckpts  params={n:,}  "
              f"architecture {'IDENTICAL' if same else 'DIFFERS'}")
        all_same &= same
    print(f"\n  => inference cost is {'IDENTICAL for all four models' if all_same else 'NOT identical'}.")
    print("     Only TRAINING cost differs.\n")

    # ── 1. parameters ───────────────────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    feat = sum(p.numel() for p in model.features.parameters())
    clf = sum(p.numel() for p in model.classifier.parameters())
    print("-" * 90)
    print("PARAMETERS")
    print("-" * 90)
    print(f"  total               {total:>12,}")
    print(f"  trainable           {trainable:>12,}")
    print(f"  features (encoder)  {feat:>12,}   ({100*feat/total:.1f}%)  <- what contrastive pretraining transfers")
    print(f"  classifier head     {clf:>12,}   ({100*clf/total:.1f}%)  <- always random init")
    print()
    print(f"  {'module':<28}{'params':>12}")
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            n = sum(p.numel() for p in mod.parameters())
            if n:
                print(f"  {name:<28}{n:>12,}")
    print()

    # ── 2-3. MACs and size ──────────────────────────────────────────────────
    macs = count_macs(model, IN_SHAPE)
    cnn_macs = sum(macs.values())
    pre_macs = cwt_macs()
    print("-" * 90)
    print("COMPUTE per beat (1 MAC = 2 FLOPs)")
    print("-" * 90)
    print(f"  {'layer':<28}{'MACs':>16}{'% of CNN':>10}")
    for k, v in macs.items():
        print(f"  {k:<28}{v:>16,}{100*v/cnn_macs:>9.1f}%")
    print(f"  {'CNN TOTAL':<28}{cnn_macs:>16,}{100.0:>9.1f}%")
    print(f"\n  CNN            {cnn_macs/1e6:8.2f} MMACs  =  {2*cnn_macs/1e6:8.2f} MFLOPs")
    print(f"  CWT preprocess {pre_macs/1e6:8.2f} MMACs  =  {2*pre_macs/1e6:8.2f} MFLOPs"
          f"   ({100*pre_macs/(pre_macs+cnn_macs):.1f}% of end-to-end)")
    print(f"  END-TO-END     {(cnn_macs+pre_macs)/1e6:8.2f} MMACs  = "
          f"{2*(cnn_macs+pre_macs)/1e6:8.2f} MFLOPs")
    print("\n  NOTE: the CWT is part of the real inference pipeline and is usually")
    print("        omitted from published FLOP counts. Reporting it is more honest.")
    print()
    print("-" * 90)
    print("MODEL SIZE")
    print("-" * 90)
    for name, bytes_per in (("fp32", 4), ("fp16", 2), ("int8 (hypothetical)", 1)):
        print(f"  {name:<22}{total*bytes_per/1024:>10.1f} KiB   "
              f"{total*bytes_per/1024/1024:>7.3f} MiB")
    ck = sorted(glob.glob("runs/*/best_macro.pth"))
    if ck:
        print(f"  checkpoint on disk    {os.path.getsize(ck[0])/1024:>10.1f} KiB   "
              f"(includes BN running stats)")
    print()

    # ── 5-6. latency ────────────────────────────────────────────────────────
    print("-" * 90)
    print("INFERENCE LATENCY / THROUGHPUT   (identical for a/b/c/d)")
    print("-" * 90)
    print("  (time-based warmup; SM clock shown to prove the GPU was not idling)")
    print(f"  {'device':<8}{'batch':>7}{'median ms':>12}{'p95 ms':>10}"
          f"{'ms/beat':>10}{'beats/s':>12}{'SM MHz':>9}")
    for dev, batches in ((dev_gpu, (1, 64, 256)), (torch.device("cpu"), (1, 64))):
        for b in batches:
            try:
                r = bench(model, dev, b)
                clk = "-" if r["sm_mhz"] is None else f"{r['sm_mhz']}"
                print(f"  {dev.type:<8}{b:>7}{r['median_ms']:>12.3f}{r['p95_ms']:>10.3f}"
                      f"{r['per_beat_ms']:>10.4f}{r['beats_per_s']:>12,.0f}{clk:>9}")
            except RuntimeError as e:
                print(f"  {dev.type:<8}{b:>7}  failed: {e}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        m = model.to(dev_gpu).eval()
        with torch.no_grad():
            m(torch.randn(64, *IN_SHAPE, device=dev_gpu))
        print(f"\n  peak GPU memory, batch 64: "
              f"{torch.cuda.max_memory_allocated()/1024/1024:.1f} MiB")
    print()

    # ── 7. training cost ────────────────────────────────────────────────────
    print("-" * 90)
    print("TRAINING COST — the only axis on which the four models differ")
    print("-" * 90)
    print(f"  {'model':<22}{'seeds':>7}{'epochs (mean)':>15}{'lambda_inv':>12}"
          f"{'fwd/bwd passes':>16}")
    for label, tag in CLASS_TAGS.items():
        ms = [json.load(open(p)) for p in sorted(glob.glob(f"runs/{tag}_seed*/metrics.json"))]
        if not ms:
            continue
        ep = np.mean([m["epochs_run"] for m in ms])
        lam = ms[0]["config"]["lambda_inv"]
        passes = "2x (clean+perturbed)" if lam > 0 else "1x"
        print(f"  {label:<22}{len(ms):>7}{ep:>15.1f}{lam:>12}{passes:>16}")
    print()
    print("  One-time contrastive pretraining (models b, d): 50 epochs.")
    print("    MIT-BIH  86,080 beats -> 111-117 min   (pretrained_encoder_*.pth)")
    print("    INCART  156,887 beats -> 198 min       (pretrained_encoder_incart.pth)")
    print("  Amortised across every downstream run that loads that encoder.")
    print()
    print("  THE TRADE, stated for the writeup:")
    print("    invariance training (c) costs ~1.6x TRAINING time and 0% INFERENCE")
    print("    cost, for +0.0499 INCART 3-class macro-F1 (p=0.035, max-T corrected).")
    print("    contrastive pretraining (b, d) adds a one-time ~2 h pretraining stage,")
    print("    also 0% inference cost.")


if __name__ == "__main__":
    main()
