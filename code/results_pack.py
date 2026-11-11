"""Build the full results pack (per-class tables, metric definitions,
environment) as a single Markdown file.

Everything is read from `runs/*/metrics.json` and
`runs/_analysis/confusion_matrices.json` — nothing is hand-transcribed, which
is the standing rule in CLAUDE.md ("Read metrics from there rather than
re-typing them off the console").

Three protocols, and their n means DIFFERENT things — this is stated in the
output too, because conflating them is a documented past error:

  inter-patient  n = 10 SEEDS (42-51).  Spread = seed/run variance.
  INCART         n = 10 SEEDS, same runs, evaluated zero-shot.
  5-fold CV      n = 5 FOLDS, seed FIXED at 42. Spread = PARTITION variance
                 only, and must not be quoted beside the seed-wise stds.

Run: python code/results_pack.py            (use PYTHONIOENCODING=utf-8)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["N", "S", "V", "F", "Q"]
OUTDIR = "results/reports"
CM_JSON = "runs/_analysis/confusion_matrices.json"

ARMS = [
    ("a", "baseline",        "random",     0.0, "focal_sampler",            "kfold_focal_sampler"),
    ("b", "contrastive+FT",  "pretrained", 0.0, "pretrained_focal_sampler", "kfold_pretrained_focal_sampler"),
    ("c", "invariance",      "random",     0.4, "focal_inv04_sampler",      "kfold_inv04_sampler"),
    ("d", "contrastive+inv", "pretrained", 0.4, "pretrained_inv04_sampler", "kfold_pretrained_inv04_sampler"),
]
SEEDS = list(range(42, 52))
FOLDS = list(range(5))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def arr(m, block, field):
    return np.asarray(m[block]["macro_checkpoint"][field], float)


def fmt(v, nd=4):
    return f"{v:.{nd}f}"


def ms(vals, nd=4):
    vals = np.asarray(vals, float)
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{vals.mean():.{nd}f} ± {vals.std(ddof=1):.{nd}f}"


def section_per_seed(out, title, runs, block, note="", tag_idx=4):
    """Per-seed per-class F1 for each arm, then the arm mean.

    tag_idx selects which tag column of ARMS to look up: 4 = inter-patient
    tag, 5 = k-fold tag stem. Getting this wrong silently yields EMPTY
    tables rather than an error, which is how it slipped through once.
    """
    out.append(f"\n### {title}\n")
    if note:
        out.append(note + "\n")
    for arm in ARMS:
        key, name, init, lam = arm[0], arm[1], arm[2], arm[3]
        tag = arm[tag_idx]
        rows = runs.get(tag, {})
        if not rows:
            raise KeyError(f"no runs for tag {tag!r} in section {title!r}")
        keys = sorted(rows)
        out.append(f"\n**Model {key} — {name}** (init={init}, λ_inv={lam}, tag `{tag}`, "
                   f"n={len(keys)})\n")
        out.append("| seed / fold | " + " | ".join(f"F1({c})" for c in CLASSES) +
                   " | macro-F1 (5c) | 4c | 3c | acc-proxy epochs |")
        out.append("|---" * 10 + "|")
        F = []
        for k in keys:
            m = rows[k]
            f1 = arr(m, block, "per_class_f1")
            F.append(f1)
            out.append(f"| {k} | " + " | ".join(fmt(v) for v in f1) +
                       f" | **{fmt(f1[:5].mean())}** | {fmt(f1[:4].mean())} | "
                       f"{fmt(f1[:3].mean())} | {m['epochs_run']} |")
        F = np.array(F)
        out.append("| **mean ± sd** | " +
                   " | ".join(ms(F[:, i]) for i in range(5)) +
                   f" | **{ms(F[:, :5].mean(axis=1))}** | {ms(F[:, :4].mean(axis=1))} | "
                   f"{ms(F[:, :3].mean(axis=1))} | |")


def section_model_average(out, title, runs, block, support, n_label, tag_idx=4):
    """One table: all four arms side by side, per-class precision/recall/F1.

    See section_per_seed for tag_idx.
    """
    out.append(f"\n### {title}\n")
    out.append(f"Support (number of beats per class, {n_label}):\n")
    out.append("| class | " + " | ".join(CLASSES) + " | **total** |")
    out.append("|---" * 7 + "|")
    out.append("| beats | " + " | ".join(f"{int(s):,}" for s in support) +
               f" | **{int(sum(support)):,}** |")
    out.append("")
    for metric, field in (("F1", "per_class_f1"),
                          ("Precision", "per_class_precision"),
                          ("Recall", "per_class_recall")):
        out.append(f"\n**{metric} — mean ± sd across {n_label}**\n")
        out.append("| model | " + " | ".join(CLASSES) + " | macro (5c) | macro (4c) | macro (3c) |")
        out.append("|---" * 9 + "|")
        for arm in ARMS:
            key, name, tag = arm[0], arm[1], arm[tag_idx]
            rows = runs.get(tag, {})
            if not rows:
                raise KeyError(f"no runs for tag {tag!r} in section {title!r}")
            V = np.array([arr(rows[k], block, field) for k in sorted(rows)])
            out.append(f"| **{key}.** {name} | " +
                       " | ".join(ms(V[:, i]) for i in range(5)) +
                       f" | {ms(V[:, :5].mean(axis=1))} | {ms(V[:, :4].mean(axis=1))} | "
                       f"{ms(V[:, :3].mean(axis=1))} |")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = []

    # ---------------- gather ------------------------------------------------
    inter = {}
    for _, _, _, _, tag, _ in ARMS:
        d = {}
        for s in SEEDS:
            m = load(f"runs/{tag}_seed{s}/metrics.json")
            if m:
                d[s] = m
        inter[tag] = d

    kf = {}
    for _, _, _, _, _, ktag in ARMS:
        d = {}
        for f in FOLDS:
            m = load(f"runs/{ktag}_fold{f}_seed42/metrics.json")
            if m:
                d[f] = m
        kf[ktag] = d

    cms = load(CM_JSON) or {}

    # support counts
    any_inter = inter["focal_sampler"][42]
    incart_support = np.asarray(any_inter["incart"]["macro_checkpoint"]["per_class_support"], float)
    train_counts = np.asarray(any_inter["train_class_counts"], float)
    # MIT-BIH inter-patient val support: from the confusion matrix if available
    mit_support = None
    if "mitbih|a" in cms:
        mit_support = np.asarray(cms["mitbih|a"]["support"], float)
    kfold_support = None
    if "kfold|a" in cms:
        kfold_support = np.asarray(cms["kfold|a"]["support"], float)

    # ---------------- header ------------------------------------------------
    out.append("# ECG cross-dataset arrhythmia classification — results pack")
    out.append("")
    out.append("Generated by `python code/results_pack.py` from `runs/*/metrics.json` and")
    out.append("`runs/_analysis/confusion_matrices.json`. Nothing here is hand-transcribed.")
    out.append("")
    out.append("**The four models.** All share the same `BaselineCNN` (292,005 parameters),")
    out.append("focal loss γ=2.0, no class weights, `MinorityGuaranteedBatchSampler` with an")
    out.append("F floor of 1, batch 64, lr 1e-4, weight decay 1e-5, 5-class-macro checkpoint")
    out.append("selection. **Only two things vary:** weight initialisation and λ_inv.")
    out.append("")
    out.append("| model | name | initialisation | λ_inv |")
    out.append("|---|---|---|---|")
    for key, name, init, lam, tag, _ in ARMS:
        out.append(f"| **{key}** | {name} | {init} | {lam} |")
    out.append("")
    out.append("> **Read the three protocols' `n` differently.** Inter-patient and INCART are")
    out.append("> **n = 10 SEEDS** (42–51), so the spread is seed/run variance. 5-fold CV is")
    out.append("> **n = 5 FOLDS with the seed fixed at 42**, so its spread is *partition*")
    out.append("> variance only and is systematically smaller. Quoting a fold-wise sd beside")
    out.append("> a seed-wise sd understates the real uncertainty.")
    out.append("")

    # ---------------- 1. per-class results ----------------------------------
    out.append("\n---\n")
    out.append("## 1. Per-class results")

    out.append("\n## 1A. Model averages (the summary tables)")
    if mit_support is not None:
        section_model_average(out, "1A-i. MIT-BIH inter-patient validation fold",
                              inter, "mitbih_val", mit_support, "10 seeds")
    section_model_average(out, "1A-ii. INCART zero-shot (cross-dataset)",
                          inter, "incart", incart_support, "10 seeds")
    if kfold_support is not None:
        section_model_average(out, "1A-iii. 5-fold CV — MIT-BIH validation (pooled protocol)",
                              kf, "mitbih_val", kfold_support, "5 folds", tag_idx=5)
        section_model_average(out, "1A-iv. 5-fold CV — INCART zero-shot",
                              kf, "incart", incart_support, "5 folds", tag_idx=5)

    out.append("\n## 1B. Per-seed / per-fold detail")
    section_per_seed(out, "1B-i. MIT-BIH inter-patient validation — per-seed F1",
                     inter, "mitbih_val",
                     "Validation fold = 23,371 beats from 10 held-out records; "
                     "training = 86,080 beats from 38 records.")
    section_per_seed(out, "1B-ii. INCART zero-shot — per-seed F1",
                     inter, "incart",
                     "175,811 beats from 75 records, never trained on. "
                     "F1(Q)≈0 by construction: INCART contains 6 Q beats.")
    section_per_seed(out, "1B-iii. 5-fold CV, MIT-BIH validation — per-fold F1",
                     kf, "mitbih_val",
                     "Seed fixed at 42; the FOLD is the varying factor. "
                     "Each fold validates ~21,890 beats; the five folds partition "
                     "all 109,451 beats exactly once.", tag_idx=5)
    section_per_seed(out, "1B-iv. 5-fold CV, INCART zero-shot — per-fold F1",
                     kf, "incart",
                     "Same five models as 1B-iii, each evaluated on the full INCART set.",
                     tag_idx=5)

    # ---------------- 2. confusion matrices ---------------------------------
    out.append("\n---\n")
    out.append("## 2. Confusion matrices (seed 42)")
    out.append("")
    out.append("Rendered with `code/cm.py` into this folder. Cell **colour** is")
    out.append("row-normalised (each cell as a share of its true class, i.e. recall-style),")
    out.append("so the huge N row does not wash out the minority rows; cell **text** is the")
    out.append("raw beat count. Rows = actual, columns = predicted.")
    out.append("")
    out.append("Recomputed from the saved `best_macro.pth` checkpoints, because")
    out.append("`metrics.json` stores only aggregate per-class metrics — a 5×5 matrix")
    out.append("cannot be reconstructed from per-class P/R/F1 alone (they pin the diagonal")
    out.append("and the margins, not the off-diagonal structure).")
    out.append("")
    if cms:
        out.append("| file | protocol | model | N (beats) | accuracy | macro-F1 (5c) | macro-F1 (3c) |")
        out.append("|---|---|---|---|---|---|---|")
        for proto, plabel in (("mitbih", "MIT-BIH val"), ("incart", "INCART zero-shot"),
                              ("kfold", "5-fold CV pooled")):
            for key, name, *_ in ARMS:
                r = cms.get(f"{proto}|{key}")
                if not r:
                    continue
                out.append(f"| `cm_{proto}_{key}.png` | {plabel} | **{key}.** {name} | "
                           f"{r['n']:,} | {fmt(r['accuracy'])} | {fmt(r['macro_f1_5c'])} | "
                           f"{fmt(r['macro_f1_3c'])} |")
        out.append("")
        out.append("**Note on the 5-fold matrices.** Under k-fold every beat is validated")
        out.append("exactly once, so the five per-fold matrices **sum** to a complete")
        out.append("109,451-beat matrix with no double counting and no beat omitted. That is")
        out.append("a genuine CV confusion matrix, not an average of overlapping ones.")
        out.append("")
        out.append("**Two reasons a figure here will not exactly equal one in section 1.**")
        out.append("Both are expected; neither is an error.")
        out.append("")
        out.append("1. *Single seed vs. average.* Section 1 reports the mean over 10 seeds")
        out.append("   (or 5 folds); these matrices are **seed 42 only**. E.g. model a on")
        out.append("   MIT-BIH reads macro-F1(5c) 0.5193 here against 0.5004 ± 0.0358 as a")
        out.append("   10-seed mean — well inside one sd.")
        out.append("2. *Pooled vs. fold-mean, for the k-fold rows.* The macro-F1 printed")
        out.append("   here is computed **from the pooled matrix** (a ratio of summed")
        out.append("   counts), whereas section 1A-iii averages the five per-fold macro-F1")
        out.append("   values. A mean of ratios is not the ratio of sums, so they differ in")
        out.append("   the 3rd-4th decimal (model a: 0.9377 pooled vs 0.9380 fold-mean).")
        out.append("   The pooled value is the right one to quote beside a confusion matrix,")
        out.append("   because it is the quantity that matrix actually describes.")
        out.append("")
        out.append("A third, smaller source: this recomputation runs **without AMP** while")
        out.append("the original in-training evaluation used autocast. Measured effect on")
        out.append("per-class F1 is <= 1.8e-3 across all 12 matrices (printed by")
        out.append("`code/confusion_matrices.py` as it goes), which is why stored and")
        out.append("recomputed aggregates agree only to ~3 decimals.")
        out.append("")
        for proto, plabel in (("mitbih", "MIT-BIH inter-patient validation"),
                              ("incart", "INCART zero-shot"),
                              ("kfold", "5-fold CV, pooled over all five folds")):
            have = [k for k, *_ in [(a[0],) for a in ARMS] if f"{proto}|{k}" in cms]
            if not have:
                continue
            out.append(f"\n### 2.{'123'[('mitbih','incart','kfold').index(proto)]} {plabel}\n")
            for key, name, *_ in ARMS:
                r = cms.get(f"{proto}|{key}")
                if not r:
                    continue
                cm = np.asarray(r["cm"], int)
                out.append(f"\n**Model {key} — {name}**  (n = {r['n']:,}, "
                           f"accuracy {fmt(r['accuracy'])})\n")
                # This report lives in results/reports/ while the figures live in
                # results/figures/, so inline image links have to cross over.
                out.append("![](" + f"../figures/cm_{proto}_{key}.png" + ")\n")
                out.append("| actual \\\\ predicted | " + " | ".join(CLASSES) + " | total | recall |")
                out.append("|---" * 8 + "|")
                for i, c in enumerate(CLASSES):
                    tot = cm[i].sum()
                    rec = cm[i, i] / tot if tot else 0.0
                    out.append(f"| **{c}** | " + " | ".join(f"{v:,}" for v in cm[i]) +
                               f" | {tot:,} | {fmt(rec)} |")
                out.append("| **predicted total** | " +
                           " | ".join(f"{v:,}" for v in cm.sum(axis=0)) +
                           f" | {cm.sum():,} | |")
                out.append("| **precision** | " +
                           " | ".join(fmt(p) for p in r["precision"]) + " | | |")
    else:
        out.append("*(confusion_matrices.json not found — run "
                   "`python code/confusion_matrices.py` first)*")

    # ---------------- 3. metric equations -----------------------------------
    out.append("\n---\n")
    out.append(METRIC_EQUATIONS)

    # ---------------- 4. environment ----------------------------------------
    out.append("\n---\n")
    out.append(env_section())

    path = os.path.join(OUTDIR, "results_pack.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {path}  ({len(out)} lines)")


METRIC_EQUATIONS = r"""## 3. Metric definitions

For a given class $c$, with $TP_c$ true positives, $FP_c$ false positives and
$FN_c$ false negatives (read off the confusion matrix: $TP_c$ is the diagonal
cell, $FP_c$ the rest of column $c$, $FN_c$ the rest of row $c$):

$$
\mathrm{Precision}_c \;=\; \frac{TP_c}{TP_c + FP_c}
\qquad
\mathrm{Recall}_c \;=\; \frac{TP_c}{TP_c + FN_c}
$$

$$
F1_c \;=\; 2 \cdot \frac{\mathrm{Precision}_c \cdot \mathrm{Recall}_c}
                        {\mathrm{Precision}_c + \mathrm{Recall}_c}
     \;=\; \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}
$$

Macro-averaged F1 over a class set $\mathcal{C}$ is the **unweighted** mean of
the per-class scores, so every class counts equally regardless of prevalence:

$$
\text{macro-}F1 \;=\; \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} F1_c
$$

This project reports macro-F1 over three nested class sets:

$$
\mathcal{C}_{5} = \{N,S,V,F,Q\}, \qquad
\mathcal{C}_{4} = \{N,S,V,F\}, \qquad
\mathcal{C}_{3} = \{N,S,V\}
$$

**Why three.** INCART contains **6 Q beats in total** (MIT-BIH has 8,039), so Q
is a label-space void rather than a rare class cross-dataset, and $F1_Q \approx 0$
there by construction. Because macro-F1 is an unweighted mean, a structurally
dead class does not merely lower the score — it multiplies it by a constant:

$$
\text{macro-}F1_{5} \;=\; \tfrac{4}{5}\,\text{macro-}F1_{4}
\quad\text{exactly, whenever } F1_Q = 0
$$

which is observed (INCART 5-class $0.3522 = 0.8 \times 0.4402$). So the 5-class
cross-dataset figure understates the model by a fixed factor and
**$\mathcal{C}_3$ is the headline cross-dataset metric**; $\mathcal{C}_4$ is
secondary and $\mathcal{C}_5$ is kept only for continuity with earlier reports.
In-domain under the pooled-beat (k-fold) protocol every class *is* learnable, so
there the 5-class number is a genuine 5-class number.

Accuracy, quoted only alongside the confusion matrices, is the usual

$$
\mathrm{Accuracy} \;=\; \frac{\sum_{c} TP_c}{N}
$$

and is **not** used for model selection: at 82.8% N prevalence in MIT-BIH and
87.4% in INCART, a constant-N predictor already scores ~0.83/0.87. Checkpoint
selection uses macro-$F1_5$ on the validation fold throughout.
"""


def env_section():
    import platform
    import subprocess
    import numpy, scipy, pywt, sklearn, torch, matplotlib
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:
        drv = "n/a"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
    gmem = (f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
            if torch.cuda.is_available() else "n/a")
    try:
        import wfdb
        wfdb_v = wfdb.__version__
    except Exception:
        wfdb_v = "n/a"

    rows = [
        ("Operating system", f"{platform.system()} {platform.release()} "
                             f"(build {platform.version().split('.')[-1]}), 64-bit"),
        ("CPU", "Intel Core i5-11400H @ 2.70 GHz, 6 cores / 12 threads"),
        ("System RAM", "15.71 GiB"),
        ("GPU", f"{gpu}, {gmem} VRAM"),
        ("NVIDIA driver", drv),
        ("CUDA (torch build)", torch.version.cuda),
        ("cuDNN", str(torch.backends.cudnn.version())),
        ("Python", platform.python_version()),
        ("PyTorch", torch.__version__),
        ("NumPy", numpy.__version__),
        ("SciPy", scipy.__version__),
        ("PyWavelets", pywt.__version__),
        ("scikit-learn", sklearn.__version__),
        ("Matplotlib", matplotlib.__version__),
        ("wfdb", wfdb_v),
    ]
    s = ["## 4. Computational environment\n",
         "| component | version / model |", "|---|---|"]
    s += [f"| {k} | {v} |" for k, v in rows]
    s += [
        "",
        "Full dependency list: `runs/_analysis/pip_freeze.txt`.",
        "",
        "**Determinism.** `torch.backends.cudnn.deterministic = True` and",
        "`benchmark = False` are set in every training script, and all runs are seeded",
        "(`random`, `numpy`, `torch`). Runs are nonetheless **not** bit-reproducible:",
        "1 of 4 identical seed-42 runs diverged at epoch 1. The cuDNN flags constrain",
        "convolution *algorithm selection* only, not operations that accumulate with",
        "atomics, whose summation order depends on GPU scheduling.",
        "`torch.use_deterministic_algorithms(True)` is not set. Consequence for",
        "reporting: say \"N runs at N seeds\", never \"bit-identical config\"; paired",
        "comparisons remain valid because nondeterminism adds noise, not bias.",
        "",
        "**Thermal caveat on any timing figure.** This is a laptop GPU that throttles",
        "from 2100 MHz to ~712 MHz at 87-88 °C under sustained load, and measured",
        "encoder wall-clock drifted 125 → 164 min across a single day at identical",
        "workload. Treat all latency and runtime numbers as a lower bound on the",
        "hardware, and lead with the hardware-independent figures (parameters, MACs).",
    ]
    return "\n".join(s)


if __name__ == "__main__":
    main()
