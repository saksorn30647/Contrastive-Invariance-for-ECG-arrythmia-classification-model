"""Effective training-record diversity against per-class F1, under three
evaluation protocols. One figure that explains S and F together.

The x-axis is the inverse-Simpson effective record count over the MIT-BIH
TRAINING distribution of each class, 1 / sum(p_i^2), i.e. "how many patients
does this class effectively draw on", discounting records contributing a
handful of beats. The y-axis is per-class F1 for model a (baseline).

The three series are the same model measured three ways:
  * pooled 5-fold CV  - patients shared between train and val (leaky)
  * MIT-BIH inter-patient - patients disjoint, same database
  * INCART zero-shot  - patients disjoint AND database disjoint

The gap between the first and second series is the part of a class's score
that is patient memorisation rather than class learning. That gap is tiny for
N and V and enormous for S and F, and it tracks effective record count.

Run: python code/diversity_vs_f1.py        (use PYTHONIOENCODING=utf-8)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTDIR = "results/figures"
CLASSES = ["N", "S", "V", "F", "Q"]
# inverse-Simpson effective record count, MIT-BIH TRAIN side (CLAUDE.md audit)
EFF = {"N": 33.25, "S": 2.57, "V": 12.33, "F": 1.39, "Q": 3.01}
TAG = "focal_sampler"
SEEDS = list(range(42, 52))


def mean_f1(block, seeds=SEEDS, tag=TAG):
    V = []
    for s in seeds:
        m = json.load(open(f"runs/{tag}_seed{s}/metrics.json"))
        V.append(m[block]["macro_checkpoint"]["per_class_f1"])
    return np.mean(V, axis=0)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    kf = np.mean([json.load(open(
        f"runs/kfold_{TAG}_fold{f}_seed42/metrics.json"))["mitbih_val"]["macro_checkpoint"]["per_class_f1"]
        for f in range(5)], axis=0)
    mit = mean_f1("mitbih_val")
    inc = mean_f1("incart")

    series = [
        ("5-fold CV (patients SHARED)", kf, "#4878A8", "o"),
        ("MIT-BIH inter-patient (patients disjoint)", mit, "#DD8452", "s"),
        ("INCART zero-shot (patients + database disjoint)", inc, "#C44E52", "^"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.array([EFF[c] for c in CLASSES])
    for label, yv, col, mk in series:
        ax.scatter(x, yv, s=130, color=col, marker=mk, label=label,
                   zorder=3, edgecolor="white", linewidth=1.2)
        order = np.argsort(x)
        ax.plot(x[order], np.asarray(yv)[order], color=col, alpha=0.35,
                lw=1.5, zorder=2)

    for i, c in enumerate(CLASSES):
        ax.annotate(c, (x[i], kf[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontweight="bold", fontsize=12)
        # drop line showing the collapse from shared to disjoint patients
        ax.plot([x[i], x[i]], [inc[i], kf[i]], color="grey",
                lw=0.8, ls=":", zorder=1)

    # Q breaks the trend in BOTH directions and must be annotated, not hidden:
    # it survives patient-disjoint MIT-BIH evaluation at 0.76 on only 3.01
    # effective records (paced morphology is highly distinctive), and its
    # INCART value is ~0 for a LABEL-SPACE reason -- INCART contains 6 Q beats
    # in total -- not because of training diversity.
    qi = CLASSES.index("Q")
    msg_mit = ("Q survives inter-patient on 3 records:"
               + os.linesep.replace("\r", "") + "paced QRS is highly distinctive")
    msg_inc = ("Q ~ 0 on INCART = label-space void"
               + os.linesep.replace("\r", "") + "(6 Q beats exist), not a diversity effect")
    ax.annotate(msg_mit,
                (x[qi], mit[qi]), textcoords="offset points", xytext=(18, 6),
                fontsize=8.5, color="#8a5a2b",
                arrowprops=dict(arrowstyle="->", color="#8a5a2b", lw=0.9))
    ax.annotate(msg_inc,
                (x[qi], inc[qi]), textcoords="offset points", xytext=(18, 34),
                fontsize=8.5, color="#8a3a3a",
                arrowprops=dict(arrowstyle="->", color="#8a3a3a", lw=0.9))
    ax.axvline(3, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 35])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "35"])
    ax.set_xlabel("Effective TRAINING records for the class  "
                  "(inverse-Simpson, MIT-BIH train side)")
    ax.set_ylabel("per-class F1  (model a, baseline)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Patient diversity in training predicts whether a class survives\n"
                 "patient-disjoint evaluation  (Q is the documented exception)", fontsize=12.5)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "diversity_vs_f1.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{'class':<7}{'eff.rec':>9}{'5-fold':>9}{'MIT inter':>11}{'INCART':>9}"
          f"{'drop (pooled->inter)':>23}")
    for i, c in enumerate(CLASSES):
        print(f"{c:<7}{EFF[c]:>9.2f}{kf[i]:>9.4f}{mit[i]:>11.4f}{inc[i]:>9.4f}"
              f"{kf[i]-mit[i]:>22.4f}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
