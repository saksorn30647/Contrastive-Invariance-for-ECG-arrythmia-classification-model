"""Render every confusion matrix in runs/_analysis/confusion_matrices.json to
PNG, using the project's existing plotter (`code/cm.py`).

Uses `plot_confusion_matrix(..., normalize="row", annotate="both")`: cell COLOUR
is row-normalised (each cell as a share of its true class, i.e. recall-style) so
the ~83%-prevalence N row does not flatten every other row to one shade, while
the cell TEXT carries the raw beat count with the row percentage underneath.
That combination is what makes a matrix readable at this level of class
imbalance — a raw-count colour scale would render S, V, F and Q as a uniform
near-white block.

Run: python code/render_cms.py      (use PYTHONIOENCODING=utf-8)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")          # no display; never block on plt.show()
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm import plot_confusion_matrix

CLASSES = ["N", "S", "V", "F", "Q"]
CM_JSON = "runs/_analysis/confusion_matrices.json"
OUTDIR = "results/figures"

PROTO_LABEL = {
    "mitbih": "MIT-BIH inter-patient val",
    "incart": "INCART zero-shot",
    "kfold": "5-fold CV (pooled)",
}
ARM_ORDER = ["a", "b", "c", "d"]


def main():
    if not os.path.exists(CM_JSON):
        raise SystemExit(f"{CM_JSON} not found - run code/confusion_matrices.py first")
    cms = json.load(open(CM_JSON))
    os.makedirs(OUTDIR, exist_ok=True)

    written = []
    for proto in ("mitbih", "incart", "kfold"):
        for key in ARM_ORDER:
            r = cms.get(f"{proto}|{key}")
            if not r:
                continue
            cm = np.asarray(r["cm"], dtype=int)
            title = (f"Model {key} — {r['model_name']} | {PROTO_LABEL[proto]}\n"
                     f"seed {r['seed']} | N = {r['n']:,} | "
                     f"macro-F1 5c {r['macro_f1_5c']:.4f} / "
                     f"3c {r['macro_f1_3c']:.4f} | acc {r['accuracy']:.4f}")
            fig, ax = plot_confusion_matrix(
                cm, CLASSES, title=title, figsize=(9, 7),
                normalize="row", annotate="both",
            )
            path = os.path.join(OUTDIR, f"cm_{proto}_{key}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append((path, r["n"], r["accuracy"]))
            print(f"  wrote {path}   N={r['n']:,}  acc={r['accuracy']:.4f}")

    print(f"\n{len(written)} figures -> {OUTDIR}/")


if __name__ == "__main__":
    main()
