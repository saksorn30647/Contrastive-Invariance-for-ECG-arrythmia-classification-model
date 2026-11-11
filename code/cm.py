import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_confusion_matrix(cm, labels, title="Confusion Matrix", figsize=(8, 6), cmap="Blues",
                           normalize="row", annotate="count"):
    """
    Plot a confusion matrix as a heatmap with value annotations.

    Cell COLOR is driven by row-normalized percentage (each cell's share of
    its true/actual class), so imbalanced classes (e.g. a huge "N" row)
    don't wash out every other row with a flat, near-uniform color.
    Cell TEXT still shows the raw count (or percentage, if you prefer).

    Parameters
    ----------
    cm        : 2D array-like  – confusion matrix (rows = actual, cols = predicted)
    labels    : list of str    – class labels in the same order as cm rows/columns
    title     : str            – plot title (default "Confusion Matrix")
    figsize   : tuple          – figure size in inches (default (8, 6))
    cmap      : str            – matplotlib colormap name (default "Blues")
    normalize : "row" | "all" | None
                "row" (default) – color = cm[i,j] / row_sum[i]   (i.e. recall-style %)
                "all"           – color = cm[i,j] / cm.sum()      (overall %)
                None            – color = raw cm[i,j] (old behavior)
    annotate  : "count" | "percent" | "both"
                what to print inside each cell (default "count")
    """
    cm = np.array(cm, dtype=float)
    n = len(labels)

    # ── Compute the matrix used for COLOR ────────────────────────────────────
    if normalize == "row":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        color_mat = cm / row_sums
        vmax = 1.0
    elif normalize == "all":
        color_mat = cm / cm.sum()
        vmax = color_mat.max()
    else:
        color_mat = cm
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(color_mat, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    if normalize in ("row", "all"):
        cbar.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

    # Axis labels & ticks
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Prediction",
        title=title,
    )

    # Move x-axis label + ticks to top (like the reference)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # ── Annotate each cell ────────────────────────────────────────────────────
    # Text color (white/black) is based on the COLOR matrix's brightness,
    # not the raw count, so it stays readable regardless of normalization.
    thresh = vmax / 2.0
    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            pct = color_mat[i, j] * 100 if normalize in ("row", "all") else None

            if annotate == "percent" and pct is not None:
                text = f"{pct:.1f}%"
            elif annotate == "both" and pct is not None:
                text = f"{count}\n({pct:.1f}%)"
            else:
                text = str(count)

            text_color = "white" if color_mat[i, j] > thresh else "black"
            ax.text(
                j, i,
                text,
                ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold",
            )

    fig.tight_layout()
    return fig, ax


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cm = [[9831, 7336 ,2015 ,  10 , 130],
 [  21 , 422  , 57  ,  0   ,16],
 [  16  , 30 ,1292  ,  0  , 17],
 [  62 , 119  ,156  , 36 ,   1],
 [ 108  , 31 , 237   , 0, 1428],
    ]
    labels = ["N", "S", "V", "F", "Q"]

    fig, ax = plot_confusion_matrix(cm, labels, normalize="row", annotate="count")
    # plt.savefig("/mnt/user-data/outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    # print("Saved → /mnt/user-data/outputs/confusion_matrix.png")
    plt.title("Invariaint Traning batch = 16, lambda = 0 (F1 = 0.4575)", fontsize=14, pad=20)
    plt.show()