"""Class composition of every split protocol used in this project.

Reads only the label and record arrays (not the 3.6/5.8 GB scalogram files), so
it is safe to run while training is in flight.

Three protocols, all reported the same way:
  1. FORWARD  inter-patient   MIT-BIH 38/10 records  -> zero-shot INCART
  2. FORWARD  random 8:2      MIT-BIH pooled beats   -> zero-shot INCART
  3. REVERSE  inter-patient   INCART  67/8  records  -> zero-shot MIT-BIH

Labels are mapped through dataset2D.LABEL_MAP before any split call. See bug #10:
random_split_indices stratifies over np.unique(y), which orders the raw <U1
characters F,N,Q,S,V but the ints N,S,V,F,Q — feeding it characters silently
returns a DIFFERENT partition.

Run: python code/split_tables.py    (use PYTHONIOENCODING=utf-8)
"""
import sys

import numpy as np

sys.path.insert(0, "code")
from utils.dataset2D import LABEL_MAP                      # noqa: E402
from utils.splits import (incart_inter_patient_indices,    # noqa: E402
                          inter_patient_indices,
                          random_split_indices)

NAMES = ["N", "S", "V", "F", "Q"]


def load(records_path, labels_path):
    rec = np.array([str(r).strip() for r in np.load(records_path)])
    y = np.array([LABEL_MAP[str(v).strip()] for v in np.load(labels_path)])
    assert len(rec) == len(y), f"{records_path}: record/beat misalignment"
    return rec, y


def eff_records(rec, y, idx, cls):
    """Inverse-Simpson effective record count for one class within `idx`."""
    sel = idx[y[idx] == cls]
    if len(sel) == 0:
        return 0.0, 0
    _, counts = np.unique(rec[sel], return_counts=True)
    p = counts / counts.sum()
    return float(1.0 / (p ** 2).sum()), len(counts)


def block(title, rec, y, sides):
    """sides: list of (label, idx_or_None). None means 'all beats'."""
    print("=" * 88)
    print(title)
    print("=" * 88)
    header = f"{'class':<6}"
    for name, _ in sides:
        header += f"{name:>26}"
    print(header)
    sub = f"{'':<6}" + "".join(f"{'n':>10}{'%':>9}{'effrec':>7}"
                              for _ in sides)
    print(sub)
    print("-" * 88)

    idxs = [np.arange(len(y)) if i is None else i for _, i in sides]
    for c, cname in enumerate(NAMES):
        row = f"{cname:<6}"
        for idx in idxs:
            n = int((y[idx] == c).sum())
            pct = 100.0 * n / len(idx) if len(idx) else 0.0
            eff, _ = eff_records(rec, y, idx, c)
            row += f"{n:>10}{pct:>8.3f}%{eff:>7.2f}"
        print(row)
    print("-" * 88)
    tot = f"{'TOTAL':<6}"
    for idx in idxs:
        tot += f"{len(idx):>10}{100.0:>8.1f}%{len(np.unique(rec[idx])):>7d}"
    print(tot + "   (last column = distinct records)")
    print()


# ── 1 & 2. MIT-BIH as source ────────────────────────────────────────────────
# records.npy at the ROOT is MIT-BIH; data/records.npy is INCART. Same name,
# different dataset -- swapping them misaligns every per-record statistic.
m_rec, m_y = load("records.npy", "data/y_cwt.npy")            # MIT-BIH, 48 recs
i_rec, i_y = load("data/records.npy", "data/y_incart_cwt.npy")  # INCART, 75 recs

tr, va = inter_patient_indices(m_rec)
block("1. FORWARD / inter-patient — train+val on MIT-BIH (38/10 records), "
      "test on INCART",
      m_rec, m_y, [("MIT-BIH TRAIN", tr), ("MIT-BIH VAL", va)])
block("   ...its zero-shot target: ALL of INCART",
      i_rec, i_y, [("INCART TEST (all)", None)])

rtr, rva = random_split_indices(m_y, val_fraction=0.2, seed=0, stratified=True)
block("2. FORWARD / random 8:2 — train+val on MIT-BIH (pooled beats), "
      "test on INCART",
      m_rec, m_y, [("MIT-BIH TRAIN", rtr), ("MIT-BIH VAL", rva)])

# ── 3. INCART as source (reverse direction) ─────────────────────────────────
itr, iva = incart_inter_patient_indices(i_rec)
block("3. REVERSE / inter-patient — train+val on INCART (67/8 records), "
      "test on MIT-BIH",
      i_rec, i_y, [("INCART TRAIN", itr), ("INCART VAL", iva)])
block("   ...its zero-shot target: ALL of MIT-BIH",
      m_rec, m_y, [("MIT-BIH TEST (all)", None)])

# ── the comparison the reverse experiment exists to make ────────────────────
print("=" * 88)
print("THE CONTRAST THAT MOTIVATES THE REVERSE DIRECTION — class F on the TRAIN side")
print("=" * 88)
print(f"{'protocol':<34}{'F beats':>9}{'records':>9}{'EFF records':>13}{'F % of train':>14}")
print("-" * 88)
for label, rec, y, idx in (
        ("FORWARD  MIT-BIH inter-patient", m_rec, m_y, tr),
        ("FORWARD  MIT-BIH random 8:2", m_rec, m_y, rtr),
        ("REVERSE  INCART inter-patient", i_rec, i_y, itr)):
    n = int((y[idx] == 3).sum())
    eff, nrec = eff_records(rec, y, idx, 3)
    print(f"{label:<34}{n:>9}{nrec:>9}{eff:>13.2f}{100.0 * n / len(idx):>13.3f}%")
print()
print("Reverse trains on 62% FEWER fusion beats but 5.4x more effective fusion")
print("patients. That is the whole experiment.")
