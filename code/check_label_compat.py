"""AAMI label-compatibility audit: MIT-BIH vs INCART.

Motivation (CLAUDE.md next-step #6, upgraded 2026-08-11):
  * Q accounts for ~90-100% of the measured 5-class domain-shift drop, while the
    3-class N/S/V drop is ~0. Q is a label-space mismatch, not domain shift.
  * F is mis-scaled on INCART by one to two orders of magnitude in every
    configuration (0 F predictions without the sampler; 16,919 against 219 true
    beats with it).

Both point at the label space rather than the model. This script asks whether
"F" (and the other AAMI classes) denote the same annotation populations in the
two databases, at three levels:

  1. Code level  — are AAMI_MAP / VALID_SYMBOLS identical in the two extraction
                   scripts? (If they differ, everything downstream is suspect.)
  2. Symbol level — what raw annotation symbols actually occur in each database,
                   how do they roll up to AAMI classes, and how many beats does
                   VALID_SYMBOLS silently discard in each?
  3. Signal level — do INCART F scalograms look like MIT-BIH F scalograms?

Run from the project root:
    python code/check_label_compat.py
    python code/check_label_compat.py --no-plot     # skip the scalogram figure
"""

import argparse
import ast
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MITBIH_DIR = os.path.join("data", "mit-bih")
INCART_DIR = os.path.join("data", "incart", "files")
OUT_DIR = os.path.join("runs", "_analysis")

MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
INCART_RECORDS = [f"I{i:02d}" for i in range(1, 76)]

CLASS_ORDER = ["N", "S", "V", "F", "Q"]


def literal_from_script(path, name):
    """Pull a top-level literal assignment out of a script without importing it.

    The extraction scripts run their whole pipeline at import time, so importing
    them here would kick off hours of preprocessing. Parsing the AST instead.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def part1_code_level():
    print("=" * 78)
    print("PART 1 — code level: are the two extraction scripts using the same map?")
    print("=" * 78)
    mit = "code/beat_extrac_label.py"
    inc = "code/beat_extrac_label_INCART.py"

    m_map = literal_from_script(mit, "AAMI_MAP")
    i_map = literal_from_script(inc, "AAMI_MAP")
    m_val = literal_from_script(mit, "VALID_SYMBOLS")
    i_val = literal_from_script(inc, "VALID_SYMBOLS")

    print(f"  AAMI_MAP identical      : {m_map == i_map}  "
          f"({len(m_map)} symbols each)")
    print(f"  VALID_SYMBOLS identical : {m_val == i_val}  "
          f"({len(m_val)} symbols each)")

    for k in sorted(set(m_map) | set(i_map)):
        a, b = m_map.get(k), i_map.get(k)
        if a != b:
            print(f"    !! '{k}': MIT-BIH -> {a}   INCART -> {b}")
    only_mapped_not_valid = sorted(set(m_map) - set(m_val))
    only_valid_not_mapped = sorted(set(m_val) - set(m_map))
    if only_mapped_not_valid:
        print(f"  NOTE mapped but not in VALID_SYMBOLS: {only_mapped_not_valid}")
    if only_valid_not_mapped:
        print(f"  NOTE in VALID_SYMBOLS but unmapped  : {only_valid_not_mapped}")
    return m_map, m_val


def scan_symbols(directory, record_names, label):
    """Count every annotation symbol actually present in a database."""
    import wfdb

    counts = Counter()
    missing = []
    for rec in record_names:
        path = os.path.join(directory, rec)
        try:
            ann = wfdb.rdann(path, "atr")
        except Exception:
            missing.append(rec)
            continue
        counts.update(ann.symbol)
    print(f"  {label}: read {len(record_names) - len(missing)}/{len(record_names)} records"
          + (f"  (missing: {missing[:6]}{'...' if len(missing) > 6 else ''})" if missing else ""))
    return counts


def part2_symbol_level(aami_map, valid_symbols):
    print()
    print("=" * 78)
    print("PART 2 — symbol level: what does each database actually contain?")
    print("=" * 78)

    mit_counts = scan_symbols(MITBIH_DIR, MITBIH_RECORDS, "MIT-BIH")
    inc_counts = scan_symbols(INCART_DIR, INCART_RECORDS, "INCART ")

    if not mit_counts or not inc_counts:
        print("  !! could not read one of the databases — skipping symbol comparison")
        return

    # Non-beat annotations (rhythm/quality markers) are not beats at all.
    all_syms = sorted(set(mit_counts) | set(inc_counts))
    print(f"\n  {'sym':<6}{'AAMI':<7}{'kept?':<8}{'MIT-BIH':>12}{'INCART':>12}"
          f"{'MIT %':>9}{'INC %':>9}")
    mit_beats = sum(v for k, v in mit_counts.items() if k in valid_symbols)
    inc_beats = sum(v for k, v in inc_counts.items() if k in valid_symbols)
    for s in all_syms:
        m, i = mit_counts.get(s, 0), inc_counts.get(s, 0)
        aami = aami_map.get(s, "-")
        kept = "kept" if s in valid_symbols else "DROPPED"
        mp = 100 * m / mit_beats if (s in valid_symbols and mit_beats) else float("nan")
        ip = 100 * i / inc_beats if (s in valid_symbols and inc_beats) else float("nan")
        mps = f"{mp:8.3f}" if mp == mp else "       -"
        ips = f"{ip:8.3f}" if ip == ip else "       -"
        print(f"  {repr(s):<6}{aami:<7}{kept:<8}{m:>12,}{i:>12,}{mps:>9}{ips:>9}")

    print(f"\n  Beats KEPT   — MIT-BIH {mit_beats:,}   INCART {inc_beats:,}")
    mit_drop = sum(v for k, v in mit_counts.items() if k not in valid_symbols)
    inc_drop = sum(v for k, v in inc_counts.items() if k not in valid_symbols)
    print(f"  Annots DROPPED — MIT-BIH {mit_drop:,}   INCART {inc_drop:,} "
          f"(includes non-beat rhythm/quality markers)")

    print(f"\n  {'AAMI':<7}{'MIT-BIH':>12}{'MIT %':>9}{'INCART':>12}{'INC %':>9}"
          f"   contributing symbols")
    for cls in CLASS_ORDER:
        syms = [s for s in valid_symbols if aami_map.get(s) == cls]
        m = sum(mit_counts.get(s, 0) for s in syms)
        i = sum(inc_counts.get(s, 0) for s in syms)
        m_detail = ", ".join(f"{s}:{mit_counts.get(s,0):,}" for s in syms if mit_counts.get(s, 0))
        i_detail = ", ".join(f"{s}:{inc_counts.get(s,0):,}" for s in syms if inc_counts.get(s, 0))
        print(f"  {cls:<7}{m:>12,}{100*m/mit_beats:>8.3f}%{i:>12,}{100*i/inc_beats:>8.3f}%")
        print(f"         MIT-BIH: {m_detail or '(none)'}")
        print(f"         INCART : {i_detail or '(none)'}")

    print("\n  Interpretation notes:")
    for cls in CLASS_ORDER:
        syms = [s for s in valid_symbols if aami_map.get(s) == cls]
        m_syms = {s for s in syms if mit_counts.get(s, 0)}
        i_syms = {s for s in syms if inc_counts.get(s, 0)}
        if m_syms != i_syms:
            print(f"    class {cls}: symbol populations DIFFER — "
                  f"MIT-BIH only {sorted(m_syms - i_syms) or '[]'}, "
                  f"INCART only {sorted(i_syms - m_syms) or '[]'}")


def part3_signal_level(n=6):
    print()
    print("=" * 78)
    print("PART 3 — signal level: do INCART F scalograms resemble MIT-BIH F?")
    print("=" * 78)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from utils.dataset2D import ECGDataset

    mit = ECGDataset("data/X_cwt.npy", "data/y_cwt.npy")
    inc = ECGDataset("data/X_incart_cwt.dat", "data/y_incart_cwt.npy",
                     shape=(175811, 64, 256), memmap=True)

    # Bug #7: labels are alphabet chars in the raw .npy; ECGDataset maps them.
    mit_f = np.where(np.asarray(mit.y) == 3)[0]
    inc_f = np.where(np.asarray(inc.y) == 3)[0]
    print(f"  MIT-BIH F beats: {len(mit_f)}   INCART F beats: {len(inc_f)}")

    rng = np.random.default_rng(0)
    mit_pick = rng.choice(mit_f, size=min(n, len(mit_f)), replace=False)
    inc_pick = rng.choice(inc_f, size=min(n, len(inc_f)), replace=False)

    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.0))
    for j in range(n):
        for row, (picks, ds, name) in enumerate(
                ((mit_pick, mit, "MIT-BIH F"), (inc_pick, inc, "INCART F"))):
            ax = axes[row, j]
            if j < len(picks):
                img = np.asarray(ds.X[picks[j]], dtype=np.float32)
                ax.imshow(img, aspect="auto", origin="lower", cmap="magma")
                ax.set_title(f"{name} #{picks[j]}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Fusion-beat CWT scalograms — MIT-BIH (top) vs INCART (bottom)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "F_scalograms_mitbih_vs_incart.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  figure -> {out}")

    # Cheap quantitative companion to the eyeball test.
    def stats(ds, idx, k=200):
        sel = idx[:k]
        arr = np.stack([np.asarray(ds.X[i], dtype=np.float32) for i in sel])
        return arr.mean(), arr.std(), arr.max()

    mm, ms, mx = stats(mit, mit_f)
    im, isd, ix = stats(inc, inc_f)
    print(f"  MIT-BIH F scalograms: mean {mm:.4f}  std {ms:.4f}  max {mx:.4f}")
    print(f"  INCART  F scalograms: mean {im:.4f}  std {isd:.4f}  max {ix:.4f}")
    print("  (Large scale differences here would mean the CWT/normalisation, not the "
          "label,\n   is what makes INCART F unrecognisable to a MIT-BIH-trained model.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    aami_map, valid_symbols = part1_code_level()
    part2_symbol_level(aami_map, valid_symbols)
    if not args.no_plot:
        part3_signal_level()


if __name__ == "__main__":
    main()
