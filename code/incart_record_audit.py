"""INCART per-record class concentration audit — the symmetric check to the
MIT-BIH one that located this project's root cause.

The MIT-BIH audit (CLAUDE.md "Per-record class concentration") showed F and S
are effectively single-patient classes on the TRAIN side: F draws on 1.39
effective records, S on 2.57, and per-class F1 rank-orders almost perfectly with
that number. That audit was never run on INCART, so it has never been checked
whether the zero-shot TEST set has the same structure.

Why it matters: if INCART's 219 F beats also come from one or two patients, then
cross-dataset F1(F) is measuring "does a boundary fitted to MIT-BIH patient 213
happen to transfer to INCART patient X" — a second single-patient lottery
stacked on the first. That is a different sentence in the writeup than "the
model cannot represent fusion beats."

Effective record count = inverse-Simpson index 1 / sum(p_i^2) over each class's
per-record proportions: how many patients a class *effectively* draws on,
discounting records contributing a handful of beats.

Bug #7 compliance: labels are read from the ALPHABET-encoded y_incart_cwt.npy
and mapped explicitly; record IDs are normalised with str().strip(); the
len(records) == len(y) assertion is present.

Run: python code/incart_record_audit.py
"""

import json
import os

import numpy as np

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
LABEL_MAP = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}

# MIT-BIH figures from CLAUDE.md, for side-by-side comparison.
MITBIH_TRAIN_EFF = {"N": 33.25, "S": 2.57, "V": 12.33, "F": 1.39, "Q": 3.01}
MITBIH_VAL_EFF = {"N": 8.91, "S": 1.63, "V": 1.75, "F": 1.01, "Q": 1.00}


def effective_records(counts):
    """Inverse-Simpson index over a per-record count dict."""
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    p = np.array([c / tot for c in counts.values()])
    return float(1.0 / np.sum(p ** 2))


def main():
    y_raw = np.load("data/y_incart_cwt.npy")
    records = np.array([str(r).strip() for r in np.load("data/records.npy")])  # INCART record ids (75 records / 175,811 beats). NOT root records.npy, which is MIT-BIH.

    assert len(records) == len(y_raw), (
        f"records ({len(records)}) != labels ({len(y_raw)}) — record labels are "
        f"misaligned with beats; every per-record result would be wrong."
    )

    y = np.array([LABEL_MAP[v] for v in y_raw])
    n_records_total = len(set(records.tolist()))
    print(f"INCART: {len(y)} beats across {n_records_total} records\n")

    print("PER-RECORD CLASS CONCENTRATION — INCART (zero-shot test set)")
    print("=" * 96)
    print(f"{'class':<7}{'beats':>8}{'%':>8}{'#recs':>8}{'eff.recs':>10}"
          f"{'top record':>14}{'top share':>11}{'MIT train eff':>15}")
    print("-" * 96)

    out = {}
    for cls, name in enumerate(CLASS_NAMES):
        mask = y == cls
        n = int(mask.sum())
        if n == 0:
            print(f"{name:<7}{0:>8}{0.0:>7.3f}%{0:>8}{'-':>10}{'-':>14}{'-':>11}")
            out[name] = {"beats": 0}
            continue
        recs = records[mask]
        uniq, cnts = np.unique(recs, return_counts=True)
        counts = dict(zip(uniq.tolist(), cnts.tolist()))
        eff = effective_records(counts)
        top = max(counts.items(), key=lambda kv: kv[1])
        print(f"{name:<7}{n:>8}{100*n/len(y):>7.3f}%{len(counts):>8}{eff:>10.2f}"
              f"{top[0]:>14}{100*top[1]/n:>10.1f}%{MITBIH_TRAIN_EFF[name]:>15.2f}")
        out[name] = {
            "beats": n,
            "pct": 100 * n / len(y),
            "n_records": len(counts),
            "effective_records": eff,
            "top_record": top[0],
            "top_share_pct": 100 * top[1] / n,
            "per_record": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }
    print("=" * 96)

    # Raw counts for the rare classes, for reproducibility in the writeup.
    for name in ("F", "S", "Q"):
        d = out.get(name, {})
        if d.get("beats"):
            items = list(d["per_record"].items())
            shown = ", ".join(f"{r}:{c}" for r, c in items[:20])
            more = "" if len(items) <= 20 else f", ... (+{len(items)-20} records)"
            print(f"\n{name} ({d['beats']} beats, {d['n_records']} records): {shown}{more}")

    os.makedirs("runs/_analysis", exist_ok=True)
    path = "runs/_analysis/incart_record_concentration.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
