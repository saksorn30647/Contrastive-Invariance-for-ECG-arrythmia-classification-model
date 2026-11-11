"""Split protocols for MIT-BIH, plus a leakage audit that quantifies the
difference between them.

Why this module exists
----------------------
This project reports inter-patient (record-disjoint) results, which are lower
than most published MIT-BIH numbers because most published numbers come from a
random split over pooled beats. The reference paper (LMUEBCNet, Xie & Lin 2023)
uses random 10-fold CV and explicitly states it "regarded every single heartbeat
as different data numbers (not subject-related)" — its own Limitations section
names the absence of subject-level CV as limitation #1.

So the honest comparison is not "our 0.58 vs their 0.82". It is the SAME model
and SAME code evaluated under BOTH protocols, with the gap between them
reported as the protocol effect. That is what this module enables.

`random_split_indices` deliberately implements the leaky protocol. It is not a
bug and it is not an alternative to the inter-patient split — it is a
measurement instrument for how much patient leakage is worth in this dataset.
`split_leakage_report` quantifies the leakage so the number can be stated rather
than asserted.

Bug #7 note (see CLAUDE.md): `records.npy` entries can be `np.bytes_`/`np.str_`,
and `y_cwt.npy` stores AAMI labels as ALPHABET characters — the integer mapping
happens inside `ECGDataset`. Every function here takes an ALREADY-MAPPED integer
`y` (i.e. `ECGDataset.y`) and normalises record IDs with `str(r).strip()`.
"""

import numpy as np

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# The 38/10 record layout this project has used throughout. Kept here so the
# inter-patient protocol has one definition instead of being re-typed per
# script. NOTE: this includes all 48 records and all four paced records
# (102/104/107 train, 217 val); the De Chazal DS1/DS2 protocol uses 22/22 and
# excludes paced records. See CLAUDE.md "Per-record class concentration".
TRAIN_RECORDS = [
    "100", "101", "102", "104", "105", "106", "107", "108", "109", "112",
    "113", "114", "116", "117", "118", "119", "123", "124", "200", "202",
    "203", "205", "207", "210", "213", "214", "215", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
VAL_RECORDS = [
    "103", "111", "115", "121", "122", "201", "208", "209", "212", "217",
]


def _norm_records(records):
    """Normalise record IDs to plain strings (bug #7: dtype can be np.bytes_)."""
    return np.array([str(r).strip() for r in records])


# ── INCART record-disjoint split (added 2026-09-02, reverse-direction expt) ──
# Built by a greedy per-class balance that assigns the RAREST class first, so F
# diversity lands on the train side - that is the entire point of the reverse
# direction. Resulting train F: 163 beats / 21 records / 7.51 EFFECTIVE records,
# against MIT-BIH train F's 428 / 15 / 1.39. Regenerate or re-audit with
# scratchpad/design_incart_split.py.
#
# Val is 8 records / 18,924 beats and exists only to select checkpoints; the real
# test set is all 109,451 MIT-BIH beats. NOTE val F is 56 beats from ONE record
# (I18), so val F1(F) is as single-patient as MIT-BIH's was - do not read it as a
# result, read the MIT-BIH test number.
INCART_TRAIN_RECORDS = [
    "I01", "I02", "I03", "I04", "I05", "I06", "I07", "I09", "I10", "I11",
    "I12", "I13", "I14", "I15", "I16", "I17", "I20", "I21", "I22", "I23",
    "I24", "I25", "I26", "I27", "I28", "I31", "I32", "I34", "I35", "I36",
    "I37", "I38", "I39", "I40", "I41", "I42", "I43", "I44", "I45", "I46",
    "I47", "I48", "I49", "I50", "I51", "I52", "I54", "I55", "I56", "I57",
    "I58", "I59", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68",
    "I69", "I70", "I71", "I72", "I73", "I74", "I75",
]
INCART_VAL_RECORDS = [
    "I08", "I18", "I19", "I29", "I30", "I33", "I53", "I60",
]


def incart_inter_patient_indices(records):
    """Record-disjoint INCART split. `records` is data/records.npy (INCART)."""
    import numpy as _np
    r = _np.array([str(x).strip() for x in records])
    tr = _np.flatnonzero(_np.isin(r, INCART_TRAIN_RECORDS))
    va = _np.flatnonzero(_np.isin(r, INCART_VAL_RECORDS))
    assert len(_np.intersect1d(tr, va)) == 0, "INCART train/val overlap"
    assert len(tr) + len(va) == len(r), (
        f"INCART split does not cover all beats: {len(tr)}+{len(va)} != {len(r)}")
    return tr, va


def inter_patient_indices(records, train_records=None, val_records=None):
    """Record-disjoint split. No patient appears on both sides.

    Returns (train_idx, val_idx) as int64 arrays.
    """
    train_records = list(TRAIN_RECORDS if train_records is None else train_records)
    val_records = list(VAL_RECORDS if val_records is None else val_records)

    overlap = set(train_records) & set(val_records)
    if overlap:
        raise ValueError(f"Inter-patient split is not disjoint: {sorted(overlap)}")

    rec = _norm_records(records)
    train_idx = np.flatnonzero(np.isin(rec, train_records))
    val_idx = np.flatnonzero(np.isin(rec, val_records))

    unassigned = len(rec) - len(train_idx) - len(val_idx)
    if unassigned:
        raise ValueError(
            f"{unassigned} beats belong to neither split — record lists are "
            f"incomplete for this data array."
        )
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def random_split_indices(y, val_fraction=0.2, seed=0, stratified=True):
    """Random split over POOLED BEATS — the literature-comparable protocol.

    Patient identity is ignored by construction, so beats from the same record
    (and therefore the same patient's characteristic morphology) land on both
    sides. That is the point: this reproduces the protocol under which most
    published MIT-BIH results — including the reference paper's — were obtained.

    Args:
        y: integer class labels (use `ECGDataset.y`, NOT raw `y_cwt.npy`).
        val_fraction: held-out proportion. 0.2 gives the 8:2 split.
        seed: split RNG seed. Kept SEPARATE from the training seed on purpose —
            see `random_split_training.py`. Holding this fixed while varying the
            training seed isolates init variance; varying it while holding the
            training seed isolates split variance. Conflating them measures
            neither.
        stratified: preserve per-class proportions exactly. Default True.
            Class F is 0.73% of the data, so an unstratified draw leaves its
            val count to chance; stratifying removes that as a source of
            run-to-run noise. Unstratified is available for a closer match to a
            naive `train_test_split`.

    Returns (train_idx, val_idx) as sorted int64 arrays.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)

    if not stratified:
        perm = rng.permutation(n)
        n_val = int(round(n * val_fraction))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
    else:
        train_parts, val_parts = [], []
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            perm = rng.permutation(cls_idx)
            n_val = int(round(len(perm) * val_fraction))
            # Never let stratification empty a class on either side: a val set
            # with zero F beats makes F1(F) undefined, and a train set with zero
            # F beats silently changes the task.
            n_val = min(max(n_val, 1), len(perm) - 1) if len(perm) > 1 else 0
            val_parts.append(perm[:n_val])
            train_parts.append(perm[n_val:])
        val_idx = np.concatenate(val_parts)
        train_idx = np.concatenate(train_parts)

    train_idx = np.sort(train_idx).astype(np.int64)
    val_idx = np.sort(val_idx).astype(np.int64)

    assert len(np.intersect1d(train_idx, val_idx)) == 0, "train/val index overlap"
    assert len(train_idx) + len(val_idx) == n, "split does not cover all beats"
    return train_idx, val_idx


def split_leakage_report(records, y, train_idx, val_idx, class_names=None):
    """Quantify patient leakage between two index sets.

    The headline number is `pct_val_beats_class_record_seen`: the share of
    validation beats whose (record, class) pair ALSO appears in training. For a
    record-disjoint split this is 0% by construction. For a random split it
    approaches 100%, which is the mechanism behind inflated minority-class
    scores — the model can match a specific patient's rendition of a class
    rather than the class itself.

    Returns a JSON-serialisable dict.
    """
    class_names = class_names or CLASS_NAMES
    rec = _norm_records(records)
    y = np.asarray(y)

    if len(rec) != len(y):
        raise ValueError(
            f"records ({len(rec)}) and y ({len(y)}) length mismatch — record "
            f"labels are misaligned with beats (see CLAUDE.md bug #7)."
        )

    rec_tr, rec_va = set(rec[train_idx]), set(rec[val_idx])
    shared = sorted(rec_tr & rec_va)

    # (record, class) pairs present in train
    train_pairs = set(zip(rec[train_idx].tolist(), y[train_idx].tolist()))

    per_class = {}
    for cls in range(len(class_names)):
        v_mask = y[val_idx] == cls
        n_val_cls = int(v_mask.sum())
        n_train_cls = int((y[train_idx] == cls).sum())
        if n_val_cls:
            v_recs = rec[val_idx][v_mask]
            seen = sum(1 for r in v_recs if (r, cls) in train_pairs)
            pct_seen = 100.0 * seen / n_val_cls
            n_val_records = len(set(v_recs.tolist()))
        else:
            pct_seen, n_val_records = float("nan"), 0
        per_class[class_names[cls]] = {
            "n_train": n_train_cls,
            "n_val": n_val_cls,
            "n_val_records": n_val_records,
            "pct_val_beats_class_record_seen": pct_seen,
        }

    n_val_seen = sum(1 for r in rec[val_idx] if r in rec_tr)
    all_pairs = sum(
        1 for r, c in zip(rec[val_idx].tolist(), y[val_idx].tolist())
        if (r, c) in train_pairs
    )

    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_records_train": len(rec_tr),
        "n_records_val": len(rec_va),
        "n_records_shared": len(shared),
        "shared_records": shared,
        "pct_val_beats_from_a_train_record": 100.0 * n_val_seen / max(len(val_idx), 1),
        "pct_val_beats_class_record_seen": 100.0 * all_pairs / max(len(val_idx), 1),
        "per_class": per_class,
    }


def format_leakage_report(rep, title="SPLIT AUDIT"):
    """Human-readable rendering of `split_leakage_report` for run logs."""
    lines = [
        "──────────────────────────────────────────────────────────────",
        f"  {title}",
        "──────────────────────────────────────────────────────────────",
        f"  train / val beats      : {rep['n_train']} / {rep['n_val']}",
        f"  train / val records    : {rep['n_records_train']} / {rep['n_records_val']}",
        f"  records on BOTH sides  : {rep['n_records_shared']}"
        + ("  (record-disjoint)" if rep["n_records_shared"] == 0 else "  ** PATIENT LEAKAGE **"),
        f"  val beats from a record also in train : "
        f"{rep['pct_val_beats_from_a_train_record']:.2f}%",
        f"  val beats whose (record,class) is in train : "
        f"{rep['pct_val_beats_class_record_seen']:.2f}%   <-- leakage headline",
        "",
        f"  {'class':<7}{'n_train':>9}{'n_val':>8}{'val recs':>10}{'(rec,cls) seen in train':>26}",
    ]
    for name, d in rep["per_class"].items():
        pct = d["pct_val_beats_class_record_seen"]
        pct_s = "   n/a" if pct != pct else f"{pct:6.2f}%"
        lines.append(
            f"  {name:<7}{d['n_train']:>9}{d['n_val']:>8}{d['n_val_records']:>10}{pct_s:>26}"
        )
    lines.append("──────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def kfold_indices(y, n_folds=5, seed=0, stratified=True):
    """Stratified k-fold over POOLED BEATS. Yields (train_idx, val_idx) per fold.

    This is the literature-comparable protocol: LMUEBCNet (Xie & Lin 2023) used
    random 10-fold CV over pooled beats, explicitly "not subject-related". With
    n_folds=5 each fold's validation side is 20% of the beats, so fold 0 is
    directly comparable in size to `random_split_indices(val_fraction=0.2)`.

    DO NOT emulate this by calling `random_split_indices` with 5 different seeds.
    That gives five INDEPENDENT 80/20 draws whose validation sets overlap: a beat
    can appear in 3 val sets and another in none. This function guarantees the
    defining property of cross-validation instead --

        every beat appears in EXACTLY ONE validation fold,
        and the k validation folds partition the dataset.

    -- which `verify_kfold` below asserts explicitly.

    Args:
        y: INTEGER class labels. Map through `dataset2D.LABEL_MAP` first; see
           bug #10 in CLAUDE.md -- passing the raw `<U1` characters silently
           produces a different partition because `np.unique` orders them
           F,N,Q,S,V instead of N,S,V,F,Q.
        n_folds: number of folds. 5 gives the 80/20 ratio used elsewhere here.
        seed: shuffling seed. Kept separate from the training seed on purpose,
           exactly as `random_split_indices`' `seed` is.
        stratified: keep each class's proportion equal across folds. Default
           True. F is 0.73% of MIT-BIH, so an unstratified 5-fold draw would
           leave per-fold F counts to chance.

    Returns:
        list of (train_idx, val_idx), both sorted int64 arrays, length n_folds.
    """
    import numpy as _np

    y = _np.asarray(y)
    n = len(y)
    if not 2 <= n_folds <= n:
        raise ValueError(f"n_folds must be in [2, {n}], got {n_folds}")
    rng = _np.random.default_rng(seed)

    # fold_of[i] = which fold beat i belongs to (as VALIDATION)
    fold_of = _np.empty(n, dtype=_np.int64)

    if stratified:
        for cls in _np.unique(y):
            idx = _np.flatnonzero(y == cls)
            perm = rng.permutation(idx)
            # Deal round-robin rather than in contiguous chunks so that when a
            # class size is not divisible by n_folds the remainder is spread one
            # beat per fold instead of landing entirely in the last fold.
            fold_of[perm] = _np.arange(len(perm)) % n_folds
    else:
        perm = rng.permutation(n)
        fold_of[perm] = _np.arange(n) % n_folds

    out = []
    all_idx = _np.arange(n)
    for k in range(n_folds):
        val_idx = _np.sort(all_idx[fold_of == k])
        train_idx = _np.sort(all_idx[fold_of != k])
        out.append((train_idx, val_idx))
    return out


def verify_kfold(folds, n, y=None, class_names=None):
    """Assert that `folds` really is a cross-validation partition.

    Checks, in order of how badly each would corrupt a CV result:
      1. every beat is in EXACTLY ONE validation fold
      2. train and val are disjoint within every fold
      3. train u val == the whole dataset, within every fold
      4. (if y given) every class is present in every validation fold

    Returns a dict of per-fold stats for reporting.
    """
    import numpy as _np

    seen = _np.zeros(n, dtype=_np.int64)
    stats = []
    for k, (tr, va) in enumerate(folds):
        assert len(_np.intersect1d(tr, va)) == 0, f"fold {k}: train/val overlap"
        assert len(tr) + len(va) == n, (
            f"fold {k}: train+val = {len(tr)}+{len(va)} != {n}")
        seen[va] += 1
        row = dict(fold=k, n_train=int(len(tr)), n_val=int(len(va)))
        if y is not None:
            yv = _np.asarray(y)[va]
            cls, cnt = _np.unique(yv, return_counts=True)
            row["val_class_counts"] = {
                (class_names[c] if class_names else int(c)): int(n_)
                for c, n_ in zip(cls, cnt)}
        stats.append(row)

    bad = int((seen != 1).sum())
    assert bad == 0, (
        f"{bad} beats are not in exactly one validation fold "
        f"(min {seen.min()}, max {seen.max()}) -- this is NOT a partition")
    return stats


def build_split(protocol, records, y, val_fraction=0.2, split_seed=0,
                stratified=True, fold=None, n_folds=5):
    """Dispatch to a protocol and return (train_idx, val_idx, leakage_report).

    protocol='kfold' requires `fold` in [0, n_folds). It is the literature-
    comparable arm: LMUEBCNet used random 10-fold CV over pooled beats. Note it
    is deliberately leaky in the same way `random` is -- the folds ignore patient
    identity by construction -- so the leakage report will show ~100% and that is
    the point, not a bug.
    """
    if protocol == "inter_patient":
        train_idx, val_idx = inter_patient_indices(records)
    elif protocol == "random":
        train_idx, val_idx = random_split_indices(
            y, val_fraction=val_fraction, seed=split_seed, stratified=stratified
        )
    elif protocol == "kfold":
        if fold is None:
            raise ValueError("protocol='kfold' requires fold=<int>")
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0,{n_folds}), got {fold}")
        train_idx, val_idx = kfold_indices(
            y, n_folds=n_folds, seed=split_seed, stratified=stratified)[fold]
    else:
        raise ValueError(f"Unknown protocol {protocol!r}")
    return train_idx, val_idx, split_leakage_report(records, y, train_idx, val_idx)
