# Data: acquisition and regeneration

Nothing under `data/` is in this repository — the preprocessed arrays total ~11 GB. This
document is enough to rebuild all of it from the public sources.

## 1. Download the databases

Both are open-access on PhysioNet.

| Database | Records | PhysioNet | Target directory |
| --- | --- | --- | --- |
| MIT-BIH Arrhythmia | 48 | https://physionet.org/content/mitdb/1.0.0/ | `data/mit-bih/` |
| St. Petersburg INCART 12-lead | 75 | https://physionet.org/content/incartdb/1.0.0/ | `data/incart/` |

```bash
python -c "import wfdb; wfdb.dl_database('mitdb',    'data/mit-bih')"
python -c "import wfdb; wfdb.dl_database('incartdb', 'data/incart')"
```

Roughly 77 MB and 591 MB zipped respectively. Each record is a `.dat`/`.hea`/`.atr` triple;
the `.atr` annotations supply both the R-peak sample indices and the beat labels.

## 2. Run the pipeline, in this order

```bash
python code/beat_extrac_label.py          # step 1a  MIT-BIH beats
python code/beat_extrac_label_INCART.py   # step 1b  INCART beats
python code/CWT_preprocess.py             # step 2a  MIT-BIH scalograms
python code/CWT_preprocess_incart.py      # step 2b  INCART scalograms
```

Each script carries a header comment restating its inputs, outputs and position in the order.

### What gets produced

| File | Shape / dtype | Size | Produced by |
| --- | --- | --- | --- |
| `data/X.npy` | (109451, 256) float | 224 MB | 1a |
| `data/y.npy` | (109451,) `<U1` **characters** | 438 KB | 1a |
| `records.npy` | (109451,) — **root, MIT-BIH** | 1.3 MB | 1a |
| `data/X_incart.npy` | (175811, 256) float | 360 MB | 1b |
| `data/y_incart.npy` | (175811,) characters | 703 KB | 1b |
| `data/records.npy` | (175811,) — **INCART** | 2.1 MB | 1b |
| `data/X_cwt.npy` | (109451, 64, 256) float16 | **3.59 GB** | 2a |
| `data/y_cwt.npy` | (109451,) characters | 438 KB | 2a |
| `data/X_incart_cwt.dat` | (175811, 64, 256) float16 **memmap** | **5.76 GB** | 2b |
| `data/y_incart_cwt.npy` | (175811,) characters | 703 KB | 2b |

> **The two `records.npy` files are different datasets.** MIT-BIH record ids live at the
> repository **root**; INCART record ids live in **`data/`**. This split is deliberate and both
> paths are load-bearing — see the table in the README for which script reads which. An earlier
> version of `beat_extrac_label_INCART.py` wrote INCART ids to the root path, silently
> overwriting the MIT-BIH array and breaking every MIT-BIH split; that is fixed here.

> **`X_incart_cwt.dat` is a raw memmap, not an `.npy`.** Readers must supply the shape:
> `ECGDataset("data/X_incart_cwt.dat", "data/y_incart_cwt.npy", shape=(175811, 64, 256), memmap=True)`

## 3. Verify the rebuild

Class counts after preprocessing — these should match exactly:

| | N | S | V | F | Q | total |
| --- | --- | --- | --- | --- | --- | --- |
| MIT-BIH | 90,593 | 2,781 | 7,235 | 802 | 8,040 | **109,451** |
| INCART | 153,621 | 1,959 | 20,006 | 219 | 6 | **175,811** |

```bash
python code/split_tables.py        # class composition of all three split protocols
python code/check_label_compat.py  # AAMI symbol audit across both databases
```

Raw annotation counts are 109,494 and 175,874; the difference is beats whose 256-sample window
would cross a record boundary and are dropped. For fusion specifically, 803 raw → 802 kept.

## 4. Processing parameters

Identical for both databases except where noted.

| | |
| --- | --- |
| Lead | MIT-BIH channel 0 (MLII in 45/48; **V5** in 102, 104, 114) · INCART channel 1 (lead II, all 75) |
| Resampling | INCART only, 257 → 360 Hz (`scipy.signal.resample`); annotation indices rescaled by 360/257 |
| Filter | Butterworth bandpass, 4–40 Hz, order 4, applied to the full record |
| Window | 256 samples (0.711 s at 360 Hz) centred on the annotated R peak |
| Normalisation | per-beat z-score, after windowing |
| Transform | Morlet (`'morl'`) CWT, scales 1–64, then `log1p(abs(coefficients))`, stored float16 |

AAMI mapping (15 symbols, byte-identical between the two extraction scripts):

```
N <- N L R e j        S <- A a J S        V <- V E        F <- F        Q <- / f Q
```

## 5. Known structural properties of this data

These are measured facts about the databases, not artifacts of the pipeline, and they explain
most of the per-class results:

- **Minority classes are effectively single-patient under the inter-patient split.** Effective
  record count (inverse-Simpson, MIT-BIH training side): N 33.25, V 12.33, Q 3.01, **S 2.57,
  F 1.39**. 85% of training F beats come from record 213; 99.5% of validation F beats come from
  record 208.
- **INCART contains no paced beats.** Its entire Q class is 6 unclassifiable beats, against
  8,040 in MIT-BIH (7,028 paced + 982 fusion-of-paced + 33 unclassifiable). Q is a label-space
  void cross-dataset, not a rare class.
- **MIT-BIH contains rhythms INCART lacks**: 8,075 `L` (LBBB) and `e` inside class N, `a` and
  `J` inside S, `E` inside V. INCART's S is 99.2% `A` with zero `a` and zero `J`.
- **Class priors differ substantially**: F is 0.733% of MIT-BIH against 0.125% of INCART
  (5.9× rarer); Q is 7.35% against 0.003%.
