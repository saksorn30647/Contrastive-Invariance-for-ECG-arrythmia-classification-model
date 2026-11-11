# Cross-dataset ECG arrhythmia classification

Contrastive pretraining and invariance training for AAMI beat classification under
**domain shift**: train on MIT-BIH, test zero-shot on INCART (St. Petersburg 12-lead).
CWT scalograms → 2D CNN, inter-patient (record-disjoint) evaluation throughout.

Wearable Technology and Mobile Healthcare Lab, National Cheng Kung University.
Saksorn Singhasenee, under Prof. Che-Wei Lin.

---

## The four compared models

All four are the **same** `BaselineCNN` (292,005 parameters, 178.5 MMACs/beat, 1.11 MiB fp32)
with the same focal loss (γ=2.0), the same minority-guaranteed sampler (F floor = 1), batch 64,
lr 1e-4, weight decay 1e-5, and 5-class-macro checkpoint selection. **Only two things vary**, so
any difference is attributable to the model:

| Model | Name | Initialisation | λ_inv | Script |
| --- | --- | --- | --- | --- |
| **a** | baseline | random | 0.0 | `code/invariance_training.py` |
| **b** | contrastive + finetune | contrastive-pretrained | 0.0 | `code/pretrained_training.py` |
| **c** | invariance training | random | 0.4 | `code/invariance_training.py --lambda-inv 0.4` |
| **d** | contrastive + invariance | contrastive-pretrained | 0.4 | `code/pretrained_training.py --lambda-inv 0.4` |

Inference cost is **identical** across all four; only training cost differs (λ_inv adds a second
forward/backward pass; contrastive pretraining adds a one-time ~2 h stage).

## Headline results

- **Invariance training significantly improves zero-shot transfer.** `c − a` on INCART 3-class
  macro-F1 = **+0.0499, p = 0.035** (max-T corrected across the three vs-baseline contrasts,
  8/10 seeds), replicating at 4-class and 5-class.
- **In-domain, invariance costs performance and pretraining recovers it.** `c − b` = −0.0395
  (p = 0.0011, 0/10 seeds) and `d − c` = +0.0449 (p = 0.0004, 9/10). The study's most robust
  result; survives correction over all six model pairs. **Nothing beats the baseline in-domain.**
- **The protocol gap is the strongest methodological result.** Same script, same seed, only the
  split differs: F1(F) goes **0.0000 → 0.8635** from a patient-leaking random split, while the
  INCART control moves only +0.0125 — a 39× ratio. The model does not improve; the ruler changes.
- **Transfer works for V, not for S.** Against INCART-supervised training on the same records,
  zero-shot reaches **78% of the ceiling for V** but only **15% for S**.
- **Minority-class failure is a data-structure problem.** Under the inter-patient split, F draws
  on 1.39 and S on 2.57 *effective* training patients (inverse-Simpson). Effective patient
  diversity rank-orders per-class F1 better than any loss, sampler or hyperparameter tried.

Full numbers: [`results/reports/results_pack.md`](results/reports/results_pack.md) (per-class,
per-seed, all protocols) and [`results/reports/model_comparison.md`](results/reports/model_comparison.md)
(every pairwise test with corrections).

---

## Setup

Python **3.10.20**. On the original machine the interpreter is *not* on PATH:

```text
C:\Users\sakso\Anaconda3\envs\ecg_env\python.exe
```

```bash
pip install -r requirements.txt
```

Key versions: torch 2.5.1+cu121, numpy 2.2.6, scipy 1.15.3, PyWavelets 1.8.0,
scikit-learn 1.7.2, wfdb 4.3.1. CUDA 12.1, cuDNN 9.1. Full freeze:
`runs/_analysis/pip_freeze.txt`.

**Run every script from the repository root**, not from inside `code/`:

```bash
python code/invariance_training.py --seed 42
```

## Data

**Not in this repository** (~11 GB). See [`docs/DATA.md`](docs/DATA.md) for download and
regeneration. Short version:

```bash
python -c "import wfdb; wfdb.dl_database('mitdb',    'data/mit-bih')"
python -c "import wfdb; wfdb.dl_database('incartdb', 'data/incart')"

python code/beat_extrac_label.py          # -> data/X.npy, data/y.npy, records.npy
python code/beat_extrac_label_INCART.py   # -> data/X_incart.npy, data/y_incart.npy, data/records.npy
python code/CWT_preprocess.py             # -> data/X_cwt.npy          (3.59 GB)
python code/CWT_preprocess_incart.py      # -> data/X_incart_cwt.dat   (5.76 GB memmap)
```

**Model weights are also not included.** Encoders rebuild with `code/contrastive_pretrain.py`
(~2 h each); fine-tuned checkpoints rebuild from the training scripts with the seeds below.

---

## Repository layout

```
code/
  models/baseline2D.py        the CNN (all four models share it)
  utils/                      cwt, dataset2D, datasetContrastivePretrain,
                              invariance (FocalLoss, InvarianceLoss), NTXentLoss,
                              samplers (MinorityGuaranteedBatchSampler), splits
  beat_extrac_label*.py       step 1: beat extraction
  CWT_preprocess*.py          step 2: scalograms
  invariance_training.py      models a and c          (inter-patient)
  pretrained_training.py      models b and d          (inter-patient)
  dual_protocol_training.py   random 8:2 and 5-fold CV protocols
  reverse_direction_training.py  INCART → MIT-BIH
  contrastive_pretrain.py     NT-Xent encoder pretraining
  <26 analysis scripts>       statistics, tables, figures
scripts/                      PowerShell runners (thermal gate, stall watchdog, sleep guard)
runs/<tag>/metrics.json       109 completed runs — full config + per-epoch history + metrics
runs/_analysis/               result tables (.csv/.json)
results/figures/              15 figures
results/reports/              three write-ups
docs/DATA.md                  data acquisition and regeneration
```

## Reproducing the results

**Everything statistical reproduces with no GPU and no data**, because all 109 `metrics.json`
files are included:

```bash
python code/four_model_ttests.py                  # four-model paired tests, n=10
python code/comparison_chart.py                   # -> results/reports/model_comparison.md
python code/results_pack.py                       # -> results/reports/results_pack.md
python code/kfold_ttests.py --arms a b c d        # 5-fold CV, paired by fold
python code/omnibus_tests.py                      # Friedman + Nemenyi
python code/vs_baseline_tests.py                  # max-T vs-baseline correction
python code/recompute_power.py                    # power analysis
python code/compute_analysis.py                   # params / MACs / latency
```

Needs the data (no GPU): `split_tables.py`, `check_label_compat.py`, `s_class_analysis.py`,
`test_kfold.py`, `incart_record_audit.py`.
Needs data **and** checkpoints: `confusion_matrices.py`, `incart_predict.py`,
`incart_bootstrap.py`, `prior_correction.py`, `incart_ceiling.py`.

### Training commands

```bash
# inter-patient, models a / b / c / d, seeds 42-51
python code/invariance_training.py --seed 42                       --tag focal_sampler
python code/pretrained_training.py --seed 42 --lambda-inv 0        --tag pretrained_focal_sampler
python code/invariance_training.py --seed 42 --lambda-inv 0.4      --tag focal_inv04_sampler
python code/pretrained_training.py --seed 42 --lambda-inv 0.4      --tag pretrained_inv04_sampler

# random 8:2 and 5-fold CV
python code/dual_protocol_training.py --protocol random --seed 42 --split-seed 0
python code/dual_protocol_training.py --protocol kfold --fold 0 --n-folds 5 --split-seed 0 --seed 42

# reverse direction (train INCART, test MIT-BIH)
python code/reverse_direction_training.py --seed 42 --lambda-inv 0.4 --tag reverse_inv04_sampler

# contrastive encoder (must match the split it will be fine-tuned on)
python code/contrastive_pretrain.py --protocol kfold --fold 0 --n-folds 5 \
       --split-seed 0 --seed 42 --epochs 50 --save-path model_saved/pretrained_encoder_kfold_fold0.pth
```

**The encoder's pretraining set must be a subset of that split's TRAIN side.** Each protocol —
and under k-fold, each *fold* — needs its own encoder: fold 0's training side wholly contains
every other fold's validation set, so one shared encoder would leak 100% of their validation
beats. `dual_protocol_training.py` refuses an encoder whose provenance sidecar does not match
the run's split, fold, split-seed and fold count.

## Caveats to carry with any number from this repo

- **Runs are not bit-reproducible.** The cuDNN determinism flags are set, but
  `torch.use_deterministic_algorithms` is not, and 1 of 4 identical seed-42 runs diverged. Say
  "N runs at N seeds", never "bit-identical". Paired comparisons are unaffected — nondeterminism
  adds noise, not bias.
- **Report 3-class (N/S/V) as the cross-dataset headline.** INCART contains 6 Q beats in total,
  so F1(Q) ≈ 0 by construction and 5-class macro-F1 is exactly 0.8 × the 4-class figure. The
  5-class cross-dataset number understates the model by a fixed factor.
- **Use record-level (cluster) bootstrap CIs, never beat-level.** 175,811 INCART beats come from
  75 patients; beat-level intervals are 7–13× too narrow.
- **Per-class F for fusion is a documented limitation, not a result.** It needs n ≈ 95 seeds to
  be measurable at this variance.
- **Fold-wise standard deviations measure partition variance only** (the training seed is fixed
  at 42) and must not be quoted beside the seed-wise ones.
