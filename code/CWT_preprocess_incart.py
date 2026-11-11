"""STEP 2b of 4 -- CWT scalograms for INCART.

CONSUMES : data/X_incart.npy, data/y_incart.npy  (from beat_extrac_label_INCART.py)
PRODUCES : data/X_incart_cwt.dat  (175811, 64, 256) float16 MEMMAP  ~5.76 GB
           data/y_incart_cwt.npy  labels copied through unchanged (CHARACTERS)

Written as a raw .dat memmap rather than .npy because of its size. Readers must
supply the shape explicitly, e.g.
    ECGDataset("data/X_incart_cwt.dat", "data/y_incart_cwt.npy",
               shape=(175811, 64, 256), memmap=True)

Same transform as the MIT-BIH stage: Morlet, scales 1..64, log1p(|coef|).
"""

import numpy as np
from tqdm import tqdm
from utils.cwt import compute_cwt

X = np.load("data/X_incart.npy")
y = np.load("data/y_incart.npy")

X_cwt = np.memmap(
    "data/X_incart_cwt.dat",
    dtype=np.float16,
    mode="w+",
    shape=(len(X), 64, 256)
)

for idx, beat in enumerate(tqdm(X)):

    X_cwt[idx] = compute_cwt(beat)

np.save("data/y_incart_cwt.npy", y)

print("Done.")