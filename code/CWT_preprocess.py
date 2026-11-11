"""STEP 2a of 4 -- CWT scalograms for MIT-BIH.

CONSUMES : data/X.npy, data/y.npy        (from beat_extrac_label.py)
PRODUCES : data/X_cwt.npy   (109451, 64, 256) float16  ~3.59 GB
           data/y_cwt.npy   labels copied through unchanged (still CHARACTERS)

Morlet ('morl') CWT at scales 1..64, then log1p(|coefficients|), stored float16.
Runtime is tens of minutes and the output does not fit in RAM comfortably --
downstream code memory-maps it.
"""

import numpy as np
from tqdm import tqdm
from utils.cwt import compute_cwt

X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("Original size (MB):")
print(X.nbytes / 1024 / 1024)

X_cwt = []

for beat in tqdm(X):

    cwt_img = compute_cwt(beat)

    X_cwt.append(cwt_img)

X_cwt = np.array(X_cwt, dtype=np.float16)

print("CWT shape:")
print(X_cwt.shape)

print("CWT size (MB):")
print(X_cwt.nbytes / 1024 / 1024)

print("Saving...")

np.save("data/X_cwt.npy", X_cwt)
np.save("data/y_cwt.npy", y)

print("Done.")