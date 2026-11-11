"""MIT-BIH / INCART scalogram dataset, and the AAMI label mapping.

READ THIS BEFORE LOADING LABELS DIRECTLY.

`data/y_cwt.npy` and `data/y_incart_cwt.npy` store AAMI labels as single
CHARACTERS ('N','S','V','F','Q'), not integers. `ECGDataset` maps them through
`LABEL_MAP` below, so anything that goes through the dataset class is fine.
Anything that calls `np.load()` on those files itself is not, and it fails
SILENTLY in two different ways:

  1. Comparing `y == 3` against a `<U1` array matches NOTHING and reports zero
     beats for every class -- no error, just all-zero counts.
  2. Worse, `utils.splits.build_split()` stratifies by iterating `np.unique(y)`.
     On characters that order is F,N,Q,S,V; on mapped integers it is N,S,V,F,Q.
     The per-class RNG draws are therefore consumed in a different order and
     the two calls return train sets sharing only ~86% of their indices --
     same size, same class proportions, no error, a DIFFERENT partition.
     Used for contrastive pretraining this silently leaks ~12k validation
     beats into the encoder's pretraining set.

So: always map through `LABEL_MAP` before calling `build_split()`, or load via
`ECGDataset` and use `.y`.

Also note `records.npy` exists twice with different contents: the project root
copy is MIT-BIH (48 records / 109,451 beats) and `data/records.npy` is INCART
(75 records / 175,811 beats). Assert `len(records) == len(y)` after loading --
if that ever fails, record ids are misaligned with beats and every per-record
result is wrong.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {
    "N":0,
    "S":1,
    "V":2,
    "F":3,
    "Q":4
}

class ECGDataset(Dataset):

    def __init__(
        self,
        x_path,
        y_path,
        shape=None,
        memmap=False
    ):

        if memmap:

            self.X = np.memmap(
                x_path,
                dtype=np.float16,
                mode='r',
                shape=shape
            )

        else:

            self.X = np.load(
                x_path,
                mmap_mode='r'
            )

        self.y = np.load(y_path)

        self.y = np.array([
            LABEL_MAP[label]
            for label in self.y
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        signal = self.X[idx]

        signal = torch.from_numpy(signal.copy()).float().unsqueeze(0)

        label = torch.tensor(self.y[idx], dtype=torch.long)

        return signal, label

    