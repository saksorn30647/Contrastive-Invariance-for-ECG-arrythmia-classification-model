import numpy as np
import torch
import random
from torch.utils.data import Dataset

from utils.cwt import compute_cwt

LABEL_MAP = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4
}


class ECGTwoViewTransform:
    """
    Applies two independent random augmentations to the same signal
    and returns both views. Used for contrastive learning.

    Args:
        transform: a callable that takes a torch.Tensor and returns
                   an augmented torch.Tensor. If None, returns the
                   signal unchanged for both views (useful for testing).
    """

    def __init__(self, transform=None):
        self.transform = transform
        

    def __call__(self, signal):

        if self.transform is None:
            return signal.clone(), signal.clone()

        return (
            self.transform(signal.clone()),
            self.transform(signal.clone())
    )

class ECGDataset(Dataset):
    def __init__(
        self,
        x_path,
        y_path,
        shape=None,
        memmap=False,
        two_view_transform=None,
            # pass ECGTwoViewTransform instance here
    ):                              # if None → standard single-view mode

        if memmap:
            self.X = np.memmap(
                x_path,
                dtype=np.float16,
                mode='r',
                shape=shape
            )
        else:
            self.X = np.load(x_path, mmap_mode='r')

        raw_y = np.load(y_path)

        # Support both string labels ("N","S",...) and pre-encoded integers
        if raw_y.dtype.kind in ('U', 'S', 'O'):   # string array
            self.y = np.array([LABEL_MAP[label] for label in raw_y])
        else:                                       # already numeric
            self.y = raw_y.astype(np.int64)

        self.two_view_transform = two_view_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # Load and convert to float32 tensor with channel dim
        signal = torch.from_numpy(
            self.X[idx].copy()
        ).float().unsqueeze(0)       # (1, L) for 1D or (1, H, W) for 2D

        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.two_view_transform is not None:

            view_1, view_2 = self.two_view_transform(signal)
            return (view_1, view_2), label
        else:
            # Standard mode — return single signal as before
            return signal, label
        
import torch
import random


class RandomGaussianNoise:
    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            noise = torch.randn_like(x) * self.std
            x = x + noise
        return x


class RandomAmplitudeScale:
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            x = x * scale
        return x


class RandomTimeShift:
    def __init__(self, max_shift=20, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            shift = random.randint(-self.max_shift, self.max_shift)
            x = torch.linear_shift(x, shifts=shift, dims=-1)
        return x


import torch.nn.functional as F
class RandomLinearTimeShift:
    def __init__(self, max_shift=10, p=0.5, padding_mode='replicate'):
        """
        Args:
            max_shift (int): Maximum number of samples to shift in either direction.
            p (float): Probability of applying the transformation.
            padding_mode (str): PyTorch padding mode. 'replicate' is highly 
                                recommended for ECG to prevent edge artifacts.
        """
        self.max_shift = max_shift
        self.p = p
        self.padding_mode = padding_mode

    def __call__(self, x):
        """
        Args:
            x (Tensor): Input ECG signal. Assumed shape is (..., T), 
                        supporting (Length,), (1, Length), or (Batch, 1, Length).
        """
        if random.random() >= self.p:
            return x
        
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        
        T = x.shape[-1] # Target original temporal length
        
        if shift > 0:
            # 1. SHIFT RIGHT (Delay in time)
            # Pad on the left by 'shift' samples, pad 0 on the right.
            # PyTorch F.pad expects tuples as (left, right) for the last dimension.
            x_padded = F.pad(x, (shift, 0), mode=self.padding_mode)
            # Retain the first T elements (the rightmost shifted-out samples drop off)
            return x_padded[..., :T]
            
        else:
            # 2. SHIFT LEFT (Advance in time)
            abs_shift = abs(shift)
            # Pad on the right by 'abs_shift' samples, pad 0 on the left.
            x_padded = F.pad(x, (0, abs_shift), mode=self.padding_mode)
            # Discard the first 'abs_shift' elements to slide the signal left
            return x_padded[..., abs_shift:]
        
class RandomBaselineWander:
    def __init__(self, amplitude=0.05, p=0.3):
        self.amplitude = amplitude
        self.p = p

    def __call__(self, x):

        if random.random() > self.p:
            return x

        length = x.shape[-1]

        t = torch.linspace(0, 2 * torch.pi, length)

        wander = self.amplitude * torch.sin(t)

        return x + wander.unsqueeze(0)


    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x