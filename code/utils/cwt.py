import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_cwt(signal):
    scales = np.arange(1, 65)

    coefficients, _ = pywt.cwt(
        signal,
        scales,
        'morl'
    )

    # Log-energy scaling ONLY: Keeps structural contrast perfectly intact
    coefficients = np.log1p(np.abs(coefficients))

    return coefficients.astype(np.float16)
    


class GPUCWT(nn.Module):
    """GPU CWT that reproduces `compute_cwt` (i.e. `pywt.cwt`) to ~1e-4.

    Used for real-time CWT during contrastive pretraining, where augmenting the
    raw 1D signal and transforming on the fly is much cheaper than materialising
    augmented scalograms. It MUST match `compute_cwt`, because that is what
    built `data/X_cwt.npy` — the representation the encoder is later fine-tuned
    and evaluated on. If the two disagree, the encoder is pretrained on a
    different input distribution than it is used with.

    TWO BUGS FIXED 2026-08-31 (verified against pywt on real beats):

    1. **Missing kernel flip.** `F.conv1d` is CROSS-CORRELATION, not
       convolution — PyTorch does not flip the kernel. `pywt.cwt` differentiates
       the integrated wavelet, which makes the effective filter ANTI-symmetric,
       so the missing flip is not a no-op: it transforms with a time-mirrored
       wavelet.
    2. **`mode='replicate'` padding.** `pywt.cwt` convolves with implicit ZERO
       padding (`np.convolve` full, then centre-crop). Replicate padding extends
       a 256-sample beat by 512 samples of its edge value on each side, which
       contaminates every scale.

    Measured against `compute_cwt` on 32 real beats (mean |diff|, % of mean
    magnitude, Pearson r):

        replicate + no flip (OLD) : 0.10238   12.8%   r=0.97384
        zero pad   + no flip      : 0.08935   11.2%   r=0.97843
        zero pad   + FLIP  (NEW)  : 0.00014    0.0%   r=1.00000

    The discrepancy was worst in the QRS centre (mean |diff| 0.158, r=0.952),
    not at the boundaries.

    `legacy=True` restores the old buggy behaviour, ONLY so that
    `pretrained_encoder_1D_aug.pth` (pretrained under it) remains reproducible.
    Do not use it for new pretraining.
    """

    def __init__(self, scales=np.arange(1, 65), wavelet='morl', legacy=False):
        super().__init__()

        mock_len = 2048
        center = mock_len // 2
        impulse = np.zeros(mock_len)
        impulse[center] = 1.0

        # Impulse response captures pywt's scale normalisation and the
        # -sqrt(a)*diff step, so the filters need no extra scaling.
        filters, _ = pywt.cwt(impulse, scales, wavelet)
        half_width = 512
        filters_cropped = filters[:, center - half_width : center + half_width + 1]
        filters_tensor = torch.tensor(filters_cropped, dtype=torch.float32).unsqueeze(1)

        self.legacy = legacy
        if not legacy:
            # Flip so that F.conv1d's cross-correlation performs a true
            # convolution, matching np.convolve inside pywt.cwt.
            filters_tensor = torch.flip(filters_tensor, dims=[-1])

        self.register_buffer('filters', filters_tensor)
        self.padding = half_width

    def forward(self, x):
        # x input shape from dataloader: (B, 1, L)
        x = x.float()

        if self.legacy:
            x_padded = F.pad(x, (self.padding, self.padding), mode='replicate')
        else:
            x_padded = F.pad(x, (self.padding, self.padding))   # zeros, as pywt

        cwt_out = F.conv1d(x_padded, self.filters)  # Shape: (B, 64, L)

        # Log-energy scaling — identical to compute_cwt.
        cwt_out = torch.log1p(torch.abs(cwt_out))

        # No manual z-scoring; BatchNorm2d in the backbone handles normalisation.
        return cwt_out  # Shape: (B, 64, L)