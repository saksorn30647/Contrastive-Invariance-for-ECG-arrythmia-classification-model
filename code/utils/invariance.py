import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017), the classification-loss
    alternative to class-weighted cross-entropy.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Cross-entropy applies a FIXED per-class multiplier decided upfront from
    the class counts: every N sample is down-weighted by the same amount,
    whether the model finds it trivial or genuinely confusing. Focal loss
    instead down-weights per EXAMPLE by how confident the model already is:
    the (1 - p_t)^gamma term shrinks the loss on samples the model already
    gets right, so gradient stays concentrated on the ones it doesn't.

    For this project that difference matters because the majority class (N)
    is both huge and mostly easy. Under weighted CE, tens of thousands of
    already-solved N samples still contribute gradient every epoch. Under
    focal loss they contribute almost nothing once p_t is high, which is the
    mechanism we're hoping stops N/V from crowding out F.

    Args:
        alpha (Tensor or array-like or None): Per-class weights, same role as
            CrossEntropyLoss's `weight`. Can be set to None to test focal
            loss's own implicit rebalancing in isolation.
        gamma (float): Focusing parameter. gamma=0 reduces exactly to
            weighted cross-entropy; 2.0 is the standard default.
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()

        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        # Registered as a buffer so it follows the module across .to(device),
        # matching how CrossEntropyLoss handles its `weight` argument.
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # log_softmax is numerically safer than log(softmax(...)) — important
        # here because the focal term can drive per-sample loss very small,
        # and we don't want that swamped by float error.
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather the log-prob assigned to each sample's TRUE class -> log(p_t)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        at = None
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * loss

        if self.reduction == 'mean':
            # Match nn.CrossEntropyLoss's normalization exactly: when class
            # weights are used it computes a WEIGHTED mean, dividing by
            # sum(weights of the batch's targets) rather than by batch size.
            #
            # This is not a cosmetic detail. Dividing by N instead inflates
            # the loss (and therefore the gradient) by sum(at)/N — with the
            # alpha=0.5 weights here that is roughly a 20-25% shift. Swapping
            # CE -> focal would then silently change the effective learning
            # rate at the same time, so any difference in results could not
            # be attributed to the loss function alone.
            if at is not None:
                return loss.sum() / at.sum().clamp_min(1e-8)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class InvarianceLoss(nn.Module):
    def __init__(self, cls_weight=None, lambda_inv=0.5, inv_type='prediction',
                 cls_loss_type='ce', focal_gamma=2.0):
        """
        Args:
            cls_weight (Tensor or array-like): Smoothed class weights for minority classes.
            lambda_inv (float): Regularization weight for the invariance constraint.
            inv_type (str): 'prediction' (KL Div) or 'feature' (MSE on embeddings).
            cls_loss_type (str): 'ce' (class-weighted cross-entropy) or 'focal'.
            focal_gamma (float): Focusing parameter, only used when cls_loss_type='focal'.
        """
        super().__init__()

        # Defensive conversion — CrossEntropyLoss's `weight` arg is registered
        # as a buffer and requires a Tensor (or None). A numpy array here
        # raises at construction time, before any training starts.
        if cls_weight is not None and not torch.is_tensor(cls_weight):
            cls_weight = torch.as_tensor(cls_weight, dtype=torch.float32)

        if cls_loss_type == 'focal':
            self.cls_criterion = FocalLoss(alpha=cls_weight, gamma=focal_gamma)
        elif cls_loss_type == 'ce':
            self.cls_criterion = nn.CrossEntropyLoss(weight=cls_weight)
        else:
            raise ValueError("cls_loss_type must be 'ce' or 'focal'")

        self.cls_loss_type = cls_loss_type
        self.lambda_inv = lambda_inv
        self.inv_type = inv_type

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, logits, logits_perturbed, targets, features=None, features_perturbed=None):
        # 1. Supervised classification loss on the clean sample.
        loss_cls = self.cls_criterion(logits, targets)

        # When lambda_inv is 0 (the pure-CE ablation), skip the invariance
        # term entirely rather than computing it and multiplying by zero —
        # this lets the caller skip building the perturbed branch at all,
        # so this path is a genuine "no invariance mechanism" baseline, not
        # just a zeroed-out term with wasted compute behind it.
        if self.lambda_inv == 0:
            loss_inv = torch.zeros((), device=logits.device)
            return loss_cls, loss_cls, loss_inv

        # 2. Invariance regularization — the CLEAN branch is detached and
        #    treated as a fixed anchor (it already has direct CE supervision).
        #    Only the PERTURBED branch is pulled toward it. Without the
        #    detach, gradient flows both ways and the invariance term can
        #    just as easily drag the clean, supervised prediction toward a
        #    noisy, unsupervised perturbed one.
        if self.inv_type == 'prediction':
            log_p_perturbed = F.log_softmax(logits_perturbed, dim=-1)
            p_clean = F.softmax(logits.detach(), dim=-1)
            loss_inv = self.kl_loss(log_p_perturbed, p_clean)
        elif self.inv_type == 'feature':
            loss_inv = self.mse_loss(features_perturbed, features.detach())
        else:
            raise ValueError("inv_type must be 'prediction' or 'feature'")

        # 3. Joint optimization.
        total_loss = loss_cls + (self.lambda_inv * loss_inv)
        return total_loss, loss_cls, loss_inv


@torch.no_grad()
def apply_cwt_perturbations(x, device, mask_prob=0.3, mask_width_frac=0.03):
    """
    Applies domain-realistic 2D perturbations on CWT scalogram batches.
    Input shape: [Batch, Channels, Height, Width]

    Wrapped in no_grad — this is data augmentation, not a learned transform,
    so there's no need to track gradients through the perturbation itself
    (only the model's forward pass on the result needs gradients).

    mask_prob / mask_width_frac default to 0.3 / 0.03 (down from an
    implicit 0.5 / 0.08). Visual inspection of real F-class samples showed
    the original settings routinely landing dead-center on the scalogram's
    main energy cone — 300-780 of the ~1280 masked pixels wiped from near
    peak value to zero, on roughly half of all forward passes. For a class
    with only ~374 training examples and very little morphological
    redundancy, that was likely deleting the exact region the model needed
    to tell F apart from N/V, not adding realistic noise.
    """
    x_perturbed = x.clone()
    batch_size, _, height, width = x.shape

    # 1. Amplitude Gain Scaling (Random scaling factor per batch element)
    scales = (0.8 + 0.4 * torch.rand(batch_size, 1, 1, 1)).to(device)  # Scale range: [0.8, 1.2]
    x_perturbed = x_perturbed * scales

    # 2. Additive High-Frequency Noise (Powerline / Electrode interface simulation)
    noise = (0.05 * torch.randn_like(x_perturbed)).to(device)
    x_perturbed = x_perturbed + noise

    # 3. Time Masking (Simulates brief temporal dropouts) — narrower and
    # rarer than before; see docstring.
    mask_width = max(1, int(width * mask_width_frac))
    for i in range(batch_size):
        if torch.rand(1).item() < mask_prob:
            start_idx = torch.randint(0, width - mask_width, (1,)).item()
            x_perturbed[i, :, :, start_idx: start_idx + mask_width] = 0.0

    return x_perturbed