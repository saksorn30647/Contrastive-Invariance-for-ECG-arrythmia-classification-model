import numpy as np
from torch.utils.data import Sampler


class MinorityGuaranteedBatchSampler(Sampler):
    """
    Batch sampler that floors the minimum per-batch exposure of specified
    minority classes, without reweighting the relative frequency of the
    remaining classes.

    For each batch: draws `min_count` samples (with replacement — the pools
    for classes like F are small) from each class in `min_per_class`, then
    fills the remainder via uniform random sampling from the full dataset
    (not excluding the already-guaranteed classes, not touching the natural
    relative frequency of the other classes).

    Deliberately different from WeightedRandomSampler, which reshapes the
    *entire* batch distribution and — per this project's history — distorted
    the shared encoder's features via over-representation of duplicated
    minority samples (see CLAUDE.md, "Ruled out: aggressive class-balanced
    sampling"). This only raises the floor on the rarest classes; it leaves
    everything else alone.

    Note on epoch semantics: because the "filler" portion of every batch is
    drawn independently per batch rather than from a single per-epoch
    permutation, this does not guarantee each majority-class sample is seen
    exactly once per epoch the way standard shuffled sampling does. That's
    an intentional trade-off of this design (see CLAUDE.md's agreed spec),
    not a bug.
    """

    def __init__(self, labels, batch_size, min_per_class, seed=None, num_batches=None):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.min_per_class = dict(min_per_class)
        self.rng = np.random.default_rng(seed)

        total_floor = sum(self.min_per_class.values())
        if total_floor > batch_size:
            raise ValueError(
                f"sum(min_per_class.values())={total_floor} exceeds "
                f"batch_size={batch_size}"
            )

        self.class_indices = {}
        for c in self.min_per_class:
            idx = np.where(self.labels == c)[0]
            if len(idx) == 0:
                raise ValueError(f"class {c} has no samples in this dataset")
            self.class_indices[c] = idx

        self.all_indices = np.arange(len(self.labels))
        self.num_batches = num_batches or int(np.ceil(len(self.labels) / batch_size))

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c, min_count in self.min_per_class.items():
                pool = self.class_indices[c]
                draw = self.rng.choice(pool, size=min_count, replace=True)
                batch.extend(draw.tolist())

            remaining = self.batch_size - len(batch)
            fill = self.rng.choice(self.all_indices, size=remaining, replace=False)
            batch.extend(fill.tolist())

            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
