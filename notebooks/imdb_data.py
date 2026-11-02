"""Dataset, collate function and length-bucketing batch sampler for the IMDB notebooks.

These live in a module instead of the notebook so DataLoader worker processes can import them:
on macOS, workers are started with "spawn", which cannot pickle classes defined inside a notebook.
"""
import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

PAD_ID = 0


class ReviewDataset(Dataset):
    def __init__(self, token_ids, labels):
        self.token_ids = [torch.tensor(ids) for ids in token_ids]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.token_ids[index], self.labels[index]


def collate_batch(batch):
    token_ids, labels = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in token_ids])
    padded = pad_sequence(token_ids, batch_first=True, padding_value=PAD_ID)
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)


class BucketBatchSampler(Sampler):
    """Puts reviews of similar length in the same batch, so little compute is spent on padding.

    Shuffled mode: shuffle all reviews, cut them into pools of `pool_batches` batches, sort each pool
    by length, slice it into batches, then shuffle the batches. Unshuffled mode: sort everything by length.
    """

    def __init__(self, lengths, batch_size, shuffle, drop_last, pool_batches=50, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pool_size = batch_size * pool_batches
        self.rng = random.Random(seed)

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(indices)
        pool_size = self.pool_size if self.shuffle else len(indices)
        batches = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start:start + pool_size], key=self.lengths.__getitem__)
            batches += [pool[i:i + self.batch_size] for i in range(0, len(pool), self.batch_size)]
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)
