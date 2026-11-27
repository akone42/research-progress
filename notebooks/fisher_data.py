"""Data preparation for next-turn speech-feature prediction on the Fisher corpus.

The data file is licensed (Fisher) and lives in notebooks/data/fisher/, which is git-ignored.
The notebook explains every choice made here; this module only holds the code so DataLoader
workers can import it.
"""
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ["pitch_mean_log", "pitch_std_log", "intensity_mean", "jitter", "shimmer", "nhr", "rate_syl_sec"]
N_FEAT = len(FEATURES)
PAD_ID, UNK_ID = 0, 1


def load_turns(path):
    df = pd.read_csv(path, index_col=0)
    df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
    # A speaking rate of 0 only occurs on turns with no recognized words: the rate is undefined, not zero.
    df.loc[df["rate_syl_sec"] == 0, "rate_syl_sec"] = np.nan
    return df


def split_conversations(conv_ids, seed=0, frac_train=0.8, frac_val=0.1):
    ids = np.array(sorted(set(conv_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(frac_train * len(ids))
    n_val = int(frac_val * len(ids))
    return set(ids[:n_train]), set(ids[n_train:n_train + n_val]), set(ids[n_train + n_val:])


def build_vocab(texts, min_count):
    counts = Counter(word for text in texts for word in text.split())
    words = sorted(w for w, c in counts.items() if c >= min_count)
    return {w: i + 2 for i, w in enumerate(words)}  # 0 = padding, 1 = unknown word


class Conversation:
    """One conversation, already shifted so that input row t is used to predict target row t.

    inputs:   (T, 7 features + 7 missing flags + 2 speaker one-hot); row 0 is the added start turn
    targets:  (T, 7) standardized features of turns 1..T
    observed: (T, 7) True where the target value was actually measured
    tokens:   word ids of the input turns, concatenated; turn_lengths says how many belong to each row
    turn_type: (T,) for each target turn: 0 = first turn, 1 = same speaker continues, 2 = speaker change
    """

    def __init__(self, inputs, targets, observed, tokens, turn_lengths, turn_type):
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)
        self.observed = torch.from_numpy(observed)
        self.tokens = torch.from_numpy(tokens)
        self.turn_lengths = torch.from_numpy(turn_lengths)
        self.turn_type = torch.from_numpy(turn_type)

    def __len__(self):
        return len(self.targets)


def build_conversations(df, conv_ids, mean, std, vocab):
    part = df[df["conv_id"].isin(conv_ids)]
    raw = part[FEATURES].to_numpy(np.float32)
    missing = np.isnan(raw)
    standardized = np.where(missing, 0.0, (raw - mean) / std).astype(np.float32)
    side = np.stack([(part["side"] == "A").to_numpy(), (part["side"] == "B").to_numpy()], axis=1).astype(np.float32)
    rows = np.concatenate([standardized, missing.astype(np.float32), side], axis=1)
    # Start turn (turn_id = 0): features at 0, every feature flagged as "not measured", no speaker.
    start_row = np.concatenate([np.zeros(N_FEAT), np.ones(N_FEAT), np.zeros(2)]).astype(np.float32)

    token_lists = [np.array([vocab.get(w, UNK_ID) for w in text.split()], dtype=np.int64)
                   for text in part["text"].fillna("")]

    sides = part["side"].to_numpy()
    conversations = []
    bounds = np.flatnonzero(np.diff(part["conv_id"].to_numpy())) + 1
    for start, end in zip(np.r_[0, bounds], np.r_[bounds, len(part)]):
        inputs = np.vstack([start_row, rows[start:end - 1]])
        turn_tokens = [np.zeros(0, dtype=np.int64)] + token_lists[start:end - 1]
        conversations.append(Conversation(
            inputs=inputs,
            targets=standardized[start:end],
            observed=~missing[start:end],
            tokens=np.concatenate(turn_tokens),
            turn_lengths=np.array([len(t) for t in turn_tokens], dtype=np.int64),
            turn_type=np.r_[0, np.where(sides[start + 1:end] == sides[start:end - 1], 1, 2)].astype(np.int64),
        ))
    return conversations


class ConversationDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        return self.conversations[index]


def collate_conversations(batch):
    """Pads conversations to the longest one in the batch.

    Words are not padded per turn: all word ids of the batch go in one flat tensor, with
    `token_turn` giving the (flattened batch x time) position of the turn each word belongs to.
    """
    lengths = torch.tensor([len(c) for c in batch])
    max_len = int(lengths.max())
    size = len(batch)
    inputs = torch.zeros(size, max_len, batch[0].inputs.shape[1])
    targets = torch.zeros(size, max_len, N_FEAT)
    observed = torch.zeros(size, max_len, N_FEAT, dtype=torch.bool)
    turn_type = torch.full((size, max_len), -1, dtype=torch.long)  # -1 = padding
    tokens, token_turn = [], []
    for i, c in enumerate(batch):
        n = len(c)
        inputs[i, :n] = c.inputs
        targets[i, :n] = c.targets
        observed[i, :n] = c.observed
        turn_type[i, :n] = c.turn_type
        tokens.append(c.tokens)
        token_turn.append(i * max_len + torch.repeat_interleave(torch.arange(n), c.turn_lengths))
    return {
        "inputs": inputs,
        "targets": targets,
        "observed": observed,
        "lengths": lengths,
        "turn_type": turn_type,
        "tokens": torch.cat(tokens),
        "token_turn": torch.cat(token_turn),
    }
