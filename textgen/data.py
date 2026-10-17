"""Corpus loading, tokenisation, vocabulary and window datasets."""

import re
from collections import Counter

import torch
from torch.utils.data import TensorDataset

PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")


def load_book(path):
    """Read a Project Gutenberg text and drop the licence header/footer."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        text = text[text.find("\n", start) + 1:]
        end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text


def tokenize(text):
    text = text.lower().replace("’", "'").replace("‘", "'")
    return TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, tokens, min_count=3):
        counts = Counter(tokens)
        kept = sorted((w for w, c in counts.items() if c >= min_count),
                      key=lambda w: (-counts[w], w))
        self.itos = [PAD, UNK] + kept
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = counts

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def split_chronologically(tokens, train=0.8, val=0.1):
    """Contiguous split so no text window appears in two splits."""
    n = len(tokens)
    a, b = int(n * train), int(n * (train + val))
    return tokens[:a], tokens[a:b], tokens[b:]


def make_windows(ids, context):
    """Every `context`-length window paired with the word that follows it."""
    ids = torch.tensor(ids, dtype=torch.long)
    x = ids.unfold(0, context, 1)[:-1]
    y = ids[context:]
    return TensorDataset(x, y)


def detokenize(tokens):
    out = " ".join(tokens)
    return re.sub(r" ([.,!?;:])", r"\1", out)
