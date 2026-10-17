"""Recurrent next-word predictors that share one embedding/output design."""

import torch
from torch import nn

CELLS = {"rnn": nn.RNN, "birnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class NextWordModel(nn.Module):
    def __init__(self, kind, vocab_size, emb_dim=128, hidden=256, dropout=0.3):
        super().__init__()
        self.kind = kind
        self.bidirectional = kind == "birnn"
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = CELLS[kind](emb_dim, hidden, batch_first=True,
                               bidirectional=self.bidirectional)
        feat = hidden * (2 if self.bidirectional else 1)
        self.norm = nn.LayerNorm(feat)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(self.drop(self.embed(x)))
        if self.bidirectional:
            # Mean over time of both directions: the backward pass only ever
            # reads the context window itself, never words after it.
            summary = out.mean(dim=1)
        else:
            summary = out[:, -1]
        return self.head(self.drop(self.norm(summary)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
