"""
model.py
---------
Bi-directional LSTM for protein secondary structure prediction (Q3 labels).
Model features:
- Embedding layer for amino acid sequences
- Optional integration of additional per-residue features
- Bidirectional LSTM layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

PAD_IDX = 0
NUM_CLASSES = 3  # Q3: H, E, C

class BiLSTMProtein(nn.Module):
    def __init__(self, vocab_size: int = 21,  # 20 AA + PAD
                 embed_dim: int = 32,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.3,
                 use_features: bool = False,
                 feat_dim: Optional[int] = None):
        super().__init__()
        self.use_features = use_features
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        input_dim = embed_dim
        if use_features:
            if feat_dim is None:
                raise ValueError("feat_dim must be provided if use_features=True")
            input_dim += feat_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), NUM_CLASSES)

    def forward(self, seqs, lengths, features=None):
        """
        seqs: (batch_size, max_len)
        lengths: (batch_size,)
        features: optional (batch_size, max_len, feat_dim)
        """
        emb = self.embedding(seqs)  # (B, L, embed_dim)
        if self.use_features:
            if features is None:
                raise ValueError("Features are required but not provided")
            # Concatenate residue embeddings with any auxiliary descriptors we engineered.
            x = torch.cat([emb, features], dim=-1)
        else:
            x = emb

        # pack sequences for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits  # (B, L, NUM_CLASSES)


if __name__ == '__main__':
    # quick sanity check
    B, L = 4, 10
    seqs = torch.randint(1, 21, (B, L))
    lengths = torch.tensor([10, 9, 8, 7])
    model = BiLSTMProtein()
    logits = model(seqs, lengths)
    print("Logits shape:", logits.shape)
