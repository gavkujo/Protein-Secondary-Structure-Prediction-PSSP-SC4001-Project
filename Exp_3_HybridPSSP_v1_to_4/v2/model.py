"""
model.py
--------
BiLSTM model definition for protein secondary structure prediction.
Model features:
- Embedding layer for amino acid sequences
- Optional integration of additional per-residue features
- Bidirectional LSTM layers
- Optional Transformer-based attention mechanism
"""

import torch
import torch.nn as nn
from typing import Optional

PAD_IDX = 0
NUM_CLASSES = 3  # Q3: H, E, C


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding injected before transformer layers."""

    def __init__(self, dim: int, max_len: int = 4000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_buffer', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Grab as many rows as the current sequence length, then broadcast across batch dimension.
        length = int(x.size(1))
        positions = torch.arange(length, device=x.device)
        pe_buffer: torch.Tensor = getattr(self, 'pe_buffer')
        pos_slice = torch.index_select(pe_buffer, 0, positions).unsqueeze(0)
        return x + pos_slice


class BiLSTMProtein(nn.Module):
    def __init__(self, vocab_size: int = 21,  # 20 AA + PAD
                 embed_dim: int = 32,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.3,
                 use_features: bool = False,
                 feat_dim: Optional[int] = None,
                 use_attention: bool = True,
                 attention_heads: int = 4,
                 transformer_layers: int = 2,
                 transformer_dropout: float = 0.2,
                 transformer_ff_dim: int = 256):
        super().__init__()
        self.use_features = use_features
        self.use_attention = use_attention and transformer_layers > 0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        input_dim = embed_dim
        if use_features:
            if feat_dim is None:
                raise ValueError("feat_dim must be provided if use_features=True")
            input_dim += feat_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        lstm_dim = hidden_dim * (2 if bidirectional else 1)

        if self.use_attention:
            # TransformerEncoder refines the biLSTM states with global context if enabled.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=lstm_dim,
                nhead=attention_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.positional_encoding = SinusoidalPositionalEncoding(lstm_dim)
            self.post_attn_norm = nn.LayerNorm(lstm_dim)
            self.attn_dropout = nn.Dropout(transformer_dropout)
        else:
            self.transformer = None

        self.classifier = nn.Linear(lstm_dim, NUM_CLASSES)

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
            # Concatenate learned embeddings with handcrafted residue descriptors.
            x = torch.cat([emb, features], dim=-1)
        else:
            x = emb

        # pack sequences for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seqs.size(1))
        out = self.dropout(out)

        if self.use_attention and self.transformer is not None:
            # transformer expects padding mask with True on padded positions
            attn_mask = self._key_padding_mask(lengths, seqs.size(1))
            out = self.positional_encoding(out)
            out = self.transformer(out, src_key_padding_mask=attn_mask)
            out = self.post_attn_norm(out)
            out = self.attn_dropout(out)

        logits = self.classifier(out)
        return logits  # (B, L, NUM_CLASSES)

    @staticmethod
    def _key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)
        if lengths.device.type != 'cpu':
            lengths_cpu = lengths.cpu()
        else:
            lengths_cpu = lengths
        range_row = torch.arange(max_len)
        # Mask padded indices so the transformer never attends to them.
        mask = range_row.unsqueeze(0) >= lengths_cpu.unsqueeze(1)
        return mask.to(lengths.device)


if __name__ == '__main__':
    # quick sanity check
    B, L = 4, 10
    seqs = torch.randint(1, 21, (B, L))
    lengths = torch.tensor([10, 9, 8, 7])
    model = BiLSTMProtein()
    logits = model(seqs, lengths)
    print("Logits shape:", logits.shape)
