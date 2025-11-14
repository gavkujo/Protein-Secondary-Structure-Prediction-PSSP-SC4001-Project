"""
model.py
--------
Hybrid CNN + Transformer model definition for protein secondary structure prediction.
Model features:
- Depthwise separable convolutional blocks for local context
- Optional integration of additional per-residue features
- Transformer encoder layers for global context
- Joint Q3 and Q8 prediction heads
"""

import torch
import torch.nn as nn
from typing import Optional

PAD_IDX = 0
NUM_Q3_CLASSES = 3  # Q3: H, E, C
NUM_Q8_CLASSES = 8  # Q8: H, E, C, T, S, G, B, I


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise separable conv block with residual connection for local context."""

    def __init__(self, dim: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise then operate along the channel dimension to keep the skip connection stable.
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return residual + x


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
        # Slice the cached table to match the current sequence length, then broadcast across the batch.
        length = int(x.size(1))
        positions = torch.arange(length, device=x.device)
        pe_buffer: torch.Tensor = getattr(self, 'pe_buffer')
        pos_slice = torch.index_select(pe_buffer, 0, positions).unsqueeze(0)
        return x + pos_slice


class ProteinHybridModel(nn.Module):
    """Hybrid CNN + Transformer encoder for joint Q3/Q8 prediction."""

    def __init__(self,
                 vocab_size: int = 21,
                 embed_dim: int = 64,
                 model_dim: int = 256,
                 conv_layers: int = 3,
                 conv_kernel: int = 9,
                 conv_dropout: float = 0.1,
                 dropout: float = 0.3,
                 use_features: bool = False,
                 feat_dim: Optional[int] = None,
                 attention_heads: int = 8,
                 transformer_layers: int = 4,
                 transformer_dropout: float = 0.2,
                 transformer_ff_dim: int = 512,
                 head_dropout: float = 0.15,
                 q8_head_dim: int = 512,
                 q8_head_layers: int = 2):
        super().__init__()
        self.use_features = use_features
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if use_features:
            if feat_dim is None:
                raise ValueError("feat_dim must be provided when use_features=True")
            self.feature_proj = nn.Linear(feat_dim, embed_dim)
        else:
            self.feature_proj = None

        input_dim = embed_dim + (embed_dim if use_features else 0)
        # Project concatenated embeddings into the shared model space expected by conv/transformer blocks.
        self.input_proj = nn.Linear(input_dim, model_dim)

        self.conv_blocks = nn.ModuleList([
            DepthwiseSeparableConvBlock(model_dim, kernel_size=conv_kernel, dropout=conv_dropout)
            for _ in range(conv_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.use_attention = transformer_layers > 0
        if self.use_attention:
            # TransformerEncoder with norm_first=True keeps the residual stack numerically stable.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=attention_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.positional_encoding = SinusoidalPositionalEncoding(model_dim)
        else:
            self.transformer = None
            self.positional_encoding = None

        self.final_norm = nn.LayerNorm(model_dim)

        if q8_head_layers < 1:
            raise ValueError("q8_head_layers must be >= 1")
        if q8_head_layers > 1 and q8_head_dim <= 0:
            raise ValueError("q8_head_dim must be positive when q8_head_layers > 1")

        self.q3_head = nn.Sequential(
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Linear(model_dim, NUM_Q3_CLASSES)
        )

        if q8_head_layers == 1:
            self.q8_head = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
                nn.Linear(model_dim, NUM_Q8_CLASSES)
            )
        else:
            # Build a small MLP so the Q8 head can model the richer eight-class boundary.
            q8_layers: list[nn.Module] = [nn.LayerNorm(model_dim)]
            in_dim = model_dim
            for _ in range(q8_head_layers - 1):
                q8_layers.append(nn.Linear(in_dim, q8_head_dim))
                q8_layers.append(nn.GELU())
                if head_dropout > 0:
                    q8_layers.append(nn.Dropout(head_dropout))
                in_dim = q8_head_dim
            q8_layers.append(nn.Linear(in_dim, NUM_Q8_CLASSES))
            self.q8_head = nn.Sequential(*q8_layers)

    def forward(self, seqs: torch.Tensor, lengths: torch.Tensor, features: Optional[torch.Tensor] = None):
        emb = self.embedding(seqs)
        if self.use_features:
            if features is None:
                raise ValueError("Features are required but not provided")
            if self.feature_proj is None:
                raise ValueError("Feature projection layer is missing")
            feat_proj = self.feature_proj(features)
            # Concatenate learned tokens with projected biochemical descriptors per residue.
            x = torch.cat([emb, feat_proj], dim=-1)
        else:
            x = emb

        x = self.input_proj(x)
        for block in self.conv_blocks:
            # Each convolutional block captures neighbourhood motifs before global attention kicks in.
            x = block(x)

        x = self.dropout(x)

        if self.use_attention and self.transformer is not None:
            # Convert lengths into a key padding mask so the transformer ignores padded residues.
            attn_mask = self._key_padding_mask(lengths, seqs.size(1))
            if self.positional_encoding is None:
                raise RuntimeError("Positional encoding module is missing")
            x = self.positional_encoding(x)
            x = self.transformer(x, src_key_padding_mask=attn_mask)

        x = self.final_norm(x)
        logits_q3 = self.q3_head(x)
        logits_q8 = self.q8_head(x)
        return {'q3': logits_q3, 'q8': logits_q8}

    @staticmethod
    def _key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)
        if lengths.device.type != 'cpu':
            lengths_cpu = lengths.cpu()
        else:
            lengths_cpu = lengths
        range_row = torch.arange(max_len)
        # True marks padded positions; the transformer will skip attention updates for them.
        mask = range_row.unsqueeze(0) >= lengths_cpu.unsqueeze(1)
        return mask.to(lengths.device)


if __name__ == '__main__':
    # quick sanity check
    B, L = 4, 10
    seqs = torch.randint(1, 21, (B, L))
    lengths = torch.tensor([10, 9, 8, 7])
    model_hybrid = ProteinHybridModel()
    out_hybrid = model_hybrid(seqs, lengths)
    print("Hybrid q3 shape:", out_hybrid['q3'].shape, "q8 shape:", out_hybrid['q8'].shape)
