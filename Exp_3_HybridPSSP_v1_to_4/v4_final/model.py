"""
model.py
--------
Hybrid Transformer + Dilated CNN model for protein secondary structure prediction (Q3 and Q8 labels).
Model features:
- Learned positional encodings for sequence residues
- Dilated convolutional blocks for local context capture
- Transformer encoder layers for global context modeling
- Joint Q3 and Q8 prediction heads
Inputs:
- features: FloatTensor of shape (batch_size, max_len, input_dim) containing per-residue features
- optional: mask tensor (batch_size, max_len) where True indicates valid residues
Outputs:
- Dictionary containing logits for Q3 and Q8 tasks keyed by 'q3' and 'q8'
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

NUM_Q3_CLASSES = 3
NUM_Q8_CLASSES = 8


class LearnedPositionalEncoding(nn.Module):
    """Learned positional offsets added to residues prior to attention."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(max_len, dim))
        # Initialise with a tiny normal distribution so positions start unique but stable.
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.weight.size(0):
            raise ValueError(f"Sequence length {length} exceeds positional encoding limit {self.weight.size(0)}")
        return x + self.weight[:length].unsqueeze(0)


class DilatedConvBlock(nn.Module):
    """Dilated 1D convolutional residual block operating on (B, L, D)."""

    def __init__(self, d_model: int, hidden_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        # Depth-wise conv reads along the sequence dimension; dilation expands receptive field.
        self.depthwise = nn.Conv1d(d_model, hidden_channels, kernel_size,
                                   padding=padding, dilation=dilation, bias=False)
        self.pointwise = nn.Conv1d(hidden_channels, d_model, kernel_size=1, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Transformer output arrives as (B, L, D); swap to (B, D, L) for convs then swap back.
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.depthwise(y)
        y = self.activation(y)
        y = self.pointwise(y)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        return residual + y


class HybridPSSP(nn.Module):
    """Hybrid Transformer + dilated CNN model for joint Q3/Q8 prediction."""

    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 ff_multiplier: float = 4.0,
                 dropout: float = 0.1,
                 cnn_channels: int = 128,
                 cnn_kernel_sizes: Sequence[int] = (3, 5, 7),
                 cnn_dilations: Sequence[int] = (1, 2, 4),
                 fuse_mode: str = 'sum',
                 head_dropout: float = 0.15,
                 max_len: int = 4096):
        super().__init__()

        if fuse_mode not in {'sum', 'concat'}:
            raise ValueError("fuse_mode must be 'sum' or 'concat'")
        if len(cnn_kernel_sizes) != len(cnn_dilations):
            raise ValueError("cnn_kernel_sizes and cnn_dilations must have the same length")

        # Feed-forward hidden size follows the common Transformer rule of 4x the model width (configurable).
        ff_dim = int(ff_multiplier * d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = LearnedPositionalEncoding(d_model, max_len=max_len)
        # Transformer layers model global residue interactions once features share the same channel width.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Parallel dilated convolutions widen the receptive field without sacrificing resolution.
        self.cnn_blocks = nn.ModuleList([
            DilatedConvBlock(d_model=d_model,
                             hidden_channels=cnn_channels,
                             kernel_size=kernel,
                             dilation=dilation,
                             dropout=dropout)
            for kernel, dilation in zip(cnn_kernel_sizes, cnn_dilations)
        ])

        self.fuse_mode = fuse_mode
        if self.fuse_mode == 'concat':
            # When concatenating branches, project back to d_model so downstream heads stay unchanged.
            self.fuse_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.fuse_proj = None

        self.final_norm = nn.LayerNorm(d_model)
        self.final_dropout = nn.Dropout(dropout)
        self.q3_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, NUM_Q3_CLASSES)
        )
        self.q8_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, NUM_Q8_CLASSES)
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, L, input_dim) stacked residue features.
            mask: optional (B, L) boolean tensor where True indicates valid residues.
        Returns:
            Dictionary containing logits for Q3 and Q8 tasks keyed by 'q3' and 'q8'.
        """
        key_padding_mask = None
        if mask is not None:
            # nn.Transformer expects False for valid tokens; invert boolean mask from the dataset.
            key_padding_mask = (~mask.bool())

        x = self.input_proj(features)
        x = self.positional_encoding(x)
        # Transformer encoder handles global residue interactions with attention.
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        y = x
        for block in self.cnn_blocks:
            # Convolutional branch refines local motifs before fusion.
            y = block(y)

        if self.fuse_mode == 'concat':
            fused = torch.cat([x, y], dim=-1)
            assert self.fuse_proj is not None
            fused = self.fuse_proj(fused)
        else:
            # Sum fusion keeps dimensionality low and mimics residual blending.
            fused = x + y

        fused = self.final_norm(fused)
        fused = self.final_dropout(fused)

        logits_q3 = self.q3_head(fused)
        logits_q8 = self.q8_head(fused)
    # Return both heads together so the training loop can decide how to weight them.
        return {'q3': logits_q3, 'q8': logits_q8}


if __name__ == '__main__':
    batch_size, seq_len, feat_dim = 2, 50, 128
    dummy_features = torch.randn(batch_size, seq_len, feat_dim)
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    model = HybridPSSP(input_dim=feat_dim)
    out = model(dummy_features, mask=dummy_mask)
    print(out['q3'].shape, out['q8'].shape)
