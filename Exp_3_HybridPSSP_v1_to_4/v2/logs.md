# Revision Log

## Overview
- Compared with the v1 baseline (pure BiLSTM + Adam), v2 layers optional transformer attention, richer optimisation utilities, and better sequence pre-processing while retaining archive-only compatibility.
- Maintains drop-in CLI parity so existing v1 training scripts continue to run, but exposes new knobs for attention depth, feature usage, and scheduler behaviour.

## Data Handling
- `ProteinDataset` can cap sequence length via `max_len`, trimming labels and features together; v1 returned full sequences even when exceeding GPU memory budgets.
- Optional per-residue `.npy` features are converted to float tensors after trimming, keeping LSTM inputs aligned; v1 passed raw arrays with mismatched lengths if the cache was longer than the sequence.
- Archived CSV parsing is unchanged, ensuring the new sampler still works when no auxiliary feature directory is provided.
- Collate logic continues to pad sequences, labels, and masks, but now preserves trimmed feature tensors so the downstream model never sees inconsistent time steps.

## Model Architecture
- Core BiLSTM remains, but v2 can append a sinusoidal positional encoding and configurable Transformer encoder (layers/heads/FF width) on top of the recurrent states, enabling long-range mixing that did not exist in v1.
- Attention stack is guarded by a key-padding mask derived from true lengths so padded tokens stay silent; v1 relied solely on packed LSTM outputs.
- Constructor now surfaces toggles for `use_attention`, `attention_heads`, `transformer_layers`, `transformer_ff_dim`, and `transformer_dropout`, letting experiments scale capacity without modifying source.
- Post-attention LayerNorm + dropout stabilise the hybrid outputs before classification, reducing overfitting observed in long runs of the plain BiLSTM.

## Training Pipeline
- Optimiser upgraded from Adam to AdamW with weight decay, paired with optional OneCycleLR scheduling and gradient clipping; v1 used a fixed-step Adam without regularisation controls.
- Mixed-precision (AMP) training is enabled by default on CUDA/MPS via `GradScaler`, trimming wall-clock time relative to v1’s full-precision loop.
- CLI introduces flags for label smoothing, warmup percentage, max token length, feature dimensions, and scheduler hyperparameters while keeping v1 defaults for continuity.
- DataLoaders gained `num_workers`, pinned memory, and persistent workers options, plus non-blocking tensor transfers to saturate accelerators; v1 always executed single-threaded host loads.
- `_infer_feature_dim` inspects cached `.npy` files when features are enabled, preventing silent shape mismatches that previously caused runtime errors.

## Evaluation
- `test.py` mirrors the richer CLI/feature flags from training, so evaluation reuses attention settings, feature directories, and `max_len` behaviour introduced in v2.
- Device resolution now recognises CUDA and Apple MPS, matching the training pipeline’s fallback logic and avoiding manual edits that were necessary in v1.
- Metric logging still reports macro-F1 but now surfaces the active configuration, aiding reproducibility when toggling attention or external features.



# RESULTS

[08:43:45] Test macro-F1: 0.6997                                          
[08:43:45] Classification Report:
              precision    recall  f1-score   support

           0     0.7356    0.7668    0.7509     76692
           1     0.6607    0.6143    0.6366     47399
           2     0.7112    0.7119    0.7115     91346

    accuracy                         0.7099    215437
   macro avg     0.7025    0.6976    0.6997    215437
weighted avg     0.7088    0.7099    0.7091    215437

[08:43:46] Confusion Matrix:
[[58806  4438 13448]
 [ 5329 29115 12955]
 [15808 10513 65025]]
