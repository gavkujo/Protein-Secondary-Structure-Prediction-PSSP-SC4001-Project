# Revision Log

## Overview
- v3 graduates from the v2 BiLSTM-with-optional-attention stack to a dual-head hybrid built for simultaneous Q3/Q8 prediction while keeping the data interface compatible with the archive CSVs.
- Q8 supervision is now first-class across dataprep, training, and evaluation, letting us learn 8-state structure directly instead of mapping back from Q3 predictions.

## Data & Features
- `dataprep.ProteinDataset` still trims sequences and optional feature tensors, but additionally constructs Q8 label indices (`SS8_MAP`) after enforcing that any truncation keeps Q3/Q8 lengths aligned—v2 silently dropped Q8.
- `collate_fn` pads Q8 labels in lockstep with Q3 whenever they exist, so downstream batches carry both targets without extra bookkeeping.

## Model Architecture
- The legacy `BiLSTMProtein` remains for backwards compatibility, yet it now emits both Q3 and Q8 logits (dict return) and can be disabled via CLI; v2 only predicted Q3.
- Added `DepthwiseSeparableConvBlock` components and the new `ProteinHybridModel`: a feature projection + CNN + Transformer encoder that fuses local receptive fields with global attention, surpassing the LSTM-only approach available in v2.
- Transformer positional encoding moved into a dedicated sinusoidal module within the hybrid, while the Q8 prediction head gained optional deep MLP layers (`q8_head_layers`, `q8_head_dim`) and configurable dropout to address minority-state imbalance.

## Training Pipeline
- Training can now select between BiLSTM and hybrid paths via `--model`. The loop computes separate losses for Q3/Q8, mixes them with `--q8_weight`, and optionally re-weights Q8 classes using inverse or effective-number heuristics (`--q8_balance`, `--q8_balance_beta`).
- Mixed-precision, gradient clipping, and OneCycleLR from v2 remain, but metric logging now surfaces individual task losses/F1 scores and stores the best checkpoint keyed to joint performance.
- When using the hybrid, new CLI flags expose CNN kernel size/count, transformer depth, head dropout, and Q8 head depth so ablations do not require source changes.

## Evaluation
- `test.py` mirrors the expanded training CLI, instantiating the same backbone and head topology to guarantee parity between validation and test.
- Reports now include macro-F1, classification reports, and confusion matrices for both Q3 and Q8, with graceful fallbacks if Q8 labels are unavailable—v2 only reported Q3 metrics by default.
- Device resolution, feature loading, and max-length truncation logic match the training script, preventing the evaluation-time divergences we saw in v2 experiments.


# RESULTS

[09:25:51] Evaluating Q3 and Q8 heads
[09:25:53] Test macro-F1 (Q3): 0.6987                                             
[09:25:53] Q3 Classification Report:
              precision    recall  f1-score   support

           0     0.7519    0.7477    0.7498     76692
           1     0.6543    0.6004    0.6262     47399
           2     0.7036    0.7370    0.7200     91346

    accuracy                         0.7108    215437
   macro avg     0.7033    0.6951    0.6987    215437
weighted avg     0.7100    0.7108    0.7100    215437

[09:25:53] Q3 Confusion Matrix:
[[57343  4780 14569]
 [ 5153 28459 13787]
 [13765 10255 67326]]
[09:25:53] Test macro-F1 (Q8): 0.2942
[09:25:53] Q8 Classification Report:
              precision    recall  f1-score   support

           0     0.6869    0.8270    0.7504     68513
           1     0.5789    0.7020    0.6346     45075
           2     0.5168    0.5807    0.5469     52245
           3     0.4076    0.3277    0.3633     22785
           4     0.3219    0.0155    0.0296     16316
           5     0.2632    0.0154    0.0290      8131
           6     0.0000    0.0000    0.0000      2324
           7     0.0000    0.0000    0.0000        48

    accuracy                         0.5871    215437
   macro avg     0.3469    0.3085    0.2942    215437
weighted avg     0.5423    0.5871    0.5458    215437

[09:25:53] Q8 Confusion Matrix:
[[56659  4988  4953  1802    42    69     0     0]
 [ 5303 31642  6527  1478    90    35     0     0]
 [ 8357  9584 30340  3664   206    94     0     0]
 [ 5733  3248  6091  7466   159    88     0     0]
 [ 2954  3006  7315  2729   253    59     0     0]
 [ 3055  1468  2454  1002    27   125     0     0]
 [  395   713  1028   175     8     5     0     0]
 [   34     6     4     3     1     0     0     0]]