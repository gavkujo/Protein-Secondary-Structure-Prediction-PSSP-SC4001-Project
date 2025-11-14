# Final Revision Log

# Changes from v3
- **Data preparation**: `ProteinDataset` now builds a single dense feature tensor by concatenating one-hot, PSSM, and ESM blocks and caches the inferred dimensionality. v3 delivered integer token sequences plus an optional feature array per sample, forcing downstream code to branch on available modalities.
- **Feature hygiene**: v4 trims mismatched label/feature lengths, tracks missing PSSM/ESM matrices with warning deduplication, and applies configurable normalisation (`none`, `zscore`, `minmax`) at load time. v3 surfaced raw numpy dumps with no scaling or diagnostics.
- **PSSM toolchain**: `generate_pssm_features` wraps PSI-BLAST invocation, temp-file management, and consistency checks, auto-saving `.npy` matrices and truncating mismatched lengths. v3 expected precomputed features dropped into `features_dir` with no orchestration.
- **ESM integration**: `generate_esm_embeddings` streams batches under a token budget, handles CPU fallback, and writes per-chain `.npy` files. v3 relied on external scripts and only surfaced a `use_features` flag.
- **Collation**: loader padding now operates directly on feature tensors and boolean masks, yielding a unified `features` tensor and optional `labels_q8`. v3 padded token IDs separately and stitched features post-hoc, duplicating masking logic in the trainer.

## Model evolution
- **Input contract**: `HybridPSSP` consumes `(B, L, D)` feature tensors with an optional validity mask, eliminating vocab embeddings and simplifying multi-feature fusion. `ProteinHybridModel` in v3 embedded integer tokens and conditionally concatenated projected features.
- **Positional encoding**: a learned positional table (size guarded by `max_len`) replaces the fixed sinusoidal buffer, allowing the network to adapt residue-relative biases beyond the hand-crafted Fourier basis used in v3.
- **Attention stack**: transformer depth defaults to six layers with configurable feed-forward expansion (`ff_multiplier`) and norm-first blocks. The earlier stack was four layers with fixed feed-forward width and no fusion with additional convolutional paths.
- **Dilated CNN blocks**: v4 introduces parallel dilated residual blocks with tunable kernel sizes and dilations, expanding the receptive field while preserving parameter efficiency. v3 relied on repeated depthwise separable convolutions with a single dilation pattern.
- **Fusion strategies**: outputs from the transformer and CNN branches can be combined via residual `sum` or `concat + linear` projection, giving explicit control over how global and local cues mix. v3 fed a single stream through LayerNorm before the heads.
- **Prediction heads**: both Q3 and Q8 heads share a streamlined dropout+linear stack, backed by `head_dropout`. v3 exposed deeper multi-layer heads only for Q8 and required manual configuration of intermediate widths.

## Training and evaluation workflow
- Data loaders feed padded feature tensors and masks directly into the model, improving compatibility with mixed precision (`torch.cuda.amp`) and reducing redundant moves between CPU and GPU. v3 reassembled embeddings inside `forward` and carried both raw token IDs and optional features.
- Hyperparameter surface grew to cover CNN dilation grids, fusion mode, positional span (`max_position`), and feed-forward expansion, enabling finer ablations without code edits.
- Class-weight computation remains but now pairs naturally with the richer feature stack, as every sequence benefits from the same number of modalities regardless of auxiliary file availability.
- Evaluation reporting now enumerates which feature sources were active, easing experiment reproducibility.

## Embedding and profile features
- **PSSM profiles**: Iterative PSI-BLAST builds position-specific scoring matrices that encode evolutionary substitution propensities and conservation. In structural biology literature these profiles correlate strongly with backbone torsion stability and beta-sheet propensity; here they complement one-hot encodings by flagging residues whose mutational tolerance is low, aiding both the transformer attention (global conservation cues) and the dilated CNN paths (local motif reinforcement).
- **ESM contextual embeddings**: Protein language models (e.g. ESM2) produce per-residue vectors that internalise remote contacts, disorder likelihood, and secondary-structure motifs by training on millions of unlabeled sequences. Incorporating these embeddings provides high-level contextual priors that are especially helpful when PSSM coverage is sparse or when sequences lack close homologs, letting the hybrid model reason over long-range dependencies without bespoke structural inputs.
- Together, stacking one-hot, PSSM, and ESM channels yields a feature tensor that blends sequence identity, evolutionary constraints, and learned biophysical priors, supplying the v4 architecture with richer signals than the v3 pipeline, which could only consume one-hot tokens plus whatever handcrafted features happened to be on disk.

# RESULTS (Using ESM Embeddings Only)
[10:31:22] Evaluating Q3 and Q8 heads
[10:31:33] Test macro-F1 (Q3): 0.8707                     
[10:31:33] Q3 Classification Report:
              precision    recall  f1-score   support

           0     0.8979    0.9028    0.9004     76692
           1     0.8842    0.8304    0.8565     47399
           2     0.8439    0.8666    0.8551     91346

    accuracy                         0.8716    215437
   macro avg     0.8753    0.8666    0.8707    215437
weighted avg     0.8720    0.8716    0.8715    215437

[10:31:33] Q3 Confusion Matrix:
[[69240   400  7052]
 [  446 39362  7591]
 [ 7424  4757 79165]]
[10:31:33] Test macro-F1 (Q8): 0.5210
[10:31:33] Q8 Classification Report:

              precision    recall  f1-score   support

           0     0.8818    0.9339    0.9071     68513
           1     0.8633    0.8673    0.8653     45075
           2     0.6769    0.7707    0.7208     52245
           3     0.6257    0.6355    0.6306     22785
           4     0.5956    0.3187    0.4152     16316
           5     0.5597    0.4006    0.4670      8131
           6     0.5362    0.0955    0.1622      2324
           7     0.0000    0.0000    0.0000        48

    accuracy                         0.7729    215437
   macro avg     0.5924    0.5028    0.5210    215437
weighted avg     0.7634    0.7729    0.7618    215437

[10:31:33] Q8 Confusion Matrix:
[[63982   317  2141  1233    73   764     3     0]
 [  346 39095  4526   502   454    84    68     0]
 [ 2701  4121 40267  2500  2070   503    83     0]
 [ 3089   342  3253 14481   720   895     5     0]
 [  702   740  6641  2712  5200   290    31     0]
 [ 1630   176  1422  1537   107  3257     2     0]
 [   80   491  1235   169   102    25   222     0]
 [   30     1     2     9     5     1     0     0]]
