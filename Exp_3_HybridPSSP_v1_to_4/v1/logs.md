# Revision Log

## Baseline Approach (v1)

- **Problem framing**: Protein Secondary Structure Prediction (PSSP) is handled as a per-residue classification task. Given a primary amino-acid sequence (`seq`), the model predicts either the coarse three-class labels (Q3: helix, strand, coil) or the finer eight-state scheme (Q8). These labels mirror what DSSP would derive from experimentally determined 3D structures, offering a computational proxy for costly lab techniques like X-ray crystallography.

- **Model architecture**: `BiLSTMProtein` embeds each residue, passes the sequence through a multi-layer bidirectional LSTM, and applies a linear classifier to every timestep. Bidirectionality lets the network see both N-terminal and C-terminal context, capturing local motifs (e.g., helix caps) as well as longer-range dependencies important for sheet pairing.

- **Sequence encoding**: Residues map to integer indices across the canonical 20 amino acids (padding index reserved for shorter chains). The embedding layer transforms these indices into dense vectors learned jointly with the classifier.

- **Optional features**: The forward pass can concatenate external per-residue feature matrices (e.g., PSI-BLAST-derived PSSM rows or language-model embeddings) with the learned embeddings. Shape compatibility is enforced during loading so each timestep has a consistent representation.

- **Data preparation**: `dataprep.py` merges the curated archive CSVs, normalises column names, filters sequences exceeding length bounds or containing non-standard residues, and computes Q3 labels when only Q8 exists. Randomised, stratified splits (`train.csv`, `val.csv`, `test.csv`) are emitted for reproducible experiments.

- **Dataset & collation**: `ProteinDataset` reads the processed CSVs, converts sequences and labels into tensors, trims to `max_len` when requested, and optionally loads `.npy` feature files keyed by `pdb_id_chain`. The `collate_fn` sorts by sequence length, pads sequences and labels to the batch maximum, and constructs boolean masks so padded positions can be ignored downstream.

- **Training routine**: `train.py` wires the dataset into a PyTorch `DataLoader`, optimises with Adam, and minimises per-residue cross-entropy while ignoring padded tokens (`ignore_index=-100`). Progress logging tracks loss, and the best checkpoint (`best_model.pt`) is saved based on validation macro-F1, which balances performance across class imbalance.

- **Evaluation**: `test.py` mirrors the training configuration, loading the saved checkpoint and reporting macro-F1 alongside a classification report to highlight how well the model recovers helices, strands, and coils from sequence alone.


# RESULTS

[08:43:45] Classification Report:
              precision    recall  f1-score   support

           0     0.7531    0.7552    0.7552     76692
           1     0.6590    0.6360    0.6360     47399
           2     0.7094    0.7308    0.7199     91346

    accuracy                         0.7147    215437
   macro avg     0.7072    0.7009    0.7037    215437
weighted avg     0.7139    0.7147    0.7140    215437
