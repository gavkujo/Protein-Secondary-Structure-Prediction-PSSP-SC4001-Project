# Hybrid Protein Secondary Structure Prediction Experiments

## Instructions to run the Final Model (Exp 3 · v4)
1. **Ensure the archive is available at the project root**
	- `archive/2018-06-06-pdb-intersect-pisces.csv`
	  - Columns: `pdb_id, chain_code, seq, sst8, sst3, len, has_nonstd_aa, Exptl., resolution, R-factor, FreeRvalue`
	- `archive/2018-06-06-ss.cleaned.csv`
	  - Columns: `pdb_id, chain_code, seq, sst8, sst3, len, has_nonstd_aa`
2. **Create and activate a virtual environment, then install dependencies**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r Exp_3_HybridPSSP_v1_to_4/v4\ (final)/requirements.txt
	```
3. **Generate ESM embeddings and processed splits**
	```bash
	cd Exp_3_HybridPSSP_v1_to_4/v4\ (final)
	python dataprep.py \
		 --archive archive \
		 --out data/processed \
		 --esm_dir data/esm_embeddings \
		 --esm_model esm2_t33_650M_UR50D \
		 --esm_device cuda \
		 --esm_batch_tokens 256
	```
4. **Train the HybridPSSP v4 model**
	```bash
	python train.py \
		 --data_dir data/processed \
		 --ckpt_dir checkpoints_esm \
		 --epochs 60 \
		 --batch_size 32 \
		 --lr 1e-4 \
		 --model_dim 512 \
		 --transformer_layers 6 \
		 --attention_heads 8 \
		 --esm_dir data/esm_embeddings \
		 --q8_weight 0.6 \
		 --dropout 0.2 \
		 --head_dropout 0.25 \
		 --label_smoothing 0.1 \
		 --weight_decay 1e-4 \
		 --device cuda
	```
5. **Evaluate on the held-out test set**
	```bash
	python test.py \
		 --data_dir data/processed \
		 --ckpt checkpoints_esm/best_model.pt \
		 --batch_size 32 \
		 --model_dim 512 \
		 --transformer_layers 6 \
		 --attention_heads 8 \
		 --esm_dir data/esm_embeddings \
		 --device cuda
	```

## Project Premise
Deep learning for protein structure prediction addresses the challenge of inferring a protein’s secondary structure directly from its amino-acid sequence. Experimental pipelines (X-ray crystallography, NMR) are accurate but slow and expensive; instead, we predict both coarse-grained Q3 states (helix, strand, coil) and fine-grained Q8 states (eight DSSP categories). Successful models balance local motif recognition with long-range dependencies and benefit from enriched residue features (evolutionary profiles, protein language-model embeddings).

## Experiment Portfolio

| Experiment | Location | Goal | Highlights | Results |
|-----------:|:---------|:-----|:-----------|:--------|
| **Exp 1** | `Exp_1_Transformer/` | Establish a pure Transformer baseline trained end-to-end on per-residue labels. | Implemented `ProteinTransformer` notebook with sinusoidal encodings, multi-head self-attention, early stopping, and joint Q3/Q8 loss tracking. Served as a sandbox for data loaders, padding strategy, and PyTorch Lightning-style training utilities. | End-to-end training pipeline validated; reliable test metrics were not persisted because experiments remained exploratory.
| **Exp 2** | `Exp_2_CNN_BiLSTM/` | Explore convolutional + BiLSTM hybrids and improve data hygiene. | Introduced cleaned CSV splits, systematic batching (`collate_batch`), and confusion-matrix plotting inside `protein_4001.ipynb`. Demonstrated that CNN front-ends sharpened helix/strand boundaries before the recurrent stack. | Prototype runs plateaued around mid-0.7 Q3 accuracy (per notebook traces); full exports were not saved, motivating tighter logging in later revisions.
| **Exp 3 · v1** | `Exp_3_HybridPSSP_v1_to_4/v1/` | PyTorch script refactor of the BiLSTM baseline with CLI tooling. | Added `ProteinDataset`, stratified splits, macro-F1 checkpointing, and Q3-only supervision with optional external features. | Q3 macro-F1 **0.7037**.
| **Exp 3 · v2** | `Exp_3_HybridPSSP_v1_to_4/v2/` | Extend v1 with optional Transformer layers and improved optimisation. | AdamW + OneCycleLR, AMP, label smoothing, configurable attention stack, length trimming, and feature dimension inference. | Q3 macro-F1 **0.6997** (attention-heavy configs overfit without richer inputs).
| **Exp 3 · v3** | `Exp_3_HybridPSSP_v1_to_4/v3/` | Transition to a true hybrid that predicts Q3 and Q8 jointly. | Introduced parallel CNN + attention fusion, Q8 heads with class weighting, data loader support for Q8 labels, and richer CLI surface. | Q3 macro-F1 **0.6987** · Q8 macro-F1 **0.2942** (highlighted need for higher-capacity features).
| **Exp 3 · v4 (final)** | `Exp_3_HybridPSSP_v1_to_4/v4 (final)/` | Final HybridPSSP architecture mixing Transformer encoders with dilated CNNs and ESM embeddings. | Learned positional encodings, dilated residual CNN blocks, flexible fusion (`sum`/`concat`), ESM embedding integration, improved masking, AMP-ready data loaders, and confusion-matrix visualisation. | Q3 macro-F1 **0.8707** · Q8 macro-F1 **0.5210** (ESM embeddings only).

### Development Narrative
- **Data foundation**: All experiments rely on the curated 2018 PDB/PISCES intersect. `dataprep.py` matured from notebook utilities to CLI tools capable of generating train/val/test CSVs, PSI-BLAST PSSMs, and ESM embeddings. Alignment, trimming, and normalisation routines now guarantee `(B, L, D)` feature stacks without manual fixes.
- **Architectural evolution**: We progressed from pure Transformers (Exp 1) and CNN-BiLSTM mixtures (Exp 2) to scriptable BiLSTM baselines (Exp 3 v1), optional Transformer add-ons (v2), joint-task hybrids (v3), and the final Transformer + dilated CNN fusion (v4). Each revision emphasised balancing global attention with local motif extraction.
- **Training infrastructure**: Early notebooks tracked accuracy curves informally. By v1 we introduced CLI-driven scripts, macro-F1 checkpointing, and deterministic splits. v2/v3 layered in AMP, gradient clipping, OneCycleLR, label smoothing, class weighting, and device auto-resolution. v4 stabilised multi-modal feature ingestion and simplified experiment reproducibility.
- **Feature strategy**: Initial attempts relied on learned embeddings alone. PSSM and ESM features were gradually added, with v4 defaulting to high-quality ESM embeddings (and optional PSSMs) that markedly improved Q8 performance.

### Result Synopsis
| Model | Q3 Macro-F1 | Q8 Macro-F1 | Notes |
|:------|:------------|:------------|:------|
| BiLSTM Baseline (v1) | 0.7037 | – | One-hot tokens + optional features; Q3-only training. |
| BiLSTM + Optional Attention (v2) | 0.6997 | – | Gains in flexibility but limited by feature quality. |
| Hybrid (CNN + Attention) v3 | 0.6987 | 0.2942 | Q8 supervision introduced; feature scarcity capped gains. |
| HybridPSSP v4 (Transformer + Dilated CNN w/ ESM) | **0.8707** | **0.5210** | Final configuration; best performing model. |

## Folder Guide
- `Exp_1_Transformer/`: Experimental notebook for the transformer-only baseline (`protein_transformer.ipynb`).
- `Exp_2_CNN_BiLSTM/`: Notebook exploring CNN + BiLSTM hybrids (`protein_4001.ipynb`, processed CSVs for reproducibility).
- `Exp_3_HybridPSSP_v1_to_4/`: Scripted experiments v1–v4 with progressively richer architectures, logs, and utilities.
- `ReadME.md`: This consolidated summary and quickstart guide.

