# HybridPSSP v4

HybridPSSP v4 is a hybrid Transformer + dilated CNN architecture for protein secondary structure prediction (PSP) that delivers concurrent Q3 and Q8 residue-level labels. This release consolidates the v1–v3 feature pipelines into a unified `(B, L, D)` tensor interface and leans on high-capacity ESM language model embeddings for long-range context while preserving convolutional sensitivity to local motifs.

## Key Improvements Over Earlier Revisions
- Learned positional encodings replace fixed sinusoidal tables, enabling residues to learn task-specific spatial priors.
- A deeper Transformer encoder (default 6 layers, norm-first) models global dependencies across long proteins.
- Parallel dilated residual CNN blocks expand the local receptive field without forfeiting parameter efficiency.
- Flexible fusion (`sum` or `concat`) reconciles Transformer and CNN representations before lightweight Q3/Q8 heads.
- Data loaders emit padded tensors with boolean masks, simplifying mixed-precision training and device transfers.

## Repository Layout
- `dataprep.py` – data cleaning, train/val/test split, optional PSSM and ESM feature generation, and `ProteinDataset` implementation.
- `model.py` – hybrid architecture definition with positional encodings, Transformer stack, dilated CNN blocks, and joint heads.
- `train.py` – training loop with `OneCycleLR`, AMP support, class-weighting utilities, and checkpoint management.
- `test.py` – evaluation script that restores a checkpoint and reports macro-F1, classification reports, and confusion matrices.
- `visualization.py` – helper to visualise the saved Q3/Q8 confusion matrices.

## Data Prerequisites
Place the `archive` directory at the project root with the following CSVs (exact filenames required):

1. `2018-06-06-pdb-intersect-pisces.csv`
	- Columns: `pdb_id, chain_code, seq, sst8, sst3, len, has_nonstd_aa, Exptl., resolution, R-factor, FreeRvalue`
2. `2018-06-06-ss.cleaned.csv`
	- Columns: `pdb_id, chain_code, seq, sst8, sst3, len, has_nonstd_aa`

`dataprep.py` will merge, clean, and split these files before feature generation.

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure you have GPU access (`cuda` or `mps`) for the provided defaults. Install PSI-BLAST only if you intend to generate PSSM profiles; ESM embeddings are sufficient for the default recipe below.

## Workflow

1. **Pre-compute ESM embeddings and Data splits** (writes `.npy` files beside the CSV splits):
	```bash
	python dataprep.py \
		 --archive archive \
		 --out data/processed \
		 --esm_dir data/esm_embeddings \
		 --esm_model esm2_t33_650M_UR50D \
		 --esm_device cuda \
		 --esm_batch_tokens 256
	```

2. **Train the hybrid model** (checkpoints saved i.a. to `checkpoints_esm/best_model.pt`):
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

3. **Evaluate on the test set** (restores the best checkpoint and prints metrics):
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

Optional: run `visualization.py` after testing to render the confusion matrices stored in `logs.md`.

## Reference Results (ESM embeddings only)
- Q3 macro-F1: **0.8707**
- Q8 macro-F1: **0.5210**

Confusion matrices and classification reports for both heads are logged in `logs.md` and replicated via `visualization.py`.

## Notes
- `train.py` automatically detects class imbalance and can apply inverse or effective-number weighting; adjust via `--q8_balance`.
- Set `--fuse_mode concat` to explore alternative fusion, and `--pssm_dir` if you have precomputed PSSM profiles.
- Increase `--max_len` or `--max_position` cautiously; the learned positional table defaults to 4096 residues.
