"""
data_prep.py
--------------
Data loading, cleaning and feature generation utilities for protein secondary structure prediction.
--------------
Functions:
- load_raw: Load and merge the two CSVs from the archive into a unified dataframe.
- clean_df: Clean dataframe: drop sequences with nonstandard AA, normalize sst3, remove trivial lengths.
- split_save: Split dataframe into train/val/test and save as CSVs.
- generate_pssm_features: Run PSI-BLAST to generate PSSM features for all sequences in a dataframe.
Classes:
- ProteinDataset: PyTorch Dataset for loading protein sequences, labels, and optional features.
- collate_fn: Collate function for DataLoader to handle variable-length sequences and optional features.
--------------
Data format:
The archive folder should contain two CSV files:
1) 2018-06-06-pdb-intersect-pisces.csv
     columns: pdb_id,chain_code,seq,sst8,sst3,len,has_nonstd_aa,Exptl.,resolution,R-factor,FreeRvalue
2) 2018-06-06-ss.cleaned.csv
    columns: pdb_id,chain_code,seq,sst8,sst3,len,has_nonstd_aa
Feature generation:
1) PSSM features via PSI-BLAST
    For each sequence, run PSI-BLAST against a large protein database (e.g., UniRef90)
    to generate a position-specific scoring matrix (PSSM). The PSSM can be saved as a
    .npy file for later loading.
2) ESM embeddings via Fairseq ESM models
    Use pre-trained ESM models to generate contextual embeddings for each residue
    in the protein sequence. These embeddings can also be saved as .npy files.
--------------
Data preparation workflow:
1) Load and clean the raw data from the archive CSVs using load_raw and clean_df.
2) Split the cleaned dataframe into train/val/test sets using split_save.
3) Generate PSSM features for all sequences using generate_pssm_features.
4) OR Generate ESM embeddings for all sequences using generate_esm_embeddings.
--------------
Data format details:
The cleaned dataframe should contain the following columns:
- pdb_id: PDB identifier of the protein
- chain_code: Chain identifier within the PDB structure
- seq: Amino acid sequence (string of single-letter codes)
- sst3: Secondary structure labels in Q3 format (H, E, C)
- sst8: Secondary structure labels in Q8 format (H, E, C, T, S, G, B, I)
- len: Length of the sequence
- has_nonstd_aa: Boolean indicating presence of nonstandard amino acids
--------------
Example Usage:
# Generate processed data ready for training
python dataprep.py --archive archive --out data/processed 
# PSSM feature generation
python dataprep.py --generate_pssm --data_csv data/processed/train.csv --features_dir data/features/pssm --blast_db uniref90 --psiblast_cmd psiblast
# ESM embedding generation
python dataprep.py --generate_esm --data_csv data/processed/train.csv --features_dir data/features/esm --esm_model esm2_t33_650M_UR50D
"""

import os
import csv
import random
import subprocess
import tempfile
import textwrap
import warnings
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

RANDOM_SEED = 42

Q8_TO_Q3 = {
    'H': 'H', 'G': 'H', 'I': 'H',  # helix types -> H
    'E': 'E', 'B': 'E',            # strand types -> E
    'T': 'C', 'S': 'C', 'C': 'C'   # turn/bend/coil -> C
}

SS8_ALPHABET = ['H', 'E', 'C', 'T', 'S', 'G', 'B', 'I']
SS8_MAP = {ch: idx for idx, ch in enumerate(SS8_ALPHABET)}

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # canonical 20 amino acids
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}  # 0 reserved for padding
AA_TO_ONEHOT = {aa: i for i, aa in enumerate(AA_ALPHABET)}
ONE_HOT_DIM = len(AA_ALPHABET)
PAD_IDX = 0
SS3_MAP = {'H': 0, 'E': 1, 'C': 2}
NUM_Q8_CLASSES = len(SS8_ALPHABET)

PSSM_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
PSSM_NUM_FEATURES = len(PSSM_AA_ORDER)
DEFAULT_PSSM_ITERATIONS = 3
DEFAULT_PSSM_EVALUE = 1e-3
ESM_DEFAULT_MODEL = "esm2_t33_650M_UR50D"


def make_entry_id(pdb_id: Optional[str], chain_code: Optional[str]) -> str:
    # Build a stable identifier so auxiliary feature files line up with CSV rows.
    pid = str(pdb_id).strip() if pd.notna(pdb_id) else 'UNK'
    chain = str(chain_code).strip() if pd.notna(chain_code) and str(chain_code).strip() else 'UNK'
    return f"{pid}_{chain}".replace('/', '-')


def _read_csv_guess(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        # Standard case: comma-separated CSV exported by our preprocessing notebooks.
        df = pd.read_csv(path)
    except Exception:
        # If the commas fail, try tab separation since some archives ship as TSV.
        df = pd.read_csv(path, sep='\t')
    return df

def load_raw(archive_dir: str) -> pd.DataFrame:
    """Load the two CSVs from the archive and merge/filter them into a unified dataframe.

    The function looks for the two exact filenames and then finds the cleaned SS file which
    contains sequence and labels. If the intersect file exists it'll filter the cleaned set.
    """
    cleaned_path = os.path.join(archive_dir, '2018-06-06-ss.cleaned.csv')
    intersect_path = os.path.join(archive_dir, '2018-06-06-pdb-intersect-pisces.csv')

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Expected cleaned SS file not found: {cleaned_path}")

    df = _read_csv_guess(cleaned_path)

    # common expected columns: pdb_id, chain_code, seq, sst8, sst3, len, has_nonstd_aa
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    required = ['pdb_id', 'chain_code', 'seq', 'sst8', 'sst3', 'len', 'has_nonstd_aa']
    # be permissive: try lowercase variants
    cols_lower = {c.lower(): c for c in df.columns}
    for r in required:
        if r not in df.columns:
            if r in cols_lower:
                df.rename(columns={cols_lower[r]: r}, inplace=True)

    # If intersect exists, filter
    if os.path.exists(intersect_path):
        try:
            df_inter = _read_csv_guess(intersect_path)
            df_inter.columns = [c.strip() for c in df_inter.columns]
            # try to detect id columns
            id_cols = [c for c in df_inter.columns if c.lower() in ('pdb_id', 'pdb', 'pdbid')]
            chain_cols = [c for c in df_inter.columns if c.lower() in ('chain_code', 'chain')]
            if id_cols:
                pid = id_cols[0]
                if chain_cols:
                    cid = chain_cols[0]
                    # Preserve only entries that exist in both archive files (PDB + chain pair).
                    keys = set(zip(df_inter[pid].astype(str).str.upper(), df_inter[cid].astype(str)))
                    # filter df
                    df = df[df.apply(lambda r: (str(r.get('pdb_id')).upper(), str(r.get('chain_code'))) in keys, axis=1)]
                else:
                    pids = set(df_inter[pid].astype(str).str.upper())
                    df = df[df['pdb_id'].astype(str).str.upper().isin(pids)]
        except Exception:
            # If filtering fails, continue with cleaned df as-is
            pass

    return df

def clean_df(df: pd.DataFrame, min_len: int = 20, max_len: int = 2000, drop_nonstd: bool = True) -> pd.DataFrame:
    """Clean dataframe: drop sequences with nonstandard AA (optionally), normalize sst3, remove trivial lengths.

    Also replaces masked nonstandard residues in sequences ("*") by X and can drop if desired.
    """
    df = df.copy()
    # ensure expected columns exist; if not, try to create from alternatives
    if 'seq' not in df.columns and 'sequence' in df.columns:
        df.rename(columns={'sequence': 'seq'}, inplace=True)
    if 'sst3' not in df.columns and 'ss3' in df.columns:
        df.rename(columns={'ss3': 'sst3'}, inplace=True)
    if 'sst8' not in df.columns and 'ss8' in df.columns:
        df.rename(columns={'ss8': 'sst8'}, inplace=True)

    # drop rows with missing sequence or labels
    # Drop completely empty sequences because they cannot form training examples.
    df = df.dropna(subset=['seq'])
    if 'sst3' not in df.columns and 'sst8' not in df.columns:
        raise ValueError('Neither sst3 nor sst8 present in the dataframe')

    # If only sst8 present, create sst3 mapping column
    if 'sst3' not in df.columns and 'sst8' in df.columns:
        def map_q8_to_q3(s8: str) -> str:
            # Collapse the detailed Q8 alphabet into coarse-grained Q3 labels residue by residue.
            return ''.join([Q8_TO_Q3.get(ch, 'C') for ch in s8])
        df['sst3'] = df['sst8'].astype(str).apply(map_q8_to_q3)

    # standardize has_nonstd_aa column
    if 'has_nonstd_aa' in df.columns:
        df['has_nonstd_aa'] = df['has_nonstd_aa'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
    else:
        # infer from seq presence of nonstandard letters B,O,U,X,Z or '*' masking
        df['has_nonstd_aa'] = df['seq'].apply(lambda s: any(ch in set('B O U X Z *'.split()) for ch in s))

    if drop_nonstd:
        # Optionally filter anything containing suspicious amino acids.
        df = df[~df['has_nonstd_aa']]

    # length checks
    df['len'] = df['seq'].astype(str).apply(len)
    df = df[df['len'] >= min_len]
    df = df[df['len'] <= max_len]

    # uppercase sequences and strip whitespace
    df['seq'] = df['seq'].astype(str).str.strip().str.upper()
    df['sst3'] = df['sst3'].astype(str).str.strip().str.upper()
    if 'sst8' in df.columns:
        df['sst8'] = df['sst8'].astype(str).str.strip().str.upper()

    # ensure lengths match labels
    def valid_row(r):
        return len(r['seq']) == len(r['sst3'])
    df = df[df.apply(valid_row, axis=1)]

    df = df.reset_index(drop=True)
    return df

def split_save(df: pd.DataFrame, out_dir: str, val_frac: float = 0.1, test_frac: float = 0.1):
    os.makedirs(out_dir, exist_ok=True)
    # stratify by length buckets to avoid weird skew
    random.seed(RANDOM_SEED)
    idx = list(df.index)
    random.shuffle(idx)
    n = len(idx)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    # Persist deterministic CSV splits so training and evaluation stay reproducible across runs.
    df.loc[train_idx].to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    df.loc[val_idx].to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    df.loc[test_idx].to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    print(f"Saved splits to {out_dir}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


def _parse_ascii_pssm(pssm_path: str) -> np.ndarray:
    rows: List[np.ndarray] = []
    pattern = re.compile(r'^\s*(\d+)\s+[A-Z]\s+')
    with open(pssm_path, 'r') as handle:
        for line in handle:
            if not pattern.match(line):
                continue
            parts = line.strip().split()
            # Each PSI-BLAST row lists log-odds followed by probabilities for the 20 amino acids.
            if len(parts) < 2 + PSSM_NUM_FEATURES:
                continue
            log_odds = parts[2:2 + PSSM_NUM_FEATURES]
            probs_start = 2 + PSSM_NUM_FEATURES
            probs = parts[probs_start:probs_start + PSSM_NUM_FEATURES]
            if len(probs) < PSSM_NUM_FEATURES:
                probs = parts[-PSSM_NUM_FEATURES:]
            try:
                log_odds_vals = [float(x) for x in log_odds]
                prob_vals = [float(x) / 100.0 for x in probs]
            except ValueError:
                continue
            row_vec = np.asarray(log_odds_vals + prob_vals, dtype=np.float32)
            rows.append(row_vec)
    if not rows:
        raise ValueError(f"No usable PSI-BLAST rows parsed from {pssm_path}. Ensure -out_ascii_pssm was set.")
    return np.vstack(rows)


def _load_pssm_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == '.npy':
        return np.load(path).astype(np.float32)
    return _parse_ascii_pssm(str(path))


def _run_psiblast(sequence: str,
                  seq_id: str,
                  out_dir: Path,
                  psiblast_cmd: str,
                  blast_db: str,
                  iterations: int = DEFAULT_PSSM_ITERATIONS,
                  evalue: float = DEFAULT_PSSM_EVALUE,
                  threads: int = 4,
                  tmp_root: Optional[str] = None) -> Path:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("PSI-BLAST received empty sequence")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
        query_path = Path(tmp_dir) / f"{seq_id}.fasta"
        pssm_path = Path(tmp_dir) / f"{seq_id}.pssm"
        with open(query_path, 'w') as handle:
            handle.write(f">{seq_id}\n{sequence}\n")
        # Compose the BLAST+ command that produces an ASCII PSSM we can parse downstream.
        cmd = [
            psiblast_cmd,
            '-query', str(query_path),
            '-db', blast_db,
            '-num_iterations', str(iterations),
            '-evalue', str(evalue),
            '-out_ascii_pssm', str(pssm_path),
            '-out', os.devnull,
            '-num_threads', str(max(1, threads)),
            '-comp_based_stats', '1'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"PSI-BLAST executable '{psiblast_cmd}' not found. Install BLAST+ and ensure psiblast is on PATH."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"PSI-BLAST failed for {seq_id} (exit code {result.returncode}).\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        matrix = _parse_ascii_pssm(str(pssm_path))
    final_path = out_dir / f"{seq_id}.npy"
    np.save(final_path, matrix.astype(np.float32))
    return final_path


def generate_pssm_features(df: pd.DataFrame,
                           out_dir: str,
                           psiblast_cmd: str = 'psiblast',
                           blast_db: Optional[str] = None,
                           iterations: int = DEFAULT_PSSM_ITERATIONS,
                           evalue: float = DEFAULT_PSSM_EVALUE,
                           threads: int = 4,
                           overwrite: bool = False,
                           tmp_root: Optional[str] = None) -> None:
    if not blast_db:
        raise ValueError("blast_db must be provided to generate PSSM features")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    total_sequences = len(df)
    processed = 0
    for _, row in df.iterrows():
        seq = str(row.get('seq', '')).strip()
        if not seq:
            continue
        entry_id = make_entry_id(row.get('pdb_id'), row.get('chain_code'))
        target_file = out_path / f"{entry_id}.npy"
        if target_file.exists() and not overwrite:
            processed += 1
            continue
        try:
            # Launch PSI-BLAST and immediately cache the resulting matrix to disk.
            saved_path = _run_psiblast(
                sequence=seq,
                seq_id=entry_id,
                out_dir=out_path,
                psiblast_cmd=psiblast_cmd,
                blast_db=blast_db,
                iterations=iterations,
                evalue=evalue,
                threads=threads,
                tmp_root=tmp_root,
            )
            matrix = np.load(saved_path)
            if matrix.shape[0] != len(seq):
                warnings.warn(
                    f"PSSM length mismatch for {entry_id}: seq_len={len(seq)} pssm_len={matrix.shape[0]} (truncating)")
                min_len = min(len(seq), matrix.shape[0])
                np.save(saved_path, matrix[:min_len])
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to generate PSSM for {entry_id}: {exc}")
            failures.append(entry_id)
            continue
        processed += 1
        if processed % 50 == 0:
            print(f"[PSSM] {processed}/{total_sequences} sequences processed")
    print(f"[PSSM] Completed with {processed} successes and {len(failures)} failures. Output dir: {out_path}")


def generate_esm_embeddings(df: pd.DataFrame,
                            out_dir: str,
                            model_name: str = ESM_DEFAULT_MODEL,
                            layer: Optional[int] = None,
                            device: str = 'cuda',
                            batch_tokens: int = 2048,
                            overwrite: bool = False) -> None:
    try:
        import torch
        import esm  # type: ignore[import]
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("ESM embeddings require the fair-esm package. Install via `pip install fair-esm`."
                          ) from exc

    if device != 'cpu' and not torch.cuda.is_available():
        warnings.warn("CUDA device requested for ESM embeddings but not available; falling back to CPU")
        device = 'cpu'
    device_obj = torch.device(device)
    # Load the pretrained language model once; reuse it for all sequences.
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device_obj)
    if layer is None:
        layer = model.num_layers
    batch_converter = alphabet.get_batch_converter()

    sequences: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        seq = str(row.get('seq', '')).strip()
        if not seq:
            continue
        entry_id = make_entry_id(row.get('pdb_id'), row.get('chain_code'))
        sequences.append((entry_id, seq))

    if not sequences:
        print("[ESM] No sequences available for embedding generation.")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    progress = 0
    total = len(sequences)
    cursor = 0
    while cursor < total:
        batch: List[Tuple[str, str]] = []
        token_budget = 0
        while cursor < total and token_budget < batch_tokens:
            entry = sequences[cursor]
            prospective = token_budget + len(entry[1]) + 2
            if batch and prospective > batch_tokens:
                break
            batch.append(entry)
            token_budget = prospective
            cursor += 1
        # Convert strings into ESM token tensors; alphabet handles padding/special tokens for us.
        labels, batch_strs, batch_tokens_tensor = batch_converter(batch)
        batch_tokens_tensor = batch_tokens_tensor.to(device_obj)
        with torch.no_grad():
            outputs = model(batch_tokens_tensor, repr_layers=[layer], return_contacts=False)
        reps = outputs['representations'][layer].cpu()
        for idx, (entry_id, seq) in enumerate(batch):
            target_file = out_path / f"{entry_id}.npy"
            if target_file.exists() and not overwrite:
                continue
            residue_repr = reps[idx, 1:len(seq) + 1, :].numpy().astype(np.float32)
            # Skip the BOS/EOS slots and keep only residue-aligned embeddings.
            np.save(target_file, residue_repr)
            progress += 1
        print(f"[ESM] {progress}/{total} embeddings saved -> {out_path}")

from torch.utils.data import Dataset
import torch


def _one_hot_encode(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), ONE_HOT_DIM), dtype=np.float32)
    for idx, aa in enumerate(seq):
        pos = AA_TO_ONEHOT.get(aa)
        if pos is not None:
            # Place a single 1.0 in the column representing the observed residue.
            arr[idx, pos] = 1.0
    return arr


def _normalise_matrix(matrix: np.ndarray, mode: str = 'zscore') -> np.ndarray:
    if mode == 'none':
        return matrix
    if mode == 'zscore':
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True) + 1e-6
        return (matrix - mean) / std
    if mode == 'minmax':
        min_val = matrix.min(axis=0, keepdims=True)
        max_val = matrix.max(axis=0, keepdims=True)
        denom = np.maximum(max_val - min_val, 1e-6)
        return (matrix - min_val) / denom
    raise ValueError(f"Unknown normalisation mode: {mode}")


class ProteinDataset(Dataset):
    """Dataset for processed CSVs with stacked per-residue feature tensors."""

    def __init__(self,
                 csv_path: str,
                 max_len: Optional[int] = None,
                 include_one_hot: bool = True,
                 pssm_dir: Optional[str] = None,
                 esm_dir: Optional[str] = None,
                 feature_norm: str = 'zscore'):
        # Load the processed split and stash optional feature directories for lazy loading.
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.include_one_hot = include_one_hot
        self.pssm_dir = Path(pssm_dir) if pssm_dir else None
        self.esm_dir = Path(esm_dir) if esm_dir else None
        self.feature_norm = feature_norm
        self._feature_dim: Optional[int] = None
        self._missing_pssm: set[str] = set()
        self._missing_esm: set[str] = set()

    def __len__(self) -> int:
        # Allow len(dataset) to report how many chains survived cleaning.
        return len(self.df)

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            _ = self[0]
        assert self._feature_dim is not None
        return self._feature_dim

    def _trim(self,
              seq: str,
              labels3: str,
              labels8: Optional[str],
              features: Optional[np.ndarray]) -> Tuple[str, str, Optional[str], Optional[np.ndarray]]:
        if self.max_len is None or len(seq) <= self.max_len:
            return seq, labels3, labels8, features
        trimmed_seq = seq[:self.max_len]
        trimmed_labels3 = labels3[:self.max_len]
        trimmed_labels8 = labels8[:self.max_len] if labels8 is not None else None
        trimmed_features = features[:self.max_len] if features is not None else None
        return trimmed_seq, trimmed_labels3, trimmed_labels8, trimmed_features

    def _load_optional_matrix(self, directory: Optional[Path], entry_id: str, missing_cache: set[str]) -> Optional[np.ndarray]:
        if directory is None:
            return None
        path_npy = directory / f"{entry_id}.npy"
        path_ascii = directory / f"{entry_id}.pssm"
        for path in (path_npy, path_ascii):
            if path.exists():
                try:
                    # Accept either cached numpy arrays or raw PSI-BLAST ASCII dumps.
                    matrix = _load_pssm_matrix(path)
                    return matrix
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"Failed to load {path}: {exc}")
                    break
        if entry_id not in missing_cache:
            warnings.warn(f"Feature matrix missing for {entry_id} in {directory}")
            missing_cache.add(entry_id)
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row['seq']).strip().upper()
        sst3 = str(row['sst3']).strip().upper()
        sst8_raw = row.get('sst8')
        sst8 = str(sst8_raw).strip().upper() if isinstance(sst8_raw, str) else None
        entry_id = make_entry_id(row.get('pdb_id'), row.get('chain_code'))

        feature_blocks: List[np.ndarray] = []
        if self.include_one_hot:
            # Baseline categorical signal so the network always has residue identity.
            feature_blocks.append(_one_hot_encode(seq))

        pssm_matrix = self._load_optional_matrix(self.pssm_dir, entry_id, self._missing_pssm)
        if pssm_matrix is not None:
            # Evolutionary profile (log-odds + probabilities) after normalisation.
            feature_blocks.append(_normalise_matrix(pssm_matrix, mode=self.feature_norm))

        if self.esm_dir is not None:
            esm_path = self.esm_dir / f"{entry_id}.npy"
            if esm_path.exists():
                esm_matrix = np.load(esm_path).astype(np.float32)
                feature_blocks.append(esm_matrix)
            elif entry_id not in self._missing_esm:
                warnings.warn(f"ESM embedding missing for {entry_id} in {self.esm_dir}")
                self._missing_esm.add(entry_id)

        if not feature_blocks:
            raise ValueError("No residue-level features available. Enable one-hot or provide feature directories.")

        min_len = min(block.shape[0] for block in feature_blocks)
        if min_len <= 0:
            raise ValueError(f"Derived zero-length feature block for {entry_id}")
        if min_len != len(seq):
            # Align all sources to the shortest block to keep tensors consistent.
            seq = seq[:min_len]
            sst3 = sst3[:min_len]
            if sst8 is not None:
                sst8 = sst8[:min_len] if len(sst8) >= min_len else None
            feature_blocks = [block[:min_len] for block in feature_blocks]

        # Concatenate along the feature axis so each residue row stacks all modalities.
        features = np.concatenate(feature_blocks, axis=1)
        seq, sst3, sst8, features = self._trim(seq, sst3, sst8, features)

        label_idx = [SS3_MAP.get(ch, SS3_MAP['C']) for ch in sst3]
        label_idx_q8 = None
        if sst8 is not None and len(sst8) == len(seq):
            # Map each character label to its integer class ID for the auxiliary head.
            label_idx_q8 = [SS8_MAP.get(ch, SS8_MAP['C']) for ch in sst8]

        assert features is not None
        features_tensor = torch.from_numpy(features.astype(np.float32))

        item = {
            'id': entry_id,
            'features': features_tensor,
            'labels': torch.tensor(label_idx, dtype=torch.long),
            'length': features_tensor.size(0),
            'seq': seq,
            'sst3': sst3
        }
        if label_idx_q8 is not None:
            item['labels_q8'] = torch.tensor(label_idx_q8, dtype=torch.long)

        if self._feature_dim is None:
            self._feature_dim = features_tensor.size(1)
        return item


def collate_fn(batch):
    # Sort by descending length so packed sequences keep padding to a minimum.
    batch_sorted = sorted(batch, key=lambda x: x['length'], reverse=True)
    features_list = [b['features'] for b in batch_sorted]
    labels_list = [b['labels'] for b in batch_sorted]
    lengths = [b['length'] for b in batch_sorted]
    max_len = lengths[0]
    batch_size = len(batch_sorted)

    feature_dim = features_list[0].size(1)
    padded_features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (feat, lab, length) in enumerate(zip(features_list, labels_list, lengths)):
        # Copy only the valid prefix for each sequence and leave padding at ignore_index.
        padded_features[i, :length, :] = feat
        padded_labels[i, :length] = lab
        mask[i, :length] = True

    labels_q8_list = [b.get('labels_q8') for b in batch_sorted]
    padded_labels_q8 = None
    if all(l is not None for l in labels_q8_list):
        padded_labels_q8 = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for i, (label_q8, length) in enumerate(zip(labels_q8_list, lengths)):
            # Keep auxiliary labels aligned with the same padding mask as the primary task.
            padded_labels_q8[i, :length] = label_q8[:length]

    # Package everything into a dictionary so the trainer can unpack tensors by name.
    return {
        'ids': [b['id'] for b in batch_sorted],
        'features': padded_features,
        'labels': padded_labels,
        'labels_q8': padded_labels_q8,
        'mask': mask,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'seqs': [b['seq'] for b in batch_sorted],
        'sst3': [b['sst3'] for b in batch_sorted]
    }

def prepare_and_split(archive_dir: str,
                      out_dir: str = 'data/processed',
                      min_len: int = 20,
                      max_len: int = 2000,
                      val_frac: float = 0.1,
                      test_frac: float = 0.1,
                      drop_nonstd: bool = True,
                      pssm_dir: Optional[str] = None,
                      psiblast_cmd: str = 'psiblast',
                      blast_db: Optional[str] = None,
                      psiblast_iterations: int = DEFAULT_PSSM_ITERATIONS,
                      psiblast_evalue: float = DEFAULT_PSSM_EVALUE,
                      psiblast_threads: int = 4,
                      pssm_overwrite: bool = False,
                      esm_dir: Optional[str] = None,
                      esm_model: str = ESM_DEFAULT_MODEL,
                      esm_layer: Optional[int] = None,
                      esm_device: str = 'cuda',
                      esm_batch_tokens: int = 2048,
                      esm_overwrite: bool = False,
                      skip_split: bool = False) -> pd.DataFrame:
    # Student-friendly shortcut: run the entire preprocessing chain in one call.
    df = load_raw(archive_dir)
    df = clean_df(df, min_len=min_len, max_len=max_len, drop_nonstd=drop_nonstd)
    if not skip_split:
        # Write the processed CSVs so train/test scripts can read them later.
        split_save(df, out_dir, val_frac=val_frac, test_frac=test_frac)
    if pssm_dir:
        # Optionally spin up PSI-BLAST to fill the PSSM cache folder.
        generate_pssm_features(
            df,
            out_dir=pssm_dir,
            psiblast_cmd=psiblast_cmd,
            blast_db=blast_db,
            iterations=psiblast_iterations,
            evalue=psiblast_evalue,
            threads=psiblast_threads,
            overwrite=pssm_overwrite
        )
    if esm_dir:
        # Likewise, generate contextual embeddings if the user provided an output directory.
        generate_esm_embeddings(
            df,
            out_dir=esm_dir,
            model_name=esm_model,
            layer=esm_layer,
            device=esm_device,
            batch_tokens=esm_batch_tokens,
            overwrite=esm_overwrite
        )
    return df

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True, help='Path to archive folder containing the two CSVs')
    p.add_argument('--out', default='data/processed', help='Directory to store processed splits (train/val/test)')
    p.add_argument('--min_len', type=int, default=20)
    p.add_argument('--max_len', type=int, default=2000)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--keep_nonstd', action='store_true', help='Keep sequences marked with non-standard amino acids')
    p.add_argument('--pssm_dir', type=str, default=None, help='Output directory for cached PSSM matrices (.npy)')
    p.add_argument('--psiblast_cmd', type=str, default='psiblast', help='Path or alias for psiblast executable')
    p.add_argument('--blast_db', type=str, default=None, help='BLAST database name/path created via makeblastdb')
    p.add_argument('--psiblast_iterations', type=int, default=DEFAULT_PSSM_ITERATIONS)
    p.add_argument('--psiblast_evalue', type=float, default=DEFAULT_PSSM_EVALUE)
    p.add_argument('--psiblast_threads', type=int, default=4)
    p.add_argument('--pssm_overwrite', action='store_true', help='Regenerate PSSM files even if cached .npy exists')
    p.add_argument('--esm_dir', type=str, default=None, help='Output directory for per-residue ESM embeddings (.npy)')
    p.add_argument('--esm_model', type=str, default=ESM_DEFAULT_MODEL)
    p.add_argument('--esm_layer', type=int, default=None, help='Layer index to extract from the ESM model (defaults to top)')
    p.add_argument('--esm_device', type=str, default='cuda', help='Device for ESM inference (cuda|cpu)')
    p.add_argument('--esm_batch_tokens', type=int, default=2048, help='Max tokens per ESM forward pass (controls memory)')
    p.add_argument('--esm_overwrite', action='store_true', help='Regenerate ESM embeddings even if cached .npy exists')
    p.add_argument('--pssm_only', action='store_true', help='Skip CSV splitting and only generate auxiliary features')
    args = p.parse_args()
    prepare_and_split(
        args.archive,
        out_dir=args.out,
        min_len=args.min_len,
        max_len=args.max_len,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        drop_nonstd=not args.keep_nonstd,
        pssm_dir=args.pssm_dir,
        psiblast_cmd=args.psiblast_cmd,
        blast_db=args.blast_db,
        psiblast_iterations=args.psiblast_iterations,
        psiblast_evalue=args.psiblast_evalue,
        psiblast_threads=args.psiblast_threads,
        pssm_overwrite=args.pssm_overwrite,
        esm_dir=args.esm_dir,
        esm_model=args.esm_model,
        esm_layer=args.esm_layer,
        esm_device=args.esm_device,
        esm_batch_tokens=args.esm_batch_tokens,
        esm_overwrite=args.esm_overwrite,
        skip_split=args.pssm_only,
    )
