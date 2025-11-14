"""
data_prep.py
--------------
Data loading and cleaning utilities.
Functions:
- load_raw: Load raw CSVs and merge/filter them
- clean_df: Clean dataframe (drop nonstandard AA, normalize labels, length checks)
- split_save: Split dataframe into train/val/test and save to CSVs
Classes:
- ProteinDataset: PyTorch Dataset for processed CSVs with optional per-residue features
- collate_fn: Collate function for DataLoader to pad sequences and features
"""

import os
import csv
import random
from typing import List, Tuple, Optional

import pandas as pd

RANDOM_SEED = 42

Q8_TO_Q3 = {
    'H': 'H', 'G': 'H', 'I': 'H',  # helix types -> H
    'E': 'E', 'B': 'E',            # strand types -> E
    'T': 'C', 'S': 'C', 'C': 'C'   # turn/bend/coil -> C
}


def _read_csv_guess(path: str) -> pd.DataFrame:
    # The archive is not consistent about delimiters, so we try CSV before settling on TSV.
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception:
        # fallback to reading as tab-separated
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
    df = df.dropna(subset=['seq'])
    if 'sst3' not in df.columns and 'sst8' not in df.columns:
        raise ValueError('Neither sst3 nor sst8 present in the dataframe')

    # If only sst8 present, create sst3 mapping column
    if 'sst3' not in df.columns and 'sst8' in df.columns:
        def map_q8_to_q3(s8: str) -> str:
            return ''.join([Q8_TO_Q3.get(ch, 'C') for ch in s8])
        df['sst3'] = df['sst8'].astype(str).apply(map_q8_to_q3)

    # standardize has_nonstd_aa column
    if 'has_nonstd_aa' in df.columns:
        df['has_nonstd_aa'] = df['has_nonstd_aa'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
    else:
        # infer from seq presence of nonstandard letters B,O,U,X,Z or '*' masking
        df['has_nonstd_aa'] = df['seq'].apply(lambda s: any(ch in set('B O U X Z *'.split()) for ch in s))

    if drop_nonstd:
        # Removing non-standard residues simplifies the label space for this baseline model.
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
    df.loc[train_idx].to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    df.loc[val_idx].to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    df.loc[test_idx].to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    print(f"Saved splits to {out_dir}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# Lightweight Dataset class that will be imported by training script
from torch.utils.data import Dataset
import torch

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # canonical 20
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}  # 0 reserved for PAD
PAD_IDX = 0
SS3_MAP = {'H': 0, 'E': 1, 'C': 2}

class ProteinDataset(Dataset):
    """Simple dataset reading the processed CSVs. Optionally supports features_dir for per-sequence .npy features.
    Each row must contain: pdb_id, chain_code, seq, sst3, (optionally sst8, len)
    """
    def __init__(self, csv_path: str, features_dir: Optional[str] = None, max_len: Optional[int] = None):
        import pandas as pd
        # The CSV is tiny enough for eager loading; workers will inherit the cached dataframe.
        self.df = pd.read_csv(csv_path)
        self.features_dir = features_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row['seq']).strip()
        sst3 = str(row['sst3']).strip()
        seq_idx = [AA_TO_IDX.get(ch, PAD_IDX) for ch in seq]
        label_idx = [SS3_MAP.get(ch, 2) for ch in sst3]
        features = None
        if self.features_dir is not None:
            # Features are optional here; if provided we load them lazily and cast to float32 once.
            sid = f"{row.get('pdb_id')}_{row.get('chain_code')}"
            feat_path = os.path.join(self.features_dir, f"{sid}.npy")
            if os.path.exists(feat_path):
                features = torch.from_numpy(__import__('numpy').load(feat_path)).float()
        return {
            'id': f"{row.get('pdb_id')}_{row.get('chain_code')}",
            'seq_idx': torch.tensor(seq_idx, dtype=torch.long),
            'labels': torch.tensor(label_idx, dtype=torch.long),
            'length': len(seq_idx),
            'features': features
        }

def collate_fn(batch):
    # Sort by descending length so packed sequences remain valid downstream.
    batch_sorted = sorted(batch, key=lambda x: x['length'], reverse=True)
    seqs = [b['seq_idx'] for b in batch_sorted]
    labels = [b['labels'] for b in batch_sorted]
    lengths = [len(s) for s in seqs]
    max_len = lengths[0]

    padded_seqs = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
    padded_labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)

    for i, (s, l) in enumerate(zip(seqs, lengths)):
        padded_seqs[i, :l] = s
        padded_labels[i, :l] = labels[i]
        mask[i, :l] = 1

    # features
    features_list = [b['features'] for b in batch_sorted]
    features_tensor = None
    available_feats = [f for f in features_list if f is not None]
    if available_feats:
        # Allocate a dense tensor so the training loop can call to(device) exactly once.
        feat_dim = available_feats[0].shape[1]
        feats = torch.zeros((len(seqs), max_len, feat_dim), dtype=torch.float)
        for i, f in enumerate(features_list):
            if f is None:
                continue
            L = f.shape[0]
            feats[i, :L, :] = f
        features_tensor = feats

    return {
        'ids': [b['id'] for b in batch_sorted],
        'seqs': padded_seqs,
        'labels': padded_labels,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'mask': mask,
        'features': features_tensor
    }

def prepare_and_split(archive_dir: str, out_dir: str = 'data/processed', min_len: int = 20, max_len: int = 2000,
                      val_frac: float = 0.1, test_frac: float = 0.1, drop_nonstd: bool = True):
    # Wrapper used in early experiments to regenerate splits with minimal boilerplate.
    df = load_raw(archive_dir)
    df = clean_df(df, min_len=min_len, max_len=max_len, drop_nonstd=drop_nonstd)
    split_save(df, out_dir, val_frac=val_frac, test_frac=test_frac)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True, help='Path to archive folder containing the two CSVs')
    p.add_argument('--out', default='data/processed')
    args = p.parse_args()
    prepare_and_split(args.archive, out_dir=args.out)
