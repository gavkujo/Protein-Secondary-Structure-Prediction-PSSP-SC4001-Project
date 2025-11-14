"""
test.py
-------
Evaluate a trained BiLSTM protein secondary structure model on the test set.
"""

import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from dataprep import ProteinDataset, collate_fn
from model import BiLSTMProtein

import argparse
import numpy as np
from typing import Optional


def _log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def _resolve_device(requested: str) -> torch.device:
    # Honour the preferred accelerator when present; otherwise fall back systematically.
    req = (requested or '').lower()
    if req == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if req in ('mps', 'metal') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if req == 'cpu' or not req:
        return torch.device('cpu')
    if torch.cuda.is_available():
        print(f"Requested device '{requested}' not available; using CUDA instead.")
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"Requested device '{requested}' not available; using MPS instead.")
        return torch.device('mps')
    print(f"Requested device '{requested}' not available; using CPU instead.")
    return torch.device('cpu')


def _infer_feature_dim(dataset: ProteinDataset) -> Optional[int]:
    # Auxiliary features are stored per chain, so we only need to inspect one file to get the width.
    if dataset.features_dir is None:
        return None
    df = getattr(dataset, 'df', None)
    if df is None or df.empty:
        return None
    for _, row in df.iterrows():
        sid = f"{row.get('pdb_id')}_{row.get('chain_code')}"
        feat_path = os.path.join(dataset.features_dir, f"{sid}.npy")
        if os.path.exists(feat_path):
            arr = np.load(feat_path, mmap_mode='r')
            if arr.ndim == 1:
                return int(arr.shape[0])
            return int(arr.shape[-1])
    return None

def evaluate(model, loader, device):
    model.eval()
    # Flattened predictions and labels feed into macro-F1 and confusion matrices later on.
    all_preds = []
    all_labels = []
    ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            # Keep a CPU copy of lengths so we can slice away pad positions for metrics.
            seqs = batch['seqs'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            lengths_cpu = batch['lengths']
            lengths = lengths_cpu.to(device, non_blocking=True)
            features = batch['features']
            if features is not None:
                features = features.to(device, non_blocking=True)
            logits = model(seqs, lengths, features)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            preds = logits.argmax(dim=-1)
            for idx, seq_len in enumerate(lengths_cpu.tolist()):
                # Slice to the true length so evaluation ignores zero-padding artefacts.
                valid_labels = labels[idx, :seq_len].detach().cpu().numpy()
                valid_preds = preds[idx, :seq_len].detach().cpu().numpy()
                all_labels.extend(valid_labels)
                all_preds.extend(valid_preds)
                ids.extend([batch['ids'][idx]] * seq_len)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return all_labels, all_preds, ids, f1

def main(args):
    device = _resolve_device(args.device)
    _log(f"Using device: {device}")

    features_dir = args.features_dir if args.use_features else None
    # Reuse the processed CSV split prepared during training to guarantee the same vocabulary.
    test_dataset = ProteinDataset(
        os.path.join(args.data_dir, 'test.csv'),
        features_dir=features_dir,
        max_len=args.max_len
    )
    _log(f"Loaded test dataset | samples={len(test_dataset)}")
    pin_memory = device.type in ('cuda', 'mps')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers,
                             pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    _log(f"Test batches={len(test_loader)} | batch_size={args.batch_size}")

    feat_dim = args.feat_dim
    if args.use_features:
        if args.features_dir is None:
            raise ValueError("--use_features requires --features_dir")
        if feat_dim is None:
            feat_dim = _infer_feature_dim(test_dataset)
        if feat_dim is None:
            raise ValueError("Could not infer feature dimension; provide --feat_dim")
        # Logging the inferred dimension is useful when we audit evaluation settings later on.
        _log(f"Using residue features | dir={args.features_dir} | feat_dim={feat_dim}")

    # Create model (same hyperparameters as training)
    model = BiLSTMProtein(embed_dim=args.embed_dim,
                          hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          use_features=args.use_features,
                          feat_dim=feat_dim,
                          use_attention=not args.no_attention,
                          attention_heads=args.attention_heads,
                          transformer_layers=args.transformer_layers,
                          transformer_dropout=args.transformer_dropout,
                          transformer_ff_dim=args.transformer_ff_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    _log(f"Loaded model from {args.ckpt}")

    all_labels, all_preds, ids, f1 = evaluate(model, test_loader, device)
    _log(f"Test macro-F1: {f1:.4f}")
    _log("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    _log("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--use_features', action='store_true')
    parser.add_argument('--features_dir', type=str, default=None)
    parser.add_argument('--feat_dim', type=int, default=None)
    parser.add_argument('--attention_heads', type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--transformer_dropout', type=float, default=0.2)
    parser.add_argument('--transformer_ff_dim', type=int, default=256)
    parser.add_argument('--no_attention', action='store_true')
    args = parser.parse_args()
    main(args)
