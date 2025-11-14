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


def _log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def _resolve_device(requested: str) -> torch.device:
    # Prefer the requested accelerator when present; otherwise drop to the next viable option.
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

def evaluate(model, loader, device):
    model.eval()
    # Accumulate predictions and labels at residue resolution for macro-F1 and confusion matrices.
    all_preds = []
    all_labels = []
    ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            # Keep lengths on CPU so trimming away padding is straightforward later on.
            seqs = batch['seqs'].to(device)
            labels = batch['labels'].to(device)
            lengths_cpu = batch['lengths']
            lengths = lengths_cpu.to(device)
            features = batch['features']
            if features is not None:
                features = features.to(device)
            logits = model(seqs, lengths, features)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            preds = logits.argmax(dim=-1)
            for idx, seq_len in enumerate(lengths_cpu.tolist()):
                # Limit evaluation to real residues and replicate IDs for optional error analysis.
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

    # Evaluation reuses the processed CSVs produced during training to stay in sync with vocab indices.
    test_dataset = ProteinDataset(os.path.join(args.data_dir, 'test.csv'))
    _log(f"Loaded test dataset | samples={len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn)
    _log(f"Test batches={len(test_loader)} | batch_size={args.batch_size}")

    # Create model (same hyperparameters as training)
    model = BiLSTMProtein(embed_dim=args.embed_dim,
                          hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers,
                          dropout=args.dropout).to(device)
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
    args = parser.parse_args()
    main(args)
