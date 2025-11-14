"""
train.py
--------
Training script for BiLSTM protein secondary structure prediction.
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from dataprep import ProteinDataset, collate_fn
from model import BiLSTMProtein

import argparse
import numpy as np


def _log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def _resolve_device(requested: str) -> torch.device:
    # Interpret the device flag once, then fall back to the best accelerator the machine offers.
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # Record the running sum of losses weighted by sequence count for consistent epoch logs.
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        # The collate function pads tensors; here we only move them onto the active device.
        seqs = batch['seqs'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)
        optimizer.zero_grad()
        logits = model(seqs, lengths, batch['features'].to(device) if batch['features'] is not None else None)
        # Flatten makes CrossEntropy compare residues individually instead of whole sequences.
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
        if step == 1:
            # Early log so we can detect suspicious padding lengths during debugging sessions.
            _log(f"First batch processed: batch_size={seqs.size(0)}, max_len={seqs.size(1)}")
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    # Gather predictions for macro-F1 and keep a mean loss for the validation summary.
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            # Maintain a CPU copy of lengths so slicing away padding stays cheap.
            seqs = batch['seqs'].to(device)
            labels = batch['labels'].to(device)
            lengths_cpu = batch['lengths']
            lengths = lengths_cpu.to(device)
            logits = model(seqs, lengths, batch['features'].to(device) if batch['features'] is not None else None)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=-1)
            for idx, seq_len in enumerate(lengths_cpu.tolist()):
                # Truncate to the true chain length so padding tokens never influence the metric.
                valid_labels = labels[idx, :seq_len].detach().cpu().numpy()
                valid_preds = preds[idx, :seq_len].detach().cpu().numpy()
                all_labels.extend(valid_labels)
                all_preds.extend(valid_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader.dataset), f1

def main(args):
    device = _resolve_device(args.device)
    _log(f"Using device: {device}")

    # Base version relies purely on sequence CSVs with no auxiliary feature directory.
    train_dataset = ProteinDataset(os.path.join(args.data_dir, 'train.csv'))
    val_dataset = ProteinDataset(os.path.join(args.data_dir, 'val.csv'))
    _log(f"Loaded datasets | train={len(train_dataset)} | val={len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)
    _log(f"Batches per epoch | train={len(train_loader)} | val={len(val_loader)}")

    model = BiLSTMProtein(embed_dim=args.embed_dim,
                          hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers,
                          dropout=args.dropout).to(device)
    _log("Model initialised and moved to device")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = Adam(model.parameters(), lr=args.lr)

    _log("Starting training")

    best_f1 = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        _log(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_f1={val_f1:.4f}")

        # Save checkpoint if improved
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.ckpt_dir, f'best_model.pt')
            torch.save(model.state_dict(), ckpt_path)
            _log(f"New best model saved to {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
