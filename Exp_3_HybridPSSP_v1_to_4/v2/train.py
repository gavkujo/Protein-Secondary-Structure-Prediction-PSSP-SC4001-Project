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
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
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
    # Respect the user's choice when possible, otherwise back off to the best available accelerator.
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
    """Infer per-residue feature dimensionality from the first available .npy file."""
    # The feature directory may be absent for sequence-only experiments.
    if dataset.features_dir is None:
        return None

    df = getattr(dataset, 'df', None)
    if df is None or df.empty:
        return None

    for _, row in df.iterrows():
        # Per-sequence feature files follow the {pdb}_{chain}.npy naming convention.
        sid = f"{row.get('pdb_id')}_{row.get('chain_code')}"
        feat_path = os.path.join(dataset.features_dir, f"{sid}.npy")
        if os.path.exists(feat_path):
            arr = np.load(feat_path, mmap_mode='r')
            if arr.ndim == 1:
                return int(arr.shape[0])
            return int(arr.shape[-1])
    return None

def train_epoch(model, loader, criterion, optimizer, device, scaler, scheduler=None,
                clip_norm: Optional[float] = None, amp_enabled: bool = True):
    model.train()
    # Track epoch-level loss weighted by sequence length to keep logs comparable across batch sizes.
    total_loss = 0.0
    dataset_size = max(len(loader.dataset), 1)

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        # Each batch dictionary already contains padded tensors; we just move them onto the device.
        seqs = batch['seqs'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        lengths = batch['lengths'].to(device, non_blocking=True)
        features = batch['features']
        if features is not None:
            features = features.to(device, non_blocking=True)

        # Clearing gradients before the forward pass is essential when using gradient scaling.
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(seqs, lengths, features)
            # Flatten so CrossEntropyLoss can treat residues as independent classification targets.
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)

        # AMP recipe: scale gradients, unscale for clipping, then step and update the scaler state.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if clip_norm is not None and clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * seqs.size(0)

        if step == 1:
            # Heuristic check so we notice abnormal padding lengths right away.
            _log(f"First batch processed: batch_size={seqs.size(0)}, max_len={seqs.size(1)}")

    return total_loss / dataset_size

def evaluate(model, loader, criterion, device, amp_enabled: bool = True):
    model.eval()
    # Collect predictions for macro-F1 while accumulating the average loss for reporting.
    total_loss = 0.0
    all_preds = []
    all_labels = []
    dataset_size = max(len(loader.dataset), 1)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            # Keep the lengths on CPU so slicing remains inexpensive later on.
            seqs = batch['seqs'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            lengths_cpu = batch['lengths']
            lengths = lengths_cpu.to(device, non_blocking=True)
            features = batch['features']
            if features is not None:
                features = features.to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(seqs, lengths, features)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(logits_flat, labels_flat)

            total_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=-1)
            for idx, seq_len in enumerate(lengths_cpu.tolist()):
                # Trim to the true sequence length to avoid counting padded residues in the metric.
                valid_labels = labels[idx, :seq_len].detach().cpu().numpy()
                valid_preds = preds[idx, :seq_len].detach().cpu().numpy()
                all_labels.extend(valid_labels)
                all_preds.extend(valid_preds)

    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / dataset_size, f1

def main(args):
    device = _resolve_device(args.device)
    _log(f"Using device: {device}")

    features_dir = args.features_dir if args.use_features else None

    # Instantiate datasets for the train/validation splits produced during preprocessing.
    train_dataset = ProteinDataset(
        os.path.join(args.data_dir, 'train.csv'),
        features_dir=features_dir,
        max_len=args.max_len
    )
    val_dataset = ProteinDataset(
        os.path.join(args.data_dir, 'val.csv'),
        features_dir=features_dir,
        max_len=args.max_len
    )
    _log(f"Loaded datasets | train={len(train_dataset)} | val={len(val_dataset)}")
    pin_memory = device.type in ('cuda', 'mps')
    # Persistent workers keep the CSV open and reduce reloading overhead during long runs.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers,
                            pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    _log(f"Batches per epoch | train={len(train_loader)} | val={len(val_loader)}")

    feat_dim = args.feat_dim
    if args.use_features:
        if args.features_dir is None:
            raise ValueError("--use_features requires --features_dir to be specified")
        if feat_dim is None:
            feat_dim = _infer_feature_dim(train_dataset)
        if feat_dim is None:
            raise ValueError("Could not infer feature dimension; please provide --feat_dim")
        # Explicit logging helps when we compare runs that toggle handcrafted features on/off.
        _log(f"Using residue features | dir={args.features_dir} | feat_dim={feat_dim}")

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
    _log("Model initialised and moved to device")

    amp_enabled = device.type in ('cuda', 'mps') and not args.disable_amp
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=amp_enabled)
    _log(f"AMP enabled: {amp_enabled}")

    scheduler = None
    if args.scheduler == 'onecycle':
        if len(train_loader) == 0:
            _log("Skipping OneCycleLR scheduler because the training loader is empty")
        else:
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                   steps_per_epoch=len(train_loader), pct_start=args.warmup_pct,
                                   anneal_strategy='cos', div_factor=args.div_factor,
                                   final_div_factor=args.final_div_factor)
            _log("Using OneCycleLR scheduler")
    if args.clip_grad_norm and args.clip_grad_norm > 0:
        # Gradient clipping is especially helpful when very long chains slip through a batch.
        _log(f"Gradient clipping set to {args.clip_grad_norm}")

    _log("Starting training")

    best_f1 = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            scheduler=scheduler,
            clip_norm=args.clip_grad_norm,
            amp_enabled=amp_enabled,
        )
        val_loss, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
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
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, choices=['none', 'onecycle'], default='onecycle')
    parser.add_argument('--warmup_pct', type=float, default=0.1)
    parser.add_argument('--div_factor', type=float, default=25.0)
    parser.add_argument('--final_div_factor', type=float, default=1e4)
    parser.add_argument('--disable_amp', action='store_true')
    args = parser.parse_args()
    main(args)
