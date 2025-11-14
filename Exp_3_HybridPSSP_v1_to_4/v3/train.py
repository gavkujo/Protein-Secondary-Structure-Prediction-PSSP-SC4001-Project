"""
train.py
--------
Training script for Hybrid CNN + Transformer protein secondary structure prediction.
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

from dataprep import ProteinDataset, collate_fn, SS8_MAP, NUM_Q8_CLASSES
from model import ProteinHybridModel

import argparse
import numpy as np
from typing import Optional


def _log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def _resolve_device(requested: str) -> torch.device:
    # Interpret the CLI flag once, then pick the best available accelerator in a predictable order.
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
    # Bail out early if the dataset was constructed without auxiliary features.
    if dataset.features_dir is None:
        return None

    df = getattr(dataset, 'df', None)
    if df is None or df.empty:
        return None

    for _, row in df.iterrows():
        # We follow the naming convention used during feature export.
        sid = f"{row.get('pdb_id')}_{row.get('chain_code')}"
        feat_path = os.path.join(dataset.features_dir, f"{sid}.npy")
        if os.path.exists(feat_path):
            arr = np.load(feat_path, mmap_mode='r')
            if arr.ndim == 1:
                return int(arr.shape[0])
            return int(arr.shape[-1])
    return None


def _compute_q8_class_weights(dataset: ProteinDataset, mode: str, beta: float) -> Optional[torch.Tensor]:
    """Derive class weights for Q8 labels to counter severe imbalance."""
    # The raw dataframe stores labels as concatenated strings; we expand them into counts here.
    df = getattr(dataset, 'df', None)
    if df is None or 'sst8' not in df.columns:
        return None

    counts = np.zeros(NUM_Q8_CLASSES, dtype=np.float64)
    for raw in df['sst8'].dropna():
        label = str(raw).strip()
        if not label:
            continue
        for ch in label:
            idx = SS8_MAP.get(ch)
            if idx is not None:
                counts[idx] += 1

    total = counts.sum()
    if total == 0:
        return None

    counts_safe = np.where(counts > 0, counts, 1.0)
    # Two common heuristics: plain inverse-frequency and "effective number" (focal) weighting.
    if mode == 'inverse':
        weights = total / counts_safe
    elif mode == 'effective':
        beta = float(min(max(beta, 0.9), 0.9999))
        effective_num = 1.0 - np.power(beta, counts_safe)
        effective_num = np.where(effective_num > 0, effective_num, 1e-8)
        weights = (1.0 - beta) / effective_num
    else:
        return None

    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def build_model(args, feat_dim: Optional[int], device: torch.device) -> nn.Module:
    # Allow the user to toggle the transformer stack entirely for ablation purposes.
    transformer_layers = 0 if args.no_attention else args.transformer_layers
    model = ProteinHybridModel(
        embed_dim=args.embed_dim,
        model_dim=args.model_dim,
        conv_layers=args.conv_layers,
        conv_kernel=args.conv_kernel,
        conv_dropout=args.conv_dropout,
        dropout=args.dropout,
        use_features=args.use_features,
        feat_dim=feat_dim,
        attention_heads=args.attention_heads,
        transformer_layers=transformer_layers,
        transformer_dropout=args.transformer_dropout,
        transformer_ff_dim=args.transformer_ff_dim,
        head_dropout=args.head_dropout,
        q8_head_dim=args.q8_head_dim,
        q8_head_layers=args.q8_head_layers,
    )
    return model.to(device)

def train_epoch(model,
                loader,
                criterion_q3,
                optimizer,
                device,
                scaler,
                scheduler=None,
                clip_norm: Optional[float] = None,
                amp_enabled: bool = True,
                criterion_q8: Optional[nn.Module] = None,
                q8_weight: float = 0.0):
    model.train()
    # We aggregate losses in absolute terms so the epoch summary scales with the number of residues.
    total_loss = 0.0
    total_loss_q3 = 0.0
    total_loss_q8 = 0.0
    dataset_size = max(len(loader.dataset), 1)
    use_q8 = criterion_q8 is not None and q8_weight > 0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        # The collate_fn already pads, so we only need to move tensors to the active device.
        seqs = batch['seqs'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        lengths = batch['lengths'].to(device, non_blocking=True)
        features = batch['features']
        if features is not None:
            features = features.to(device, non_blocking=True)
        labels_q8 = batch.get('labels_q8')
        if use_q8 and labels_q8 is not None:
            labels_q8 = labels_q8.to(device, non_blocking=True)
        else:
            labels_q8 = None

        # Clearing grads before the autocast context keeps buffers fresh when mixed precision is on.
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            # The model can return a tensor (Q3 only) or a dict with both heads.
            outputs = model(seqs, lengths, features)
            if isinstance(outputs, torch.Tensor):
                logits_q3 = outputs
                logits_q8 = None
            else:
                logits_q3 = outputs['q3']
                logits_q8 = outputs.get('q8')

            # Flatten the time dimension so CrossEntropyLoss can compare class logits per residue.
            logits_q3_flat = logits_q3.view(-1, logits_q3.size(-1))
            labels_flat = labels.view(-1)
            loss_q3 = criterion_q3(logits_q3_flat, labels_flat)
            loss = loss_q3

            loss_q8 = None
            if use_q8 and logits_q8 is not None and labels_q8 is not None:
                if criterion_q8 is None:
                    raise RuntimeError("criterion_q8 is required when using Q8 supervision")
                logits_q8_flat = logits_q8.view(-1, logits_q8.size(-1))
                labels_q8_flat = labels_q8.view(-1)
                loss_q8 = criterion_q8(logits_q8_flat, labels_q8_flat)
                # The auxiliary head is scaled so we can explore different trade-offs during tuning.
                loss = loss + q8_weight * loss_q8

        # Standard AMP recipe: scale, backprop, unscale, clip (optional), optimise, then update scaler.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if clip_norm is not None and clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * seqs.size(0)
        total_loss_q3 += loss_q3.item() * seqs.size(0)
        if loss_q8 is not None:
            total_loss_q8 += loss_q8.item() * seqs.size(0)

        if step == 1:
            # Quick sanity log so we notice if padding lengths look off early in training.
            _log(f"First batch processed: batch_size={seqs.size(0)}, max_len={seqs.size(1)}")

    avg_loss = total_loss / dataset_size
    avg_loss_q3 = total_loss_q3 / dataset_size
    avg_loss_q8 = (total_loss_q8 / dataset_size) if total_loss_q8 > 0 else None
    return avg_loss, avg_loss_q3, avg_loss_q8

def evaluate(model,
             loader,
             criterion_q3,
             device,
             amp_enabled: bool = True,
             criterion_q8: Optional[nn.Module] = None,
             q8_weight: float = 0.0):
    model.eval()
    # Accumulate per-head losses and labels to compute macro-F1 outside torch.no_grad().
    total_loss = 0.0
    total_loss_q3 = 0.0
    total_loss_q8 = 0.0
    all_preds_q3 = []
    all_labels_q3 = []
    all_preds_q8 = []
    all_labels_q8 = []
    dataset_size = max(len(loader.dataset), 1)
    use_q8 = criterion_q8 is not None and q8_weight > 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            # Same data movement pattern as training, minus gradient tracking.
            seqs = batch['seqs'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            lengths_cpu = batch['lengths']
            lengths = lengths_cpu.to(device, non_blocking=True)
            features = batch['features']
            if features is not None:
                features = features.to(device, non_blocking=True)
            labels_q8 = batch.get('labels_q8')
            if use_q8 and labels_q8 is not None:
                labels_q8 = labels_q8.to(device, non_blocking=True)
            else:
                labels_q8 = None

            with autocast(enabled=amp_enabled):
                outputs = model(seqs, lengths, features)
                if isinstance(outputs, torch.Tensor):
                    logits_q3 = outputs
                    logits_q8 = None
                else:
                    logits_q3 = outputs['q3']
                    logits_q8 = outputs.get('q8')

                logits_q3_flat = logits_q3.view(-1, logits_q3.size(-1))
                labels_flat = labels.view(-1)
                loss_q3 = criterion_q3(logits_q3_flat, labels_flat)
                loss = loss_q3

                loss_q8 = None
                if use_q8 and logits_q8 is not None and labels_q8 is not None:
                    if criterion_q8 is None:
                        raise RuntimeError("criterion_q8 is required when using Q8 supervision")
                    logits_q8_flat = logits_q8.view(-1, logits_q8.size(-1))
                    labels_q8_flat = labels_q8.view(-1)
                    loss_q8 = criterion_q8(logits_q8_flat, labels_q8_flat)
                    loss = loss + q8_weight * loss_q8

            total_loss += loss.item() * seqs.size(0)
            total_loss_q3 += loss_q3.item() * seqs.size(0)
            if loss_q8 is not None:
                total_loss_q8 += loss_q8.item() * seqs.size(0)

            preds_q3 = logits_q3.argmax(dim=-1)
            preds_q3_cpu = preds_q3.detach().cpu()
            labels_cpu = labels.detach().cpu()
            if labels_q8 is not None and logits_q8 is not None:
                preds_q8 = logits_q8.argmax(dim=-1)
                preds_q8_cpu = preds_q8.detach().cpu()
                labels_q8_cpu = labels_q8.detach().cpu()
            else:
                preds_q8_cpu = None
                labels_q8_cpu = None

            for idx, seq_len in enumerate(lengths_cpu.tolist()):
                # Slice away pad positions so the metrics reflect real residues only.
                valid_labels = labels_cpu[idx, :seq_len].numpy()
                valid_preds = preds_q3_cpu[idx, :seq_len].numpy()
                all_labels_q3.extend(valid_labels)
                all_preds_q3.extend(valid_preds)

                if labels_q8_cpu is not None and preds_q8_cpu is not None:
                    valid_labels_q8 = labels_q8_cpu[idx, :seq_len].numpy()
                    valid_preds_q8 = preds_q8_cpu[idx, :seq_len].numpy()
                    all_labels_q8.extend(valid_labels_q8)
                    all_preds_q8.extend(valid_preds_q8)

    avg_loss = total_loss / dataset_size
    avg_loss_q3 = total_loss_q3 / dataset_size
    avg_loss_q8 = (total_loss_q8 / dataset_size) if total_loss_q8 > 0 else None
    f1_q3 = f1_score(all_labels_q3, all_preds_q3, average='macro') if all_labels_q3 else 0.0
    f1_q8 = f1_score(all_labels_q8, all_preds_q8, average='macro') if all_labels_q8 else None
    return avg_loss, avg_loss_q3, avg_loss_q8, f1_q3, f1_q8

def main(args):
    device = _resolve_device(args.device)
    _log(f"Using device: {device}")

    features_dir = args.features_dir if args.use_features else None

    # Construct the split datasets; max_len trims at load time when requested.
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
    # Pin memory and persistent workers help when shuttling large batches to GPU repeatedly.
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
        # Logging dimensions early helps catch mismatches between exported and expected feature shapes.
        _log(f"Using residue features | dir={args.features_dir} | feat_dim={feat_dim}")

    q8_class_weights = None
    if not args.disable_q8 and args.q8_balance != 'none':
        q8_class_weights = _compute_q8_class_weights(train_dataset, args.q8_balance, args.q8_balance_beta)
        if q8_class_weights is not None:
            weights_preview = ", ".join(f"{w:.2f}" for w in q8_class_weights.tolist())
            _log(f"Q8 class weights ({args.q8_balance}): [{weights_preview}]")
        else:
            _log("Q8 class balancing requested but Q8 labels were unavailable; falling back to uniform weights")

    model = build_model(args, feat_dim, device)
    _log("Hybrid model initialised and moved to device")

    amp_enabled = device.type in ('cuda', 'mps') and not args.disable_amp
    criterion_q3 = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing).to(device)
    criterion_q8: Optional[nn.Module] = None
    q8_weight = 0.0
    if not args.disable_q8:
        requested_weight = max(args.q8_weight, 0.0)
        if requested_weight > 0:
            weight_tensor = q8_class_weights.to(device) if q8_class_weights is not None else None
            criterion_q8 = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=args.label_smoothing,
                weight=weight_tensor
            ).to(device)
            # Balance mode is reported for the lab book so we can justify metric changes later.
            balance_mode = args.q8_balance if q8_class_weights is not None else 'uniform'
            _log(f"Q8 supervision enabled | weight={requested_weight:.3f} | balance={balance_mode}")
            if q8_class_weights is None and args.q8_balance != 'none':
                _log("Requested Q8 balancing but valid counts were not found; using uniform weighting")
            q8_weight = requested_weight
        else:
            _log("Q8 weight is non-positive; the auxiliary head will not influence the loss")
    else:
        _log("Q8 supervision disabled")

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
        # Gradient clipping prevents exploding updates on very long chains.
        _log(f"Gradient clipping set to {args.clip_grad_norm}")

    _log("Starting training")

    best_f1_q3 = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss_total, train_loss_q3, train_loss_q8 = train_epoch(
            model,
            train_loader,
            criterion_q3,
            optimizer,
            device,
            scaler,
            scheduler=scheduler,
            clip_norm=args.clip_grad_norm,
            amp_enabled=amp_enabled,
            criterion_q8=criterion_q8,
            q8_weight=q8_weight,
        )
        (val_loss_total,
         val_loss_q3,
         val_loss_q8,
         val_f1_q3,
         val_f1_q8) = evaluate(
            model,
            val_loader,
            criterion_q3,
            device,
            amp_enabled=amp_enabled,
            criterion_q8=criterion_q8,
            q8_weight=q8_weight,
        )
        log_parts = [
            f"Epoch {epoch:02d}",
            f"train_total={train_loss_total:.4f}",
            f"train_q3={train_loss_q3:.4f}"
        ]
        if train_loss_q8 is not None:
            log_parts.append(f"train_q8={train_loss_q8:.4f}")
        log_parts.extend([
            f"val_total={val_loss_total:.4f}",
            f"val_q3={val_loss_q3:.4f}",
            f"val_f1_q3={val_f1_q3:.4f}"
        ])
        if val_loss_q8 is not None:
            log_parts.insert(len(log_parts) - 1, f"val_q8={val_loss_q8:.4f}")
        if val_f1_q8 is not None:
            log_parts.append(f"val_f1_q8={val_f1_q8:.4f}")
        _log(" | ".join(log_parts))

        # Save checkpoint if improved
        if val_f1_q3 > best_f1_q3:
            best_f1_q3 = val_f1_q3
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
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--conv_layers', type=int, default=4)
    parser.add_argument('--conv_kernel', type=int, default=9)
    parser.add_argument('--conv_dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--head_dropout', type=float, default=0.15)
    parser.add_argument('--q8_head_dim', type=int, default=512)
    parser.add_argument('--q8_head_layers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--use_features', action='store_true')
    parser.add_argument('--features_dir', type=str, default=None)
    parser.add_argument('--feat_dim', type=int, default=None)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--transformer_dropout', type=float, default=0.2)
    parser.add_argument('--transformer_ff_dim', type=int, default=512)
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, choices=['none', 'onecycle'], default='onecycle')
    parser.add_argument('--warmup_pct', type=float, default=0.1)
    parser.add_argument('--div_factor', type=float, default=25.0)
    parser.add_argument('--final_div_factor', type=float, default=1e4)
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--disable_q8', action='store_true')
    parser.add_argument('--q8_weight', type=float, default=0.6)
    parser.add_argument('--q8_balance', choices=['none', 'inverse', 'effective'], default='effective')
    parser.add_argument('--q8_balance_beta', type=float, default=0.999)
    args = parser.parse_args()
    if getattr(args, 'hidden_dim', None) is not None:
        args.model_dim = args.hidden_dim
    main(args)
