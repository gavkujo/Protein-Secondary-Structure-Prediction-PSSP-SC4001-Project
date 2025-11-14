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
from model import HybridPSSP

import argparse
import numpy as np
from typing import Optional, Sequence


def _log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def _resolve_device(requested: str) -> torch.device:
    req = (requested or '').lower()
    if req == 'cuda' and torch.cuda.is_available():
        # Preferred path: let the model stretch its legs on a proper GPU.
        return torch.device('cuda')
    if req in ('mps', 'metal') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon fallback so the script still benefits from hardware acceleration.
        return torch.device('mps')
    if req == 'cpu' or not req:
        # Explicit CPU request or nothing specified: keep things simple.
        return torch.device('cpu')
    if torch.cuda.is_available():
        print(f"Requested device '{requested}' not available; using CUDA instead.")
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"Requested device '{requested}' not available; using MPS instead.")
        return torch.device('mps')
    print(f"Requested device '{requested}' not available; using CPU instead.")
    return torch.device('cpu')


def _compute_q8_class_weights(dataset: ProteinDataset, mode: str, beta: float) -> Optional[torch.Tensor]:
    """Derive class weights for Q8 labels to counter severe imbalance."""
    df = getattr(dataset, 'df', None)
    if df is None or 'sst8' not in df.columns:
        return None

    counts = np.zeros(NUM_Q8_CLASSES, dtype=np.float64)
    # Step through every label sequence once and tally how many residues land in each bucket.
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

    # Guard against zero divisions by pretending empty buckets had one residue.
    counts_safe = np.where(counts > 0, counts, 1.0)
    if mode == 'inverse':
        weights = total / counts_safe
    elif mode == 'effective':
        beta = float(min(max(beta, 0.9), 0.9999))
        effective_num = 1.0 - np.power(beta, counts_safe)
        effective_num = np.where(effective_num > 0, effective_num, 1e-8)
        weights = (1.0 - beta) / effective_num
    else:
        return None

    # Renormalise so the average class weight stays around 1.0.
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _parse_int_sequence(raw: str, name: str) -> Sequence[int]:
    parts = [p.strip() for p in str(raw).split(',') if p.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one integer")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc


def build_model(args, input_dim: int, device: torch.device) -> nn.Module:
    kernel_sizes = _parse_int_sequence(args.cnn_kernel_sizes, 'cnn_kernel_sizes')
    dilations = _parse_int_sequence(args.cnn_dilations, 'cnn_dilations')
    if len(kernel_sizes) != len(dilations):
        raise ValueError("cnn_kernel_sizes and cnn_dilations must have the same number of elements")

    # Instantiate the hybrid network so the Transformer and CNN see a consistent feature width.
    model = HybridPSSP(
        input_dim=input_dim,
        d_model=args.model_dim,
        n_heads=args.attention_heads,
        n_layers=args.transformer_layers,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        cnn_channels=args.cnn_channels,
        cnn_kernel_sizes=kernel_sizes,
        cnn_dilations=dilations,
        fuse_mode=args.fuse_mode,
        head_dropout=args.head_dropout,
        max_len=args.max_position
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
    # Flip the module into training mode for dropout / layer norm updates.
    model.train()
    total_loss = 0.0
    total_loss_q3 = 0.0
    total_loss_q8 = 0.0
    dataset_size = max(len(loader.dataset), 1)
    use_q8 = criterion_q8 is not None and q8_weight > 0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        # Move variable-length batch tensors to the accelerator using pinned-memory transfers.
        features = batch['features'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        labels_q8 = batch.get('labels_q8')
        if use_q8 and labels_q8 is not None:
            labels_q8 = labels_q8.to(device, non_blocking=True)
        else:
            labels_q8 = None

        # Clearing gradients at the top keeps the step independent of previous batches.
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            # Forward pass through the hybrid backbone; returns dict when both heads are active.
            outputs = model(features, mask=mask)
            if isinstance(outputs, torch.Tensor):
                logits_q3 = outputs
                logits_q8 = None
            else:
                logits_q3 = outputs['q3']
                logits_q8 = outputs.get('q8')

            # Flatten sequence predictions so padding can be dropped via CrossEntropy's ignore_index.
            logits_q3_flat = logits_q3.view(-1, logits_q3.size(-1))
            labels_flat = labels.view(-1)
            loss_q3 = criterion_q3(logits_q3_flat, labels_flat)
            loss = loss_q3

            loss_q8 = None
            if use_q8 and logits_q8 is not None and labels_q8 is not None:
                if criterion_q8 is None:
                    raise RuntimeError("criterion_q8 is required when using Q8 supervision")
                # Same padding trick for the Q8 branch so both heads train on identical residues.
                logits_q8_flat = logits_q8.view(-1, logits_q8.size(-1))
                labels_q8_flat = labels_q8.view(-1)
                loss_q8 = criterion_q8(logits_q8_flat, labels_q8_flat)
                loss = loss + q8_weight * loss_q8

        # Mixed precision: scale, backprop, then unscale so gradient clipping works correctly.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if clip_norm is not None and clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            # OneCycleLR expects a scheduler step per optimizer update.
            scheduler.step()

        batch_size = features.size(0)
        # Accumulate losses using batch size as weight so the final average is sample-aware.
        total_loss += loss.item() * batch_size
        total_loss_q3 += loss_q3.item() * batch_size
        if loss_q8 is not None:
            total_loss_q8 += loss_q8.item() * batch_size

        if step == 1:
            _log(
                "First batch processed: batch_size=%d, max_len=%d, feature_dim=%d" %
                (batch_size, features.size(1), features.size(2))
            )

    # Sample-weighted averages give a fair comparison when the last batch is smaller.
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
    # Evaluation mode disables dropout and turns off running-stat updates.
    model.eval()
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
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            lengths_cpu = batch['lengths']
            labels_q8 = batch.get('labels_q8')
            if use_q8 and labels_q8 is not None:
                labels_q8 = labels_q8.to(device, non_blocking=True)
            else:
                labels_q8 = None

            with autocast(enabled=amp_enabled):
                # Inference mirrors the training forward so we can reuse loss bookkeeping.
                outputs = model(features, mask=mask)
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

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_loss_q3 += loss_q3.item() * batch_size
            if loss_q8 is not None:
                total_loss_q8 += loss_q8.item() * batch_size

            # Convert logits to discrete class predictions for later F1 computation.
            preds_q3 = logits_q3.argmax(dim=-1)
            preds_q3_cpu = preds_q3.detach().cpu()
            labels_cpu = labels.detach().cpu()
            mask_cpu = mask.detach().cpu()
            if labels_q8 is not None and logits_q8 is not None:
                preds_q8 = logits_q8.argmax(dim=-1)
                preds_q8_cpu = preds_q8.detach().cpu()
                labels_q8_cpu = labels_q8.detach().cpu()
            else:
                preds_q8_cpu = None
                labels_q8_cpu = None

            for idx, seq_len in enumerate(lengths_cpu.tolist()):
                valid_len = int(seq_len)
                valid_mask = mask_cpu[idx, :valid_len]
                # Use the boolean mask to discard padded residues before scoring.
                valid_labels = labels_cpu[idx, :valid_len][valid_mask].numpy()
                valid_preds = preds_q3_cpu[idx, :valid_len][valid_mask].numpy()
                all_labels_q3.extend(valid_labels)
                all_preds_q3.extend(valid_preds)

                if labels_q8_cpu is not None and preds_q8_cpu is not None:
                    valid_labels_q8 = labels_q8_cpu[idx, :valid_len][valid_mask].numpy()
                    valid_preds_q8 = preds_q8_cpu[idx, :valid_len][valid_mask].numpy()
                    all_labels_q8.extend(valid_labels_q8)
                    all_preds_q8.extend(valid_preds_q8)

    # Mirror the training metrics so logs line up nicely.
    avg_loss = total_loss / dataset_size
    avg_loss_q3 = total_loss_q3 / dataset_size
    avg_loss_q8 = (total_loss_q8 / dataset_size) if total_loss_q8 > 0 else None
    f1_q3 = f1_score(all_labels_q3, all_preds_q3, average='macro') if all_labels_q3 else 0.0
    f1_q8 = f1_score(all_labels_q8, all_preds_q8, average='macro') if all_labels_q8 else None
    return avg_loss, avg_loss_q3, avg_loss_q8, f1_q3, f1_q8

def main(args):
    # Decide where the tensors will live; fallback rules are handled inside the helper.
    device = _resolve_device(args.device)
    _log(f"Using device: {device}")

    include_one_hot = not args.no_one_hot
    # Instantiate dataset objects; they lazily load auxiliary features on demand.
    train_dataset = ProteinDataset(
        os.path.join(args.data_dir, 'train.csv'),
        max_len=args.max_len,
        include_one_hot=include_one_hot,
        pssm_dir=args.pssm_dir,
        esm_dir=args.esm_dir,
        feature_norm=args.feature_norm
    )
    val_dataset = ProteinDataset(
        os.path.join(args.data_dir, 'val.csv'),
        max_len=args.max_len,
        include_one_hot=include_one_hot,
        pssm_dir=args.pssm_dir,
        esm_dir=args.esm_dir,
        feature_norm=args.feature_norm
    )
    _log(f"Loaded datasets | train={len(train_dataset)} | val={len(val_dataset)}")
    pin_memory = device.type in ('cuda', 'mps')
    # Persistent workers keep feature tensors cached between iterations when num_workers > 0.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers,
                            pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    _log(f"Batches per epoch | train={len(train_loader)} | val={len(val_loader)}")

    feature_sources = []
    if include_one_hot:
        # Always include the classic one-hot profile unless explicitly disabled.
        feature_sources.append('one-hot')
    if args.pssm_dir:
        # Pointer to precomputed evolutionary profiles (if available).
        feature_sources.append(f"pssm:{args.pssm_dir}")
    if args.esm_dir:
        # Pointer to contextual protein language model embeddings.
        feature_sources.append(f"esm:{args.esm_dir}")
    feature_dim = train_dataset.feature_dim
    _log(f"Residue feature stack | sources={'+'.join(feature_sources) if feature_sources else 'none'} | dim={feature_dim}")

    q8_class_weights = None
    if not args.disable_q8 and args.q8_balance != 'none':
        q8_class_weights = _compute_q8_class_weights(train_dataset, args.q8_balance, args.q8_balance_beta)
        if q8_class_weights is not None:
            weights_preview = ", ".join(f"{w:.2f}" for w in q8_class_weights.tolist())
            _log(f"Q8 class weights ({args.q8_balance}): [{weights_preview}]")
        else:
            _log("Q8 class balancing requested but Q8 labels were unavailable; falling back to uniform weights")

    model = build_model(args, feature_dim, device)
    _log("Hybrid model initialised and moved to device")

    amp_enabled = device.type in ('cuda', 'mps') and not args.disable_amp
    # Cross-entropy handles padding tokens via ignore_index so masked residues are skipped automatically.
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
            balance_mode = args.q8_balance if q8_class_weights is not None else 'uniform'
            _log(f"Q8 supervision enabled | weight={requested_weight:.3f} | balance={balance_mode}")
            if q8_class_weights is None and args.q8_balance != 'none':
                _log("Requested Q8 balancing but valid counts were not found; using uniform weighting")
            q8_weight = requested_weight
        else:
            _log("Q8 weight is non-positive; the auxiliary head will not influence the loss")
    else:
        _log("Q8 supervision disabled")

    # AdamW is a safe default for Transformer-style architectures.
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
        _log(f"Gradient clipping set to {args.clip_grad_norm}")

    _log("Starting training")

    best_f1_q3 = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # === Train epoch ===
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
            # === Validation pass mirrors training but with gradients disabled ===
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
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--ff_multiplier', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--head_dropout', type=float, default=0.15)
    parser.add_argument('--cnn_channels', type=int, default=128)
    parser.add_argument('--cnn_kernel_sizes', type=str, default='3,5,7')
    parser.add_argument('--cnn_dilations', type=str, default='1,2,4')
    parser.add_argument('--fuse_mode', choices=['sum', 'concat'], default='sum')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--max_position', type=int, default=4096)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=4)
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
    parser.add_argument('--pssm_dir', type=str, default=None)
    parser.add_argument('--esm_dir', type=str, default=None)
    parser.add_argument('--no_one_hot', action='store_true')
    parser.add_argument('--feature_norm', choices=['zscore', 'minmax', 'none'], default='zscore')
    args = parser.parse_args()
    if getattr(args, 'hidden_dim', None) is not None:
        args.model_dim = args.hidden_dim
    main(args)
