"""
test.py
-------
Evaluate a trained Hybrid CNN + Transformer protein secondary structure model on the test set.
"""

import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from dataprep import ProteinDataset, collate_fn
from model import HybridPSSP

import argparse
from typing import Optional, Sequence, List


def _log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def _resolve_device(requested: str) -> torch.device:
    req = (requested or '').lower()
    if req == 'cuda' and torch.cuda.is_available():
        # Ideal scenario: run the evaluation on GPU for speed.
        return torch.device('cuda')
    if req in ('mps', 'metal') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Metal acceleration slot for M-series laptops in the lab.
        return torch.device('mps')
    if req == 'cpu' or not req:
        # Either the student requested CPU or nothing was specified.
        return torch.device('cpu')
    if torch.cuda.is_available():
        print(f"Requested device '{requested}' not available; using CUDA instead.")
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"Requested device '{requested}' not available; using MPS instead.")
        return torch.device('mps')
    print(f"Requested device '{requested}' not available; using CPU instead.")
    return torch.device('cpu')

def _parse_int_sequence(raw: str, name: str) -> Sequence[int]:
    parts = [p.strip() for p in str(raw).split(',') if p.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one integer")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc


def build_model(args, input_dim: int, device: torch.device) -> torch.nn.Module:
    kernel_sizes = _parse_int_sequence(args.cnn_kernel_sizes, 'cnn_kernel_sizes')
    dilations = _parse_int_sequence(args.cnn_dilations, 'cnn_dilations')
    if len(kernel_sizes) != len(dilations):
        raise ValueError("cnn_kernel_sizes and cnn_dilations must have the same number of elements")

    # Mirror the training-side architecture so weights load cleanly.
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

def evaluate(model, loader, device, include_q8: bool = True):
    # Disable dropout and such so evaluation stays deterministic.
    model.eval()
    all_preds_q3 = []
    all_labels_q3 = []
    all_preds_q8 = []
    all_labels_q8 = []
    # Keep track of sequence IDs so we can trace back any interesting misclassifications later.
    ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            # Each batch already contains padded tensors and mask emitted by collate_fn.
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            lengths_cpu = batch['lengths']
            labels_q8 = batch.get('labels_q8') if include_q8 else None
            if labels_q8 is not None:
                labels_q8 = labels_q8.to(device, non_blocking=True)

            # Forward pass: HybridPSSP returns either a tensor (Q3-only) or a dict with both heads.
            outputs = model(features, mask=mask)
            if isinstance(outputs, torch.Tensor):
                logits_q3 = outputs
                logits_q8 = None
            else:
                logits_q3 = outputs['q3']
                logits_q8 = outputs.get('q8') if include_q8 else None

            # Argmax over the class dimension to convert logits into discrete predictions.
            preds_q3 = logits_q3.argmax(dim=-1)
            preds_q8 = logits_q8.argmax(dim=-1) if logits_q8 is not None else None

            mask_cpu = mask.detach().cpu()
            labels_cpu = labels.detach().cpu()
            preds_q3_cpu = preds_q3.detach().cpu()
            labels_q8_cpu = labels_q8.detach().cpu() if labels_q8 is not None else None
            preds_q8_cpu = preds_q8.detach().cpu() if preds_q8 is not None else None

            for idx in range(len(lengths_cpu)):
                valid_mask = mask_cpu[idx].bool()
                if not valid_mask.any():
                    continue
                # Trim predictions back to the original residue count before aggregating metrics.
                valid_labels = labels_cpu[idx][valid_mask].tolist()
                valid_preds = preds_q3_cpu[idx][valid_mask].tolist()
                all_labels_q3.extend(valid_labels)
                all_preds_q3.extend(valid_preds)
                ids.extend([batch['ids'][idx]] * len(valid_labels))

                if labels_q8_cpu is not None and preds_q8_cpu is not None:
                    valid_labels_q8 = labels_q8_cpu[idx][valid_mask].tolist()
                    valid_preds_q8 = preds_q8_cpu[idx][valid_mask].tolist()
                    all_labels_q8.extend(valid_labels_q8)
                    all_preds_q8.extend(valid_preds_q8)

    f1_q3 = f1_score(all_labels_q3, all_preds_q3, average='macro') if all_labels_q3 else 0.0
    f1_q8 = f1_score(all_labels_q8, all_preds_q8, average='macro') if all_labels_q8 else None
    return {
        'labels_q3': all_labels_q3,
        'preds_q3': all_preds_q3,
        'labels_q8': all_labels_q8,
        'preds_q8': all_preds_q8,
        'ids': ids,
        'f1_q3': f1_q3,
        'f1_q8': f1_q8,
    }

def main(args):
    # Pick a compute device using the shared helper (CUDA > MPS > CPU).
    device = _resolve_device(args.device)
    _log(f"Using device: {device}")

    include_one_hot = not args.no_one_hot
    # Build the test dataset; just like training but without shuffling or augmentation.
    test_dataset = ProteinDataset(
        os.path.join(args.data_dir, 'test.csv'),
        max_len=args.max_len,
        include_one_hot=include_one_hot,
        pssm_dir=args.pssm_dir,
        esm_dir=args.esm_dir,
        feature_norm=args.feature_norm
    )
    _log(f"Loaded test dataset | samples={len(test_dataset)}")
    pin_memory = device.type in ('cuda', 'mps')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers,
                             pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    _log(f"Test batches={len(test_loader)} | batch_size={args.batch_size}")

    feature_sources = []
    if include_one_hot:
        # Classic amino-acid identity features.
        feature_sources.append('one-hot')
    if args.pssm_dir:
        # Point to the folder containing evolutionary PSSM matrices.
        feature_sources.append(f"pssm:{args.pssm_dir}")
    if args.esm_dir:
        # Include learned representations from ESM if generated.
        feature_sources.append(f"esm:{args.esm_dir}")
    feature_dim = test_dataset.feature_dim
    _log(f"Residue feature stack | sources={'+'.join(feature_sources) if feature_sources else 'none'} | dim={feature_dim}")

    model = build_model(args, feature_dim, device)
    # Load the checkpoint produced during training; map_location ensures compatibility across devices.
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    _log(f"Loaded model from {args.ckpt}")

    include_q8 = not args.disable_q8
    if include_q8:
        _log("Evaluating Q3 and Q8 heads")
    else:
        _log("Evaluating Q3 head only")
    # Run one full pass over the test split and gather per-residue predictions.
    metrics = evaluate(model, test_loader, device, include_q8=include_q8)
    _log(f"Test macro-F1 (Q3): {metrics['f1_q3']:.4f}")
    if metrics['labels_q3']:
        _log("Q3 Classification Report:")
        print(classification_report(metrics['labels_q3'], metrics['preds_q3'], digits=4))
        _log("Q3 Confusion Matrix:")
        print(confusion_matrix(metrics['labels_q3'], metrics['preds_q3']))
    else:
        _log("No Q3 labels available; skipping detailed metrics")

    if include_q8 and metrics['labels_q8']:
        _log(f"Test macro-F1 (Q8): {metrics['f1_q8']:.4f}")
        _log("Q8 Classification Report:")
        print(classification_report(metrics['labels_q8'], metrics['preds_q8'], digits=4))
        _log("Q8 Confusion Matrix:")
        print(confusion_matrix(metrics['labels_q8'], metrics['preds_q8']))
    elif include_q8:
        _log("Q8 labels not present in dataset; skipping auxiliary metrics")

    if args.metrics_path:
        report = {
            'f1_q3': metrics['f1_q3'],
            'f1_q8': metrics['f1_q8'],
            'samples': len(test_dataset),
            'feature_dim': feature_dim,
            'feature_sources': feature_sources,
            'ckpt': os.path.abspath(args.ckpt),
        }
        # Persist a lightweight JSON dump so downstream notebooks can load summary metrics quickly.
        with open(args.metrics_path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        _log(f"Metrics written to {args.metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
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
    parser.add_argument('--disable_q8', action='store_true')
    parser.add_argument('--pssm_dir', type=str, default=None)
    parser.add_argument('--esm_dir', type=str, default=None)
    parser.add_argument('--no_one_hot', action='store_true')
    parser.add_argument('--feature_norm', choices=['zscore', 'minmax', 'none'], default='zscore')
    parser.add_argument('--metrics_path', type=str, default=None, help='Optional path to save a JSON metrics summary')
    args = parser.parse_args()
    if getattr(args, 'hidden_dim', None) is not None:
        args.model_dim = args.hidden_dim
    main(args)
