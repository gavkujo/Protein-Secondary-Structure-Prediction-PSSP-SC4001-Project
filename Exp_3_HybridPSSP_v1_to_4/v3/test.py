"""
test.py
-------
Evaluate a trained Hybrid CNN + Transformer protein secondary structure model on the test set.
"""

import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from dataprep import ProteinDataset, collate_fn
from model import ProteinHybridModel

import argparse
import numpy as np
from typing import Optional


def _log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def _resolve_device(requested: str) -> torch.device:
    # Interpret the request once, then gracefully downgrade if the preferred accelerator is missing.
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
    # We inspect the first available feature file because all residues share the same width.
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


def build_model(args, feat_dim: Optional[int], device: torch.device) -> torch.nn.Module:
    # Mirrors the training configuration so the checkpoint weights align with the architecture.
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

def evaluate(model, loader, device, include_q8: bool = True):
    model.eval()
    # Lists will hold flattened residue-level predictions for downstream metrics and reports.
    all_preds_q3 = []
    all_labels_q3 = []
    all_preds_q8 = []
    all_labels_q8 = []
    ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            # Move padded tensors to the compute device while keeping the CPU copy of lengths.
            seqs = batch['seqs'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            lengths_cpu = batch['lengths']
            lengths = lengths_cpu.to(device, non_blocking=True)
            features = batch['features']
            if features is not None:
                features = features.to(device, non_blocking=True)
            labels_q8 = batch.get('labels_q8') if include_q8 else None

            outputs = model(seqs, lengths, features)
            if isinstance(outputs, torch.Tensor):
                logits_q3 = outputs
                logits_q8 = None
            else:
                logits_q3 = outputs['q3']
                logits_q8 = outputs.get('q8') if include_q8 else None

            preds_q3 = logits_q3.argmax(dim=-1)
            preds_q8 = logits_q8.argmax(dim=-1) if logits_q8 is not None else None

            for idx, seq_len in enumerate(lengths_cpu.tolist()):
                # Trim away padding so macro-F1 is computed strictly on observed residues.
                valid_labels = labels[idx, :seq_len].detach().cpu().numpy()
                valid_preds = preds_q3[idx, :seq_len].detach().cpu().numpy()
                all_labels_q3.extend(valid_labels)
                all_preds_q3.extend(valid_preds)
                ids.extend([batch['ids'][idx]] * seq_len)
                if logits_q8 is not None and labels_q8 is not None:
                    if preds_q8 is None:
                        raise RuntimeError("Q8 predictions missing despite logits being present")
                    valid_labels_q8 = labels_q8[idx, :seq_len].detach().cpu().numpy()
                    valid_preds_q8 = preds_q8[idx, :seq_len].detach().cpu().numpy()
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
    device = _resolve_device(args.device)
    _log(f"Using device: {device}")

    features_dir = args.features_dir if args.use_features else None
    # Rely on the processed CSV splits that were generated during training time.
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
        # Logging confirms that the auxiliary feature tensors match the expected size at inference.
        _log(f"Using residue features | dir={args.features_dir} | feat_dim={feat_dim}")

    # Create model (same hyperparameters as training)
    model = build_model(args, feat_dim, device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    _log(f"Loaded model from {args.ckpt}")

    include_q8 = not args.disable_q8
    if include_q8:
        _log("Evaluating Q3 and Q8 heads")
    else:
        _log("Evaluating Q3 head only")
    metrics = evaluate(model, test_loader, device, include_q8=include_q8)
    _log(f"Test macro-F1 (Q3): {metrics['f1_q3']:.4f}")
    _log("Q3 Classification Report:")
    print(classification_report(metrics['labels_q3'], metrics['preds_q3'], digits=4))
    _log("Q3 Confusion Matrix:")
    print(confusion_matrix(metrics['labels_q3'], metrics['preds_q3']))
    if include_q8 and metrics['labels_q8']:
        _log(f"Test macro-F1 (Q8): {metrics['f1_q8']:.4f}")
        _log("Q8 Classification Report:")
        print(classification_report(metrics['labels_q8'], metrics['preds_q8'], digits=4))
        _log("Q8 Confusion Matrix:")
        print(confusion_matrix(metrics['labels_q8'], metrics['preds_q8']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
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
    parser.add_argument('--disable_q8', action='store_true')
    args = parser.parse_args()
    if getattr(args, 'hidden_dim', None) is not None:
        args.model_dim = args.hidden_dim
    main(args)
