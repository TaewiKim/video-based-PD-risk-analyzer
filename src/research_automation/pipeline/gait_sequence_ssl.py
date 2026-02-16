"""
CARE-PD Self-Supervised + Fine-tune Sequence Model
==================================================

Two-stage training:
1) SSL pretraining with temporal order verification
2) Supervised fine-tuning for UPDRS risk/severity
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .gait_sequence_model import build_sequence_dataset
except ImportError:
    from gait_sequence_model import build_sequence_dataset


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class Encoder1D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z.squeeze(-1)


class SSLOrderModel(nn.Module):
    """Binary SSL task: original order vs temporal-chunk shuffled order."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder = Encoder1D(in_channels)
        self.head = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class ClassifierModel(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.encoder = Encoder1D(in_channels)
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


@dataclass
class SSLConfig:
    seq_len: int = 128
    target_fps: float = 30.0
    n_splits: int = 5
    ssl_epochs: int = 12
    finetune_epochs: int = 32
    batch_size: int = 64
    lr_ssl: float = 1e-3
    lr_finetune: float = 8e-4
    weight_decay: float = 1e-4
    seed: int = 42
    optimize_threshold: bool = True
    binary: bool = True


def _make_ssl_pairs(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build SSL pseudo labels:
    - 0: original sequence
    - 1: chunk-shuffled sequence
    """
    n, c, t = X.shape
    x_orig = X.copy()
    x_perm = X.copy()
    n_chunks = 4
    chunk_size = max(1, t // n_chunks)

    for i in range(n):
        idx = np.arange(n_chunks)
        np.random.shuffle(idx)
        chunks = []
        for j in idx:
            s = j * chunk_size
            e = t if j == n_chunks - 1 else min(t, (j + 1) * chunk_size)
            chunks.append(X[i, :, s:e])
        x_perm[i] = np.concatenate(chunks, axis=1)[:, :t]

    X_ssl = np.concatenate([x_orig, x_perm], axis=0)
    y_ssl = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)], axis=0)
    return X_ssl, y_ssl


def _standardize_train_val(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # X shape: (N, C, T)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_val - mean) / std


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_acc, best_f1 = 0.5, -1.0, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")
        if acc > best_acc or (acc == best_acc and mf1 > best_f1):
            best_t, best_acc, best_f1 = float(t), float(acc), float(mf1)
    return best_t


def run_ssl_finetune_cv(dataset_dir: str | Path, config: SSLConfig | None = None) -> dict[str, Any]:
    config = config or SSLConfig()
    _set_seed(config.seed)

    X, y = build_sequence_dataset(
        dataset_dir=dataset_dir,
        binary=config.binary,
        seq_len=config.seq_len,
        target_fps=config.target_fps,
    )
    # Conv1d input: (N, C, T)
    X = np.transpose(X, (0, 2, 1))

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    n_classes = int(np.max(y)) + 1
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    oof_probs = np.zeros((len(y), n_classes), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]
        X_tr, X_va = _standardize_train_val(X_tr, X_va)

        # 1) SSL pretraining on train split
        X_ssl, y_ssl = _make_ssl_pairs(X_tr)
        ssl_loader = DataLoader(SequenceDataset(X_ssl, y_ssl), batch_size=config.batch_size, shuffle=True)
        ssl_model = SSLOrderModel(in_channels=X_tr.shape[1]).to(device)
        opt_ssl = torch.optim.AdamW(ssl_model.parameters(), lr=config.lr_ssl, weight_decay=config.weight_decay)
        crit_ssl = nn.CrossEntropyLoss()

        ssl_model.train()
        for _ in range(config.ssl_epochs):
            for xb, yb in ssl_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt_ssl.zero_grad(set_to_none=True)
                loss = crit_ssl(ssl_model(xb), yb)
                loss.backward()
                opt_ssl.step()

        # 2) Fine-tuning on supervised labels
        clf = ClassifierModel(in_channels=X_tr.shape[1], n_classes=n_classes).to(device)
        clf.encoder.load_state_dict(ssl_model.encoder.state_dict())

        tr_loader = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=config.batch_size, shuffle=True)
        va_loader = DataLoader(SequenceDataset(X_va, y_va), batch_size=config.batch_size, shuffle=False)

        opt = torch.optim.AdamW(clf.parameters(), lr=config.lr_finetune, weight_decay=config.weight_decay)
        class_weight = None
        bincount = np.bincount(y_tr, minlength=n_classes).astype(float)
        bincount = np.maximum(bincount, 1.0)
        weight = len(y_tr) / (n_classes * bincount)
        class_weight = torch.tensor(weight, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)

        clf.train()
        for _ in range(config.finetune_epochs):
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = clf(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

        clf.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in va_loader:
                xb = xb.to(device)
                p = torch.softmax(clf(xb), dim=1).cpu().numpy()
                probs.append(p)
        oof_probs[va] = np.vstack(probs)
        print(f"[fold {fold}/{config.n_splits}] ssl+finetune done on {device}")

    y_pred = np.argmax(oof_probs, axis=1)
    res: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "n_samples": int(len(y)),
        "n_classes": int(n_classes),
    }

    if n_classes == 2:
        y_prob = oof_probs[:, 1]
        threshold = _best_threshold(y, y_prob) if config.optimize_threshold else 0.5
        y_pred_opt = (y_prob >= threshold).astype(int)
        res.update(
            {
                "roc_auc": float(roc_auc_score(y, y_prob)),
                "threshold": float(threshold),
                "accuracy_opt": float(accuracy_score(y, y_pred_opt)),
                "balanced_accuracy_opt": float(balanced_accuracy_score(y, y_pred_opt)),
                "macro_f1_opt": float(f1_score(y, y_pred_opt, average="macro")),
            }
        )
    return res


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CARE-PD SSL + fine-tune sequence model")
    parser.add_argument("dataset_dir", nargs="?", default="data/datasets/CARE-PD")
    parser.add_argument("--multiclass", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ssl-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = SSLConfig(
        seq_len=args.seq_len,
        target_fps=args.target_fps,
        n_splits=args.folds,
        ssl_epochs=args.ssl_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        binary=not args.multiclass,
    )
    result = run_ssl_finetune_cv(args.dataset_dir, cfg)
    print(json.dumps(result, indent=2))
