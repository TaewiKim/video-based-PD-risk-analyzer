"""
CARE-PD UPDRS Sequence Model
============================

Temporal model (1D CNN) over normalized gait sequences.
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
    from .gait_baseline import _apply_domain_normalization, load_care_pd_data
except ImportError:
    from gait_baseline import _apply_domain_normalization, load_care_pd_data


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TemporalCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.head(x)


@dataclass
class SequenceTrainingConfig:
    seq_len: int = 128
    target_fps: float = 30.0
    n_splits: int = 5
    epochs: int = 18
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 1e-4
    use_class_weight: bool = True
    use_scheduler: bool = True
    seed: int = 42
    optimize_threshold: bool = True


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resize_sequence(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """Linear interpolation of temporal axis to fixed length."""
    if len(seq) == seq_len:
        return seq
    if len(seq) <= 1:
        return np.repeat(seq[:1], seq_len, axis=0)
    x_old = np.linspace(0.0, 1.0, len(seq))
    x_new = np.linspace(0.0, 1.0, seq_len)
    out = np.empty((seq_len, seq.shape[1]), dtype=float)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(x_new, x_old, seq[:, c])
    return out


def _sequence_from_walk(walk_data: dict[str, Any], target_fps: float, seq_len: int) -> np.ndarray:
    """Build per-frame feature sequence from one walk."""
    pose = np.asarray(walk_data["pose"], dtype=float)
    trans = np.asarray(walk_data["trans"], dtype=float)
    fps = float(walk_data.get("fps", 30))
    pose, trans, fps = _apply_domain_normalization(pose, trans, fps, target_fps=target_fps)

    pose3 = pose.reshape(len(pose), 24, 3)
    # Key gait joints: pelvis/hips/knees/ankles/spine
    joint_idx = [0, 1, 2, 4, 5, 7, 8, 3, 6, 9]
    joint_feat = pose3[:, joint_idx, :].reshape(len(pose3), -1)  # (T, 30)

    vel = np.vstack([np.zeros((1, trans.shape[1])), np.diff(trans, axis=0) * fps])
    speed = np.linalg.norm(vel, axis=1, keepdims=True)

    seq = np.concatenate([joint_feat, trans, vel, speed], axis=1)
    return _resize_sequence(seq, seq_len)


def build_sequence_dataset(
    dataset_dir: str | Path,
    binary: bool = True,
    seq_len: int = 128,
    target_fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    walks, labels, _ = load_care_pd_data(dataset_dir)
    X = np.stack([_sequence_from_walk(w, target_fps=target_fps, seq_len=seq_len) for w in walks], axis=0)
    y = np.asarray(labels, dtype=int)
    if binary:
        y = (y > 0).astype(int)
    return X, y


def _standardize_sequence(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize channel-wise statistics from train split only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    return X_train_n, X_val_n


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_acc, best_f1 = 0.5, -1.0, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")
        if acc > best_acc or (acc == best_acc and mf1 > best_f1):
            best_t, best_acc, best_f1 = float(t), float(acc), float(mf1)
    return best_t


def train_sequence_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: SequenceTrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or SequenceTrainingConfig()
    _set_seed(config.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    n_classes = int(np.max(y)) + 1
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    oof_probs = np.zeros((len(y), n_classes), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]
        X_tr, X_va = _standardize_sequence(X_tr, X_va)

        # Conv1d expects (N, C, T)
        X_tr_t = np.transpose(X_tr, (0, 2, 1))
        X_va_t = np.transpose(X_va, (0, 2, 1))

        train_loader = DataLoader(
            SequenceDataset(X_tr_t, y_tr), batch_size=config.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            SequenceDataset(X_va_t, y_va), batch_size=config.batch_size, shuffle=False, drop_last=False
        )

        model = TemporalCNN(in_channels=X_tr_t.shape[1], n_classes=n_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        class_weight = None
        if config.use_class_weight:
            bincount = np.bincount(y_tr, minlength=n_classes).astype(float)
            bincount = np.maximum(bincount, 1.0)
            weight = len(y_tr) / (n_classes * bincount)
            class_weight = torch.tensor(weight, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        scheduler = None
        if config.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=0.5,
                patience=2,
            )

        best_state = None
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for _ in range(config.epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    val_losses.append(float(criterion(logits, yb).item()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss + config.min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        fold_probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                fold_probs.append(probs)
        oof_probs[va] = np.vstack(fold_probs)
        print(f"[fold {fold}/{config.n_splits}] done on {device}")

    y_pred_default = np.argmax(oof_probs, axis=1)
    results: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred_default)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred_default)),
        "macro_f1": float(f1_score(y, y_pred_default, average="macro")),
        "n_samples": int(len(y)),
        "n_classes": int(n_classes),
    }

    if n_classes == 2:
        y_prob = oof_probs[:, 1]
        threshold = 0.5
        if config.optimize_threshold:
            threshold = _best_threshold(y, y_prob)
        y_pred_opt = (y_prob >= threshold).astype(int)
        results.update(
            {
                "roc_auc": float(roc_auc_score(y, y_prob)),
                "threshold": float(threshold),
                "accuracy_opt": float(accuracy_score(y, y_pred_opt)),
                "balanced_accuracy_opt": float(balanced_accuracy_score(y, y_pred_opt)),
                "macro_f1_opt": float(f1_score(y, y_pred_opt, average="macro")),
            }
        )

    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train/evaluate temporal sequence model for CARE-PD")
    parser.add_argument("dataset_dir", nargs="?", default="data/datasets/CARE-PD")
    parser.add_argument("--multiclass", action="store_true", help="Use 4-class UPDRS instead of binary risk")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--no-scheduler", action="store_true")
    args = parser.parse_args()

    X, y = build_sequence_dataset(
        args.dataset_dir,
        binary=not args.multiclass,
        seq_len=args.seq_len,
        target_fps=args.target_fps,
    )
    cfg = SequenceTrainingConfig(
        seq_len=args.seq_len,
        target_fps=args.target_fps,
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_class_weight=not args.no_class_weight,
        use_scheduler=not args.no_scheduler,
    )
    res = train_sequence_cv(X, y, cfg)
    print(json.dumps(res, indent=2))
