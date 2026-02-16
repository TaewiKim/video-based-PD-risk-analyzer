"""
CARE-PD Gait UPDRS Prediction Baseline
======================================

Simple baseline using handcrafted gait features + Random Forest
for predicting UPDRS gait scores from SMPL pose sequences.

Usage:
    from research_automation.pipeline.gait_baseline import (
        load_care_pd_data,
        extract_gait_features,
        train_updrs_classifier,
        evaluate_model,
    )

    # Load data
    walks, labels, subjects = load_care_pd_data("data/datasets/vida-adl_CARE-PD")

    # Extract features
    X, feature_names = extract_features_batch(walks)

    # Train and evaluate
    results = train_updrs_classifier(X, labels, n_splits=6)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


DATASETS_WITH_UPDRS = ["3DGait.pkl", "BMCLab.pkl", "PD-GaM.pkl", "T-SDU-PD.pkl"]
DEFAULT_TARGET_FPS = 30.0
DEFAULT_SPEED_CLIP_PERCENTILE = 99.0
DEFAULT_PD_RISK_CLASSIFIER = "histgb"

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
]


def _canonicalize_translation(trans: np.ndarray) -> np.ndarray:
    """
    Canonicalize trajectory to reduce site/camera axis mismatch.

    - Origin: first frame
    - Horizontal axes: rotate so principal walking direction becomes +X
    """
    if trans.ndim != 2 or trans.shape[0] == 0:
        return trans

    t = trans.astype(float, copy=True)
    t = t - t[0:1]

    if t.shape[1] < 2 or t.shape[0] < 3:
        return t

    horiz = [0, 2] if t.shape[1] >= 3 else [0, 1]
    xy = t[:, horiz]
    dxy = np.diff(xy, axis=0)

    if np.allclose(dxy, 0):
        return t

    # PCA/SVD principal motion direction in horizontal plane.
    _, _, vh = np.linalg.svd(dxy, full_matrices=False)
    forward = vh[0]
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        return t
    forward = forward / norm

    # Keep forward sign consistent with overall displacement.
    net_disp = xy[-1] - xy[0]
    if float(np.dot(net_disp, forward)) < 0:
        forward = -forward

    lateral = np.array([-forward[1], forward[0]], dtype=float)

    xy_rot = np.stack(
        [
            np.dot(xy, forward),
            np.dot(xy, lateral),
        ],
        axis=1,
    )
    t[:, horiz[0]] = xy_rot[:, 0]
    t[:, horiz[1]] = xy_rot[:, 1]
    return t


def _resample_timeseries(arr: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    """Linearly resample frame-wise array from src_fps to target_fps."""
    if src_fps <= 0 or target_fps <= 0 or arr.ndim < 1 or len(arr) < 2:
        return arr
    if abs(src_fps - target_fps) < 1e-6:
        return arr

    n_src = arr.shape[0]
    duration = max((n_src - 1) / src_fps, 0.0)
    n_tgt = max(2, int(round(duration * target_fps)) + 1)
    t_src = np.linspace(0.0, duration, n_src)
    t_tgt = np.linspace(0.0, duration, n_tgt)

    flat = arr.reshape(n_src, -1)
    out = np.empty((n_tgt, flat.shape[1]), dtype=float)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(t_tgt, t_src, flat[:, i])
    return out.reshape((n_tgt,) + arr.shape[1:])


def _apply_domain_normalization(
    pose: np.ndarray,
    trans: np.ndarray,
    fps: float,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Domain normalization for cross-site consistency:
    - fixed temporal rate (resample to target_fps)
    - unified trajectory frame (origin + forward-axis alignment)
    """
    pose_norm = _resample_timeseries(np.asarray(pose, dtype=float), float(fps), target_fps)
    trans_norm = _resample_timeseries(np.asarray(trans, dtype=float), float(fps), target_fps)
    trans_norm = _canonicalize_translation(trans_norm)
    return pose_norm, trans_norm, float(target_fps)


def load_care_pd_data(
    dataset_dir: str | Path,
    datasets: list[str] | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Load CARE-PD data with UPDRS labels.

    Args:
        dataset_dir: Path to CARE-PD dataset directory
        datasets: List of dataset filenames to load (default: all with UPDRS)

    Returns:
        walks: List of walk dictionaries with pose, trans, etc.
        labels: UPDRS gait scores (0-3)
        subjects: Subject identifiers for grouping
    """
    dataset_dir = Path(dataset_dir)
    if datasets is None:
        datasets = DATASETS_WITH_UPDRS

    walks = []
    labels = []
    subjects = []

    for pkl_name in datasets:
        pkl_path = dataset_dir / pkl_name
        if not pkl_path.exists():
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for subj_id, subj_data in data.items():
            for walk_id, walk_data in subj_data.items():
                updrs = walk_data.get("UPDRS_GAIT")
                if updrs is not None:
                    walks.append(walk_data)
                    labels.append(updrs)
                    subjects.append(f"{pkl_name}_{subj_id}")

    return walks, np.array(labels), np.array(subjects)


def extract_gait_features(
    walk_data: dict[str, Any],
    domain_normalize: bool = True,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> dict[str, float]:
    """
    Extract handcrafted gait features from a single walk.

    Args:
        walk_data: Dictionary with 'pose', 'trans', 'fps' keys

    Returns:
        Dictionary of feature name -> value
    """
    pose = np.asarray(walk_data["pose"], dtype=float)  # (T, 72) - 24 joints × 3 axis-angle
    trans = np.asarray(walk_data["trans"], dtype=float)  # (T, 3) - translation
    fps = float(walk_data.get("fps", 30))
    if domain_normalize:
        pose, trans, fps = _apply_domain_normalization(pose, trans, fps, target_fps=target_fps)
    else:
        trans = _canonicalize_translation(trans)

    features = {}

    # --- Temporal features ---
    features["duration"] = pose.shape[0] / fps
    features["n_frames"] = pose.shape[0]

    # --- Translation/velocity features ---
    trans_diff = np.diff(trans, axis=0)
    velocity = trans_diff * fps
    speed = np.linalg.norm(velocity, axis=1)
    if len(speed) > 0:
        speed_clip = np.percentile(speed, DEFAULT_SPEED_CLIP_PERCENTILE)
        if speed_clip > 0:
            speed = np.clip(speed, 0, speed_clip)

    features["speed_mean"] = speed.mean()
    features["speed_std"] = speed.std()
    features["speed_max"] = speed.max()
    features["speed_min"] = speed.min() if len(speed) > 0 else 0

    # Canonical velocity axes (forward/lateral/vertical) after trajectory alignment.
    features["vel_forward_mean"] = np.abs(velocity[:, 0]).mean()
    features["vel_vertical_mean"] = np.abs(velocity[:, 1]).mean() if velocity.shape[1] > 1 else 0.0
    features["vel_lateral_mean"] = np.abs(velocity[:, 2]).mean() if velocity.shape[1] > 2 else (
        np.abs(velocity[:, 1]).mean() if velocity.shape[1] > 1 else 0.0
    )

    # --- Pose statistics ---
    pose_reshaped = pose.reshape(-1, 24, 3)

    features["pose_std_mean"] = pose.std(axis=0).mean()
    features["pose_range_mean"] = (pose.max(axis=0) - pose.min(axis=0)).mean()

    # Per-joint variability (key joints for gait)
    joint_indices = {
        "pelvis": 0, "left_hip": 1, "right_hip": 2,
        "left_knee": 4, "right_knee": 5,
        "left_ankle": 7, "right_ankle": 8,
        "spine1": 3, "spine2": 6, "spine3": 9,
    }

    for joint_name, idx in joint_indices.items():
        joint_pose = pose_reshaped[:, idx, :]
        features[f"{joint_name}_std"] = joint_pose.std()
        features[f"{joint_name}_range"] = joint_pose.max() - joint_pose.min()

    # --- Symmetry features ---
    features["hip_asymmetry"] = np.abs(
        pose_reshaped[:, 1, :] - pose_reshaped[:, 2, :]
    ).mean()
    features["knee_asymmetry"] = np.abs(
        pose_reshaped[:, 4, :] - pose_reshaped[:, 5, :]
    ).mean()
    features["ankle_asymmetry"] = np.abs(
        pose_reshaped[:, 7, :] - pose_reshaped[:, 8, :]
    ).mean()

    # --- Frequency domain features ---
    pelvis_z = trans[:, 2] if trans.shape[1] > 2 else trans[:, 1]
    if len(pelvis_z) > 10:
        fft = np.fft.fft(pelvis_z - pelvis_z.mean())
        freqs = np.fft.fftfreq(len(pelvis_z), 1 / fps)
        pos_mask = (freqs > 0.3) & (freqs < 3)
        if pos_mask.any():
            fft_mag = np.abs(fft)[pos_mask]
            features["gait_frequency"] = freqs[pos_mask][np.argmax(fft_mag)]
            features["gait_regularity"] = fft_mag.max() / (fft_mag.mean() + 1e-6)
        else:
            features["gait_frequency"] = 0
            features["gait_regularity"] = 0
    else:
        features["gait_frequency"] = 0
        features["gait_regularity"] = 0

    # --- Acceleration features ---
    if len(velocity) > 1:
        accel = np.diff(velocity, axis=0) * fps
        accel_mag = np.linalg.norm(accel, axis=1)
        features["accel_mean"] = accel_mag.mean()
        features["accel_std"] = accel_mag.std()
        features["accel_max"] = accel_mag.max()
    else:
        features["accel_mean"] = 0
        features["accel_std"] = 0
        features["accel_max"] = 0

    # --- Jerk (smoothness) ---
    if len(velocity) > 2:
        jerk = np.diff(np.diff(velocity, axis=0), axis=0) * fps * fps
        jerk_mag = np.linalg.norm(jerk, axis=1)
        features["jerk_mean"] = jerk_mag.mean()
        features["jerk_std"] = jerk_mag.std()
    else:
        features["jerk_mean"] = 0
        features["jerk_std"] = 0

    return features


def extract_features_batch(
    walks: list[dict],
    domain_normalize: bool = True,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features from multiple walks.

    Returns:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
    """
    feature_list = [
        extract_gait_features(w, domain_normalize=domain_normalize, target_fps=target_fps)
        for w in walks
    ]
    feature_names = list(feature_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in feature_list])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return X, feature_names


def train_updrs_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 6,
    binary: bool = False,
    classifier: str = DEFAULT_PD_RISK_CLASSIFIER,
    optimize_threshold: bool = False,
) -> dict[str, Any]:
    """
    Train and evaluate UPDRS classifier with cross-validation.

    Args:
        X: Feature matrix
        y: Labels (UPDRS 0-3)
        n_splits: Number of CV folds
        binary: If True, convert to binary (Normal vs Impaired)
        optimize_threshold: If True and binary, sweep threshold for max accuracy.

    Returns:
        Dictionary with accuracy, predictions, and model
    """
    if binary:
        y = (y > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = _make_classifier(classifier)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    threshold = 0.5

    # Fit final model on all data
    clf.fit(X_scaled, y)

    y_prob = None
    if binary:
        y_prob = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        if optimize_threshold:
            threshold = _find_best_binary_threshold(y, y_prob)
            y_pred = (y_prob >= threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "y_true": y,
        "y_pred": y_pred,
        "model": clf,
        "scaler": scaler,
        "confusion_matrix": confusion_matrix(y, y_pred),
    }

    if binary and y_prob is not None:
        results["roc_auc"] = roc_auc_score(y, y_prob)
        results["y_prob"] = y_prob
        results["threshold"] = float(threshold)

    return results


def _find_best_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select threshold that maximizes accuracy; tie-break with macro-F1."""
    best_threshold = 0.5
    best_acc = -1.0
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        if acc > best_acc or (acc == best_acc and macro_f1 > best_f1):
            best_acc = acc
            best_f1 = macro_f1
            best_threshold = float(threshold)
    return best_threshold


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    top_k: int = 15,
) -> list[tuple[str, float]]:
    """Get top-k most important features."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).mean(axis=0)
    else:
        return []
    indices = np.argsort(importances)[::-1][:top_k]
    return [(feature_names[i], importances[i]) for i in indices]


def print_results(results: dict[str, Any], feature_names: list[str] | None = None):
    """Print evaluation results."""
    print(f"Accuracy:          {results['accuracy']:.3f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.3f}")
    print(f"Macro-F1:          {results['macro_f1']:.3f}")

    if "roc_auc" in results:
        print(f"ROC-AUC:           {results['roc_auc']:.3f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    if feature_names:
        print("\nTop 10 Important Features:")
        top = get_feature_importance(results["model"], feature_names, 10)
        if not top:
            print("  (not available for this classifier)")
        for name, imp in top:
            print(f"  {name:<20s} {imp:.4f}")


def _make_classifier(name: str = DEFAULT_PD_RISK_CLASSIFIER):
    name = name.lower()
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    if name in {"gb", "gbm", "gradient_boosting"}:
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    if name in {"histgb", "hist_gradient_boosting"}:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_depth=6,
            random_state=42,
        )
    if name in {"et", "extra_trees"}:
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    if name in {"svm", "svc"}:
        return SVC(
            C=2.0,
            kernel="rbf",
            class_weight="balanced",
            gamma="scale",
            probability=True,
            random_state=42,
        )
    if name in {"lr", "logistic", "logistic_regression"}:
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            multi_class="auto",
        )
    if name in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ValueError("xgboost is not installed. Install with: ./web/.venv312/bin/pip install xgboost") from exc
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def get_current_baseline_params() -> dict[str, Any]:
    """Return current default preprocessing/model parameters used by baseline."""
    return {
        "domain_normalization": {
            "enabled": True,
            "target_fps": DEFAULT_TARGET_FPS,
            "trajectory_alignment": "origin+forward_axis(PCA)",
            "speed_clip_percentile": DEFAULT_SPEED_CLIP_PERCENTILE,
        },
        "default_classifier": DEFAULT_PD_RISK_CLASSIFIER,
        "cv": {
            "type": "StratifiedKFold",
            "shuffle": True,
            "random_state": 42,
        },
    }


def _load_dataset_with_subject_groups(
    dataset_dir: str | Path,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one CARE-PD dataset with subject-level groups."""
    walks, labels, subjects = load_care_pd_data(dataset_dir, datasets=[dataset_name])
    if len(walks) == 0:
        return np.empty((0, 0)), np.array([]), np.array([])
    X, _ = extract_features_batch(walks)
    # Strip dataset prefix (e.g., BMCLab.pkl_subjectA -> subjectA) for LOSO grouping.
    groups = np.array([s.split("_", 1)[1] if "_" in s else s for s in subjects])
    return X, labels, groups


def evaluate_within_dataset_loso(
    dataset_dir: str | Path,
    datasets: list[str] | None = None,
    classifier: str = DEFAULT_PD_RISK_CLASSIFIER,
) -> dict[str, dict[str, float]]:
    """
    Literature-style within-dataset evaluation (LOSO, subject-wise).

    Returns per-dataset metrics and macro-F1 mean across datasets.
    """
    if datasets is None:
        datasets = DATASETS_WITH_UPDRS

    results: dict[str, dict[str, float]] = {}
    macro_values = []

    for dataset_name in datasets:
        X, y, groups = _load_dataset_with_subject_groups(dataset_dir, dataset_name)
        if len(y) == 0:
            continue

        logo = LeaveOneGroupOut()
        y_pred = np.empty_like(y)

        for train_idx, test_idx in logo.split(X, y, groups=groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = _make_classifier(classifier)
            clf.fit(X_train, y[train_idx])
            y_pred[test_idx] = clf.predict(X_test)

        metrics = {
            "macro_f1": float(f1_score(y, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "n_subjects": int(len(np.unique(groups))),
            "n_walks": int(len(y)),
        }
        results[dataset_name] = metrics
        macro_values.append(metrics["macro_f1"])

    if macro_values:
        results["summary"] = {"macro_f1_mean": float(np.mean(macro_values))}
    return results


def evaluate_lodo(
    dataset_dir: str | Path,
    datasets: list[str] | None = None,
    classifier: str = DEFAULT_PD_RISK_CLASSIFIER,
) -> dict[str, dict[str, float]]:
    """
    Literature-style Leave-One-Dataset-Out (LODO) evaluation.
    """
    if datasets is None:
        datasets = DATASETS_WITH_UPDRS

    cached: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dataset_name in datasets:
        cached[dataset_name] = _load_dataset_with_subject_groups(dataset_dir, dataset_name)

    results: dict[str, dict[str, float]] = {}
    macro_values = []

    for test_dataset in datasets:
        X_test, y_test, _ = cached[test_dataset]
        if len(y_test) == 0:
            continue

        train_parts = [cached[name] for name in datasets if name != test_dataset and len(cached[name][1]) > 0]
        X_train = np.concatenate([part[0] for part in train_parts], axis=0)
        y_train = np.concatenate([part[1] for part in train_parts], axis=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = _make_classifier(classifier)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        metrics = {
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "n_walks_test": int(len(y_test)),
        }
        results[test_dataset] = metrics
        macro_values.append(metrics["macro_f1"])

    if macro_values:
        results["summary"] = {"macro_f1_mean": float(np.mean(macro_values))}
    return results


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="CARE-PD baseline evaluation")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/datasets/CARE-PD",
        help="Directory containing CARE-PD .pkl files",
    )
    parser.add_argument(
        "--protocol",
        choices=["literature", "all", "legacy"],
        default="literature",
        help="Evaluation protocol. literature = LOSO + LODO (recommended).",
    )
    parser.add_argument(
        "--method",
        choices=["rf", "carepd_official", "auto"],
        default="auto",
        help="rf: handcrafted baseline, carepd_official: paper code path, auto: prefer official if ready.",
    )
    parser.add_argument(
        "--carepd-code-dir",
        default="data/datasets/CARE-PD-code",
        help="Path to official CARE-PD code repository.",
    )
    parser.add_argument(
        "--backbone",
        default="motionbert",
        help="Official CARE-PD backbone for run.py (e.g. motionbert, mixste, motionagformer, poseformerv2, momask).",
    )
    parser.add_argument(
        "--config",
        default="BMCLab_backright.json",
        help="Official CARE-PD config file for selected backbone.",
    )
    parser.add_argument(
        "--classifier",
        choices=["rf", "gbm", "histgb", "et", "svm", "lr", "xgb"],
        default=DEFAULT_PD_RISK_CLASSIFIER,
        help="Classifier for handcrafted baseline methods.",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="For binary setting, tune decision threshold on out-of-fold probabilities.",
    )
    parser.add_argument(
        "--disable-domain-normalization",
        action="store_true",
        help="Disable domain normalization (resampling + axis alignment).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Target FPS for domain normalization.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    selected_method = args.method
    if selected_method == "auto":
        try:
            from carepd_official import official_env_ready
        except Exception:
            selected_method = "rf"
        else:
            ready, _ = official_env_ready(args.carepd_code_dir)
            selected_method = "carepd_official" if ready else "rf"

    if selected_method == "carepd_official":
        from carepd_official import official_env_ready, run_official_eval

        ready, issues = official_env_ready(args.carepd_code_dir)
        if not ready:
            print("CARE-PD official method is not ready:")
            for issue in issues:
                print(f"  - {issue}")
            print("Falling back to RF baseline method.")
            selected_method = "rf"
        else:
            protocol = "within" if args.protocol in {"literature", "all"} else "within"
            print("\n" + "=" * 50)
            print("CARE-PD Official Method")
            print("=" * 50)
            print(f"code_dir:  {args.carepd_code_dir}")
            print(f"protocol:  {protocol}")
            print(f"backbone:  {args.backbone}")
            print(f"config:    {args.config}")
            run = run_official_eval(
                code_dir=args.carepd_code_dir,
                backbone=args.backbone,
                config=args.config,
                protocol=protocol,
                python_executable=sys.executable,
            )
            print("\n[stdout]")
            print(run.stdout[-8000:] if run.stdout else "(empty)")
            print("\n[stderr]")
            print(run.stderr[-8000:] if run.stderr else "(empty)")
            print(f"\nexit_code: {run.returncode}")
            summary = {
                "method": "carepd_official",
                "protocol": protocol,
                "backbone": args.backbone,
                "config": args.config,
                "exit_code": run.returncode,
            }
            print("\nsummary:", json.dumps(summary, ensure_ascii=False))
            raise SystemExit(run.returncode)

    if selected_method == "rf" and args.protocol in {"literature", "all"}:
        print("\n" + "=" * 50)
        print("Within-Dataset LOSO (Literature Protocol)")
        print("=" * 50)
        print(f"Classifier: {args.classifier}")
        loso_results = evaluate_within_dataset_loso(dataset_dir, classifier=args.classifier)
        for ds in DATASETS_WITH_UPDRS:
            if ds not in loso_results:
                continue
            r = loso_results[ds]
            print(
                f"{ds:<12s} Macro-F1={r['macro_f1']:.3f}  "
                f"Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}  "
                f"Subjects={r['n_subjects']}  Walks={r['n_walks']}"
            )
        if "summary" in loso_results:
            print(f"LOSO Macro-F1 mean: {loso_results['summary']['macro_f1_mean']:.3f}")

        print("\n" + "=" * 50)
        print("LODO (Literature Protocol)")
        print("=" * 50)
        lodo_results = evaluate_lodo(dataset_dir, classifier=args.classifier)
        for ds in DATASETS_WITH_UPDRS:
            if ds not in lodo_results:
                continue
            r = lodo_results[ds]
            print(
                f"Test={ds:<12s} Macro-F1={r['macro_f1']:.3f}  "
                f"Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}  "
                f"N_test={r['n_walks_test']}"
            )
        if "summary" in lodo_results:
            print(f"LODO Macro-F1 mean: {lodo_results['summary']['macro_f1_mean']:.3f}")

    if selected_method == "rf" and args.protocol in {"legacy", "all"}:
        print("\n" + "=" * 50)
        print("Legacy Pooled CV (Not Literature Protocol)")
        print("=" * 50)
        print("Loading CARE-PD data...")
        walks, labels, _ = load_care_pd_data(dataset_dir)
        print(f"Loaded {len(walks)} walks")
        domain_normalize = not args.disable_domain_normalization
        X, feature_names = extract_features_batch(
            walks,
            domain_normalize=domain_normalize,
            target_fps=args.target_fps,
        )
        print(f"Feature matrix: {X.shape}")
        print(f"Domain normalization: {domain_normalize} (target_fps={args.target_fps:g})")

        print("\n4-Class UPDRS Prediction")
        results = train_updrs_classifier(X, labels, binary=False, classifier=args.classifier)
        print_results(results, feature_names)

        print("\nBinary: Normal vs Impaired")
        results_binary = train_updrs_classifier(
            X,
            labels,
            binary=True,
            classifier=args.classifier,
            optimize_threshold=args.optimize_threshold,
        )
        print_results(results_binary, feature_names)
        if "threshold" in results_binary:
            print(f"Decision threshold: {results_binary['threshold']:.3f}")
