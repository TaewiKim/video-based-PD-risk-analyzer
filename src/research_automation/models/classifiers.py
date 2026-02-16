"""
Scikit-learn Classifier Wrappers
================================

Wrappers for common ML classifiers for PD symptom classification.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import InferenceResult, ModelConfig, SequenceModel, register_model


@dataclass
class ClassifierConfig(ModelConfig):
    """Classifier configuration."""

    classifier_type: str = "random_forest"  # rf, svm, lr, gbm
    n_estimators: int = 100
    max_depth: int | None = 10
    C: float = 1.0
    kernel: str = "rbf"
    random_state: int = 42
    class_names: list[str] = field(default_factory=list)


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    class_report: str
    roc_auc: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc if self.roc_auc else 0.0,
        }


class FeatureClassifier:
    """
    Classifier for handcrafted features.

    Supports Random Forest, SVM, Logistic Regression, and Gradient Boosting.
    """

    CLASSIFIERS = {
        "random_forest": RandomForestClassifier,
        "rf": RandomForestClassifier,
        "svm": SVC,
        "logistic_regression": LogisticRegression,
        "lr": LogisticRegression,
        "gradient_boosting": GradientBoostingClassifier,
        "gbm": GradientBoostingClassifier,
    }

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig(name="classifier")
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _create_model(self):
        """Create classifier based on config."""
        clf_type = self.config.classifier_type.lower()
        clf_class = self.CLASSIFIERS.get(clf_type)

        if clf_class is None:
            raise ValueError(f"Unknown classifier: {clf_type}")

        if clf_type in ("random_forest", "rf"):
            return clf_class(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif clf_type == "svm":
            return clf_class(
                C=self.config.C,
                kernel=self.config.kernel,
                probability=True,
                random_state=self.config.random_state,
            )
        elif clf_type in ("logistic_regression", "lr"):
            return clf_class(
                C=self.config.C,
                max_iter=1000,
                random_state=self.config.random_state,
            )
        elif clf_type in ("gradient_boosting", "gbm"):
            return clf_class(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth or 3,
                random_state=self.config.random_state,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale: bool = True,
    ) -> "FeatureClassifier":
        """
        Fit classifier.

        Args:
            X: (N, D) feature matrix
            y: (N,) labels
            scale: Whether to standardize features

        Returns:
            Self
        """
        self.model = self._create_model()

        if scale:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict labels."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if scale:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if scale:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale: bool = True,
    ) -> ClassificationMetrics:
        """
        Evaluate classifier.

        Args:
            X: Feature matrix
            y: True labels
            scale: Whether to scale features

        Returns:
            ClassificationMetrics
        """
        y_pred = self.predict(X, scale)
        y_proba = self.predict_proba(X, scale)

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, zero_division=0)

        # ROC-AUC (binary or multi-class)
        try:
            if len(np.unique(y)) == 2:
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = None

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            class_report=report,
            roc_auc=roc_auc,
        )

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        scale: bool = True,
    ) -> dict[str, float]:
        """
        Cross-validate classifier.

        Args:
            X: Feature matrix
            y: Labels
            n_folds: Number of CV folds
            scale: Whether to scale

        Returns:
            Dict of metric name -> (mean, std)
        """
        if scale:
            X = self.scaler.fit_transform(X)

        self.model = self._create_model()
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_state)

        # Compute CV scores
        accuracy = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        f1 = cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted")

        return {
            "accuracy": accuracy.mean(),
            "accuracy_std": accuracy.std(),
            "f1": f1.mean(),
            "f1_std": f1.std(),
        }

    def feature_importance(self) -> np.ndarray | None:
        """Get feature importances (if available)."""
        if not self._is_fitted:
            return None

        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_).mean(axis=0)

        return None

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        path = Path(path)
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load model from file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.config = data["config"]
        self._is_fitted = True


@register_model("updrs_classifier")
class UPDRSClassifier(SequenceModel):
    """Classifier for UPDRS severity levels."""

    def __init__(self, config: ClassifierConfig | None = None):
        config = config or ClassifierConfig(name="updrs_classifier")
        super().__init__(config)
        self.classifier = FeatureClassifier(config)

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        if checkpoint_path:
            self.classifier.load(checkpoint_path)
        self._is_loaded = True

    def predict(self, inputs: np.ndarray) -> InferenceResult:
        """Predict UPDRS level from features."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        predictions = self.classifier.predict(inputs)
        probas = self.classifier.predict_proba(inputs)
        confidence = np.max(probas, axis=1).mean()

        return InferenceResult(
            predictions=predictions,
            confidence=confidence,
            metadata={"probabilities": probas},
        )

    def classify_sequence(
        self,
        sequence: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Classify a sequence of features."""
        result = self.predict(sequence)
        return int(result.predictions[0]), result.metadata["probabilities"][0]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ClassificationMetrics:
        """Train classifier and return metrics."""
        self.classifier.fit(X, y)
        return self.classifier.evaluate(X, y)
