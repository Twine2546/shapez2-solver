"""
Model training and inference for routing quality prediction.

Supports multiple backends:
- scikit-learn (default, always available)
- XGBoost (preferred if installed)
- Simple heuristic fallback
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
import math

from .features import SolutionFeatures, ConnectionFeatures


# Try to import ML libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class ModelMetrics:
    """Metrics from model training/evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mse: float = 0.0
    r2: float = 0.0
    cross_val_mean: float = 0.0
    cross_val_std: float = 0.0
    feature_importances: Optional[Dict[str, float]] = None


class HeuristicPredictor:
    """
    Simple heuristic-based predictor when ML libraries aren't available.
    Uses hand-tuned rules based on feature patterns.
    """

    def predict_success(self, features: SolutionFeatures) -> float:
        """Predict routing success probability (0-1)."""
        score = 0.7  # Base probability

        # Penalize high congestion
        if features.max_local_density_3x3 > 0.6:
            score -= 0.3
        elif features.max_local_density_3x3 > 0.4:
            score -= 0.15

        # Penalize many connections
        if features.num_connections > 6:
            score -= 0.1 * (features.num_connections - 6)

        # Reward good separation
        if features.input_output_separation > 0.5:
            score += 0.1

        # Penalize high crossing potential
        if features.crossing_potential > 0.5:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def predict_throughput(self, features: SolutionFeatures) -> float:
        """Predict throughput (items/min)."""
        if features.num_outputs == 0:
            return 0.0

        # Base throughput per output
        base = 180.0 / features.num_outputs

        # Reduce for congestion
        congestion_factor = 1.0 - features.max_local_density_3x3 * 0.5

        return base * congestion_factor

    def predict_difficulty(self, features: ConnectionFeatures) -> float:
        """Predict connection routing difficulty (0-1, higher = harder)."""
        difficulty = 0.3  # Base difficulty

        # Distance increases difficulty
        difficulty += features.normalized_distance * 0.3

        # Congestion increases difficulty
        difficulty += features.src_local_density * 0.15
        difficulty += features.dst_local_density * 0.15

        # Crossing center increases difficulty
        if features.crosses_center:
            difficulty += 0.1

        # Floor changes increase difficulty
        difficulty += features.floor_change * 0.1

        return max(0.0, min(1.0, difficulty))


class QualityPredictor:
    """
    Predicts routing solution quality.

    Can predict:
    - routing_success: Will all connections route successfully?
    - throughput: Expected throughput (items/min)
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor."""
        self.success_model = None
        self.throughput_model = None
        self.scaler = None
        self.heuristic = HeuristicPredictor()
        self.feature_names = SolutionFeatures.feature_names()
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        X: List[List[float]],
        y_success: List[bool],
        y_throughput: List[float],
        use_xgboost: bool = True,
    ) -> ModelMetrics:
        """
        Train the predictor on labeled data.

        Args:
            X: Feature vectors from SolutionFeatures.to_feature_vector()
            y_success: Success labels (True/False)
            y_throughput: Throughput values

        Returns:
            Training metrics
        """
        if not HAS_SKLEARN:
            print("Warning: scikit-learn not installed. Using heuristic predictor.")
            return ModelMetrics()

        if not HAS_NUMPY:
            print("Warning: numpy not installed. Using heuristic predictor.")
            return ModelMetrics()

        X = np.array(X)
        y_success = np.array(y_success, dtype=int)
        y_throughput = np.array(y_throughput)

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_success_train, y_success_test, y_tp_train, y_tp_test = \
            train_test_split(X_scaled, y_success, y_throughput, test_size=0.2, random_state=42)

        metrics = ModelMetrics()

        # Train success classifier
        if use_xgboost and HAS_XGBOOST:
            self.success_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
            )
        else:
            self.success_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )

        self.success_model.fit(X_train, y_success_train)

        # Evaluate success model
        y_pred = self.success_model.predict(X_test)
        metrics.accuracy = accuracy_score(y_success_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.success_model, X_scaled, y_success, cv=5)
        metrics.cross_val_mean = cv_scores.mean()
        metrics.cross_val_std = cv_scores.std()

        # Feature importances
        if hasattr(self.success_model, 'feature_importances_'):
            importances = self.success_model.feature_importances_
            metrics.feature_importances = {
                name: float(imp)
                for name, imp in zip(self.feature_names, importances)
            }

        # Train throughput regressor (only on successful examples)
        success_mask_train = y_success_train == 1
        success_mask_test = y_success_test == 1

        if success_mask_train.sum() > 10:
            if use_xgboost and HAS_XGBOOST:
                self.throughput_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                )
            else:
                self.throughput_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                )

            self.throughput_model.fit(X_train[success_mask_train], y_tp_train[success_mask_train])

            if success_mask_test.sum() > 0:
                y_tp_pred = self.throughput_model.predict(X_test[success_mask_test])
                metrics.mse = mean_squared_error(y_tp_test[success_mask_test], y_tp_pred)

        self.is_trained = True
        return metrics

    def predict(self, features: SolutionFeatures) -> Tuple[float, float]:
        """
        Predict success probability and throughput.

        Args:
            features: Extracted solution features

        Returns:
            (success_probability, predicted_throughput)
        """
        if not self.is_trained or not HAS_SKLEARN:
            # Fall back to heuristic
            return (
                self.heuristic.predict_success(features),
                self.heuristic.predict_throughput(features),
            )

        X = np.array([features.to_feature_vector()])
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        # Predict success probability
        if hasattr(self.success_model, 'predict_proba'):
            success_prob = self.success_model.predict_proba(X_scaled)[0, 1]
        else:
            success_prob = float(self.success_model.predict(X_scaled)[0])

        # Predict throughput
        if self.throughput_model is not None:
            throughput = float(self.throughput_model.predict(X_scaled)[0])
        else:
            throughput = self.heuristic.predict_throughput(features)

        return success_prob, throughput

    def predict_batch(
        self,
        features_list: List[SolutionFeatures]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple solutions."""
        return [self.predict(f) for f in features_list]

    def save(self, path: str):
        """Save trained model to file."""
        data = {
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
        }

        if self.is_trained and HAS_SKLEARN:
            data['success_model'] = self.success_model
            data['throughput_model'] = self.throughput_model
            data['scaler'] = self.scaler

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load trained model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.is_trained = data.get('is_trained', False)
        self.feature_names = data.get('feature_names', SolutionFeatures.feature_names())

        if self.is_trained:
            self.success_model = data.get('success_model')
            self.throughput_model = data.get('throughput_model')
            self.scaler = data.get('scaler')


class DifficultyPredictor:
    """
    Predicts routing difficulty for individual connections.
    Used to order connections for sequential routing.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor."""
        self.model = None
        self.scaler = None
        self.heuristic = HeuristicPredictor()
        self.feature_names = ConnectionFeatures.feature_names()
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        X: List[List[float]],
        y: List[bool],  # True = routed successfully
    ) -> ModelMetrics:
        """
        Train difficulty predictor.

        Learns to predict which connections are harder to route.
        """
        if not HAS_SKLEARN or not HAS_NUMPY:
            return ModelMetrics()

        X = np.array(X)
        y = np.array(y, dtype=int)

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                random_state=42,
            )

        self.model.fit(X_train, y_train)

        metrics = ModelMetrics()
        y_pred = self.model.predict(X_test)
        metrics.accuracy = accuracy_score(y_test, y_pred)

        if hasattr(self.model, 'feature_importances_'):
            metrics.feature_importances = {
                name: float(imp)
                for name, imp in zip(self.feature_names, self.model.feature_importances_)
            }

        self.is_trained = True
        return metrics

    def predict(self, features: ConnectionFeatures) -> float:
        """
        Predict routing difficulty (0-1, higher = harder).

        Returns probability of failure, which can be used as difficulty score.
        """
        if not self.is_trained or not HAS_SKLEARN:
            return self.heuristic.predict_difficulty(features)

        X = np.array([features.to_feature_vector()])
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, 'predict_proba'):
            # Return probability of failure (class 0)
            probs = self.model.predict_proba(X_scaled)[0]
            return float(probs[0]) if len(probs) > 1 else 1.0 - float(probs[0])
        else:
            return 1.0 - float(self.model.predict(X_scaled)[0])

    def rank_connections(
        self,
        connections: List[Tuple],
        grid_width: int,
        grid_height: int,
        occupied: set,
    ) -> List[int]:
        """
        Rank connections by difficulty (easiest first).

        Returns indices sorted by predicted difficulty.
        """
        from .features import extract_connection_features

        difficulties = []
        for i, conn in enumerate(connections):
            features = extract_connection_features(
                connection=conn,
                grid_width=grid_width,
                grid_height=grid_height,
                occupied=occupied,
                connection_index=i,
                total_connections=len(connections),
            )
            difficulty = self.predict(features)
            difficulties.append((i, difficulty))

        # Sort by difficulty (easiest first)
        difficulties.sort(key=lambda x: x[1])
        return [idx for idx, _ in difficulties]

    def save(self, path: str):
        """Save model to file."""
        data = {
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'model': self.model,
            'scaler': self.scaler,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.is_trained = data.get('is_trained', False)
        self.feature_names = data.get('feature_names', ConnectionFeatures.feature_names())
        self.model = data.get('model')
        self.scaler = data.get('scaler')


def train_quality_model(
    db_path: str = "routing_data.db",
    model_path: str = "quality_model.pkl",
) -> Tuple[QualityPredictor, ModelMetrics]:
    """
    Train quality predictor from logged data.

    Args:
        db_path: Path to SQLite database with routing attempts
        model_path: Where to save trained model

    Returns:
        (trained_predictor, metrics)
    """
    from .data_logger import DataStore

    store = DataStore(db_path)
    X, y_success, y_throughput = store.export_feature_vectors()

    if len(X) < 10:
        print(f"Warning: Only {len(X)} samples. Need more data for reliable training.")

    predictor = QualityPredictor()
    metrics = predictor.train(X, y_success, y_throughput)
    predictor.save(model_path)

    print(f"Trained quality model on {len(X)} samples")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Cross-val: {metrics.cross_val_mean:.3f} (+/- {metrics.cross_val_std:.3f})")

    if metrics.feature_importances:
        print("\nTop 10 important features:")
        sorted_features = sorted(
            metrics.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for name, imp in sorted_features:
            print(f"  {name}: {imp:.4f}")

    return predictor, metrics


def train_difficulty_model(
    db_path: str = "routing_data.db",
    model_path: str = "difficulty_model.pkl",
) -> Tuple[DifficultyPredictor, ModelMetrics]:
    """
    Train difficulty predictor from logged data.

    Args:
        db_path: Path to SQLite database
        model_path: Where to save trained model

    Returns:
        (trained_predictor, metrics)
    """
    from .data_logger import DataStore

    store = DataStore(db_path)
    attempts = store.get_attempts(limit=100000)

    X = []
    y = []

    for attempt in attempts:
        if attempt.connection_features:
            for cf in attempt.connection_features:
                features = ConnectionFeatures(**cf)
                X.append(features.to_feature_vector())
                y.append(features.routed_successfully)

    if len(X) < 10:
        print(f"Warning: Only {len(X)} connection samples.")

    predictor = DifficultyPredictor()
    metrics = predictor.train(X, y)
    predictor.save(model_path)

    print(f"Trained difficulty model on {len(X)} connections")
    print(f"Accuracy: {metrics.accuracy:.3f}")

    return predictor, metrics
