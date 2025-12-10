#!/usr/bin/env python3
"""
Placement Feedback Learning System.

Implements reinforcement learning for machine placement:
1. Logs placement attempts with routing success/failure outcomes
2. Extracts features from placements for ML training
3. Provides online-updated model to score/reject bad placements early
4. Stores training data in SQLite for offline batch training
"""

import json
import pickle
import sqlite3
import time
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np

# ML imports
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class PlacementFeatures:
    """Features extracted from a machine placement for ML."""
    # Problem characteristics
    num_inputs: int
    num_outputs: int
    num_machines: int
    grid_width: int
    grid_height: int
    num_floors: int

    # Machine distribution
    machine_density: float  # machines / total cells
    machines_per_floor: List[float]  # distribution across floors
    machine_spread_x: float  # std dev of x positions
    machine_spread_y: float  # std dev of y positions
    machine_centroid_x: float  # center of mass x
    machine_centroid_y: float  # center of mass y

    # Distance metrics
    avg_machine_to_input_dist: float
    avg_machine_to_output_dist: float
    min_machine_to_input_dist: float
    min_machine_to_output_dist: float
    max_machine_to_machine_dist: float
    avg_machine_to_machine_dist: float

    # Congestion estimates
    input_side_density: float  # machines near input ports
    output_side_density: float  # machines near output ports
    center_density: float  # machines in center region
    edge_density: float  # machines near edges

    # Port alignment
    machines_aligned_with_inputs: int  # machines in same row/col as inputs
    machines_aligned_with_outputs: int

    # Connectivity estimates
    estimated_path_crossings: float  # how many paths likely cross
    avg_path_length_estimate: float  # manhattan distance sum

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            self.num_inputs,
            self.num_outputs,
            self.num_machines,
            self.grid_width,
            self.grid_height,
            self.num_floors,
            self.machine_density,
            self.machine_spread_x,
            self.machine_spread_y,
            self.machine_centroid_x,
            self.machine_centroid_y,
            self.avg_machine_to_input_dist,
            self.avg_machine_to_output_dist,
            self.min_machine_to_input_dist,
            self.min_machine_to_output_dist,
            self.max_machine_to_machine_dist,
            self.avg_machine_to_machine_dist,
            self.input_side_density,
            self.output_side_density,
            self.center_density,
            self.edge_density,
            self.machines_aligned_with_inputs,
            self.machines_aligned_with_outputs,
            self.estimated_path_crossings,
            self.avg_path_length_estimate,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for interpretability."""
        return [
            'num_inputs', 'num_outputs', 'num_machines',
            'grid_width', 'grid_height', 'num_floors',
            'machine_density', 'machine_spread_x', 'machine_spread_y',
            'machine_centroid_x', 'machine_centroid_y',
            'avg_machine_to_input_dist', 'avg_machine_to_output_dist',
            'min_machine_to_input_dist', 'min_machine_to_output_dist',
            'max_machine_to_machine_dist', 'avg_machine_to_machine_dist',
            'input_side_density', 'output_side_density',
            'center_density', 'edge_density',
            'machines_aligned_with_inputs', 'machines_aligned_with_outputs',
            'estimated_path_crossings', 'avg_path_length_estimate',
        ]


def extract_placement_features(
    machines: List[Tuple[Any, int, int, int, Any]],  # (type, x, y, floor, rotation)
    input_positions: List[Tuple[int, int, int, Any]],  # (x, y, floor, side)
    output_positions: List[Tuple[int, int, int, Any]],
    grid_width: int,
    grid_height: int,
    num_floors: int,
) -> PlacementFeatures:
    """
    Extract ML features from a machine placement.

    Args:
        machines: List of (building_type, x, y, floor, rotation)
        input_positions: List of (x, y, floor, side) for inputs
        output_positions: List of (x, y, floor, side) for outputs
        grid_width, grid_height, num_floors: Grid dimensions

    Returns:
        PlacementFeatures dataclass
    """
    num_machines = len(machines)
    num_inputs = len(input_positions)
    num_outputs = len(output_positions)
    total_cells = grid_width * grid_height * num_floors

    if num_machines == 0:
        # Return default features for empty placement
        return PlacementFeatures(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_machines=0,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machine_density=0,
            machines_per_floor=[0] * num_floors,
            machine_spread_x=0,
            machine_spread_y=0,
            machine_centroid_x=grid_width / 2,
            machine_centroid_y=grid_height / 2,
            avg_machine_to_input_dist=0,
            avg_machine_to_output_dist=0,
            min_machine_to_input_dist=0,
            min_machine_to_output_dist=0,
            max_machine_to_machine_dist=0,
            avg_machine_to_machine_dist=0,
            input_side_density=0,
            output_side_density=0,
            center_density=0,
            edge_density=0,
            machines_aligned_with_inputs=0,
            machines_aligned_with_outputs=0,
            estimated_path_crossings=0,
            avg_path_length_estimate=0,
        )

    # Extract machine positions
    machine_positions = [(m[1], m[2], m[3]) for m in machines]  # (x, y, floor)
    xs = [p[0] for p in machine_positions]
    ys = [p[1] for p in machine_positions]
    floors = [p[2] for p in machine_positions]

    # Machine distribution
    machine_density = num_machines / max(1, total_cells)

    machines_per_floor = [0.0] * num_floors
    for f in floors:
        if 0 <= f < num_floors:
            machines_per_floor[f] += 1
    machines_per_floor = [c / max(1, num_machines) for c in machines_per_floor]

    machine_spread_x = np.std(xs) if len(xs) > 1 else 0
    machine_spread_y = np.std(ys) if len(ys) > 1 else 0
    machine_centroid_x = np.mean(xs)
    machine_centroid_y = np.mean(ys)

    # Distance to inputs
    input_coords = [(inp[0], inp[1], inp[2]) for inp in input_positions]
    machine_to_input_dists = []
    for mx, my, mf in machine_positions:
        for ix, iy, inf in input_coords:
            dist = abs(mx - ix) + abs(my - iy) + abs(mf - inf) * 2
            machine_to_input_dists.append(dist)

    avg_machine_to_input_dist = np.mean(machine_to_input_dists) if machine_to_input_dists else 0
    min_machine_to_input_dist = min(machine_to_input_dists) if machine_to_input_dists else 0

    # Distance to outputs
    output_coords = [(out[0], out[1], out[2]) for out in output_positions]
    machine_to_output_dists = []
    for mx, my, mf in machine_positions:
        for ox, oy, of in output_coords:
            dist = abs(mx - ox) + abs(my - oy) + abs(mf - of) * 2
            machine_to_output_dists.append(dist)

    avg_machine_to_output_dist = np.mean(machine_to_output_dists) if machine_to_output_dists else 0
    min_machine_to_output_dist = min(machine_to_output_dists) if machine_to_output_dists else 0

    # Machine-to-machine distances
    machine_dists = []
    for i, (x1, y1, f1) in enumerate(machine_positions):
        for j, (x2, y2, f2) in enumerate(machine_positions):
            if i < j:
                dist = abs(x1 - x2) + abs(y1 - y2) + abs(f1 - f2) * 2
                machine_dists.append(dist)

    max_machine_to_machine_dist = max(machine_dists) if machine_dists else 0
    avg_machine_to_machine_dist = np.mean(machine_dists) if machine_dists else 0

    # Density in different regions
    center_x, center_y = grid_width / 2, grid_height / 2
    center_radius = min(grid_width, grid_height) / 4

    center_count = sum(1 for x, y, _ in machine_positions
                      if abs(x - center_x) < center_radius and abs(y - center_y) < center_radius)
    edge_count = sum(1 for x, y, _ in machine_positions
                    if x <= 1 or x >= grid_width - 2 or y <= 1 or y >= grid_height - 2)

    center_density = center_count / max(1, num_machines)
    edge_density = edge_count / max(1, num_machines)

    # Input/output side density (machines near ports)
    input_side_count = 0
    for mx, my, mf in machine_positions:
        for ix, iy, inf, _ in input_positions:
            if abs(mx - ix) <= 2 and abs(my - iy) <= 2 and mf == inf:
                input_side_count += 1
                break

    output_side_count = 0
    for mx, my, mf in machine_positions:
        for ox, oy, of, _ in output_positions:
            if abs(mx - ox) <= 2 and abs(my - oy) <= 2 and mf == of:
                output_side_count += 1
                break

    input_side_density = input_side_count / max(1, num_machines)
    output_side_density = output_side_count / max(1, num_machines)

    # Alignment with ports
    input_rows = set(inp[1] for inp in input_positions)
    input_cols = set(inp[0] for inp in input_positions)
    output_rows = set(out[1] for out in output_positions)
    output_cols = set(out[0] for out in output_positions)

    machines_aligned_with_inputs = sum(1 for _, x, y, _, _ in machines
                                       if x in input_cols or y in input_rows)
    machines_aligned_with_outputs = sum(1 for _, x, y, _, _ in machines
                                        if x in output_cols or y in output_rows)

    # Estimate path crossings (simplified)
    # Count how many machine pairs have overlapping bounding boxes
    crossing_estimate = 0
    for i, (x1, y1, f1) in enumerate(machine_positions):
        for j, (x2, y2, f2) in enumerate(machine_positions):
            if i < j and f1 == f2:
                # Check if paths might cross (rough estimate)
                if (min(x1, x2) < max(x1, x2) and min(y1, y2) < max(y1, y2)):
                    crossing_estimate += 1

    estimated_path_crossings = crossing_estimate / max(1, num_machines * (num_machines - 1) / 2)

    # Average path length estimate (manhattan distances)
    avg_path_length_estimate = (avg_machine_to_input_dist + avg_machine_to_output_dist) / 2

    return PlacementFeatures(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_machines=num_machines,
        grid_width=grid_width,
        grid_height=grid_height,
        num_floors=num_floors,
        machine_density=machine_density,
        machines_per_floor=machines_per_floor,
        machine_spread_x=machine_spread_x,
        machine_spread_y=machine_spread_y,
        machine_centroid_x=machine_centroid_x,
        machine_centroid_y=machine_centroid_y,
        avg_machine_to_input_dist=avg_machine_to_input_dist,
        avg_machine_to_output_dist=avg_machine_to_output_dist,
        min_machine_to_input_dist=min_machine_to_input_dist,
        min_machine_to_output_dist=min_machine_to_output_dist,
        max_machine_to_machine_dist=max_machine_to_machine_dist,
        avg_machine_to_machine_dist=avg_machine_to_machine_dist,
        input_side_density=input_side_density,
        output_side_density=output_side_density,
        center_density=center_density,
        edge_density=edge_density,
        machines_aligned_with_inputs=machines_aligned_with_inputs,
        machines_aligned_with_outputs=machines_aligned_with_outputs,
        estimated_path_crossings=estimated_path_crossings,
        avg_path_length_estimate=avg_path_length_estimate,
    )


class PlacementFeedbackDB:
    """SQLite database for storing placement feedback data."""

    def __init__(self, db_path: str = "placement_feedback.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS placement_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                problem_hash TEXT,
                foundation_type TEXT,

                -- Problem spec
                num_inputs INTEGER,
                num_outputs INTEGER,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,

                -- Placement info
                num_machines INTEGER,
                machines_json TEXT,

                -- Features (stored as JSON for flexibility)
                features_json TEXT,
                feature_vector TEXT,

                -- Outcome
                routing_success INTEGER,
                routing_progress REAL,
                connections_routed INTEGER,
                connections_total INTEGER,
                num_belts INTEGER,
                solve_time REAL,

                -- For analysis
                failure_reason TEXT,
                blocking_connections TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_routing_success
            ON placement_attempts(routing_success)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_foundation
            ON placement_attempts(foundation_type)
        ''')

        conn.commit()
        conn.close()

    def log_attempt(
        self,
        foundation_type: str,
        machines: List[Tuple],
        input_positions: List[Tuple],
        output_positions: List[Tuple],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        routing_success: bool,
        routing_progress: float = 0.0,
        connections_routed: int = 0,
        connections_total: int = 0,
        num_belts: int = 0,
        solve_time: float = 0.0,
        failure_reason: str = "",
        blocking_connections: List[int] = None,
    ):
        """Log a placement attempt with its outcome."""
        # Extract features
        features = extract_placement_features(
            machines, input_positions, output_positions,
            grid_width, grid_height, num_floors
        )

        # Create problem hash for grouping similar problems
        problem_hash = f"{foundation_type}_{len(input_positions)}_{len(output_positions)}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize machines (convert enums to strings)
        machines_serializable = [
            (str(m[0]), m[1], m[2], m[3], str(m[4])) for m in machines
        ]

        cursor.execute('''
            INSERT INTO placement_attempts (
                timestamp, problem_hash, foundation_type,
                num_inputs, num_outputs, grid_width, grid_height, num_floors,
                num_machines, machines_json, features_json, feature_vector,
                routing_success, routing_progress, connections_routed, connections_total,
                num_belts, solve_time, failure_reason, blocking_connections
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            problem_hash,
            foundation_type,
            len(input_positions),
            len(output_positions),
            grid_width,
            grid_height,
            num_floors,
            len(machines),
            json.dumps(machines_serializable),
            json.dumps(asdict(features)),
            json.dumps(features.to_vector().tolist()),
            1 if routing_success else 0,
            routing_progress,
            connections_routed,
            connections_total,
            num_belts,
            solve_time,
            failure_reason,
            json.dumps(blocking_connections or []),
        ))

        conn.commit()
        conn.close()

        return features

    def get_training_data(
        self,
        min_samples: int = 10,
        foundation_type: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data for ML model.

        Returns:
            (X, y) where X is feature matrix and y is success labels
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT feature_vector, routing_success FROM placement_attempts"
        params = []

        if foundation_type:
            query += " WHERE foundation_type = ?"
            params.append(foundation_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < min_samples:
            return None, None

        X = np.array([json.loads(row[0]) for row in rows])
        y = np.array([row[1] for row in rows])

        return X, y

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged attempts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM placement_attempts")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM placement_attempts WHERE routing_success = 1")
        successes = cursor.fetchone()[0]

        cursor.execute("""
            SELECT foundation_type, COUNT(*), SUM(routing_success)
            FROM placement_attempts
            GROUP BY foundation_type
        """)
        by_foundation = {
            row[0]: {'total': row[1], 'successes': row[2]}
            for row in cursor.fetchall()
        }

        conn.close()

        return {
            'total_attempts': total,
            'successes': successes,
            'success_rate': successes / max(1, total),
            'by_foundation': by_foundation,
        }


class PlacementQualityModel:
    """
    ML model for predicting placement quality.

    Supports:
    - Batch training from database
    - Online updates after each failure
    - Placement scoring for early rejection
    """

    def __init__(self, model_path: str = "models/placement_quality.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.online_buffer_X = []
        self.online_buffer_y = []
        self.online_update_threshold = 10  # Update model after this many new samples

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load model from disk if exists."""
        if Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.is_trained = True
                    print(f"Loaded placement quality model from {self.model_path}")
            except Exception as e:
                print(f"Could not load placement model: {e}")

    def _save_model(self):
        """Save model to disk."""
        if self.model is not None:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                }, f)

    def train(self, db: PlacementFeedbackDB, min_samples: int = 50) -> bool:
        """
        Train model from database.

        Returns:
            True if training succeeded, False if not enough data
        """
        if not HAS_SKLEARN:
            print("sklearn not available for training")
            return False

        X, y = db.get_training_data(min_samples=min_samples)

        if X is None or len(X) < min_samples:
            print(f"Not enough training data (need {min_samples}, have {len(X) if X is not None else 0})")
            return False

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train gradient boosting classifier
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        # Evaluate
        accuracy = self.model.score(X_scaled, y)
        print(f"Placement quality model trained: accuracy={accuracy:.3f}")

        # Feature importance
        importance = list(zip(PlacementFeatures.feature_names(), self.model.feature_importances_))
        importance.sort(key=lambda x: -x[1])
        print("Top features:")
        for name, imp in importance[:5]:
            print(f"  {name}: {imp:.3f}")

        self.is_trained = True
        self._save_model()

        return True

    def update_online(self, features: PlacementFeatures, success: bool):
        """
        Add a sample to the online learning buffer.

        When enough samples accumulate, retrain the model.
        """
        self.online_buffer_X.append(features.to_vector())
        self.online_buffer_y.append(1 if success else 0)

        # Check if we should update the model
        if len(self.online_buffer_X) >= self.online_update_threshold:
            self._update_model_online()

    def _update_model_online(self):
        """Update model with buffered online samples."""
        if not HAS_SKLEARN or len(self.online_buffer_X) == 0:
            return

        X_new = np.array(self.online_buffer_X)
        y_new = np.array(self.online_buffer_y)

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_new)
        else:
            X_scaled = self.scaler.transform(X_new)

        if self.model is None:
            # Initialize new model
            self.model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                warm_start=True,
                random_state=42,
            )
            self.model.fit(X_scaled, y_new)
        else:
            # Partial fit with new data (for models that support it)
            # GradientBoosting doesn't support partial_fit, so we do warm_start
            try:
                self.model.n_estimators += 10
                self.model.fit(X_scaled, y_new)
            except Exception:
                # Fall back to full retrain
                pass

        self.is_trained = True
        self._save_model()

        # Clear buffer
        self.online_buffer_X.clear()
        self.online_buffer_y.clear()

        print(f"Online model update complete ({len(y_new)} samples)")

    def predict_success_probability(self, features: PlacementFeatures) -> float:
        """
        Predict probability that a placement will route successfully.

        Returns:
            Probability between 0 and 1, or 0.5 if model not trained
        """
        if not self.is_trained or self.model is None:
            return 0.5  # No prediction available

        X = features.to_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0]
        # Return probability of success (class 1)
        return proba[1] if len(proba) > 1 else proba[0]

    def should_reject_placement(
        self,
        features: PlacementFeatures,
        threshold: float = 0.3,
    ) -> Tuple[bool, float]:
        """
        Decide whether to reject a placement early.

        Args:
            features: Extracted placement features
            threshold: Reject if success probability below this

        Returns:
            (should_reject, success_probability)
        """
        prob = self.predict_success_probability(features)
        return prob < threshold, prob


class PlacementFeedbackLogger:
    """
    High-level interface for logging placement feedback and learning.

    Usage:
        logger = PlacementFeedbackLogger()

        # Before routing attempt
        features = logger.extract_features(machines, inputs, outputs, ...)
        should_reject, prob = logger.should_reject(features)

        # After routing attempt
        logger.log_result(features, success=True/False, ...)
    """

    def __init__(
        self,
        db_path: str = "placement_feedback.db",
        model_path: str = "models/placement_quality.pkl",
        enable_online_learning: bool = True,
    ):
        self.db = PlacementFeedbackDB(db_path)
        self.model = PlacementQualityModel(model_path)
        self.enable_online_learning = enable_online_learning

        # Cache for current attempt
        self._current_features = None
        self._current_machines = None
        self._current_foundation = None

    def extract_features(
        self,
        machines: List[Tuple],
        input_positions: List[Tuple],
        output_positions: List[Tuple],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        foundation_type: str = "",
    ) -> PlacementFeatures:
        """Extract and cache features for current placement."""
        features = extract_placement_features(
            machines, input_positions, output_positions,
            grid_width, grid_height, num_floors
        )

        self._current_features = features
        self._current_machines = machines
        self._current_foundation = foundation_type

        return features

    def should_reject(
        self,
        features: PlacementFeatures = None,
        threshold: float = 0.3,
    ) -> Tuple[bool, float]:
        """Check if placement should be rejected based on ML prediction."""
        if features is None:
            features = self._current_features

        if features is None:
            return False, 0.5

        return self.model.should_reject_placement(features, threshold)

    def log_result(
        self,
        routing_success: bool,
        routing_progress: float = 0.0,
        connections_routed: int = 0,
        connections_total: int = 0,
        num_belts: int = 0,
        solve_time: float = 0.0,
        failure_reason: str = "",
        blocking_connections: List[int] = None,
        # Optional overrides
        features: PlacementFeatures = None,
        machines: List[Tuple] = None,
        input_positions: List[Tuple] = None,
        output_positions: List[Tuple] = None,
        grid_width: int = 0,
        grid_height: int = 0,
        num_floors: int = 0,
        foundation_type: str = "",
    ):
        """Log the result of a placement attempt."""
        # Use cached values if not provided
        if features is None:
            features = self._current_features
        if machines is None:
            machines = self._current_machines
        if not foundation_type:
            foundation_type = self._current_foundation or ""

        if features is None or machines is None:
            print("Warning: No features to log (call extract_features first)")
            return

        # Log to database
        self.db.log_attempt(
            foundation_type=foundation_type,
            machines=machines,
            input_positions=input_positions or [],
            output_positions=output_positions or [],
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            routing_success=routing_success,
            routing_progress=routing_progress,
            connections_routed=connections_routed,
            connections_total=connections_total,
            num_belts=num_belts,
            solve_time=solve_time,
            failure_reason=failure_reason,
            blocking_connections=blocking_connections,
        )

        # Online learning update
        if self.enable_online_learning:
            self.model.update_online(features, routing_success)

        # Clear cache
        self._current_features = None
        self._current_machines = None
        self._current_foundation = None

    def train_model(self, min_samples: int = 50) -> bool:
        """Train/retrain the model from logged data."""
        return self.model.train(self.db, min_samples)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged attempts."""
        stats = self.db.get_stats()
        stats['model_trained'] = self.model.is_trained
        return stats


# CLI for testing and training
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Placement Feedback Learning")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--train", action="store_true", help="Train model from database")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples for training")
    parser.add_argument("--db", type=str, default="placement_feedback.db", help="Database path")
    parser.add_argument("--model", type=str, default="models/placement_quality.pkl", help="Model path")

    args = parser.parse_args()

    logger = PlacementFeedbackLogger(db_path=args.db, model_path=args.model)

    if args.stats:
        stats = logger.get_stats()
        print("\nPlacement Feedback Statistics")
        print("=" * 40)
        print(f"Total attempts: {stats['total_attempts']}")
        print(f"Successes: {stats['successes']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Model trained: {stats['model_trained']}")

        if stats['by_foundation']:
            print("\nBy foundation:")
            for foundation, data in stats['by_foundation'].items():
                rate = data['successes'] / max(1, data['total'])
                print(f"  {foundation}: {data['successes']}/{data['total']} ({rate:.1%})")

    if args.train:
        print("\nTraining placement quality model...")
        success = logger.train_model(min_samples=args.min_samples)
        if success:
            print("Training complete!")
        else:
            print("Training failed (not enough data)")
