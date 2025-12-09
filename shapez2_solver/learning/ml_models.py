#!/usr/bin/env python3
"""
ML Models for Shapez 2 Routing Optimization.

Phase 1: Difficulty/Solvability Classifier
  - Predicts whether a problem is solvable
  - Predicts difficulty level (easy/medium/hard/extreme)
  - Used to route problems to appropriate solver mode

Phase 2: Policy Network for Belt Direction
  - Predicts best belt direction at each position
  - Used as A* heuristic or CP-SAT warm start
"""

import json
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Using sklearn fallback.")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available.")


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class ProblemFeatures:
    """Features extracted from a routing problem for ML."""
    # Grid features
    grid_width: int
    grid_height: int
    num_floors: int
    grid_area: int
    grid_volume: int

    # Machine features
    num_machines: int
    machine_density: float  # machines per cell

    # Connection features
    num_connections: int
    avg_manhattan_distance: float
    max_manhattan_distance: float
    min_manhattan_distance: float

    # I/O features
    num_inputs: int
    num_outputs: int
    io_spread: float  # how spread out I/O is

    # Foundation features
    foundation_type: str
    is_rectangular: bool
    is_multi_floor: bool

    # Complexity estimates
    connection_density: float  # connections per cell
    estimated_belt_length: float
    crossing_potential: float  # how likely paths cross

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            self.grid_width,
            self.grid_height,
            self.num_floors,
            self.grid_area,
            self.grid_volume,
            self.num_machines,
            self.machine_density,
            self.num_connections,
            self.avg_manhattan_distance,
            self.max_manhattan_distance,
            self.min_manhattan_distance,
            self.num_inputs,
            self.num_outputs,
            self.io_spread,
            1 if self.is_rectangular else 0,
            1 if self.is_multi_floor else 0,
            self.connection_density,
            self.estimated_belt_length,
            self.crossing_potential,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for interpretability."""
        return [
            'grid_width', 'grid_height', 'num_floors', 'grid_area', 'grid_volume',
            'num_machines', 'machine_density', 'num_connections',
            'avg_manhattan_dist', 'max_manhattan_dist', 'min_manhattan_dist',
            'num_inputs', 'num_outputs', 'io_spread',
            'is_rectangular', 'is_multi_floor',
            'connection_density', 'estimated_belt_length', 'crossing_potential',
        ]


def extract_features(problem_dict: Dict[str, Any]) -> ProblemFeatures:
    """
    Extract ML features from a problem dictionary.

    Args:
        problem_dict: Problem dictionary from database

    Returns:
        ProblemFeatures instance
    """
    grid_w = problem_dict['grid_width']
    grid_h = problem_dict['grid_height']
    num_floors = problem_dict['num_floors']
    grid_area = grid_w * grid_h
    grid_volume = grid_area * num_floors

    num_machines = problem_dict['num_machines']
    machine_density = num_machines / max(1, grid_volume)

    connections = problem_dict.get('connections', [])
    num_connections = len(connections)

    # Calculate manhattan distances
    manhattan_dists = []
    for conn in connections:
        src = conn['src']
        dst = conn['dst']
        dist = abs(src[0] - dst[0]) + abs(src[1] - dst[1]) + abs(src[2] - dst[2])
        manhattan_dists.append(dist)

    avg_dist = np.mean(manhattan_dists) if manhattan_dists else 0
    max_dist = max(manhattan_dists) if manhattan_dists else 0
    min_dist = min(manhattan_dists) if manhattan_dists else 0

    # I/O features
    inputs = problem_dict.get('input_positions', [])
    outputs = problem_dict.get('output_positions', [])

    # Calculate I/O spread (how spread out they are)
    io_spread = 0
    if inputs:
        input_ys = [p[1] if isinstance(p, (list, tuple)) else p['y'] for p in inputs]
        io_spread = max(input_ys) - min(input_ys) if len(input_ys) > 1 else 0

    # Foundation info
    foundation_type = problem_dict['foundation_type']
    is_rectangular = foundation_type in ['1x1', '2x1', '1x2', '2x2', '3x1', '1x3', '3x2', '2x3', '3x3', '4x2', '2x4']
    is_multi_floor = num_floors > 1

    # Complexity estimates
    connection_density = num_connections / max(1, grid_volume)
    estimated_belt_length = sum(manhattan_dists) if manhattan_dists else 0

    # Crossing potential: higher when paths likely cross
    # Simple heuristic: more connections + smaller area = more crossings
    crossing_potential = (num_connections ** 2) / max(1, grid_area)

    return ProblemFeatures(
        grid_width=grid_w,
        grid_height=grid_h,
        num_floors=num_floors,
        grid_area=grid_area,
        grid_volume=grid_volume,
        num_machines=num_machines,
        machine_density=machine_density,
        num_connections=num_connections,
        avg_manhattan_distance=avg_dist,
        max_manhattan_distance=max_dist,
        min_manhattan_distance=min_dist,
        num_inputs=len(inputs),
        num_outputs=len(outputs),
        io_spread=io_spread,
        foundation_type=foundation_type,
        is_rectangular=is_rectangular,
        is_multi_floor=is_multi_floor,
        connection_density=connection_density,
        estimated_belt_length=estimated_belt_length,
        crossing_potential=crossing_potential,
    )


# =============================================================================
# Phase 1: Solvability & Difficulty Classifier
# =============================================================================

class SolvabilityClassifier:
    """
    Predicts whether a routing problem is solvable and its difficulty.

    Uses Random Forest or Gradient Boosting for interpretability and robustness.
    """

    DIFFICULTY_LABELS = ['easy', 'medium', 'hard', 'extreme']

    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for SolvabilityClassifier")

        self.solvability_model = None
        self.difficulty_model = None
        self.solve_time_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def load_training_data(self, db_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data from the synthetic database.

        Returns:
            X: Feature matrix
            y_solvable: Solvability labels (0/1)
            y_difficulty: Difficulty labels (0-3)
            y_time: Solve time (for regression)
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.problem_json, r.success, r.solve_time, p.difficulty
            FROM synthetic_problems p
            JOIN solver_results r ON p.problem_id = r.problem_id
        """)

        X_list = []
        y_solvable = []
        y_difficulty = []
        y_time = []

        for row in cursor.fetchall():
            problem = json.loads(row[0])
            features = extract_features(problem)

            X_list.append(features.to_vector())
            y_solvable.append(1 if row[1] else 0)
            y_difficulty.append(self.DIFFICULTY_LABELS.index(row[3]))
            y_time.append(row[2])

        conn.close()

        return (
            np.array(X_list),
            np.array(y_solvable),
            np.array(y_difficulty),
            np.array(y_time),
        )

    def train(self, db_path: str, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the classifier models.

        Args:
            db_path: Path to synthetic training database
            test_size: Fraction of data for testing

        Returns:
            Dict with training metrics
        """
        print("Loading training data...")
        X, y_solvable, y_difficulty, y_time = self.load_training_data(db_path)

        print(f"Loaded {len(X)} examples")
        print(f"  Solvable: {sum(y_solvable)} / {len(y_solvable)}")
        print(f"  Difficulty distribution: {np.bincount(y_difficulty)}")

        # Split data
        X_train, X_test, y_solv_train, y_solv_test, y_diff_train, y_diff_test, y_time_train, y_time_test = \
            train_test_split(X, y_solvable, y_difficulty, y_time, test_size=test_size, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        metrics = {}

        # Train solvability classifier
        print("\nTraining solvability classifier...")
        self.solvability_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        self.solvability_model.fit(X_train_scaled, y_solv_train)

        solv_pred = self.solvability_model.predict(X_test_scaled)
        solv_acc = np.mean(solv_pred == y_solv_test)
        print(f"  Solvability accuracy: {solv_acc:.3f}")
        metrics['solvability_accuracy'] = solv_acc

        # Train difficulty classifier
        print("\nTraining difficulty classifier...")
        self.difficulty_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        self.difficulty_model.fit(X_train_scaled, y_diff_train)

        diff_pred = self.difficulty_model.predict(X_test_scaled)
        diff_acc = np.mean(diff_pred == y_diff_test)
        print(f"  Difficulty accuracy: {diff_acc:.3f}")
        metrics['difficulty_accuracy'] = diff_acc

        # Feature importance
        feature_names = ProblemFeatures.feature_names()
        importances = self.solvability_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        print("\nTop features for solvability:")
        for i in sorted_idx[:5]:
            print(f"  {feature_names[i]}: {importances[i]:.3f}")

        metrics['feature_importances'] = {
            feature_names[i]: float(importances[i]) for i in sorted_idx
        }

        self.is_trained = True
        return metrics

    def predict(self, problem_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict solvability and difficulty for a problem.

        Args:
            problem_dict: Problem dictionary

        Returns:
            Dict with predictions:
                - solvable: bool
                - solvable_prob: float (0-1)
                - difficulty: str
                - difficulty_probs: dict
                - recommended_mode: str
                - recommended_timeout: float
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        features = extract_features(problem_dict)
        X = features.to_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Solvability prediction
        solvable_prob = self.solvability_model.predict_proba(X_scaled)[0][1]
        solvable = solvable_prob > 0.5

        # Difficulty prediction
        diff_probs = self.difficulty_model.predict_proba(X_scaled)[0]
        diff_idx = np.argmax(diff_probs)
        difficulty = self.DIFFICULTY_LABELS[diff_idx]

        # Recommendations
        if not solvable:
            recommended_mode = 'skip'
            recommended_timeout = 0
        elif difficulty == 'easy':
            recommended_mode = 'astar'
            recommended_timeout = 5
        elif difficulty == 'medium':
            recommended_mode = 'hybrid'
            recommended_timeout = 15
        elif difficulty == 'hard':
            recommended_mode = 'global'
            recommended_timeout = 30
        else:  # extreme
            recommended_mode = 'global'
            recommended_timeout = 60

        return {
            'solvable': bool(solvable),
            'solvable_prob': float(solvable_prob),
            'difficulty': difficulty,
            'difficulty_probs': {
                self.DIFFICULTY_LABELS[i]: float(p) for i, p in enumerate(diff_probs)
            },
            'recommended_mode': recommended_mode,
            'recommended_timeout': recommended_timeout,
        }

    def save(self, path: str):
        """Save trained models to file."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        with open(path, 'wb') as f:
            pickle.dump({
                'solvability_model': self.solvability_model,
                'difficulty_model': self.difficulty_model,
                'scaler': self.scaler,
            }, f)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load trained models from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.solvability_model = data['solvability_model']
        self.difficulty_model = data['difficulty_model']
        self.scaler = data['scaler']
        self.is_trained = True
        print(f"Loaded model from {path}")


# =============================================================================
# Phase 2: Policy Network for Belt Direction
# =============================================================================

# Direction encoding used by all models
# Extended to 11 classes: 4 belt + 4 jump + up + down + none
DIRECTIONS = [
    'north', 'east', 'south', 'west',           # Regular belt (0-3)
    'jump_north', 'jump_east', 'jump_south', 'jump_west',  # Belt port jumps (4-7)
    'up', 'down',                                # Floor changes (8-9)
    'none',                                      # No movement (10)
]

# Movement deltas (x, y, floor)
DIR_TO_DELTA = {
    'north': (0, -1, 0),
    'east': (1, 0, 0),
    'south': (0, 1, 0),
    'west': (-1, 0, 0),
    'jump_north': (0, -2, 0),  # Min jump distance
    'jump_east': (2, 0, 0),
    'jump_south': (0, 2, 0),
    'jump_west': (-2, 0, 0),
    'up': (0, 0, 1),
    'down': (0, 0, -1),
    'none': (0, 0, 0),
}

# For backward compatibility
DIRECTIONS_SIMPLE = ['north', 'east', 'south', 'west', 'none']
DIR_TO_DELTA_2D = {
    'north': (0, -1),
    'east': (1, 0),
    'south': (0, 1),
    'west': (-1, 0),
    'none': (0, 0),
}


class DirectionPredictor:
    """
    Sklearn-based direction predictor (fallback when PyTorch unavailable).

    Uses per-cell features to predict belt direction using Random Forest.
    """

    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required")

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_cell_features(
        self,
        x: int,
        y: int,
        grid_w: int,
        grid_h: int,
        occupied: Set[Tuple[int, int]],
        inputs: List[Tuple[int, int]],
        outputs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Extract features for a single cell."""
        max_dist = grid_w + grid_h

        # Distance to nearest input
        min_src_dist = max_dist
        for (ix, iy) in inputs:
            d = abs(x - ix) + abs(y - iy)
            min_src_dist = min(min_src_dist, d)

        # Distance to nearest output
        min_dst_dist = max_dist
        for (ox, oy) in outputs:
            d = abs(x - ox) + abs(y - oy)
            min_dst_dist = min(min_dst_dist, d)

        # Average output direction
        if outputs:
            avg_out_x = np.mean([ox for ox, oy in outputs])
            avg_out_y = np.mean([oy for ox, oy in outputs])
            dx_to_out = avg_out_x - x
            dy_to_out = avg_out_y - y
        else:
            dx_to_out = 0
            dy_to_out = 0

        # Neighbor occupancy (8 neighbors)
        neighbors_occupied = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in occupied:
                    neighbors_occupied += 1

        return np.array([
            x / grid_w,  # Normalized position
            y / grid_h,
            min_src_dist / max_dist,  # Normalized distances
            min_dst_dist / max_dist,
            dx_to_out / max(1, grid_w),  # Direction to output
            dy_to_out / max(1, grid_h),
            1 if (x, y) in occupied else 0,  # Is occupied
            neighbors_occupied / 8,  # Neighbor density
            x / grid_w,  # Edge distances
            (grid_w - 1 - x) / grid_w,
            y / grid_h,
            (grid_h - 1 - y) / grid_h,
        ], dtype=np.float32)

    def _get_heuristic_direction(
        self,
        x: int,
        y: int,
        occupied: Set[Tuple[int, int]],
        outputs: List[Tuple[int, int]],
    ) -> int:
        """Get heuristic direction toward outputs."""
        if (x, y) in occupied:
            return 4  # none

        if not outputs:
            return 4

        avg_out_x = np.mean([ox for ox, oy in outputs])
        avg_out_y = np.mean([oy for ox, oy in outputs])

        dx = avg_out_x - x
        dy = avg_out_y - y

        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3  # east or west
        elif abs(dy) > 0:
            return 2 if dy > 0 else 0  # south or north
        return 4  # at destination

    def prepare_training_data(self, db_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from solved problems."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.problem_json
            FROM synthetic_problems p
            JOIN solver_results r ON p.problem_id = r.problem_id
            WHERE r.success = 1
            AND p.grid_width <= 20
            AND p.grid_height <= 20
        """)

        X_list = []
        y_list = []

        for (problem_json,) in cursor.fetchall():
            problem = json.loads(problem_json)
            grid_w = problem['grid_width']
            grid_h = problem['grid_height']

            # Get occupied cells (floor 0 only for simplicity)
            occupied = set()
            for m in problem.get('machines', []):
                if m.get('floor', 0) == 0:
                    occupied.add((m['x'], m['y']))

            # Get I/O (floor 0)
            inputs = []
            for p in problem.get('input_positions', []):
                if isinstance(p, list):
                    if len(p) < 3 or p[2] == 0:
                        inputs.append((p[0], p[1]))
                else:
                    if p.get('floor', 0) == 0:
                        inputs.append((p['x'], p['y']))

            outputs = []
            for p in problem.get('output_positions', []):
                if isinstance(p, list):
                    if len(p) < 3 or p[2] == 0:
                        outputs.append((p[0], p[1]))
                else:
                    if p.get('floor', 0) == 0:
                        outputs.append((p['x'], p['y']))

            if not outputs:
                continue

            # Sample cells (not all to avoid huge dataset)
            for y in range(0, grid_h, 2):
                for x in range(0, grid_w, 2):
                    features = self._extract_cell_features(
                        x, y, grid_w, grid_h, occupied, inputs, outputs
                    )
                    direction = self._get_heuristic_direction(x, y, occupied, outputs)

                    X_list.append(features)
                    y_list.append(direction)

        conn.close()
        return np.array(X_list), np.array(y_list)

    def train(self, db_path: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the direction predictor."""
        print("Preparing training data...")
        X, y = self.prepare_training_data(db_path)

        print(f"Training data: {len(X)} cells")
        print(f"Direction distribution: {np.bincount(y, minlength=5)}")

        if len(X) == 0:
            print("No training data available!")
            return {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Training direction model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        print(f"Direction prediction accuracy: {accuracy:.3f}")

        self.is_trained = True
        return {'accuracy': accuracy}

    def predict_direction(
        self,
        x: int,
        y: int,
        grid_w: int,
        grid_h: int,
        occupied: Set[Tuple[int, int]],
        inputs: List[Tuple[int, int]],
        outputs: List[Tuple[int, int]],
    ) -> Tuple[int, np.ndarray]:
        """
        Predict direction for a single cell.

        Returns:
            (direction_idx, probabilities)
        """
        if not self.is_trained:
            # Fallback to heuristic
            direction = self._get_heuristic_direction(x, y, occupied, outputs)
            probs = np.zeros(5)
            probs[direction] = 1.0
            return direction, probs

        features = self._extract_cell_features(
            x, y, grid_w, grid_h, occupied, inputs, outputs
        )
        X = self.scaler.transform(features.reshape(1, -1))
        probs = self.model.predict_proba(X)[0]
        direction = np.argmax(probs)
        return direction, probs

    def predict_grid(
        self,
        grid_w: int,
        grid_h: int,
        occupied: Set[Tuple[int, int]],
        inputs: List[Tuple[int, int]],
        outputs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Predict directions for entire grid.

        Returns:
            np.ndarray of shape (height, width) with direction indices
        """
        directions = np.zeros((grid_h, grid_w), dtype=np.int32)

        for y in range(grid_h):
            for x in range(grid_w):
                dir_idx, _ = self.predict_direction(
                    x, y, grid_w, grid_h, occupied, inputs, outputs
                )
                directions[y, x] = dir_idx

        return directions

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
            }, f)
        print(f"Saved direction predictor to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        print(f"Loaded direction predictor from {path}")


class DirectionPredictor3D:
    """
    3D-aware direction predictor with jump and floor change support.

    Predicts one of 11 movement types:
    - 0-3: Regular belt (north, east, south, west)
    - 4-7: Belt port jump (jump_north, jump_east, jump_south, jump_west)
    - 8-9: Floor change (up, down)
    - 10: No movement (none)
    """

    NUM_CLASSES = 11

    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required")

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_cell_features_3d(
        self,
        x: int,
        y: int,
        floor: int,
        grid_w: int,
        grid_h: int,
        num_floors: int,
        occupied: Set[Tuple[int, int, int]],
        src: Tuple[int, int, int],  # Specific source we're routing FROM
        dst: Tuple[int, int, int],  # Specific destination we're routing TO
    ) -> np.ndarray:
        """
        Extract 3D features for a single cell for a SPECIFIC connection.

        This is connection-aware: it knows exactly which src→dst pair we're routing.
        """
        max_dist = grid_w + grid_h + num_floors
        src_x, src_y, src_z = src
        dst_x, dst_y, dst_z = dst

        # Basic position features
        pos_features = [
            x / max(1, grid_w - 1),
            y / max(1, grid_h - 1),
            floor / max(1, num_floors - 1),
        ]

        # Distance from current position to source (how far we've come)
        dist_from_src = abs(x - src_x) + abs(y - src_y) + abs(floor - src_z)

        # Distance from current position to destination (how far to go)
        dist_to_dst = abs(x - dst_x) + abs(y - dst_y) + abs(floor - dst_z)

        # Total path length estimate
        total_dist = abs(src_x - dst_x) + abs(src_y - dst_y) + abs(src_z - dst_z)

        # Progress along the path (0 = at source, 1 = at destination)
        progress = dist_from_src / max(1, total_dist)

        dist_features = [
            dist_from_src / max_dist,
            dist_to_dst / max_dist,
            progress,
            (dst_z - floor) / max(1, num_floors),  # Floor diff to destination
        ]

        # Direction TO the specific destination (not average of all outputs)
        dx_to_dst = dst_x - x
        dy_to_dst = dst_y - y
        dz_to_dst = dst_z - floor

        # Normalized direction vector
        dir_features = [
            dx_to_dst / max(1, grid_w),
            dy_to_dst / max(1, grid_h),
            dz_to_dst / max(1, num_floors),
        ]

        # Obstacles in each cardinal direction (for jump decisions)
        # Check 1-4 tiles ahead in each direction
        obstacle_features = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # N, E, S, W
            obstacles_ahead = 0
            first_obstacle_dist = 5  # Beyond max jump
            can_jump = True

            for dist in range(1, 5):
                nx, ny = x + dx * dist, y + dy * dist
                if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                    can_jump = False
                    break
                if (nx, ny, floor) in occupied:
                    obstacles_ahead += 1
                    if dist < first_obstacle_dist:
                        first_obstacle_dist = dist
                    # Can still jump over if obstacle is in middle
                    if dist >= 2:
                        can_jump = True

            obstacle_features.extend([
                obstacles_ahead / 4,
                first_obstacle_dist / 5,
                1.0 if can_jump and obstacles_ahead > 0 else 0.0,
            ])

        # Floor-specific features for this connection
        dst_above = dst_z > floor
        dst_below = dst_z < floor
        dst_same_floor = dst_z == floor

        # Can move up/down from this position?
        can_go_up = floor < num_floors - 1 and (x, y, floor + 1) not in occupied
        can_go_down = floor > 0 and (x, y, floor - 1) not in occupied

        floor_features = [
            1.0 if dst_above else 0.0,
            1.0 if dst_below else 0.0,
            1.0 if dst_same_floor else 0.0,
            1.0 if can_go_up else 0.0,
            1.0 if can_go_down else 0.0,
        ]

        # Local congestion (8 neighbors on same floor)
        neighbors_occupied = sum(
            1 for ddx in [-1, 0, 1] for ddy in [-1, 0, 1]
            if (ddx != 0 or ddy != 0) and (x + ddx, y + ddy, floor) in occupied
        )

        congestion_features = [
            neighbors_occupied / 8,
            1.0 if (x, y, floor) in occupied else 0.0,
        ]

        # Edge distances
        edge_features = [
            x / max(1, grid_w - 1),
            (grid_w - 1 - x) / max(1, grid_w - 1),
            y / max(1, grid_h - 1),
            (grid_h - 1 - y) / max(1, grid_h - 1),
        ]

        all_features = (
            pos_features +      # 3
            dist_features +     # 4 (added progress)
            dir_features +      # 3 (added z direction)
            obstacle_features + # 12 (3 per direction * 4 directions)
            floor_features +    # 5 (changed to connection-specific)
            congestion_features + # 2
            edge_features       # 4
        )  # Total: 33 features

        return np.array(all_features, dtype=np.float32)

    def _get_heuristic_direction_3d(
        self,
        x: int,
        y: int,
        floor: int,
        grid_w: int,
        grid_h: int,
        num_floors: int,
        occupied: Set[Tuple[int, int, int]],
        dst: Tuple[int, int, int],  # Specific destination
    ) -> int:
        """Get heuristic direction toward a SPECIFIC destination."""
        if (x, y, floor) in occupied:
            return 10  # none

        tx, ty, tz = dst

        # Already at destination
        if x == tx and y == ty and floor == tz:
            return 10  # none

        # Check if floor change needed
        if tz > floor and floor < num_floors - 1:
            if (x, y, floor + 1) not in occupied:
                return 8  # up
        elif tz < floor and floor > 0:
            if (x, y, floor - 1) not in occupied:
                return 9  # down

        # Determine primary direction on current floor
        dx = tx - x
        dy = ty - y

        # Check if jump would help (obstacle ahead but clear landing)
        if abs(dx) > abs(dy):
            direction = 1 if dx > 0 else 3  # east or west
            step_x = 1 if dx > 0 else -1

            # Check for obstacle that could be jumped
            if (x + step_x, y, floor) in occupied:
                # Check if we can jump over
                for jump_dist in [2, 3, 4]:
                    land_x = x + step_x * jump_dist
                    if 0 <= land_x < grid_w and (land_x, y, floor) not in occupied:
                        return 4 + direction  # jump in that direction
        else:
            direction = 2 if dy > 0 else 0  # south or north
            step_y = 1 if dy > 0 else -1

            # Check for obstacle that could be jumped
            if (x, y + step_y, floor) in occupied:
                for jump_dist in [2, 3, 4]:
                    land_y = y + step_y * jump_dist
                    if 0 <= land_y < grid_h and (x, land_y, floor) not in occupied:
                        return 4 + direction  # jump in that direction

        # Regular belt movement
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3  # east or west
        elif abs(dy) > 0:
            return 2 if dy > 0 else 0  # south or north

        return 10  # at destination

    def prepare_training_data(
        self,
        db_path: str,
        use_solved_paths: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare CONNECTION-AWARE training data from solved problems.

        For each connection (src→dst), we sample points along potential paths
        and create training examples with the specific src/dst context.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.problem_json, r.partial_belt_positions
            FROM synthetic_problems p
            JOIN solver_results r ON p.problem_id = r.problem_id
            WHERE r.success = 1
            AND p.grid_width <= 20
            AND p.grid_height <= 20
        """)

        X_list = []
        y_list = []

        for row in cursor.fetchall():
            problem = json.loads(row[0])
            belt_positions_json = row[1] if len(row) > 1 else "[]"

            grid_w = problem['grid_width']
            grid_h = problem['grid_height']
            num_floors = problem.get('num_floors', 1)

            # Get occupied cells (all floors)
            occupied = set()
            for m in problem.get('machines', []):
                occupied.add((m['x'], m['y'], m.get('floor', 0)))

            # Get connections (src→dst pairs)
            connections = problem.get('connections', [])
            if not connections:
                continue

            # Parse belt positions for actual path data
            belt_positions = []
            if use_solved_paths and belt_positions_json:
                try:
                    belts = json.loads(belt_positions_json)
                    for b in belts:
                        if isinstance(b, list):
                            belt_positions.append((b[0], b[1], b[2] if len(b) > 2 else 0))
                except (json.JSONDecodeError, TypeError):
                    pass

            # Build direction map from actual belt paths
            belt_path_directions = {}
            if belt_positions and len(belt_positions) > 1:
                for i in range(len(belt_positions) - 1):
                    curr = belt_positions[i]
                    next_pos = belt_positions[i + 1]
                    cx, cy, cz = curr
                    nx, ny, nz = next_pos
                    dx, dy, dz = nx - cx, ny - cy, nz - cz

                    if dz > 0:
                        dir_idx = 8  # up
                    elif dz < 0:
                        dir_idx = 9  # down
                    elif abs(dx) > 1 or abs(dy) > 1:
                        # Jump
                        if dy < 0:
                            dir_idx = 4  # jump_north
                        elif dx > 0:
                            dir_idx = 5  # jump_east
                        elif dy > 0:
                            dir_idx = 6  # jump_south
                        else:
                            dir_idx = 7  # jump_west
                    else:
                        # Regular move
                        if dy < 0:
                            dir_idx = 0  # north
                        elif dx > 0:
                            dir_idx = 1  # east
                        elif dy > 0:
                            dir_idx = 2  # south
                        elif dx < 0:
                            dir_idx = 3  # west
                        else:
                            dir_idx = 10  # none

                    belt_path_directions[curr] = dir_idx

            # For each connection, generate training samples
            for conn_data in connections:
                src = conn_data.get('src', conn_data.get('from'))
                dst = conn_data.get('dst', conn_data.get('to'))

                if not src or not dst:
                    continue

                # Normalize to tuples
                if isinstance(src, list):
                    src = (src[0], src[1], src[2] if len(src) > 2 else 0)
                elif isinstance(src, dict):
                    src = (src['x'], src['y'], src.get('floor', 0))

                if isinstance(dst, list):
                    dst = (dst[0], dst[1], dst[2] if len(dst) > 2 else 0)
                elif isinstance(dst, dict):
                    dst = (dst['x'], dst['y'], dst.get('floor', 0))

                # Sample points in the bounding box of this connection
                min_x = max(0, min(src[0], dst[0]) - 2)
                max_x = min(grid_w, max(src[0], dst[0]) + 3)
                min_y = max(0, min(src[1], dst[1]) - 2)
                max_y = min(grid_h, max(src[1], dst[1]) + 3)
                min_z = max(0, min(src[2], dst[2]))
                max_z = min(num_floors, max(src[2], dst[2]) + 1)

                for floor in range(min_z, max_z):
                    for y in range(min_y, max_y):
                        for x in range(min_x, max_x):
                            # Extract features for this cell with THIS specific connection
                            features = self._extract_cell_features_3d(
                                x, y, floor, grid_w, grid_h, num_floors,
                                occupied, src, dst
                            )

                            # Use actual path direction if available
                            if (x, y, floor) in belt_path_directions:
                                direction = belt_path_directions[(x, y, floor)]
                            else:
                                # Use heuristic for this specific destination
                                direction = self._get_heuristic_direction_3d(
                                    x, y, floor, grid_w, grid_h, num_floors,
                                    occupied, dst
                                )

                            X_list.append(features)
                            y_list.append(direction)

        conn.close()
        return np.array(X_list), np.array(y_list)

    def train(self, db_path: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the 3D direction predictor."""
        print("Preparing 3D training data...")
        X, y = self.prepare_training_data(db_path)

        print(f"Training data: {len(X)} cells")
        print(f"Direction distribution: {np.bincount(y, minlength=self.NUM_CLASSES)}")
        print(f"  Belt: {sum(y < 4)}, Jump: {sum((y >= 4) & (y < 8))}, "
              f"Floor: {sum((y >= 8) & (y < 10))}, None: {sum(y == 10)}")

        if len(X) == 0:
            print("No training data available!")
            return {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Training 3D direction model (11 classes)...")
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # Handle class imbalance
        )
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        print(f"Overall accuracy: {accuracy:.3f}")

        # Per-class accuracy
        for i, name in enumerate(DIRECTIONS):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == y_test[mask])
                print(f"  {name}: {class_acc:.3f} ({mask.sum()} samples)")

        self.is_trained = True
        return {'accuracy': accuracy}

    def predict_direction(
        self,
        x: int,
        y: int,
        floor: int,
        grid_w: int,
        grid_h: int,
        num_floors: int,
        occupied: Set[Tuple[int, int, int]],
        src: Tuple[int, int, int],  # Specific source of this connection
        dst: Tuple[int, int, int],  # Specific destination of this connection
    ) -> Tuple[int, np.ndarray]:
        """Predict direction for a single cell for a SPECIFIC connection."""
        if not self.is_trained:
            direction = self._get_heuristic_direction_3d(
                x, y, floor, grid_w, grid_h, num_floors,
                occupied, dst
            )
            probs = np.zeros(self.NUM_CLASSES)
            probs[direction] = 1.0
            return direction, probs

        features = self._extract_cell_features_3d(
            x, y, floor, grid_w, grid_h, num_floors,
            occupied, src, dst
        )
        X = self.scaler.transform(features.reshape(1, -1))
        probs = self.model.predict_proba(X)[0]

        # Ensure probs has all 11 classes
        if len(probs) < self.NUM_CLASSES:
            full_probs = np.zeros(self.NUM_CLASSES)
            for i, cls in enumerate(self.model.classes_):
                full_probs[cls] = probs[i]
            probs = full_probs

        direction = np.argmax(probs)
        return direction, probs

    def predict_path_directions(
        self,
        grid_w: int,
        grid_h: int,
        num_floors: int,
        occupied: Set[Tuple[int, int, int]],
        src: Tuple[int, int, int],
        dst: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Predict directions for the path between src and dst.

        Returns:
            np.ndarray of shape (num_floors, height, width) with direction indices
            Only meaningful in the bounding box of src→dst
        """
        directions = np.full((num_floors, grid_h, grid_w), 10, dtype=np.int32)  # 10 = none

        # Focus on bounding box of the connection
        min_x = max(0, min(src[0], dst[0]) - 2)
        max_x = min(grid_w, max(src[0], dst[0]) + 3)
        min_y = max(0, min(src[1], dst[1]) - 2)
        max_y = min(grid_h, max(src[1], dst[1]) + 3)
        min_z = max(0, min(src[2], dst[2]))
        max_z = min(num_floors, max(src[2], dst[2]) + 1)

        for floor in range(min_z, max_z):
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    dir_idx, _ = self.predict_direction(
                        x, y, floor, grid_w, grid_h, num_floors,
                        occupied, src, dst
                    )
                    directions[floor, y, x] = dir_idx

        return directions

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
            }, f)
        print(f"Saved 3D direction predictor to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        print(f"Loaded 3D direction predictor from {path}")


class GridEncoder:
    """Encodes problem state into grid representation for neural network."""

    # Channel definitions for input tensor
    CHANNELS = {
        'occupied': 0,      # Cell is occupied by machine
        'input': 1,         # Cell is an input
        'output': 2,        # Cell is an output
        'src_dist': 3,      # Distance to nearest source (normalized)
        'dst_dist': 4,      # Distance to nearest destination
        'edge_west': 5,     # Distance to west edge
        'edge_east': 6,     # Distance to east edge
        'edge_north': 7,    # Distance to north edge
        'edge_south': 8,    # Distance to south edge
    }
    NUM_CHANNELS = 9

    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height

    def encode(
        self,
        occupied: Set[Tuple[int, int, int]],
        inputs: List[Tuple[int, int, int]],
        outputs: List[Tuple[int, int, int]],
        floor: int = 0,
    ) -> np.ndarray:
        """
        Encode a single floor as multi-channel image.

        Returns:
            np.ndarray of shape (NUM_CHANNELS, grid_height, grid_width)
        """
        grid = np.zeros((self.NUM_CHANNELS, self.grid_height, self.grid_width), dtype=np.float32)

        # Occupied channel
        for (x, y, z) in occupied:
            if z == floor and 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                grid[self.CHANNELS['occupied'], y, x] = 1.0

        # Input/output channels
        for (x, y, z) in inputs:
            if z == floor and 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                grid[self.CHANNELS['input'], y, x] = 1.0

        for (x, y, z) in outputs:
            if z == floor and 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                grid[self.CHANNELS['output'], y, x] = 1.0

        # Distance to sources/destinations
        max_dist = self.grid_width + self.grid_height
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Distance to nearest input
                min_src_dist = max_dist
                for (ix, iy, iz) in inputs:
                    if iz == floor:
                        d = abs(x - ix) + abs(y - iy)
                        min_src_dist = min(min_src_dist, d)
                grid[self.CHANNELS['src_dist'], y, x] = min_src_dist / max_dist

                # Distance to nearest output
                min_dst_dist = max_dist
                for (ox, oy, oz) in outputs:
                    if oz == floor:
                        d = abs(x - ox) + abs(y - oy)
                        min_dst_dist = min(min_dst_dist, d)
                grid[self.CHANNELS['dst_dist'], y, x] = min_dst_dist / max_dist

                # Edge distances
                grid[self.CHANNELS['edge_west'], y, x] = x / self.grid_width
                grid[self.CHANNELS['edge_east'], y, x] = (self.grid_width - 1 - x) / self.grid_width
                grid[self.CHANNELS['edge_north'], y, x] = y / self.grid_height
                grid[self.CHANNELS['edge_south'], y, x] = (self.grid_height - 1 - y) / self.grid_height

        return grid


if HAS_TORCH:
    class PolicyNetwork(nn.Module):
        """
        CNN-based policy network that predicts belt direction at each position.

        Architecture:
        - Convolutional layers to extract spatial features
        - Outputs 5 channels: 4 directions (N,E,S,W) + no-belt
        """

        # Direction encoding
        DIRECTIONS = ['north', 'east', 'south', 'west', 'none']
        DIR_TO_DELTA = {
            'north': (0, -1),
            'east': (1, 0),
            'south': (0, 1),
            'west': (-1, 0),
            'none': (0, 0),
        }

        def __init__(self, in_channels: int = GridEncoder.NUM_CHANNELS):
            super().__init__()

            # Convolutional backbone
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

            # Output layer: 5 directions per cell
            self.output = nn.Conv2d(32, 5, kernel_size=1)

            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input tensor of shape (batch, channels, height, width)

            Returns:
                Tensor of shape (batch, 5, height, width) with direction logits
            """
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.output(x)
            return x

        def predict_directions(
            self,
            grid: np.ndarray,
            mask: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """
            Predict belt directions for a grid.

            Args:
                grid: Input grid from GridEncoder (channels, height, width)
                mask: Optional mask of valid cells

            Returns:
                np.ndarray of direction indices (height, width)
            """
            self.eval()
            with torch.no_grad():
                x = torch.from_numpy(grid).unsqueeze(0)
                logits = self(x)
                probs = F.softmax(logits, dim=1)
                directions = torch.argmax(probs, dim=1).squeeze(0).numpy()

            if mask is not None:
                directions = np.where(mask, directions, 4)  # 4 = none

            return directions


    class PolicyNetworkTrainer:
        """Trains the policy network using solved routing problems."""

        def __init__(self, device: str = 'cpu'):
            self.device = torch.device(device)
            self.model = PolicyNetwork().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.criterion = nn.CrossEntropyLoss()

        def prepare_training_data(
            self,
            db_path: str,
            max_grid_size: int = 20,
        ) -> List[Tuple[np.ndarray, np.ndarray]]:
            """
            Prepare training data from solved problems.

            Note: This requires belt path data which we'll need to extract
            from successful solves. For now, we'll use a simplified approach
            based on problem structure.
            """
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get successful solves
            cursor.execute("""
                SELECT p.problem_json
                FROM synthetic_problems p
                JOIN solver_results r ON p.problem_id = r.problem_id
                WHERE r.success = 1
                AND p.grid_width <= ?
                AND p.grid_height <= ?
            """, (max_grid_size, max_grid_size))

            training_data = []

            for (problem_json,) in cursor.fetchall():
                problem = json.loads(problem_json)

                grid_w = problem['grid_width']
                grid_h = problem['grid_height']

                # Get occupied cells
                occupied = set()
                for m in problem.get('machines', []):
                    occupied.add((m['x'], m['y'], m.get('floor', 0)))

                # Get I/O positions
                inputs = [tuple(p) if isinstance(p, list) else (p['x'], p['y'], p.get('floor', 0))
                         for p in problem.get('input_positions', [])]
                outputs = [tuple(p) if isinstance(p, list) else (p['x'], p['y'], p.get('floor', 0))
                          for p in problem.get('output_positions', [])]

                # Create encoder
                encoder = GridEncoder(grid_w, grid_h)

                # For each floor
                for floor in range(problem['num_floors']):
                    grid = encoder.encode(occupied, inputs, outputs, floor)

                    # Create target direction map based on heuristic
                    # (In production, this would come from actual solved paths)
                    target = self._create_heuristic_target(
                        grid_w, grid_h, inputs, outputs, occupied, floor
                    )

                    training_data.append((grid, target))

            conn.close()
            return training_data

        def _create_heuristic_target(
            self,
            grid_w: int,
            grid_h: int,
            inputs: List[Tuple[int, int, int]],
            outputs: List[Tuple[int, int, int]],
            occupied: Set[Tuple[int, int, int]],
            floor: int,
        ) -> np.ndarray:
            """
            Create heuristic direction targets based on flow direction.

            Direction encoding:
            0 = north (-y)
            1 = east (+x)
            2 = south (+y)
            3 = west (-x)
            4 = none
            """
            target = np.full((grid_h, grid_w), 4, dtype=np.int64)  # Default: none

            # Get I/O on this floor
            floor_inputs = [(x, y) for x, y, z in inputs if z == floor]
            floor_outputs = [(x, y) for x, y, z in outputs if z == floor]

            if not floor_inputs or not floor_outputs:
                return target

            # Calculate average output position
            avg_out_x = np.mean([x for x, y in floor_outputs])
            avg_out_y = np.mean([y for x, y in floor_outputs])

            # For each cell, point toward outputs
            for y in range(grid_h):
                for x in range(grid_w):
                    if (x, y, floor) in occupied:
                        target[y, x] = 4  # No belt on occupied
                        continue

                    # Direction toward output
                    dx = avg_out_x - x
                    dy = avg_out_y - y

                    if abs(dx) > abs(dy):
                        target[y, x] = 1 if dx > 0 else 3  # East or West
                    elif abs(dy) > 0:
                        target[y, x] = 2 if dy > 0 else 0  # South or North
                    else:
                        target[y, x] = 4  # At destination

            return target

        def train(
            self,
            training_data: List[Tuple[np.ndarray, np.ndarray]],
            epochs: int = 50,
            batch_size: int = 32,
        ) -> List[float]:
            """
            Train the policy network.

            Returns:
                List of loss values per epoch
            """
            losses = []

            for epoch in range(epochs):
                epoch_loss = 0.0
                np.random.shuffle(training_data)

                for i in range(0, len(training_data), batch_size):
                    batch = training_data[i:i+batch_size]

                    # Stack batch
                    grids = torch.tensor(
                        np.stack([g for g, t in batch]),
                        dtype=torch.float32,
                        device=self.device
                    )
                    targets = torch.tensor(
                        np.stack([t for g, t in batch]),
                        dtype=torch.long,
                        device=self.device
                    )

                    # Forward
                    self.optimizer.zero_grad()
                    logits = self.model(grids)

                    # Reshape for cross entropy: (N, C, H, W) -> (N*H*W, C)
                    N, C, H, W = logits.shape
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
                    targets_flat = targets.reshape(-1)

                    loss = self.criterion(logits_flat, targets_flat)

                    # Backward
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / max(1, len(training_data) // batch_size)
                losses.append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            return losses

        def save(self, path: str):
            """Save model weights."""
            torch.save(self.model.state_dict(), path)
            print(f"Saved policy network to {path}")

        def load(self, path: str):
            """Load model weights."""
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded policy network from {path}")


# =============================================================================
# Integration with Solver
# =============================================================================

class MLGuidedRouter:
    """
    Router that uses ML models to guide the solving process.

    Uses:
    1. SolvabilityClassifier to decide solver mode and timeout
    2. DirectionPredictor (sklearn) or PolicyNetwork (PyTorch) for A* heuristic
    """

    def __init__(
        self,
        solvability_model_path: Optional[str] = None,
        direction_model_path: Optional[str] = None,
        policy_model_path: Optional[str] = None,
    ):
        self.solvability_classifier = None
        self.direction_predictor = None
        self.policy_network = None

        if solvability_model_path and Path(solvability_model_path).exists():
            self.solvability_classifier = SolvabilityClassifier()
            self.solvability_classifier.load(solvability_model_path)

        # Try sklearn direction predictor first
        if direction_model_path and Path(direction_model_path).exists():
            self.direction_predictor = DirectionPredictor()
            self.direction_predictor.load(direction_model_path)

        # PyTorch policy network if available
        if HAS_TORCH and policy_model_path and Path(policy_model_path).exists():
            self.policy_network = PolicyNetwork()
            self.policy_network.load_state_dict(torch.load(policy_model_path))
            self.policy_network.eval()

    def get_recommendations(self, problem_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ML recommendations for solving a problem.

        Returns:
            Dict with recommendations for solver mode, timeout, etc.
        """
        result = {
            'has_solvability_model': self.solvability_classifier is not None,
            'has_direction_model': self.direction_predictor is not None,
            'has_policy_model': self.policy_network is not None,
        }

        if self.solvability_classifier:
            prediction = self.solvability_classifier.predict(problem_dict)
            result.update(prediction)
        else:
            # Default recommendations without ML
            result.update({
                'solvable': True,
                'solvable_prob': 0.5,
                'difficulty': 'medium',
                'recommended_mode': 'hybrid',
                'recommended_timeout': 30,
            })

        return result

    def get_direction_heuristic(
        self,
        grid_width: int,
        grid_height: int,
        occupied: Set[Tuple[int, int, int]],
        inputs: List[Tuple[int, int, int]],
        outputs: List[Tuple[int, int, int]],
        floor: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Get ML-predicted direction preferences for A* heuristic.

        Returns:
            Array of shape (height, width) with direction indices, or None
        """
        # Convert 3D positions to 2D for the floor
        occupied_2d = {(x, y) for (x, y, z) in occupied if z == floor}
        inputs_2d = [(x, y) for (x, y, z) in inputs if z == floor]
        outputs_2d = [(x, y) for (x, y, z) in outputs if z == floor]

        # Try sklearn predictor first (always available when trained)
        if self.direction_predictor and self.direction_predictor.is_trained:
            return self.direction_predictor.predict_grid(
                grid_width, grid_height, occupied_2d, inputs_2d, outputs_2d
            )

        # Fall back to PyTorch policy network
        if self.policy_network:
            encoder = GridEncoder(grid_width, grid_height)
            grid = encoder.encode(occupied, inputs, outputs, floor)
            return self.policy_network.predict_directions(grid)

        return None

    def get_direction_at(
        self,
        x: int,
        y: int,
        grid_width: int,
        grid_height: int,
        occupied: Set[Tuple[int, int]],
        inputs: List[Tuple[int, int]],
        outputs: List[Tuple[int, int]],
    ) -> Tuple[int, float]:
        """
        Get predicted direction for a single cell with confidence.

        Returns:
            (direction_idx, confidence)
        """
        if self.direction_predictor and self.direction_predictor.is_trained:
            dir_idx, probs = self.direction_predictor.predict_direction(
                x, y, grid_width, grid_height, occupied, inputs, outputs
            )
            return dir_idx, float(np.max(probs))

        # Fallback heuristic
        if outputs:
            avg_x = np.mean([ox for ox, oy in outputs])
            avg_y = np.mean([oy for ox, oy in outputs])
            dx, dy = avg_x - x, avg_y - y
            if abs(dx) > abs(dy):
                return (1 if dx > 0 else 3), 0.5  # east or west
            elif abs(dy) > 0:
                return (2 if dy > 0 else 0), 0.5  # south or north
        return 4, 0.5  # none


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train ML models for routing")
    parser.add_argument("--db", type=str, default="synthetic_training.db",
                       help="Path to training database")
    parser.add_argument("--train-classifier", action="store_true",
                       help="Train solvability/difficulty classifier (Phase 1)")
    parser.add_argument("--train-direction", action="store_true",
                       help="Train sklearn direction predictor (2D, Phase 2 fallback)")
    parser.add_argument("--train-direction-3d", action="store_true",
                       help="Train 3D direction predictor with jumps and floor changes")
    parser.add_argument("--train-policy", action="store_true",
                       help="Train PyTorch policy network (Phase 2)")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory for saved models")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs for policy network")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.train_classifier:
        print("="*60)
        print("Phase 1: Training Solvability/Difficulty Classifier")
        print("="*60)

        classifier = SolvabilityClassifier()
        metrics = classifier.train(args.db)

        model_path = output_dir / "solvability_classifier.pkl"
        classifier.save(str(model_path))

        print(f"\nTraining complete!")
        print(f"  Solvability accuracy: {metrics['solvability_accuracy']:.3f}")
        print(f"  Difficulty accuracy: {metrics['difficulty_accuracy']:.3f}")

    if args.train_direction:
        print("\n" + "="*60)
        print("Phase 2: Training Direction Predictor (sklearn, 2D)")
        print("="*60)

        predictor = DirectionPredictor()
        metrics = predictor.train(args.db)

        if metrics:
            model_path = output_dir / "direction_predictor.pkl"
            predictor.save(str(model_path))

            print(f"\nTraining complete!")
            print(f"  Direction accuracy: {metrics.get('accuracy', 0):.3f}")
        else:
            print("Training failed - no data available")

    if args.train_direction_3d:
        print("\n" + "="*60)
        print("Phase 2: Training 3D Direction Predictor")
        print("  (11 classes: belt, jump, up/down, none)")
        print("="*60)

        predictor = DirectionPredictor3D()
        metrics = predictor.train(args.db)

        if metrics:
            model_path = output_dir / "direction_predictor_3d.pkl"
            predictor.save(str(model_path))

            print(f"\nTraining complete!")
            print(f"  Overall accuracy: {metrics.get('accuracy', 0):.3f}")
        else:
            print("Training failed - no data available")

    if args.train_policy:
        if not HAS_TORCH:
            print("ERROR: PyTorch required for policy network training")
            print("Use --train-direction for sklearn fallback instead")
            return

        print("\n" + "="*60)
        print("Phase 2: Training Policy Network (PyTorch)")
        print("="*60)

        trainer = PolicyNetworkTrainer()

        print("Preparing training data...")
        training_data = trainer.prepare_training_data(args.db)
        print(f"Prepared {len(training_data)} training examples")

        if training_data:
            print("\nTraining policy network...")
            losses = trainer.train(training_data, epochs=args.epochs)

            model_path = output_dir / "policy_network.pt"
            trainer.save(str(model_path))

            print(f"\nTraining complete!")
            print(f"  Final loss: {losses[-1]:.4f}")
        else:
            print("No training data available")

    if not any([args.train_classifier, args.train_direction,
                 args.train_direction_3d, args.train_policy]):
        parser.print_help()


if __name__ == "__main__":
    main()
