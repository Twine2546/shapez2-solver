"""
Machine Learning Evaluators for CP-SAT Solver and A* Routing.

This module provides ML-based implementations of the evaluation interfaces
defined in evaluation.py. These can be trained on successful solutions to
improve solver performance.

Features:
- Unified feature extraction for placements, solutions, and routing
- Support for multiple model backends (sklearn, PyTorch)
- Training data collection hooks
- Model persistence and loading
"""

import json
import sqlite3
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime

from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS
from .evaluation import (
    SolutionEvaluator,
    PlacementEvaluator,
    RoutingHeuristic,
    MoveCostFunction,
    SolutionInfo,
    PlacementInfo,
    RoutingInfo,
    DefaultSolutionEvaluator,
    DefaultRoutingHeuristic,
    DefaultMoveCostFunction,
)

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    GradientBoostingRegressor = None
    GradientBoostingClassifier = None
    StandardScaler = None
    joblib = None


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class PlacementFeatures:
    """Features extracted from a machine placement."""
    # Grid-relative position (normalized 0-1)
    norm_x: float
    norm_y: float
    norm_floor: float

    # Distance to edges
    dist_to_west: float
    dist_to_east: float
    dist_to_north: float
    dist_to_south: float

    # Machine type encoding (one-hot style)
    is_cutter: int
    is_rotator: int
    is_stacker: int
    is_other: int

    # Rotation encoding
    rot_east: int
    rot_south: int
    rot_west: int
    rot_north: int

    # Machine dimensions
    width: int
    height: int

    def to_array(self) -> np.ndarray:
        return np.array([
            self.norm_x, self.norm_y, self.norm_floor,
            self.dist_to_west, self.dist_to_east,
            self.dist_to_north, self.dist_to_south,
            self.is_cutter, self.is_rotator, self.is_stacker, self.is_other,
            self.rot_east, self.rot_south, self.rot_west, self.rot_north,
            self.width, self.height,
        ], dtype=np.float32)


@dataclass
class SolutionFeatures:
    """Features extracted from a complete solution."""
    # Grid info
    grid_width: int
    grid_height: int
    num_floors: int

    # I/O counts
    num_inputs: int
    num_outputs: int

    # Machine summary
    num_machines: int
    num_cutters: int
    num_rotators: int
    num_stackers: int

    # Placement quality metrics
    avg_machine_x: float
    avg_machine_y: float
    machine_spread_x: float  # std dev
    machine_spread_y: float

    # Routing metrics
    num_belts: int
    routing_success: int

    # Throughput
    throughput_per_output: float

    # Derived metrics
    belt_to_machine_ratio: float
    density: float  # machines / grid_area

    def to_array(self) -> np.ndarray:
        return np.array([
            self.grid_width, self.grid_height, self.num_floors,
            self.num_inputs, self.num_outputs,
            self.num_machines, self.num_cutters, self.num_rotators, self.num_stackers,
            self.avg_machine_x, self.avg_machine_y,
            self.machine_spread_x, self.machine_spread_y,
            self.num_belts, self.routing_success,
            self.throughput_per_output,
            self.belt_to_machine_ratio, self.density,
        ], dtype=np.float32)


@dataclass
class RoutingFeatures:
    """Features for A* routing decisions."""
    # Current position (normalized)
    curr_norm_x: float
    curr_norm_y: float
    curr_floor: int

    # Goal position (normalized)
    goal_norm_x: float
    goal_norm_y: float
    goal_floor: int

    # Distance metrics
    manhattan_dist: float
    euclidean_dist: float
    floor_diff: int

    # Direction to goal
    dx_to_goal: float  # -1 to 1
    dy_to_goal: float

    # Local occupancy (8-directional)
    occupied_n: int
    occupied_s: int
    occupied_e: int
    occupied_w: int
    occupied_ne: int
    occupied_nw: int
    occupied_se: int
    occupied_sw: int

    def to_array(self) -> np.ndarray:
        return np.array([
            self.curr_norm_x, self.curr_norm_y, self.curr_floor,
            self.goal_norm_x, self.goal_norm_y, self.goal_floor,
            self.manhattan_dist, self.euclidean_dist, self.floor_diff,
            self.dx_to_goal, self.dy_to_goal,
            self.occupied_n, self.occupied_s, self.occupied_e, self.occupied_w,
            self.occupied_ne, self.occupied_nw, self.occupied_se, self.occupied_sw,
        ], dtype=np.float32)


class FeatureExtractor:
    """Extracts features from placements, solutions, and routing states."""

    def __init__(self, grid_width: int, grid_height: int, num_floors: int = 3):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_floors = num_floors

    def extract_placement_features(self, placement: PlacementInfo) -> PlacementFeatures:
        """Extract features from a single machine placement."""
        spec = BUILDING_SPECS.get(placement.building_type)
        width = spec.width if spec else 1
        height = spec.height if spec else 1

        # Normalized position
        norm_x = placement.x / max(1, self.grid_width - 1)
        norm_y = placement.y / max(1, self.grid_height - 1)
        norm_floor = placement.floor / max(1, self.num_floors - 1)

        # Distance to edges
        dist_to_west = placement.x / self.grid_width
        dist_to_east = (self.grid_width - placement.x - width) / self.grid_width
        dist_to_north = placement.y / self.grid_height
        dist_to_south = (self.grid_height - placement.y - height) / self.grid_height

        # Machine type
        bt = placement.building_type
        is_cutter = 1 if bt in (BuildingType.CUTTER, BuildingType.HALF_CUTTER) else 0
        is_rotator = 1 if bt in (BuildingType.ROTATOR_CW, BuildingType.ROTATOR_CCW, BuildingType.ROTATOR_180) else 0
        is_stacker = 1 if bt in (BuildingType.STACKER, BuildingType.STACKER_BENT, BuildingType.UNSTACKER) else 0
        is_other = 1 if not (is_cutter or is_rotator or is_stacker) else 0

        # Rotation
        rot = placement.rotation
        rot_east = 1 if rot == Rotation.EAST else 0
        rot_south = 1 if rot == Rotation.SOUTH else 0
        rot_west = 1 if rot == Rotation.WEST else 0
        rot_north = 1 if rot == Rotation.NORTH else 0

        return PlacementFeatures(
            norm_x=norm_x, norm_y=norm_y, norm_floor=norm_floor,
            dist_to_west=dist_to_west, dist_to_east=dist_to_east,
            dist_to_north=dist_to_north, dist_to_south=dist_to_south,
            is_cutter=is_cutter, is_rotator=is_rotator,
            is_stacker=is_stacker, is_other=is_other,
            rot_east=rot_east, rot_south=rot_south,
            rot_west=rot_west, rot_north=rot_north,
            width=width, height=height,
        )

    def extract_solution_features(self, solution: SolutionInfo) -> SolutionFeatures:
        """Extract features from a complete solution."""
        machines = solution.machines

        # Count machine types
        num_cutters = sum(1 for m in machines if m.building_type in
                         (BuildingType.CUTTER, BuildingType.HALF_CUTTER))
        num_rotators = sum(1 for m in machines if m.building_type in
                          (BuildingType.ROTATOR_CW, BuildingType.ROTATOR_CCW, BuildingType.ROTATOR_180))
        num_stackers = sum(1 for m in machines if m.building_type in
                          (BuildingType.STACKER, BuildingType.STACKER_BENT, BuildingType.UNSTACKER))

        # Position statistics
        if machines:
            xs = [m.x for m in machines]
            ys = [m.y for m in machines]
            avg_x = np.mean(xs)
            avg_y = np.mean(ys)
            spread_x = np.std(xs) if len(xs) > 1 else 0
            spread_y = np.std(ys) if len(ys) > 1 else 0
        else:
            avg_x = avg_y = spread_x = spread_y = 0

        # Derived metrics
        num_belts = solution.routing.total_length
        belt_ratio = num_belts / max(1, len(machines))
        grid_area = solution.grid_width * solution.grid_height
        density = len(machines) / grid_area

        return SolutionFeatures(
            grid_width=solution.grid_width,
            grid_height=solution.grid_height,
            num_floors=solution.num_floors,
            num_inputs=solution.num_inputs,
            num_outputs=solution.num_outputs,
            num_machines=len(machines),
            num_cutters=num_cutters,
            num_rotators=num_rotators,
            num_stackers=num_stackers,
            avg_machine_x=avg_x,
            avg_machine_y=avg_y,
            machine_spread_x=spread_x,
            machine_spread_y=spread_y,
            num_belts=num_belts,
            routing_success=1 if solution.routing.success else 0,
            throughput_per_output=solution.throughput_per_output,
            belt_to_machine_ratio=belt_ratio,
            density=density,
        )

    def extract_routing_features(
        self,
        current: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        occupied: Optional[Set[Tuple[int, int, int]]] = None,
    ) -> RoutingFeatures:
        """Extract features for A* routing decision."""
        occupied = occupied or set()

        cx, cy, cf = current
        gx, gy, gf = goal

        # Normalized positions
        curr_norm_x = cx / max(1, self.grid_width - 1)
        curr_norm_y = cy / max(1, self.grid_height - 1)
        goal_norm_x = gx / max(1, self.grid_width - 1)
        goal_norm_y = gy / max(1, self.grid_height - 1)

        # Distances
        dx = gx - cx
        dy = gy - cy
        manhattan = abs(dx) + abs(dy) + abs(gf - cf) * 2
        euclidean = np.sqrt(dx*dx + dy*dy + (gf - cf)**2 * 4)

        # Direction to goal (normalized)
        max_dist = max(self.grid_width, self.grid_height)
        dx_norm = dx / max_dist if max_dist > 0 else 0
        dy_norm = dy / max_dist if max_dist > 0 else 0

        # Local occupancy
        def is_occ(dx, dy):
            return 1 if (cx + dx, cy + dy, cf) in occupied else 0

        return RoutingFeatures(
            curr_norm_x=curr_norm_x, curr_norm_y=curr_norm_y, curr_floor=cf,
            goal_norm_x=goal_norm_x, goal_norm_y=goal_norm_y, goal_floor=gf,
            manhattan_dist=manhattan / max_dist,
            euclidean_dist=euclidean / max_dist,
            floor_diff=abs(gf - cf),
            dx_to_goal=dx_norm, dy_to_goal=dy_norm,
            occupied_n=is_occ(0, -1), occupied_s=is_occ(0, 1),
            occupied_e=is_occ(1, 0), occupied_w=is_occ(-1, 0),
            occupied_ne=is_occ(1, -1), occupied_nw=is_occ(-1, -1),
            occupied_se=is_occ(1, 1), occupied_sw=is_occ(-1, 1),
        )


# =============================================================================
# Training Data Collection
# =============================================================================

@dataclass
class TrainingSample:
    """A single training sample with features and label."""
    features: np.ndarray
    label: float
    sample_type: str  # 'solution', 'placement', 'routing', 'move_cost'
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingDataCollector:
    """Collects training data from solver runs."""

    def __init__(self, db_path: str = "ml_training.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for training data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_type TEXT NOT NULL,
                features BLOB NOT NULL,
                label REAL NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sample_type ON training_samples(sample_type)
        ''')

        conn.commit()
        conn.close()

    def add_sample(self, sample: TrainingSample):
        """Add a training sample to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO training_samples (sample_type, features, label, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            sample.sample_type,
            sample.features.tobytes(),
            sample.label,
            json.dumps(sample.metadata),
        ))

        conn.commit()
        conn.close()

    def get_samples(self, sample_type: str, limit: int = None) -> List[TrainingSample]:
        """Get training samples of a specific type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = 'SELECT features, label, metadata FROM training_samples WHERE sample_type = ?'
        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, (sample_type,))
        rows = cursor.fetchall()
        conn.close()

        samples = []
        for features_blob, label, metadata_json in rows:
            features = np.frombuffer(features_blob, dtype=np.float32)
            metadata = json.loads(metadata_json) if metadata_json else {}
            samples.append(TrainingSample(
                features=features,
                label=label,
                sample_type=sample_type,
                metadata=metadata,
            ))

        return samples

    def get_sample_count(self, sample_type: str = None) -> int:
        """Get count of training samples."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if sample_type:
            cursor.execute(
                'SELECT COUNT(*) FROM training_samples WHERE sample_type = ?',
                (sample_type,)
            )
        else:
            cursor.execute('SELECT COUNT(*) FROM training_samples')

        count = cursor.fetchone()[0]
        conn.close()
        return count


# =============================================================================
# ML Models
# =============================================================================

class BaseMLModel(ABC):
    """Base class for ML models."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """Make a prediction from features."""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model on data."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from file."""
        pass


class GradientBoostingModel(BaseMLModel):
    """Gradient Boosting model using sklearn."""

    def __init__(self, is_classifier: bool = False):
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for GradientBoostingModel")

        self.is_classifier = is_classifier
        if is_classifier:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def predict(self, features: np.ndarray) -> float:
        if not self.is_fitted:
            return 0.5  # Default prediction

        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        if self.is_classifier:
            probs = self.model.predict_proba(features_scaled)
            return probs[0][1]  # Probability of positive class
        else:
            return self.model.predict(features_scaled)[0]

    def train(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'is_classifier': self.is_classifier,
        }, path)

    def load(self, path: str):
        if not Path(path).exists():
            return False

        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        self.is_classifier = data['is_classifier']
        return True


# =============================================================================
# ML Evaluators
# =============================================================================

class MLSolutionEvaluator(SolutionEvaluator):
    """
    ML-based solution evaluator.

    Predicts solution quality based on learned patterns from successful solutions.
    Falls back to default evaluator if model is not trained.
    """

    def __init__(
        self,
        model_path: str = "models/solution_evaluator.pkl",
        db_path: str = "ml_training.db",
        collect_training_data: bool = True,
        fallback_weight: float = 0.3,  # Weight for fallback evaluator
    ):
        self.model_path = model_path
        self.db_path = db_path
        self.collect_training_data = collect_training_data
        self.fallback_weight = fallback_weight

        # Initialize model
        if HAS_SKLEARN:
            self.model = GradientBoostingModel(is_classifier=False)
            self.model.load(model_path)
        else:
            self.model = None

        # Feature extractor (will be set based on solution info)
        self._extractor = None

        # Fallback evaluator
        self.fallback = DefaultSolutionEvaluator()

        # Training data collector
        if collect_training_data:
            self.collector = TrainingDataCollector(db_path)
        else:
            self.collector = None

    def evaluate(self, solution: SolutionInfo) -> float:
        # Get fallback score
        fallback_score = self.fallback.evaluate(solution)

        if self.model is None or not self.model.is_fitted:
            return fallback_score

        # Extract features
        if self._extractor is None or \
           self._extractor.grid_width != solution.grid_width or \
           self._extractor.grid_height != solution.grid_height:
            self._extractor = FeatureExtractor(
                solution.grid_width,
                solution.grid_height,
                solution.num_floors,
            )

        features = self._extractor.extract_solution_features(solution)

        # ML prediction
        ml_score = self.model.predict(features.to_array())

        # Blend ML and fallback scores
        blended = (1 - self.fallback_weight) * ml_score + self.fallback_weight * fallback_score

        return blended

    def on_solution_found(self, solution: SolutionInfo, fitness: float):
        """Collect training data from successful solutions."""
        if not self.collect_training_data or self.collector is None:
            return

        if self._extractor is None:
            self._extractor = FeatureExtractor(
                solution.grid_width,
                solution.grid_height,
                solution.num_floors,
            )

        features = self._extractor.extract_solution_features(solution)

        sample = TrainingSample(
            features=features.to_array(),
            label=fitness,
            sample_type='solution',
            metadata={
                'num_machines': len(solution.machines),
                'num_belts': solution.routing.total_length,
                'routing_success': solution.routing.success,
                'throughput': solution.throughput_per_output,
            }
        )

        self.collector.add_sample(sample)

    def train(self, min_samples: int = 50) -> bool:
        """Train the model on collected data."""
        if self.collector is None or self.model is None:
            return False

        samples = self.collector.get_samples('solution')
        if len(samples) < min_samples:
            print(f"Not enough samples ({len(samples)} < {min_samples})")
            return False

        X = np.array([s.features for s in samples])
        y = np.array([s.label for s in samples])

        self.model.train(X, y)
        self.model.save(self.model_path)

        print(f"Trained solution evaluator on {len(samples)} samples")
        return True


class MLPlacementEvaluator(PlacementEvaluator):
    """
    ML-based placement evaluator.

    Predicts whether a placement will route successfully.
    Used for early rejection of bad placements.
    """

    def __init__(
        self,
        model_path: str = "models/placement_evaluator.pkl",
        db_path: str = "ml_training.db",
        collect_training_data: bool = True,
        reject_threshold: float = 0.3,  # Reject if predicted success < this
    ):
        self.model_path = model_path
        self.db_path = db_path
        self.collect_training_data = collect_training_data
        self.reject_threshold = reject_threshold

        # Initialize model
        if HAS_SKLEARN:
            self.model = GradientBoostingModel(is_classifier=True)
            self.model.load(model_path)
        else:
            self.model = None

        self._extractor = None

        if collect_training_data:
            self.collector = TrainingDataCollector(db_path)
        else:
            self.collector = None

    def evaluate(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
    ) -> Tuple[float, bool]:
        if self.model is None or not self.model.is_fitted:
            return 0.5, False  # Accept by default

        if self._extractor is None or \
           self._extractor.grid_width != grid_width:
            self._extractor = FeatureExtractor(grid_width, grid_height, num_floors)

        # Extract features for each machine and aggregate
        if not machines:
            return 0.5, False

        machine_features = [
            self._extractor.extract_placement_features(m).to_array()
            for m in machines
        ]

        # Aggregate: mean of all machine features
        agg_features = np.mean(machine_features, axis=0)

        # Add global features
        num_machines = len(machines)
        num_inputs = len(input_positions)
        num_outputs = len(output_positions)

        global_features = np.array([
            num_machines / 10,  # Normalize
            num_inputs / 4,
            num_outputs / 4,
        ], dtype=np.float32)

        features = np.concatenate([agg_features, global_features])

        # Predict
        score = self.model.predict(features)
        should_reject = score < self.reject_threshold

        return score, should_reject

    def record_outcome(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
        routing_success: bool,
    ):
        """Record a placement outcome for training."""
        if not self.collect_training_data or self.collector is None:
            return

        if self._extractor is None:
            self._extractor = FeatureExtractor(grid_width, grid_height, num_floors)

        machine_features = [
            self._extractor.extract_placement_features(m).to_array()
            for m in machines
        ]

        agg_features = np.mean(machine_features, axis=0) if machine_features else np.zeros(17)

        global_features = np.array([
            len(machines) / 10,
            len(input_positions) / 4,
            len(output_positions) / 4,
        ], dtype=np.float32)

        features = np.concatenate([agg_features, global_features])

        sample = TrainingSample(
            features=features,
            label=1.0 if routing_success else 0.0,
            sample_type='placement',
            metadata={
                'num_machines': len(machines),
                'routing_success': routing_success,
            }
        )

        self.collector.add_sample(sample)

    def train(self, min_samples: int = 100) -> bool:
        """Train the model on collected data."""
        if self.collector is None or self.model is None:
            return False

        samples = self.collector.get_samples('placement')
        if len(samples) < min_samples:
            print(f"Not enough samples ({len(samples)} < {min_samples})")
            return False

        X = np.array([s.features for s in samples])
        y = np.array([s.label for s in samples])

        self.model.train(X, y)
        self.model.save(self.model_path)

        print(f"Trained placement evaluator on {len(samples)} samples")
        return True


class MLRoutingHeuristic(RoutingHeuristic):
    """
    ML-enhanced A* heuristic.

    Learns to estimate true path cost more accurately than Manhattan distance.
    """

    def __init__(
        self,
        model_path: str = "models/routing_heuristic.pkl",
        db_path: str = "ml_training.db",
        collect_training_data: bool = True,
        ml_weight: float = 0.5,  # Blend with default heuristic
    ):
        self.model_path = model_path
        self.db_path = db_path
        self.collect_training_data = collect_training_data
        self.ml_weight = ml_weight

        if HAS_SKLEARN:
            self.model = GradientBoostingModel(is_classifier=False)
            self.model.load(model_path)
        else:
            self.model = None

        self._extractor = None
        self._grid_width = 42  # Default
        self._grid_height = 28

        self.fallback = DefaultRoutingHeuristic()

        if collect_training_data:
            self.collector = TrainingDataCollector(db_path)
        else:
            self.collector = None

    def set_grid_size(self, width: int, height: int, num_floors: int = 3):
        """Set grid size for feature extraction."""
        self._grid_width = width
        self._grid_height = height
        self._extractor = FeatureExtractor(width, height, num_floors)

    def __call__(
        self,
        current: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        # Default heuristic
        default_h = self.fallback(current, goal, context)

        if self.model is None or not self.model.is_fitted:
            return default_h

        if self._extractor is None:
            self._extractor = FeatureExtractor(
                self._grid_width, self._grid_height, 4
            )

        # Extract features
        occupied = context.get('occupied', set()) if context else set()
        features = self._extractor.extract_routing_features(current, goal, occupied)

        # ML prediction
        ml_h = self.model.predict(features.to_array())

        # Ensure admissibility: take min of ML and default (never overestimate)
        # But blend for better guidance
        blended = (1 - self.ml_weight) * default_h + self.ml_weight * ml_h

        # Ensure we never overestimate (admissibility)
        return min(blended, default_h * 1.1)  # Allow slight overestimate for speed

    def on_path_found(
        self,
        path: List[Tuple[int, int, int, str]],
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
    ):
        """Record successful path for training."""
        if not self.collect_training_data or self.collector is None:
            return

        if self._extractor is None:
            return

        # The true cost is the path length
        true_cost = len(path)

        # Record samples along the path
        for i, (x, y, z, _) in enumerate(path[:-1]):
            current = (x, y, z)
            remaining_cost = len(path) - i - 1  # Steps to goal

            features = self._extractor.extract_routing_features(current, goal)

            sample = TrainingSample(
                features=features.to_array(),
                label=remaining_cost,
                sample_type='routing',
                metadata={
                    'path_length': len(path),
                    'position_in_path': i,
                }
            )

            self.collector.add_sample(sample)

    def train(self, min_samples: int = 500) -> bool:
        """Train the model on collected data."""
        if self.collector is None or self.model is None:
            return False

        samples = self.collector.get_samples('routing')
        if len(samples) < min_samples:
            print(f"Not enough samples ({len(samples)} < {min_samples})")
            return False

        X = np.array([s.features for s in samples])
        y = np.array([s.label for s in samples])

        self.model.train(X, y)
        self.model.save(self.model_path)

        print(f"Trained routing heuristic on {len(samples)} samples")
        return True


class MLMoveCostFunction(MoveCostFunction):
    """
    ML-based move cost function.

    Learns optimal move costs based on path quality outcomes.
    """

    def __init__(
        self,
        model_path: str = "models/move_cost.pkl",
        db_path: str = "ml_training.db",
        collect_training_data: bool = True,
    ):
        self.model_path = model_path
        self.db_path = db_path
        self.collect_training_data = collect_training_data

        if HAS_SKLEARN:
            self.model = GradientBoostingModel(is_classifier=False)
            self.model.load(model_path)
        else:
            self.model = None

        self.fallback = DefaultMoveCostFunction()

        if collect_training_data:
            self.collector = TrainingDataCollector(db_path)
        else:
            self.collector = None

    def __call__(
        self,
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
        move_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        # Default cost
        default_cost = self.fallback(current, neighbor, move_type, context)

        if self.model is None or not self.model.is_fitted:
            return default_cost

        # Extract features for this move
        features = self._extract_move_features(current, neighbor, move_type)

        # ML prediction
        ml_cost = self.model.predict(features)

        # Ensure positive cost
        return max(0.1, ml_cost)

    def _extract_move_features(
        self,
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
        move_type: str,
    ) -> np.ndarray:
        """Extract features for a single move."""
        cx, cy, cf = current
        nx, ny, nf = neighbor

        dx = nx - cx
        dy = ny - cy
        dz = nf - cf

        # Move type encoding
        is_horizontal = 1 if move_type == 'horizontal' else 0
        is_lift_up = 1 if move_type == 'lift_up' else 0
        is_lift_down = 1 if move_type == 'lift_down' else 0
        is_belt_port = 1 if move_type == 'belt_port' else 0

        # Distance for belt ports
        jump_distance = abs(dx) + abs(dy) if move_type == 'belt_port' else 0

        return np.array([
            dx, dy, dz,
            is_horizontal, is_lift_up, is_lift_down, is_belt_port,
            jump_distance,
        ], dtype=np.float32)

    def record_move(
        self,
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
        move_type: str,
        path_quality: float,  # Higher = better path (e.g., shorter)
    ):
        """Record a move outcome for training."""
        if not self.collect_training_data or self.collector is None:
            return

        features = self._extract_move_features(current, neighbor, move_type)

        # Inverse of path quality as cost (better paths = lower cost moves)
        cost_label = 1.0 / max(0.1, path_quality)

        sample = TrainingSample(
            features=features,
            label=cost_label,
            sample_type='move_cost',
            metadata={
                'move_type': move_type,
                'path_quality': path_quality,
            }
        )

        self.collector.add_sample(sample)

    def train(self, min_samples: int = 200) -> bool:
        """Train the model on collected data."""
        if self.collector is None or self.model is None:
            return False

        samples = self.collector.get_samples('move_cost')
        if len(samples) < min_samples:
            print(f"Not enough samples ({len(samples)} < {min_samples})")
            return False

        X = np.array([s.features for s in samples])
        y = np.array([s.label for s in samples])

        self.model.train(X, y)
        self.model.save(self.model_path)

        print(f"Trained move cost function on {len(samples)} samples")
        return True


# =============================================================================
# Unified ML System
# =============================================================================

class MLEvaluatorSystem:
    """
    Unified system for all ML evaluators.

    Provides a single interface to create, train, and use all ML evaluators.
    """

    def __init__(
        self,
        model_dir: str = "models",
        db_path: str = "ml_training.db",
        collect_training_data: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path

        # Create all evaluators
        self.solution_evaluator = MLSolutionEvaluator(
            model_path=str(self.model_dir / "solution_evaluator.pkl"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

        self.placement_evaluator = MLPlacementEvaluator(
            model_path=str(self.model_dir / "placement_evaluator.pkl"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

        self.routing_heuristic = MLRoutingHeuristic(
            model_path=str(self.model_dir / "routing_heuristic.pkl"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

        self.move_cost_function = MLMoveCostFunction(
            model_path=str(self.model_dir / "move_cost.pkl"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

        self.collector = TrainingDataCollector(db_path)

    def get_evaluators(self) -> Dict[str, Any]:
        """Get all evaluators for use with solve_with_cpsat."""
        return {
            'solution_evaluator': self.solution_evaluator,
            'placement_evaluator': self.placement_evaluator,
            'routing_heuristic': self.routing_heuristic,
            'move_cost_function': self.move_cost_function,
        }

    def train_all(self, min_samples: Dict[str, int] = None) -> Dict[str, bool]:
        """Train all models that have enough data."""
        defaults = {
            'solution': 50,
            'placement': 100,
            'routing': 500,
            'move_cost': 200,
        }
        min_samples = min_samples or defaults

        results = {}

        results['solution'] = self.solution_evaluator.train(min_samples['solution'])
        results['placement'] = self.placement_evaluator.train(min_samples['placement'])
        results['routing'] = self.routing_heuristic.train(min_samples['routing'])
        results['move_cost'] = self.move_cost_function.train(min_samples['move_cost'])

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get training data statistics."""
        return {
            'solution_samples': self.collector.get_sample_count('solution'),
            'placement_samples': self.collector.get_sample_count('placement'),
            'routing_samples': self.collector.get_sample_count('routing'),
            'move_cost_samples': self.collector.get_sample_count('move_cost'),
            'total_samples': self.collector.get_sample_count(),
        }
