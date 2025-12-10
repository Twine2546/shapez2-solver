#!/usr/bin/env python3
"""
Raw Coordinate Placement Model.

A simpler ML model that takes raw placement data:
- Foundation type (encoded)
- Machine positions (x, y, floor)
- Machine types
- Connection coordinates (input/output ports)

Predicts: probability of successful routing
"""

import json
import pickle
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Foundation type encoding
FOUNDATION_TYPES = [
    "1x1", "2x1", "3x1", "4x1",
    "1x2", "1x3", "1x4",
    "2x2", "3x2", "4x2", "2x3", "2x4", "3x3",
    "T", "L", "L4", "S4", "Cross",
]

# Machine type encoding
MACHINE_TYPES = [
    "CUTTER", "STACKER", "SWAPPER", "ROTATOR",
    "PAINTER", "MIXER", "CRYSTAL_GENERATOR",
]

# Max values for padding
MAX_MACHINES = 10
MAX_INPUTS = 16
MAX_OUTPUTS = 16


@dataclass
class RawPlacementData:
    """Raw placement data for ML model."""
    # Foundation
    foundation_type: str
    grid_width: int
    grid_height: int
    num_floors: int

    # Machines: list of (type, x, y, floor)
    machines: List[Tuple[str, int, int, int]]

    # Connections: input ports (x, y, floor), output ports (x, y, floor)
    input_ports: List[Tuple[int, int, int]]
    output_ports: List[Tuple[int, int, int]]

    # Outcome
    routing_success: bool

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to fixed-size feature vector for ML.

        Layout:
        - [0]: foundation_type_id (0-17)
        - [1]: grid_width (normalized)
        - [2]: grid_height (normalized)
        - [3]: num_floors
        - [4]: num_machines
        - [5]: num_inputs
        - [6]: num_outputs
        - [7-46]: machine data (MAX_MACHINES * 4: type_id, x, y, floor)
        - [47-94]: input ports (MAX_INPUTS * 3: x, y, floor)
        - [95-142]: output ports (MAX_OUTPUTS * 3: x, y, floor)
        """
        features = []

        # Foundation encoding
        try:
            foundation_id = FOUNDATION_TYPES.index(self.foundation_type)
        except ValueError:
            foundation_id = 0
        features.append(foundation_id)

        # Grid dimensions (normalized by typical max ~100)
        features.append(self.grid_width / 100.0)
        features.append(self.grid_height / 100.0)
        features.append(self.num_floors)

        # Counts
        features.append(len(self.machines))
        features.append(len(self.input_ports))
        features.append(len(self.output_ports))

        # Machine data (padded to MAX_MACHINES)
        for i in range(MAX_MACHINES):
            if i < len(self.machines):
                m_type, x, y, floor = self.machines[i]
                try:
                    type_id = MACHINE_TYPES.index(m_type)
                except ValueError:
                    type_id = 0
                features.extend([
                    type_id,
                    x / max(1, self.grid_width),  # Normalized position
                    y / max(1, self.grid_height),
                    floor / max(1, self.num_floors),
                ])
            else:
                features.extend([0, 0, 0, 0])  # Padding

        # Input ports (padded to MAX_INPUTS)
        for i in range(MAX_INPUTS):
            if i < len(self.input_ports):
                x, y, floor = self.input_ports[i]
                features.extend([
                    x / max(1, self.grid_width),
                    y / max(1, self.grid_height),
                    floor / max(1, self.num_floors),
                ])
            else:
                features.extend([0, 0, 0])  # Padding

        # Output ports (padded to MAX_OUTPUTS)
        for i in range(MAX_OUTPUTS):
            if i < len(self.output_ports):
                x, y, floor = self.output_ports[i]
                features.extend([
                    x / max(1, self.grid_width),
                    y / max(1, self.grid_height),
                    floor / max(1, self.num_floors),
                ])
            else:
                features.extend([0, 0, 0])  # Padding

        return np.array(features, dtype=np.float32)

    @staticmethod
    def feature_size() -> int:
        """Total feature vector size."""
        return 7 + (MAX_MACHINES * 4) + (MAX_INPUTS * 3) + (MAX_OUTPUTS * 3)


class PlacementDatabase:
    """SQLite database for raw placement data."""

    def __init__(self, db_path: str = "placement_raw.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS placements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                foundation_type TEXT NOT NULL,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,
                machines_json TEXT,
                input_ports_json TEXT,
                output_ports_json TEXT,
                feature_vector TEXT,
                routing_success INTEGER,
                solve_time REAL,
                num_belts INTEGER,
                failure_reason TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_foundation
            ON placements(foundation_type)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_success
            ON placements(routing_success)
        ''')

        conn.commit()
        conn.close()

    def log_placement(
        self,
        foundation_type: str,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        machines: List[Tuple[str, int, int, int]],
        input_ports: List[Tuple[int, int, int]],
        output_ports: List[Tuple[int, int, int]],
        routing_success: bool,
        solve_time: float = 0.0,
        num_belts: int = 0,
        failure_reason: str = "",
    ):
        """Log a placement attempt."""
        data = RawPlacementData(
            foundation_type=foundation_type,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machines=machines,
            input_ports=input_ports,
            output_ports=output_ports,
            routing_success=routing_success,
        )

        feature_vector = data.to_feature_vector()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO placements (
                timestamp, foundation_type, grid_width, grid_height, num_floors,
                machines_json, input_ports_json, output_ports_json,
                feature_vector, routing_success, solve_time, num_belts, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            foundation_type,
            grid_width,
            grid_height,
            num_floors,
            json.dumps(machines),
            json.dumps(input_ports),
            json.dumps(output_ports),
            json.dumps(feature_vector.tolist()),
            1 if routing_success else 0,
            solve_time,
            num_belts,
            failure_reason,
        ))

        conn.commit()
        conn.close()

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data as (X, y) arrays."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT feature_vector, routing_success FROM placements")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None, None

        X = np.array([json.loads(row[0]) for row in rows])
        y = np.array([row[1] for row in rows])

        return X, y

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM placements")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM placements WHERE routing_success = 1")
        successes = cursor.fetchone()[0]

        cursor.execute("""
            SELECT foundation_type, COUNT(*), SUM(routing_success)
            FROM placements GROUP BY foundation_type
        """)
        by_foundation = {
            row[0]: {'total': row[1], 'successes': row[2]}
            for row in cursor.fetchall()
        }

        conn.close()

        return {
            'total': total,
            'successes': successes,
            'failures': total - successes,
            'success_rate': successes / max(1, total),
            'by_foundation': by_foundation,
        }


class RawPlacementModel:
    """
    ML model for predicting placement success from raw coordinates.
    """

    def __init__(self, model_path: str = "models/placement_raw.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False

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
                    print(f"Loaded placement model from {self.model_path}")
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

    def train(self, db: PlacementDatabase, min_samples: int = 50) -> bool:
        """Train model from database."""
        if not HAS_SKLEARN:
            print("sklearn not available")
            return False

        X, y = db.get_training_data()

        if X is None or len(X) < min_samples:
            print(f"Not enough data (need {min_samples}, have {len(X) if X is not None else 0})")
            return False

        # Check for both classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Need both success and failure samples (have only class {unique_classes})")
            return False

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)

        print(f"Training accuracy: {train_acc:.1%}")
        print(f"Validation accuracy: {val_acc:.1%}")

        self.is_trained = True
        self._save_model()

        return True

    def predict(self, data: RawPlacementData) -> float:
        """Predict success probability for a placement."""
        if not self.is_trained or self.model is None:
            return 0.5  # Default when not trained

        X = data.to_feature_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0]
        # Return probability of success (class 1)
        return proba[1] if len(proba) > 1 else proba[0]

    def predict_from_raw(
        self,
        foundation_type: str,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        machines: List[Tuple[str, int, int, int]],
        input_ports: List[Tuple[int, int, int]],
        output_ports: List[Tuple[int, int, int]],
    ) -> float:
        """Predict directly from raw values."""
        data = RawPlacementData(
            foundation_type=foundation_type,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machines=machines,
            input_ports=input_ports,
            output_ports=output_ports,
            routing_success=False,  # Dummy value
        )
        return self.predict(data)


class PlacementPredictor:
    """
    High-level interface for placement prediction and logging.
    """

    def __init__(
        self,
        db_path: str = "placement_raw.db",
        model_path: str = "models/placement_raw.pkl",
    ):
        self.db = PlacementDatabase(db_path)
        self.model = RawPlacementModel(model_path)

    def predict(
        self,
        foundation_type: str,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        machines: List[Tuple[str, int, int, int]],
        input_ports: List[Tuple[int, int, int]],
        output_ports: List[Tuple[int, int, int]],
    ) -> float:
        """Predict success probability for a placement."""
        return self.model.predict_from_raw(
            foundation_type, grid_width, grid_height, num_floors,
            machines, input_ports, output_ports
        )

    def log_result(
        self,
        foundation_type: str,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        machines: List[Tuple[str, int, int, int]],
        input_ports: List[Tuple[int, int, int]],
        output_ports: List[Tuple[int, int, int]],
        routing_success: bool,
        solve_time: float = 0.0,
        num_belts: int = 0,
        failure_reason: str = "",
    ):
        """Log a placement result to the database."""
        self.db.log_placement(
            foundation_type=foundation_type,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machines=machines,
            input_ports=input_ports,
            output_ports=output_ports,
            routing_success=routing_success,
            solve_time=solve_time,
            num_belts=num_belts,
            failure_reason=failure_reason,
        )

    def train(self, min_samples: int = 50) -> bool:
        """Train model from logged data."""
        return self.model.train(self.db, min_samples)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged placements."""
        stats = self.db.get_stats()
        stats['model_trained'] = self.model.is_trained
        return stats


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Raw Placement Model")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--min-samples", type=int, default=50, help="Min samples for training")
    parser.add_argument("--db", type=str, default="placement_raw.db", help="Database path")
    parser.add_argument("--model", type=str, default="models/placement_raw.pkl", help="Model path")

    args = parser.parse_args()

    predictor = PlacementPredictor(db_path=args.db, model_path=args.model)

    if args.stats:
        stats = predictor.get_stats()
        print("\nPlacement Database Statistics")
        print("=" * 40)
        print(f"Total placements: {stats['total']}")
        print(f"Successes: {stats['successes']}")
        print(f"Failures: {stats['failures']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Model trained: {stats['model_trained']}")

        if stats['by_foundation']:
            print("\nBy foundation:")
            for foundation, data in stats['by_foundation'].items():
                rate = data['successes'] / max(1, data['total'])
                print(f"  {foundation}: {data['successes']}/{data['total']} ({rate:.1%})")

    if args.train:
        print("\nTraining placement model...")
        success = predictor.train(min_samples=args.min_samples)
        if success:
            print("Training complete!")
        else:
            print("Training failed")
