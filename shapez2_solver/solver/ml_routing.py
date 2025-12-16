"""
ML-Enhanced Routing Components.

Provides learnable functions for A* routing that consider:
- Cell value (which cells should be kept clear for future paths)
- Congestion awareness (avoid crowded areas)
- Path ordering (which connection to route first)

These components plug into the existing router via heuristic_fn and move_cost_fn.
"""

import json
import sqlite3
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import time

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RoutingState:
    """Current state of routing for ML decision making."""
    grid_width: int
    grid_height: int
    num_floors: int
    occupied: Set[Tuple[int, int, int]]  # Currently occupied cells
    remaining_connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]  # (start, goal) pairs
    current_connection_idx: int
    all_goals: List[Tuple[int, int, int]]  # All goal positions

    def get_occupancy_grid(self) -> np.ndarray:
        """Convert occupied set to 3D grid."""
        grid = np.zeros((self.num_floors, self.grid_height, self.grid_width), dtype=np.float32)
        for x, y, f in self.occupied:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height and 0 <= f < self.num_floors:
                grid[f, y, x] = 1.0
        return grid

    def get_goals_grid(self) -> np.ndarray:
        """Convert remaining goals to 3D grid."""
        grid = np.zeros((self.num_floors, self.grid_height, self.grid_width), dtype=np.float32)
        for start, goal in self.remaining_connections:
            gx, gy, gf = goal
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height and 0 <= gf < self.num_floors:
                grid[gf, gy, gx] = 1.0
        return grid


@dataclass
class RoutingOutcome:
    """Outcome of a routing attempt for training."""
    grid_width: int
    grid_height: int
    num_floors: int
    connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    connection_order: List[int]  # Order in which connections were routed
    paths: List[List[Tuple[int, int, int]]]  # Resulting paths (empty if failed)
    success: bool
    failed_at_connection: Optional[int]  # Which connection failed (if any)
    cells_used: Set[Tuple[int, int, int]]  # All cells used by successful paths


# =============================================================================
# Cell Value Predictor (CNN)
# =============================================================================

class CellValueCNN(nn.Module):
    """CNN to predict cell value (importance for future routing)."""

    def __init__(self, num_floors: int = 4, hidden_dim: int = 32):
        super().__init__()

        # Input: 3 channels per floor (occupancy, current_goal, remaining_goals)
        in_channels = 3 * num_floors

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)

        # Output: 1 channel per floor (cell value)
        self.conv_out = nn.Conv2d(hidden_dim, num_floors, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width) - stacked floor channels

        Returns:
            (batch, num_floors, height, width) - cell values per floor
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.sigmoid(self.conv_out(x))  # 0-1 cell values
        return x


class CellValuePredictor:
    """
    Predicts which cells are valuable for future routing.

    High-value cells should be avoided to keep them free for future paths.
    Used to add penalty to move cost for valuable cells.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        db_path: str = "routing_ml.db",
        hidden_dim: int = 32,
        num_floors: int = 4,
        collect_training_data: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for CellValuePredictor")

        self.model_path = Path(model_path) if model_path else None
        self.db_path = db_path
        self.hidden_dim = hidden_dim
        self.num_floors = num_floors
        self.collect_training_data = collect_training_data

        self.model = CellValueCNN(num_floors=num_floors, hidden_dim=hidden_dim)
        self.model.eval()

        if self.model_path and self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

        self._init_db()

        # Cache for current routing
        self._current_values: Optional[np.ndarray] = None
        self._state: Optional[RoutingState] = None

    def _init_db(self):
        """Initialize training database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cell_value_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grid_data BLOB NOT NULL,
                label_data BLOB NOT NULL,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def update_state(self, state: RoutingState):
        """Update internal state and recompute cell values."""
        self._state = state
        self._current_values = self._compute_values(state)

    def _compute_values(self, state: RoutingState) -> np.ndarray:
        """Compute cell values for current state."""
        # Build input tensor
        occupancy = state.get_occupancy_grid()
        goals = state.get_goals_grid()

        # Current goal channel
        current_goal = np.zeros_like(occupancy)
        if state.remaining_connections:
            _, goal = state.remaining_connections[0]
            gx, gy, gf = goal
            if 0 <= gx < state.grid_width and 0 <= gy < state.grid_height and 0 <= gf < state.num_floors:
                current_goal[gf, gy, gx] = 1.0

        # Stack channels: [occ_f0, goal_f0, remaining_f0, occ_f1, ...]
        channels = []
        for f in range(state.num_floors):
            if f < occupancy.shape[0]:
                channels.extend([occupancy[f], current_goal[f], goals[f]])
            else:
                channels.extend([np.zeros((state.grid_height, state.grid_width), dtype=np.float32)] * 3)

        # Pad to expected number of floors
        while len(channels) < 3 * self.num_floors:
            channels.append(np.zeros((state.grid_height, state.grid_width), dtype=np.float32))

        x = torch.tensor(np.stack(channels[:3*self.num_floors]), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            values = self.model(x).squeeze(0).numpy()

        return values

    def get_cell_value(self, x: int, y: int, floor: int) -> float:
        """Get predicted value of a cell (0-1, higher = more valuable = avoid)."""
        if self._current_values is None:
            return 0.0
        if 0 <= floor < self._current_values.shape[0]:
            if 0 <= y < self._current_values.shape[1] and 0 <= x < self._current_values.shape[2]:
                return float(self._current_values[floor, y, x])
        return 0.0

    def record_outcome(self, outcome: RoutingOutcome):
        """Record routing outcome for training."""
        if not self.collect_training_data:
            return

        # Generate training labels: cells that were on successful paths = 1
        # cells that blocked future paths = high value
        labels = np.zeros((outcome.num_floors, outcome.grid_height, outcome.grid_width), dtype=np.float32)

        # Mark cells used by successful paths
        for cell in outcome.cells_used:
            x, y, f = cell
            if 0 <= x < outcome.grid_width and 0 <= y < outcome.grid_height and 0 <= f < outcome.num_floors:
                labels[f, y, x] = 1.0

        # If routing failed, we'd ideally know which cells blocked
        # For now, just record the successful path cells

        # Build input (initial state before routing)
        occupancy = np.zeros((outcome.num_floors, outcome.grid_height, outcome.grid_width), dtype=np.float32)
        goals = np.zeros_like(occupancy)
        for start, goal in outcome.connections:
            gx, gy, gf = goal
            if 0 <= gx < outcome.grid_width and 0 <= gy < outcome.grid_height and 0 <= gf < outcome.num_floors:
                goals[gf, gy, gx] = 1.0

        # Stack input channels
        channels = []
        for f in range(outcome.num_floors):
            channels.extend([occupancy[f], np.zeros_like(occupancy[f]), goals[f]])

        grid_data = np.stack(channels).astype(np.float32)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO cell_value_samples (grid_data, label_data, grid_width, grid_height, num_floors) VALUES (?, ?, ?, ?, ?)",
            (grid_data.tobytes(), labels.tobytes(), outcome.grid_width, outcome.grid_height, outcome.num_floors)
        )
        conn.commit()
        conn.close()

    def train(self, epochs: int = 50, batch_size: int = 16, lr: float = 0.001) -> Dict[str, Any]:
        """Train the cell value model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT grid_data, label_data, grid_width, grid_height, num_floors FROM cell_value_samples")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return {"trained": False, "reason": "Insufficient data", "samples": len(rows)}

        # Find max dimensions for padding
        max_w = max(r[2] for r in rows)
        max_h = max(r[3] for r in rows)

        X = []
        y = []

        for grid_bytes, label_bytes, gw, gh, nf in rows:
            grid = np.frombuffer(grid_bytes, dtype=np.float32).reshape(3 * nf, gh, gw)
            labels = np.frombuffer(label_bytes, dtype=np.float32).reshape(nf, gh, gw)

            # Pad to max size
            padded_grid = np.zeros((3 * self.num_floors, max_h, max_w), dtype=np.float32)
            padded_labels = np.zeros((self.num_floors, max_h, max_w), dtype=np.float32)

            c = min(3 * nf, 3 * self.num_floors)
            padded_grid[:c, :gh, :gw] = grid[:c]
            padded_labels[:min(nf, self.num_floors), :gh, :gw] = labels[:min(nf, self.num_floors)]

            X.append(padded_grid)
            y.append(padded_labels)

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        self.model = CellValueCNN(num_floors=self.num_floors, hidden_dim=self.hidden_dim)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        losses = []
        for epoch in range(epochs):
            perm = torch.randperm(len(X))
            epoch_loss = 0.0

            for i in range(0, len(X), batch_size):
                batch_X = X[perm[i:i+batch_size]]
                batch_y = y[perm[i:i+batch_size]]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss)

        self.model.eval()

        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)

        return {"trained": True, "samples": len(rows), "final_loss": losses[-1], "epochs": epochs}


# =============================================================================
# Congestion-Aware Move Cost
# =============================================================================

class CongestionMoveCost:
    """
    Move cost function that considers local congestion.

    Higher cost for moves into congested areas or toward cells needed by future paths.
    """

    def __init__(
        self,
        cell_value_predictor: Optional[CellValuePredictor] = None,
        base_horizontal_cost: float = 1.0,
        base_lift_cost: float = 1.5,
        base_belt_port_cost: float = 2.0,
        congestion_weight: float = 2.0,
        cell_value_weight: float = 3.0,
        window_size: int = 5,
    ):
        self.cell_value_predictor = cell_value_predictor
        self.base_horizontal_cost = base_horizontal_cost
        self.base_lift_cost = base_lift_cost
        self.base_belt_port_cost = base_belt_port_cost
        self.congestion_weight = congestion_weight
        self.cell_value_weight = cell_value_weight
        self.window_size = window_size

        # State
        self._state: Optional[RoutingState] = None
        self._congestion_map: Optional[np.ndarray] = None

    def update_state(self, state: RoutingState):
        """Update internal state and recompute congestion map."""
        self._state = state
        self._congestion_map = self._compute_congestion(state)

        if self.cell_value_predictor:
            self.cell_value_predictor.update_state(state)

    def _compute_congestion(self, state: RoutingState) -> np.ndarray:
        """Compute local congestion for each cell (fraction of occupied neighbors)."""
        occupancy = state.get_occupancy_grid()
        congestion = np.zeros_like(occupancy)

        half = self.window_size // 2

        for f in range(state.num_floors):
            for y in range(state.grid_height):
                for x in range(state.grid_width):
                    # Count occupied neighbors in window
                    count = 0
                    total = 0
                    for dy in range(-half, half + 1):
                        for dx in range(-half, half + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < state.grid_width and 0 <= ny < state.grid_height:
                                total += 1
                                if f < occupancy.shape[0] and occupancy[f, ny, nx] > 0:
                                    count += 1
                    congestion[f, y, x] = count / max(total, 1)

        return congestion

    def get_congestion(self, x: int, y: int, floor: int) -> float:
        """Get congestion at a cell (0-1)."""
        if self._congestion_map is None:
            return 0.0
        if 0 <= floor < self._congestion_map.shape[0]:
            if 0 <= y < self._congestion_map.shape[1] and 0 <= x < self._congestion_map.shape[2]:
                return float(self._congestion_map[floor, y, x])
        return 0.0

    def __call__(
        self,
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
        move_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate move cost with congestion and cell value awareness.

        Args:
            current: (x, y, floor) current position
            neighbor: (x, y, floor) next position
            move_type: 'horizontal', 'lift_up', 'lift_down', 'belt_port'
            context: Optional routing context

        Returns:
            Move cost (higher = less desirable)
        """
        # Base cost
        if move_type in ('lift_up', 'lift_down'):
            base_cost = self.base_lift_cost
        elif move_type == 'belt_port':
            distance = abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1])
            base_cost = self.base_belt_port_cost + 0.1 * distance
        else:
            base_cost = self.base_horizontal_cost

        # Add congestion penalty
        nx, ny, nf = neighbor
        congestion = self.get_congestion(nx, ny, nf)
        congestion_penalty = congestion * self.congestion_weight

        # Add cell value penalty (avoid valuable cells)
        cell_value_penalty = 0.0
        if self.cell_value_predictor:
            cell_value = self.cell_value_predictor.get_cell_value(nx, ny, nf)
            cell_value_penalty = cell_value * self.cell_value_weight

        return base_cost + congestion_penalty + cell_value_penalty


# =============================================================================
# Path Ordering Predictor
# =============================================================================

class PathOrderingPredictor:
    """
    Predicts optimal order to route connections.

    Goal: Route connections that are easy to block LAST,
    and connections with more flexibility FIRST.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        db_path: str = "routing_ml.db",
        collect_training_data: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.db_path = db_path
        self.collect_training_data = collect_training_data

        # Simple MLP for scoring connections
        if HAS_TORCH:
            self.model = nn.Sequential(
                nn.Linear(12, 32),  # Features per connection
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),  # Priority score
            )
            self.model.eval()

            if self.model_path and self.model_path.exists():
                self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        else:
            self.model = None

        self._init_db()

    def _init_db(self):
        """Initialize training database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS path_ordering_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                connection_features BLOB NOT NULL,
                optimal_order TEXT NOT NULL,
                success INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _extract_features(
        self,
        connection: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        state: RoutingState,
    ) -> np.ndarray:
        """Extract features for a single connection."""
        start, goal = connection
        sx, sy, sf = start
        gx, gy, gf = goal

        # Normalize positions
        norm_sx = sx / max(state.grid_width - 1, 1)
        norm_sy = sy / max(state.grid_height - 1, 1)
        norm_sf = sf / max(state.num_floors - 1, 1)
        norm_gx = gx / max(state.grid_width - 1, 1)
        norm_gy = gy / max(state.grid_height - 1, 1)
        norm_gf = gf / max(state.num_floors - 1, 1)

        # Distances
        manhattan = abs(gx - sx) + abs(gy - sy) + abs(gf - sf)
        floor_diff = abs(gf - sf)

        # Edge proximity (connections near edges may be harder to route around)
        edge_dist_start = min(sx, sy, state.grid_width - 1 - sx, state.grid_height - 1 - sy)
        edge_dist_goal = min(gx, gy, state.grid_width - 1 - gx, state.grid_height - 1 - gy)

        # Occupancy near start/goal
        occ_near_start = sum(1 for cell in state.occupied
                            if abs(cell[0] - sx) <= 2 and abs(cell[1] - sy) <= 2 and cell[2] == sf)
        occ_near_goal = sum(1 for cell in state.occupied
                           if abs(cell[0] - gx) <= 2 and abs(cell[1] - gy) <= 2 and cell[2] == gf)

        return np.array([
            norm_sx, norm_sy, norm_sf,
            norm_gx, norm_gy, norm_gf,
            manhattan / 20.0,  # Normalized distance
            floor_diff / max(state.num_floors - 1, 1),
            edge_dist_start / 5.0,
            edge_dist_goal / 5.0,
            occ_near_start / 10.0,
            occ_near_goal / 10.0,
        ], dtype=np.float32)

    def predict_order(
        self,
        connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
        state: RoutingState,
    ) -> List[int]:
        """
        Predict optimal order to route connections.

        Returns:
            List of connection indices in suggested order.
        """
        if not connections:
            return []

        if not HAS_TORCH or self.model is None:
            # Fallback: sort by distance (longer first - gives more flexibility)
            distances = []
            for i, (start, goal) in enumerate(connections):
                d = abs(goal[0] - start[0]) + abs(goal[1] - start[1]) + abs(goal[2] - start[2])
                distances.append((d, i))
            distances.sort(reverse=True)
            return [i for _, i in distances]

        # Extract features for all connections
        features = np.array([self._extract_features(c, state) for c in connections])
        x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            scores = self.model(x).squeeze(-1).numpy()

        # Sort by score (higher = route first)
        order = np.argsort(-scores).tolist()
        return order

    def record_outcome(self, outcome: RoutingOutcome):
        """Record routing outcome for training."""
        if not self.collect_training_data:
            return

        # Create initial state
        state = RoutingState(
            grid_width=outcome.grid_width,
            grid_height=outcome.grid_height,
            num_floors=outcome.num_floors,
            occupied=set(),
            remaining_connections=outcome.connections,
            current_connection_idx=0,
            all_goals=[g for _, g in outcome.connections],
        )

        # Extract features for each connection
        features = np.array([self._extract_features(c, state) for c in outcome.connections])

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO path_ordering_samples (connection_features, optimal_order, success) VALUES (?, ?, ?)",
            (features.tobytes(), json.dumps(outcome.connection_order), 1 if outcome.success else 0)
        )
        conn.commit()
        conn.close()

    def train(self, epochs: int = 100, lr: float = 0.001) -> Dict[str, Any]:
        """Train the path ordering model."""
        if not HAS_TORCH:
            return {"trained": False, "reason": "PyTorch not available"}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT connection_features, optimal_order, success FROM path_ordering_samples WHERE success = 1")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return {"trained": False, "reason": "Insufficient successful examples", "samples": len(rows)}

        # Build training data: connection features -> order score
        # Connections routed first get higher scores
        X = []
        y = []

        for feat_bytes, order_json, _ in rows:
            order = json.loads(order_json)
            num_conn = len(order)
            features = np.frombuffer(feat_bytes, dtype=np.float32).reshape(num_conn, 12)

            for i, conn_idx in enumerate(order):
                X.append(features[conn_idx])
                # Score: higher for earlier in order
                y.append((num_conn - i) / num_conn)

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        self.model = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            perm = torch.randperm(len(X))
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            optimizer.zero_grad()
            outputs = self.model(X_shuffled)
            loss = criterion(outputs, y_shuffled)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        self.model.eval()

        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)

        return {"trained": True, "samples": len(rows), "final_loss": losses[-1], "epochs": epochs}


# =============================================================================
# ML-Enhanced Heuristic
# =============================================================================

class MLRoutingHeuristic:
    """
    A* heuristic that considers cell values and congestion.

    Guides routing toward paths that avoid blocking future connections.
    """

    def __init__(
        self,
        cell_value_predictor: Optional[CellValuePredictor] = None,
        base_floor_weight: float = 2.0,
        cell_value_weight: float = 1.0,
    ):
        self.cell_value_predictor = cell_value_predictor
        self.base_floor_weight = base_floor_weight
        self.cell_value_weight = cell_value_weight

        self._state: Optional[RoutingState] = None

    def update_state(self, state: RoutingState):
        """Update internal state."""
        self._state = state
        if self.cell_value_predictor:
            self.cell_value_predictor.update_state(state)

    def __call__(
        self,
        current: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Heuristic function for A*.

        Returns estimated cost from current to goal, biased away from valuable cells.
        """
        cx, cy, cf = current
        gx, gy, gf = goal

        # Base Manhattan distance
        base_h = abs(gx - cx) + abs(gy - cy) + abs(gf - cf) * self.base_floor_weight

        # Add penalty for routing through valuable areas
        # (This is a rough estimate - we add some value for cells along the direct path)
        if self.cell_value_predictor and self._state:
            # Sample a few cells along direct path
            steps = max(abs(gx - cx), abs(gy - cy), 1)
            value_sum = 0.0
            for i in range(steps + 1):
                t = i / steps
                px = int(cx + (gx - cx) * t)
                py = int(cy + (gy - cy) * t)
                # Average floors if crossing
                pf = cf if cf == gf else (cf if t < 0.5 else gf)
                value_sum += self.cell_value_predictor.get_cell_value(px, py, pf)
            avg_value = value_sum / (steps + 1)
            base_h += avg_value * self.cell_value_weight * steps

        return base_h


# =============================================================================
# Unified Routing ML System
# =============================================================================

class RoutingMLSystem:
    """
    Unified system for ML-enhanced routing.

    Coordinates all ML components and provides simple interface for training and inference.
    """

    def __init__(
        self,
        model_dir: str = "models",
        db_path: str = "routing_ml.db",
        collect_training_data: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.db_path = db_path
        self.collect_training_data = collect_training_data

        # Initialize components
        self.cell_value_predictor = CellValuePredictor(
            model_path=str(self.model_dir / "cell_value.pt"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        ) if HAS_TORCH else None

        self.path_ordering = PathOrderingPredictor(
            model_path=str(self.model_dir / "path_ordering.pt"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

        # Create move cost and heuristic functions
        self.move_cost = CongestionMoveCost(
            cell_value_predictor=self.cell_value_predictor,
        )

        self.heuristic = MLRoutingHeuristic(
            cell_value_predictor=self.cell_value_predictor,
        )

    def update_state(self, state: RoutingState):
        """Update all components with new routing state."""
        self.move_cost.update_state(state)
        self.heuristic.update_state(state)

    def get_connection_order(
        self,
        connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        occupied: Optional[Set[Tuple[int, int, int]]] = None,
    ) -> List[int]:
        """Get suggested order to route connections."""
        state = RoutingState(
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            occupied=occupied or set(),
            remaining_connections=connections,
            current_connection_idx=0,
            all_goals=[g for _, g in connections],
        )
        return self.path_ordering.predict_order(connections, state)

    def record_outcome(self, outcome: RoutingOutcome):
        """Record routing outcome for training all components."""
        if self.cell_value_predictor:
            self.cell_value_predictor.record_outcome(outcome)
        self.path_ordering.record_outcome(outcome)

    def train_all(self, epochs: int = 50) -> Dict[str, Any]:
        """Train all ML components."""
        results = {}

        if self.cell_value_predictor:
            results['cell_value'] = self.cell_value_predictor.train(epochs=epochs)

        results['path_ordering'] = self.path_ordering.train(epochs=epochs)

        return results

    def get_functions(self) -> Dict[str, Any]:
        """Get heuristic and move cost functions for use with router."""
        return {
            'heuristic_fn': self.heuristic,
            'move_cost_fn': self.move_cost,
        }


# =============================================================================
# Convenience function to create ML-enhanced router
# =============================================================================

def create_ml_router_functions(
    model_dir: str = "models",
    db_path: str = "routing_ml.db",
    collect_training_data: bool = True,
) -> Tuple[Any, Any, RoutingMLSystem]:
    """
    Create ML-enhanced heuristic and move cost functions.

    Returns:
        (heuristic_fn, move_cost_fn, ml_system)

    Usage:
        heuristic, move_cost, ml_system = create_ml_router_functions()
        router = BeltRouter(..., heuristic_fn=heuristic, move_cost_fn=move_cost)
    """
    ml_system = RoutingMLSystem(
        model_dir=model_dir,
        db_path=db_path,
        collect_training_data=collect_training_data,
    )

    return ml_system.heuristic, ml_system.move_cost, ml_system


# =============================================================================
# Helper function to extract RoutingOutcome from router
# =============================================================================

def extract_routing_outcome_from_router(
    router,
    connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    routing_success: bool,
) -> RoutingOutcome:
    """
    Extract a RoutingOutcome from a router after route_all completes.

    This function reads the router's internal state to build a complete
    training sample without modifying the router itself.

    Args:
        router: BeltRouter instance after routing
        connections: List of (start, goal) tuples that were routed
        routing_success: Whether all connections were successfully routed

    Returns:
        RoutingOutcome with full routing data for ML training
    """
    # Get paths for each connection (in order they were routed)
    paths = []
    connection_order = []

    # Router stores paths by connection index
    for i in range(len(connections)):
        if i in router.connection_paths:
            paths.append(router.connection_paths[i])
            connection_order.append(i)
        else:
            paths.append([])  # Failed connection

    # Determine which connection failed (if any)
    failed_at = None
    if not routing_success and router.failed_connections:
        # First failed connection
        failed_at = router.failed_connections[0].get('index', 0)

    # Collect all cells used by successful paths
    cells_used = set()
    for path in paths:
        cells_used.update(path)

    return RoutingOutcome(
        grid_width=router.grid_width,
        grid_height=router.grid_height,
        num_floors=router.num_floors,
        connections=connections,
        connection_order=connection_order,
        paths=paths,
        success=routing_success,
        failed_at_connection=failed_at,
        cells_used=cells_used,
    )


# =============================================================================
# Routing Data Storage
# =============================================================================

class RoutingDataStore:
    """
    Stores routing outcomes for ML training.

    Provides methods to:
    - Store full routing outcomes with conflict analysis
    - Query training data by success/failure
    - Generate training samples for cell value and path ordering
    """

    def __init__(self, db_path: str = "routing_ml.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables for routing data."""
        conn = sqlite3.connect(self.db_path)

        # Full routing outcomes table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,
                connections_json TEXT NOT NULL,
                connection_order_json TEXT NOT NULL,
                paths_json TEXT NOT NULL,
                success INTEGER,
                failed_at_connection INTEGER,
                cells_used_json TEXT NOT NULL,
                conflict_analysis_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Cell blocking data - which cells blocked which connections
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cell_blocking_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                outcome_id INTEGER,
                cell_x INTEGER,
                cell_y INTEGER,
                cell_floor INTEGER,
                placed_by_connection INTEGER,
                blocked_connections_json TEXT,
                FOREIGN KEY (outcome_id) REFERENCES routing_outcomes(id)
            )
        """)

        conn.commit()
        conn.close()

    def store_outcome(
        self,
        outcome: RoutingOutcome,
        conflict_analysis: Optional[Dict] = None,
    ) -> int:
        """
        Store a routing outcome.

        Args:
            outcome: RoutingOutcome to store
            conflict_analysis: Optional conflict analysis from router.get_conflict_analysis()

        Returns:
            ID of the stored outcome
        """
        conn = sqlite3.connect(self.db_path)

        # Serialize data
        connections_json = json.dumps([
            [list(start), list(goal)]
            for start, goal in outcome.connections
        ])
        paths_json = json.dumps([
            [list(pos) for pos in path]
            for path in outcome.paths
        ])
        cells_used_json = json.dumps([list(cell) for cell in outcome.cells_used])
        conflict_json = json.dumps(conflict_analysis) if conflict_analysis else None

        cursor = conn.execute("""
            INSERT INTO routing_outcomes (
                grid_width, grid_height, num_floors,
                connections_json, connection_order_json, paths_json,
                success, failed_at_connection, cells_used_json,
                conflict_analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome.grid_width, outcome.grid_height, outcome.num_floors,
            connections_json, json.dumps(outcome.connection_order), paths_json,
            1 if outcome.success else 0, outcome.failed_at_connection,
            cells_used_json, conflict_json,
        ))

        outcome_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return outcome_id

    def store_cell_blocking(
        self,
        outcome_id: int,
        blocking_info: Dict[Tuple[int, int, int], Dict],
    ):
        """
        Store cell blocking information for conflict analysis.

        Args:
            outcome_id: ID of the routing outcome
            blocking_info: Dict mapping cell -> {'placed_by': int, 'blocked': [int, ...]}
        """
        conn = sqlite3.connect(self.db_path)

        for cell, info in blocking_info.items():
            x, y, floor = cell
            placed_by = info.get('placed_by', -1)
            blocked = info.get('blocked', [])

            conn.execute("""
                INSERT INTO cell_blocking_data (
                    outcome_id, cell_x, cell_y, cell_floor,
                    placed_by_connection, blocked_connections_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (outcome_id, x, y, floor, placed_by, json.dumps(blocked)))

        conn.commit()
        conn.close()

    def load_outcomes(
        self,
        success_only: bool = False,
        failure_only: bool = False,
        limit: int = 1000,
    ) -> List[RoutingOutcome]:
        """Load routing outcomes from database."""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM routing_outcomes"
        conditions = []

        if success_only:
            conditions.append("success = 1")
        elif failure_only:
            conditions.append("success = 0")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY id DESC LIMIT {limit}"

        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()

        outcomes = []
        for row in rows:
            (id_, gw, gh, nf, conn_json, order_json, paths_json,
             success, failed_at, cells_json, conflict_json, created) = row

            connections = [
                (tuple(start), tuple(goal))
                for start, goal in json.loads(conn_json)
            ]
            paths = [
                [tuple(pos) for pos in path]
                for path in json.loads(paths_json)
            ]
            cells_used = {tuple(cell) for cell in json.loads(cells_json)}

            outcomes.append(RoutingOutcome(
                grid_width=gw,
                grid_height=gh,
                num_floors=nf,
                connections=connections,
                connection_order=json.loads(order_json),
                paths=paths,
                success=bool(success),
                failed_at_connection=failed_at,
                cells_used=cells_used,
            ))

        return outcomes

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored routing data."""
        conn = sqlite3.connect(self.db_path)

        total = conn.execute("SELECT COUNT(*) FROM routing_outcomes").fetchone()[0]
        successful = conn.execute(
            "SELECT COUNT(*) FROM routing_outcomes WHERE success = 1"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM routing_outcomes WHERE success = 0"
        ).fetchone()[0]
        blocking_records = conn.execute(
            "SELECT COUNT(*) FROM cell_blocking_data"
        ).fetchone()[0]

        conn.close()

        return {
            "total_outcomes": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / max(total, 1),
            "blocking_records": blocking_records,
        }


# =============================================================================
# Enhanced RoutingMLSystem with data storage
# =============================================================================

class EnhancedRoutingMLSystem(RoutingMLSystem):
    """
    Extended RoutingMLSystem with full routing data storage.

    Adds:
    - Storage of complete routing outcomes
    - Conflict analysis recording
    - Cell blocking pattern learning
    """

    def __init__(
        self,
        model_dir: str = "models",
        db_path: str = "routing_ml.db",
        collect_training_data: bool = True,
    ):
        super().__init__(model_dir, db_path, collect_training_data)
        self.data_store = RoutingDataStore(db_path)

    def record_full_outcome(
        self,
        router,
        connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
        routing_success: bool,
    ) -> int:
        """
        Record a complete routing outcome from a router instance.

        This extracts all relevant data from the router and stores it
        for ML training.

        Args:
            router: BeltRouter instance after routing
            connections: List of (start, goal) tuples
            routing_success: Whether routing succeeded

        Returns:
            ID of the stored outcome
        """
        if not self.collect_training_data:
            return -1

        # Extract outcome
        outcome = extract_routing_outcome_from_router(
            router, connections, routing_success
        )

        # Get conflict analysis
        conflict_analysis = None
        if hasattr(router, 'get_conflict_analysis'):
            conflict_analysis = router.get_conflict_analysis()

        # Store outcome
        outcome_id = self.data_store.store_outcome(outcome, conflict_analysis)

        # Store cell blocking info
        if conflict_analysis and not routing_success:
            blocking_info = {}
            for cell, owner in router.belt_owner.items():
                # Check if this cell blocked any failed connections
                blocked = []
                for failed in router.failed_connections:
                    if cell in failed.get('blocked_positions', []):
                        blocked.append(failed['index'])

                if blocked:
                    blocking_info[cell] = {
                        'placed_by': owner,
                        'blocked': blocked,
                    }

            if blocking_info:
                self.data_store.store_cell_blocking(outcome_id, blocking_info)

        # Also record to component-specific stores
        self.record_outcome(outcome)

        return outcome_id

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about stored routing data."""
        return self.data_store.get_stats()


# =============================================================================
# Routing Trainer - Direct routing training without cpsat_solver
# =============================================================================

class RoutingTrainer:
    """
    Trains routing ML by running routing directly on placement data.

    This bypasses cpsat_solver to get detailed routing data for ML training.
    Takes machine placements and connections, runs routing, records outcomes.
    """

    def __init__(
        self,
        ml_system: Optional[EnhancedRoutingMLSystem] = None,
        db_path: str = "routing_ml.db",
    ):
        """
        Initialize routing trainer.

        Args:
            ml_system: Optional ML system to use (creates one if not provided)
            db_path: Database path for training data
        """
        # Import router here to avoid circular imports
        from .router import BeltRouter, Connection

        self.BeltRouter = BeltRouter
        self.Connection = Connection

        if ml_system is None:
            self.ml_system = EnhancedRoutingMLSystem(db_path=db_path)
        else:
            self.ml_system = ml_system

        self.stats = {
            'total_trained': 0,
            'routing_success': 0,
            'routing_failed': 0,
        }

    def train_from_placement(
        self,
        machines: List[Tuple[Any, int, int, int, Any]],  # (type, x, y, floor, rotation)
        connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Any, Any]],
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        valid_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Train routing ML from a single placement.

        Args:
            machines: List of (building_type, x, y, floor, rotation)
            connections: List of (from_pos, to_pos, from_dir, to_dir)
            grid_width: Grid width
            grid_height: Grid height
            num_floors: Number of floors
            valid_cells: Valid cells for irregular foundations

        Returns:
            Training result with routing success and stats
        """
        # Create router with ML functions
        router = self.BeltRouter(
            grid_width, grid_height, num_floors,
            use_belt_ports=True, max_belt_ports=4,
            valid_cells=valid_cells,
            heuristic_fn=self.ml_system.heuristic,
            move_cost_fn=self.ml_system.move_cost,
        )

        # Set occupied positions from machines
        occupied = set()
        for building_type, x, y, floor, rotation in machines:
            occupied.add((x, y, floor))
        router.set_occupied(occupied)

        # Convert to Connection objects
        conn_objects = []
        for from_pos, to_pos, from_dir, to_dir in connections:
            conn_objects.append(self.Connection(
                from_pos=from_pos,
                to_pos=to_pos,
                from_direction=from_dir,
                to_direction=to_dir,
            ))

        # Route all connections using indexed routing (tracks ownership)
        results = []
        simple_connections = []

        for i, conn in enumerate(conn_objects):
            result = router.route_connection_indexed(conn, i)
            results.append(result)
            simple_connections.append((conn.from_pos, conn.to_pos))

        # Check overall success
        routing_success = all(r.success for r in results)

        # Record to ML system
        outcome_id = self.ml_system.record_full_outcome(
            router, simple_connections, routing_success
        )

        # Update stats
        self.stats['total_trained'] += 1
        if routing_success:
            self.stats['routing_success'] += 1
        else:
            self.stats['routing_failed'] += 1

        return {
            'routing_success': routing_success,
            'connections_routed': sum(1 for r in results if r.success),
            'connections_total': len(connections),
            'outcome_id': outcome_id,
            'conflict_analysis': router.get_conflict_analysis() if not routing_success else None,
        }

    def train_from_solution(
        self,
        solution,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        valid_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Train routing ML from a CPSATSolution.

        Extracts machines and reconstructs connections from the solution.

        Args:
            solution: CPSATSolution with machines and belts
            input_positions: List of input positions
            output_positions: List of output positions
            grid_width: Grid width
            grid_height: Grid height
            num_floors: Number of floors
            valid_cells: Valid cells for irregular foundations

        Returns:
            Training result
        """
        # We need to reconstruct connections from belt paths
        # For now, store the solution's success and belt positions
        # Full connection reconstruction requires more solution data

        from .router import Rotation

        # Build simple connections from inputs to outputs via machines
        # This is approximate - full reconstruction would need flow analysis

        machines = solution.machines
        routing_success = solution.routing_success

        # Store outcome with available data
        outcome = RoutingOutcome(
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            connections=[],  # Not available without reconstruction
            connection_order=[],
            paths=[],  # Belt positions are available but not as paths
            success=routing_success,
            failed_at_connection=None,
            cells_used={tuple(b[:3]) for b in solution.belts} if solution.belts else set(),
        )

        self.ml_system.data_store.store_outcome(outcome)

        self.stats['total_trained'] += 1
        if routing_success:
            self.stats['routing_success'] += 1
        else:
            self.stats['routing_failed'] += 1

        return {
            'routing_success': routing_success,
            'belts_count': len(solution.belts) if solution.belts else 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        total = self.stats['total_trained']
        return {
            **self.stats,
            'success_rate': self.stats['routing_success'] / max(total, 1),
            **self.ml_system.get_routing_stats(),
        }


# =============================================================================
# Integration with training_runner.py
# =============================================================================

def create_routing_trainer(db_path: str = "routing_ml.db") -> RoutingTrainer:
    """
    Create a routing trainer for use with training_runner.py.

    Usage in training_runner.py:
        from .ml_routing import create_routing_trainer

        routing_trainer = create_routing_trainer(db_path=self.db_path)

        # After solving:
        if solution:
            routing_trainer.train_from_solution(
                solution=solution,
                input_positions=input_positions,
                output_positions=output_positions,
                grid_width=spec.grid_width,
                grid_height=spec.grid_height,
                num_floors=spec.num_floors,
                valid_cells=valid_cells,
            )
    """
    return RoutingTrainer(db_path=db_path)
