"""
Advanced ML Models for Global Placement and Routing Evaluation.

This module provides neural network models that assess placements and routing
globally rather than through hand-crafted features:

- CNNPlacementEvaluator: Treats grid as multi-channel image
- GNNRoutingPredictor: Graph neural network over machines/connections
- TransformerEvaluator: Self-attention over all grid elements

Requires: torch, torch_geometric
"""

import json
import sqlite3
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Local imports
from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS
from .evaluation import (
    PlacementEvaluator,
    SolutionEvaluator,
    PlacementInfo,
    SolutionInfo,
    RoutingInfo,
)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

# PyTorch Geometric imports
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch_geometric = None
    GCNConv = None
    GATConv = None
    PyGDataLoader = None


# =============================================================================
# Grid Encoding for CNN
# =============================================================================

# Building type to channel index mapping
BUILDING_CHANNELS = {
    # Cutters - channel 0
    BuildingType.CUTTER: 0,
    BuildingType.CUTTER_MIRRORED: 0,
    BuildingType.HALF_CUTTER: 0,
    # Rotators - channel 1
    BuildingType.ROTATOR_CW: 1,
    BuildingType.ROTATOR_CCW: 1,
    BuildingType.ROTATOR_180: 1,
    # Stackers - channel 2
    BuildingType.STACKER: 2,
    BuildingType.STACKER_BENT: 2,
    BuildingType.STACKER_BENT_MIRRORED: 2,
    BuildingType.UNSTACKER: 2,
    # Painter - channel 3
    BuildingType.PAINTER: 3,
    BuildingType.PAINTER_MIRRORED: 3,
    # Splitters/Mergers - channel 4
    BuildingType.SPLITTER: 4,
    BuildingType.SPLITTER_LEFT: 4,
    BuildingType.SPLITTER_RIGHT: 4,
    BuildingType.MERGER: 4,
    # Swapper/Trash - channel 5
    BuildingType.SWAPPER: 5,
    BuildingType.TRASH: 5,
    BuildingType.PIN_PUSHER: 5,
    # Belts - channel 6
    BuildingType.BELT_FORWARD: 6,
    BuildingType.BELT_LEFT: 6,
    BuildingType.BELT_RIGHT: 6,
    # Lifts - channel 7
    BuildingType.LIFT_UP: 7,
    BuildingType.LIFT_DOWN: 7,
    # Belt ports - channel 8
    BuildingType.BELT_PORT_SENDER: 8,
    BuildingType.BELT_PORT_RECEIVER: 8,
}

NUM_BUILDING_CHANNELS = 9  # Building type channels
NUM_ROTATION_CHANNELS = 4  # NORTH, SOUTH, EAST, WEST
NUM_IO_CHANNELS = 2  # Inputs, Outputs
NUM_TOTAL_CHANNELS = NUM_BUILDING_CHANNELS + NUM_ROTATION_CHANNELS + NUM_IO_CHANNELS  # 15


def encode_grid_for_cnn(
    machines: List[PlacementInfo],
    grid_width: int,
    grid_height: int,
    num_floors: int,
    input_positions: List[Tuple[int, int, int]],
    output_positions: List[Tuple[int, int, int]],
) -> np.ndarray:
    """
    Encode placement as multi-channel grid image.

    Returns:
        Array of shape (num_floors, num_channels, grid_height, grid_width)
    """
    grid = np.zeros((num_floors, NUM_TOTAL_CHANNELS, grid_height, grid_width), dtype=np.float32)

    # Encode machines
    for m in machines:
        if 0 <= m.x < grid_width and 0 <= m.y < grid_height and 0 <= m.floor < num_floors:
            # Building type channel
            channel = BUILDING_CHANNELS.get(m.building_type, 0)
            grid[m.floor, channel, m.y, m.x] = 1.0

            # Rotation channel
            rot_channel = NUM_BUILDING_CHANNELS + {
                Rotation.NORTH: 0,
                Rotation.SOUTH: 1,
                Rotation.EAST: 2,
                Rotation.WEST: 3,
            }.get(m.rotation, 0)
            grid[m.floor, rot_channel, m.y, m.x] = 1.0

    # Encode I/O positions
    input_channel = NUM_BUILDING_CHANNELS + NUM_ROTATION_CHANNELS
    output_channel = input_channel + 1

    for x, y, f in input_positions:
        if 0 <= x < grid_width and 0 <= y < grid_height and 0 <= f < num_floors:
            grid[f, input_channel, y, x] = 1.0

    for x, y, f in output_positions:
        if 0 <= x < grid_width and 0 <= y < grid_height and 0 <= f < num_floors:
            grid[f, output_channel, y, x] = 1.0

    return grid


# =============================================================================
# Graph Encoding for GNN
# =============================================================================

def encode_graph_for_gnn(
    machines: List[PlacementInfo],
    grid_width: int,
    grid_height: int,
    num_floors: int,
    input_positions: List[Tuple[int, int, int]],
    output_positions: List[Tuple[int, int, int]],
    connections: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Encode placement as graph for GNN.

    Nodes: machines + I/O ports
    Edges: spatial proximity + required connections

    Returns:
        Dict with 'x' (node features), 'edge_index', 'edge_attr'
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for GNN encoding")

    nodes = []
    node_positions = []

    # Add machine nodes
    for i, m in enumerate(machines):
        # Node features: [type_onehot(10), rotation_onehot(4), norm_x, norm_y, norm_floor]
        type_onehot = [0.0] * 10
        type_idx = min(m.building_type.value, 9) if hasattr(m.building_type, 'value') else 0
        type_onehot[type_idx] = 1.0

        rot_onehot = [0.0] * 4
        rot_idx = {Rotation.NORTH: 0, Rotation.SOUTH: 1, Rotation.EAST: 2, Rotation.WEST: 3}.get(m.rotation, 0)
        rot_onehot[rot_idx] = 1.0

        norm_x = m.x / max(grid_width - 1, 1)
        norm_y = m.y / max(grid_height - 1, 1)
        norm_floor = m.floor / max(num_floors - 1, 1)

        features = type_onehot + rot_onehot + [norm_x, norm_y, norm_floor, 0.0, 0.0]  # Last 2: is_input, is_output
        nodes.append(features)
        node_positions.append((m.x, m.y, m.floor))

    num_machine_nodes = len(nodes)

    # Add input port nodes
    for x, y, f in input_positions:
        features = [0.0] * 10 + [0.0] * 4 + [
            x / max(grid_width - 1, 1),
            y / max(grid_height - 1, 1),
            f / max(num_floors - 1, 1),
            1.0,  # is_input
            0.0,  # is_output
        ]
        nodes.append(features)
        node_positions.append((x, y, f))

    # Add output port nodes
    for x, y, f in output_positions:
        features = [0.0] * 10 + [0.0] * 4 + [
            x / max(grid_width - 1, 1),
            y / max(grid_height - 1, 1),
            f / max(num_floors - 1, 1),
            0.0,  # is_input
            1.0,  # is_output
        ]
        nodes.append(features)
        node_positions.append((x, y, f))

    # Build edges based on spatial proximity
    edges = []
    edge_attrs = []

    for i, (x1, y1, f1) in enumerate(node_positions):
        for j, (x2, y2, f2) in enumerate(node_positions):
            if i >= j:
                continue

            # Manhattan distance
            dist = abs(x1 - x2) + abs(y1 - y2) + abs(f1 - f2) * 2

            # Connect nodes within threshold
            if dist <= 5:  # Adjustable threshold
                edges.append([i, j])
                edges.append([j, i])  # Bidirectional
                edge_attrs.append([dist / 10.0, 1.0 if f1 == f2 else 0.0])  # Normalized dist, same_floor
                edge_attrs.append([dist / 10.0, 1.0 if f1 == f2 else 0.0])

    # Add explicit connection edges if provided
    if connections:
        for src, dst in connections:
            if src < len(node_positions) and dst < len(node_positions):
                edges.append([src, dst])
                edge_attrs.append([0.0, 0.0])  # Special edge for required connection

    # Handle empty graph
    if not nodes:
        nodes = [[0.0] * 19]
    if not edges:
        edges = [[0, 0]]
        edge_attrs = [[0.0, 0.0]]

    return {
        'x': torch.tensor(nodes, dtype=torch.float32),
        'edge_index': torch.tensor(edges, dtype=torch.long).t().contiguous(),
        'edge_attr': torch.tensor(edge_attrs, dtype=torch.float32),
        'num_machines': num_machine_nodes,
    }


# =============================================================================
# Sequence Encoding for Transformer
# =============================================================================

def encode_sequence_for_transformer(
    machines: List[PlacementInfo],
    grid_width: int,
    grid_height: int,
    num_floors: int,
    input_positions: List[Tuple[int, int, int]],
    output_positions: List[Tuple[int, int, int]],
    max_seq_len: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode placement as sequence for Transformer.

    Each element: [type_onehot, rotation_onehot, position]

    Returns:
        (sequence, mask) - sequence of shape (max_seq_len, feature_dim), mask of shape (max_seq_len,)
    """
    feature_dim = 10 + 4 + 3 + 2  # type + rotation + position + io_flags = 19
    sequence = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
    mask = np.zeros(max_seq_len, dtype=np.float32)

    idx = 0

    # Encode machines
    for m in machines:
        if idx >= max_seq_len:
            break

        # Type one-hot
        type_idx = min(m.building_type.value, 9) if hasattr(m.building_type, 'value') else 0
        sequence[idx, type_idx] = 1.0

        # Rotation one-hot
        rot_idx = {Rotation.NORTH: 0, Rotation.SOUTH: 1, Rotation.EAST: 2, Rotation.WEST: 3}.get(m.rotation, 0)
        sequence[idx, 10 + rot_idx] = 1.0

        # Position
        sequence[idx, 14] = m.x / max(grid_width - 1, 1)
        sequence[idx, 15] = m.y / max(grid_height - 1, 1)
        sequence[idx, 16] = m.floor / max(num_floors - 1, 1)

        mask[idx] = 1.0
        idx += 1

    # Encode inputs
    for x, y, f in input_positions:
        if idx >= max_seq_len:
            break
        sequence[idx, 14] = x / max(grid_width - 1, 1)
        sequence[idx, 15] = y / max(grid_height - 1, 1)
        sequence[idx, 16] = f / max(num_floors - 1, 1)
        sequence[idx, 17] = 1.0  # is_input
        mask[idx] = 1.0
        idx += 1

    # Encode outputs
    for x, y, f in output_positions:
        if idx >= max_seq_len:
            break
        sequence[idx, 14] = x / max(grid_width - 1, 1)
        sequence[idx, 15] = y / max(grid_height - 1, 1)
        sequence[idx, 16] = f / max(num_floors - 1, 1)
        sequence[idx, 18] = 1.0  # is_output
        mask[idx] = 1.0
        idx += 1

    return sequence, mask


# =============================================================================
# CNN Model
# =============================================================================

class CNNModel(nn.Module):
    """CNN for evaluating grid-based placements."""

    def __init__(
        self,
        in_channels: int = NUM_TOTAL_CHANNELS,
        num_floors: int = 4,
        hidden_dim: int = 64,
        num_classes: int = 1,  # 1 for regression, 2 for classification
    ):
        super().__init__()

        self.num_floors = num_floors

        # Process each floor with shared conv layers
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim * 4)

        # Cross-floor attention
        self.floor_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 4,
            num_heads=4,
            batch_first=True,
        )

        # Final prediction head
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, num_floors, channels, height, width)

        Returns:
            Tensor of shape (batch, num_classes)
        """
        batch_size = x.shape[0]
        num_floors = x.shape[1]

        # Process each floor
        floor_features = []
        for f in range(num_floors):
            h = x[:, f]  # (batch, channels, height, width)

            h = F.relu(self.bn1(self.conv1(h)))
            h = F.max_pool2d(h, 2)

            h = F.relu(self.bn2(self.conv2(h)))
            h = F.max_pool2d(h, 2)

            h = F.relu(self.bn3(self.conv3(h)))

            # Global average pooling
            h = F.adaptive_avg_pool2d(h, (1, 1))
            h = h.view(batch_size, -1)  # (batch, hidden_dim * 4)

            floor_features.append(h)

        # Stack floor features: (batch, num_floors, hidden_dim * 4)
        floor_features = torch.stack(floor_features, dim=1)

        # Cross-floor attention
        attended, _ = self.floor_attention(floor_features, floor_features, floor_features)

        # Mean pool across floors
        pooled = attended.mean(dim=1)  # (batch, hidden_dim * 4)

        # Final prediction
        h = F.relu(self.fc1(self.dropout(pooled)))
        out = self.fc2(self.dropout(h))

        return out


# =============================================================================
# GNN Model
# =============================================================================

class GNNModel(nn.Module):
    """Graph Neural Network for evaluating placements as graphs."""

    def __init__(
        self,
        in_features: int = 19,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 1,
        use_gat: bool = True,
    ):
        super().__init__()

        self.use_gat = use_gat

        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if use_gat and HAS_PYG:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif HAS_PYG:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                # Fallback to simple linear
                self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge connectivity (2, num_edges)
            batch: Batch assignment for each node (num_nodes,)

        Returns:
            Graph-level predictions (batch_size, num_classes)
        """
        # Input projection
        h = F.relu(self.input_proj(x))

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            if HAS_PYG and isinstance(conv, (GCNConv, GATConv)):
                h = conv(h, edge_index)
            else:
                h = conv(h)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

        # Global pooling
        if batch is not None and HAS_PYG:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)

        # Output
        h = F.relu(self.fc1(h))
        out = self.fc2(self.dropout(h))

        return out


# =============================================================================
# Transformer Model
# =============================================================================

class TransformerModel(nn.Module):
    """Transformer for evaluating placements as sequences."""

    def __init__(
        self,
        in_features: int = 19,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 128,
        num_classes: int = 1,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence features (batch, seq_len, in_features)
            mask: Attention mask (batch, seq_len), 1 for valid, 0 for padding

        Returns:
            Predictions (batch, num_classes)
        """
        # Input projection
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        h = h + self.pos_encoding[:, :h.shape[1], :]

        # Create attention mask (True = ignore)
        if mask is not None:
            attn_mask = (mask == 0)
        else:
            attn_mask = None

        # Transformer encoding
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        # Mean pooling over valid positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        # Output
        h = F.relu(self.fc1(h))
        out = self.fc2(self.dropout(h))

        return out


# =============================================================================
# Unified Evaluator Classes
# =============================================================================

class CNNPlacementEvaluator(PlacementEvaluator):
    """CNN-based placement evaluator."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 64,
        collect_training_data: bool = True,
        db_path: str = "advanced_ml_training.db",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for CNNPlacementEvaluator")

        self.model_path = Path(model_path) if model_path else None
        self.db_path = db_path
        self.collect_training_data = collect_training_data
        self.hidden_dim = hidden_dim

        self.model = CNNModel(hidden_dim=hidden_dim, num_classes=1)
        self.model.eval()

        if self.model_path and self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

        self._init_db()
        self._training_data = []

    def _init_db(self):
        """Initialize training database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cnn_training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grid_data BLOB NOT NULL,
                label REAL NOT NULL,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def evaluate(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
    ) -> Tuple[float, bool]:
        """Evaluate placement using CNN."""
        # Encode grid
        grid = encode_grid_for_cnn(
            machines, grid_width, grid_height, num_floors,
            input_positions, output_positions
        )

        # Convert to tensor
        x = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # Add batch dim

        # Predict
        with torch.no_grad():
            score = torch.sigmoid(self.model(x)).item()

        # Store for training
        if self.collect_training_data:
            self._training_data.append({
                'grid': grid,
                'grid_width': grid_width,
                'grid_height': grid_height,
                'num_floors': num_floors,
            })

        return score, score < 0.3  # Reject if low score

    def record_outcome(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
        routing_success: bool,
    ) -> None:
        """Record outcome for training."""
        if not self.collect_training_data:
            return

        grid = encode_grid_for_cnn(
            machines, grid_width, grid_height, num_floors,
            input_positions, output_positions
        )

        label = 1.0 if routing_success else 0.0

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO cnn_training_samples (grid_data, label, grid_width, grid_height, num_floors) VALUES (?, ?, ?, ?, ?)",
            (grid.tobytes(), label, grid_width, grid_height, num_floors)
        )
        conn.commit()
        conn.close()

    def train(self, epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> Dict[str, Any]:
        """Train CNN model on collected data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT grid_data, label, grid_width, grid_height, num_floors FROM cnn_training_samples")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return {"trained": False, "reason": "Insufficient data", "samples": len(rows)}

        # Prepare data - group by grid dimensions
        max_width = max(r[2] for r in rows)
        max_height = max(r[3] for r in rows)
        max_floors = max(r[4] for r in rows)

        X = []
        y = []

        for grid_bytes, label, gw, gh, nf in rows:
            # Decode grid
            grid = np.frombuffer(grid_bytes, dtype=np.float32).reshape(nf, NUM_TOTAL_CHANNELS, gh, gw)

            # Pad to max size
            padded = np.zeros((max_floors, NUM_TOTAL_CHANNELS, max_height, max_width), dtype=np.float32)
            padded[:nf, :, :gh, :gw] = grid

            X.append(padded)
            y.append(label)

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        # Create new model with correct dimensions
        self.model = CNNModel(hidden_dim=self.hidden_dim, num_classes=1, num_floors=max_floors)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        losses = []
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(len(X))
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            epoch_loss = 0.0
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss)

        self.model.eval()

        # Save model
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)

        return {"trained": True, "samples": len(rows), "final_loss": losses[-1], "epochs": epochs}


class GNNPlacementEvaluator(PlacementEvaluator):
    """GNN-based placement evaluator."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 64,
        collect_training_data: bool = True,
        db_path: str = "advanced_ml_training.db",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GNNPlacementEvaluator")

        self.model_path = Path(model_path) if model_path else None
        self.db_path = db_path
        self.collect_training_data = collect_training_data
        self.hidden_dim = hidden_dim

        self.model = GNNModel(hidden_dim=hidden_dim, num_classes=1)
        self.model.eval()

        if self.model_path and self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

        self._init_db()

    def _init_db(self):
        """Initialize training database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gnn_training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_data TEXT NOT NULL,
                label REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def evaluate(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
    ) -> Tuple[float, bool]:
        """Evaluate placement using GNN."""
        graph_data = encode_graph_for_gnn(
            machines, grid_width, grid_height, num_floors,
            input_positions, output_positions
        )

        with torch.no_grad():
            score = torch.sigmoid(self.model(
                graph_data['x'],
                graph_data['edge_index'],
            )).item()

        return score, score < 0.3

    def record_outcome(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
        routing_success: bool,
    ) -> None:
        """Record outcome for training."""
        if not self.collect_training_data:
            return

        # Serialize placement for storage
        placement_data = {
            'machines': [(m.building_type.name, m.x, m.y, m.floor, m.rotation.name) for m in machines],
            'grid_width': grid_width,
            'grid_height': grid_height,
            'num_floors': num_floors,
            'inputs': input_positions,
            'outputs': output_positions,
        }

        label = 1.0 if routing_success else 0.0

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO gnn_training_samples (graph_data, label) VALUES (?, ?)",
            (json.dumps(placement_data), label)
        )
        conn.commit()
        conn.close()

    def train(self, epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> Dict[str, Any]:
        """Train GNN model on collected data."""
        if not HAS_PYG:
            return {"trained": False, "reason": "PyTorch Geometric required"}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT graph_data, label FROM gnn_training_samples")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return {"trained": False, "reason": "Insufficient data", "samples": len(rows)}

        # Prepare graph data
        graphs = []
        for graph_json, label in rows:
            try:
                placement = json.loads(graph_json)

                machines = [
                    PlacementInfo(
                        building_type=BuildingType[m[0]],
                        x=int(m[1]), y=int(m[2]), floor=int(m[3]),
                        rotation=Rotation[m[4]]
                    )
                    for m in placement['machines']
                ]

                # Ensure positions are integers
                input_positions = [(int(p[0]), int(p[1]), int(p[2])) for p in placement['inputs']]
                output_positions = [(int(p[0]), int(p[1]), int(p[2])) for p in placement['outputs']]

                graph_data = encode_graph_for_gnn(
                    machines,
                    int(placement['grid_width']),
                    int(placement['grid_height']),
                    int(placement['num_floors']),
                    input_positions,
                    output_positions,
                )

                data = Data(
                    x=graph_data['x'],
                    edge_index=graph_data['edge_index'],
                    y=torch.tensor([[label]], dtype=torch.float32),
                )
                graphs.append(data)
            except Exception as e:
                # Skip problematic samples
                continue

        if len(graphs) < 10:
            return {"trained": False, "reason": f"Insufficient valid graphs ({len(graphs)})", "samples": len(rows)}

        # Training
        self.model = GNNModel(hidden_dim=self.hidden_dim, num_classes=1)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Use PyTorch Geometric's DataLoader for graph data
        loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

        self.model.eval()

        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)

        return {"trained": True, "samples": len(rows), "final_loss": losses[-1], "epochs": epochs}


class TransformerPlacementEvaluator(PlacementEvaluator):
    """Transformer-based placement evaluator."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 64,
        max_seq_len: int = 128,
        collect_training_data: bool = True,
        db_path: str = "advanced_ml_training.db",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TransformerPlacementEvaluator")

        self.model_path = Path(model_path) if model_path else None
        self.db_path = db_path
        self.collect_training_data = collect_training_data
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.model = TransformerModel(hidden_dim=hidden_dim, max_seq_len=max_seq_len, num_classes=1)
        self.model.eval()

        if self.model_path and self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

        self._init_db()

    def _init_db(self):
        """Initialize training database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transformer_training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_data BLOB NOT NULL,
                mask_data BLOB NOT NULL,
                label REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def evaluate(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
    ) -> Tuple[float, bool]:
        """Evaluate placement using Transformer."""
        sequence, mask = encode_sequence_for_transformer(
            machines, grid_width, grid_height, num_floors,
            input_positions, output_positions,
            max_seq_len=self.max_seq_len,
        )

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score = torch.sigmoid(self.model(x, m)).item()

        return score, score < 0.3

    def record_outcome(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
        routing_success: bool,
    ) -> None:
        """Record outcome for training."""
        if not self.collect_training_data:
            return

        sequence, mask = encode_sequence_for_transformer(
            machines, grid_width, grid_height, num_floors,
            input_positions, output_positions,
            max_seq_len=self.max_seq_len,
        )

        label = 1.0 if routing_success else 0.0

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO transformer_training_samples (sequence_data, mask_data, label) VALUES (?, ?, ?)",
            (sequence.tobytes(), mask.tobytes(), label)
        )
        conn.commit()
        conn.close()

    def train(self, epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> Dict[str, Any]:
        """Train Transformer model on collected data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT sequence_data, mask_data, label FROM transformer_training_samples")
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return {"trained": False, "reason": "Insufficient data", "samples": len(rows)}

        X = []
        masks = []
        y = []

        for seq_bytes, mask_bytes, label in rows:
            seq = np.frombuffer(seq_bytes, dtype=np.float32).reshape(self.max_seq_len, 19)
            mask = np.frombuffer(mask_bytes, dtype=np.float32)
            X.append(seq)
            masks.append(mask)
            y.append(label)

        X = torch.tensor(np.array(X), dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        self.model = TransformerModel(hidden_dim=self.hidden_dim, max_seq_len=self.max_seq_len, num_classes=1)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        losses = []
        for epoch in range(epochs):
            perm = torch.randperm(len(X))
            X_shuffled = X[perm]
            masks_shuffled = masks[perm]
            y_shuffled = y[perm]

            epoch_loss = 0.0
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_masks = masks_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X, batch_masks)
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
# Model Comparison Framework
# =============================================================================

@dataclass
class ModelComparisonResult:
    """Result of comparing multiple models."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_inference_time_ms: float
    num_test_samples: int


def compare_models(
    models: Dict[str, PlacementEvaluator],
    test_data: List[Tuple[List[PlacementInfo], int, int, int, List, List, bool]],
) -> List[ModelComparisonResult]:
    """
    Compare multiple models on test data.

    Args:
        models: Dict mapping model name to evaluator instance
        test_data: List of (machines, grid_w, grid_h, num_floors, inputs, outputs, label)

    Returns:
        List of comparison results for each model
    """
    import time

    results = []

    for name, model in models.items():
        predictions = []
        labels = []
        times = []

        for machines, gw, gh, nf, inputs, outputs, label in test_data:
            start = time.perf_counter()
            score, _ = model.evaluate(machines, gw, gh, nf, inputs, outputs)
            elapsed = (time.perf_counter() - start) * 1000

            predictions.append(1 if score >= 0.5 else 0)
            labels.append(1 if label else 0)
            times.append(elapsed)

        # Compute metrics
        predictions = np.array(predictions)
        labels = np.array(labels)

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))

        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append(ModelComparisonResult(
            model_name=name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_inference_time_ms=np.mean(times),
            num_test_samples=len(test_data),
        ))

    return sorted(results, key=lambda r: r.f1_score, reverse=True)


class AdvancedMLSystem:
    """Unified system for managing advanced ML models."""

    def __init__(
        self,
        model_dir: str = "models/advanced",
        db_path: str = "advanced_ml_training.db",
        collect_training_data: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.db_path = db_path
        self.collect_training_data = collect_training_data

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators
        self.cnn_evaluator = CNNPlacementEvaluator(
            model_path=str(self.model_dir / "cnn_placement.pt"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

        self.gnn_evaluator = GNNPlacementEvaluator(
            model_path=str(self.model_dir / "gnn_placement.pt"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

        self.transformer_evaluator = TransformerPlacementEvaluator(
            model_path=str(self.model_dir / "transformer_placement.pt"),
            db_path=db_path,
            collect_training_data=collect_training_data,
        )

    def get_evaluators(self) -> Dict[str, PlacementEvaluator]:
        """Get all evaluators."""
        return {
            "cnn": self.cnn_evaluator,
            "gnn": self.gnn_evaluator,
            "transformer": self.transformer_evaluator,
        }

    def record_outcome(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
        routing_success: bool,
    ) -> None:
        """Record outcome to all evaluators."""
        for evaluator in [self.cnn_evaluator, self.gnn_evaluator, self.transformer_evaluator]:
            evaluator.record_outcome(
                machines, grid_width, grid_height, num_floors,
                input_positions, output_positions, routing_success
            )

    def train_all(self, epochs: int = 50) -> Dict[str, Any]:
        """Train all models."""
        results = {}

        print("Training CNN model...")
        results["cnn"] = self.cnn_evaluator.train(epochs=epochs)

        print("Training GNN model...")
        results["gnn"] = self.gnn_evaluator.train(epochs=epochs)

        print("Training Transformer model...")
        results["transformer"] = self.transformer_evaluator.train(epochs=epochs)

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get training data statistics."""
        conn = sqlite3.connect(self.db_path)

        stats = {}
        for table in ["cnn_training_samples", "gnn_training_samples", "transformer_training_samples"]:
            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table.replace("_training_samples", "")] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                stats[table.replace("_training_samples", "")] = 0

        conn.close()
        return stats
