#!/usr/bin/env python3
"""
Transformer-based Placement Model.

Uses attention mechanism to handle variable numbers of machines
and learn spatial relationships between them.

Architecture:
- Machine embeddings: type + position + rotation encoded as vectors
- Connection embeddings: source -> destination pairs
- Port embeddings: position + direction encoded
- Transformer encoder learns relationships between all elements
- Classification head predicts routing success probability
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Foundation type encoding
FOUNDATION_TYPES = [
    "1x1", "2x1", "3x1", "4x1",
    "1x2", "1x3", "1x4",
    "2x2", "3x2", "4x2", "2x3", "2x4", "3x3",
    "T", "L", "L4", "S4", "Cross",
]

# Machine type encoding (must match BuildingType enum values)
MACHINE_TYPES = [
    "CUTTER", "CUTTER_HALF", "STACKER", "SWAPPER", "ROTATOR",
    "ROTATOR_CCW", "ROTATOR_180",
    "PAINTER", "MIXER", "CRYSTAL_GENERATOR",
    "SPLITTER",  # Also used for splitting
]

# Connection types
CONN_INPUT_TO_MACHINE = 0   # Input port -> Machine
CONN_MACHINE_TO_MACHINE = 1  # Machine -> Machine
CONN_MACHINE_TO_OUTPUT = 2   # Machine -> Output port

# Element type tokens for unified sequence
TOKEN_FOUNDATION = 0
TOKEN_MACHINE = 1
TOKEN_INPUT_PORT = 2
TOKEN_OUTPUT_PORT = 3
TOKEN_CONNECTION = 4


@dataclass
class PlacementSample:
    """A single placement sample for training."""
    foundation_type: str
    grid_width: int
    grid_height: int
    num_floors: int

    # Machines: (type, x, y, floor, rotation, is_mirrored)
    # rotation: 0=EAST, 1=SOUTH, 2=WEST, 3=NORTH
    machines: List[Tuple[str, int, int, int, int, bool]]

    # Input/output port positions: (x, y, floor, side)
    # side: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
    input_ports: List[Tuple[int, int, int, int]]
    output_ports: List[Tuple[int, int, int, int]]

    # Connections: (conn_type, src_idx, dst_idx, src_pos, dst_pos)
    # conn_type: CONN_INPUT_TO_MACHINE, CONN_MACHINE_TO_MACHINE, CONN_MACHINE_TO_OUTPUT
    # src_idx/dst_idx: index into input_ports, machines, or output_ports depending on type
    # src_pos/dst_pos: (x, y, floor) actual positions for the connection endpoints
    connections: List[Tuple[int, int, int, Tuple[int, int, int], Tuple[int, int, int]]] = field(default_factory=list)

    # Outcome
    routing_success: bool = False

    # Optional metadata
    solve_time: float = 0.0
    num_belts: int = 0
    failure_reason: str = ""


if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for spatial coordinates."""

        def __init__(self, d_model: int, max_len: int = 100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, positions: torch.Tensor) -> torch.Tensor:
            """Encode integer positions to vectors."""
            # positions: (batch, seq_len) or (batch, seq_len, 3) for x,y,floor
            if positions.dim() == 3:
                # Encode each coordinate separately and sum
                x_enc = self.pe[positions[:, :, 0].clamp(0, self.pe.size(0)-1)]
                y_enc = self.pe[positions[:, :, 1].clamp(0, self.pe.size(0)-1)]
                f_enc = self.pe[positions[:, :, 2].clamp(0, self.pe.size(0)-1)]
                return x_enc + y_enc + f_enc
            return self.pe[positions.clamp(0, self.pe.size(0)-1)]


    class PlacementTransformer(nn.Module):
        """
        Transformer model for placement quality prediction.

        Takes a variable-length sequence of:
        - 1 foundation token (grid size, type)
        - N machine tokens (type, position, rotation, mirror)
        - M input port tokens (position, side)
        - K output port tokens (position, side)
        - C connection tokens (type, src_pos, dst_pos)

        Uses attention to learn spatial relationships and connection routing feasibility.
        """

        def __init__(
            self,
            d_model: int = 128,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.d_model = d_model

            # Token type embedding (5 types now including connections)
            self.token_type_embed = nn.Embedding(5, d_model)

            # Foundation type embedding
            self.foundation_embed = nn.Embedding(len(FOUNDATION_TYPES) + 1, d_model)

            # Machine type embedding
            self.machine_type_embed = nn.Embedding(len(MACHINE_TYPES) + 1, d_model)

            # Rotation embedding (4 directions: EAST, SOUTH, WEST, NORTH)
            self.rotation_embed = nn.Embedding(4, d_model)

            # Mirror embedding (True/False)
            self.mirror_embed = nn.Embedding(2, d_model)

            # Side embedding for ports (NORTH, EAST, SOUTH, WEST)
            self.side_embed = nn.Embedding(4, d_model)

            # Connection type embedding
            self.conn_type_embed = nn.Embedding(3, d_model)  # input->machine, machine->machine, machine->output

            # Spatial position encoding
            self.pos_encoding = PositionalEncoding(d_model)

            # Grid dimension projection (width, height, floors -> d_model)
            self.grid_proj = nn.Linear(3, d_model)

            # Connection position projection (encodes both src and dst positions)
            self.conn_pos_proj = nn.Linear(6, d_model)  # src(x,y,z) + dst(x,y,z)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, 1),
            )

        def forward(
            self,
            foundation_type: torch.Tensor,      # (batch,) - foundation type index
            grid_dims: torch.Tensor,            # (batch, 3) - width, height, floors
            machine_types: torch.Tensor,        # (batch, max_machines) - type indices
            machine_positions: torch.Tensor,    # (batch, max_machines, 3) - x, y, floor
            machine_rotations: torch.Tensor,    # (batch, max_machines) - rotation (0-3)
            machine_mirrors: torch.Tensor,      # (batch, max_machines) - is_mirrored (0/1)
            input_positions: torch.Tensor,      # (batch, max_inputs, 3) - x, y, floor
            input_sides: torch.Tensor,          # (batch, max_inputs) - side (0-3)
            output_positions: torch.Tensor,     # (batch, max_outputs, 3) - x, y, floor
            output_sides: torch.Tensor,         # (batch, max_outputs) - side (0-3)
            conn_types: torch.Tensor,           # (batch, max_conns) - connection type (0-2)
            conn_positions: torch.Tensor,       # (batch, max_conns, 6) - src(x,y,z) + dst(x,y,z)
            machine_mask: torch.Tensor,         # (batch, max_machines) - valid machines
            input_mask: torch.Tensor,           # (batch, max_inputs) - valid inputs
            output_mask: torch.Tensor,          # (batch, max_outputs) - valid outputs
            conn_mask: torch.Tensor,            # (batch, max_conns) - valid connections
        ) -> torch.Tensor:
            """
            Forward pass.

            Returns: (batch,) success probability
            """
            batch_size = foundation_type.size(0)
            device = foundation_type.device

            # Build sequence of embeddings
            embeddings = []
            masks = []

            # 1. Foundation token (always present)
            foundation_emb = (
                self.token_type_embed(torch.full((batch_size,), TOKEN_FOUNDATION, device=device))
                + self.foundation_embed(foundation_type)
                + self.grid_proj(grid_dims.float())
            )
            embeddings.append(foundation_emb.unsqueeze(1))
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

            # 2. Machine tokens (with rotation and mirror)
            num_machines = machine_types.size(1)
            machine_token_type = torch.full((batch_size, num_machines), TOKEN_MACHINE, device=device)
            machine_emb = (
                self.token_type_embed(machine_token_type)
                + self.machine_type_embed(machine_types)
                + self.pos_encoding(machine_positions)
                + self.rotation_embed(machine_rotations)
                + self.mirror_embed(machine_mirrors)
            )
            embeddings.append(machine_emb)
            masks.append(machine_mask)

            # 3. Input port tokens (with side)
            num_inputs = input_positions.size(1)
            input_token_type = torch.full((batch_size, num_inputs), TOKEN_INPUT_PORT, device=device)
            input_emb = (
                self.token_type_embed(input_token_type)
                + self.pos_encoding(input_positions)
                + self.side_embed(input_sides)
            )
            embeddings.append(input_emb)
            masks.append(input_mask)

            # 4. Output port tokens (with side)
            num_outputs = output_positions.size(1)
            output_token_type = torch.full((batch_size, num_outputs), TOKEN_OUTPUT_PORT, device=device)
            output_emb = (
                self.token_type_embed(output_token_type)
                + self.pos_encoding(output_positions)
                + self.side_embed(output_sides)
            )
            embeddings.append(output_emb)
            masks.append(output_mask)

            # 5. Connection tokens
            num_conns = conn_types.size(1)
            conn_token_type = torch.full((batch_size, num_conns), TOKEN_CONNECTION, device=device)
            conn_emb = (
                self.token_type_embed(conn_token_type)
                + self.conn_type_embed(conn_types)
                + self.conn_pos_proj(conn_positions.float())
            )
            embeddings.append(conn_emb)
            masks.append(conn_mask)

            # Concatenate all embeddings
            x = torch.cat(embeddings, dim=1)  # (batch, seq_len, d_model)
            mask = torch.cat(masks, dim=1)     # (batch, seq_len)

            # Create attention mask (True = ignore)
            attn_mask = ~mask

            # Transformer forward
            x = self.transformer(x, src_key_padding_mask=attn_mask)

            # Pool: take mean over valid tokens
            mask_expanded = mask.unsqueeze(-1).float()
            x_masked = x * mask_expanded
            x_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

            # Classify
            logits = self.classifier(x_pooled).squeeze(-1)
            return torch.sigmoid(logits)


    class PlacementDataset(Dataset):
        """Dataset for placement samples with rich connection data."""

        def __init__(
            self,
            samples: List[PlacementSample],
            max_machines: int = 10,
            max_inputs: int = 16,
            max_outputs: int = 16,
            max_connections: int = 32,
        ):
            self.samples = samples
            self.max_machines = max_machines
            self.max_inputs = max_inputs
            self.max_outputs = max_outputs
            self.max_connections = max_connections

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # Foundation
            try:
                foundation_idx = FOUNDATION_TYPES.index(sample.foundation_type)
            except ValueError:
                foundation_idx = 0

            grid_dims = np.array([
                sample.grid_width,
                sample.grid_height,
                sample.num_floors
            ], dtype=np.float32)

            # Machines: (type, x, y, floor, rotation, is_mirrored)
            machine_types = np.zeros(self.max_machines, dtype=np.int64)
            machine_positions = np.zeros((self.max_machines, 3), dtype=np.int64)
            machine_rotations = np.zeros(self.max_machines, dtype=np.int64)
            machine_mirrors = np.zeros(self.max_machines, dtype=np.int64)
            machine_mask = np.zeros(self.max_machines, dtype=np.bool_)

            for i, machine in enumerate(sample.machines[:self.max_machines]):
                # Handle both old format (type, x, y, floor) and new format (type, x, y, floor, rot, mirror)
                if len(machine) >= 6:
                    m_type, x, y, floor, rot, mirror = machine[:6]
                elif len(machine) >= 4:
                    m_type, x, y, floor = machine[:4]
                    rot, mirror = 0, False
                else:
                    continue

                try:
                    machine_types[i] = MACHINE_TYPES.index(str(m_type).replace('BuildingType.', ''))
                except ValueError:
                    machine_types[i] = 0
                machine_positions[i] = [x, y, floor]
                machine_rotations[i] = rot if isinstance(rot, int) else 0
                machine_mirrors[i] = 1 if mirror else 0
                machine_mask[i] = True

            # Input ports: (x, y, floor, side)
            input_positions = np.zeros((self.max_inputs, 3), dtype=np.int64)
            input_sides = np.zeros(self.max_inputs, dtype=np.int64)
            input_mask = np.zeros(self.max_inputs, dtype=np.bool_)

            for i, port in enumerate(sample.input_ports[:self.max_inputs]):
                if len(port) >= 4:
                    x, y, floor, side = port[:4]
                elif len(port) >= 3:
                    x, y, floor = port[:3]
                    side = 0
                else:
                    continue
                input_positions[i] = [x, y, floor]
                input_sides[i] = side if isinstance(side, int) else 0
                input_mask[i] = True

            # Output ports: (x, y, floor, side)
            output_positions = np.zeros((self.max_outputs, 3), dtype=np.int64)
            output_sides = np.zeros(self.max_outputs, dtype=np.int64)
            output_mask = np.zeros(self.max_outputs, dtype=np.bool_)

            for i, port in enumerate(sample.output_ports[:self.max_outputs]):
                if len(port) >= 4:
                    x, y, floor, side = port[:4]
                elif len(port) >= 3:
                    x, y, floor = port[:3]
                    side = 0
                else:
                    continue
                output_positions[i] = [x, y, floor]
                output_sides[i] = side if isinstance(side, int) else 0
                output_mask[i] = True

            # Connections: (conn_type, src_idx, dst_idx, src_pos, dst_pos)
            conn_types = np.zeros(self.max_connections, dtype=np.int64)
            conn_positions = np.zeros((self.max_connections, 6), dtype=np.float32)
            conn_mask = np.zeros(self.max_connections, dtype=np.bool_)

            for i, conn in enumerate(sample.connections[:self.max_connections]):
                if len(conn) >= 5:
                    conn_type, src_idx, dst_idx, src_pos, dst_pos = conn[:5]
                    conn_types[i] = conn_type
                    # Normalize positions by grid size
                    conn_positions[i] = [
                        src_pos[0] / max(1, sample.grid_width),
                        src_pos[1] / max(1, sample.grid_height),
                        src_pos[2] / max(1, sample.num_floors),
                        dst_pos[0] / max(1, sample.grid_width),
                        dst_pos[1] / max(1, sample.grid_height),
                        dst_pos[2] / max(1, sample.num_floors),
                    ]
                    conn_mask[i] = True

            return {
                'foundation_type': foundation_idx,
                'grid_dims': grid_dims,
                'machine_types': machine_types,
                'machine_positions': machine_positions,
                'machine_rotations': machine_rotations,
                'machine_mirrors': machine_mirrors,
                'machine_mask': machine_mask,
                'input_positions': input_positions,
                'input_sides': input_sides,
                'input_mask': input_mask,
                'output_positions': output_positions,
                'output_sides': output_sides,
                'output_mask': output_mask,
                'conn_types': conn_types,
                'conn_positions': conn_positions,
                'conn_mask': conn_mask,
                'label': float(sample.routing_success),
            }


class PlacementTransformerDB:
    """SQLite database for storing placement training data."""

    def __init__(self, db_path: str = "placement_transformer.db"):
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
                connections_json TEXT,
                routing_success INTEGER,
                solve_time REAL,
                num_belts INTEGER,
                failure_reason TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_foundation_type
            ON placements(foundation_type)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_routing_success
            ON placements(routing_success)
        ''')

        conn.commit()
        conn.close()

    def log_placement(self, sample: PlacementSample):
        """Log a placement sample to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO placements (
                timestamp, foundation_type, grid_width, grid_height, num_floors,
                machines_json, input_ports_json, output_ports_json, connections_json,
                routing_success, solve_time, num_belts, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            sample.foundation_type,
            sample.grid_width,
            sample.grid_height,
            sample.num_floors,
            json.dumps(sample.machines),
            json.dumps(sample.input_ports),
            json.dumps(sample.output_ports),
            json.dumps(sample.connections),
            1 if sample.routing_success else 0,
            sample.solve_time,
            sample.num_belts,
            sample.failure_reason,
        ))

        conn.commit()
        conn.close()

    def get_samples(self) -> List[PlacementSample]:
        """Load all samples from the database."""
        samples = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT foundation_type, grid_width, grid_height, num_floors,
                   machines_json, input_ports_json, output_ports_json, connections_json,
                   routing_success, solve_time, num_belts, failure_reason
            FROM placements
        """)

        for row in cursor.fetchall():
            try:
                # Parse machines - convert to tuples
                machines_raw = json.loads(row[4]) if row[4] else []
                machines = [tuple(m) for m in machines_raw]

                # Parse ports
                input_ports_raw = json.loads(row[5]) if row[5] else []
                input_ports = [tuple(p) for p in input_ports_raw]

                output_ports_raw = json.loads(row[6]) if row[6] else []
                output_ports = [tuple(p) for p in output_ports_raw]

                # Parse connections
                connections_raw = json.loads(row[7]) if row[7] else []
                connections = []
                for c in connections_raw:
                    if len(c) >= 5:
                        connections.append((c[0], c[1], c[2], tuple(c[3]), tuple(c[4])))

                sample = PlacementSample(
                    foundation_type=row[0],
                    grid_width=row[1],
                    grid_height=row[2],
                    num_floors=row[3],
                    machines=machines,
                    input_ports=input_ports,
                    output_ports=output_ports,
                    connections=connections,
                    routing_success=bool(row[8]),
                    solve_time=row[9] or 0.0,
                    num_belts=row[10] or 0,
                    failure_reason=row[11] or "",
                )
                samples.append(sample)
            except Exception as e:
                print(f"Error parsing sample: {e}")
                continue

        conn.close()
        return samples

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
            row[0]: {'total': row[1], 'successes': row[2] or 0}
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


class PlacementTransformerModel:
    """
    High-level interface for the transformer placement model.

    Supports:
    - Training from database
    - Online training after each solve attempt
    - Predicting success probability for a placement
    """

    def __init__(
        self,
        model_path: str = "models/placement_transformer.pt",
        db_path: str = "placement_transformer.db",
    ):
        self.model_path = model_path
        self.db = PlacementTransformerDB(db_path)
        self.model = None
        self.is_trained = False
        self.device = 'cpu'

        # Online training buffer
        self.online_buffer: List[PlacementSample] = []
        self.online_buffer_size = 32  # Retrain after this many samples

        # Model config (used for creating new models)
        self.config = {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 256,
            'dropout': 0.1,
        }

        if HAS_TORCH:
            if torch.cuda.is_available():
                self.device = 'cuda'
            self._load_model()

    def _load_model(self):
        """Load model from disk if exists."""
        if not HAS_TORCH:
            return

        if Path(self.model_path).exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.config = checkpoint.get('config', self.config)
                self.model = PlacementTransformer(**self.config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                self.is_trained = True
                print(f"Loaded transformer placement model from {self.model_path}")
            except Exception as e:
                print(f"Could not load transformer model: {e}")

    def _save_model(self):
        """Save model to disk."""
        if self.model is not None:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
            }, self.model_path)

    def log_sample(self, sample: PlacementSample, trigger_training: bool = True):
        """Log a placement sample and optionally trigger online training."""
        # Store in database
        self.db.log_placement(sample)

        # Add to online buffer
        self.online_buffer.append(sample)

        # Trigger training if buffer is full
        if trigger_training and len(self.online_buffer) >= self.online_buffer_size:
            self._online_train()

    def _online_train(self):
        """Perform online training with buffered samples."""
        if not HAS_TORCH or len(self.online_buffer) < 10:
            return

        # Check for both classes
        successes = sum(1 for s in self.online_buffer if s.routing_success)
        if successes == 0 or successes == len(self.online_buffer):
            # Need both classes
            return

        print(f"Online training with {len(self.online_buffer)} samples...")

        # Quick training on buffer
        self.train(
            samples=self.online_buffer,
            epochs=10,
            batch_size=min(16, len(self.online_buffer)),
            learning_rate=1e-4,  # Lower learning rate for fine-tuning
            val_split=0.0,  # No validation for online training
            verbose=False,
        )

        # Clear buffer
        self.online_buffer.clear()

    def _forward_batch(self, batch: dict) -> 'torch.Tensor':
        """Run forward pass on a batch."""
        foundation_type = batch['foundation_type'].to(self.device)
        grid_dims = batch['grid_dims'].to(self.device)
        machine_types = batch['machine_types'].to(self.device)
        machine_positions = batch['machine_positions'].to(self.device)
        machine_rotations = batch['machine_rotations'].to(self.device)
        machine_mirrors = batch['machine_mirrors'].to(self.device)
        machine_mask = batch['machine_mask'].to(self.device)
        input_positions = batch['input_positions'].to(self.device)
        input_sides = batch['input_sides'].to(self.device)
        input_mask = batch['input_mask'].to(self.device)
        output_positions = batch['output_positions'].to(self.device)
        output_sides = batch['output_sides'].to(self.device)
        output_mask = batch['output_mask'].to(self.device)
        conn_types = batch['conn_types'].to(self.device)
        conn_positions = batch['conn_positions'].to(self.device)
        conn_mask = batch['conn_mask'].to(self.device)

        return self.model(
            foundation_type, grid_dims,
            machine_types, machine_positions, machine_rotations, machine_mirrors,
            input_positions, input_sides,
            output_positions, output_sides,
            conn_types, conn_positions,
            machine_mask, input_mask, output_mask, conn_mask,
        )

    def train(
        self,
        samples: List[PlacementSample] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> bool:
        """Train model from samples (or from database if samples not provided)."""
        if not HAS_TORCH:
            print("PyTorch not available")
            return False

        # Load from database if no samples provided
        if samples is None:
            samples = self.db.get_samples()

        if len(samples) < 20:
            if verbose:
                print(f"Not enough samples (have {len(samples)}, need at least 20)")
            return False

        # Check for both classes
        successes = sum(1 for s in samples if s.routing_success)
        failures = len(samples) - successes
        if successes == 0 or failures == 0:
            if verbose:
                print(f"Need both success and failure samples (have {successes} successes, {failures} failures)")
            return False

        # Split data
        samples_copy = list(samples)
        np.random.shuffle(samples_copy)

        if val_split > 0:
            val_size = max(1, int(len(samples_copy) * val_split))
            train_samples = samples_copy[val_size:]
            val_samples = samples_copy[:val_size]
        else:
            train_samples = samples_copy
            val_samples = []

        train_dataset = PlacementDataset(train_samples)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_samples:
            val_dataset = PlacementDataset(val_samples)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None

        # Create or reuse model
        if self.model is None:
            self.model = PlacementTransformer(**self.config)
            self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        best_val_acc = 0.0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                optimizer.zero_grad()

                predictions = self._forward_batch(batch)
                labels = batch['label'].to(self.device)

                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += ((predictions > 0.5) == labels).sum().item()
                train_total += labels.size(0)

            train_acc = train_correct / max(1, train_total)

            # Validation
            val_acc = 0.0
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        predictions = self._forward_batch(batch)
                        labels = batch['label'].to(self.device)
                        val_correct += ((predictions > 0.5) == labels).sum().item()
                        val_total += labels.size(0)

                val_acc = val_correct / max(1, val_total)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"loss={train_loss/len(train_loader):.4f}, "
                      f"train_acc={train_acc:.1%}" +
                      (f", val_acc={val_acc:.1%}" if val_loader else ""))

            # Save best model
            if val_acc > best_val_acc or (val_loader is None and epoch == epochs - 1):
                best_val_acc = val_acc
                self._save_model()

        self.is_trained = True
        if verbose:
            print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1%}")

        return True

    def predict(self, sample: PlacementSample) -> float:
        """Predict success probability for a placement."""
        if not HAS_TORCH or not self.is_trained or self.model is None:
            return 0.5  # Default when not trained

        # Create single-sample batch
        dataset = PlacementDataset([sample])
        batch = dataset[0]

        # Convert to tensors with batch dimension
        batch_tensors = {
            k: torch.tensor([v], device=self.device) if not isinstance(v, np.ndarray)
            else torch.tensor(np.array([v]), device=self.device)
            for k, v in batch.items() if k != 'label'
        }

        self.model.eval()
        with torch.no_grad():
            prob = self._forward_batch(batch_tensors)
            return prob.item()

    def predict_from_raw(
        self,
        foundation_type: str,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        machines: List[Tuple],
        input_ports: List[Tuple],
        output_ports: List[Tuple],
        connections: List[Tuple] = None,
    ) -> float:
        """Predict directly from raw values."""
        sample = PlacementSample(
            foundation_type=foundation_type,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machines=machines,
            input_ports=input_ports,
            output_ports=output_ports,
            connections=connections or [],
            routing_success=False,
        )
        return self.predict(sample)

    def get_stats(self) -> Dict[str, Any]:
        """Get training data statistics."""
        stats = self.db.get_stats()
        stats['model_trained'] = self.is_trained
        stats['online_buffer_size'] = len(self.online_buffer)
        return stats


def load_samples_from_db(db_path: str = "placement_transformer.db") -> List[PlacementSample]:
    """Load training samples from the placement transformer database."""
    db = PlacementTransformerDB(db_path)
    return db.get_samples()


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transformer Placement Model")
    parser.add_argument("--train", action="store_true", help="Train model from database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--db", type=str, default="placement_transformer.db", help="Database path")
    parser.add_argument("--model", type=str, default="models/placement_transformer.pt", help="Model path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    model = PlacementTransformerModel(model_path=args.model, db_path=args.db)

    if args.stats:
        stats = model.get_stats()
        print("\nPlacement Transformer Statistics")
        print("=" * 40)
        print(f"Total samples: {stats['total']}")
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
        print(f"\nLoading samples from {args.db}...")
        samples = load_samples_from_db(args.db)
        print(f"Loaded {len(samples)} samples")

        if samples:
            successes = sum(1 for s in samples if s.routing_success)
            print(f"Success rate: {successes}/{len(samples)} ({100*successes/len(samples):.1f}%)")

            print("\nTraining transformer model...")
            success = model.train(samples, epochs=args.epochs, batch_size=args.batch_size)

            if success:
                print("Training complete!")
            else:
                print("Training failed")
        else:
            print("No samples found in database")
