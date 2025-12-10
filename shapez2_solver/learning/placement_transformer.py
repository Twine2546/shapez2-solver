#!/usr/bin/env python3
"""
Transformer-based Placement Model.

Uses attention mechanism to handle variable numbers of machines
and learn spatial relationships between them.

Architecture:
- Machine embeddings: type + position encoded as vectors
- Port embeddings: position + direction encoded
- Transformer encoder learns relationships between all elements
- Classification head predicts routing success
"""

import json
import sqlite3
from dataclasses import dataclass
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

# Machine type encoding
MACHINE_TYPES = [
    "CUTTER", "STACKER", "SWAPPER", "ROTATOR",
    "PAINTER", "MIXER", "CRYSTAL_GENERATOR",
]

# Element type tokens for unified sequence
TOKEN_FOUNDATION = 0
TOKEN_MACHINE = 1
TOKEN_INPUT_PORT = 2
TOKEN_OUTPUT_PORT = 3


@dataclass
class PlacementSample:
    """A single placement sample for training."""
    foundation_type: str
    grid_width: int
    grid_height: int
    num_floors: int
    machines: List[Tuple[str, int, int, int]]  # (type, x, y, floor)
    input_ports: List[Tuple[int, int, int]]    # (x, y, floor)
    output_ports: List[Tuple[int, int, int]]   # (x, y, floor)
    routing_success: bool


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
        - N machine tokens (type, position)
        - M input port tokens (position)
        - K output port tokens (position)

        Uses attention to learn spatial relationships.
        """

        def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 3,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.d_model = d_model

            # Token type embedding
            self.token_type_embed = nn.Embedding(4, d_model)  # foundation, machine, input, output

            # Foundation type embedding
            self.foundation_embed = nn.Embedding(len(FOUNDATION_TYPES) + 1, d_model)

            # Machine type embedding
            self.machine_type_embed = nn.Embedding(len(MACHINE_TYPES) + 1, d_model)

            # Spatial position encoding
            self.pos_encoding = PositionalEncoding(d_model)

            # Grid dimension projection (width, height, floors -> d_model)
            self.grid_proj = nn.Linear(3, d_model)

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
                nn.Linear(dim_feedforward, 1),
            )

        def forward(
            self,
            foundation_type: torch.Tensor,      # (batch,) - foundation type index
            grid_dims: torch.Tensor,            # (batch, 3) - width, height, floors
            machine_types: torch.Tensor,        # (batch, max_machines) - type indices
            machine_positions: torch.Tensor,    # (batch, max_machines, 3) - x, y, floor
            input_positions: torch.Tensor,      # (batch, max_inputs, 3) - x, y, floor
            output_positions: torch.Tensor,     # (batch, max_outputs, 3) - x, y, floor
            machine_mask: torch.Tensor,         # (batch, max_machines) - valid machines
            input_mask: torch.Tensor,           # (batch, max_inputs) - valid inputs
            output_mask: torch.Tensor,          # (batch, max_outputs) - valid outputs
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

            # 2. Machine tokens
            num_machines = machine_types.size(1)
            machine_token_type = torch.full((batch_size, num_machines), TOKEN_MACHINE, device=device)
            machine_emb = (
                self.token_type_embed(machine_token_type)
                + self.machine_type_embed(machine_types)
                + self.pos_encoding(machine_positions)
            )
            embeddings.append(machine_emb)
            masks.append(machine_mask)

            # 3. Input port tokens
            num_inputs = input_positions.size(1)
            input_token_type = torch.full((batch_size, num_inputs), TOKEN_INPUT_PORT, device=device)
            input_emb = (
                self.token_type_embed(input_token_type)
                + self.pos_encoding(input_positions)
            )
            embeddings.append(input_emb)
            masks.append(input_mask)

            # 4. Output port tokens
            num_outputs = output_positions.size(1)
            output_token_type = torch.full((batch_size, num_outputs), TOKEN_OUTPUT_PORT, device=device)
            output_emb = (
                self.token_type_embed(output_token_type)
                + self.pos_encoding(output_positions)
            )
            embeddings.append(output_emb)
            masks.append(output_mask)

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
        """Dataset for placement samples."""

        def __init__(
            self,
            samples: List[PlacementSample],
            max_machines: int = 10,
            max_inputs: int = 16,
            max_outputs: int = 16,
        ):
            self.samples = samples
            self.max_machines = max_machines
            self.max_inputs = max_inputs
            self.max_outputs = max_outputs

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

            # Machines
            machine_types = np.zeros(self.max_machines, dtype=np.int64)
            machine_positions = np.zeros((self.max_machines, 3), dtype=np.int64)
            machine_mask = np.zeros(self.max_machines, dtype=np.bool_)

            for i, (m_type, x, y, floor) in enumerate(sample.machines[:self.max_machines]):
                try:
                    machine_types[i] = MACHINE_TYPES.index(m_type)
                except ValueError:
                    machine_types[i] = 0
                machine_positions[i] = [x, y, floor]
                machine_mask[i] = True

            # Input ports
            input_positions = np.zeros((self.max_inputs, 3), dtype=np.int64)
            input_mask = np.zeros(self.max_inputs, dtype=np.bool_)

            for i, (x, y, floor) in enumerate(sample.input_ports[:self.max_inputs]):
                input_positions[i] = [x, y, floor]
                input_mask[i] = True

            # Output ports
            output_positions = np.zeros((self.max_outputs, 3), dtype=np.int64)
            output_mask = np.zeros(self.max_outputs, dtype=np.bool_)

            for i, (x, y, floor) in enumerate(sample.output_ports[:self.max_outputs]):
                output_positions[i] = [x, y, floor]
                output_mask[i] = True

            return {
                'foundation_type': foundation_idx,
                'grid_dims': grid_dims,
                'machine_types': machine_types,
                'machine_positions': machine_positions,
                'machine_mask': machine_mask,
                'input_positions': input_positions,
                'input_mask': input_mask,
                'output_positions': output_positions,
                'output_mask': output_mask,
                'label': float(sample.routing_success),
            }


class PlacementTransformerModel:
    """
    High-level interface for the transformer placement model.
    """

    def __init__(self, model_path: str = "models/placement_transformer.pt"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.device = 'cpu'

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
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = PlacementTransformer(**checkpoint.get('config', {}))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                self.is_trained = True
                print(f"Loaded transformer placement model from {self.model_path}")
            except Exception as e:
                print(f"Could not load transformer model: {e}")

    def _save_model(self, config: dict):
        """Save model to disk."""
        if self.model is not None:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': config,
            }, self.model_path)

    def train(
        self,
        samples: List[PlacementSample],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> bool:
        """Train model from samples."""
        if not HAS_TORCH:
            print("PyTorch not available")
            return False

        if len(samples) < 20:
            print(f"Not enough samples (have {len(samples)}, need at least 20)")
            return False

        # Check for both classes
        successes = sum(1 for s in samples if s.routing_success)
        failures = len(samples) - successes
        if successes == 0 or failures == 0:
            print(f"Need both success and failure samples (have {successes} successes, {failures} failures)")
            return False

        # Split data
        np.random.shuffle(samples)
        val_size = int(len(samples) * val_split)
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]

        train_dataset = PlacementDataset(train_samples)
        val_dataset = PlacementDataset(val_samples)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Model config
        config = {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 3,
            'dim_feedforward': 128,
            'dropout': 0.1,
        }

        self.model = PlacementTransformer(**config)
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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

                # Move to device
                foundation_type = batch['foundation_type'].to(self.device)
                grid_dims = batch['grid_dims'].to(self.device)
                machine_types = batch['machine_types'].to(self.device)
                machine_positions = batch['machine_positions'].to(self.device)
                machine_mask = batch['machine_mask'].to(self.device)
                input_positions = batch['input_positions'].to(self.device)
                input_mask = batch['input_mask'].to(self.device)
                output_positions = batch['output_positions'].to(self.device)
                output_mask = batch['output_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward
                predictions = self.model(
                    foundation_type, grid_dims,
                    machine_types, machine_positions, machine_mask,
                    input_positions, input_mask,
                    output_positions, output_mask,
                )

                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += ((predictions > 0.5) == labels).sum().item()
                train_total += labels.size(0)

            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    foundation_type = batch['foundation_type'].to(self.device)
                    grid_dims = batch['grid_dims'].to(self.device)
                    machine_types = batch['machine_types'].to(self.device)
                    machine_positions = batch['machine_positions'].to(self.device)
                    machine_mask = batch['machine_mask'].to(self.device)
                    input_positions = batch['input_positions'].to(self.device)
                    input_mask = batch['input_mask'].to(self.device)
                    output_positions = batch['output_positions'].to(self.device)
                    output_mask = batch['output_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    predictions = self.model(
                        foundation_type, grid_dims,
                        machine_types, machine_positions, machine_mask,
                        input_positions, input_mask,
                        output_positions, output_mask,
                    )

                    val_correct += ((predictions > 0.5) == labels).sum().item()
                    val_total += labels.size(0)

            train_acc = train_correct / train_total
            val_acc = val_correct / max(1, val_total)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"loss={train_loss/len(train_loader):.4f}, "
                      f"train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model(config)

        self.is_trained = True
        if verbose:
            print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1%}")

        return True

    def predict(self, sample: PlacementSample) -> float:
        """Predict success probability for a placement."""
        if not self.is_trained or self.model is None:
            return 0.5  # Default when not trained

        # Create single-sample batch
        dataset = PlacementDataset([sample])
        batch = dataset[0]

        self.model.eval()
        with torch.no_grad():
            foundation_type = torch.tensor([batch['foundation_type']], device=self.device)
            grid_dims = torch.tensor([batch['grid_dims']], device=self.device)
            machine_types = torch.tensor([batch['machine_types']], device=self.device)
            machine_positions = torch.tensor([batch['machine_positions']], device=self.device)
            machine_mask = torch.tensor([batch['machine_mask']], device=self.device)
            input_positions = torch.tensor([batch['input_positions']], device=self.device)
            input_mask = torch.tensor([batch['input_mask']], device=self.device)
            output_positions = torch.tensor([batch['output_positions']], device=self.device)
            output_mask = torch.tensor([batch['output_mask']], device=self.device)

            prob = self.model(
                foundation_type, grid_dims,
                machine_types, machine_positions, machine_mask,
                input_positions, input_mask,
                output_positions, output_mask,
            )

            return prob.item()

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
        sample = PlacementSample(
            foundation_type=foundation_type,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machines=machines,
            input_ports=input_ports,
            output_ports=output_ports,
            routing_success=False,  # Dummy
        )
        return self.predict(sample)


def load_samples_from_db(db_path: str = "placement_feedback.db") -> List[PlacementSample]:
    """Load training samples from the placement feedback database."""
    samples = []

    if not Path(db_path).exists():
        return samples

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT foundation_type, grid_width, grid_height, num_floors,
                   machines_json, input_ports_json, output_ports_json, routing_success
            FROM placements
        """)

        for row in cursor.fetchall():
            try:
                sample = PlacementSample(
                    foundation_type=row[0],
                    grid_width=row[1],
                    grid_height=row[2],
                    num_floors=row[3],
                    machines=json.loads(row[4]) if row[4] else [],
                    input_ports=json.loads(row[5]) if row[5] else [],
                    output_ports=json.loads(row[6]) if row[6] else [],
                    routing_success=bool(row[7]),
                )
                samples.append(sample)
            except Exception as e:
                print(f"Error parsing sample: {e}")
                continue
    except sqlite3.OperationalError:
        # Table might not exist or have different schema
        pass

    conn.close()
    return samples


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transformer Placement Model")
    parser.add_argument("--train", action="store_true", help="Train model from database")
    parser.add_argument("--db", type=str, default="placement_feedback.db", help="Database path")
    parser.add_argument("--model", type=str, default="models/placement_transformer.pt", help="Model path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    model = PlacementTransformerModel(model_path=args.model)

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
