#!/usr/bin/env python3
"""
Synthetic problem generator for ML training data.

Generates random but valid routing problems on different foundation types
and runs the CP-SAT solver to create positive/negative training examples.
"""

import random
import time
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional, Any
from pathlib import Path
import sys
import gc

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, FoundationSpec, Side
from shapez2_solver.blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS, BUILDING_PORTS


# Machine types we can place (including mirrored variants)
PLACEABLE_MACHINES = [
    BuildingType.CUTTER,
    BuildingType.CUTTER_MIRRORED,
    BuildingType.HALF_CUTTER,
    BuildingType.ROTATOR_CW,
    BuildingType.ROTATOR_CCW,
    BuildingType.ROTATOR_180,
    BuildingType.STACKER,
    BuildingType.STACKER_BENT,
    BuildingType.STACKER_BENT_MIRRORED,
    BuildingType.UNSTACKER,
    BuildingType.SWAPPER,
    BuildingType.SPLITTER,
    BuildingType.MERGER,
]

# Machine specs for sizing (width, height, floors)
MACHINE_SIZES = {
    BuildingType.CUTTER: (1, 2, 1),
    BuildingType.CUTTER_MIRRORED: (1, 2, 1),  # Same size as CUTTER
    BuildingType.HALF_CUTTER: (1, 1, 1),
    BuildingType.ROTATOR_CW: (1, 1, 1),
    BuildingType.ROTATOR_CCW: (1, 1, 1),
    BuildingType.ROTATOR_180: (1, 1, 1),
    BuildingType.STACKER: (1, 1, 2),
    BuildingType.STACKER_BENT: (1, 1, 2),
    BuildingType.STACKER_BENT_MIRRORED: (1, 1, 2),  # Same size as STACKER_BENT
    BuildingType.UNSTACKER: (1, 1, 2),
    BuildingType.SWAPPER: (2, 2, 1),
    BuildingType.SPLITTER: (1, 1, 1),
    BuildingType.MERGER: (1, 1, 1),
}


@dataclass
class PlacedMachine:
    """A machine placement in the problem."""
    machine_type: BuildingType
    x: int
    y: int
    floor: int
    rotation: Rotation

    def _get_rotated_size(self) -> Tuple[int, int, int]:
        """Get machine size accounting for rotation (width/height swap for 90° rotations)."""
        size = MACHINE_SIZES.get(self.machine_type, (1, 1, 1))
        w, h, d = size
        # For NORTH/SOUTH (90° rotations), swap width and height
        if self.rotation in (Rotation.NORTH, Rotation.SOUTH):
            return (h, w, d)
        return (w, h, d)

    def get_occupied_cells(self) -> Set[Tuple[int, int, int]]:
        """Get all cells occupied by this machine (rotation-aware)."""
        w, h, d = self._get_rotated_size()
        cells = set()
        for dx in range(w):
            for dy in range(h):
                for dz in range(d):
                    cells.add((self.x + dx, self.y + dy, self.floor + dz))
        return cells

    def _rotate_offset(self, dx: int, dy: int) -> Tuple[int, int]:
        """Rotate a relative offset based on machine rotation."""
        # EAST is default (0°), others rotate clockwise
        if self.rotation == Rotation.EAST:
            return (dx, dy)
        elif self.rotation == Rotation.SOUTH:
            return (-dy, dx)
        elif self.rotation == Rotation.WEST:
            return (-dx, -dy)
        elif self.rotation == Rotation.NORTH:
            return (dy, -dx)
        return (dx, dy)

    def get_input_positions(self) -> List[Tuple[int, int, int]]:
        """Get world positions of input ports (cells adjacent to machine for routing TO)."""
        ports = BUILDING_PORTS.get(self.machine_type, {
            'inputs': [(-1, 0, 0)],
            'outputs': [(1, 0, 0)],
        })
        positions = []
        for rel_x, rel_y, rel_z in ports.get('inputs', [(-1, 0, 0)]):
            rot_x, rot_y = self._rotate_offset(rel_x, rel_y)
            positions.append((self.x + rot_x, self.y + rot_y, self.floor + rel_z))
        return positions

    def get_output_positions(self) -> List[Tuple[int, int, int]]:
        """Get world positions of output ports (cells adjacent to machine for routing FROM)."""
        ports = BUILDING_PORTS.get(self.machine_type, {
            'inputs': [(-1, 0, 0)],
            'outputs': [(1, 0, 0)],
        })
        positions = []
        for rel_x, rel_y, rel_z in ports.get('outputs', [(1, 0, 0)]):
            rot_x, rot_y = self._rotate_offset(rel_x, rel_y)
            positions.append((self.x + rot_x, self.y + rot_y, self.floor + rel_z))
        return positions


@dataclass
class SyntheticProblem:
    """A synthetically generated routing problem."""
    problem_id: str
    foundation_type: str
    grid_width: int
    grid_height: int
    num_floors: int
    machines: List[PlacedMachine]
    input_positions: List[Tuple[int, int, int]]  # (x, y, floor)
    output_positions: List[Tuple[int, int, int]]
    connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    difficulty: str  # 'easy', 'medium', 'hard', 'extreme'

    def get_occupied(self) -> Set[Tuple[int, int, int]]:
        """Get all occupied cells."""
        occupied = set()
        for m in self.machines:
            occupied.update(m.get_occupied_cells())
        return occupied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'problem_id': self.problem_id,
            'foundation_type': self.foundation_type,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'num_floors': self.num_floors,
            'num_machines': len(self.machines),
            'num_connections': len(self.connections),
            'difficulty': self.difficulty,
            'machines': [
                {
                    'type': m.machine_type.name,
                    'x': m.x, 'y': m.y, 'floor': m.floor,
                    'rotation': m.rotation.name
                }
                for m in self.machines
            ],
            'input_positions': self.input_positions,
            'output_positions': self.output_positions,
            'connections': [
                {'src': list(c[0]), 'dst': list(c[1])}
                for c in self.connections
            ],
        }


@dataclass
class SolverResult:
    """Result from running the solver on a problem."""
    problem_id: str
    success: bool
    solve_time: float
    num_belts: int
    error_message: str = ""
    # Partial progress tracking for continuous learning
    connections_attempted: int = 0
    connections_routed: int = 0
    routing_progress: float = 0.0  # 0.0 to 1.0 (connections_routed / attempted)
    failed_connection_indices: str = ""  # JSON list of indices that failed
    partial_belt_positions: str = ""  # JSON list of successfully placed belts
    solver_iterations: int = 0  # How many iterations/nodes explored
    best_objective: float = 0.0  # Best objective value found (even if not optimal)

    # === Enhanced ML Training Data ===
    # Per-connection data (for both success and failure)
    connection_results: str = ""  # JSON list of per-connection results with features
    # Format: [{"index": 0, "success": true, "features": {...}, "path": [...], "belt_directions": [...],
    #          "nodes_explored": 10, "blocked_positions": [...]}]

    # Placement features (extracted before routing)
    placement_features: str = ""  # JSON dict of placement features for PlacementPredictor

    # Aggregated A* search statistics
    total_nodes_explored: int = 0
    total_blocked_positions: int = 0


class SyntheticProblemGenerator:
    """
    Generates synthetic routing problems with varying complexity.

    Difficulty levels:
    - easy: 1-2 machines, 2-3 connections, small foundations
    - medium: 3-5 machines, 4-6 connections, medium foundations
    - hard: 6-10 machines, 7-12 connections, larger foundations
    - extreme: 10+ machines, 12+ connections, multi-floor
    """

    DIFFICULTY_CONFIG = {
        'easy': {
            'min_machines': 1,
            'max_machines': 2,
            'min_connections': 2,
            'max_connections': 3,
            'foundations': ['1x1', '2x1', '1x2'],
            'max_floors': 1,
        },
        'medium': {
            'min_machines': 3,
            'max_machines': 5,
            'min_connections': 4,
            'max_connections': 6,
            'foundations': ['2x1', '1x2', '2x2', '3x1', 'L'],
            'max_floors': 2,
        },
        'hard': {
            'min_machines': 6,
            'max_machines': 10,
            'min_connections': 7,
            'max_connections': 12,
            'foundations': ['2x2', '3x2', '2x3', '3x3', 'T', 'L4', 'Cross'],
            'max_floors': 3,
        },
        'extreme': {
            'min_machines': 10,
            'max_machines': 20,
            'min_connections': 12,
            'max_connections': 20,
            'foundations': ['3x3', '4x2', '2x4', '3x2', 'Cross'],
            'max_floors': 4,
        },
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        self.problem_counter = 0

    def generate_problem(
        self,
        difficulty: str = 'medium',
        foundation_type: Optional[str] = None,
    ) -> SyntheticProblem:
        """
        Generate a single synthetic problem.

        Args:
            difficulty: 'easy', 'medium', 'hard', or 'extreme'
            foundation_type: Optional specific foundation type

        Returns:
            SyntheticProblem instance
        """
        config = self.DIFFICULTY_CONFIG[difficulty]

        # Select foundation
        if foundation_type is None:
            foundation_type = random.choice(config['foundations'])

        spec = FOUNDATION_SPECS[foundation_type]
        grid_w = spec.grid_width
        grid_h = spec.grid_height
        max_floors = min(config['max_floors'], spec.num_floors)

        # Generate unique problem ID
        self.problem_counter += 1
        problem_id = f"syn_{foundation_type}_{difficulty}_{self.problem_counter:05d}"

        # Place machines
        num_machines = random.randint(config['min_machines'], config['max_machines'])
        machines = self._place_machines(grid_w, grid_h, max_floors, num_machines)

        # Get occupied cells
        occupied = set()
        for m in machines:
            occupied.update(m.get_occupied_cells())

        # Generate input/output positions on edges
        num_connections = random.randint(config['min_connections'], config['max_connections'])
        inputs, outputs = self._generate_io_positions(
            grid_w, grid_h, max_floors, num_connections, occupied
        )

        # Create connections (pair inputs with outputs)
        connections = []
        for i in range(min(len(inputs), len(outputs))):
            connections.append((inputs[i], outputs[i]))

        return SyntheticProblem(
            problem_id=problem_id,
            foundation_type=foundation_type,
            grid_width=grid_w,
            grid_height=grid_h,
            num_floors=max_floors,
            machines=machines,
            input_positions=inputs,
            output_positions=outputs,
            connections=connections,
            difficulty=difficulty,
        )

    def _place_machines(
        self,
        grid_w: int,
        grid_h: int,
        num_floors: int,
        num_machines: int,
    ) -> List[PlacedMachine]:
        """Place machines randomly without overlaps."""
        machines = []
        occupied = set()

        # Leave margin on edges for I/O
        margin = 2

        attempts = 0
        while len(machines) < num_machines and attempts < 1000:
            attempts += 1

            # Pick random machine type
            machine_type = random.choice(PLACEABLE_MACHINES)
            size = MACHINE_SIZES.get(machine_type, (1, 1, 1))

            # Pick random position
            x = random.randint(margin, grid_w - margin - size[0])
            y = random.randint(margin, grid_h - margin - size[1])
            floor = random.randint(0, max(0, num_floors - size[2]))

            # Check if fits
            new_cells = set()
            fits = True
            for dx in range(size[0]):
                for dy in range(size[1]):
                    for dz in range(size[2]):
                        cell = (x + dx, y + dy, floor + dz)
                        if cell in occupied:
                            fits = False
                            break
                        new_cells.add(cell)
                    if not fits:
                        break
                if not fits:
                    break

            if fits:
                rotation = random.choice(list(Rotation))
                machines.append(PlacedMachine(
                    machine_type=machine_type,
                    x=x, y=y, floor=floor,
                    rotation=rotation,
                ))
                occupied.update(new_cells)

        return machines

    def _generate_io_positions(
        self,
        grid_w: int,
        grid_h: int,
        num_floors: int,
        num_connections: int,
        occupied: Set[Tuple[int, int, int]],
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """Generate input and output positions on grid edges."""
        inputs = []
        outputs = []

        # Inputs on west side, outputs on east side (typical flow)
        # But also allow some on north/south for variety

        used_positions = set()

        # Generate inputs (west and north sides)
        for _ in range(num_connections):
            attempts = 0
            while attempts < 100:
                attempts += 1
                floor = random.randint(0, num_floors - 1)

                if random.random() < 0.7:
                    # West side
                    x = 0
                    y = random.randint(2, grid_h - 3)
                else:
                    # North side
                    x = random.randint(2, grid_w - 3)
                    y = 0

                pos = (x, y, floor)
                if pos not in occupied and pos not in used_positions:
                    inputs.append(pos)
                    used_positions.add(pos)
                    break

        # Generate outputs (east and south sides)
        for _ in range(num_connections):
            attempts = 0
            while attempts < 100:
                attempts += 1
                floor = random.randint(0, num_floors - 1)

                if random.random() < 0.7:
                    # East side
                    x = grid_w - 1
                    y = random.randint(2, grid_h - 3)
                else:
                    # South side
                    x = random.randint(2, grid_w - 3)
                    y = grid_h - 1

                pos = (x, y, floor)
                if pos not in occupied and pos not in used_positions:
                    outputs.append(pos)
                    used_positions.add(pos)
                    break

        return inputs, outputs

    def generate_batch(
        self,
        count: int,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ) -> List[SyntheticProblem]:
        """
        Generate a batch of problems with given difficulty distribution.

        Args:
            count: Number of problems to generate
            difficulty_distribution: Dict mapping difficulty -> probability
                Default: {'easy': 0.3, 'medium': 0.4, 'hard': 0.2, 'extreme': 0.1}

        Returns:
            List of SyntheticProblem instances
        """
        if difficulty_distribution is None:
            difficulty_distribution = {
                'easy': 0.25,
                'medium': 0.35,
                'hard': 0.25,
                'extreme': 0.15,
            }

        problems = []
        difficulties = list(difficulty_distribution.keys())
        weights = list(difficulty_distribution.values())

        for _ in range(count):
            difficulty = random.choices(difficulties, weights=weights)[0]
            problems.append(self.generate_problem(difficulty))

        return problems


class SyntheticDataStore:
    """SQLite storage for synthetic problems and solver results."""

    def __init__(self, db_path: str = "synthetic_training.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synthetic_problems (
                problem_id TEXT PRIMARY KEY,
                foundation_type TEXT,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,
                num_machines INTEGER,
                num_connections INTEGER,
                difficulty TEXT,
                problem_json TEXT,
                created_at TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS solver_results (
                problem_id TEXT PRIMARY KEY,
                success INTEGER,
                solve_time REAL,
                num_belts INTEGER,
                error_message TEXT,
                solved_at TEXT,
                -- Partial progress tracking for continuous learning
                connections_attempted INTEGER DEFAULT 0,
                connections_routed INTEGER DEFAULT 0,
                routing_progress REAL DEFAULT 0.0,
                failed_connection_indices TEXT DEFAULT '',
                partial_belt_positions TEXT DEFAULT '',
                solver_iterations INTEGER DEFAULT 0,
                best_objective REAL DEFAULT 0.0,
                -- Enhanced ML training data
                connection_results TEXT DEFAULT '',
                placement_features TEXT DEFAULT '',
                total_nodes_explored INTEGER DEFAULT 0,
                total_blocked_positions INTEGER DEFAULT 0,
                FOREIGN KEY (problem_id) REFERENCES synthetic_problems(problem_id)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_difficulty
            ON synthetic_problems(difficulty)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_foundation
            ON synthetic_problems(foundation_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_success
            ON solver_results(success)
        ''')

        conn.commit()
        conn.close()

    def save_problem(self, problem: SyntheticProblem):
        """Save a problem to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO synthetic_problems
            (problem_id, foundation_type, grid_width, grid_height, num_floors,
             num_machines, num_connections, difficulty, problem_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            problem.problem_id,
            problem.foundation_type,
            problem.grid_width,
            problem.grid_height,
            problem.num_floors,
            len(problem.machines),
            len(problem.connections),
            problem.difficulty,
            json.dumps(problem.to_dict()),
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def save_result(self, result: SolverResult):
        """Save a solver result with partial progress tracking and ML training data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO solver_results
            (problem_id, success, solve_time, num_belts, error_message, solved_at,
             connections_attempted, connections_routed, routing_progress,
             failed_connection_indices, partial_belt_positions, solver_iterations, best_objective,
             connection_results, placement_features, total_nodes_explored, total_blocked_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.problem_id,
            1 if result.success else 0,
            result.solve_time,
            result.num_belts,
            result.error_message,
            datetime.now().isoformat(),
            result.connections_attempted,
            result.connections_routed,
            result.routing_progress,
            result.failed_connection_indices,
            result.partial_belt_positions,
            result.solver_iterations,
            result.best_objective,
            result.connection_results,
            result.placement_features,
            result.total_nodes_explored,
            result.total_blocked_positions,
        ))

        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Add new columns to existing database if needed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if new columns exist
        cursor.execute("PRAGMA table_info(solver_results)")
        columns = {row[1] for row in cursor.fetchall()}

        new_columns = [
            ('connections_attempted', 'INTEGER DEFAULT 0'),
            ('connections_routed', 'INTEGER DEFAULT 0'),
            ('routing_progress', 'REAL DEFAULT 0.0'),
            ('failed_connection_indices', 'TEXT DEFAULT ""'),
            ('partial_belt_positions', 'TEXT DEFAULT ""'),
            ('solver_iterations', 'INTEGER DEFAULT 0'),
            ('best_objective', 'REAL DEFAULT 0.0'),
            # Enhanced ML training data columns
            ('connection_results', 'TEXT DEFAULT ""'),
            ('placement_features', 'TEXT DEFAULT ""'),
            ('total_nodes_explored', 'INTEGER DEFAULT 0'),
            ('total_blocked_positions', 'INTEGER DEFAULT 0'),
        ]

        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    cursor.execute(f'ALTER TABLE solver_results ADD COLUMN {col_name} {col_type}')
                except sqlite3.OperationalError:
                    pass  # Column already exists

        conn.commit()
        conn.close()

    def save_result_old(self, result: SolverResult):
        """Old save method without partial progress (deprecated)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO solver_results
            (problem_id, success, solve_time, num_belts, error_message, solved_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result.problem_id,
            1 if result.success else 0,
            result.solve_time,
            result.num_belts,
            result.error_message,
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM synthetic_problems")
        stats['total_problems'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT difficulty, COUNT(*) FROM synthetic_problems
            GROUP BY difficulty
        """)
        stats['by_difficulty'] = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT foundation_type, COUNT(*) FROM synthetic_problems
            GROUP BY foundation_type
        """)
        stats['by_foundation'] = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT COUNT(*), SUM(success), AVG(solve_time)
            FROM solver_results
        """)
        row = cursor.fetchone()
        stats['total_solved'] = row[0] or 0
        stats['successes'] = row[1] or 0
        stats['avg_solve_time'] = row[2] or 0

        cursor.execute("""
            SELECT p.difficulty, COUNT(*), SUM(r.success), AVG(r.solve_time)
            FROM synthetic_problems p
            JOIN solver_results r ON p.problem_id = r.problem_id
            GROUP BY p.difficulty
        """)
        stats['results_by_difficulty'] = {
            row[0]: {
                'total': row[1],
                'successes': row[2] or 0,
                'avg_time': row[3] or 0,
            }
            for row in cursor.fetchall()
        }

        conn.close()
        return stats

    def export_training_data(self, output_path: str) -> int:
        """Export training data to JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.problem_json, r.success, r.solve_time, r.num_belts, r.error_message
            FROM synthetic_problems p
            JOIN solver_results r ON p.problem_id = r.problem_id
        """)

        data = []
        for row in cursor.fetchall():
            problem = json.loads(row[0])
            data.append({
                'problem': problem,
                'solver': {
                    'success': bool(row[1]),
                    'solve_time': row[2],
                    'num_belts': row[3],
                    'error_message': row[4],
                },
                'label': 'positive' if row[1] else 'negative',
            })

        conn.close()

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return len(data)


@dataclass
class MachinePort:
    """A port on a machine (input or output)."""
    position: Tuple[int, int, int]  # (x, y, floor)
    direction: Rotation  # Direction items flow
    port_type: str  # 'input' or 'output'
    machine_id: str  # ID of the machine this belongs to


@dataclass
class PlacedIOBlock:
    """A multi-port I/O block (e.g., 4-port input/output)."""
    machine_id: str
    block_type: str  # 'input_4port', 'output_4port', etc.
    base_x: int
    base_y: int
    floor: int
    side: Side  # Which foundation side
    ports: List[MachinePort]


class RealisticProblemGenerator:
    """
    Generates realistic routing problems that model actual Shapez 2 scenarios.

    Features:
    - Multi-port input/output machines on foundation edges
    - Processing machines in the interior
    - Machine-to-machine connections
    - Multi-floor routing with lifts
    - Jump/teleport usage when blocked

    Typical scenarios:
    - 4-port input → 4 processors → 4-port output
    - Input on one side, output on opposite/adjacent side
    - Cross-floor processing chains
    """

    # I/O block configurations
    IO_CONFIGS = {
        'input_4port': {
            'num_ports': 4,
            'width': 4,
            'height': 1,
            'is_input': True,
        },
        'output_4port': {
            'num_ports': 4,
            'width': 4,
            'height': 1,
            'is_input': False,
        },
        'input_2port': {
            'num_ports': 2,
            'width': 2,
            'height': 1,
            'is_input': True,
        },
        'output_2port': {
            'num_ports': 2,
            'width': 2,
            'height': 1,
            'is_input': False,
        },
    }

    # Scenarios define how problems are structured
    SCENARIOS = {
        'simple_passthrough': {
            'description': '4-port input to 4-port output on opposite sides',
            'min_floors': 1,
            'input_sides': [Side.SOUTH],
            'output_sides': [Side.NORTH],
            'machines_per_floor': 0,
            'num_io_ports': 4,  # Ports per I/O block
            'difficulty': 'easy',
        },
        'single_processor': {
            'description': 'Input → processor → output chain',
            'min_floors': 1,
            'input_sides': [Side.SOUTH],
            'output_sides': [Side.NORTH, Side.EAST],
            'machines_per_floor': 2,
            'num_io_ports': 4,
            'difficulty': 'medium',
        },
        'multi_processor': {
            'description': 'Input → multiple processors → output',
            'min_floors': 1,
            'input_sides': [Side.SOUTH, Side.WEST],
            'output_sides': [Side.NORTH, Side.EAST],
            'machines_per_floor': 5,  # More machines = more blocking
            'num_io_ports': 4,
            'difficulty': 'hard',
        },
        'cross_floor': {
            'description': 'Input on floor 0, processing on floor 1, output on floor 0/2',
            'min_floors': 2,
            'input_sides': [Side.SOUTH],
            'output_sides': [Side.NORTH, Side.EAST, Side.WEST],
            'machines_per_floor': 4,
            'num_io_ports': 4,
            'difficulty': 'hard',
        },
        'full_factory': {
            'description': '3-floor production with multiple I/O per floor',
            'min_floors': 3,
            'input_sides': [Side.SOUTH],
            'output_sides': [Side.NORTH, Side.EAST, Side.WEST],
            'machines_per_floor': 6,  # Pack in more machines
            'num_io_ports': 4,
            'difficulty': 'extreme',
        },
        'crowded': {
            'description': 'Densely packed machines with many crossing paths',
            'min_floors': 1,
            'input_sides': [Side.SOUTH, Side.WEST],
            'output_sides': [Side.NORTH, Side.EAST],
            'machines_per_floor': 8,  # Very crowded!
            'num_io_ports': 6,  # More I/O = more paths to route
            'difficulty': 'extreme',
        },
        'nightmare': {
            'description': 'Maximum density - designed to fail often',
            'min_floors': 2,
            'input_sides': [Side.SOUTH, Side.WEST, Side.NORTH],
            'output_sides': [Side.NORTH, Side.EAST, Side.SOUTH],
            'machines_per_floor': 12,  # Packed tight
            'num_io_ports': 8,  # Lots of paths
            'difficulty': 'extreme',
        },
    }

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.problem_counter = 0

    def generate_problem(
        self,
        foundation_type: str = 'T',
        scenario: str = 'simple_passthrough',
        num_floors: int = None,
    ) -> SyntheticProblem:
        """
        Generate a realistic problem with machine chains.

        Args:
            foundation_type: Foundation type (e.g., 'T', '3x2', 'Cross')
            scenario: Problem scenario from SCENARIOS
            num_floors: Number of floors (defaults based on scenario)

        Returns:
            SyntheticProblem with realistic machine connections
        """
        spec = FOUNDATION_SPECS.get(foundation_type, FOUNDATION_SPECS['3x2'])
        scenario_config = self.SCENARIOS.get(scenario, self.SCENARIOS['simple_passthrough'])

        if num_floors is None:
            num_floors = min(scenario_config['min_floors'], spec.num_floors)
        num_floors = max(1, min(num_floors, spec.num_floors))

        grid_w = spec.grid_width
        grid_h = spec.grid_height

        self.problem_counter += 1
        problem_id = f"realistic_{foundation_type}_{scenario}_{self.problem_counter:05d}"

        # Get valid grid cells for irregular foundations
        valid_cells = self._get_valid_cells(spec, num_floors)

        # Place I/O blocks on edges
        io_blocks, occupied = self._place_io_blocks(
            spec, num_floors, scenario_config, valid_cells
        )

        # Place processing machines
        machines, occupied = self._place_processing_machines(
            spec, num_floors, scenario_config, occupied, valid_cells
        )

        # Generate connections (machine chains)
        connections = self._generate_machine_chains(
            io_blocks, machines, num_floors
        )

        # Extract input/output positions
        inputs = []
        outputs = []
        for block in io_blocks:
            for port in block.ports:
                if port.port_type == 'input':
                    inputs.append(port.position)
                else:
                    outputs.append(port.position)

        return SyntheticProblem(
            problem_id=problem_id,
            foundation_type=foundation_type,
            grid_width=grid_w,
            grid_height=grid_h,
            num_floors=num_floors,
            machines=machines,
            input_positions=inputs,
            output_positions=outputs,
            connections=connections,
            difficulty=scenario_config['difficulty'],
        )

    def _get_valid_cells(
        self, spec: FoundationSpec, num_floors: int
    ) -> Set[Tuple[int, int, int]]:
        """Get all valid grid cells for this foundation."""
        valid = set()

        if spec.present_cells is None:
            # Rectangular foundation - all cells valid
            for x in range(spec.grid_width):
                for y in range(spec.grid_height):
                    for floor in range(num_floors):
                        valid.add((x, y, floor))
        else:
            # Irregular foundation - calculate valid regions
            for unit_x, unit_y in spec.present_cells:
                # Each unit is 14x14 (or 20x20 with overlap)
                base_x = unit_x * 20 if unit_x > 0 else 0
                base_y = unit_y * 20 if unit_y > 0 else 0
                width = 14 if unit_x == 0 else 20
                height = 14 if unit_y == 0 else 20

                # Adjust for first unit
                if unit_x > 0:
                    base_x = 14 + (unit_x - 1) * 20
                    width = 20
                if unit_y > 0:
                    base_y = 14 + (unit_y - 1) * 20
                    height = 20

                for x in range(base_x, min(base_x + width, spec.grid_width)):
                    for y in range(base_y, min(base_y + height, spec.grid_height)):
                        for floor in range(num_floors):
                            valid.add((x, y, floor))

        return valid

    def _place_io_blocks(
        self,
        spec: FoundationSpec,
        num_floors: int,
        scenario: Dict,
        valid_cells: Set[Tuple[int, int, int]],
    ) -> Tuple[List[PlacedIOBlock], Set[Tuple[int, int, int]]]:
        """Place input and output blocks on foundation edges."""
        blocks = []
        occupied = set()

        input_sides = scenario['input_sides']
        output_sides = scenario['output_sides']

        # Place inputs on specified sides for each floor
        for floor in range(num_floors):
            for side in input_sides:
                block = self._place_io_on_side(
                    spec, side, floor, 'input_4port', valid_cells, occupied
                )
                if block:
                    blocks.append(block)
                    for port in block.ports:
                        occupied.add(port.position)

        # Place outputs on specified sides for each floor
        for floor in range(num_floors):
            for side in output_sides:
                block = self._place_io_on_side(
                    spec, side, floor, 'output_4port', valid_cells, occupied
                )
                if block:
                    blocks.append(block)
                    for port in block.ports:
                        occupied.add(port.position)

        return blocks, occupied

    def _place_io_on_side(
        self,
        spec: FoundationSpec,
        side: Side,
        floor: int,
        block_type: str,
        valid_cells: Set[Tuple[int, int, int]],
        occupied: Set[Tuple[int, int, int]],
    ) -> Optional[PlacedIOBlock]:
        """Place an I/O block on a specific side."""
        config = self.IO_CONFIGS[block_type]
        is_input = config['is_input']
        num_ports = config['num_ports']

        # Calculate positions based on side
        ports = []
        machine_id = f"{block_type}_{side.value}_{floor}"

        if side == Side.SOUTH:
            # South side: y = grid_height - 1, x varies
            y = spec.grid_height - 1
            base_x = (spec.grid_width - num_ports) // 2
            direction = Rotation.NORTH  # Items flow north into the foundation

            for i in range(num_ports):
                x = base_x + i
                pos = (x, y, floor)
                if pos in occupied or pos not in valid_cells:
                    continue
                ports.append(MachinePort(
                    position=pos,
                    direction=direction,
                    port_type='input' if is_input else 'output',
                    machine_id=machine_id,
                ))

        elif side == Side.NORTH:
            y = 0
            base_x = (spec.grid_width - num_ports) // 2
            direction = Rotation.SOUTH  # Items flow south into the foundation

            for i in range(num_ports):
                x = base_x + i
                pos = (x, y, floor)
                if pos in occupied or pos not in valid_cells:
                    continue
                ports.append(MachinePort(
                    position=pos,
                    direction=direction,
                    port_type='input' if is_input else 'output',
                    machine_id=machine_id,
                ))

        elif side == Side.WEST:
            x = 0
            base_y = (spec.grid_height - num_ports) // 2
            direction = Rotation.EAST  # Items flow east into the foundation

            for i in range(num_ports):
                y = base_y + i
                pos = (x, y, floor)
                if pos in occupied or pos not in valid_cells:
                    continue
                ports.append(MachinePort(
                    position=pos,
                    direction=direction,
                    port_type='input' if is_input else 'output',
                    machine_id=machine_id,
                ))

        elif side == Side.EAST:
            x = spec.grid_width - 1
            base_y = (spec.grid_height - num_ports) // 2
            direction = Rotation.WEST  # Items flow west into the foundation

            for i in range(num_ports):
                y = base_y + i
                pos = (x, y, floor)
                if pos in occupied or pos not in valid_cells:
                    continue
                ports.append(MachinePort(
                    position=pos,
                    direction=direction,
                    port_type='input' if is_input else 'output',
                    machine_id=machine_id,
                ))

        if not ports:
            return None

        return PlacedIOBlock(
            machine_id=machine_id,
            block_type=block_type,
            base_x=ports[0].position[0],
            base_y=ports[0].position[1],
            floor=floor,
            side=side,
            ports=ports,
        )

    def _place_processing_machines(
        self,
        spec: FoundationSpec,
        num_floors: int,
        scenario: Dict,
        occupied: Set[Tuple[int, int, int]],
        valid_cells: Set[Tuple[int, int, int]],
    ) -> Tuple[List[PlacedMachine], Set[Tuple[int, int, int]]]:
        """Place processing machines in the interior."""
        machines = []
        machines_per_floor = scenario.get('machines_per_floor', 0)

        if machines_per_floor == 0:
            return machines, occupied

        # Smaller margin = more crowded placement
        margin = 2  # Reduced from 3 - stay closer to edges

        for floor in range(num_floors):
            for _ in range(machines_per_floor):
                # Try to place a machine
                for attempt in range(100):
                    machine_type = random.choice(PLACEABLE_MACHINES)
                    rotation = random.choice(list(Rotation))

                    # Create temporary machine to get rotated size
                    temp_machine = PlacedMachine(
                        machine_type=machine_type,
                        x=0, y=0, floor=floor,
                        rotation=rotation,
                    )
                    w, h, d = temp_machine._get_rotated_size()

                    # Check bounds with rotated size
                    if spec.grid_width - margin - w < margin or spec.grid_height - margin - h < margin:
                        continue  # Machine too big for this rotation

                    x = random.randint(margin, spec.grid_width - margin - w)
                    y = random.randint(margin, spec.grid_height - margin - h)

                    # Create actual machine at position
                    machine = PlacedMachine(
                        machine_type=machine_type,
                        x=x, y=y, floor=floor,
                        rotation=rotation,
                    )

                    # Get actual occupied cells (rotation-aware)
                    new_cells = machine.get_occupied_cells()

                    # Check all cells are valid
                    fits = True
                    for cell in new_cells:
                        if cell in occupied or cell not in valid_cells:
                            fits = False
                            break

                    if fits:
                        # Also check that port positions aren't already occupied
                        port_positions = set(machine.get_input_positions() + machine.get_output_positions())
                        if port_positions & occupied:
                            continue  # Port would overlap with existing occupied cell

                        machines.append(machine)
                        occupied.update(new_cells)
                        # Reserve port positions so other machines don't block them
                        occupied.update(port_positions)
                        break

        return machines, occupied

    def _generate_machine_chains(
        self,
        io_blocks: List[PlacedIOBlock],
        machines: List[PlacedMachine],
        num_floors: int,
    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Generate connections forming machine chains.

        Creates connections:
        - input_port → machine_input OR output_port (if no machines)
        - machine_output → output_port OR another_machine_input
        """
        connections = []

        # Separate inputs and outputs
        input_ports = []
        output_ports = []

        for block in io_blocks:
            for port in block.ports:
                if port.port_type == 'input':
                    input_ports.append(port)
                else:
                    output_ports.append(port)

        if not machines:
            # Direct input → output connections (by floor, then position)
            # Group by floor
            inputs_by_floor = {}
            outputs_by_floor = {}

            for port in input_ports:
                floor = port.position[2]
                if floor not in inputs_by_floor:
                    inputs_by_floor[floor] = []
                inputs_by_floor[floor].append(port)

            for port in output_ports:
                floor = port.position[2]
                if floor not in outputs_by_floor:
                    outputs_by_floor[floor] = []
                outputs_by_floor[floor].append(port)

            # Connect same-floor ports first
            for floor in inputs_by_floor:
                floor_inputs = inputs_by_floor[floor]
                floor_outputs = outputs_by_floor.get(floor, [])

                for i, inp in enumerate(floor_inputs):
                    if i < len(floor_outputs):
                        connections.append((inp.position, floor_outputs[i].position))

            # Cross-floor connections for remaining ports
            remaining_inputs = []
            remaining_outputs = []

            for floor, ports in inputs_by_floor.items():
                used = len(outputs_by_floor.get(floor, []))
                remaining_inputs.extend(ports[used:])

            for floor, ports in outputs_by_floor.items():
                used = len(inputs_by_floor.get(floor, []))
                remaining_outputs.extend(ports[used:])

            for i, inp in enumerate(remaining_inputs):
                if i < len(remaining_outputs):
                    connections.append((inp.position, remaining_outputs[i].position))

        else:
            # Connect through machines using proper input/output port positions
            # Machine input ports are where belts route TO the machine
            # Machine output ports are where belts route FROM the machine

            # Build set of all occupied cells (machine cells)
            all_machine_cells = set()
            for m in machines:
                all_machine_cells.update(m.get_occupied_cells())

            # Build list of (machine_index, input_port_pos, output_port_pos)
            # Only include machines whose ports are not on occupied cells
            machine_ports = []
            for i, m in enumerate(machines):
                input_positions = m.get_input_positions()
                output_positions = m.get_output_positions()
                # Use first input/output port (most machines have just one)
                input_pos = input_positions[0] if input_positions else None
                output_pos = output_positions[0] if output_positions else None
                # Skip if port positions are on occupied cells (rotated multi-cell machine issue)
                if input_pos and output_pos:
                    if input_pos in all_machine_cells or output_pos in all_machine_cells:
                        continue  # Invalid port position, skip this machine
                    machine_ports.append((i, input_pos, output_pos, m))

            # Pair IO ports with closest machines
            used_machines = set()

            for inp in input_ports:
                # Find closest machine input port on same floor or any floor
                best_machine_idx = None
                best_dist = float('inf')

                for i, (machine_idx, m_input, m_output, m) in enumerate(machine_ports):
                    if machine_idx in used_machines:
                        continue
                    dist = abs(inp.position[0] - m_input[0]) + abs(inp.position[1] - m_input[1])
                    dist += abs(inp.position[2] - m_input[2]) * 5  # Penalize floor changes
                    if dist < best_dist:
                        best_dist = dist
                        best_machine_idx = machine_idx

                if best_machine_idx is not None:
                    # Connect input to machine's input port position
                    for machine_idx, m_input, m_output, m in machine_ports:
                        if machine_idx == best_machine_idx:
                            connections.append((inp.position, m_input))
                            used_machines.add(best_machine_idx)
                            break

            # Connect machines to outputs
            used_outputs = set()
            for machine_idx, m_input, m_output, m in machine_ports:
                if machine_idx not in used_machines:
                    continue

                # Find closest output to machine's output port
                best_out = None
                best_dist = float('inf')

                for j, out in enumerate(output_ports):
                    if j in used_outputs:
                        continue
                    dist = abs(m_output[0] - out.position[0]) + abs(m_output[1] - out.position[1])
                    dist += abs(m_output[2] - out.position[2]) * 5
                    if dist < best_dist:
                        best_dist = dist
                        best_out = j

                if best_out is not None:
                    connections.append((m_output, output_ports[best_out].position))
                    used_outputs.add(best_out)

        return connections

    def generate_t_foundation_problem(
        self,
        num_floors: int = 3,
        inputs_per_floor: int = 4,
        outputs_per_side: int = 4,
    ) -> SyntheticProblem:
        """
        Generate a T 3x2 foundation problem as described by user:
        - 12 inputs on South side (4 per floor × 3 floors)
        - 48 outputs on W/N/E/extra N sides across 3 floors
        """
        spec = FOUNDATION_SPECS['T']
        self.problem_counter += 1
        problem_id = f"realistic_T_fullscale_{self.problem_counter:05d}"

        valid_cells = self._get_valid_cells(spec, num_floors)
        occupied = set()

        inputs = []
        outputs = []
        connections = []

        # Place inputs on South side (middle unit at y=grid_height-1)
        # T-shape: units at (0,0), (1,0), (2,0), (1,1)
        # South side of unit (1,1) is exposed
        for floor in range(num_floors):
            # South ports on the bottom unit (1,1)
            center_x = 7 + 1 * 20  # Unit 1 center
            y = spec.grid_height - 1

            for i in range(inputs_per_floor):
                x = center_x - 2 + i
                pos = (x, y, floor)
                if pos in valid_cells:
                    inputs.append(pos)
                    occupied.add(pos)

        # Place outputs on multiple sides
        # West side of unit (0,0)
        for floor in range(num_floors):
            center_y = 7  # Unit 0 center y
            for i in range(outputs_per_side):
                y = center_y - 2 + i
                pos = (0, y, floor)
                if pos in valid_cells:
                    outputs.append(pos)
                    occupied.add(pos)

        # North side of units (0,0), (1,0), (2,0)
        for floor in range(num_floors):
            for unit in range(3):
                center_x = 7 + unit * 20
                for i in range(outputs_per_side):
                    x = center_x - 2 + i
                    pos = (x, 0, floor)
                    if pos in valid_cells and pos not in occupied:
                        outputs.append(pos)
                        occupied.add(pos)

        # East side of unit (2,0)
        for floor in range(num_floors):
            center_y = 7
            x = spec.grid_width - 1
            for i in range(outputs_per_side):
                y = center_y - 2 + i
                pos = (x, y, floor)
                if pos in valid_cells and pos not in occupied:
                    outputs.append(pos)
                    occupied.add(pos)

        # Create connections (input[i] → output[i] pattern)
        for i, inp in enumerate(inputs):
            if i < len(outputs):
                connections.append((inp, outputs[i]))

        return SyntheticProblem(
            problem_id=problem_id,
            foundation_type='T',
            grid_width=spec.grid_width,
            grid_height=spec.grid_height,
            num_floors=num_floors,
            machines=[],
            input_positions=inputs,
            output_positions=outputs,
            connections=connections,
            difficulty='extreme',
        )

    def generate_batch(
        self,
        count: int,
        foundation_types: List[str] = None,
        scenarios: List[str] = None,
    ) -> List[SyntheticProblem]:
        """Generate a batch of realistic problems."""
        # Smaller foundations = harder problems (more crowded)
        small_foundations = ['1x1', '2x1', '1x2', '2x2']
        medium_foundations = ['3x1', '1x3', '3x2', '2x3', 'L', 'T']
        large_foundations = ['4x1', '1x4', '4x2', '2x4', '3x3', 'L4', 'S4', 'Cross']

        if foundation_types is None:
            foundation_types = list(FOUNDATION_SPECS.keys())
        if scenarios is None:
            scenarios = list(self.SCENARIOS.keys())

        problems = []
        for _ in range(count):
            scenario = random.choice(scenarios)
            scenario_config = self.SCENARIOS[scenario]

            # For harder scenarios, prefer smaller foundations
            difficulty = scenario_config['difficulty']
            if difficulty == 'extreme':
                # 60% small, 30% medium, 10% large
                foundation_pool = small_foundations * 6 + medium_foundations * 3 + large_foundations
            elif difficulty == 'hard':
                # 40% small, 40% medium, 20% large
                foundation_pool = small_foundations * 4 + medium_foundations * 4 + large_foundations * 2
            else:
                # Use all equally
                foundation_pool = foundation_types

            # Filter to only valid foundations
            valid_pool = [f for f in foundation_pool if f in foundation_types]
            if not valid_pool:
                valid_pool = foundation_types

            foundation = random.choice(valid_pool)

            # Filter scenarios by foundation capabilities
            spec = FOUNDATION_SPECS.get(foundation, FOUNDATION_SPECS['3x2'])

            if scenario_config['min_floors'] > spec.num_floors:
                scenario = 'simple_passthrough'
                scenario_config = self.SCENARIOS[scenario]

            num_floors = random.randint(1, min(3, spec.num_floors))

            problems.append(self.generate_problem(
                foundation_type=foundation,
                scenario=scenario,
                num_floors=num_floors,
            ))

        return problems


def generate_realistic_data(
    count: int = 50,
    db_path: str = "synthetic_training.db",
    time_limit: float = 30.0,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Generate realistic problems and solve them with CP-SAT.

    Args:
        count: Number of problems
        db_path: Database path
        time_limit: Time limit per problem
        verbose: Print progress

    Returns:
        (successes, failures)
    """
    generator = RealisticProblemGenerator()
    store = SyntheticDataStore(db_path)

    if verbose:
        print(f"Generating {count} realistic routing problems...")
        print(f"Time limit: {time_limit}s per problem\n")

    successes = 0
    failures = 0

    problems = generator.generate_batch(count)

    for i, problem in enumerate(problems):
        if verbose:
            print(f"[{i+1}/{count}] {problem.problem_id} ({problem.difficulty})...", end=" ", flush=True)

        store.save_problem(problem)
        result = solve_problem(problem, time_limit)
        store.save_result(result)

        if result.success:
            successes += 1
            if verbose:
                print(f"OK ({result.solve_time:.1f}s, {result.num_belts} belts)")
        else:
            failures += 1
            if verbose:
                progress = f"{result.routing_progress*100:.0f}%" if result.routing_progress > 0 else ""
                msg = result.error_message[:30] if result.error_message else "Failed"
                print(f"FAIL {progress}: {msg}")

        # Clean up memory after each problem
        del result
        gc.collect()

    if verbose:
        print(f"\nResults: {successes} successes, {failures} failures")
        print(f"Success rate: {100*successes/max(1,count):.1f}%")

    return successes, failures


def _extract_placement_features(problem: SyntheticProblem) -> Dict[str, Any]:
    """
    Extract placement features from a problem for PlacementPredictor training.

    Returns features about machine placement quality before routing is attempted.
    """
    occupied = problem.get_occupied()
    grid_area = problem.grid_width * problem.grid_height
    grid_volume = grid_area * problem.num_floors

    features = {
        'num_machines': len(problem.machines),
        'num_connections': len(problem.connections),
        'grid_width': problem.grid_width,
        'grid_height': problem.grid_height,
        'num_floors': problem.num_floors,
        'machine_density': len(occupied) / max(1, grid_volume),
    }

    # Calculate machine spread and clustering
    if problem.machines:
        xs = [m.x for m in problem.machines]
        ys = [m.y for m in problem.machines]

        features['center_of_mass_x'] = sum(xs) / len(xs) / max(1, problem.grid_width - 1)
        features['center_of_mass_y'] = sum(ys) / len(ys) / max(1, problem.grid_height - 1)

        if len(problem.machines) > 1:
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            variance = sum((x - mean_x)**2 + (y - mean_y)**2 for x, y in zip(xs, ys)) / len(problem.machines)
            features['machine_spread'] = (variance ** 0.5) / ((problem.grid_width**2 + problem.grid_height**2) ** 0.5)
        else:
            features['machine_spread'] = 0.0
    else:
        features['center_of_mass_x'] = 0.5
        features['center_of_mass_y'] = 0.5
        features['machine_spread'] = 0.0

    # Calculate connection complexity
    if problem.connections:
        manhattan_dists = []
        for src, dst in problem.connections:
            dist = abs(src[0] - dst[0]) + abs(src[1] - dst[1]) + abs(src[2] - dst[2])
            manhattan_dists.append(dist)

        features['avg_connection_distance'] = sum(manhattan_dists) / len(manhattan_dists)
        features['max_connection_distance'] = max(manhattan_dists)
        features['min_connection_distance'] = min(manhattan_dists)
        features['total_manhattan_distance'] = sum(manhattan_dists)
    else:
        features['avg_connection_distance'] = 0
        features['max_connection_distance'] = 0
        features['min_connection_distance'] = 0
        features['total_manhattan_distance'] = 0

    # Estimate path crossing potential
    if len(problem.connections) > 1:
        overlap_count = 0
        for i, (src1, dst1) in enumerate(problem.connections):
            box1 = (min(src1[0], dst1[0]), min(src1[1], dst1[1]),
                    max(src1[0], dst1[0]), max(src1[1], dst1[1]))
            for src2, dst2 in problem.connections[i+1:]:
                box2 = (min(src2[0], dst2[0]), min(src2[1], dst2[1]),
                        max(src2[0], dst2[0]), max(src2[1], dst2[1]))
                # Check bounding box overlap
                if not (box1[2] < box2[0] or box2[2] < box1[0] or
                        box1[3] < box2[1] or box2[3] < box1[1]):
                    overlap_count += 1
        max_pairs = len(problem.connections) * (len(problem.connections) - 1) / 2
        features['crossing_potential'] = overlap_count / max(1, max_pairs)
    else:
        features['crossing_potential'] = 0.0

    # I/O spread
    if problem.input_positions and problem.output_positions:
        all_io = problem.input_positions + problem.output_positions
        io_xs = [p[0] for p in all_io]
        io_ys = [p[1] for p in all_io]
        features['io_spread_x'] = (max(io_xs) - min(io_xs)) / max(1, problem.grid_width - 1) if len(io_xs) > 1 else 0
        features['io_spread_y'] = (max(io_ys) - min(io_ys)) / max(1, problem.grid_height - 1) if len(io_ys) > 1 else 0
    else:
        features['io_spread_x'] = 0
        features['io_spread_y'] = 0

    return features


def solve_problem(problem: SyntheticProblem, time_limit: float = 30.0) -> SolverResult:
    """
    Run the A* router on a synthetic problem with rich ML training data capture.

    Args:
        problem: The problem to solve
        time_limit: Solver time limit in seconds

    Returns:
        SolverResult with outcome, partial progress, and ML training data
    """
    start_time = time.time()
    num_connections = len(problem.connections)

    # Extract placement features BEFORE routing
    placement_features = _extract_placement_features(problem)

    try:
        # Use enhanced routing with ML data capture
        (partial_belts, partial_success, failed_indices,
         connection_results, total_nodes, total_blocked) = _solve_with_ml_data(
            problem, time_limit
        )

        connections_routed = num_connections - len(failed_indices)
        routing_progress = connections_routed / max(1, num_connections)

        all_routed = len(failed_indices) == 0

        return SolverResult(
            problem_id=problem.problem_id,
            success=all_routed,
            solve_time=time.time() - start_time,
            num_belts=len(partial_belts),
            error_message="" if all_routed else f"Failed {len(failed_indices)}/{num_connections} connections",
            connections_attempted=num_connections,
            connections_routed=connections_routed,
            routing_progress=routing_progress,
            failed_connection_indices=json.dumps(failed_indices),
            partial_belt_positions=json.dumps([(x, y, z) for x, y, z, _, _ in partial_belts]),
            solver_iterations=total_nodes,
            best_objective=float(len(partial_belts)),
            # Enhanced ML data
            connection_results=json.dumps(connection_results),
            placement_features=json.dumps(placement_features),
            total_nodes_explored=total_nodes,
            total_blocked_positions=total_blocked,
        )

    except Exception as e:
        return SolverResult(
            problem_id=problem.problem_id,
            success=False,
            solve_time=time.time() - start_time,
            num_belts=0,
            error_message=str(e)[:200],
            connections_attempted=num_connections,
            connections_routed=0,
            routing_progress=0.0,
            failed_connection_indices=json.dumps(list(range(num_connections))),
            partial_belt_positions="[]",
            solver_iterations=0,
            best_objective=0.0,
            connection_results="[]",
            placement_features=json.dumps(placement_features),
            total_nodes_explored=0,
            total_blocked_positions=0,
        )


def _solve_with_partial_tracking(
    problem: SyntheticProblem,
    time_limit: float,
) -> Tuple[List, bool, List[int]]:
    """
    Try to route connections sequentially to track which ones fail.

    Returns:
        (partial_belts, all_success, failed_indices)
    """
    from ..evolution.router import BeltRouter, Connection

    router = BeltRouter(
        problem.grid_width,
        problem.grid_height,
        problem.num_floors,
    )
    router.set_occupied(problem.get_occupied())

    all_belts = []
    failed_indices = []

    for i, (src, dst) in enumerate(problem.connections):
        # Create a connection object
        conn = Connection(
            from_pos=src,
            to_pos=dst,
            from_direction=Rotation.EAST,  # Default
            to_direction=Rotation.EAST,
        )

        result = router.route_connection(conn)

        if result.success:
            all_belts.extend(result.belts)
        else:
            failed_indices.append(i)

    all_success = len(failed_indices) == 0
    return all_belts, all_success, failed_indices


def _solve_with_ml_data(
    problem: SyntheticProblem,
    time_limit: float,
) -> Tuple[List, bool, List[int], List[Dict], int, int]:
    """
    Route connections while capturing rich ML training data.

    Returns:
        (partial_belts, all_success, failed_indices, connection_results,
         total_nodes_explored, total_blocked_positions)

    connection_results is a list of dicts with per-connection data:
        - index: connection index
        - success: whether routing succeeded
        - src/dst: source and destination positions
        - features: connection features (distance, congestion, etc.)
        - path: the routed path (if successful)
        - belt_directions: belt types and rotations along path
        - nodes_explored: A* nodes expanded for this connection
        - blocked_positions: positions that blocked the search
        - grid_state_before: occupied cells before this routing attempt
    """
    from ..evolution.router import BeltRouter, Connection
    from .features import extract_connection_features

    router = BeltRouter(
        problem.grid_width,
        problem.grid_height,
        problem.num_floors,
    )
    initial_occupied = problem.get_occupied()
    router.set_occupied(initial_occupied)

    all_belts = []
    failed_indices = []
    connection_results = []
    total_nodes_explored = 0
    total_blocked_positions = 0

    # Track current occupied state (grows as we route)
    current_occupied = initial_occupied.copy()

    for i, (src, dst) in enumerate(problem.connections):
        # Extract connection features BEFORE routing
        conn_features = extract_connection_features(
            connection=(src, dst),
            grid_width=problem.grid_width,
            grid_height=problem.grid_height,
            occupied=current_occupied,
            connection_index=i,
            total_connections=len(problem.connections),
            connections_routed=i - len(failed_indices),
        )

        # Create connection object
        conn = Connection(
            from_pos=src,
            to_pos=dst,
            from_direction=Rotation.EAST,
            to_direction=Rotation.EAST,
        )

        # Route with statistics capture
        result = router.route_connection_with_stats(conn)

        # Build connection result entry
        conn_result = {
            'index': i,
            'success': result.success,
            'src': list(src),
            'dst': list(dst),
            'features': {
                'manhattan_distance': conn_features.manhattan_distance,
                'normalized_distance': conn_features.normalized_distance,
                'src_local_density': conn_features.src_local_density,
                'dst_local_density': conn_features.dst_local_density,
                'path_corridor_density': conn_features.path_corridor_density,
                'crosses_center': conn_features.crosses_center,
                'floor_change': conn_features.floor_change,
                'direction_complexity': conn_features.direction_complexity,
            },
            'nodes_explored': result.nodes_explored,
            'blocked_positions': [list(p) for p in result.blocked_positions[:20]],  # Limit to 20
        }

        total_nodes_explored += result.nodes_explored
        total_blocked_positions += len(result.blocked_positions)

        if result.success:
            all_belts.extend(result.belts)

            # Store path and belt directions for successful routes
            conn_result['path'] = [list(p) for p in result.path]
            conn_result['belt_directions'] = [
                {'pos': [b[0], b[1], b[2]], 'type': b[3].name, 'rotation': b[4].name}
                for b in result.belts
            ]
            conn_result['path_length'] = len(result.path)

            # Update occupied set with new belt positions
            for b in result.belts:
                current_occupied.add((b[0], b[1], b[2]))

            # Update features with outcome
            conn_features.routed_successfully = True
            conn_features.actual_path_length = len(result.path)
            if conn_features.manhattan_distance > 0:
                conn_features.path_stretch = len(result.path) / conn_features.manhattan_distance
        else:
            failed_indices.append(i)
            conn_result['path'] = []
            conn_result['belt_directions'] = []
            conn_result['failure_reason'] = 'no_path_found'
            conn_features.routed_successfully = False

        connection_results.append(conn_result)

    all_success = len(failed_indices) == 0
    return all_belts, all_success, failed_indices, connection_results, total_nodes_explored, total_blocked_positions


def generate_and_solve(
    count: int = 100,
    db_path: str = "synthetic_training.db",
    time_limit: float = 30.0,
    difficulty_distribution: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Generate problems and run the solver.

    Args:
        count: Number of problems to generate
        db_path: Database path
        time_limit: Solver time limit per problem
        difficulty_distribution: Distribution of difficulties
        verbose: Print progress

    Returns:
        Tuple of (successes, failures)
    """
    generator = SyntheticProblemGenerator()
    store = SyntheticDataStore(db_path)

    if verbose:
        print(f"Generating and solving {count} problems...")
        print(f"Time limit: {time_limit}s per problem\n")

    successes = 0
    failures = 0

    problems = generator.generate_batch(count, difficulty_distribution)

    for i, problem in enumerate(problems):
        if verbose:
            print(f"[{i+1}/{count}] {problem.problem_id} ({problem.difficulty})...", end=" ", flush=True)

        # Save problem
        store.save_problem(problem)

        # Solve
        result = solve_problem(problem, time_limit)

        # Save result
        store.save_result(result)

        if result.success:
            successes += 1
            if verbose:
                print(f"OK ({result.solve_time:.1f}s, {result.num_belts} belts)")
        else:
            failures += 1
            if verbose:
                msg = result.error_message[:30] if result.error_message else "Failed"
                print(f"FAIL: {msg}")

        # Clean up memory after each problem
        del result
        gc.collect()

    if verbose:
        print(f"\nResults: {successes} successes, {failures} failures")
        print(f"Success rate: {100*successes/count:.1f}%")

    return successes, failures


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of problems to generate")
    parser.add_argument("--db", type=str, default="synthetic_training.db",
                       help="Database path")
    parser.add_argument("--time-limit", type=float, default=30.0,
                       help="Solver time limit per problem")
    parser.add_argument("--difficulty", type=str, default=None,
                       choices=['easy', 'medium', 'hard', 'extreme'],
                       help="Generate only this difficulty")
    parser.add_argument("--stats", action="store_true",
                       help="Show database statistics")
    parser.add_argument("--export", type=str, metavar="FILE",
                       help="Export training data to JSON")

    # Realistic problem generation
    parser.add_argument("--realistic", action="store_true",
                       help="Generate realistic problems with machine chains")
    parser.add_argument("--foundation", type=str, default=None,
                       choices=list(FOUNDATION_SPECS.keys()),
                       help="Foundation type for realistic problems")
    parser.add_argument("--scenario", type=str, default=None,
                       choices=list(RealisticProblemGenerator.SCENARIOS.keys()),
                       help="Problem scenario for realistic problems")
    parser.add_argument("--t-foundation", action="store_true",
                       help="Generate T foundation full-scale problems")

    args = parser.parse_args()

    if args.stats:
        store = SyntheticDataStore(args.db)
        stats = store.get_stats()
        print("\n" + "="*60)
        print("Synthetic Training Database Statistics")
        print("="*60)
        print(f"\nTotal problems: {stats['total_problems']}")
        print(f"Total solved: {stats['total_solved']}")
        print(f"Successes: {stats['successes']}")
        if stats['total_solved'] > 0:
            print(f"Success rate: {100*stats['successes']/stats['total_solved']:.1f}%")
            print(f"Avg solve time: {stats['avg_solve_time']:.2f}s")

        print("\nBy difficulty:")
        for diff, count in stats.get('by_difficulty', {}).items():
            results = stats.get('results_by_difficulty', {}).get(diff, {})
            rate = 100 * results.get('successes', 0) / max(1, results.get('total', 1))
            print(f"  {diff}: {count} problems, {rate:.1f}% success")

        print("\nBy foundation:")
        for found, count in stats.get('by_foundation', {}).items():
            print(f"  {found}: {count}")
        return

    if args.export:
        store = SyntheticDataStore(args.db)
        count = store.export_training_data(args.export)
        print(f"Exported {count} examples to {args.export}")
        return

    # T-foundation full-scale problem
    if args.t_foundation:
        print("Generating T foundation full-scale problems...")
        generator = RealisticProblemGenerator()
        store = SyntheticDataStore(args.db)

        for i in range(args.count):
            print(f"[{i+1}/{args.count}] Generating T foundation problem...", end=" ", flush=True)
            problem = generator.generate_t_foundation_problem(num_floors=3)
            store.save_problem(problem)
            result = solve_problem(problem, args.time_limit)
            store.save_result(result)

            if result.success:
                print(f"OK ({result.solve_time:.1f}s, {result.num_belts} belts)")
            else:
                progress = f"{result.routing_progress*100:.0f}%" if result.routing_progress > 0 else ""
                print(f"FAIL {progress}: {result.error_message[:30]}")

            # Clean up memory
            del result, problem
            gc.collect()
        return

    # Realistic problem generation
    if args.realistic:
        generate_realistic_data(
            count=args.count,
            db_path=args.db,
            time_limit=args.time_limit,
            verbose=True,
        )
        return

    # Generate and solve
    if args.difficulty:
        dist = {args.difficulty: 1.0}
    else:
        dist = None

    generate_and_solve(
        count=args.count,
        db_path=args.db,
        time_limit=args.time_limit,
        difficulty_distribution=dist,
        verbose=True,
    )


if __name__ == "__main__":
    main()
