"""
Data logging and storage for routing training data.

Logs routing attempts with features and outcomes for later training.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional, Any
import os

from .features import (
    SolutionFeatures,
    ConnectionFeatures,
    extract_solution_features,
    extract_connection_features,
)


@dataclass
class RoutingAttempt:
    """Record of a single routing attempt."""

    # Identifiers
    attempt_id: str = ""
    timestamp: str = ""

    # Configuration
    foundation_type: str = ""
    grid_width: int = 0
    grid_height: int = 0
    num_floors: int = 4
    routing_mode: str = ""  # 'astar', 'global', 'hybrid'

    # Input specification
    num_inputs: int = 0
    num_outputs: int = 0
    num_machines: int = 0
    num_connections: int = 0

    # Extracted features (serialized)
    solution_features: Optional[Dict] = None
    connection_features: Optional[List[Dict]] = None

    # Outcome
    routing_success: bool = False
    connections_routed: int = 0
    total_belt_length: int = 0
    throughput: float = 0.0
    solve_time: float = 0.0

    # Raw data (for debugging/retraining)
    machines_json: str = ""
    belts_json: str = ""
    connections_json: str = ""
    paths_json: str = ""


class DataStore:
    """
    SQLite-based storage for routing training data.

    Provides:
    - Efficient storage of many routing attempts
    - Querying by foundation type, success rate, etc.
    - Export to pandas DataFrame for training
    """

    def __init__(self, db_path: str = "routing_data.db"):
        """Initialize data store."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routing_attempts (
                attempt_id TEXT PRIMARY KEY,
                timestamp TEXT,
                foundation_type TEXT,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,
                routing_mode TEXT,
                num_inputs INTEGER,
                num_outputs INTEGER,
                num_machines INTEGER,
                num_connections INTEGER,
                routing_success INTEGER,
                connections_routed INTEGER,
                total_belt_length INTEGER,
                throughput REAL,
                solve_time REAL,
                solution_features TEXT,
                connection_features TEXT,
                machines_json TEXT,
                belts_json TEXT,
                connections_json TEXT,
                paths_json TEXT
            )
        ''')

        # Index for common queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_foundation
            ON routing_attempts(foundation_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_success
            ON routing_attempts(routing_success)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON routing_attempts(timestamp)
        ''')

        conn.commit()
        conn.close()

    def save_attempt(self, attempt: RoutingAttempt):
        """Save a routing attempt to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO routing_attempts
            (attempt_id, timestamp, foundation_type, grid_width, grid_height,
             num_floors, routing_mode, num_inputs, num_outputs, num_machines,
             num_connections, routing_success, connections_routed,
             total_belt_length, throughput, solve_time, solution_features,
             connection_features, machines_json, belts_json, connections_json,
             paths_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            attempt.attempt_id,
            attempt.timestamp,
            attempt.foundation_type,
            attempt.grid_width,
            attempt.grid_height,
            attempt.num_floors,
            attempt.routing_mode,
            attempt.num_inputs,
            attempt.num_outputs,
            attempt.num_machines,
            attempt.num_connections,
            1 if attempt.routing_success else 0,
            attempt.connections_routed,
            attempt.total_belt_length,
            attempt.throughput,
            attempt.solve_time,
            json.dumps(attempt.solution_features) if attempt.solution_features else None,
            json.dumps(attempt.connection_features) if attempt.connection_features else None,
            attempt.machines_json,
            attempt.belts_json,
            attempt.connections_json,
            attempt.paths_json,
        ))

        conn.commit()
        conn.close()

    def get_attempts(
        self,
        foundation_type: Optional[str] = None,
        routing_success: Optional[bool] = None,
        limit: int = 1000,
    ) -> List[RoutingAttempt]:
        """Query routing attempts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM routing_attempts WHERE 1=1"
        params = []

        if foundation_type:
            query += " AND foundation_type = ?"
            params.append(foundation_type)

        if routing_success is not None:
            query += " AND routing_success = ?"
            params.append(1 if routing_success else 0)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        attempts = []
        for row in rows:
            attempt = RoutingAttempt(
                attempt_id=row[0],
                timestamp=row[1],
                foundation_type=row[2],
                grid_width=row[3],
                grid_height=row[4],
                num_floors=row[5],
                routing_mode=row[6],
                num_inputs=row[7],
                num_outputs=row[8],
                num_machines=row[9],
                num_connections=row[10],
                routing_success=bool(row[11]),
                connections_routed=row[12],
                total_belt_length=row[13],
                throughput=row[14],
                solve_time=row[15],
                solution_features=json.loads(row[16]) if row[16] else None,
                connection_features=json.loads(row[17]) if row[17] else None,
                machines_json=row[18],
                belts_json=row[19],
                connections_json=row[20],
                paths_json=row[21],
            )
            attempts.append(attempt)

        return attempts

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics of stored data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total attempts
        cursor.execute("SELECT COUNT(*) FROM routing_attempts")
        stats['total_attempts'] = cursor.fetchone()[0]

        # Success rate
        cursor.execute("SELECT AVG(routing_success) FROM routing_attempts")
        stats['success_rate'] = cursor.fetchone()[0] or 0

        # By foundation type
        cursor.execute('''
            SELECT foundation_type, COUNT(*), AVG(routing_success)
            FROM routing_attempts
            GROUP BY foundation_type
        ''')
        stats['by_foundation'] = {
            row[0]: {'count': row[1], 'success_rate': row[2]}
            for row in cursor.fetchall()
        }

        # Average metrics
        cursor.execute('''
            SELECT AVG(throughput), AVG(solve_time), AVG(total_belt_length)
            FROM routing_attempts
            WHERE routing_success = 1
        ''')
        row = cursor.fetchone()
        stats['avg_throughput'] = row[0] or 0
        stats['avg_solve_time'] = row[1] or 0
        stats['avg_belt_length'] = row[2] or 0

        conn.close()
        return stats

    def export_to_csv(self, filepath: str, include_features: bool = True):
        """Export data to CSV for external analysis."""
        import csv

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM routing_attempts")
        rows = cursor.fetchall()

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        conn.close()

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

    def export_feature_vectors(self) -> Tuple[List[List[float]], List[bool], List[float]]:
        """
        Export feature vectors for ML training.

        Returns:
            X: List of feature vectors
            y_success: List of success labels (bool)
            y_throughput: List of throughput values
        """
        attempts = self.get_attempts(limit=100000)

        X = []
        y_success = []
        y_throughput = []

        for attempt in attempts:
            if attempt.solution_features:
                # Reconstruct SolutionFeatures and get vector
                features = SolutionFeatures(**attempt.solution_features)
                X.append(features.to_feature_vector())
                y_success.append(attempt.routing_success)
                y_throughput.append(attempt.throughput)

        return X, y_success, y_throughput


class RoutingLogger:
    """
    Logger to capture routing attempts during solving.

    Usage:
        logger = RoutingLogger()

        # In your solver:
        logger.start_attempt(foundation_type, grid_size, ...)

        # After routing:
        logger.log_result(machines, belts, paths, success, throughput)

        # Save to database
        logger.save()
    """

    def __init__(self, db_path: str = "routing_data.db"):
        """Initialize logger."""
        self.store = DataStore(db_path)
        self.current_attempt: Optional[RoutingAttempt] = None
        self._start_time: float = 0

    def start_attempt(
        self,
        foundation_type: str,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        routing_mode: str,
        input_positions: List[Tuple],
        output_positions: List[Tuple],
        connections: List[Tuple],
    ):
        """Start logging a new routing attempt."""
        self._start_time = time.time()

        self.current_attempt = RoutingAttempt(
            attempt_id=f"{foundation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now().isoformat(),
            foundation_type=foundation_type,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            routing_mode=routing_mode,
            num_inputs=len(input_positions),
            num_outputs=len(output_positions),
            num_connections=len(connections),
            connections_json=json.dumps([
                ((s[0], s[1], s[2]), (d[0], d[1], d[2]))
                for s, d in connections
            ]),
        )

    def log_machines(self, machines: List[Tuple]):
        """Log machine placements."""
        if self.current_attempt:
            self.current_attempt.num_machines = len(machines)
            self.current_attempt.machines_json = json.dumps([
                (str(m[0]), m[1], m[2], m[3], str(m[4])) for m in machines
            ])

    def log_result(
        self,
        machines: List[Tuple],
        belts: List[Tuple],
        paths: List[List[Tuple]],
        input_positions: List[Tuple],
        output_positions: List[Tuple],
        connections: List[Tuple],
        occupied: Set[Tuple],
        routing_success: bool,
        throughput: float,
    ):
        """Log the result of a routing attempt."""
        if not self.current_attempt:
            return

        solve_time = time.time() - self._start_time

        # Store raw data
        self.current_attempt.machines_json = json.dumps([
            (str(m[0]) if hasattr(m[0], 'name') else str(m[0]),
             m[1], m[2], m[3],
             str(m[4]) if hasattr(m[4], 'name') else str(m[4]))
            for m in machines
        ])
        self.current_attempt.belts_json = json.dumps([
            (b[0], b[1], b[2],
             str(b[3]) if hasattr(b[3], 'name') else str(b[3]),
             str(b[4]) if hasattr(b[4], 'name') else str(b[4]))
            for b in belts
        ])
        self.current_attempt.paths_json = json.dumps(paths)

        # Calculate metrics
        self.current_attempt.routing_success = routing_success
        self.current_attempt.connections_routed = sum(1 for p in paths if p)
        self.current_attempt.total_belt_length = sum(len(p) for p in paths if p)
        self.current_attempt.throughput = throughput
        self.current_attempt.solve_time = solve_time

        # Extract features
        solution_features = extract_solution_features(
            grid_width=self.current_attempt.grid_width,
            grid_height=self.current_attempt.grid_height,
            num_floors=self.current_attempt.num_floors,
            machines=machines,
            belts=belts,
            input_positions=input_positions,
            output_positions=output_positions,
            connections=connections,
            paths=paths,
            occupied=occupied,
            routing_success=routing_success,
            throughput=throughput,
            solve_time=solve_time,
            foundation_type=self.current_attempt.foundation_type,
        )
        self.current_attempt.solution_features = solution_features.to_dict()

        # Extract per-connection features
        conn_features = []
        for i, (conn, path) in enumerate(zip(connections, paths)):
            cf = extract_connection_features(
                connection=conn,
                grid_width=self.current_attempt.grid_width,
                grid_height=self.current_attempt.grid_height,
                occupied=occupied,
                connection_index=i,
                total_connections=len(connections),
                connections_routed=i,  # approximation
                actual_path=path,
            )
            conn_features.append(asdict(cf))
        self.current_attempt.connection_features = conn_features

    def save(self):
        """Save current attempt to database."""
        if self.current_attempt:
            self.store.save_attempt(self.current_attempt)
            self.current_attempt = None

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.store.get_stats()


def create_sample_data(db_path: str = "routing_data.db", num_samples: int = 100):
    """
    Generate sample data for testing.
    Creates synthetic routing attempts with varied parameters.
    """
    import random

    store = DataStore(db_path)

    foundation_types = ['1x1', '2x1', '2x2', '3x1', '3x2', '3x3', 'T', 'L', 'Cross']
    routing_modes = ['astar', 'global', 'hybrid']

    for i in range(num_samples):
        foundation = random.choice(foundation_types)

        # Determine grid size based on foundation
        if foundation == '1x1':
            width, height = 14, 14
        elif foundation == '2x1':
            width, height = 34, 14
        elif foundation == '2x2':
            width, height = 34, 34
        elif foundation == '3x1':
            width, height = 54, 14
        elif foundation == '3x2':
            width, height = 54, 34
        elif foundation == '3x3':
            width, height = 54, 54
        else:
            width, height = 34, 34

        num_connections = random.randint(2, 8)
        success = random.random() > 0.3  # 70% success rate

        # Create synthetic features
        features = SolutionFeatures(
            foundation_type=foundation,
            grid_width=width,
            grid_height=height,
            num_floors=4,
            total_cells=width * height * 4,
            num_machines=random.randint(1, 5),
            machine_density=random.uniform(0.01, 0.1),
            machine_spread=random.uniform(0.1, 0.5),
            center_of_mass_x=random.uniform(0.3, 0.7),
            center_of_mass_y=random.uniform(0.3, 0.7),
            num_belts=random.randint(10, 100),
            belt_density=random.uniform(0.05, 0.2),
            avg_path_length=random.uniform(5, 20),
            max_local_density_3x3=random.uniform(0.1, 0.8),
            avg_local_density_3x3=random.uniform(0.05, 0.3),
            num_connections=num_connections,
            routing_success=success,
            throughput=random.uniform(20, 180) if success else 0,
            solve_time=random.uniform(0.1, 30),
        )

        attempt = RoutingAttempt(
            attempt_id=f"sample_{i:04d}",
            timestamp=datetime.now().isoformat(),
            foundation_type=foundation,
            grid_width=width,
            grid_height=height,
            num_floors=4,
            routing_mode=random.choice(routing_modes),
            num_inputs=random.randint(1, 4),
            num_outputs=random.randint(1, 8),
            num_machines=features.num_machines,
            num_connections=num_connections,
            routing_success=success,
            connections_routed=num_connections if success else random.randint(0, num_connections - 1),
            total_belt_length=int(features.avg_path_length * num_connections),
            throughput=features.throughput,
            solve_time=features.solve_time,
            solution_features=features.to_dict(),
        )

        store.save_attempt(attempt)

    print(f"Created {num_samples} sample routing attempts in {db_path}")
    return store
