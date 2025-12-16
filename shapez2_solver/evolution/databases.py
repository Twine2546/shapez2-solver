"""Database utilities for training and ML systems.

This module contains database classes for storing training samples,
routing outcomes, and other persistent data.
"""

import pickle
import sqlite3
from typing import Dict, List, Any, Optional
import numpy as np


class TrainingSampleDB:
    """SQLite database for storing training samples with checkpointing."""

    def __init__(self, db_path: str = "training_samples.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_name TEXT,
                foundation TEXT,
                grid_width INTEGER,
                grid_height INTEGER,
                num_floors INTEGER,
                machines BLOB,
                input_positions BLOB,
                output_positions BLOB,
                routing_success INTEGER,
                is_test INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS run_progress (
                id INTEGER PRIMARY KEY,
                run_id TEXT,
                total_problems INTEGER,
                problems_processed INTEGER,
                solved_count INTEGER,
                routed_count INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_test ON training_samples(is_test)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_routing_success ON training_samples(routing_success)')

        conn.commit()
        conn.close()

    def add_sample(self, sample: Dict[str, Any], is_test: bool = False):
        """Add a training sample to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize complex objects
        machines_blob = pickle.dumps(sample['machines'])
        input_pos_blob = pickle.dumps(sample['input_positions'])
        output_pos_blob = pickle.dumps(sample['output_positions'])

        cursor.execute('''
            INSERT INTO training_samples
            (problem_name, foundation, grid_width, grid_height, num_floors,
             machines, input_positions, output_positions, routing_success, is_test)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample.get('problem_name', ''),
            sample.get('foundation', ''),
            sample['grid_width'],
            sample['grid_height'],
            sample['num_floors'],
            machines_blob,
            input_pos_blob,
            output_pos_blob,
            1 if sample['routing_success'] else 0,
            1 if is_test else 0,
        ))

        conn.commit()
        conn.close()

    def get_samples(self, is_test: bool = False) -> List[Dict[str, Any]]:
        """Retrieve samples from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT problem_name, foundation, grid_width, grid_height, num_floors,
                   machines, input_positions, output_positions, routing_success
            FROM training_samples
            WHERE is_test = ?
        ''', (1 if is_test else 0,))

        samples = []
        for row in cursor.fetchall():
            sample = {
                'problem_name': row[0],
                'foundation': row[1],
                'grid_width': row[2],
                'grid_height': row[3],
                'num_floors': row[4],
                'machines': pickle.loads(row[5]),
                'input_positions': pickle.loads(row[6]),
                'output_positions': pickle.loads(row[7]),
                'routing_success': bool(row[8]),
            }
            samples.append(sample)

        conn.close()
        return samples

    def get_sample_count(self, is_test: Optional[bool] = None) -> int:
        """Get count of samples."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if is_test is None:
            cursor.execute('SELECT COUNT(*) FROM training_samples')
        else:
            cursor.execute('SELECT COUNT(*) FROM training_samples WHERE is_test = ?',
                          (1 if is_test else 0,))

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def update_progress(self, run_id: str, total: int, processed: int, solved: int, routed: int):
        """Update run progress for checkpointing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO run_progress
            (id, run_id, total_problems, problems_processed, solved_count, routed_count, last_updated)
            VALUES (1, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (run_id, total, processed, solved, routed))

        conn.commit()
        conn.close()

    def get_progress(self, run_id: str) -> Optional[Dict[str, int]]:
        """Get run progress for resuming."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT total_problems, problems_processed, solved_count, routed_count
            FROM run_progress WHERE run_id = ?
        ''', (run_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'total_problems': row[0],
                'problems_processed': row[1],
                'solved_count': row[2],
                'routed_count': row[3],
            }
        return None

    def clear_samples(self):
        """Clear all samples (for fresh start)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM training_samples')
        cursor.execute('DELETE FROM run_progress')
        conn.commit()
        conn.close()

    def assign_test_split(self, test_ratio: float = 0.2, seed: int = 42):
        """Randomly assign samples to test set."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all sample IDs
        cursor.execute('SELECT id FROM training_samples')
        ids = [row[0] for row in cursor.fetchall()]

        if not ids:
            conn.close()
            return

        # Randomly select test samples
        np.random.seed(seed)
        np.random.shuffle(ids)
        test_count = int(len(ids) * test_ratio)
        test_ids = ids[:test_count]

        # Reset all to training
        cursor.execute('UPDATE training_samples SET is_test = 0')

        # Mark test samples
        if test_ids:
            placeholders = ','.join('?' * len(test_ids))
            cursor.execute(f'UPDATE training_samples SET is_test = 1 WHERE id IN ({placeholders})', test_ids)

        conn.commit()
        conn.close()
