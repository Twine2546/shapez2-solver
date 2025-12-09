#!/usr/bin/env python3
"""
Continuous Learning for Shapez 2 Routing.

This module provides:
1. Automatic logging of routing attempts to the training database
2. Incremental model retraining based on new data
3. Model performance tracking over time

Usage:
    # Retrain models with latest data
    python -m shapez2_solver.learning.continuous_learning --retrain

    # Log a routing attempt
    python -m shapez2_solver.learning.continuous_learning --log-problem problem.json

    # Show training stats
    python -m shapez2_solver.learning.continuous_learning --stats
"""

import argparse
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Imports
try:
    from .synthetic_generator import (
        SyntheticProblem, SolverResult, SyntheticDataStore, solve_problem
    )
    from .ml_models import (
        SolvabilityClassifier, DirectionPredictor, MLGuidedRouter
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shapez2_solver.learning.synthetic_generator import (
        SyntheticProblem, SolverResult, SyntheticDataStore, solve_problem
    )
    from shapez2_solver.learning.ml_models import (
        SolvabilityClassifier, DirectionPredictor, MLGuidedRouter
    )


class ContinuousLearner:
    """
    Manages continuous learning for routing models.

    Tracks training history and enables incremental retraining.
    """

    def __init__(
        self,
        db_path: str = "synthetic_training.db",
        model_dir: str = None,
    ):
        self.db_path = db_path

        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self.model_dir.mkdir(exist_ok=True)

        # Model paths
        self.solvability_path = self.model_dir / "solvability_classifier.pkl"
        self.direction_path = self.model_dir / "direction_predictor.pkl"
        self.history_path = self.model_dir / "training_history.json"

        # Load training history
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        """Load training history."""
        if self.history_path.exists():
            with open(self.history_path) as f:
                return json.load(f)
        return {
            'trainings': [],
            'last_problem_count': 0,
            'created_at': datetime.now().isoformat(),
        }

    def _save_history(self):
        """Save training history."""
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        store = SyntheticDataStore(self.db_path)
        stats = store.get_stats()

        # Check for new data since last training
        last_count = self.history.get('last_problem_count', 0)
        new_problems = stats['total_problems'] - last_count

        # Calculate success rate by routing_progress
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT AVG(routing_progress), COUNT(*)
            FROM solver_results
            WHERE routing_progress > 0 AND routing_progress < 1.0
        """)
        row = cursor.fetchone()
        partial_avg = row[0] or 0
        partial_count = row[1] or 0

        conn.close()

        return {
            'total_problems': stats['total_problems'],
            'total_solved': stats['total_solved'],
            'successes': stats['successes'],
            'success_rate': stats['successes'] / max(1, stats['total_solved']),
            'avg_solve_time': stats['avg_solve_time'],
            'new_since_training': new_problems,
            'partial_successes': partial_count,
            'avg_partial_progress': partial_avg,
            'by_difficulty': stats.get('results_by_difficulty', {}),
            'last_training': self.history['trainings'][-1] if self.history['trainings'] else None,
        }

    def should_retrain(self, min_new_problems: int = 50) -> Tuple[bool, str]:
        """
        Determine if models should be retrained.

        Returns:
            (should_retrain, reason)
        """
        stats = self.get_training_stats()

        # Check if enough new data
        if stats['new_since_training'] >= min_new_problems:
            return True, f"{stats['new_since_training']} new problems since last training"

        # Check if no models exist
        if not self.solvability_path.exists():
            return True, "No solvability model exists"

        if not self.direction_path.exists():
            return True, "No direction model exists"

        return False, f"Only {stats['new_since_training']} new problems (need {min_new_problems})"

    def retrain(self, force: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """
        Retrain models with latest data.

        Args:
            force: Retrain even if not enough new data
            verbose: Print progress

        Returns:
            Training metrics
        """
        should, reason = self.should_retrain()

        if not should and not force:
            if verbose:
                print(f"Skipping retraining: {reason}")
            return {'skipped': True, 'reason': reason}

        if verbose:
            print(f"Retraining models: {reason}")
            print("=" * 60)

        results = {'skipped': False, 'started_at': datetime.now().isoformat()}

        # Train solvability classifier
        if verbose:
            print("\n[Phase 1] Training Solvability Classifier...")

        try:
            classifier = SolvabilityClassifier()
            metrics = classifier.train(self.db_path)
            classifier.save(str(self.solvability_path))

            results['solvability'] = {
                'accuracy': metrics.get('solvability_accuracy', 0),
                'difficulty_accuracy': metrics.get('difficulty_accuracy', 0),
            }

            if verbose:
                print(f"  Solvability accuracy: {metrics.get('solvability_accuracy', 0):.3f}")
                print(f"  Difficulty accuracy: {metrics.get('difficulty_accuracy', 0):.3f}")

        except Exception as e:
            results['solvability'] = {'error': str(e)}
            if verbose:
                print(f"  Error: {e}")

        # Train direction predictor
        if verbose:
            print("\n[Phase 2] Training Direction Predictor...")

        try:
            predictor = DirectionPredictor()
            metrics = predictor.train(self.db_path)

            if metrics:
                predictor.save(str(self.direction_path))
                results['direction'] = {
                    'accuracy': metrics.get('accuracy', 0),
                }
                if verbose:
                    print(f"  Direction accuracy: {metrics.get('accuracy', 0):.3f}")
            else:
                results['direction'] = {'error': 'No training data'}
                if verbose:
                    print("  No training data available")

        except Exception as e:
            results['direction'] = {'error': str(e)}
            if verbose:
                print(f"  Error: {e}")

        # Update history
        stats = self.get_training_stats()
        results['finished_at'] = datetime.now().isoformat()
        results['problem_count'] = stats['total_problems']

        self.history['trainings'].append({
            'timestamp': datetime.now().isoformat(),
            'problem_count': stats['total_problems'],
            'results': results,
        })
        self.history['last_problem_count'] = stats['total_problems']
        self._save_history()

        if verbose:
            print("\n" + "=" * 60)
            print("Training complete!")
            print(f"Models saved to: {self.model_dir}")

        return results

    def log_routing_attempt(
        self,
        problem_dict: Dict[str, Any],
        success: bool,
        solve_time: float,
        num_belts: int,
        connections_routed: int = 0,
        total_connections: int = 0,
        partial_belts: List = None,
        error_message: str = "",
    ):
        """
        Log a routing attempt to the training database.

        This can be called from the solver to continuously collect training data.
        """
        store = SyntheticDataStore(self.db_path)

        # Generate problem ID if not present
        problem_id = problem_dict.get('problem_id', f"runtime_{int(time.time()*1000)}")

        # Create problem entry
        problem = SyntheticProblem(
            problem_id=problem_id,
            foundation_type=problem_dict.get('foundation_type', 'unknown'),
            grid_width=problem_dict.get('grid_width', 10),
            grid_height=problem_dict.get('grid_height', 10),
            num_floors=problem_dict.get('num_floors', 1),
            machines=problem_dict.get('machines', []),
            connections=problem_dict.get('connections', []),
            input_positions=problem_dict.get('input_positions', []),
            output_positions=problem_dict.get('output_positions', []),
            difficulty=problem_dict.get('difficulty', 'unknown'),
        )

        store.save_problem(problem)

        # Calculate routing progress
        if total_connections > 0:
            routing_progress = connections_routed / total_connections
        else:
            routing_progress = 1.0 if success else 0.0

        # Create result entry
        result = SolverResult(
            problem_id=problem_id,
            success=success,
            solve_time=solve_time,
            num_belts=num_belts,
            error_message=error_message,
            connections_attempted=total_connections,
            connections_routed=connections_routed,
            routing_progress=routing_progress,
            failed_connection_indices="[]",
            partial_belt_positions=json.dumps(partial_belts or []),
            solver_iterations=0,
            best_objective=float(num_belts),
        )

        store.save_result(result)

        return problem_id


def main():
    parser = argparse.ArgumentParser(
        description="Continuous learning for routing models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show training stats
    python -m shapez2_solver.learning.continuous_learning --stats

    # Retrain models with latest data
    python -m shapez2_solver.learning.continuous_learning --retrain

    # Force retrain even with little new data
    python -m shapez2_solver.learning.continuous_learning --retrain --force
        """
    )

    parser.add_argument("--db", type=str, default="synthetic_training.db",
                       help="Path to training database")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Directory for model files")

    parser.add_argument("--stats", action="store_true",
                       help="Show training statistics")
    parser.add_argument("--retrain", action="store_true",
                       help="Retrain models with latest data")
    parser.add_argument("--force", action="store_true",
                       help="Force retrain even if not enough new data")
    parser.add_argument("--check", action="store_true",
                       help="Check if retraining is recommended")

    args = parser.parse_args()

    learner = ContinuousLearner(db_path=args.db, model_dir=args.model_dir)

    if args.stats:
        stats = learner.get_training_stats()

        print("=" * 60)
        print("Training Statistics")
        print("=" * 60)
        print(f"\nDatabase: {args.db}")
        print(f"Models: {learner.model_dir}")
        print(f"\nTotal problems: {stats['total_problems']}")
        print(f"Total solved: {stats['total_solved']}")
        print(f"Successes: {stats['successes']} ({stats['success_rate']*100:.1f}%)")
        print(f"Avg solve time: {stats['avg_solve_time']:.2f}s")
        print(f"\nPartial successes: {stats['partial_successes']}")
        print(f"Avg partial progress: {stats['avg_partial_progress']*100:.1f}%")
        print(f"\nNew since training: {stats['new_since_training']}")

        if stats['last_training']:
            print(f"\nLast training: {stats['last_training']['timestamp']}")

        print("\nBy difficulty:")
        for diff, data in stats['by_difficulty'].items():
            rate = 100 * data.get('successes', 0) / max(1, data.get('total', 1))
            print(f"  {diff}: {data.get('total', 0)} problems, {rate:.1f}% success")

    if args.check:
        should, reason = learner.should_retrain()
        print(f"Should retrain: {should}")
        print(f"Reason: {reason}")

    if args.retrain:
        learner.retrain(force=args.force)

    if not any([args.stats, args.check, args.retrain]):
        parser.print_help()


if __name__ == "__main__":
    main()
