"""
Model Comparison Script.

Trains and compares all ML models (Gradient Boosting, CNN, GNN, Transformer)
on the same training data to determine which performs best.

Usage:
    python -m shapez2_solver.evolution.model_comparison --problems 100 --epochs 50
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .problem_generator import ProblemGenerator
from .training_runner import TrainingRunner, extract_specs_from_problem, SolveResult
from .foundation_config import FOUNDATION_SPECS, Side
from .cpsat_solver import CPSATFullSolver
from .evaluation import PlacementInfo, DefaultSolutionEvaluator
from .ml_evaluators import MLEvaluatorSystem, MLPlacementEvaluator
from .advanced_ml_models import (
    AdvancedMLSystem,
    CNNPlacementEvaluator,
    GNNPlacementEvaluator,
    TransformerPlacementEvaluator,
)
from ..blueprint.building_types import BuildingType, Rotation


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_inference_time_ms: float
    training_time_s: float
    num_train_samples: int
    num_test_samples: int


@dataclass
class ComparisonReport:
    """Full comparison report."""
    timestamp: str
    num_problems: int
    num_train: int
    num_test: int
    solve_success_rate: float
    routing_success_rate: float
    models: List[ModelMetrics]
    winner: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "num_problems": self.num_problems,
            "num_train": self.num_train,
            "num_test": self.num_test,
            "solve_success_rate": self.solve_success_rate,
            "routing_success_rate": self.routing_success_rate,
            "models": [asdict(m) for m in self.models],
            "winner": self.winner,
        }

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "MODEL COMPARISON REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Total problems: {self.num_problems}",
            f"Train/Test split: {self.num_train}/{self.num_test}",
            f"Solve success rate: {self.solve_success_rate*100:.1f}%",
            f"Routing success rate: {self.routing_success_rate*100:.1f}%",
            "",
            f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Infer(ms)':>10}",
            "-" * 70,
        ]

        for m in self.models:
            lines.append(
                f"{m.model_name:<20} {m.accuracy*100:>9.1f}% {m.precision*100:>9.1f}% "
                f"{m.recall*100:>9.1f}% {m.f1_score*100:>9.1f}% {m.avg_inference_time_ms:>9.2f}"
            )

        lines.extend([
            "-" * 70,
            f"WINNER: {self.winner}",
            "=" * 70,
        ])

        return "\n".join(lines)


class ModelComparer:
    """Compares multiple ML models for placement evaluation."""

    def __init__(
        self,
        db_path: str = "comparison_training.db",
        model_dir: str = "models/comparison",
    ):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Training data storage
        self.training_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []

    def collect_training_data(
        self,
        num_problems: int = 100,
        time_limit: float = 60.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Generate problems and collect training data."""
        generator = ProblemGenerator(seed=seed)

        if verbose:
            print(f"Generating {num_problems} problems (may be more with coverage guarantee)...")

        problems = generator.generate_training_set(
            num_problems=num_problems,
            ensure_coverage=True,
        )

        if verbose:
            print(f"Generated {len(problems)} problems (coverage ensures min ~25)")
            print(f"Solving problems to collect training data...")

        all_samples = []
        solved_count = 0
        routed_count = 0

        for i, problem in enumerate(problems):
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{num_problems}] Solved: {solved_count}, Routed: {routed_count}")

            try:
                foundation_name = problem.get("foundation", "2x2")
                input_specs, output_specs = extract_specs_from_problem(problem)

                if not input_specs or not output_specs:
                    continue

                # Create solver
                solver = CPSATFullSolver(
                    foundation_type=foundation_name,
                    input_specs=input_specs,
                    output_specs=output_specs,
                    time_limit_seconds=time_limit,
                    enable_placement_feedback=False,
                    enable_transformer_logging=False,
                    enable_logging=False,
                )

                solution = solver.solve(verbose=False)

                if solution:
                    solved_count += 1
                    if solution.routing_success:
                        routed_count += 1

                    spec = FOUNDATION_SPECS[foundation_name]

                    # Convert solution to training sample
                    machines = [
                        PlacementInfo(
                            building_type=bt,
                            x=x, y=y, floor=f,
                            rotation=rot
                        )
                        for bt, x, y, f, rot in solution.machines
                    ]

                    side_map = {'N': Side.NORTH, 'S': Side.SOUTH, 'E': Side.EAST, 'W': Side.WEST}
                    input_positions = [
                        (spec.get_port_grid_position(side_map[s], p)[0],
                         spec.get_port_grid_position(side_map[s], p)[1], f)
                        for s, p, f, _ in input_specs
                    ]
                    output_positions = [
                        (spec.get_port_grid_position(side_map[s], p)[0],
                         spec.get_port_grid_position(side_map[s], p)[1], f)
                        for s, p, f, _ in output_specs
                    ]

                    sample = {
                        'machines': machines,
                        'grid_width': spec.grid_width,
                        'grid_height': spec.grid_height,
                        'num_floors': spec.num_floors,
                        'input_positions': input_positions,
                        'output_positions': output_positions,
                        'routing_success': solution.routing_success,
                        'problem_name': problem.get('name', f'problem_{i}'),
                    }
                    all_samples.append(sample)

            except Exception as e:
                if verbose:
                    print(f"  Error on problem {i}: {e}")
                continue

        # Split into train/test (80/20)
        np.random.seed(seed or 42)
        np.random.shuffle(all_samples)

        split_idx = int(len(all_samples) * 0.8)
        self.training_samples = all_samples[:split_idx]
        self.test_samples = all_samples[split_idx:]

        actual_problems = len(problems)
        stats = {
            "total_problems": actual_problems,
            "solved": solved_count,
            "routed": routed_count,
            "solve_rate": solved_count / actual_problems if actual_problems > 0 else 0,
            "routing_rate": routed_count / solved_count if solved_count > 0 else 0,
            "train_samples": len(self.training_samples),
            "test_samples": len(self.test_samples),
        }

        if verbose:
            print(f"\nData collection complete:")
            print(f"  Solved: {solved_count}/{actual_problems} ({stats['solve_rate']*100:.1f}%)")
            print(f"  Routed: {routed_count}/{solved_count} ({stats['routing_rate']*100:.1f}%)")
            print(f"  Train samples: {len(self.training_samples)}")
            print(f"  Test samples: {len(self.test_samples)}")

        return stats

    def _train_gradient_boosting(self, epochs: int, verbose: bool) -> Tuple[MLPlacementEvaluator, Dict]:
        """Train gradient boosting model."""
        if verbose:
            print("\nTraining Gradient Boosting model...")

        model = MLPlacementEvaluator(
            model_path=str(self.model_dir / "gb_placement.pkl"),
            db_path=self.db_path,
            collect_training_data=True,
        )

        # Record training samples
        start = time.time()
        for sample in self.training_samples:
            model.record_outcome(
                machines=sample['machines'],
                grid_width=sample['grid_width'],
                grid_height=sample['grid_height'],
                num_floors=sample['num_floors'],
                input_positions=sample['input_positions'],
                output_positions=sample['output_positions'],
                routing_success=sample['routing_success'],
            )

        # MLPlacementEvaluator.train() returns bool, not dict
        trained = model.train(min_samples=5)
        train_time = time.time() - start

        if verbose:
            print(f"  Trained: {trained}, Samples: {len(self.training_samples)}")

        return model, {"trained": trained, "train_time": train_time}

    def _train_cnn(self, epochs: int, verbose: bool) -> Tuple[CNNPlacementEvaluator, Dict]:
        """Train CNN model."""
        if verbose:
            print("\nTraining CNN model...")

        model = CNNPlacementEvaluator(
            model_path=str(self.model_dir / "cnn_placement.pt"),
            db_path=self.db_path,
            collect_training_data=True,
        )

        # Record training samples
        for sample in self.training_samples:
            model.record_outcome(
                machines=sample['machines'],
                grid_width=sample['grid_width'],
                grid_height=sample['grid_height'],
                num_floors=sample['num_floors'],
                input_positions=sample['input_positions'],
                output_positions=sample['output_positions'],
                routing_success=sample['routing_success'],
            )

        start = time.time()
        result = model.train(epochs=epochs)
        train_time = time.time() - start

        if verbose:
            print(f"  Trained: {result.get('trained', False)}, Loss: {result.get('final_loss', 'N/A')}")

        return model, {"trained": result.get('trained', False), "train_time": train_time}

    def _train_gnn(self, epochs: int, verbose: bool) -> Tuple[GNNPlacementEvaluator, Dict]:
        """Train GNN model."""
        if verbose:
            print("\nTraining GNN model...")

        model = GNNPlacementEvaluator(
            model_path=str(self.model_dir / "gnn_placement.pt"),
            db_path=self.db_path,
            collect_training_data=True,
        )

        # Record training samples
        for sample in self.training_samples:
            model.record_outcome(
                machines=sample['machines'],
                grid_width=sample['grid_width'],
                grid_height=sample['grid_height'],
                num_floors=sample['num_floors'],
                input_positions=sample['input_positions'],
                output_positions=sample['output_positions'],
                routing_success=sample['routing_success'],
            )

        start = time.time()
        result = model.train(epochs=epochs)
        train_time = time.time() - start

        if verbose:
            print(f"  Trained: {result.get('trained', False)}, Loss: {result.get('final_loss', 'N/A')}")

        return model, {"trained": result.get('trained', False), "train_time": train_time}

    def _train_transformer(self, epochs: int, verbose: bool) -> Tuple[TransformerPlacementEvaluator, Dict]:
        """Train Transformer model."""
        if verbose:
            print("\nTraining Transformer model...")

        model = TransformerPlacementEvaluator(
            model_path=str(self.model_dir / "transformer_placement.pt"),
            db_path=self.db_path,
            collect_training_data=True,
        )

        # Record training samples
        for sample in self.training_samples:
            model.record_outcome(
                machines=sample['machines'],
                grid_width=sample['grid_width'],
                grid_height=sample['grid_height'],
                num_floors=sample['num_floors'],
                input_positions=sample['input_positions'],
                output_positions=sample['output_positions'],
                routing_success=sample['routing_success'],
            )

        start = time.time()
        result = model.train(epochs=epochs)
        train_time = time.time() - start

        if verbose:
            print(f"  Trained: {result.get('trained', False)}, Loss: {result.get('final_loss', 'N/A')}")

        return model, {"trained": result.get('trained', False), "train_time": train_time}

    def _evaluate_model(
        self,
        model,
        model_name: str,
        train_info: Dict,
        verbose: bool,
    ) -> ModelMetrics:
        """Evaluate a model on test data."""
        predictions = []
        labels = []
        times = []

        for sample in self.test_samples:
            start = time.perf_counter()
            try:
                score, _ = model.evaluate(
                    machines=sample['machines'],
                    grid_width=sample['grid_width'],
                    grid_height=sample['grid_height'],
                    num_floors=sample['num_floors'],
                    input_positions=sample['input_positions'],
                    output_positions=sample['output_positions'],
                )
            except Exception as e:
                if verbose:
                    print(f"  Eval error for {model_name}: {e}")
                score = 0.5
            elapsed = (time.perf_counter() - start) * 1000

            predictions.append(1 if score >= 0.5 else 0)
            labels.append(1 if sample['routing_success'] else 0)
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

        return ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_inference_time_ms=np.mean(times),
            training_time_s=train_info.get('train_time', 0),
            num_train_samples=len(self.training_samples),
            num_test_samples=len(self.test_samples),
        )

    def run_comparison(
        self,
        num_problems: int = 100,
        epochs: int = 50,
        time_limit: float = 60.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> ComparisonReport:
        """Run full model comparison."""

        # Collect training data
        data_stats = self.collect_training_data(
            num_problems=num_problems,
            time_limit=time_limit,
            seed=seed,
            verbose=verbose,
        )

        if len(self.training_samples) < 10:
            raise ValueError(f"Insufficient training data: {len(self.training_samples)} samples")

        # Train all models
        models_info = []

        # 1. Gradient Boosting
        try:
            gb_model, gb_info = self._train_gradient_boosting(epochs, verbose)
            if gb_info['trained']:
                metrics = self._evaluate_model(gb_model, "GradientBoosting", gb_info, verbose)
                models_info.append(metrics)
        except Exception as e:
            if verbose:
                print(f"  GB training failed: {e}")

        # 2. CNN
        try:
            cnn_model, cnn_info = self._train_cnn(epochs, verbose)
            if cnn_info['trained']:
                metrics = self._evaluate_model(cnn_model, "CNN", cnn_info, verbose)
                models_info.append(metrics)
        except Exception as e:
            if verbose:
                print(f"  CNN training failed: {e}")

        # 3. GNN
        try:
            gnn_model, gnn_info = self._train_gnn(epochs, verbose)
            if gnn_info['trained']:
                metrics = self._evaluate_model(gnn_model, "GNN", gnn_info, verbose)
                models_info.append(metrics)
        except Exception as e:
            if verbose:
                print(f"  GNN training failed: {e}")

        # 4. Transformer
        try:
            transformer_model, transformer_info = self._train_transformer(epochs, verbose)
            if transformer_info['trained']:
                metrics = self._evaluate_model(transformer_model, "Transformer", transformer_info, verbose)
                models_info.append(metrics)
        except Exception as e:
            if verbose:
                print(f"  Transformer training failed: {e}")

        # Sort by F1 score
        models_info.sort(key=lambda m: m.f1_score, reverse=True)

        winner = models_info[0].model_name if models_info else "None"

        report = ComparisonReport(
            timestamp=datetime.now().isoformat(),
            num_problems=num_problems,
            num_train=len(self.training_samples),
            num_test=len(self.test_samples),
            solve_success_rate=data_stats['solve_rate'],
            routing_success_rate=data_stats['routing_rate'],
            models=models_info,
            winner=winner,
        )

        if verbose:
            print(f"\n{report.summary()}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Compare ML models for placement evaluation")
    parser.add_argument("--problems", type=int, default=100, help="Number of problems to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for neural networks")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Per-problem solve time limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for report")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    comparer = ModelComparer()

    report = comparer.run_comparison(
        num_problems=args.problems,
        epochs=args.epochs,
        time_limit=args.time_limit,
        seed=args.seed,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
