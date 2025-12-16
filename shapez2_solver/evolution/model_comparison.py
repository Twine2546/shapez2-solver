"""
Model Comparison Script.

Trains and compares ML models for BOTH placement and routing:
- Placement ML: Gradient Boosting, CNN, GNN, Transformer
- Routing ML: CellValuePredictor, PathOrderingPredictor

Compares 4 configurations:
1. Baseline (no ML)
2. Placement ML only
3. Routing ML only
4. Both (placement + routing ML)

Usage:
    python -m shapez2_solver.evolution.model_comparison --problems 100 --epochs 50
    python -m shapez2_solver.evolution.model_comparison --problems 100 --ab-test  # Run A/B comparison
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class InsufficientMemoryError(Exception):
    """Raised when available memory drops below threshold."""
    pass


def check_memory(min_available_mb: int = 500, verbose: bool = False) -> bool:
    """
    Check if there's sufficient available memory.

    Args:
        min_available_mb: Minimum required available memory in MB
        verbose: Print memory stats

    Returns:
        True if sufficient memory, False otherwise

    Raises:
        InsufficientMemoryError: If memory is critically low
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value_kb = int(parts[1])
                    meminfo[key] = value_kb

        # Get available memory (MemAvailable is more accurate than MemFree)
        available_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
        available_mb = available_kb // 1024
        total_mb = meminfo.get('MemTotal', 0) // 1024

        if verbose:
            print(f"  Memory: {available_mb}MB available / {total_mb}MB total")

        if available_mb < min_available_mb:
            raise InsufficientMemoryError(
                f"Insufficient memory: {available_mb}MB available, "
                f"minimum {min_available_mb}MB required. "
                f"Consider reducing --problems or freeing memory."
            )

        return True

    except FileNotFoundError:
        # Not on Linux, skip memory check
        return True
    except Exception as e:
        if isinstance(e, InsufficientMemoryError):
            raise
        # Other errors, skip memory check
        return True


def get_memory_usage_mb() -> int:
    """Get current process memory usage in MB."""
    try:
        with open(f'/proc/{os.getpid()}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0

from .databases import TrainingSampleDB
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
from .ml_routing import (
    EnhancedRoutingMLSystem,
    RoutingDataStore,
    RoutingOutcome,
    create_routing_trainer,
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
        samples_db_path: str = "training_samples.db",
        model_dir: str = "models/comparison",
    ):
        self.db_path = db_path
        self.samples_db = TrainingSampleDB(samples_db_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache (loaded from DB)
        self.training_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []

    def collect_training_data(
        self,
        num_problems: int = 100,
        time_limit: float = 60.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        resume: bool = True,
        fresh_start: bool = False,
    ) -> Dict[str, Any]:
        """Generate problems and collect training data with database checkpointing.

        Args:
            num_problems: Number of problems to generate
            time_limit: Time limit per problem solve
            seed: Random seed for reproducibility
            verbose: Print progress
            resume: Resume from previous run if available
            fresh_start: Clear existing samples and start fresh
        """
        run_id = f"run_{num_problems}_{seed or 'none'}"

        if fresh_start:
            if verbose:
                print("Fresh start requested - clearing existing samples...")
            self.samples_db.clear_samples()

        # Check for existing progress
        existing_samples = self.samples_db.get_sample_count()
        if existing_samples > 0 and resume and not fresh_start:
            if verbose:
                print(f"Found {existing_samples} existing samples in database")
                print("Use --fresh to start over, or continuing with existing data...")
            # Just load existing data and proceed to training
            self.samples_db.assign_test_split(test_ratio=0.2, seed=seed or 42)
            self.training_samples = self.samples_db.get_samples(is_test=False)
            self.test_samples = self.samples_db.get_samples(is_test=True)

            # Get stats from existing data
            total_samples = len(self.training_samples) + len(self.test_samples)
            routed = sum(1 for s in self.training_samples + self.test_samples if s['routing_success'])

            stats = {
                "total_problems": total_samples,
                "solved": total_samples,
                "routed": routed,
                "solve_rate": 1.0,
                "routing_rate": routed / total_samples if total_samples > 0 else 0,
                "train_samples": len(self.training_samples),
                "test_samples": len(self.test_samples),
                "resumed": True,
            }

            if verbose:
                print(f"Loaded {total_samples} samples from database")
                print(f"  Training: {len(self.training_samples)}, Test: {len(self.test_samples)}")
                print(f"  Routing success rate: {stats['routing_rate']*100:.1f}%")

            return stats

        generator = ProblemGenerator(seed=seed)

        if verbose:
            print(f"Generating {num_problems} problems (may be more with coverage guarantee)...")

        problems = generator.generate_training_set(
            num_problems=num_problems,
            ensure_coverage=True,
        )

        actual_problems = len(problems)
        if verbose:
            print(f"Generated {actual_problems} problems (coverage ensures min ~25)")
            print(f"Solving problems and saving to database...")

        solved_count = 0
        routed_count = 0

        for i, problem in enumerate(problems):
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{actual_problems}] Solved: {solved_count}, Routed: {routed_count}")
                # Update progress in database
                self.samples_db.update_progress(run_id, actual_problems, i + 1, solved_count, routed_count)

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
                        'foundation': foundation_name,
                    }

                    # Save to database immediately
                    self.samples_db.add_sample(sample)

            except Exception as e:
                if verbose:
                    print(f"  Error on problem {i}: {e}")
                continue

        # Final progress update
        self.samples_db.update_progress(run_id, actual_problems, actual_problems, solved_count, routed_count)

        # Assign train/test split in database
        self.samples_db.assign_test_split(test_ratio=0.2, seed=seed or 42)

        # Load samples from database
        self.training_samples = self.samples_db.get_samples(is_test=False)
        self.test_samples = self.samples_db.get_samples(is_test=True)

        stats = {
            "total_problems": actual_problems,
            "solved": solved_count,
            "routed": routed_count,
            "solve_rate": solved_count / actual_problems if actual_problems > 0 else 0,
            "routing_rate": routed_count / solved_count if solved_count > 0 else 0,
            "train_samples": len(self.training_samples),
            "test_samples": len(self.test_samples),
            "resumed": False,
        }

        if verbose:
            print(f"\nData collection complete:")
            print(f"  Solved: {solved_count}/{actual_problems} ({stats['solve_rate']*100:.1f}%)")
            print(f"  Routed: {routed_count}/{solved_count} ({stats['routing_rate']*100:.1f}%)")
            print(f"  Train samples: {len(self.training_samples)}")
            print(f"  Test samples: {len(self.test_samples)}")
            print(f"  Samples saved to: {self.samples_db.db_path}")

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
        fresh_start: bool = False,
    ) -> ComparisonReport:
        """Run full model comparison."""

        # Collect training data
        data_stats = self.collect_training_data(
            num_problems=num_problems,
            time_limit=time_limit,
            seed=seed,
            verbose=verbose,
            fresh_start=fresh_start,
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


@dataclass
class ConfigurationResult:
    """Result of testing a single configuration."""
    config_name: str
    problems_tested: int
    solved: int
    routed: int
    solve_rate: float
    routing_rate: float
    avg_solve_time: float
    uses_placement_ml: bool
    uses_routing_ml: bool


@dataclass
class ABTestReport:
    """Full A/B test report comparing configurations."""
    timestamp: str
    num_problems: int
    training_problems: int
    test_problems: int
    configurations: List[ConfigurationResult]
    winner: str
    improvement_over_baseline: float

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "A/B TEST REPORT: PLACEMENT ML vs ROUTING ML",
            "=" * 80,
            f"Timestamp: {self.timestamp}",
            f"Training problems: {self.training_problems}",
            f"Test problems: {self.test_problems}",
            "",
            f"{'Configuration':<25} {'Solved':>8} {'Routed':>8} {'Route%':>10} {'AvgTime':>10}",
            "-" * 80,
        ]

        for cfg in self.configurations:
            ml_flags = ""
            if cfg.uses_placement_ml:
                ml_flags += "P"
            if cfg.uses_routing_ml:
                ml_flags += "R"
            if not ml_flags:
                ml_flags = "-"

            lines.append(
                f"{cfg.config_name:<25} {cfg.solved:>8}/{cfg.problems_tested:<4} "
                f"{cfg.routed:>8} {cfg.routing_rate*100:>9.1f}% {cfg.avg_solve_time:>9.2f}s"
            )

        lines.extend([
            "-" * 80,
            f"WINNER: {self.winner}",
            f"Improvement over baseline: {self.improvement_over_baseline:+.1f}%",
            "=" * 80,
        ])

        return "\n".join(lines)


class ABTester:
    """
    A/B Testing framework for comparing ML configurations.

    Tests 4 configurations:
    1. Baseline - No ML
    2. Placement ML - Uses best placement evaluator
    3. Routing ML - Uses routing heuristic/move cost
    4. Combined - Uses both
    """

    def __init__(
        self,
        db_path: str = "ab_test.db",
        model_dir: str = "models/ab_test",
    ):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ML Systems
        self.placement_evaluator = None
        self.routing_ml = None

    def collect_and_train(
        self,
        num_problems: int = 100,
        epochs: int = 30,
        time_limit: float = 30.0,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Collect training data and train both placement and routing ML.
        """
        generator = ProblemGenerator(seed=seed)

        if verbose:
            print(f"Generating {num_problems} training problems...")

        problems = generator.generate_training_set(num_problems=num_problems, ensure_coverage=True)

        if verbose:
            print(f"Generated {len(problems)} problems")
            print("Collecting training data...")

        # Initialize ML systems
        placement_db = str(self.model_dir / "placement.db")
        routing_db = str(self.model_dir / "routing.db")

        self.placement_evaluator = CNNPlacementEvaluator(
            model_path=str(self.model_dir / "placement_cnn.pt"),
            db_path=placement_db,
            collect_training_data=True,
        )

        self.routing_ml = EnhancedRoutingMLSystem(
            model_dir=str(self.model_dir),
            db_path=routing_db,
            collect_training_data=True,
        )

        solved_count = 0
        routed_count = 0

        for i, problem in enumerate(problems):
            if verbose and (i + 1) % 10 == 0:
                mem_mb = get_memory_usage_mb()
                print(f"  [{i+1}/{len(problems)}] Solved: {solved_count}, Routed: {routed_count} (Mem: {mem_mb}MB)")

            # Check memory every 100 problems
            if (i + 1) % 100 == 0:
                check_memory(min_available_mb=500, verbose=verbose)

            try:
                foundation_name = problem.get("foundation", "2x2")
                input_specs, output_specs = extract_specs_from_problem(problem)

                if not input_specs or not output_specs:
                    continue

                spec = FOUNDATION_SPECS[foundation_name]

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

                    # Convert to PlacementInfo
                    machines = [
                        PlacementInfo(building_type=bt, x=x, y=y, floor=f, rotation=rot)
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

                    # Record placement training data
                    self.placement_evaluator.record_outcome(
                        machines=machines,
                        grid_width=spec.grid_width,
                        grid_height=spec.grid_height,
                        num_floors=spec.num_floors,
                        input_positions=input_positions,
                        output_positions=output_positions,
                        routing_success=solution.routing_success,
                    )

                    # Record routing training data
                    cells_used = {(b[0], b[1], b[2]) for b in solution.belts} if solution.belts else set()
                    routing_outcome = RoutingOutcome(
                        grid_width=spec.grid_width,
                        grid_height=spec.grid_height,
                        num_floors=spec.num_floors,
                        connections=[],
                        connection_order=[],
                        paths=[],
                        success=solution.routing_success,
                        failed_at_connection=None,
                        cells_used=cells_used,
                    )
                    self.routing_ml.record_outcome(routing_outcome)

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                continue

        if verbose:
            print(f"\nData collection complete:")
            print(f"  Solved: {solved_count}/{len(problems)}")
            print(f"  Routed: {routed_count}/{solved_count}")

        # Train models
        if verbose:
            print("\nTraining placement ML...")

        placement_result = self.placement_evaluator.train(epochs=epochs)

        if verbose:
            print(f"  Placement: {placement_result}")
            print("\nTraining routing ML...")

        routing_result = self.routing_ml.train_all(epochs=epochs)

        if verbose:
            print(f"  Routing: {routing_result}")

        return {
            "problems": len(problems),
            "solved": solved_count,
            "routed": routed_count,
            "placement_trained": placement_result,
            "routing_trained": routing_result,
        }

    def run_ab_test(
        self,
        num_test_problems: int = 50,
        time_limit: float = 30.0,
        seed: int = 12345,
        verbose: bool = True,
    ) -> ABTestReport:
        """
        Run A/B test comparing 4 configurations on fresh problems.
        """
        generator = ProblemGenerator(seed=seed)

        if verbose:
            print(f"\nGenerating {num_test_problems} TEST problems (seed={seed})...")

        test_problems = generator.generate_training_set(num_problems=num_test_problems, ensure_coverage=True)

        if verbose:
            print(f"Generated {len(test_problems)} test problems")

        # Define configurations to test
        configurations = [
            ("Baseline", False, False),
            ("Placement ML Only", True, False),
            ("Routing ML Only", False, True),
            ("Combined (P+R)", True, True),
        ]

        results = []

        for config_name, use_placement, use_routing in configurations:
            if verbose:
                print(f"\n=== Testing: {config_name} ===")

            cfg_result = self._test_configuration(
                problems=test_problems,
                use_placement_ml=use_placement,
                use_routing_ml=use_routing,
                time_limit=time_limit,
                verbose=verbose,
            )

            cfg_result.config_name = config_name
            results.append(cfg_result)

            if verbose:
                print(f"  Result: {cfg_result.routed}/{cfg_result.solved} routed "
                      f"({cfg_result.routing_rate*100:.1f}%)")

        # Find winner (highest routing rate)
        results.sort(key=lambda r: r.routing_rate, reverse=True)
        winner = results[0].config_name

        # Calculate improvement over baseline
        baseline = next(r for r in results if r.config_name == "Baseline")
        improvement = (results[0].routing_rate - baseline.routing_rate) * 100

        report = ABTestReport(
            timestamp=datetime.now().isoformat(),
            num_problems=len(test_problems),
            training_problems=0,  # Set by caller
            test_problems=len(test_problems),
            configurations=results,
            winner=winner,
            improvement_over_baseline=improvement,
        )

        if verbose:
            print(f"\n{report.summary()}")

        return report

    def _test_configuration(
        self,
        problems: List[dict],
        use_placement_ml: bool,
        use_routing_ml: bool,
        time_limit: float,
        verbose: bool,
    ) -> ConfigurationResult:
        """Test a single configuration on problems."""
        solved = 0
        routed = 0
        total_time = 0.0

        for i, problem in enumerate(problems):
            if verbose and (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(problems)}] Solved: {solved}, Routed: {routed}")

            try:
                foundation_name = problem.get("foundation", "2x2")
                input_specs, output_specs = extract_specs_from_problem(problem)

                if not input_specs or not output_specs:
                    continue

                # Set up evaluators based on configuration
                placement_eval = None
                routing_heuristic = None
                routing_move_cost = None

                if use_placement_ml and self.placement_evaluator:
                    placement_eval = self.placement_evaluator

                if use_routing_ml and self.routing_ml:
                    routing_heuristic = self.routing_ml.heuristic
                    routing_move_cost = self.routing_ml.move_cost

                solver = CPSATFullSolver(
                    foundation_type=foundation_name,
                    input_specs=input_specs,
                    output_specs=output_specs,
                    time_limit_seconds=time_limit,
                    enable_placement_feedback=False,
                    enable_transformer_logging=False,
                    enable_logging=False,
                    placement_evaluator=placement_eval,
                    routing_heuristic=routing_heuristic,
                    move_cost_function=routing_move_cost,
                )

                start = time.time()
                solution = solver.solve(verbose=False)
                elapsed = time.time() - start
                total_time += elapsed

                if solution:
                    solved += 1
                    if solution.routing_success:
                        routed += 1

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                continue

        return ConfigurationResult(
            config_name="",  # Set by caller
            problems_tested=len(problems),
            solved=solved,
            routed=routed,
            solve_rate=solved / len(problems) if problems else 0,
            routing_rate=routed / solved if solved > 0 else 0,
            avg_solve_time=total_time / len(problems) if problems else 0,
            uses_placement_ml=use_placement_ml,
            uses_routing_ml=use_routing_ml,
        )


def main():
    parser = argparse.ArgumentParser(description="Compare ML models for placement and routing")
    parser.add_argument("--problems", type=int, default=100, help="Number of problems to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for neural networks")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Per-problem solve time limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for report")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--fresh", action="store_true", help="Clear existing samples and start fresh")
    parser.add_argument("--samples-db", type=str, default="training_samples.db",
                       help="Database file for training samples")
    parser.add_argument("--ab-test", action="store_true",
                       help="Run A/B test comparing placement vs routing ML")
    parser.add_argument("--test-problems", type=int, default=50,
                       help="Number of test problems for A/B testing")
    parser.add_argument("--min-memory-mb", type=int, default=500,
                       help="Minimum available memory (MB) before stopping")
    args = parser.parse_args()

    # Initial memory check
    try:
        check_memory(min_available_mb=args.min_memory_mb, verbose=not args.quiet)
    except InsufficientMemoryError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.ab_test:
            # Run A/B test comparing placement vs routing vs both
            tester = ABTester()

            print("=" * 60)
            print("PHASE 1: Collecting training data and training models")
            print("=" * 60)

            train_stats = tester.collect_and_train(
                num_problems=args.problems,
                epochs=args.epochs,
                time_limit=args.time_limit,
                seed=args.seed,
                verbose=not args.quiet,
            )

            # Check memory before phase 2
            check_memory(min_available_mb=args.min_memory_mb, verbose=not args.quiet)

            print("\n" + "=" * 60)
            print("PHASE 2: A/B Testing configurations")
            print("=" * 60)

            report = tester.run_ab_test(
                num_test_problems=args.test_problems,
                time_limit=args.time_limit,
                seed=args.seed + 10000,  # Different seed for test problems
                verbose=not args.quiet,
            )

            report.training_problems = train_stats["problems"]

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
                print(f"\nReport saved to {args.output}")

        else:
            # Original placement model comparison
            comparer = ModelComparer(samples_db_path=args.samples_db)

            report = comparer.run_comparison(
                num_problems=args.problems,
                epochs=args.epochs,
                time_limit=args.time_limit,
                seed=args.seed,
                verbose=not args.quiet,
                fresh_start=args.fresh,
            )

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2)
                print(f"\nReport saved to {args.output}")

    except InsufficientMemoryError as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print("CRITICAL: OUT OF MEMORY", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"\n{e}", file=sys.stderr)
        print("\nProgress has been saved. You can resume later.", file=sys.stderr)
        sys.exit(1)
    except MemoryError as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print("CRITICAL: PYTHON MEMORY ERROR", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"\nSystem ran out of memory: {e}", file=sys.stderr)
        print("Consider reducing --problems or freeing system memory.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
