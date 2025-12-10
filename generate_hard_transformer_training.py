#!/usr/bin/env python3
"""
Generate extremely hard placement problems for transformer training.

This script generates challenging routing/placement problems and uses the
CP-SAT solver to attempt solutions. The solver automatically logs all
attempts (success and failure) to the placement transformer database.

Usage:
    python generate_hard_transformer_training.py --count 100 --db hard_training.db
    python generate_hard_transformer_training.py --count 50 --difficulty nightmare
    python generate_hard_transformer_training.py --stats  # Show database stats
"""

import argparse
import random
import time
import sys
import gc
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, Side
from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat
from shapez2_solver.learning.placement_transformer import (
    PlacementTransformerDB,
    PlacementTransformerModel,
)


# Extremely hard problem configurations
HARD_CONFIGS = {
    # Densely packed with many crossing paths
    'crowded_small': {
        'foundations': ['1x1', '2x1', '1x2'],
        'num_inputs': (3, 5),
        'num_outputs': (4, 8),
        'description': 'Small foundation with many I/O ports',
    },
    # Maximum density on medium foundations
    'crowded_medium': {
        'foundations': ['2x2', '3x1', '1x3', 'L'],
        'num_inputs': (4, 8),
        'num_outputs': (6, 12),
        'description': 'Medium foundation with dense I/O',
    },
    # Multi-floor nightmare
    'multi_floor_chaos': {
        'foundations': ['2x2', '3x2', '2x3', 'T', 'L4'],
        'num_inputs': (6, 10),
        'num_outputs': (8, 16),
        'num_floors': (2, 4),
        'description': 'Multi-floor with crossing paths',
    },
    # Corner splitting (the classic hard problem)
    'corner_splitter': {
        'foundations': ['2x2', '3x3', 'Cross'],
        'num_inputs': (1, 2),
        'num_outputs': (4, 8),
        'force_corner_split': True,
        'description': 'Input splits to many outputs',
    },
    # Opposing flows (paths must cross)
    'opposing_flows': {
        'foundations': ['2x2', '3x2', '2x3', '3x3'],
        'num_inputs': (4, 8),
        'num_outputs': (4, 8),
        'opposing': True,
        'description': 'Inputs/outputs on opposite sides with crossing',
    },
    # Full-scale nightmare
    'nightmare': {
        'foundations': ['3x3', 'Cross', 'T', 'L4', 'S4'],
        'num_inputs': (8, 12),
        'num_outputs': (12, 20),
        'num_floors': (2, 4),
        'description': 'Maximum complexity problems',
    },
    # Extreme density (designed to fail often)
    'extreme': {
        'foundations': ['1x1', '2x1', '1x2', '2x2'],
        'num_inputs': (6, 10),
        'num_outputs': (10, 16),
        'description': 'Extreme density - expect many failures',
    },
}


@dataclass
class HardProblem:
    """A hard placement problem."""
    problem_id: str
    foundation_type: str
    input_specs: List[Tuple[str, int, int, str]]  # (side, pos, floor, shape_code)
    output_specs: List[Tuple[str, int, int, str]]
    difficulty: str


class HardProblemGenerator:
    """Generates extremely challenging placement problems."""

    # Shape codes for variety
    SHAPE_CODES = [
        'CuCuCuCu',  # Full circle
        'RuRuRuRu',  # Full square
        'SuSuSuSu',  # Full star
        'WuWuWuWu',  # Full diamond
        'Cu------',  # Quarter circle
        '--Cu----',
        '----Cu--',
        '------Cu',
        'CuCu----',  # Half shapes
        '----CuCu',
        'Cu--Cu--',
        '--Cu--Cu',
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.problem_counter = 0

    def generate_problem(
        self,
        difficulty: str = 'crowded_medium',
        foundation_type: Optional[str] = None,
    ) -> HardProblem:
        """Generate a single hard problem."""
        config = HARD_CONFIGS.get(difficulty, HARD_CONFIGS['crowded_medium'])

        # Select foundation
        if foundation_type is None:
            foundation_type = random.choice(config['foundations'])

        spec = FOUNDATION_SPECS.get(foundation_type)
        if spec is None:
            foundation_type = '2x2'
            spec = FOUNDATION_SPECS[foundation_type]

        self.problem_counter += 1
        problem_id = f"hard_{difficulty}_{foundation_type}_{self.problem_counter:05d}"

        # Determine number of inputs/outputs
        num_inputs = random.randint(*config.get('num_inputs', (4, 8)))
        num_outputs = random.randint(*config.get('num_outputs', (4, 8)))

        # Determine floors
        floor_range = config.get('num_floors', (1, 1))
        max_floors = min(floor_range[1], spec.num_floors)
        num_floors = random.randint(floor_range[0], max_floors)

        # Generate I/O based on problem type
        if config.get('force_corner_split'):
            input_specs, output_specs = self._generate_corner_split(
                spec, num_inputs, num_outputs, num_floors
            )
        elif config.get('opposing'):
            input_specs, output_specs = self._generate_opposing(
                spec, num_inputs, num_outputs, num_floors
            )
        else:
            input_specs, output_specs = self._generate_random_hard(
                spec, num_inputs, num_outputs, num_floors
            )

        return HardProblem(
            problem_id=problem_id,
            foundation_type=foundation_type,
            input_specs=input_specs,
            output_specs=output_specs,
            difficulty=difficulty,
        )

    def _get_side_char(self, side: Side) -> str:
        """Convert Side enum to character."""
        return {Side.NORTH: 'N', Side.SOUTH: 'S', Side.EAST: 'E', Side.WEST: 'W'}[side]

    def _generate_corner_split(
        self,
        spec,
        num_inputs: int,
        num_outputs: int,
        num_floors: int,
    ) -> Tuple[List, List]:
        """Generate corner splitter problem (input splits to many outputs)."""
        inputs = []
        outputs = []

        # Input(s) on one side (West typically)
        input_side = Side.WEST
        max_port = spec.ports_per_side[input_side]

        for i in range(min(num_inputs, max_port)):
            floor = i % num_floors
            pos = (i // num_floors) % max_port
            shape = random.choice(self.SHAPE_CODES[:4])  # Use full shapes
            inputs.append((self._get_side_char(input_side), pos, floor, shape))

        # Outputs spread across multiple sides
        output_sides = [Side.EAST, Side.NORTH, Side.SOUTH]
        outputs_per_side = num_outputs // len(output_sides) + 1

        for side in output_sides:
            max_port = spec.ports_per_side[side]
            for i in range(min(outputs_per_side, max_port)):
                if len(outputs) >= num_outputs:
                    break
                floor = i % num_floors
                pos = (i // num_floors) % max_port
                # Use quarter shapes for splitting
                shape = random.choice(self.SHAPE_CODES[4:8])
                outputs.append((self._get_side_char(side), pos, floor, shape))

        return inputs, outputs

    def _generate_opposing(
        self,
        spec,
        num_inputs: int,
        num_outputs: int,
        num_floors: int,
    ) -> Tuple[List, List]:
        """Generate opposing flow problem (paths must cross)."""
        inputs = []
        outputs = []

        # Inputs on West AND North
        input_sides = [Side.WEST, Side.NORTH]
        for side in input_sides:
            max_port = spec.ports_per_side[side]
            count = num_inputs // len(input_sides)
            for i in range(min(count, max_port)):
                if len(inputs) >= num_inputs:
                    break
                floor = random.randint(0, num_floors - 1)
                pos = i % max_port
                shape = random.choice(self.SHAPE_CODES[:4])
                inputs.append((self._get_side_char(side), pos, floor, shape))

        # Outputs on East AND South (opposing)
        output_sides = [Side.EAST, Side.SOUTH]
        for side in output_sides:
            max_port = spec.ports_per_side[side]
            count = num_outputs // len(output_sides)
            for i in range(min(count, max_port)):
                if len(outputs) >= num_outputs:
                    break
                floor = random.randint(0, num_floors - 1)
                pos = i % max_port
                shape = random.choice(self.SHAPE_CODES)
                outputs.append((self._get_side_char(side), pos, floor, shape))

        return inputs, outputs

    def _generate_random_hard(
        self,
        spec,
        num_inputs: int,
        num_outputs: int,
        num_floors: int,
    ) -> Tuple[List, List]:
        """Generate random but hard problem with many ports."""
        inputs = []
        outputs = []

        all_sides = [Side.NORTH, Side.SOUTH, Side.EAST, Side.WEST]
        used_positions = set()

        # Generate inputs
        for _ in range(num_inputs):
            for attempt in range(50):
                side = random.choice(all_sides)
                max_port = spec.ports_per_side[side]
                if max_port == 0:
                    continue
                pos = random.randint(0, max_port - 1)
                floor = random.randint(0, num_floors - 1)

                key = (side, pos, floor)
                if key not in used_positions:
                    used_positions.add(key)
                    shape = random.choice(self.SHAPE_CODES)
                    inputs.append((self._get_side_char(side), pos, floor, shape))
                    break

        # Generate outputs
        for _ in range(num_outputs):
            for attempt in range(50):
                side = random.choice(all_sides)
                max_port = spec.ports_per_side[side]
                if max_port == 0:
                    continue
                pos = random.randint(0, max_port - 1)
                floor = random.randint(0, num_floors - 1)

                key = (side, pos, floor)
                if key not in used_positions:
                    used_positions.add(key)
                    shape = random.choice(self.SHAPE_CODES)
                    outputs.append((self._get_side_char(side), pos, floor, shape))
                    break

        return inputs, outputs

    def generate_batch(
        self,
        count: int,
        difficulty_weights: Optional[Dict[str, float]] = None,
    ) -> List[HardProblem]:
        """Generate a batch of hard problems with given difficulty distribution."""
        if difficulty_weights is None:
            # Default: focus on hardest problems
            difficulty_weights = {
                'crowded_small': 0.15,
                'crowded_medium': 0.15,
                'multi_floor_chaos': 0.15,
                'corner_splitter': 0.15,
                'opposing_flows': 0.15,
                'nightmare': 0.15,
                'extreme': 0.10,
            }

        difficulties = list(difficulty_weights.keys())
        weights = list(difficulty_weights.values())

        problems = []
        for _ in range(count):
            difficulty = random.choices(difficulties, weights=weights)[0]
            problems.append(self.generate_problem(difficulty))

        return problems


def solve_and_log(
    problem: HardProblem,
    time_limit: float,
    db_path: str,
    verbose: bool = False,
) -> Tuple[bool, float, int]:
    """
    Solve a problem and log to transformer database.

    The solve_with_cpsat function automatically logs to the transformer
    database when enable_transformer_logging=True (default).

    Returns: (success, solve_time, num_belts)
    """
    start_time = time.time()

    try:
        # solve_with_cpsat automatically logs to transformer database
        result = solve_with_cpsat(
            foundation_type=problem.foundation_type,
            input_specs=problem.input_specs,
            output_specs=problem.output_specs,
            max_machines=20,
            time_limit=time_limit,
            verbose=verbose,
            routing_mode='reroute',  # Use reroute for better success
            # Transformer logging is enabled by default
            enable_transformer_logging=True,
            transformer_db_path=db_path,
            # Also enable placement feedback
            enable_placement_feedback=True,
            placement_db_path=db_path.replace('.db', '_placement.db'),
        )

        solve_time = time.time() - start_time

        if result is not None and hasattr(result, 'routing_success'):
            success = result.routing_success
            num_belts = len(result.buildings) if hasattr(result, 'buildings') else 0
            return success, solve_time, num_belts

        return False, solve_time, 0

    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return False, time.time() - start_time, 0


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def run_training_generation(
    count: int,
    db_path: str,
    time_limit: float,
    difficulty: Optional[str] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Generate hard problems and run solver to collect training data.

    Returns: (successes, failures)
    """
    generator = HardProblemGenerator(seed=seed)

    # Set up difficulty weights
    if difficulty is not None:
        difficulty_weights = {difficulty: 1.0}
    else:
        difficulty_weights = None  # Use default distribution

    if verbose:
        print("=" * 70)
        print("HARD PLACEMENT TRAINING DATA GENERATOR")
        print("=" * 70)
        print(f"Problems to generate: {count}")
        print(f"Database: {db_path}")
        print(f"Time limit per problem: {time_limit}s")
        if difficulty:
            print(f"Difficulty: {difficulty}")
        else:
            print("Difficulty: mixed (default distribution)")
        print("=" * 70)
        print()

    successes = 0
    failures = 0
    total_solve_time = 0.0
    generation_start = time.time()

    # Track stats by difficulty and foundation
    stats_by_difficulty: Dict[str, Dict[str, int]] = {}
    stats_by_foundation: Dict[str, Dict[str, int]] = {}

    problems = generator.generate_batch(count, difficulty_weights)

    for i, problem in enumerate(problems):
        # Initialize stats tracking
        if problem.difficulty not in stats_by_difficulty:
            stats_by_difficulty[problem.difficulty] = {'success': 0, 'fail': 0}
        if problem.foundation_type not in stats_by_foundation:
            stats_by_foundation[problem.foundation_type] = {'success': 0, 'fail': 0}

        # Calculate progress info
        progress_pct = (i / count) * 100
        elapsed = time.time() - generation_start
        avg_time = elapsed / (i + 1) if i > 0 else time_limit
        eta = avg_time * (count - i - 1)
        current_rate = (successes / (i + 1) * 100) if i > 0 else 0

        if verbose:
            # Compact single-line progress
            print(f"[{i+1:4d}/{count}] {progress_pct:5.1f}% | "
                  f"Rate: {current_rate:5.1f}% | "
                  f"ETA: {format_time(eta):>8s} | "
                  f"{problem.foundation_type:6s} {problem.difficulty:18s} | ",
                  end="", flush=True)

        success, solve_time, num_belts = solve_and_log(
            problem, time_limit, db_path, verbose=False
        )
        total_solve_time += solve_time

        if success:
            successes += 1
            stats_by_difficulty[problem.difficulty]['success'] += 1
            stats_by_foundation[problem.foundation_type]['success'] += 1
            if verbose:
                print(f"OK  {solve_time:5.1f}s {num_belts:3d} buildings")
        else:
            failures += 1
            stats_by_difficulty[problem.difficulty]['fail'] += 1
            stats_by_foundation[problem.foundation_type]['fail'] += 1
            if verbose:
                print(f"FAIL {solve_time:5.1f}s")

        # Print periodic summary every 100 problems
        if verbose and (i + 1) % 100 == 0 and i + 1 < count:
            print()
            print(f"  --- Progress Update ({i+1}/{count}) ---")
            print(f"  Success: {successes}/{i+1} ({successes/(i+1)*100:.1f}%)")
            print(f"  Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")
            print(f"  Avg solve time: {total_solve_time/(i+1):.1f}s")
            print(f"  By difficulty:")
            for diff, stats in sorted(stats_by_difficulty.items()):
                total = stats['success'] + stats['fail']
                rate = stats['success'] / total * 100 if total > 0 else 0
                print(f"    {diff:18s}: {stats['success']:3d}/{total:3d} ({rate:5.1f}%)")
            print()

        # Clean up memory
        gc.collect()

    total_elapsed = time.time() - generation_start

    if verbose:
        print()
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Total problems:  {count}")
        print(f"Successes:       {successes} ({successes/count*100:.1f}%)")
        print(f"Failures:        {failures} ({failures/count*100:.1f}%)")
        print(f"Total time:      {format_time(total_elapsed)}")
        print(f"Avg solve time:  {total_solve_time/count:.1f}s")
        print()
        print("BY DIFFICULTY:")
        print("-" * 50)
        for diff in sorted(stats_by_difficulty.keys()):
            stats = stats_by_difficulty[diff]
            total = stats['success'] + stats['fail']
            rate = stats['success'] / total * 100 if total > 0 else 0
            bar_len = int(rate / 5)  # 20 char max bar
            bar = '#' * bar_len + '-' * (20 - bar_len)
            print(f"  {diff:18s}: {stats['success']:4d}/{total:4d} [{bar}] {rate:5.1f}%")
        print()
        print("BY FOUNDATION:")
        print("-" * 50)
        for found in sorted(stats_by_foundation.keys()):
            stats = stats_by_foundation[found]
            total = stats['success'] + stats['fail']
            rate = stats['success'] / total * 100 if total > 0 else 0
            bar_len = int(rate / 5)
            bar = '#' * bar_len + '-' * (20 - bar_len)
            print(f"  {found:8s}: {stats['success']:4d}/{total:4d} [{bar}] {rate:5.1f}%")
        print()
        print("=" * 70)
        print("Training data logged to:", db_path)
        print("Both successes AND failures are valuable for training!")
        print("=" * 70)

    return successes, failures


def show_stats(db_path: str):
    """Show statistics from the transformer database."""
    try:
        model = PlacementTransformerModel(db_path=db_path)
        stats = model.get_stats()

        print()
        print("=" * 60)
        print("TRANSFORMER TRAINING DATABASE STATISTICS")
        print("=" * 60)
        print(f"Database: {db_path}")
        print()
        print(f"Total samples: {stats['total']}")
        print(f"Successes: {stats['successes']}")
        print(f"Failures: {stats['failures']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Model trained: {stats['model_trained']}")
        print(f"Online buffer: {stats['online_buffer_size']} samples")

        if stats['by_foundation']:
            print("\nBy foundation:")
            for foundation, data in sorted(stats['by_foundation'].items()):
                rate = data['successes'] / max(1, data['total'])
                print(f"  {foundation:8s}: {data['successes']:4d}/{data['total']:4d} ({rate:.1%})")

        print("=" * 60)

    except Exception as e:
        print(f"Error reading database: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate extremely hard placement problems for transformer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 mixed-difficulty hard problems
  python generate_hard_transformer_training.py --count 100

  # Generate 50 nightmare-level problems
  python generate_hard_transformer_training.py --count 50 --difficulty nightmare

  # Generate problems with longer time limit
  python generate_hard_transformer_training.py --count 100 --time-limit 120

  # Show database statistics
  python generate_hard_transformer_training.py --stats

Difficulty levels:
  crowded_small      - Small foundation, many I/O ports
  crowded_medium     - Medium foundation, dense I/O
  multi_floor_chaos  - Multi-floor with crossing paths
  corner_splitter    - Input splits to many outputs
  opposing_flows     - Inputs/outputs must cross
  nightmare          - Maximum complexity
  extreme            - Extreme density (expect failures)
        """
    )

    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of problems to generate (default: 100)"
    )
    parser.add_argument(
        "--db", type=str, default="hard_training.db",
        help="Transformer database path (default: hard_training.db)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=60.0,
        help="Solver time limit per problem in seconds (default: 60)"
    )
    parser.add_argument(
        "--difficulty", type=str, default=None,
        choices=list(HARD_CONFIGS.keys()),
        help="Generate only this difficulty level"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show database statistics and exit"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--train-after", action="store_true",
        help="Train the transformer model after generation"
    )

    args = parser.parse_args()

    if args.stats:
        show_stats(args.db)
        return

    # Run generation
    successes, failures = run_training_generation(
        count=args.count,
        db_path=args.db,
        time_limit=args.time_limit,
        difficulty=args.difficulty,
        verbose=not args.quiet,
        seed=args.seed,
    )

    # Optionally train after generation
    if args.train_after:
        print("\nTraining transformer model...")
        try:
            model = PlacementTransformerModel(db_path=args.db)
            success = model.train(epochs=50, verbose=True)
            if success:
                print("Training complete!")
            else:
                print("Training failed - not enough data or missing classes")
        except Exception as e:
            print(f"Training error: {e}")


if __name__ == "__main__":
    main()
