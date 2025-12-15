"""
Training data collection runner.

Generates diverse problems and solves them to collect training data
for the ML evaluators.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import traceback

from .problem_generator import ProblemGenerator, ProblemType
from .foundation_config import FOUNDATION_SPECS, FoundationConfig, Side, PortType
from .cpsat_solver import CPSATFullSolver
from .ml_evaluators import MLEvaluatorSystem
from .evaluation import DefaultSolutionEvaluator


def extract_specs_from_problem(problem: dict) -> tuple:
    """Extract input_specs and output_specs from a problem JSON.

    Returns:
        Tuple of (input_specs, output_specs) where each is a list of
        (side, pos, floor, shape_code) tuples.
    """
    foundation_name = problem["foundation"]
    spec = FOUNDATION_SPECS[foundation_name]

    input_specs = []
    output_specs = []

    # Process inputs
    for inp in problem.get("inputs", []):
        x, y = inp["x"], inp["y"]
        floor = inp["floor"]
        shape = inp.get("shape", "CuCuCuCu")

        side, position = _position_to_port(spec, x, y)
        if side and position is not None:
            input_specs.append((side.value, position, floor, shape))

    # Process outputs
    for out in problem.get("outputs", []):
        x, y = out["x"], out["y"]
        floor = out["floor"]
        expected = out.get("expected_shape", "")

        side, position = _position_to_port(spec, x, y)
        if side and position is not None:
            output_specs.append((side.value, position, floor, expected))

    return input_specs, output_specs


def _position_to_port(spec, x: int, y: int) -> tuple:
    """Convert external position to (side, port_index)."""
    grid_w = spec.grid_width
    grid_h = spec.grid_height

    # North side: y = -1
    if y < 0:
        # Find closest port position on north side
        for port_idx in range(spec.ports_per_side[Side.NORTH]):
            px, py = spec.get_port_grid_position(Side.NORTH, port_idx)
            if abs(px - x) <= 2:
                return Side.NORTH, port_idx

    # South side: y >= grid_h
    if y >= grid_h:
        for port_idx in range(spec.ports_per_side[Side.SOUTH]):
            px, py = spec.get_port_grid_position(Side.SOUTH, port_idx)
            if abs(px - x) <= 2:
                return Side.SOUTH, port_idx

    # West side: x < 0
    if x < 0:
        for port_idx in range(spec.ports_per_side[Side.WEST]):
            px, py = spec.get_port_grid_position(Side.WEST, port_idx)
            if abs(py - y) <= 2:
                return Side.WEST, port_idx

    # East side: x >= grid_w
    if x >= grid_w:
        for port_idx in range(spec.ports_per_side[Side.EAST]):
            px, py = spec.get_port_grid_position(Side.EAST, port_idx)
            if abs(py - y) <= 2:
                return Side.EAST, port_idx

    return None, None


class TrainingRunner:
    """Runs solver on generated problems to collect training data."""

    def __init__(
        self,
        db_path: str = "training_data.db",
        use_ml_evaluators: bool = False
    ):
        """Initialize training runner.

        Args:
            db_path: Path to SQLite database for training data
            use_ml_evaluators: If True, use ML evaluators (for evaluation),
                              if False, use default evaluators (for initial data collection)
        """
        self.db_path = db_path
        self.use_ml_evaluators = use_ml_evaluators
        self.ml_system = MLEvaluatorSystem(db_path=db_path)

        self.stats = {
            "problems_attempted": 0,
            "problems_solved": 0,
            "problems_failed": 0,
            "total_time": 0.0,
        }

    def solve_problem(
        self,
        problem: dict,
        time_limit: float = 60.0,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Solve a single problem and collect training data.

        Args:
            problem: Problem JSON dict
            time_limit: Solver time limit in seconds
            verbose: Print progress

        Returns:
            Dict with solution results and statistics
        """
        result = {
            "problem_name": problem.get("name", "Unknown"),
            "foundation": problem.get("foundation", "Unknown"),
            "solved": False,
            "time": 0.0,
            "buildings": 0,
            "routing_success": False,
            "error": None
        }

        try:
            # Extract input/output specs from problem
            foundation_name = problem.get("foundation", "2x2")
            input_specs, output_specs = extract_specs_from_problem(problem)

            if verbose:
                print(f"Solving: {problem['name']}")
                print(f"  Foundation: {foundation_name}")
                print(f"  Inputs: {len(input_specs)}")
                print(f"  Outputs: {len(output_specs)}")

            # Skip if no valid inputs/outputs
            if not input_specs or not output_specs:
                result["error"] = "No valid inputs or outputs"
                if verbose:
                    print(f"  Skipping: no valid I/O ports")
                return result

            # Create solver with evaluators
            # Disable internal ML features (use our pluggable evaluators instead)
            if self.use_ml_evaluators:
                solver = CPSATFullSolver(
                    foundation_type=foundation_name,
                    input_specs=input_specs,
                    output_specs=output_specs,
                    time_limit_seconds=time_limit,
                    enable_placement_feedback=False,  # Disable internal ML
                    enable_transformer_logging=False,
                    enable_logging=False,
                    solution_evaluator=self.ml_system.solution_evaluator,
                    placement_evaluator=self.ml_system.placement_evaluator,
                    routing_heuristic=self.ml_system.routing_heuristic,
                    move_cost_function=self.ml_system.move_cost_function
                )
            else:
                solver = CPSATFullSolver(
                    foundation_type=foundation_name,
                    input_specs=input_specs,
                    output_specs=output_specs,
                    time_limit_seconds=time_limit,
                    enable_placement_feedback=False,  # Disable internal ML
                    enable_transformer_logging=False,
                    enable_logging=False,
                    solution_evaluator=DefaultSolutionEvaluator(),
                    placement_evaluator=self.ml_system.placement_evaluator,
                    routing_heuristic=self.ml_system.routing_heuristic,
                    move_cost_function=self.ml_system.move_cost_function
                )

            # Solve
            start_time = time.time()
            solution = solver.solve(verbose=verbose)
            elapsed = time.time() - start_time

            result["time"] = elapsed

            if solution:
                result["solved"] = True
                # CPSATSolution is a dataclass with machines, belts, routing_success attributes
                result["buildings"] = len(solution.machines) + len(solution.belts)
                result["routing_success"] = solution.routing_success

                if verbose:
                    print(f"  Solved in {elapsed:.2f}s")
                    print(f"  Machines: {len(solution.machines)}, Belts: {len(solution.belts)}")
                    print(f"  Routing: {'Success' if result['routing_success'] else 'Failed'}")

                # Record training data for ML evaluators
                self._record_training_data(
                    solution=solution,
                    foundation_name=foundation_name,
                    input_specs=input_specs,
                    output_specs=output_specs
                )
            else:
                if verbose:
                    print(f"  No solution found in {elapsed:.2f}s")

        except Exception as e:
            result["error"] = str(e)
            if verbose:
                print(f"  Error: {e}")
                traceback.print_exc()

        return result

    def _record_training_data(
        self,
        solution,
        foundation_name: str,
        input_specs: List,
        output_specs: List
    ):
        """Record training data from a solved problem."""
        from .evaluation import PlacementInfo, SolutionInfo, RoutingInfo

        spec = FOUNDATION_SPECS[foundation_name]

        # Convert solution machines to PlacementInfo
        machines = []
        for building_type, x, y, floor, rotation in solution.machines:
            machines.append(PlacementInfo(
                building_type=building_type,
                x=x, y=y, floor=floor,
                rotation=rotation
            ))

        # Create RoutingInfo
        routing_info = RoutingInfo(
            belts=solution.belts,
            success=solution.routing_success,
            total_length=len(solution.belts)
        )

        # Get input/output positions for placement evaluator
        # Map side codes ('N', 'S', 'E', 'W') to Side enum
        side_map = {'N': Side.NORTH, 'S': Side.SOUTH, 'E': Side.EAST, 'W': Side.WEST}
        input_positions = [(spec.get_port_grid_position(side_map[s], p)[0],
                           spec.get_port_grid_position(side_map[s], p)[1], f)
                          for s, p, f, _ in input_specs]
        output_positions = [(spec.get_port_grid_position(side_map[s], p)[0],
                            spec.get_port_grid_position(side_map[s], p)[1], f)
                           for s, p, f, _ in output_specs]

        # Create SolutionInfo
        solution_info = SolutionInfo(
            machines=machines,
            routing=routing_info,
            grid_width=spec.grid_width,
            grid_height=spec.grid_height,
            num_floors=spec.num_floors,
            num_inputs=len(input_specs),
            num_outputs=len(output_specs),
            throughput_per_output=45.0 / len(output_specs) if output_specs else 0.0
        )

        # Record solution outcome (method is on_solution_found)
        self.ml_system.solution_evaluator.on_solution_found(
            solution_info,
            fitness=solution.fitness if hasattr(solution, 'fitness') else 1.0
        )

        # Record placement outcome
        self.ml_system.placement_evaluator.record_outcome(
            machines=machines,
            grid_width=spec.grid_width,
            grid_height=spec.grid_height,
            num_floors=spec.num_floors,
            input_positions=input_positions,
            output_positions=output_positions,
            routing_success=solution.routing_success
        )

    def run_training_batch(
        self,
        num_problems: int = 100,
        time_limit: float = 60.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        save_solutions: bool = False,
        solutions_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run a batch of training problems.

        Args:
            num_problems: Number of problems to generate and solve
            time_limit: Per-problem time limit
            seed: Random seed for reproducibility
            verbose: Print progress
            save_solutions: Save solved problems to files
            solutions_dir: Directory for saved solutions

        Returns:
            Batch statistics
        """
        generator = ProblemGenerator(seed=seed)

        if verbose:
            print(f"Generating {num_problems} training problems...")

        problems = generator.generate_training_set(
            num_problems=num_problems,
            ensure_coverage=True
        )

        if save_solutions and solutions_dir:
            solutions_dir = Path(solutions_dir)
            solutions_dir.mkdir(parents=True, exist_ok=True)

        results = []
        batch_start = time.time()

        for i, problem in enumerate(problems):
            if verbose:
                print(f"\n[{i+1}/{len(problems)}]")

            result = self.solve_problem(problem, time_limit, verbose)
            results.append(result)

            self.stats["problems_attempted"] += 1
            if result["solved"]:
                self.stats["problems_solved"] += 1

                if save_solutions and solutions_dir:
                    # Save successful solution
                    filepath = solutions_dir / f"solution_{i:04d}.json"
                    with open(filepath, 'w') as f:
                        json.dump(problem, f, indent=2)
            else:
                self.stats["problems_failed"] += 1

        batch_time = time.time() - batch_start
        self.stats["total_time"] += batch_time

        # Compute batch statistics
        solved = sum(1 for r in results if r["solved"])
        routed = sum(1 for r in results if r["routing_success"])
        avg_time = sum(r["time"] for r in results) / len(results) if results else 0
        avg_buildings = sum(r["buildings"] for r in results if r["solved"]) / solved if solved else 0

        batch_stats = {
            "total_problems": len(problems),
            "solved": solved,
            "success_rate": solved / len(problems) if problems else 0,
            "routing_success": routed,
            "routing_rate": routed / solved if solved else 0,
            "avg_time": avg_time,
            "avg_buildings": avg_buildings,
            "total_time": batch_time
        }

        if verbose:
            print(f"\n{'='*50}")
            print(f"Batch Complete")
            print(f"  Problems: {batch_stats['total_problems']}")
            print(f"  Solved: {batch_stats['solved']} ({batch_stats['success_rate']*100:.1f}%)")
            print(f"  Routing Success: {batch_stats['routing_success']} ({batch_stats['routing_rate']*100:.1f}%)")
            print(f"  Avg Time: {batch_stats['avg_time']:.2f}s")
            print(f"  Avg Buildings: {batch_stats['avg_buildings']:.1f}")
            print(f"  Total Time: {batch_stats['total_time']:.1f}s")

        # Print training data statistics
        data_stats = self.ml_system.get_stats()
        if verbose:
            print(f"\nTraining Data Collected:")
            print(f"  Solution samples: {data_stats['solution_samples']}")
            print(f"  Placement samples: {data_stats['placement_samples']}")
            print(f"  Routing samples: {data_stats['routing_samples']}")
            print(f"  Move cost samples: {data_stats['move_cost_samples']}")

        return batch_stats

    def run_progressive_training(
        self,
        rounds: int = 5,
        problems_per_round: int = 50,
        time_limit: float = 60.0,
        train_after_rounds: int = 2,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Run progressive training with periodic model updates.

        Args:
            rounds: Number of training rounds
            problems_per_round: Problems per round
            time_limit: Per-problem time limit
            train_after_rounds: Train models after this many rounds
            verbose: Print progress

        Returns:
            List of per-round statistics
        """
        all_stats = []

        for round_num in range(1, rounds + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"TRAINING ROUND {round_num}/{rounds}")
                print(f"{'='*60}")

            # Use different seed each round for variety
            seed = round_num * 1000

            # Run batch
            stats = self.run_training_batch(
                num_problems=problems_per_round,
                time_limit=time_limit,
                seed=seed,
                verbose=verbose
            )
            stats["round"] = round_num
            all_stats.append(stats)

            # Train models periodically
            if round_num % train_after_rounds == 0 and round_num < rounds:
                if verbose:
                    print(f"\nTraining ML models...")

                self.ml_system.train_all()
                self.use_ml_evaluators = True  # Start using ML after first training

                if verbose:
                    print("Models trained successfully")

        # Final training
        if verbose:
            print(f"\nFinal model training...")
        self.ml_system.train_all()

        # Print overall statistics
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Total rounds: {rounds}")
            print(f"Total problems: {self.stats['problems_attempted']}")
            print(f"Total solved: {self.stats['problems_solved']}")
            print(f"Overall success rate: {self.stats['problems_solved']/self.stats['problems_attempted']*100:.1f}%")
            print(f"Total time: {self.stats['total_time']:.1f}s")

            # Show improvement over rounds
            if len(all_stats) > 1:
                first_rate = all_stats[0]["success_rate"]
                last_rate = all_stats[-1]["success_rate"]
                print(f"\nSuccess rate improvement:")
                print(f"  Round 1: {first_rate*100:.1f}%")
                print(f"  Round {rounds}: {last_rate*100:.1f}%")
                if last_rate > first_rate:
                    print(f"  Improvement: +{(last_rate-first_rate)*100:.1f}%")

        return all_stats


def main():
    """Run training data collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect training data for ML evaluators")
    parser.add_argument("--problems", type=int, default=50, help="Problems per batch")
    parser.add_argument("--rounds", type=int, default=3, help="Training rounds")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Per-problem time limit")
    parser.add_argument("--db", type=str, default="training_data.db", help="Database path")
    parser.add_argument("--progressive", action="store_true", help="Use progressive training")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    runner = TrainingRunner(db_path=args.db, use_ml_evaluators=False)

    if args.progressive:
        runner.run_progressive_training(
            rounds=args.rounds,
            problems_per_round=args.problems,
            time_limit=args.time_limit,
            verbose=not args.quiet
        )
    else:
        runner.run_training_batch(
            num_problems=args.problems,
            time_limit=args.time_limit,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
