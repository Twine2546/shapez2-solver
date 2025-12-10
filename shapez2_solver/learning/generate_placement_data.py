#!/usr/bin/env python3
"""
Generate training data for the placement ML model.

Runs the solver on various foundation types and captures placement outcomes.
"""

import argparse
import random
import time
from typing import List, Tuple

from shapez2_solver.evolution.cpsat_solver import CPSATFullSolver
from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, Side


# Define test scenarios with different shapes and foundation types
# num_ports controls how many inputs/outputs to use (more = harder)
# Multi-layer shapes (CrCu) require stacker, making routing harder
SCENARIOS = [
    # Simple scenarios (easy) - 4 ports
    {'foundation': '1x1', 'shape': 'CrCrCrCr', 'difficulty': 'easy', 'num_ports': 4},
    {'foundation': '2x1', 'shape': 'CrCrCrCr', 'difficulty': 'easy', 'num_ports': 4},

    # Medium scenarios - 8 ports
    {'foundation': '2x2', 'shape': 'CrCrCrCr', 'difficulty': 'medium', 'num_ports': 8},
    {'foundation': '3x1', 'shape': 'CuCuCuCu', 'difficulty': 'medium', 'num_ports': 8},

    # Hard scenarios - multi-layer shapes require stacker + cutter
    # These require 2+ machines which is much harder to route
    {'foundation': '2x2', 'shape': 'CrCr:CuCu', 'difficulty': 'hard', 'num_ports': 8},
    {'foundation': '3x1', 'shape': 'CrCr:CuCu', 'difficulty': 'hard', 'num_ports': 8},
    {'foundation': 'T', 'shape': 'CrCr:CuCu', 'difficulty': 'hard', 'num_ports': 12},

    # Extreme - many ports + multi-machine
    {'foundation': '3x3', 'shape': 'CrCr:CuCu', 'difficulty': 'extreme', 'num_ports': 16},
    {'foundation': '3x2', 'shape': 'CrCr:CuCu', 'difficulty': 'extreme', 'num_ports': 12},

    # Cramped - small foundation with many ports
    {'foundation': '1x1', 'shape': 'CrCrCrCr', 'difficulty': 'cramped', 'num_ports': 4},
    {'foundation': '1x1', 'shape': 'CuCuCuCu', 'difficulty': 'cramped', 'num_ports': 4},

    # Dense - fully utilize all ports
    {'foundation': '2x2', 'shape': 'CrCrCrCr', 'difficulty': 'dense', 'num_ports': 32},
    {'foundation': '2x1', 'shape': 'CuCuCuCu', 'difficulty': 'dense', 'num_ports': 16},
]


def get_foundation_ports(
    foundation_name: str,
    shape: str,
    num_inputs: int = 4,
    num_outputs: int = 4,
    floor: int = 0,
) -> Tuple[List, List]:
    """
    Get input and output port specs for a foundation.

    Returns port specs in format expected by CPSATFullSolver:
      [(side_str, port_index, floor, shape_code), ...]
    """
    spec = FOUNDATION_SPECS.get(foundation_name)
    if spec is None:
        raise ValueError(f"Unknown foundation: {foundation_name}")

    input_ports = []
    output_ports = []

    # Get port positions from the foundation spec
    # Inputs on WEST side, outputs on EAST side
    ports_west = spec.ports_per_side.get(Side.WEST, 0)
    ports_east = spec.ports_per_side.get(Side.EAST, 0)

    # Generate input ports on WEST side
    # Format: (side_str, port_index, floor, shape_code)
    for i in range(min(num_inputs, ports_west)):
        input_ports.append(("WEST", i, floor, shape))

    # Generate output ports on EAST side
    for i in range(min(num_outputs, ports_east)):
        output_ports.append(("EAST", i, floor, shape))

    return input_ports, output_ports


def run_scenario(
    foundation: str,
    shape: str,
    num_ports: int = 4,
    max_machines: int = 10,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> bool:
    """Run a single scenario and return success/failure."""

    try:
        # get_foundation_ports now returns specs in correct format
        input_specs, output_specs = get_foundation_ports(
            foundation, shape, num_inputs=num_ports, num_outputs=num_ports
        )
    except Exception as e:
        if verbose:
            print(f"  Error getting ports for {foundation}: {e}")
        return False

    if not input_specs or not output_specs:
        if verbose:
            print(f"  No ports found for {foundation}")
        return False

    if verbose:
        print(f"  Inputs: {len(input_specs)}, Outputs: {len(output_specs)}")

    try:
        solver = CPSATFullSolver(
            foundation_type=foundation,
            input_specs=input_specs,
            output_specs=output_specs,
            max_machines=max_machines,
            time_limit_seconds=time_limit,
            routing_mode='reroute',  # Use best routing mode
            enable_placement_feedback=True,  # Log to placement DB
            reject_bad_placements=False,  # Don't reject - we want to learn from failures
        )

        solution = solver.solve(verbose=verbose)

        return solution is not None and solution.status in ['optimal', 'feasible']

    except Exception as e:
        if verbose:
            print(f"  Solver error: {e}")
        return False


def generate_training_data(
    num_problems: int = 100,
    time_limit: float = 30.0,
    verbose: bool = False,
    seed: int = None,
):
    """Generate training data by running solver on various scenarios."""

    if seed is not None:
        random.seed(seed)

    print("=" * 60)
    print("GENERATING PLACEMENT TRAINING DATA")
    print("=" * 60)
    print(f"Problems to generate: {num_problems}")
    print(f"Time limit per problem: {time_limit}s")
    print(f"Available scenarios: {len(SCENARIOS)}")

    # Filter to available foundations
    available_scenarios = []
    for scenario in SCENARIOS:
        if scenario['foundation'] in FOUNDATION_SPECS:
            available_scenarios.append(scenario)

    print(f"Available (matching foundations): {len(available_scenarios)}")

    if not available_scenarios:
        print("ERROR: No matching foundation specs found!")
        return

    successes = 0
    failures = 0
    start_time = time.time()

    for i in range(num_problems):
        # Pick a random scenario
        scenario = random.choice(available_scenarios)

        foundation = scenario['foundation']
        shape = scenario['shape']
        difficulty = scenario['difficulty']
        num_ports = scenario.get('num_ports', 4)

        print(f"\n[{i+1}/{num_problems}] {foundation} ({difficulty}, {num_ports} ports): {shape}")

        success = run_scenario(
            foundation=foundation,
            shape=shape,
            num_ports=num_ports,
            max_machines=10,
            time_limit=time_limit,
            verbose=verbose,
        )

        if success:
            successes += 1
            print(f"  Result: SUCCESS")
        else:
            failures += 1
            print(f"  Result: FAILED")

        # Progress stats every 10 problems
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (num_problems - i - 1) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {successes}/{i+1} ({100*successes/(i+1):.1f}%), "
                  f"ETA: {remaining:.0f}s ---")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total problems: {num_problems}")
    print(f"Successes: {successes} ({100*successes/num_problems:.1f}%)")
    print(f"Failures: {failures} ({100*failures/num_problems:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/num_problems:.2f}s per problem)")


def main():
    parser = argparse.ArgumentParser(description="Generate placement training data")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of problems to generate")
    parser.add_argument("--time-limit", type=float, default=30.0,
                       help="Time limit per problem in seconds")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--train", action="store_true",
                       help="Train model after generating data")
    parser.add_argument("--min-samples", type=int, default=50,
                       help="Minimum samples required for training")

    args = parser.parse_args()

    # Generate data
    generate_training_data(
        num_problems=args.count,
        time_limit=args.time_limit,
        verbose=args.verbose,
        seed=args.seed,
    )

    # Train model if requested
    if args.train:
        print("\n" + "=" * 60)
        print("TRAINING PLACEMENT MODEL")
        print("=" * 60)

        from shapez2_solver.learning.placement_feedback import PlacementFeedbackLogger

        logger = PlacementFeedbackLogger()
        stats = logger.get_stats()

        print(f"Training data: {stats['total_attempts']} samples")
        print(f"Success rate in data: {stats['success_rate']:.1%}")

        if stats['total_attempts'] >= args.min_samples:
            success = logger.train_model(min_samples=args.min_samples)
            if success:
                print("\nModel trained successfully!")
            else:
                print("\nModel training failed")
        else:
            print(f"\nNot enough samples (need {args.min_samples}, have {stats['total_attempts']})")


if __name__ == "__main__":
    main()
