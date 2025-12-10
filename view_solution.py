#!/usr/bin/env python3
"""
View a CP-SAT solution in the pygame viewer.

Usage:
    python view_solution.py                    # Default L4 nightmare example
    python view_solution.py --foundation 2x2   # Different foundation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat
from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, FoundationConfig
from shapez2_solver.visualization.pygame_layout_viewer import show_layout_pygame


class MockEvolution:
    """Mock evolution object for the viewer."""
    def __init__(self, candidate, foundation_type: str):
        self.top_solutions = [candidate] if candidate else []
        self.config = FoundationConfig(FOUNDATION_SPECS[foundation_type])


def main():
    parser = argparse.ArgumentParser(description="View CP-SAT solution")
    parser.add_argument("--foundation", "-f", default="L4",
                        choices=list(FOUNDATION_SPECS.keys()),
                        help="Foundation type")
    parser.add_argument("--inputs", "-i", type=int, default=8,
                        help="Number of inputs")
    parser.add_argument("--outputs", "-o", type=int, default=12,
                        help="Number of outputs")
    parser.add_argument("--time-limit", "-t", type=float, default=60,
                        help="Solver time limit in seconds")
    args = parser.parse_args()

    print(f"Solving {args.foundation} with {args.inputs} inputs, {args.outputs} outputs...")
    print()

    # Generate input/output specs
    input_specs = [('W', i % 10, i // 10, 'CuCuCuCu') for i in range(args.inputs)]
    output_specs = [('E', i % 10, i // 10, 'Cu------') for i in range(args.outputs)]

    result = solve_with_cpsat(
        foundation_type=args.foundation,
        input_specs=input_specs,
        output_specs=output_specs,
        time_limit=args.time_limit,
        verbose=True,
        enable_placement_feedback=False,  # Skip broken model
        enable_transformer_logging=False,  # Don't log this test
    )

    if result:
        print(f"\nSolution found with {len(result.buildings)} buildings")
        print("Opening viewer...")

        # Count building types
        from collections import Counter
        counts = Counter(b.building_type.name for b in result.buildings)
        print("\nBuilding counts:")
        for bt, count in sorted(counts.items()):
            print(f"  {bt}: {count}")

        mock = MockEvolution(result, args.foundation)
        show_layout_pygame(mock)
    else:
        print("\nNo solution found!")


if __name__ == "__main__":
    main()
