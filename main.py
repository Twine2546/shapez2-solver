#!/usr/bin/env python3
"""
Shapez 2 Solver - Main Entry Point

A tool for finding optimal machine layouts in Shapez 2 using constraint programming.
"""

import argparse
import sys


def run_cpsat_gui():
    """Run the CP-SAT solver GUI."""
    from shapez2_solver.ui.cpsat_app import main
    main()


def run_parse(args):
    """Parse and display a shape code."""
    from shapez2_solver.shapes.parser import ShapeCodeParser
    from shapez2_solver.shapes.encoder import ShapeCodeEncoder

    try:
        shape = ShapeCodeParser.parse(args.code)
        print(f"Shape code: {args.code}")
        print(f"Normalized: {shape.to_code()}")
        print(f"Layers: {shape.num_layers}")
        print()
        print(ShapeCodeEncoder.format_for_display(shape, multiline=True))
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    return 0


def run_foundations():
    """List available foundation types."""
    from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS

    print("Available foundation types:")
    print()
    for name, spec in FOUNDATION_SPECS.items():
        print(f"  {name:8s} - {spec.grid_width}x{spec.grid_height} grid, {spec.num_floors} floors")
    return 0


def run_solve(args):
    """Solve using CP-SAT."""
    from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

    # Parse input/output specifications
    inputs = []
    for spec in args.input:
        parts = spec.split(',')
        if len(parts) == 4:
            inputs.append((parts[0].strip(), int(parts[1]), int(parts[2]), parts[3].strip()))
        else:
            print(f"Error: Invalid input spec '{spec}'. Format: Side,Pos,Floor,ShapeCode")
            return 1

    outputs = []
    for spec in args.output:
        parts = spec.split(',')
        if len(parts) == 4:
            outputs.append((parts[0].strip(), int(parts[1]), int(parts[2]), parts[3].strip()))
        else:
            print(f"Error: Invalid output spec '{spec}'. Format: Side,Pos,Floor,ShapeCode")
            return 1

    print("=" * 60)
    print("  Shapez 2 CP-SAT Solver")
    print("=" * 60)
    print(f"Foundation: {args.foundation}")
    print(f"Inputs: {len(inputs)}")
    print(f"Outputs: {len(outputs)}")
    print(f"Timeout: {args.timeout} seconds")
    print("=" * 60)

    solution = solve_with_cpsat(
        foundation_type=args.foundation,
        input_specs=inputs,
        output_specs=outputs,
        max_machines=args.max_machines,
        time_limit=args.timeout,
        verbose=True
    )

    if solution and solution.routing_success:
        print("\n✓ Solution found!")
        print(f"Buildings: {len(solution.buildings)}")
        print(f"Fitness: {solution.fitness:.2f}")
    else:
        print("\n✗ No solution found")
        return 1

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shapez 2 Solver - Find optimal machine layouts using CP-SAT"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # GUI command
    subparsers.add_parser("gui", help="Run the graphical interface")

    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve using CP-SAT")
    solve_parser.add_argument(
        "-f", "--foundation",
        default="2x2",
        help="Foundation type (default: 2x2)"
    )
    solve_parser.add_argument(
        "-i", "--input",
        action="append",
        required=True,
        help="Input spec: Side,Pos,Floor,ShapeCode (e.g., W,0,0,CuCuCuCu)"
    )
    solve_parser.add_argument(
        "-o", "--output",
        action="append",
        required=True,
        help="Output spec: Side,Pos,Floor,ShapeCode (e.g., E,0,0,Cu------)"
    )
    solve_parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds (default: 60)"
    )
    solve_parser.add_argument(
        "-m", "--max-machines",
        type=int,
        default=20,
        help="Maximum machines to place (default: 20)"
    )

    # Parse shape command
    parse_parser = subparsers.add_parser("parse", help="Parse and display a shape code")
    parse_parser.add_argument("code", help="Shape code to parse")

    # List foundations command
    subparsers.add_parser("foundations", help="List available foundation types")

    args = parser.parse_args()

    if args.command == "gui" or args.command is None:
        run_cpsat_gui()
        return 0
    elif args.command == "solve":
        return run_solve(args)
    elif args.command == "parse":
        return run_parse(args)
    elif args.command == "foundations":
        return run_foundations()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
