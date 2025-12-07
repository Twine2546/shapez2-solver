#!/usr/bin/env python3
"""
Shapez 2 Solver - Main Entry Point

A tool for evolving solutions to transform shapes in Shapez 2.
"""

import argparse
import sys
from shapez2_solver.ui.app import SolverApp, main as run_ui
from shapez2_solver.shapes.parser import ShapeCodeParser
from shapez2_solver.shapes.encoder import ShapeCodeEncoder
from shapez2_solver.foundations.layouts import get_foundation, list_available_foundations
from shapez2_solver.foundations.foundation import FoundationType
from shapez2_solver.evolution.algorithm import EvolutionaryAlgorithm, EvolutionConfig


def run_cli(args):
    """Run the solver in CLI mode."""
    print("=" * 60)
    print("  Shapez 2 Solver - CLI Mode")
    print("=" * 60)

    # Parse shapes
    try:
        input_shape = ShapeCodeParser.parse(args.input)
        print(f"\nInput shape:  {args.input}")
        print(ShapeCodeEncoder.format_for_display(input_shape, multiline=True))
    except ValueError as e:
        print(f"Error parsing input shape: {e}")
        return 1

    try:
        target_shape = ShapeCodeParser.parse(args.target)
        print(f"\nTarget shape: {args.target}")
        print(ShapeCodeEncoder.format_for_display(target_shape, multiline=True))
    except ValueError as e:
        print(f"Error parsing target shape: {e}")
        return 1

    # Get foundation
    try:
        foundation_type = FoundationType(args.foundation)
        foundation = get_foundation(foundation_type)
        print(f"\nFoundation: {foundation_type.value}")
    except (ValueError, KeyError):
        print(f"Unknown foundation type: {args.foundation}")
        print(f"Available: {[f.value for f in list_available_foundations()]}")
        return 1

    # Configure evolution
    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
    )

    print(f"\nEvolution settings:")
    print(f"  Population: {config.population_size}")
    print(f"  Generations: {config.generations}")
    print(f"  Mutation rate: {config.mutation_rate}")

    # Run evolution
    algorithm = EvolutionaryAlgorithm(
        foundation=foundation,
        input_shapes={"input": input_shape},
        expected_outputs={"output": target_shape},
        config=config,
    )

    print("\n" + "=" * 60)
    print("Starting evolution...")
    print("=" * 60)

    def progress(gen: int, fitness: float) -> bool:
        stats = algorithm.history[-1] if algorithm.history else {}
        avg_ops = stats.get('avg_ops', 0)
        max_ops = stats.get('max_ops', 0)
        mut_rate = stats.get('mutation_rate', config.mutation_rate)

        if gen % 10 == 0 or fitness >= 1.0:
            print(f"Gen {gen:4d} | Fitness: {fitness:.4f} | Ops: avg={avg_ops:.1f} max={max_ops} | MutRate: {mut_rate:.2f}")
        return True

    def show_top_solutions(top_candidates):
        # Show top solutions every 20 generations
        if algorithm.generation % 20 == 0:
            print("\n--- Top Solutions ---")
            for i, candidate in enumerate(top_candidates[:3]):
                ops = [op.operation.__class__.__name__.replace("Operation", "")
                       for op in candidate.design.operations]
                print(f"  #{i+1}: Fitness={candidate.fitness:.4f}, Ops={len(ops)}: {', '.join(ops) or 'none'}")
            print("---------------------\n")

    algorithm.on_top_solutions = show_top_solutions
    result = algorithm.run(callback=progress)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    if result:
        print(f"Best fitness: {result.fitness:.4f}")
        print(f"Operations used: {len(result.design.operations)}")
        print(f"Connections: {len(result.design.connections)}")

        if result.fitness >= 1.0:
            print("\nPerfect solution found!")
        elif result.fitness > 0.8:
            print("\nGood solution found.")
        else:
            print("\nPartial solution found.")

        # Display the solution pipeline
        from shapez2_solver.visualization.solution_display import display_solution
        print("\n")
        display_solution(result.design, {"in_0": input_shape})
    else:
        print("No solution found.")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shapez 2 Solver - Evolve solutions for shape transformations"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Run the graphical interface")

    # CLI command
    cli_parser = subparsers.add_parser("solve", help="Solve a shape transformation")
    cli_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input shape code (e.g., CuCuCuCu)"
    )
    cli_parser.add_argument(
        "-t", "--target",
        required=True,
        help="Target shape code (e.g., CrCrCrCr)"
    )
    cli_parser.add_argument(
        "-f", "--foundation",
        default="1x1",
        help="Foundation type (default: 1x1)"
    )
    cli_parser.add_argument(
        "-p", "--population",
        type=int,
        default=50,
        help="Population size (default: 50)"
    )
    cli_parser.add_argument(
        "-g", "--generations",
        type=int,
        default=100,
        help="Number of generations (default: 100)"
    )
    cli_parser.add_argument(
        "-m", "--mutation-rate",
        type=float,
        default=0.3,
        help="Mutation rate (default: 0.3)"
    )

    # Parse shape command
    parse_parser = subparsers.add_parser("parse", help="Parse and display a shape code")
    parse_parser.add_argument("code", help="Shape code to parse")

    # List foundations command
    subparsers.add_parser("foundations", help="List available foundation types")

    args = parser.parse_args()

    if args.command == "gui" or args.command is None:
        run_ui()
        return 0
    elif args.command == "solve":
        return run_cli(args)
    elif args.command == "parse":
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
    elif args.command == "foundations":
        print("Available foundation types:")
        for ft in list_available_foundations():
            f = get_foundation(ft)
            print(f"  {ft.value:8s} - {f.width}x{f.height} tiles, {len(f.ports)} ports")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
