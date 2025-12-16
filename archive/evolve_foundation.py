#!/usr/bin/env python3
"""
CLI tool for evolving Shapez 2 layouts on foundations.

Usage:
    python evolve_foundation.py --help

Examples:
    # Simple corner splitter on 2x2 foundation
    python evolve_foundation.py --foundation 2x2 \
        --input "W,0,0,CuCuCuCu" \
        --output "E,0,0,Cu------" \
        --output "E,0,1,--Cu----" \
        --output "E,1,0,----Cu--" \
        --output "E,1,1,------Cu"

    # 3-floor corner splitter
    python evolve_foundation.py --foundation 2x2 \
        --input "W,0,0,CuCuCuCu" --input "W,0,1,CuCuCuCu" --input "W,0,2,CuCuCuCu" \
        --output "E,0,0,Cu------" --output "E,0,1,--Cu----" \
        --output "E,1,0,----Cu--" --output "E,1,1,------Cu" \
        --generations 100

    # Interactive mode
    python evolve_foundation.py --interactive
"""

import sys
import argparse
from typing import List, Tuple

sys.path.insert(0, '/config/projects/programming/games/shape2')

from shapez2_solver.evolution.foundation_config import (
    FoundationConfig, FoundationSpec, Side, PortType, FOUNDATION_SPECS
)
from shapez2_solver.evolution.foundation_evolution import (
    FoundationEvolution, create_evolution_from_spec
)


def parse_port_spec(spec: str) -> Tuple[str, int, int, str]:
    """
    Parse a port specification string.

    Format: "SIDE,POSITION,FLOOR,SHAPE_CODE"
    Examples:
        "W,0,0,CuCuCuCu" - West side, position 0, floor 0, full circle
        "E,1,2,Cu------" - East side, position 1, floor 2, NE corner
    """
    parts = spec.split(',', 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid port spec: {spec}. Expected SIDE,POS,FLOOR,SHAPE")

    side = parts[0].strip().upper()
    pos = int(parts[1].strip())
    floor = int(parts[2].strip())
    shape_code = parts[3].strip()

    return (side, pos, floor, shape_code)


def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "=" * 70)
    print("SHAPEZ 2 FOUNDATION EVOLUTION - Interactive Mode")
    print("=" * 70)

    # Select foundation
    print("\nAvailable foundations:")
    foundations = list(FOUNDATION_SPECS.keys())
    for i, name in enumerate(foundations):
        spec = FOUNDATION_SPECS[name]
        print(f"  {i+1}. {name} ({spec.units_x}x{spec.units_y} units, {spec.grid_width}x{spec.grid_height} grid)")

    while True:
        try:
            choice = input("\nSelect foundation (number or name): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(foundations):
                    foundation_type = foundations[idx]
                    break
            elif choice in FOUNDATION_SPECS:
                foundation_type = choice
                break
            print("Invalid selection. Try again.")
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    spec = FOUNDATION_SPECS[foundation_type]
    config = FoundationConfig(spec)

    print(f"\nSelected: {foundation_type}")
    print(f"  Units: {spec.units_x}x{spec.units_y}")
    print(f"  Grid size: {spec.grid_width}x{spec.grid_height}")
    print(f"  Floors: {spec.num_floors}")
    print(f"  Ports per side: N={spec.ports_per_side[Side.NORTH]}, "
          f"E={spec.ports_per_side[Side.EAST]}, "
          f"S={spec.ports_per_side[Side.SOUTH]}, "
          f"W={spec.ports_per_side[Side.WEST]}")

    # Configure inputs
    print("\n" + "-" * 40)
    print("CONFIGURE INPUTS")
    print("-" * 40)
    print("Format: SIDE,POSITION,FLOOR,SHAPE_CODE")
    print("Example: W,0,0,CuCuCuCu")
    print("Enter empty line when done.")

    inputs = []
    while True:
        try:
            line = input("Input: ").strip()
            if not line:
                break
            side, pos, floor, shape_code = parse_port_spec(line)
            inputs.append((side, pos, floor, shape_code))
            print(f"  Added input: {side}[{pos}] Floor {floor}: {shape_code}")
        except ValueError as e:
            print(f"  Error: {e}")
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    # Configure outputs
    print("\n" + "-" * 40)
    print("CONFIGURE OUTPUTS")
    print("-" * 40)
    print("Format: SIDE,POSITION,FLOOR,SHAPE_CODE")
    print("Example: E,0,0,Cu------")
    print("Enter empty line when done.")

    outputs = []
    while True:
        try:
            line = input("Output: ").strip()
            if not line:
                break
            side, pos, floor, shape_code = parse_port_spec(line)
            outputs.append((side, pos, floor, shape_code))
            print(f"  Added output: {side}[{pos}] Floor {floor}: {shape_code}")
        except ValueError as e:
            print(f"  Error: {e}")
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    if not inputs or not outputs:
        print("Error: Must have at least one input and one output.")
        return

    # Configure evolution parameters
    print("\n" + "-" * 40)
    print("EVOLUTION PARAMETERS")
    print("-" * 40)

    try:
        generations = int(input("Number of generations [100]: ").strip() or "100")
        population = int(input("Population size [50]: ").strip() or "50")
        max_buildings = int(input("Max buildings [20]: ").strip() or "20")
    except ValueError:
        print("Invalid number. Using defaults.")
        generations, population, max_buildings = 100, 50, 20

    # Create and run evolution
    evolution = create_evolution_from_spec(
        foundation_type,
        inputs,
        outputs,
        population_size=population,
        max_buildings=max_buildings
    )

    print("\n" + "=" * 70)
    evolution.print_goal()

    input("\nPress Enter to start evolution...")

    top_solutions = evolution.run(generations, verbose=True)

    # Print blueprints
    print("\n" + "=" * 70)
    print("BLUEPRINT CODES")
    print("=" * 70)

    for i, sol in enumerate(top_solutions):
        print(f"\nSolution {i+1} (Fitness: {sol.fitness:.2f}):")
        print(evolution.export_blueprint(sol))

    # Show GUI if requested
    show_gui = input("\nShow GUI viewer? (y/n): ").strip().lower()
    if show_gui == 'y':
        from shapez2_solver.visualization import show_layout
        show_layout(evolution)


def batch_mode(args):
    """Run in batch mode with command-line arguments."""
    inputs = []
    outputs = []

    for spec in args.input:
        inputs.append(parse_port_spec(spec))

    for spec in args.output:
        outputs.append(parse_port_spec(spec))

    if not inputs:
        print("Error: At least one --input is required.")
        return

    if not outputs:
        print("Error: At least one --output is required.")
        return

    # Create evolution
    evolution = create_evolution_from_spec(
        args.foundation,
        inputs,
        outputs,
        population_size=args.population,
        max_buildings=args.max_buildings
    )

    # Print goal
    evolution.print_goal()

    # Run evolution
    top_solutions = evolution.run(args.generations, verbose=not args.quiet)

    # Print blueprints
    print("\n" + "=" * 70)
    print("BLUEPRINT CODES")
    print("=" * 70)

    for i, sol in enumerate(top_solutions):
        print(f"\nSolution {i+1} (Fitness: {sol.fitness:.2f}):")
        print(evolution.export_blueprint(sol))

    # Show GUI if requested
    if args.gui:
        from shapez2_solver.visualization import show_layout
        show_layout(evolution)


def show_foundations():
    """Show available foundations."""
    print("\nAvailable Foundations:")
    print("-" * 70)
    print(f"{'Name':<10} {'Units':<10} {'Grid Size':<15} {'Floors':<8} {'Ports/Side'}")
    print("-" * 70)

    for name, spec in FOUNDATION_SPECS.items():
        ports = f"N={spec.ports_per_side[Side.NORTH]}, E={spec.ports_per_side[Side.EAST]}"
        units = f"{spec.units_x}x{spec.units_y}"
        grid = f"{spec.grid_width}x{spec.grid_height}"
        print(f"{name:<10} {units:<10} {grid:<15} {spec.num_floors:<8} {ports}")

    print("-" * 70)
    print("\nFoundation internals:")
    print("  - Each 1x1 unit = 14x14 internal grid tiles")
    print("  - Additional units add 20 tiles per axis")
    print("  - Each 1x1 unit has 4 ports centered on each edge")
    print("\nPort positions:")
    print("  - Ports indexed 0-3 for unit 0, 4-7 for unit 1, etc.")
    print("  - Each position has floors 0 to 3")
    print("\nExample for 2x2 foundation:")
    print("  North side: 8 ports (positions 0-7)")
    print("  East side: 8 ports (positions 0-7)")
    print("  Grid size: 34x34 = 1156 tiles")


def show_shapes():
    """Show shape code format."""
    print("\nShape Code Format:")
    print("-" * 60)
    print("Each shape code has 4 parts for the 4 quadrants:")
    print("  Position 0: NE (top-right)")
    print("  Position 1: NW (top-left)")
    print("  Position 2: SW (bottom-left)")
    print("  Position 3: SE (bottom-right)")
    print()
    print("Shape types:")
    print("  C = Circle")
    print("  R = Rectangle")
    print("  S = Star")
    print("  W = Windmill")
    print("  - = Empty")
    print()
    print("Colors (optional, after shape):")
    print("  u = Uncolored")
    print("  r = Red")
    print("  g = Green")
    print("  b = Blue")
    print("  y = Yellow")
    print("  p = Purple")
    print("  c = Cyan")
    print("  w = White")
    print()
    print("Examples:")
    print("  CuCuCuCu - Full circle (uncolored)")
    print("  Cu------ - NE corner only")
    print("  --Cu---- - NW corner only")
    print("  ----Cu-- - SW corner only")
    print("  ------Cu - SE corner only")
    print("  CrCgCbCy - Circle with 4 different colors")
    print("  RuRu---- - Half rectangle (top)")


def main():
    parser = argparse.ArgumentParser(
        description="Evolve Shapez 2 layouts on foundations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Corner splitter on 2x2 foundation
  python evolve_foundation.py --foundation 2x2 \\
      --input "W,0,0,CuCuCuCu" \\
      --output "E,0,0,Cu------" \\
      --output "E,0,1,--Cu----" \\
      --output "E,1,0,----Cu--" \\
      --output "E,1,1,------Cu"

  # Interactive mode
  python evolve_foundation.py --interactive

  # Show available foundations
  python evolve_foundation.py --list-foundations

  # Show shape code format
  python evolve_foundation.py --list-shapes
        """
    )

    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--list-foundations', action='store_true',
                        help='List available foundation types')
    parser.add_argument('--list-shapes', action='store_true',
                        help='Show shape code format')

    parser.add_argument('--foundation', '-f', default='2x2',
                        help='Foundation type (default: 2x2)')
    parser.add_argument('--input', '-I', action='append', default=[],
                        help='Input port: SIDE,POS,FLOOR,SHAPE (can repeat)')
    parser.add_argument('--output', '-O', action='append', default=[],
                        help='Output port: SIDE,POS,FLOOR,SHAPE (can repeat)')

    parser.add_argument('--generations', '-g', type=int, default=100,
                        help='Number of generations (default: 100)')
    parser.add_argument('--population', '-p', type=int, default=50,
                        help='Population size (default: 50)')
    parser.add_argument('--max-buildings', '-m', type=int, default=20,
                        help='Maximum buildings per solution (default: 20)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Less verbose output')
    parser.add_argument('--gui', action='store_true',
                        help='Show GUI viewer after evolution')

    args = parser.parse_args()

    if args.list_foundations:
        show_foundations()
        return

    if args.list_shapes:
        show_shapes()
        return

    if args.interactive:
        interactive_mode()
        return

    if not args.input or not args.output:
        parser.print_help()
        print("\nError: Use --interactive mode or provide --input and --output")
        return

    batch_mode(args)


if __name__ == "__main__":
    main()
