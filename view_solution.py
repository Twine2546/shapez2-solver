#!/usr/bin/env python3
"""
View a CP-SAT solution in the pygame viewer or ASCII.

Usage:
    python view_solution.py                    # Default L4 nightmare example
    python view_solution.py --foundation 2x2   # Different foundation
    python view_solution.py --ascii            # ASCII output instead of pygame
    python view_solution.py --ascii --floor 0  # ASCII for specific floor
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat
from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, FoundationConfig
from shapez2_solver.blueprint.building_types import BuildingType, Rotation


class MockEvolution:
    """Mock evolution object for the viewer."""
    def __init__(self, candidate, foundation_type: str):
        self.top_solutions = [candidate] if candidate else []
        self.config = FoundationConfig(FOUNDATION_SPECS[foundation_type])


# ASCII symbols for buildings
ASCII_SYMBOLS = {
    BuildingType.BELT_FORWARD: {
        Rotation.EAST: '→', Rotation.WEST: '←',
        Rotation.SOUTH: '↓', Rotation.NORTH: '↑'
    },
    BuildingType.BELT_LEFT: 'L',
    BuildingType.BELT_RIGHT: 'R',
    BuildingType.LIFT_UP: '⬆',
    BuildingType.LIFT_DOWN: '⬇',
    BuildingType.CUTTER: 'C',
    BuildingType.CUTTER_MIRRORED: 'c',
    BuildingType.HALF_CUTTER: 'H',
    BuildingType.STACKER: 'S',
    BuildingType.STACKER_BENT: 'B',
    BuildingType.STACKER_BENT_MIRRORED: 'b',
    BuildingType.ROTATOR_CW: '↻',
    BuildingType.ROTATOR_CCW: '↺',
    BuildingType.ROTATOR_180: '⟲',
    BuildingType.SWAPPER: 'X',
    BuildingType.UNSTACKER: 'U',
    BuildingType.BELT_PORT_SENDER: '⊳',
    BuildingType.BELT_PORT_RECEIVER: '⊲',
    BuildingType.SPLITTER: 'Y',
    BuildingType.MERGER: 'M',
}


def render_ascii(candidate, spec, floor: int = 0, valid_cells=None):
    """Render solution as ASCII art."""
    grid_w = spec.grid_width
    grid_h = spec.grid_height

    # Create empty grid
    grid = [['.' for _ in range(grid_w)] for _ in range(grid_h)]

    # Mark invalid cells for irregular foundations
    if valid_cells is not None:
        for y in range(grid_h):
            for x in range(grid_w):
                if (x, y) not in valid_cells:
                    grid[y][x] = ' '

    # Place buildings
    for b in candidate.buildings:
        if b.floor != floor:
            continue

        x, y = b.x, b.y
        if 0 <= x < grid_w and 0 <= y < grid_h:
            sym = ASCII_SYMBOLS.get(b.building_type)
            if isinstance(sym, dict):
                sym = sym.get(b.rotation, '?')
            elif sym is None:
                sym = '?'
            grid[y][x] = sym

    # Build output string
    lines = []

    # Header with x coordinates (every 10)
    header = '    '
    for x in range(grid_w):
        if x % 10 == 0:
            header += str(x // 10)
        else:
            header += ' '
    lines.append(header)

    header2 = '    '
    for x in range(grid_w):
        header2 += str(x % 10)
    lines.append(header2)
    lines.append('    ' + '-' * grid_w)

    # Grid rows
    for y in range(grid_h):
        line = f'{y:3d}|'
        for x in range(grid_w):
            line += grid[y][x]
        line += '|'
        lines.append(line)

    lines.append('    ' + '-' * grid_w)

    return '\n'.join(lines)


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
    parser.add_argument("--ascii", "-a", action="store_true",
                        help="Show ASCII rendering instead of pygame")
    parser.add_argument("--floor", type=int, default=0,
                        help="Floor to display (for ASCII mode)")
    args = parser.parse_args()

    # Check port limits
    spec = FOUNDATION_SPECS[args.foundation]
    from shapez2_solver.evolution.foundation_config import Side
    max_west = spec.ports_per_side[Side.WEST]
    max_east = spec.ports_per_side[Side.EAST]

    if args.inputs > max_west:
        print(f"Warning: {args.foundation} only has {max_west} WEST ports, reducing inputs from {args.inputs}")
        args.inputs = max_west
    if args.outputs > max_east:
        print(f"Warning: {args.foundation} only has {max_east} EAST ports, reducing outputs from {args.outputs}")
        args.outputs = max_east

    print(f"Solving {args.foundation} with {args.inputs} inputs, {args.outputs} outputs...")
    print(f"  Grid: {spec.grid_width}x{spec.grid_height}, {spec.num_floors} floors")
    if spec.present_cells:
        print(f"  Shape: {spec.present_cells} (irregular)")
    print()

    # Generate input/output specs
    input_specs = [('W', i % 4, i // 4, 'CuCuCuCu') for i in range(args.inputs)]
    output_specs = [('E', i % 4, i // 4, 'Cu------') for i in range(args.outputs)]

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

        # Count building types
        counts = Counter(b.building_type.name for b in result.buildings)
        print("\nBuilding counts:")
        for bt, count in sorted(counts.items()):
            print(f"  {bt}: {count}")

        spec = FOUNDATION_SPECS[args.foundation]

        if args.ascii:
            # ASCII rendering
            valid_cells = spec.get_valid_grid_cells()
            print(f"\n=== Floor {args.floor} ===")
            print(render_ascii(result, spec, args.floor, valid_cells))

            # Legend
            print("\nLegend:")
            print("  → ← ↓ ↑  Belt (direction)")
            print("  L R      Belt turn left/right")
            print("  ⬆ ⬇      Lift up/down")
            print("  C c      Cutter / mirrored")
            print("  S B b    Stacker / bent / bent mirrored")
            print("  ↻ ↺ ⟲    Rotator CW/CCW/180")
            print("  X U      Swapper / Unstacker")
            print("  ⊳ ⊲      Belt port sender/receiver")
            print("  .        Empty valid cell")
            print("  (space)  Invalid (outside foundation)")
        else:
            # Pygame viewer
            print("Opening viewer...")
            from shapez2_solver.visualization.pygame_layout_viewer import show_layout_pygame
            mock = MockEvolution(result, args.foundation)
            show_layout_pygame(mock)
    else:
        print("\nNo solution found!")


if __name__ == "__main__":
    main()
