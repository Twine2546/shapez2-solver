#!/usr/bin/env python3
"""Solve the T 3x2 foundation problem."""

import sys
import json
from datetime import datetime
sys.path.insert(0, '/config/projects/programming/games/shape2')

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

# T 3x2 foundation problem - MINIMAL test: 1 input, 4 outputs
# Just testing with 1 full copper input split into 4 quarters
input_specs = [
    ("S", 4, 0, "CuCuCuCu"),  # Single input
]

# 4 quarter outputs - one to each direction
output_specs = [
    ("W", 0, 0, "Cu------"),    # Top-left quarter to West
    ("N", 0, 0, "--Cu----"),    # Top-right quarter to North
    ("E", 0, 0, "----Cu--"),    # Bottom-left quarter to East
    ("N", 8, 0, "------Cu"),    # Bottom-right quarter to North
]

print(f"Inputs: {len(input_specs)}")
print(f"Outputs: {len(output_specs)}")
print()

# Run the solver
print("Running CP-SAT solver...")
solution = solve_with_cpsat(
    foundation_type="T",
    input_specs=input_specs,
    output_specs=output_specs,
    max_machines=20,
    time_limit=120.0,
    verbose=True,
    routing_mode='astar',
    enable_placement_feedback=False,
    enable_transformer_logging=False,
)

if solution:
    print("\n" + "=" * 50)
    print("SOLUTION FOUND!")
    print("=" * 50)
    print(f"Buildings: {len(solution.buildings)}")
    print(f"Routing success: {solution.routing_success}")

    # Create simulation JSON
    sample_data = {
        "name": "T 3x2 Corner Splitter",
        "created": datetime.now().isoformat(),
        "foundation": "T",
        "buildings": [],
        "inputs": [],
        "outputs": []
    }

    # Add all buildings (machines + belts)
    for b in solution.buildings:
        building = {
            "type": b.building_type.name if hasattr(b.building_type, 'name') else str(b.building_type),
            "x": b.x,
            "y": b.y,
            "floor": b.floor,
            "rotation": b.rotation.name if hasattr(b.rotation, 'name') else str(b.rotation)
        }
        sample_data["buildings"].append(building)

    # Add inputs
    for side, pos, floor, shape in input_specs:
        # Convert side+position to grid coordinates
        from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, Side
        spec = FOUNDATION_SPECS["T"]
        side_enum = {"N": Side.NORTH, "E": Side.EAST, "S": Side.SOUTH, "W": Side.WEST}[side]
        gx, gy = spec.get_port_grid_position(side_enum, pos)

        # Offset for external position
        if side == "N":
            gy -= 1
        elif side == "S":
            gy += 1
        elif side == "W":
            gx -= 1
        elif side == "E":
            gx += 1

        sample_data["inputs"].append({
            "x": gx,
            "y": gy,
            "floor": floor,
            "shape": shape,
            "throughput": 45.0
        })

    # Add outputs
    for side, pos, floor, shape in output_specs:
        spec = FOUNDATION_SPECS["T"]
        side_enum = {"N": Side.NORTH, "E": Side.EAST, "S": Side.SOUTH, "W": Side.WEST}[side]
        gx, gy = spec.get_port_grid_position(side_enum, pos)

        # Offset for external position
        if side == "N":
            gy -= 1
        elif side == "S":
            gy += 1
        elif side == "W":
            gx -= 1
        elif side == "E":
            gx += 1

        sample_data["outputs"].append({
            "x": gx,
            "y": gy,
            "floor": floor,
            "expected_shape": shape
        })

    # Save to simulation samples
    output_path = "/config/projects/programming/games/shape2/shapez2_solver/simulation/samples/sample_01.json"
    with open(output_path, "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"\nSimulation saved to: {output_path}")
    print("Run: python -m shapez2_solver.simulation.pygame_flow_viewer")
    print("Then press 1 to load slot 1")
else:
    print("\nNo solution found!")
