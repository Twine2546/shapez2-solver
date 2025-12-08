#!/usr/bin/env python3
"""Test script to generate a corner splitter blueprint with proper belt routing."""

import sys
sys.path.insert(0, '/config/projects/programming/games/shape2')

from shapez2_solver.foundations.foundation import Foundation, FoundationType, Port, PortDirection
from shapez2_solver.simulator.design import Design
from shapez2_solver.operations.cutter import CutOperation
from shapez2_solver.operations.rotator import RotateOperation
from shapez2_solver.blueprint.placer import GridPlacer
from shapez2_solver.blueprint.router import ConveyorRouter
from shapez2_solver.blueprint.encoder import BlueprintEncoder


def create_corner_splitter_design():
    """
    Create a corner splitter design:

    Input: CuCuCuCu (full shape)
           │
           ▼
        [Cutter 1] ─────┐
           │            │
           │ (east)     │ (west)
           ▼            ▼
        [Rotate CW]  [Rotate CW]
           │            │
           ▼            ▼
        [Cutter 2]   [Cutter 3]
         │     │     │     │
         ▼     ▼     ▼     ▼
        Out0  Out1  Out2  Out3

    Output: 4 individual corners
    """
    # Create a foundation
    foundation = Foundation.create_3x3()

    # Create design
    design = Design(foundation=foundation)

    # Add input port
    input_port = Port(PortDirection.WEST, 0, 0, is_input=True)
    input_id = design.add_input(input_port)

    # Add output ports
    out0_port = Port(PortDirection.EAST, 0, 0, is_input=False)
    out1_port = Port(PortDirection.EAST, 0, 1, is_input=False)
    out2_port = Port(PortDirection.EAST, 0, 2, is_input=False)
    out3_port = Port(PortDirection.EAST, 1, 0, is_input=False)  # On floor 1 for spacing

    out0_id = design.add_output(out0_port)
    out1_id = design.add_output(out1_port)
    out2_id = design.add_output(out2_port)
    out3_id = design.add_output(out3_port)

    # Add operations
    # First cutter: splits input into east/west halves
    cutter1_id = design.add_operation(CutOperation(), position=(4, 0), floor=0)

    # Two rotators: rotate each half 90° CW
    rotate1_id = design.add_operation(RotateOperation(steps=1), position=(8, 0), floor=0)
    rotate2_id = design.add_operation(RotateOperation(steps=1), position=(8, 2), floor=0)

    # Two cutters: split each rotated half
    cutter2_id = design.add_operation(CutOperation(), position=(12, 0), floor=0)
    cutter3_id = design.add_operation(CutOperation(), position=(12, 2), floor=0)

    # Connect input to first cutter
    design.connect(input_id, 0, cutter1_id, 0)

    # Connect first cutter outputs to rotators
    # CutOperation output 0 = east half, output 1 = west half
    design.connect(cutter1_id, 0, rotate1_id, 0)  # East half to first rotator
    design.connect(cutter1_id, 1, rotate2_id, 0)  # West half to second rotator

    # Connect rotators to second set of cutters
    design.connect(rotate1_id, 0, cutter2_id, 0)
    design.connect(rotate2_id, 0, cutter3_id, 0)

    # Connect second cutters to outputs
    design.connect(cutter2_id, 0, out0_id, 0)  # East of first rotated half
    design.connect(cutter2_id, 1, out1_id, 0)  # West of first rotated half
    design.connect(cutter3_id, 0, out2_id, 0)  # East of second rotated half
    design.connect(cutter3_id, 1, out3_id, 0)  # West of second rotated half

    return design


def generate_blueprint(design):
    """Generate blueprint code from design."""
    # Place buildings on grid
    placer = GridPlacer(width=32, height=32, num_floors=3)
    placements = placer.place_design(design)

    print("Placements:")
    for node_id, placement in placements.items():
        print(f"  {node_id}: {placement.building_type.name} at ({placement.x}, {placement.y}, L{placement.layer})")
        print(f"    Inputs: {placement.input_positions}")
        print(f"    Outputs: {placement.output_positions}")

    # Create encoder and add buildings
    encoder = BlueprintEncoder()
    encoder.add_placements(placements)

    # Route belts
    router = ConveyorRouter(placer, num_floors=3)
    routes = router.route_connections(design)

    print(f"\nRoutes: {len(routes)}")
    for route in routes:
        print(f"  {route.source_id}[{route.source_output}] -> {route.target_id}[{route.target_input}]: {len(route.segments)} segments")

    encoder.add_routes(routes)

    # Generate blueprint code
    blueprint_code = encoder.encode()

    # Print grid visualization
    print("\nGrid Layout:")
    print(placer.print_grid())

    return blueprint_code


def main():
    print("=" * 60)
    print("Corner Splitter Blueprint Generator")
    print("=" * 60)

    # Create design
    design = create_corner_splitter_design()

    # Validate
    valid, errors = design.validate()
    if not valid:
        print("Design validation errors:")
        for error in errors:
            print(f"  - {error}")
        return

    print(f"Design: {design}")
    print(f"  Operations: {len(design.operations)}")
    print(f"  Connections: {len(design.connections)}")

    # Generate blueprint
    blueprint_code = generate_blueprint(design)

    print("\n" + "=" * 60)
    print("BLUEPRINT CODE:")
    print("=" * 60)
    print(blueprint_code)
    print("=" * 60)

    # Decode and verify
    decoded = BlueprintEncoder.decode(blueprint_code)
    print(f"\nDecoded entries: {len(decoded['BP']['Entries'])}")
    for entry in decoded['BP']['Entries'][:10]:
        print(f"  {entry}")
    if len(decoded['BP']['Entries']) > 10:
        print(f"  ... and {len(decoded['BP']['Entries']) - 10} more")


if __name__ == "__main__":
    main()
