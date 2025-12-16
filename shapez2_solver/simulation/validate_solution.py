"""
Validate a CP-SAT solver solution using the flow simulator.

Takes a solution (Candidate) and input/output specs, runs flow simulation,
and reports any issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Tuple, Optional
from shapez2_solver.simulation.flow_simulator import FlowSimulator, FlowReport
from shapez2_solver.blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS
from shapez2_solver.solver.foundation_config import FOUNDATION_SPECS, Side


def validate_solution(
    solution,  # Candidate or CPSATSolution.to_candidate()
    input_specs: List[Tuple[str, int, int, str]],  # (side, port_x, port_y, shape)
    output_specs: List[Tuple[str, int, int, str]],  # (side, port_x, port_y, expected_shape)
    foundation_type: str = "1x1",
    verbose: bool = True,
) -> FlowReport:
    """
    Validate a solver solution using flow simulation.
    
    Args:
        solution: The placement solution (Candidate with buildings)
        input_specs: List of (side, port_x, port_y, shape) for inputs
        output_specs: List of (side, port_x, port_y, expected_shape) for outputs
        foundation_type: Foundation type for grid dimensions
        verbose: Print detailed output
    
    Returns:
        FlowReport with validation results
    """
    spec = FOUNDATION_SPECS[foundation_type]
    sim = FlowSimulator(spec.grid_width, spec.grid_height, spec.num_floors)
    
    if verbose:
        print(f"\n{'='*70}")
        print("VALIDATING SOLUTION")
        print(f"{'='*70}")
        print(f"Foundation: {foundation_type} ({spec.grid_width}x{spec.grid_height})")
        print(f"Buildings: {len(solution.buildings)}")
    
    # Add all buildings from solution
    for b in solution.buildings:
        sim.place_building(
            b.building_type,
            b.x, b.y, b.floor,
            b.rotation
        )
    
    # Add inputs
    for side_str, px, py, shape in input_specs:
        # Convert side string to Side enum
        side_map = {'W': Side.WEST, 'E': Side.EAST, 'N': Side.NORTH, 'S': Side.SOUTH,
                    'WEST': Side.WEST, 'EAST': Side.EAST, 'NORTH': Side.NORTH, 'SOUTH': Side.SOUTH}
        side = side_map.get(side_str.upper(), Side.WEST)

        # Calculate port index from px, py (port_x within unit, port_y = unit index)
        port_index = py * 4 + px

        # Get grid position for this port
        x, y = spec.get_port_grid_position(side, port_index)

        sim.set_input(x, y, 0, shape, 180.0)
        if verbose:
            print(f"  Input: side={side.value} port={port_index} -> ({x}, {y}) = {shape}")

    # Add outputs
    for side_str, px, py, expected_shape in output_specs:
        side_map = {'W': Side.WEST, 'E': Side.EAST, 'N': Side.NORTH, 'S': Side.SOUTH,
                    'WEST': Side.WEST, 'EAST': Side.EAST, 'NORTH': Side.NORTH, 'SOUTH': Side.SOUTH}
        side = side_map.get(side_str.upper(), Side.EAST)

        port_index = py * 4 + px
        x, y = spec.get_port_grid_position(side, port_index)

        sim.set_output(x, y, 0, expected_shape)
        if verbose:
            print(f"  Output: side={side.value} port={port_index} -> ({x}, {y}) expects {expected_shape}")
    
    # Run simulation
    report = sim.simulate()
    
    if verbose:
        sim.print_grid(0)
        print(report)
    
    return report


def test_with_solver():
    """Test validation with actual solver output."""
    from shapez2_solver.solver.cpsat_solver import solve_with_cpsat
    
    print("Running solver...")
    
    input_specs = [('W', 0, 0, 'CuCuCuCu'), ('W', 0, 1, 'CuCuCuCu')]
    output_specs = [('E', 0, 0, 'Cu------'), ('E', 0, 1, 'Cu------')]
    
    result = solve_with_cpsat(
        foundation_type='1x1',
        input_specs=input_specs,
        output_specs=output_specs,
        time_limit=30,
        verbose=False,
    )
    
    if result:
        print(f"\nSolver found solution with {len(result.buildings)} buildings")
        
        # Validate
        report = validate_solution(
            result,
            input_specs,
            output_specs,
            foundation_type='1x1',
            verbose=True,
        )
        
        if report.is_valid():
            print("\n🎉 SOLUTION IS VALID!")
        else:
            print("\n💥 SOLUTION HAS ISSUES!")
    else:
        print("Solver failed to find solution")


if __name__ == "__main__":
    test_with_solver()
