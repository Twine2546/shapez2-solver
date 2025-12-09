#!/usr/bin/env python3
"""Test improved solver heuristics with iteration tracking."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

# Simple 4→4 corner split
inputs = [("W", 0, 0, "CuCuCuCu")]
outputs = [
    ("E", 0, 0, "Cu------"),
    ("E", 1, 0, "--Cu----"),
    ("E", 2, 0, "----Cu--"),
    ("E", 3, 0, "------Cu"),
]

print("="*70)
print("TEST: Improved Solver Heuristics")
print("="*70)
print("Testing: 1 input → 4 outputs corner split")
print("Expected: 3 cutters, should succeed in multiple iterations")
print("Timeout: 60 seconds")
print("="*70)
print("\nImprovement Features:")
print("  ✓ No hard iteration limit (was 10, now unlimited)")
print("  ✓ Each placement gets 30s max (was 60s)")
print("  ✓ Early iterations spread machines out more")
print("  ✓ Later iterations try compact placements")
print("  ✓ Different random seed each iteration for diversity")
print("="*70)

solution = solve_with_cpsat(
    foundation_type="2x2",
    input_specs=inputs,
    output_specs=outputs,
    max_machines=10,
    time_limit=60,
    verbose=True
)

if solution:
    print(f"\n{'='*70}")
    print("RESULT:")
    print(f"{'='*70}")
    print(f"Routing success: {solution.routing_success}")
    print(f"Fitness: {solution.fitness}")
    machines = len([b for b in solution.buildings if 'CUTTER' in str(b.building_type)])
    belts = len([b for b in solution.buildings if 'BELT' in str(b.building_type)])
    print(f"Machines: {machines}")
    print(f"Belts: {belts}")
else:
    print("\n✗ No solution found")
