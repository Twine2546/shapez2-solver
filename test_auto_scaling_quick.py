#!/usr/bin/env python3
"""Quick test of auto-scaling with 1 input → 4 outputs."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

# Simple test: 1 input, 4 outputs
inputs = [("W", 0, 0, "CuCuCuCu")]
outputs = [
    ("E", 0, 0, "Cu------"),
    ("E", 1, 0, "--Cu----"),
    ("E", 2, 0, "----Cu--"),
    ("E", 3, 0, "------Cu"),
]

print("Testing auto-scaling with 1→4 corner split")
print("Expected: 3 cutters in binary tree")

# Try 2x2 directly (should succeed immediately)
print("\n" + "="*70)
print("TRYING 2x2 FOUNDATION")
print("="*70)

solution = solve_with_cpsat(
    foundation_type="2x2",
    input_specs=inputs,
    output_specs=outputs,
    max_machines=10,
    time_limit=10,  # Short timeout
    verbose=False  # Less output
)

if solution:
    print(f"\n✓ SOLUTION STATUS:")
    print(f"  routing_success = {solution.routing_success}")
    print(f"  fitness = {solution.fitness}")
    print(f"  machines = {len([b for b in solution.buildings if 'CUTTER' in str(b.building_type)])}")
    print(f"  belts = {len([b for b in solution.buildings if 'BELT' in str(b.building_type)])}")

    if solution.routing_success:
        print("\n✓✓✓ SUCCESS - routing_success is True!")
    else:
        print("\n✗✗✗ FAILED - routing_success is False!")
else:
    print("\n✗ No solution returned")
