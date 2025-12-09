#!/usr/bin/env python3
"""Test auto-scaling feature with 4 inputs → 16 outputs (corner splitting)."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

# 4 inputs: full circles on west side
inputs = [
    ("W", 0, 0, "CuCuCuCu"),
    ("W", 1, 0, "CuCuCuCu"),
    ("W", 2, 0, "CuCuCuCu"),
    ("W", 3, 0, "CuCuCuCu"),
]

# 16 outputs: 4 corners × 4 inputs, distributed across east side
outputs = [
    # From input 0
    ("E", 0, 0, "Cu------"),
    ("E", 1, 0, "--Cu----"),
    ("E", 2, 0, "----Cu--"),
    ("E", 3, 0, "------Cu"),
    # From input 1
    ("E", 4, 0, "Cu------"),
    ("E", 5, 0, "--Cu----"),
    ("E", 6, 0, "----Cu--"),
    ("E", 7, 0, "------Cu"),
    # From input 2
    ("E", 0, 1, "Cu------"),
    ("E", 1, 1, "--Cu----"),
    ("E", 2, 1, "----Cu--"),
    ("E", 3, 1, "------Cu"),
    # From input 3
    ("E", 4, 1, "Cu------"),
    ("E", 5, 1, "--Cu----"),
    ("E", 6, 1, "----Cu--"),
    ("E", 7, 1, "------Cu"),
]

print("="*70)
print("AUTO-SCALING TEST: 4 Inputs → 16 Outputs (Corner Splitting)")
print("="*70)
print(f"Inputs: {len(inputs)}")
print(f"Outputs: {len(outputs)}")
print(f"Expected machines: 12 cutters (4 inputs × 3 cutters per tree)")
print("="*70)

# Foundation progression to try
foundations = ["1x1", "2x2", "3x3"]

for foundation in foundations:
    print(f"\n{'='*70}")
    print(f"TRYING FOUNDATION: {foundation}")
    print(f"{'='*70}")

    solution = solve_with_cpsat(
        foundation_type=foundation,
        input_specs=inputs,
        output_specs=outputs,
        max_machines=50,
        time_limit=60,
        verbose=True
    )

    if solution and hasattr(solution, 'routing_success') and solution.routing_success:
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS with {foundation}!")
        print(f"{'='*70}")
        print(f"Machines: {len([b for b in solution.buildings if 'CUTTER' in str(b.building_type)])}")
        print(f"Belts: {len([b for b in solution.buildings if 'BELT' in str(b.building_type)])}")
        print(f"Fitness: {solution.fitness}")
        break
    else:
        print(f"\n⚠ {foundation} FAILED - routing incomplete")
        print(f"Trying next foundation size...")
else:
    print(f"\n{'='*70}")
    print("✗ All foundations exhausted without success")
    print(f"{'='*70}")
