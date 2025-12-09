#!/usr/bin/env python3
"""Test 12 inputs → 48 outputs corner split scenario."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

# 12 inputs on South side (4 ports × 3 floors)
inputs = []
for floor in range(3):
    for pos in range(4):
        inputs.append(("S", pos, floor, "CuCuCuCu"))

# 48 outputs (12 per side: W, N, E, and top of N)
outputs = []

# 12 on West
for floor in range(3):
    for pos in range(4):
        outputs.append(("W", pos, floor, "Cu------"))

# 12 on North (bottom positions 0-3)
for floor in range(3):
    for pos in range(4):
        outputs.append(("N", pos, floor, "--Cu----"))

# 12 on East
for floor in range(3):
    for pos in range(4):
        outputs.append(("E", pos, floor, "----Cu--"))

# 12 on North (top positions 8-11)
for floor in range(3):
    for pos in range(8, 12):
        outputs.append(("N", pos, floor, "------Cu"))

print("="*70)
print("TEST: 12 inputs → 48 outputs on T foundation")
print("="*70)
print(f"Inputs: {len(inputs)} (South side, 4 ports × 3 floors)")
print(f"Outputs: {len(outputs)} (12 on W, 12 on N bottom, 12 on E, 12 on N top)")
print("Expected: 36 cutters (12 binary trees × 3 cutters each)")
print("="*70)

solution = solve_with_cpsat(
    foundation_type="T",
    input_specs=inputs,
    output_specs=outputs,
    max_machines=50,
    time_limit=120,  # 2 minutes timeout
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
    print(f"Cutters: {machines}")
    print(f"Belts: {belts}")

    if solution.routing_success:
        print(f"\n✓✓✓ SUCCESS!")
    else:
        print(f"\n✗ Routing failed - foundation may be too small or timeout too short")
else:
    print("\n✗ No solution found")
