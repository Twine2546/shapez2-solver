#!/usr/bin/env python3
"""Test port counts for irregular foundations."""

from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, FoundationConfig

print("="*70)
print("IRREGULAR FOUNDATION PORT COUNTS")
print("="*70)

irregular_foundations = ["T", "L", "L4", "S4", "Cross"]

for name in irregular_foundations:
    spec = FOUNDATION_SPECS[name]
    config = FoundationConfig(spec)

    pps = spec.ports_per_side
    from shapez2_solver.evolution.foundation_config import Side

    print(f"\n{name} Foundation:")
    print(f"  Layout: {spec.present_cells}")
    print(f"  North ports: {pps[Side.NORTH]} (indices 0-{pps[Side.NORTH]-1})")
    print(f"  South ports: {pps[Side.SOUTH]} (indices 0-{pps[Side.SOUTH]-1})")
    print(f"  East ports: {pps[Side.EAST]} (indices 0-{pps[Side.EAST]-1})")
    print(f"  West ports: {pps[Side.WEST]} (indices 0-{pps[Side.WEST]-1})")
    print(f"  Total: {spec.total_ports_per_floor} per floor")

print("\n" + "="*70)
print("CROSS FOUNDATION DETAIL")
print("="*70)

# Verify Cross specifically
spec = FOUNDATION_SPECS["Cross"]
cells = set(spec.present_cells)

print(f"\nCross units: {sorted(spec.present_cells)}")
print(f"  (1,0): top middle")
print(f"  (0,1), (1,1), (2,1): middle row (left, center, right)")
print(f"  (1,2): bottom middle")

print(f"\nExposed faces:")
for x, y in sorted(cells):
    faces = []
    if (x, y - 1) not in cells:
        faces.append("North")
    if (x, y + 1) not in cells:
        faces.append("South")
    if (x - 1, y) not in cells:
        faces.append("West")
    if (x + 1, y) not in cells:
        faces.append("East")
    print(f"  Unit ({x},{y}): {', '.join(faces) if faces else 'none (center)'}")

print(f"\nExpected: 48 total ports (12 per side)")
print(f"Actual: {spec.total_ports_per_floor} total ports")
print(f"  North: {pps[Side.NORTH]}")
print(f"  South: {pps[Side.SOUTH]}")
print(f"  East: {pps[Side.EAST]}")
print(f"  West: {pps[Side.WEST]}")
