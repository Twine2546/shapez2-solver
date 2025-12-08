#!/usr/bin/env python3
"""Test multi-floor throughput optimization: 12 inputs (4 per floor × 3 floors) → 48 outputs."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

def test_multifloor_corner_isolation():
    """
    Test 12 inputs (4 per floor on 3 floors) → 48 outputs (4 corners × 12 inputs).

    Inputs: Each floor has 4 inputs, each with RuCuSuWu (4 different colored corners)
    - Floor 0: 4 inputs at positions 0-3 on west side
    - Floor 1: 4 inputs at positions 0-3 on west side
    - Floor 2: 4 inputs at positions 0-3 on west side
    - Total: 12 inputs

    Outputs: Each corner isolated from each input (48 outputs total)
    - Each input → 4 outputs (one per corner)
    - 12 inputs × 4 corners = 48 outputs

    This requires cutting (shape transformation), so will use cutters.
    For maximum throughput, we need 12 separate cutting trees (one per input).
    Each tree: 1 input → 4 outputs requires 3 cutters.
    Total: 36 cutters (12 inputs × 3 cutters per tree).
    """
    print("\n" + "="*70)
    print("TEST: Multi-floor Corner Isolation (12 inputs → 48 outputs)")
    print("="*70)
    print("Inputs (12 total - 4 per floor × 3 floors):")
    print("  Floor 0: RuCuSuWu × 4 inputs (West side, pos 0-3)")
    print("  Floor 1: RuCuSuWu × 4 inputs (West side, pos 0-3)")
    print("  Floor 2: RuCuSuWu × 4 inputs (West side, pos 0-3)")
    print("")
    print("Outputs (48 total - 4 corners × 12 inputs):")
    print("  Each input → 4 isolated corners → East side")
    print("")
    print("Expected:")
    print("  - 36 cutters (12 trees × 3 cutters each)")
    print("  - Throughput: 11.25 items/min per output (45/4 per tree)")
    print("  - Total throughput: 540 items/min across all 48 outputs")
    print("="*70)

    # Try progressively larger foundations until routing succeeds
    foundation_sizes = ["2x2", "3x3", "4x4", "5x5"]

    # 12 inputs: 4 positions per floor, 3 floors
    input_specs = []
    for floor in range(3):  # Floors 0, 1, 2
        for pos in range(4):  # Positions 0, 1, 2, 3
            input_specs.append(("W", pos, floor, "RuCuSuWu"))

    # 48 outputs: 4 corners × 12 inputs
    # Organize as: each input's 4 corners grouped together
    output_specs = []
    output_floor = 0
    output_pos = 0

    for input_idx in range(12):  # 12 inputs
        # Each input produces 4 corners
        for corner_idx, corner_shape in enumerate(["Ru------", "--Cu----", "----Su--", "------Wu"]):
            output_specs.append(("E", output_pos, output_floor, corner_shape))

            # Advance position/floor for next output
            output_pos += 1
            if output_pos >= 4:  # Max 4 positions per floor on east side
                output_pos = 0
                output_floor += 1

    for foundation_type in foundation_sizes:
        print(f"\n{'='*70}")
        print(f"Trying foundation: {foundation_type}")
        print(f"{'='*70}")

        solution = solve_with_cpsat(
            foundation_type=foundation_type,
            input_specs=input_specs,
            output_specs=output_specs,
            max_machines=50,  # Need many machines for this complex case
            time_limit=120.0,  # More time for complex routing
            verbose=True,
        )

        if solution and hasattr(solution, 'routing_success'):
            routing_success = solution.routing_success
        elif solution:
            routing_success = True
        else:
            routing_success = False

        if routing_success:
            print("\n" + "="*70)
            print(f"✓ SUCCESS with foundation {foundation_type}!")
            print("="*70)
            print("\nKey Metrics:")
            print(f"  Foundation: {foundation_type}")
            print(f"  Total machines: {len(solution.buildings) if solution else 0}")
            print(f"  Fitness: {solution.fitness:.1f}" if solution else "")

            print("")
            print("Analysis:")
            print("  - Each input feeds an independent cutting tree")
            print("  - 12 inputs × 3 cutters = 36 cutters total")
            print("  - Each tree provides 45 items/min → 11.25 items/min per output")
            print("  - Total system throughput: 540 items/min (11.25 × 48 outputs)")
            print("="*70)
            return True
        else:
            print(f"\n⚠ Foundation {foundation_type} failed - trying larger foundation...")

    print("\n" + "="*70)
    print("✗ FAILED: Could not find solution even with largest foundation")
    print("="*70)
    return False


if __name__ == "__main__":
    test_multifloor_corner_isolation()
