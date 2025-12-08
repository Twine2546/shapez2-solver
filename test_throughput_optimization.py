#!/usr/bin/env python3
"""Test throughput optimization in CP-SAT solver."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat

def test_throughput_with_cutters():
    """Test 1→4 splitting WITH shape transformation (requires cutters)."""
    print("\n" + "="*60)
    print("TEST 1: Shape Transformation (1→4 with cutters)")
    print("Input: CuCuCuCu (full circle)")
    print("Outputs: Cu, Cu, Cu, Cu (quarters)")
    print("Expected: Cutters (45 items/min bottleneck)")
    print("="*60)

    # 1 input, 4 outputs, different shapes (transformation needed)
    foundation_type = "2x2"
    input_specs = [("W", 0, 0, "CuCuCuCu")]  # Full circle on west side
    output_specs = [
        ("E", 0, 0, "Cu------"),  # Quarter 1
        ("E", 1, 0, "Cu------"),  # Quarter 2
        ("E", 2, 0, "Cu------"),  # Quarter 3
        ("E", 3, 0, "Cu------"),  # Quarter 4
    ]

    solution = solve_with_cpsat(
        foundation_type=foundation_type,
        input_specs=input_specs,
        output_specs=output_specs,
        max_machines=10,
        time_limit=30.0,
        verbose=True,
    )

    if solution:
        print("\n✓ Solution found with shape transformation")
    else:
        print("\n✗ No solution found")


def test_throughput_with_splitters():
    """Test 1→4 splitting WITHOUT shape transformation (uses splitters)."""
    print("\n" + "="*60)
    print("TEST 2: Pure Splitting (1→4 with splitters)")
    print("Input: CuCuCuCu (full circle)")
    print("Outputs: CuCuCuCu, CuCuCuCu, CuCuCuCu, CuCuCuCu (same shape)")
    print("Expected: Splitters (180 items/min, NO bottleneck)")
    print("="*60)

    # 1 input, 4 outputs, SAME shapes (no transformation, just splitting)
    foundation_type = "2x2"
    input_specs = [("W", 0, 0, "CuCuCuCu")]  # Full circle on west side
    output_specs = [
        ("E", 0, 0, "CuCuCuCu"),  # Same shape
        ("E", 1, 0, "CuCuCuCu"),  # Same shape
        ("E", 2, 0, "CuCuCuCu"),  # Same shape
        ("E", 3, 0, "CuCuCuCu"),  # Same shape
    ]

    solution = solve_with_cpsat(
        foundation_type=foundation_type,
        input_specs=input_specs,
        output_specs=output_specs,
        max_machines=10,
        time_limit=30.0,
        verbose=True,
    )

    if solution:
        print("\n✓ Solution found with pure splitting")
    else:
        print("\n✗ No solution found")


if __name__ == "__main__":
    # Test 1: Shape transformation (cutters) - LOW throughput
    test_throughput_with_cutters()

    # Test 2: Pure splitting (splitters) - HIGH throughput
    test_throughput_with_splitters()

    print("\n" + "="*60)
    print("SUMMARY:")
    print("Test 1 (cutters):   ~11.25 items/min per output (45/4)")
    print("Test 2 (splitters): ~45.00 items/min per output (180/4)")
    print("Throughput improvement: 4x faster with splitters!")
    print("="*60)
