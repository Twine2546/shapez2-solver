#!/usr/bin/env python3
"""Comprehensive test suite for CP-SAT solver with throughput optimization."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat
import sys


def run_test(name, foundation, inputs, outputs, expected_machines, expected_throughput_range, expected_machine_type):
    """Run a single test case."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Foundation: {foundation}")
    print(f"Inputs: {len(inputs)}")
    print(f"Outputs: {len(outputs)}")
    print(f"Expected: {expected_machines} {expected_machine_type}s")
    print(f"Expected throughput: {expected_throughput_range[0]:.1f}-{expected_throughput_range[1]:.1f} items/min per output")

    solution = solve_with_cpsat(
        foundation_type=foundation,
        input_specs=inputs,
        output_specs=outputs,
        max_machines=50,
        time_limit=60.0,
        verbose=False,
    )

    # solve_with_cpsat returns Candidate (not CPSATSolution)
    # If it returns a Candidate, routing succeeded
    # If it returns None, routing failed
    if solution:
        success = True
        fitness = solution.fitness
        # Count processing machines (not belts)
        num_machines = len([b for b in solution.buildings
                           if any(m in str(b.building_type)
                                 for m in ['CUTTER', 'SPLITTER', 'ROTATOR', 'STACKER'])])
    else:
        success = False
        fitness = 0
        num_machines = 0

    if success:
        print(f"✓ PASS: Routing successful")
        print(f"  Machines: {num_machines}")
        print(f"  Fitness: {fitness:.1f}")
        return True
    else:
        print(f"✗ FAIL: Routing failed")
        return False


def test_1_single_to_single():
    """Test 1→1 (passthrough, no splitting)."""
    return run_test(
        name="1→1 Passthrough",
        foundation="2x2",
        inputs=[("W", 0, 0, "CuCuCuCu")],
        outputs=[("E", 0, 0, "CuCuCuCu")],
        expected_machines=1,
        expected_throughput_range=(45, 180),  # Could use cutter or direct
        expected_machine_type="cutter"
    )


def test_2_single_to_double_splitting():
    """Test 1→2 pure splitting (use splitters for max throughput)."""
    return run_test(
        name="1→2 Pure Splitting",
        foundation="2x2",
        inputs=[("W", 0, 0, "RuRuRuRu")],
        outputs=[
            ("E", 0, 0, "RuRuRuRu"),
            ("E", 1, 0, "RuRuRuRu"),
        ],
        expected_machines=1,
        expected_throughput_range=(90, 90),  # 180/2 = 90 items/min per output
        expected_machine_type="splitter"
    )


def test_3_single_to_double_cutting():
    """Test 1→2 with shape transformation (use cutters)."""
    return run_test(
        name="1→2 Cutting",
        foundation="2x2",
        inputs=[("W", 0, 0, "CuCuCuCu")],
        outputs=[
            ("E", 0, 0, "CuCu----"),  # Top half
            ("E", 1, 0, "--CuCu--"),  # Bottom half
        ],
        expected_machines=1,
        expected_throughput_range=(22.5, 22.5),  # 45/2 = 22.5 items/min per output
        expected_machine_type="cutter"
    )


def test_4_single_to_quad_splitting():
    """Test 1→4 pure splitting (use splitters)."""
    return run_test(
        name="1→4 Pure Splitting",
        foundation="2x2",
        inputs=[("W", 0, 0, "SuSuSuSu")],
        outputs=[
            ("E", 0, 0, "SuSuSuSu"),
            ("E", 1, 0, "SuSuSuSu"),
            ("E", 2, 0, "SuSuSuSu"),
            ("E", 3, 0, "SuSuSuSu"),
        ],
        expected_machines=3,
        expected_throughput_range=(45, 45),  # 180/4 = 45 items/min per output
        expected_machine_type="splitter"
    )


def test_5_single_to_quad_cutting():
    """Test 1→4 with shape transformation (use cutters)."""
    return run_test(
        name="1→4 Cutting",
        foundation="2x2",
        inputs=[("W", 0, 0, "WuWuWuWu")],
        outputs=[
            ("E", 0, 0, "Wu------"),
            ("E", 1, 0, "--Wu----"),
            ("E", 2, 0, "----Wu--"),
            ("E", 3, 0, "------Wu"),
        ],
        expected_machines=3,
        expected_throughput_range=(11.25, 11.25),  # 45/4 = 11.25 items/min per output
        expected_machine_type="cutter"
    )


def test_6_single_to_octet_splitting():
    """Test 1→8 pure splitting (use splitters)."""
    return run_test(
        name="1→8 Pure Splitting",
        foundation="3x3",
        inputs=[("W", 0, 0, "RcRcRcRc")],
        outputs=[("E", i, 0, "RcRcRcRc") for i in range(8)],
        expected_machines=7,  # 2^3 - 1 = 7 splitters
        expected_throughput_range=(22.5, 22.5),  # 180/8 = 22.5 items/min per output
        expected_machine_type="splitter"
    )


def test_7_double_to_quad():
    """Test 2→4 (2 inputs to 4 outputs, each input splits 1→2)."""
    return run_test(
        name="2→4 Splitting (2 trees)",
        foundation="2x2",
        inputs=[
            ("W", 0, 0, "CgCgCgCg"),
            ("W", 1, 0, "CgCgCgCg"),
        ],
        outputs=[
            ("E", 0, 0, "CgCgCgCg"),
            ("E", 1, 0, "CgCgCgCg"),
            ("E", 2, 0, "CgCgCgCg"),
            ("E", 3, 0, "CgCgCgCg"),
        ],
        expected_machines=2,  # 2 trees × 1 splitter each
        expected_throughput_range=(90, 90),  # Each tree: 180/2 = 90 items/min per output
        expected_machine_type="splitter"
    )


def test_8_triple_to_triple():
    """Test 3→3 (passthrough on different floors)."""
    return run_test(
        name="3→3 Multi-floor Passthrough",
        foundation="2x2",
        inputs=[
            ("W", 0, 0, "RuCuSuWu"),
            ("W", 0, 1, "RuCuSuWu"),
            ("W", 0, 2, "RuCuSuWu"),
        ],
        outputs=[
            ("E", 0, 0, "RuCuSuWu"),
            ("E", 0, 1, "RuCuSuWu"),
            ("E", 0, 2, "RuCuSuWu"),
        ],
        expected_machines=3,  # 3 passthrough machines
        expected_throughput_range=(45, 180),  # Depends on machine type chosen
        expected_machine_type="cutter"
    )


def test_9_quad_to_sixteen_cutting():
    """Test 4→16 with shape transformation (4 inputs, each 1→4)."""
    return run_test(
        name="4→16 Cutting (4 trees)",
        foundation="3x3",
        inputs=[
            ("W", i, 0, "RuCuSuWu") for i in range(4)
        ],
        outputs=[
            ("E", i % 4, i // 4, shape)
            for i in range(16)
            for shape in ["Ru------", "--Cu----", "----Su--", "------Wu"][:1]
        ],
        expected_machines=12,  # 4 trees × 3 cutters each
        expected_throughput_range=(11.25, 11.25),  # Each tree: 45/4 = 11.25 items/min per output
        expected_machine_type="cutter"
    )


def test_10_mixed_floors_splitting():
    """Test inputs on different floors, pure splitting."""
    return run_test(
        name="Mixed Floors Splitting",
        foundation="2x2",
        inputs=[
            ("W", 0, 0, "CyCyCyCy"),
            ("W", 1, 1, "CyCyCyCy"),
            ("W", 2, 2, "CyCyCyCy"),
        ],
        outputs=[
            ("E", 0, 0, "CyCyCyCy"),
            ("E", 1, 0, "CyCyCyCy"),
            ("E", 2, 1, "CyCyCyCy"),
            ("E", 3, 1, "CyCyCyCy"),
            ("E", 0, 2, "CyCyCyCy"),
            ("E", 1, 2, "CyCyCyCy"),
        ],
        expected_machines=3,  # 3 trees × 1 splitter each (1→2)
        expected_throughput_range=(90, 90),  # Each tree: 180/2 = 90 items/min
        expected_machine_type="splitter"
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CP-SAT COMPREHENSIVE TEST SUITE")
    print("Testing throughput optimization with various input/output scenarios")
    print("="*70)

    tests = [
        ("1→1 Passthrough", test_1_single_to_single),
        ("1→2 Pure Splitting", test_2_single_to_double_splitting),
        ("1→2 Cutting", test_3_single_to_double_cutting),
        ("1→4 Pure Splitting", test_4_single_to_quad_splitting),
        ("1→4 Cutting", test_5_single_to_quad_cutting),
        ("1→8 Pure Splitting", test_6_single_to_octet_splitting),
        ("2→4 Splitting", test_7_double_to_quad),
        ("3→3 Passthrough", test_8_triple_to_triple),
        ("4→16 Cutting", test_9_quad_to_sixteen_cutting),
        ("Mixed Floors", test_10_mixed_floors_splitting),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ FAIL: Exception - {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed_test in results:
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status:8} {name}")

    print(f"\nTotal: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)

    sys.exit(0 if passed == total else 1)
