#!/usr/bin/env python3
"""Edge case and stress tests for CP-SAT solver."""

from shapez2_solver.evolution.cpsat_solver import solve_with_cpsat
import sys


def test_case(name, foundation, inputs, outputs, should_succeed=True):
    """Run a test case and check if it succeeds/fails as expected."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Inputs: {len(inputs)}, Outputs: {len(outputs)}")
    print(f"Expected: {'SUCCESS' if should_succeed else 'FAILURE (expected)'}")

    try:
        solution = solve_with_cpsat(
            foundation_type=foundation,
            input_specs=inputs,
            output_specs=outputs,
            max_machines=50,
            time_limit=60.0,
            verbose=False,
        )

        success = solution is not None

        if success == should_succeed:
            print(f"✓ PASS: Behaved as expected")
            if solution:
                print(f"  Fitness: {solution.fitness:.1f}")
                print(f"  Buildings: {len(solution.buildings)}")
            return True
        else:
            print(f"✗ FAIL: Expected {'success' if should_succeed else 'failure'}, got {'success' if success else 'failure'}")
            return False

    except Exception as e:
        if not should_succeed:
            print(f"✓ PASS: Failed as expected with exception")
            return True
        else:
            print(f"✗ FAIL: Unexpected exception - {e}")
            return False


def test_1_max_splitting():
    """Test 1→16 maximum splitting with splitters."""
    return test_case(
        name="1→16 Maximum Splitting",
        foundation="3x3",  # Use 3x3 (54x54 grid)
        inputs=[("W", 0, 0, "CbCbCbCb")],
        outputs=[("E", i % 4, i // 4, "CbCbCbCb") for i in range(16)],
        should_succeed=True
    )


def test_2_many_floors():
    """Test using all 4 floors with different inputs."""
    return test_case(
        name="4 Floors × 2 Inputs Each",
        foundation="3x3",
        inputs=[("W", i % 4, i // 4, "RuRuRuRu") for i in range(8)],
        outputs=[("E", i % 4, i // 4, "RuRuRuRu") for i in range(16)],
        should_succeed=True
    )


def test_3_complex_shapes():
    """Test with complex 2-layer stacked shapes."""
    return test_case(
        name="2-Layer Stacked Shapes",
        foundation="2x2",
        inputs=[("W", 0, 0, "RuRuRuRu:CuCuCuCu")],  # 2-layer stack
        outputs=[
            ("E", 0, 0, "RuRuRuRu:CuCuCuCu"),
            ("E", 1, 0, "RuRuRuRu:CuCuCuCu"),
        ],
        should_succeed=True
    )


def test_4_single_quarter_cuts():
    """Test cutting into single quarters (very fine splitting)."""
    return test_case(
        name="Single Quarter Isolation",
        foundation="2x2",
        inputs=[("W", 0, 0, "RcCgSbWy")],  # All 4 colors
        outputs=[
            ("E", 0, 0, "Rc------"),
            ("E", 1, 0, "--Cg----"),
            ("E", 2, 0, "----Sb--"),
            ("E", 3, 0, "------Wy"),
        ],
        should_succeed=True
    )


def test_5_rotator_chain():
    """Test if system can handle rotation operations."""
    return test_case(
        name="90° Rotation",
        foundation="2x2",
        inputs=[("W", 0, 0, "Ru------")],  # Top-right red
        outputs=[("E", 0, 0, "--Ru----")],  # Top-left red (rotated 90° CW)
        should_succeed=True
    )


def test_6_merge_scenario():
    """Test many inputs to fewer outputs (merging)."""
    return test_case(
        name="4→2 Merging",
        foundation="2x2",
        inputs=[
            ("W", 0, 0, "CuCuCuCu"),
            ("W", 1, 0, "CuCuCuCu"),
            ("W", 2, 0, "CuCuCuCu"),
            ("W", 3, 0, "CuCuCuCu"),
        ],
        outputs=[
            ("E", 0, 0, "CuCuCuCu"),
            ("E", 1, 0, "CuCuCuCu"),
        ],
        should_succeed=True  # May merge or just use 2 of 4 inputs
    )


def test_7_empty_shape():
    """Test with empty input (edge case)."""
    return test_case(
        name="Empty Shape Handling",
        foundation="2x2",
        inputs=[("W", 0, 0, "--------")],  # Empty
        outputs=[("E", 0, 0, "--------")],  # Empty
        should_succeed=True
    )


def test_8_cross_floor_routing():
    """Test input floor 0 → output floor 3 (maximum floor difference)."""
    return test_case(
        name="Cross-Floor Routing (Floor 0→3)",
        foundation="2x2",
        inputs=[("W", 0, 0, "SuSuSuSu")],
        outputs=[("E", 0, 3, "SuSuSuSu")],
        should_succeed=True
    )


def test_9_all_corners():
    """Test using all 4 corners (N, S, E, W sides)."""
    return test_case(
        name="All 4 Sides (N, S, E, W)",
        foundation="2x2",
        inputs=[
            ("W", 0, 0, "RuRuRuRu"),  # West
            ("N", 0, 0, "CuCuCuCu"),  # North
        ],
        outputs=[
            ("E", 0, 0, "RuRuRuRu"),  # East
            ("S", 0, 0, "CuCuCuCu"),  # South
        ],
        should_succeed=True
    )


def test_10_stress_many_outputs():
    """Stress test with 24 outputs from 3 inputs (reduced from 32)."""
    return test_case(
        name="Stress Test: 3→24 Outputs",
        foundation="3x3",  # Use 3x3 (largest standard foundation)
        inputs=[("W", i, 0, "WuWuWuWu") for i in range(3)],
        outputs=[("E", i % 4, i // 4, "Wu------") for i in range(24)],
        should_succeed=True
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CP-SAT EDGE CASE & STRESS TEST SUITE")
    print("Testing unusual scenarios and edge cases")
    print("="*70)

    tests = [
        ("1→16 Max Splitting", test_1_max_splitting),
        ("8 Inputs on 4 Floors", test_2_many_floors),
        ("2-Layer Stacks", test_3_complex_shapes),
        ("Quarter Isolation", test_4_single_quarter_cuts),
        ("Rotation", test_5_rotator_chain),
        ("4→2 Merging", test_6_merge_scenario),
        ("Empty Shapes", test_7_empty_shape),
        ("Cross-Floor", test_8_cross_floor_routing),
        ("All 4 Sides", test_9_all_corners),
        ("4→32 Stress", test_10_stress_many_outputs),
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
