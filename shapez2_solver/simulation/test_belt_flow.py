"""
Test cases for belt merging and splitting flow.

Run with: python3 -m shapez2_solver.simulation.test_belt_flow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shapez2_solver.simulation.flow_simulator import FlowSimulator
from shapez2_solver.blueprint.building_types import BuildingType, Rotation


def print_grid(sim, floor=0, width=None, height=None):
    """Print a simple ASCII grid showing throughputs."""
    width = width or sim.width
    height = height or sim.height

    print(f"\n  ", end="")
    for x in range(width):
        print(f"{x:>4}", end="")
    print()

    for y in range(height):
        print(f"{y:>2}", end="")
        for x in range(width):
            cell = sim.cells.get((x, y, floor))
            if cell and cell.building_type:
                tp = cell.throughput
                if tp > 0:
                    print(f"{tp:>4.0f}", end="")
                else:
                    # Show direction
                    arrows = {
                        Rotation.EAST: "  →",
                        Rotation.WEST: "  ←",
                        Rotation.NORTH: "  ↑",
                        Rotation.SOUTH: "  ↓",
                    }
                    print(f"{arrows.get(cell.rotation, '  ?'):>4}", end="")
            else:
                print("   .", end="")
        print()


def test_simple_belt_line():
    """Test: Simple line of belts going right."""
    print("\n" + "="*60)
    print("TEST: Simple belt line (5 belts going EAST)")
    print("Expected: All belts should show 45 throughput")
    print("="*60)

    sim = FlowSimulator(7, 3, 1)

    # Place 5 belts in a row
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x + 1, 1, 0, Rotation.EAST)

    # Input on left, output on right
    sim.set_input(1, 1, 0, "CuCuCuCu", 45.0)
    sim.set_output(5, 1, 0)

    sim.simulate()
    print_grid(sim, width=7, height=3)

    # Check results
    output = sim.outputs[0]
    print(f"\nOutput throughput: {output['throughput']} (expected: 45)")
    print(f"Output shape: {output['actual_shape']}")

    return output['throughput'] == 45.0


def test_belt_merge_two_inputs():
    """Test: Two belts merging into one."""
    print("\n" + "="*60)
    print("TEST: Two belts merging into one")
    print("  45 →")
    print("      →→→ (should be 90)")
    print("  45 →")
    print("Expected: Downstream belt should show 90 throughput")
    print("="*60)

    sim = FlowSimulator(8, 5, 1)

    # Top input belt
    sim.place_building(BuildingType.BELT_FORWARD, 1, 1, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 1, 0, Rotation.SOUTH)  # Turn down

    # Bottom input belt
    sim.place_building(BuildingType.BELT_FORWARD, 1, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 3, 0, Rotation.NORTH)  # Turn up

    # Merge point and continuation
    sim.place_building(BuildingType.BELT_FORWARD, 2, 2, 0, Rotation.EAST)  # Merge point
    sim.place_building(BuildingType.BELT_FORWARD, 3, 2, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 2, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 2, 0, Rotation.EAST)

    # Two inputs
    sim.set_input(1, 1, 0, "CuCuCuCu", 45.0)
    sim.set_input(1, 3, 0, "CuCuCuCu", 45.0)
    sim.set_output(5, 2, 0)

    sim.simulate()
    print_grid(sim, width=8, height=5)

    output = sim.outputs[0]
    print(f"\nOutput throughput: {output['throughput']} (expected: 90)")

    # Check merge point
    merge_cell = sim.cells.get((2, 2, 0))
    print(f"Merge point (2,2) throughput: {merge_cell.throughput if merge_cell else 'N/A'}")

    return output['throughput'] == 90.0


def test_belt_split_one_side():
    """Test: Belt can split to ONE perpendicular side branch."""
    print("\n" + "="*60)
    print("TEST: Belt splits to primary + ONE side branch")
    print("         → 22.5 (accepts from south)")
    print("  45 →→→→→ 22.5")
    print("         → (second side - should NOT get flow)")
    print("Expected: Split between primary and first side branch only")
    print("="*60)

    sim = FlowSimulator(8, 5, 1)

    # Input belt line going EAST
    sim.place_building(BuildingType.BELT_FORWARD, 1, 2, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 2, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 2, 0, Rotation.EAST)  # This one will split
    sim.place_building(BuildingType.BELT_FORWARD, 4, 2, 0, Rotation.EAST)  # Primary continues

    # Top branch - accepts from south (where main belt is)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 1, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 1, 0, Rotation.EAST)

    # Bottom branch - also could accept from north, but only ONE side branch allowed
    sim.place_building(BuildingType.BELT_FORWARD, 3, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 3, 0, Rotation.EAST)

    sim.set_input(1, 2, 0, "CuCuCuCu", 45.0)
    sim.set_output(4, 1, 0)
    sim.set_output(4, 2, 0)
    sim.set_output(4, 3, 0)

    sim.simulate()
    print_grid(sim, width=8, height=5)

    print("\nOutput throughputs:")
    for i, out in enumerate(sim.outputs):
        print(f"  Output {i} at {out['position']}: {out['throughput']}")

    # Should split to primary + ONE side (first found)
    main_output = sim.outputs[1]['throughput']  # (4,2)
    top_output = sim.outputs[0]['throughput']   # (4,1)
    bot_output = sim.outputs[2]['throughput']   # (4,3)

    total = main_output + top_output + bot_output
    print(f"\nMain (4,2): {main_output}")
    print(f"Top (4,1): {top_output}")
    print(f"Bottom (4,3): {bot_output}")
    print(f"Total: {total} (expected: 45)")

    # Should have exactly 2 outputs (primary + one side), total should be 45
    num_outputs = (1 if main_output > 0 else 0) + (1 if top_output > 0 else 0) + (1 if bot_output > 0 else 0)
    return abs(total - 45.0) < 0.1 and num_outputs == 2


def test_straight_belt_no_split():
    """Test: A straight belt should NOT split to perpendicular belts unless they accept."""
    print("\n" + "="*60)
    print("TEST: Straight belt should not split randomly")
    print("       ← (separate, going west)")
    print("  45 →→→→ (should stay 45, not split)")
    print("       ← (separate, going west)")
    print("="*60)

    sim = FlowSimulator(8, 5, 1)

    # Main line going EAST
    for x in range(1, 6):
        sim.place_building(BuildingType.BELT_FORWARD, x, 2, 0, Rotation.EAST)

    # Perpendicular belts going WEST (should NOT connect)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 1, 0, Rotation.WEST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 3, 0, Rotation.WEST)

    sim.set_input(1, 2, 0, "CuCuCuCu", 45.0)
    sim.set_output(5, 2, 0)

    sim.simulate()
    print_grid(sim, width=8, height=5)

    output = sim.outputs[0]
    print(f"\nOutput throughput: {output['throughput']} (expected: 45, no splitting)")

    # Check perpendicular belts have 0 throughput
    top_belt = sim.cells.get((3, 1, 0))
    bot_belt = sim.cells.get((3, 3, 0))
    print(f"Top perpendicular (3,1) throughput: {top_belt.throughput if top_belt else 'N/A'} (expected: 0)")
    print(f"Bottom perpendicular (3,3) throughput: {bot_belt.throughput if bot_belt else 'N/A'} (expected: 0)")

    return output['throughput'] == 45.0


def test_t_junction_merge():
    """Test: T-junction where side belts merge into main."""
    print("\n" + "="*60)
    print("TEST: T-junction merge")
    print("         ↓ 45")
    print("  45 →→→→→→→ (should be 90)")
    print("Expected: After junction, throughput should be 90")
    print("="*60)

    sim = FlowSimulator(10, 5, 1)

    # Main line going EAST
    for x in range(1, 8):
        sim.place_building(BuildingType.BELT_FORWARD, x, 2, 0, Rotation.EAST)

    # Side belt coming from NORTH (going SOUTH into junction at x=4)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 0, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 1, 0, Rotation.SOUTH)

    sim.set_input(1, 2, 0, "CuCuCuCu", 45.0)
    sim.set_input(4, 0, 0, "CuCuCuCu", 45.0)
    sim.set_output(7, 2, 0)

    sim.simulate()
    print_grid(sim, width=10, height=5)

    output = sim.outputs[0]
    print(f"\nOutput throughput: {output['throughput']} (expected: 90)")

    # Check belt after merge
    after_merge = sim.cells.get((5, 2, 0))
    print(f"Belt after merge (5,2) throughput: {after_merge.throughput if after_merge else 'N/A'}")

    return output['throughput'] == 90.0


def test_180_conflict():
    """Test: Can't split to opposite directions (180 degrees)."""
    print("\n" + "="*60)
    print("TEST: 180-degree split conflict")
    print("  ↑ (should NOT get flow)")
    print("  → (main direction)")
    print("  ↓ (should NOT get flow)")
    print("Expected: Flow only goes in primary direction, not N+S")
    print("="*60)

    sim = FlowSimulator(8, 5, 1)

    # Main belt going EAST
    sim.place_building(BuildingType.BELT_FORWARD, 2, 2, 0, Rotation.EAST)

    # Belt going NORTH (would be from south)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 1, 0, Rotation.NORTH)

    # Belt going SOUTH (would be from north)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 3, 0, Rotation.SOUTH)

    # Continue EAST
    sim.place_building(BuildingType.BELT_FORWARD, 3, 2, 0, Rotation.EAST)

    sim.set_input(2, 2, 0, "CuCuCuCu", 45.0)
    sim.set_output(3, 2, 0)
    sim.set_output(2, 0, 0)  # Would receive from north belt
    sim.set_output(2, 4, 0)  # Would receive from south belt

    sim.simulate()
    print_grid(sim, width=8, height=5)

    print("\nOutput throughputs:")
    for out in sim.outputs:
        print(f"  {out['position']}: {out['throughput']}")

    east_output = sim.outputs[0]['throughput']  # (3,2)
    north_output = sim.outputs[1]['throughput']  # (2,0)
    south_output = sim.outputs[2]['throughput']  # (2,4)

    print(f"\nEast: {east_output}, North: {north_output}, South: {south_output}")
    print("Expected: East=45 (or split with one perpendicular), N+S should NOT both have flow")

    # 180 conflict means N and S can't both get flow
    return not (north_output > 0 and south_output > 0)


def run_all_tests():
    """Run all belt flow tests."""
    tests = [
        ("Simple belt line", test_simple_belt_line),
        ("Belt merge two inputs", test_belt_merge_two_inputs),
        ("Belt split one side", test_belt_split_one_side),
        ("Straight belt no split", test_straight_belt_no_split),
        ("T-junction merge", test_t_junction_merge),
        ("180-degree conflict", test_180_conflict),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")


if __name__ == "__main__":
    run_all_tests()
