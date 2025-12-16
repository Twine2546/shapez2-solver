#!/usr/bin/env python3
"""
Comprehensive test suite for the grid simulation system.

Tests various designs to ensure:
1. Belts must physically connect to transfer shapes
2. Buildings process shapes correctly
3. Disconnected belts don't move shapes
4. Complex layouts work correctly
"""

import sys
sys.path.insert(0, '/config/projects/programming/games/shape2')

from shapez2_solver.simulation.grid_simulator import GridSimulator, SimulationResult
from shapez2_solver.shapes.parser import ShapeCodeParser
from shapez2_solver.blueprint.building_types import BuildingType, Rotation


class SimulationTester:
    """Test harness for simulation tests."""

    def __init__(self):
        self.parser = ShapeCodeParser()
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0

    def parse(self, code: str):
        """Parse a shape code."""
        return self.parser.parse(code)

    def run_test(self, name: str, sim: GridSimulator, inputs: dict, expected: dict,
                 max_steps: int = 50, debug: bool = False) -> bool:
        """
        Run a single simulation test.

        Args:
            name: Test name
            sim: Configured GridSimulator
            inputs: Dict of (x,y,floor) -> shape_code for inputs
            expected: Dict of (x,y,floor) -> shape_code for expected outputs
            max_steps: Maximum simulation steps
            debug: Enable debug logging
        """
        self.tests_run += 1
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")

        # Set up inputs
        for pos, shape_code in inputs.items():
            shape = self.parse(shape_code)
            sim.set_input(pos[0], pos[1], pos[2], shape)
            print(f"Input at {pos}: {shape_code}")

        # Set up outputs
        expected_shapes = {}
        for pos, shape_code in expected.items():
            sim.set_output(pos[0], pos[1], pos[2])
            if shape_code:
                expected_shapes[pos] = self.parse(shape_code)
            else:
                expected_shapes[pos] = None
            print(f"Expected output at {pos}: {shape_code}")

        # Print grid
        for floor in range(sim.num_floors):
            grid_str = sim.print_grid(floor)
            if "R" in grid_str or "C" in grid_str or "→" in grid_str:
                print(f"\n{grid_str}")

        # Run simulation
        sim.debug = debug
        result = sim.simulate(max_steps=max_steps, expected_outputs=expected_shapes)

        # Print results
        print(f"\nSimulation completed in {result.steps} steps")
        print(f"Success: {result.success}")

        # Compare outputs
        all_match = True
        for pos, expected_shape in expected_shapes.items():
            actual_shape = result.output_shapes.get(pos)
            actual_code = actual_shape.to_code() if actual_shape else "None"
            expected_code = expected_shape.to_code() if expected_shape else "None"

            if actual_code == expected_code:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
                all_match = False

            print(f"  {pos}: Expected={expected_code}, Actual={actual_code} {status}")

        if result.errors:
            print(f"\nErrors:")
            for err in result.errors:
                print(f"  - {err}")

        if debug and result.debug_log:
            print(f"\nDebug Log:")
            for log in result.debug_log[:50]:  # Limit output
                print(f"  {log}")

        if all_match:
            self.tests_passed += 1
            print(f"\n>>> TEST PASSED <<<")
            return True
        else:
            self.tests_failed += 1
            print(f"\n>>> TEST FAILED <<<")
            return False

    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        if self.tests_run > 0:
            print(f"Pass rate: {self.tests_passed/self.tests_run*100:.1f}%")


def test_simple_belt_passthrough():
    """Test: Simple belt passes shape through."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    # Layout: Input -> Belt -> Belt -> Belt -> Output
    #         (0,0)   (1,0)   (2,0)   (3,0)   (4,0)
    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(2, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(3, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    tester.run_test(
        "Simple Belt Passthrough",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={(4, 0, 0): "CuCuCuCu"}
    )

    return tester


def test_disconnected_belt():
    """Test: Disconnected belt should NOT pass shape."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    # Layout: Input -> Belt -> GAP -> Belt -> Output
    #         (0,0)   (1,0)   (2,0)  (3,0)   (4,0)
    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    # Gap at (2,0)
    sim.add_belt(3, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    tester.run_test(
        "Disconnected Belt (should NOT pass)",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={(4, 0, 0): None}  # Shape should NOT reach output
    )

    return tester


def test_wrong_direction_belt():
    """Test: Belt facing wrong direction should NOT pass shape."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    # Layout: Input -> Belt(E) -> Belt(W) -> Output
    # Second belt faces wrong direction!
    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(2, 0, 0, BuildingType.BELT_FORWARD, Rotation.WEST)  # Wrong!

    tester.run_test(
        "Wrong Direction Belt (should NOT pass)",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={(3, 0, 0): None}
    )

    return tester


def test_rotator_cw():
    """Test: Rotator CW rotates shape 90 degrees clockwise."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    # Layout: Input -> Belt -> Rotator -> Belt -> Output
    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(1, BuildingType.ROTATOR_CW, 2, 0, 0, Rotation.EAST)
    sim.add_belt(3, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    # CuCuCuCu rotated CW: positions shift 0->3->2->1->0
    # NE->SE, SE->SW, SW->NW, NW->NE
    tester.run_test(
        "Rotator CW",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={(4, 0, 0): "CuCuCuCu"},  # Full circle unchanged
        debug=True
    )

    return tester


def test_rotator_with_asymmetric_shape():
    """Test: Rotator CW with asymmetric shape."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(1, BuildingType.ROTATOR_CW, 2, 0, 0, Rotation.EAST)
    sim.add_belt(3, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    # Cu------ (NE only) rotated CW should become ------Cu (SE only)
    tester.run_test(
        "Rotator CW Asymmetric Shape",
        sim,
        inputs={(0, 0, 0): "Cu------"},
        expected={(4, 0, 0): "------Cu"},
        debug=True
    )

    return tester


def test_cutter():
    """Test: Cutter splits shape into two halves."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    # Layout:
    #   Input -> Belt -> Cutter -> Belt -> Output1 (east half)
    #                       |
    #                       +-> Belt -> Output2 (west half)
    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(1, BuildingType.CUTTER, 2, 0, 0, Rotation.EAST)
    # Cutter outputs: (3,0) = east half, (3,1) = west half
    sim.add_belt(3, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(3, 1, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    # CuCuCuCu split:
    # East half (NE+SE): Cu----Cu (positions 0 and 3)
    # West half (NW+SW): --CuCu-- (positions 1 and 2)
    tester.run_test(
        "Cutter",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={
            (4, 0, 0): "Cu----Cu",  # East half
            (4, 1, 0): "--CuCu--",  # West half
        },
        debug=True
    )

    return tester


def test_half_cutter():
    """Test: Half cutter destroys west half."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(1, BuildingType.HALF_CUTTER, 2, 0, 0, Rotation.EAST)
    sim.add_belt(3, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    # CuCuCuCu with west half (NW+SW) destroyed = Cu----Cu
    tester.run_test(
        "Half Cutter",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={(4, 0, 0): "Cu----Cu"},
        debug=True
    )

    return tester


def test_corner_splitter_simple():
    """Test: Simple corner splitter using cutter + rotator + cutter."""
    tester = SimulationTester()
    sim = GridSimulator(width=20, height=10, num_floors=1)

    # First cutter at (2,2)
    sim.add_belt(0, 2, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(1, 2, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(1, BuildingType.CUTTER, 2, 2, 0, Rotation.EAST)

    # Top path (east half): Belt -> Rotator -> Belt -> Cutter -> Outputs
    sim.add_belt(3, 2, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(2, BuildingType.ROTATOR_CW, 4, 2, 0, Rotation.EAST)
    sim.add_belt(5, 2, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(3, BuildingType.CUTTER, 6, 2, 0, Rotation.EAST)
    sim.add_belt(7, 2, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(7, 3, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    # Bottom path (west half): Turn -> Belt -> Rotator -> Belt -> Cutter -> Outputs
    sim.add_belt(3, 3, 0, BuildingType.BELT_RIGHT, Rotation.EAST)
    sim.add_belt(3, 4, 0, BuildingType.BELT_FORWARD, Rotation.SOUTH)
    sim.add_belt(3, 5, 0, BuildingType.BELT_LEFT, Rotation.SOUTH)
    sim.add_building(4, BuildingType.ROTATOR_CW, 4, 5, 0, Rotation.EAST)
    sim.add_belt(5, 5, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(5, BuildingType.CUTTER, 6, 5, 0, Rotation.EAST)
    sim.add_belt(7, 5, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(7, 6, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    # Expected outputs:
    # Input: CuCuCuCu
    # After cutter1: Cu----Cu (east=NE+SE), --CuCu-- (west=NW+SW)
    #
    # Top path (Cu----Cu):
    #   Rotate CW: NE+SE → SW+SE = ----CuCu
    #   Cutter: ----CuCu → ------Cu (east=SE), ----Cu-- (west=SW)
    #
    # Bottom path (--CuCu--):
    #   Rotate CW: NW+SW → NE+NW = CuCu----
    #   Cutter: CuCu---- → Cu------ (east=NE), --Cu---- (west=NW)

    tester.run_test(
        "Corner Splitter",
        sim,
        inputs={(0, 2, 0): "CuCuCuCu"},
        expected={
            (8, 2, 0): "------Cu",  # SE corner (from top path)
            (8, 3, 0): "----Cu--",  # SW corner (from top path)
            (8, 5, 0): "Cu------",  # NE corner (from bottom path)
            (8, 6, 0): "--Cu----",  # NW corner (from bottom path)
        },
        max_steps=100,
        debug=True
    )

    return tester


def test_belt_turn():
    """Test: Belt turn corners work correctly."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=1)

    # Layout: Input -> Belt -> Turn Right -> Belt down -> Output
    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_belt(2, 0, 0, BuildingType.BELT_RIGHT, Rotation.EAST)  # Turn south
    sim.add_belt(2, 1, 0, BuildingType.BELT_FORWARD, Rotation.SOUTH)
    sim.add_belt(2, 2, 0, BuildingType.BELT_FORWARD, Rotation.SOUTH)

    tester.run_test(
        "Belt Turn Right",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={(2, 3, 0): "CuCuCuCu"},
        debug=True
    )

    return tester


def test_lift_up():
    """Test: Lift moves shapes between floors."""
    tester = SimulationTester()
    sim = GridSimulator(width=10, height=10, num_floors=3)

    # Floor 0: Input -> Belt -> Lift
    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_lift(2, 0, 0, BuildingType.LIFT_UP, Rotation.EAST)

    # Floor 1: Lift output -> Belt -> Output
    sim.add_belt(3, 0, 1, BuildingType.BELT_FORWARD, Rotation.EAST)

    tester.run_test(
        "Lift Up",
        sim,
        inputs={(0, 0, 0): "CuCuCuCu"},
        expected={(4, 0, 1): "CuCuCuCu"},
        debug=True
    )

    return tester


def test_multiple_rotations():
    """Test: Multiple rotations (180 degrees)."""
    tester = SimulationTester()
    sim = GridSimulator(width=15, height=10, num_floors=1)

    sim.add_belt(1, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(1, BuildingType.ROTATOR_CW, 2, 0, 0, Rotation.EAST)
    sim.add_belt(3, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)
    sim.add_building(2, BuildingType.ROTATOR_CW, 4, 0, 0, Rotation.EAST)
    sim.add_belt(5, 0, 0, BuildingType.BELT_FORWARD, Rotation.EAST)

    # Cu------ rotated 180 degrees becomes ----Cu--
    tester.run_test(
        "Double Rotation (180°)",
        sim,
        inputs={(0, 0, 0): "Cu------"},
        expected={(6, 0, 0): "----Cu--"},
        debug=True
    )

    return tester


def run_all_tests():
    """Run all simulation tests."""
    print("=" * 70)
    print("SHAPEZ 2 SIMULATION TEST SUITE")
    print("=" * 70)

    all_testers = []

    # Run individual tests
    all_testers.append(test_simple_belt_passthrough())
    all_testers.append(test_disconnected_belt())
    all_testers.append(test_wrong_direction_belt())
    all_testers.append(test_belt_turn())
    all_testers.append(test_rotator_cw())
    all_testers.append(test_rotator_with_asymmetric_shape())
    all_testers.append(test_half_cutter())
    all_testers.append(test_cutter())
    all_testers.append(test_multiple_rotations())
    all_testers.append(test_lift_up())
    all_testers.append(test_corner_splitter_simple())

    # Aggregate results
    total_run = sum(t.tests_run for t in all_testers)
    total_passed = sum(t.tests_passed for t in all_testers)
    total_failed = sum(t.tests_failed for t in all_testers)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total_run}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    if total_run > 0:
        print(f"Pass rate: {total_passed/total_run*100:.1f}%")

    return total_passed == total_run


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
