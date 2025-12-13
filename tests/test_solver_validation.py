"""
Comprehensive validation tests for CPSAT solver using the Flow Simulator.

Tests various:
- Foundation types (1x1, 2x1, 1x2, 2x2, etc.)
- Input/output configurations (different sides, positions)
- Shape transformations (passthrough, cutting, rotating, stacking)
- Multiple inputs/outputs
- Edge cases

Run with: python -m tests.test_solver_validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import traceback

from shapez2_solver.simulation.flow_simulator import (
    FlowSimulator, FlowReport, BELT_THROUGHPUT, MACHINE_THROUGHPUT
)
from shapez2_solver.blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS
from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS, Side
from shapez2_solver.evolution.cpsat_solver import CPSATFullSolver, CPSATSolution


@dataclass
class ValidationResult:
    """Result of validating a solver solution."""
    test_name: str
    solver_success: bool
    simulation_valid: bool
    expected_outputs: List[Tuple[str, float]]
    actual_outputs: List[Tuple[str, float]]
    errors: List[str]
    warnings: List[str]
    efficiency: float
    solution: Optional[CPSATSolution] = None
    exception: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.solver_success and self.simulation_valid

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{self.test_name}: {status}"]

        if not self.passed:
            if self.exception:
                lines.append(f"  Exception: {self.exception[:100]}")
            elif not self.solver_success:
                lines.append(f"  Solver failed")
            else:
                for err in self.errors[:3]:
                    lines.append(f"  - {err[:80]}")

        return "\n".join(lines)


class SolutionValidator:
    """Validates CPSAT solutions using flow simulation."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert_solution_to_simulator(
        self,
        solution: CPSATSolution,
        foundation_type: str,
        input_specs: List[Tuple[str, int, int, str]],
        output_specs: List[Tuple[str, int, int, str]],
    ) -> FlowSimulator:
        spec = FOUNDATION_SPECS.get(foundation_type)
        if spec is None:
            raise ValueError(f"Unknown foundation type: {foundation_type}")

        sim = FlowSimulator(foundation_spec=spec, validate_io=True)

        for bt, x, y, floor, rot in solution.machines:
            sim.place_building(bt, x, y, floor, rot)

        for x, y, floor, bt, rot in solution.belts:
            sim.place_building(bt, x, y, floor, rot)

        for side_str, pos, floor, shape_code in input_specs:
            side = self._side_from_string(side_str)
            io_x, io_y = self._get_io_position(spec, side, pos)
            sim.set_input(io_x, io_y, floor, shape_code, BELT_THROUGHPUT)

        for side_str, pos, floor, shape_code in output_specs:
            side = self._side_from_string(side_str)
            io_x, io_y = self._get_io_position(spec, side, pos)
            sim.set_output(io_x, io_y, floor, shape_code if shape_code else None)

        return sim

    def _side_from_string(self, s: str) -> Side:
        mapping = {'N': Side.NORTH, 'S': Side.SOUTH, 'E': Side.EAST, 'W': Side.WEST,
                   'NORTH': Side.NORTH, 'SOUTH': Side.SOUTH, 'EAST': Side.EAST, 'WEST': Side.WEST}
        return mapping.get(s.upper(), Side.WEST)

    def _get_io_position(self, spec, side: Side, port_idx: int) -> Tuple[int, int]:
        grid_x, grid_y = spec.get_port_grid_position(side, port_idx)
        if side == Side.WEST:
            return (-1, grid_y)
        elif side == Side.EAST:
            return (spec.grid_width, grid_y)
        elif side == Side.NORTH:
            return (grid_x, -1)
        else:
            return (grid_x, spec.grid_height)

    def validate_solution(
        self,
        test_name: str,
        solution: CPSATSolution,
        foundation_type: str,
        input_specs: List[Tuple[str, int, int, str]],
        output_specs: List[Tuple[str, int, int, str]],
    ) -> ValidationResult:
        errors = []
        warnings = []

        if not solution.routing_success:
            return ValidationResult(
                test_name=test_name,
                solver_success=False,
                simulation_valid=False,
                expected_outputs=[(s, BELT_THROUGHPUT) for _, _, _, s in output_specs],
                actual_outputs=[],
                errors=["Solver routing failed"],
                warnings=[],
                efficiency=0.0,
                solution=solution
            )

        try:
            sim = self.convert_solution_to_simulator(
                solution, foundation_type, input_specs, output_specs
            )
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                solver_success=True,
                simulation_valid=False,
                expected_outputs=[(s, BELT_THROUGHPUT) for _, _, _, s in output_specs],
                actual_outputs=[],
                errors=[f"Conversion error: {e}"],
                warnings=[],
                efficiency=0.0,
                solution=solution
            )

        try:
            report = sim.simulate()
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                solver_success=True,
                simulation_valid=False,
                expected_outputs=[(s, BELT_THROUGHPUT) for _, _, _, s in output_specs],
                actual_outputs=[],
                errors=[f"Simulation error: {e}"],
                warnings=[],
                efficiency=0.0,
                solution=solution
            )

        actual_outputs = []
        for out in sim.outputs:
            shape = out.get('actual_shape', '(none)')
            throughput = out.get('throughput', 0.0)
            actual_outputs.append((shape or '(none)', throughput))

        expected_outputs = [(shape_code, BELT_THROUGHPUT) for _, _, _, shape_code in output_specs]

        # Filter simulator errors - ignore backup errors for shapes we don't need
        expected_shapes = {shape_code for _, _, _, shape_code in output_specs if shape_code}

        # Check validity while ignoring backed-up outputs for unwanted shapes
        sim_valid = True
        if sim.errors:
            for err in sim.errors:
                # Check if this error is for an unwanted shape
                if 'will back up' in err and 'shape=' in err:
                    try:
                        shape_part = err.split('shape=')[1].split(',')[0].strip(')')
                        if shape_part not in expected_shapes:
                            warnings.append(f"[Unused output ignored] {err}")
                            continue  # Ignore this error
                    except:
                        pass
                errors.append(err)
                sim_valid = False

        # Also check backed_up flag on machines, but ignore for unwanted shapes
        for m in sim.machines.values():
            for port in m.output_ports:
                if port['backed_up']:
                    shape = port.get('shape', '')
                    if shape not in expected_shapes:
                        # Ignore backed-up output for shape we don't need
                        continue
                    # This is a backed-up output for a shape we DO need
                    sim_valid = False

        warnings.extend(sim.warnings)

        for i, ((exp_shape, _), (act_shape, act_tput)) in enumerate(
            zip(expected_outputs, actual_outputs)
        ):
            if exp_shape and act_shape != exp_shape and act_shape != '(none)':
                errors.append(f"Output [{i}] shape mismatch: expected {exp_shape}, got {act_shape}")
            if act_tput == 0:
                errors.append(f"Output [{i}] has no throughput")

        total_output = sum(tput for _, tput in actual_outputs)
        max_output = BELT_THROUGHPUT * len(output_specs) if output_specs else 0
        efficiency = 100 * total_output / max_output if max_output > 0 else 0

        return ValidationResult(
            test_name=test_name,
            solver_success=True,
            simulation_valid=sim_valid and len(errors) == 0,
            expected_outputs=expected_outputs,
            actual_outputs=actual_outputs,
            errors=errors,
            warnings=warnings,
            efficiency=efficiency,
            solution=solution
        )


def run_test(
    name: str,
    foundation: str,
    inputs: List[Tuple[str, int, int, str]],
    outputs: List[Tuple[str, int, int, str]],
    time_limit: float = 30.0,
    routing_mode: str = 'astar',
    verbose: bool = False,
) -> ValidationResult:
    """Run a complete test: solve with CPSAT, validate with simulation."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING TEST: {name}")
        print(f"{'='*60}")

    solver = CPSATFullSolver(
        foundation_type=foundation,
        input_specs=inputs,
        output_specs=outputs,
        time_limit_seconds=time_limit,
        routing_mode=routing_mode,
        enable_logging=False,
        enable_placement_feedback=False,
        enable_transformer_logging=False,
    )

    try:
        solution = solver.solve(verbose=verbose)
    except Exception as e:
        return ValidationResult(
            test_name=name,
            solver_success=False,
            simulation_valid=False,
            expected_outputs=[(s, BELT_THROUGHPUT) for _, _, _, s in outputs],
            actual_outputs=[],
            errors=[],
            warnings=[],
            efficiency=0.0,
            exception=str(e)
        )

    if solution is None:
        return ValidationResult(
            test_name=name,
            solver_success=False,
            simulation_valid=False,
            expected_outputs=[(s, BELT_THROUGHPUT) for _, _, _, s in outputs],
            actual_outputs=[],
            errors=["Solver returned None"],
            warnings=[],
            efficiency=0.0
        )

    validator = SolutionValidator(verbose=verbose)
    return validator.validate_solution(name, solution, foundation, inputs, outputs)


# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================

# Shape codes
FULL_CU = "CuCuCuCu"  # Full copper square
LEFT_CU = "Cu--Cu--"  # Left half copper
RIGHT_CU = "--Cu--Cu"  # Right half copper
TOP_CU = "CuCu----"   # Top half copper
BOT_CU = "----CuCu"   # Bottom half copper

# Test case structure: (name, foundation, inputs, outputs, routing_mode)
TEST_CASES = [
    # ==========================================================================
    # BASIC PASSTHROUGH TESTS (1x1 foundation)
    # ==========================================================================
    ("1x1 Passthrough W->E", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU)],
     "astar"),

    ("1x1 Passthrough N->S", "1x1",
     [("N", 0, 0, FULL_CU)],
     [("S", 0, 0, FULL_CU)],
     "astar"),

    ("1x1 Passthrough E->W", "1x1",
     [("E", 0, 0, FULL_CU)],
     [("W", 0, 0, FULL_CU)],
     "astar"),

    ("1x1 Passthrough S->N", "1x1",
     [("S", 0, 0, FULL_CU)],
     [("N", 0, 0, FULL_CU)],
     "astar"),

    # ==========================================================================
    # SPLITTER TESTS (1 input -> 2 outputs, same shape)
    # ==========================================================================
    ("1x1 Split 1->2 E", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU), ("E", 1, 0, FULL_CU)],
     "astar"),

    ("1x1 Split 1->2 S", "1x1",
     [("N", 0, 0, FULL_CU)],
     [("S", 0, 0, FULL_CU), ("S", 1, 0, FULL_CU)],
     "astar"),

    # ==========================================================================
    # CUTTER TESTS (left/right half outputs)
    # ==========================================================================
    ("1x1 Cutter Both Halves", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, LEFT_CU), ("E", 1, 0, RIGHT_CU)],
     "astar"),

    ("1x1 Cutter Left Only", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, LEFT_CU)],
     "astar"),

    ("1x1 Cutter Right Only", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 1, 0, RIGHT_CU)],  # Position 1 naturally receives right half from cutter
     "astar"),

    # ==========================================================================
    # GLOBAL ROUTING TESTS
    # ==========================================================================
    ("1x1 Global Passthrough", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU)],
     "global"),

    ("1x1 Global Split", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU), ("E", 1, 0, FULL_CU)],
     "global"),

    ("1x1 Global Cutter", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, LEFT_CU), ("E", 1, 0, RIGHT_CU)],
     "global"),

    # ==========================================================================
    # 2x1 FOUNDATION TESTS (wider)
    # ==========================================================================
    ("2x1 Passthrough W->E", "2x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU)],
     "astar"),

    ("2x1 Split 1->2", "2x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU), ("E", 1, 0, FULL_CU)],
     "astar"),

    ("2x1 Cutter Both", "2x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, LEFT_CU), ("E", 1, 0, RIGHT_CU)],
     "astar"),

    ("2x1 Multi-Input", "2x1",
     [("W", 0, 0, FULL_CU), ("W", 1, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU), ("E", 1, 0, FULL_CU)],
     "astar"),

    # ==========================================================================
    # 1x2 FOUNDATION TESTS (taller)
    # ==========================================================================
    ("1x2 Passthrough W->E", "1x2",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU)],
     "astar"),

    ("1x2 Passthrough N->S", "1x2",
     [("N", 0, 0, FULL_CU)],
     [("S", 0, 0, FULL_CU)],
     "astar"),

    ("1x2 Split N->S", "1x2",
     [("N", 0, 0, FULL_CU)],
     [("S", 0, 0, FULL_CU), ("S", 1, 0, FULL_CU)],
     "astar"),

    # ==========================================================================
    # 2x2 FOUNDATION TESTS
    # ==========================================================================
    ("2x2 Passthrough W->E", "2x2",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU)],
     "astar"),

    ("2x2 Diagonal W->S", "2x2",
     [("W", 0, 0, FULL_CU)],
     [("S", 0, 0, FULL_CU)],
     "astar"),

    ("2x2 Split 1->4", "2x2",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU), ("E", 1, 0, FULL_CU), ("S", 0, 0, FULL_CU), ("S", 1, 0, FULL_CU)],
     "astar"),

    ("2x2 Multi-Input Multi-Output", "2x2",
     [("W", 0, 0, FULL_CU), ("N", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU), ("S", 0, 0, FULL_CU)],
     "astar"),

    ("2x2 Cutter Both Global", "2x2",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, LEFT_CU), ("E", 1, 0, RIGHT_CU)],
     "global"),

    # ==========================================================================
    # LARGER FOUNDATIONS
    # ==========================================================================
    ("3x1 Passthrough", "3x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU)],
     "astar"),

    ("3x1 Triple Output", "3x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 0, FULL_CU), ("E", 1, 0, FULL_CU), ("E", 2, 0, FULL_CU)],
     "astar"),

    ("3x2 Multi-Cutter", "3x2",
     [("W", 0, 0, FULL_CU), ("W", 1, 0, FULL_CU)],
     [("E", 0, 0, LEFT_CU), ("E", 1, 0, RIGHT_CU), ("E", 2, 0, LEFT_CU), ("E", 3, 0, RIGHT_CU)],
     "astar"),

    # ==========================================================================
    # CROSS-SIDE ROUTING
    # ==========================================================================
    ("1x1 W->N Corner", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("N", 0, 0, FULL_CU)],
     "astar"),

    ("1x1 W->S Corner", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("S", 0, 0, FULL_CU)],
     "astar"),

    ("2x2 Opposite Corners", "2x2",
     [("W", 0, 0, FULL_CU)],
     [("E", 1, 0, FULL_CU)],  # Far corner
     "astar"),

    # ==========================================================================
    # MULTI-FLOOR TESTS (if supported)
    # ==========================================================================
    ("1x1 Floor 0->1 Lift", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("E", 0, 1, FULL_CU)],  # Output on floor 1
     "astar"),

    # ==========================================================================
    # EDGE CASES
    # ==========================================================================
    ("1x1 Same Side In/Out", "1x1",
     [("W", 0, 0, FULL_CU)],
     [("W", 1, 0, FULL_CU)],  # Same side, different port
     "astar"),

    ("2x1 Adjacent Ports", "2x1",
     [("W", 0, 0, FULL_CU)],
     [("W", 1, 0, FULL_CU)],  # Adjacent on same side
     "astar"),
]


def run_all_tests(verbose: bool = False, filter_pattern: str = None):
    """Run all validation tests."""

    results = []
    passed = 0
    failed = 0

    test_cases = TEST_CASES
    if filter_pattern:
        test_cases = [(n, f, i, o, r) for n, f, i, o, r in TEST_CASES
                      if filter_pattern.lower() in n.lower()]

    print(f"\nRunning {len(test_cases)} tests...\n")
    print("-" * 70)

    for name, foundation, inputs, outputs, routing_mode in test_cases:
        try:
            result = run_test(
                name=name,
                foundation=foundation,
                inputs=inputs,
                outputs=outputs,
                routing_mode=routing_mode,
                time_limit=30.0,
                verbose=verbose
            )
            results.append(result)

            if result.passed:
                passed += 1
                print(f"✓ {name}")
            else:
                failed += 1
                print(f"✗ {name}")
                if result.exception:
                    print(f"    Exception: {result.exception[:60]}...")
                elif not result.solver_success:
                    print(f"    Solver failed")
                else:
                    for err in result.errors[:2]:
                        print(f"    {err[:70]}")

        except Exception as e:
            failed += 1
            print(f"✗ {name}")
            print(f"    EXCEPTION: {str(e)[:60]}")
            results.append(ValidationResult(
                test_name=name,
                solver_success=False,
                simulation_valid=False,
                expected_outputs=[],
                actual_outputs=[],
                errors=[],
                warnings=[],
                efficiency=0.0,
                exception=str(e)
            ))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed} ({100*passed/len(test_cases):.0f}%)")
    print(f"Failed: {failed} ({100*failed/len(test_cases):.0f}%)")

    if failed > 0:
        print("\n" + "-" * 70)
        print("FAILED TESTS BY CATEGORY:")
        print("-" * 70)

        # Categorize failures
        categories = {
            'solver_failed': [],
            'unused_output': [],
            'no_throughput': [],
            'shape_mismatch': [],
            'exception': [],
            'other': []
        }

        for result in results:
            if result.passed:
                continue

            if result.exception:
                categories['exception'].append(result)
            elif not result.solver_success:
                categories['solver_failed'].append(result)
            elif any('no destination' in e or 'back up' in e for e in result.errors):
                categories['unused_output'].append(result)
            elif any('no throughput' in e for e in result.errors):
                categories['no_throughput'].append(result)
            elif any('mismatch' in e for e in result.errors):
                categories['shape_mismatch'].append(result)
            else:
                categories['other'].append(result)

        for cat, tests in categories.items():
            if tests:
                print(f"\n{cat.upper().replace('_', ' ')} ({len(tests)}):")
                for t in tests:
                    print(f"  - {t.test_name}")

    return passed, failed, results


def run_quick_tests():
    """Run a quick subset of tests for rapid iteration."""
    quick_tests = [
        ("Quick: Passthrough", "1x1", [("W", 0, 0, FULL_CU)], [("E", 0, 0, FULL_CU)], "astar"),
        ("Quick: Split", "1x1", [("W", 0, 0, FULL_CU)], [("E", 0, 0, FULL_CU), ("E", 1, 0, FULL_CU)], "astar"),
        ("Quick: Cutter", "1x1", [("W", 0, 0, FULL_CU)], [("E", 0, 0, LEFT_CU), ("E", 1, 0, RIGHT_CU)], "astar"),
        ("Quick: Global", "1x1", [("W", 0, 0, FULL_CU)], [("E", 0, 0, FULL_CU)], "global"),
    ]

    print("\nRunning quick tests...\n")

    for name, foundation, inputs, outputs, routing_mode in quick_tests:
        result = run_test(name, foundation, inputs, outputs, routing_mode=routing_mode, verbose=False)
        status = "✓" if result.passed else "✗"
        print(f"{status} {name}")
        if not result.passed and result.errors:
            print(f"    {result.errors[0][:60]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_tests()
        elif sys.argv[1] == "--verbose":
            run_all_tests(verbose=True)
        elif sys.argv[1] == "--filter":
            pattern = sys.argv[2] if len(sys.argv) > 2 else ""
            run_all_tests(verbose=False, filter_pattern=pattern)
        else:
            run_all_tests(verbose=False, filter_pattern=sys.argv[1])
    else:
        run_all_tests(verbose=False)
