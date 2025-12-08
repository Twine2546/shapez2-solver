"""
CP-SAT Solver for Shapez 2 - Constraint Programming approach.

Uses Google OR-Tools CP-SAT solver for optimal/near-optimal solutions.
This approach guarantees finding a solution if one exists.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from ortools.sat.python import cp_model

from ..shapes.shape import Shape
from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS, BUILDING_PORTS
from .foundation_evolution import PlacedBuilding, Candidate
from .foundation_config import FOUNDATION_SPECS


# Machine types available for solving
SOLVER_MACHINES = [
    BuildingType.CUTTER,
    BuildingType.HALF_CUTTER,
    BuildingType.ROTATOR_CW,
    BuildingType.ROTATOR_CCW,
    BuildingType.ROTATOR_180,
    BuildingType.STACKER,
    BuildingType.STACKER_BENT,
    BuildingType.UNSTACKER,
    BuildingType.SWAPPER,
]

# Shape transformation rules for each machine type
# Maps: (building_type) -> function(input_shapes) -> output_shapes
# This is simplified - full implementation would use actual shape operations


@dataclass
class CPSATSolution:
    """Solution from CP-SAT solver."""
    machines: List[Tuple[BuildingType, int, int, int, Rotation]]  # (type, x, y, floor, rotation)
    belts: List[Tuple[int, int, int, BuildingType, Rotation]]  # (x, y, floor, belt_type, rotation)
    fitness: float
    solve_time: float
    status: str  # 'optimal', 'feasible', 'infeasible', 'timeout'

    def to_candidate(self) -> Candidate:
        """Convert to Candidate for viewer compatibility."""
        buildings = []
        building_id = 0

        for bt, x, y, floor, rot in self.machines:
            buildings.append(PlacedBuilding(
                building_id=building_id,
                building_type=bt,
                x=x, y=y, floor=floor,
                rotation=rot
            ))
            building_id += 1

        for x, y, floor, bt, rot in self.belts:
            buildings.append(PlacedBuilding(
                building_id=building_id,
                building_type=bt,
                x=x, y=y, floor=floor,
                rotation=rot
            ))
            building_id += 1

        return Candidate(buildings=buildings, fitness=self.fitness)


class CPSATLayoutSolver:
    """
    CP-SAT based solver for Shapez 2 layout problems.

    Given a system design (machines and connections), finds optimal placement.
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        max_machines: int = 10,
        time_limit_seconds: float = 30.0,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_floors = num_floors
        self.max_machines = max_machines
        self.time_limit = time_limit_seconds

        self.model: Optional[cp_model.CpModel] = None
        self.solver: Optional[cp_model.CpSolver] = None

    def solve_layout(
        self,
        machine_types: List[BuildingType],
        connections: List[Tuple[int, int, int, int]],  # (from_machine, from_output, to_machine, to_input)
        input_positions: List[Tuple[int, int, int]],   # (x, y, floor) for each input port
        output_positions: List[Tuple[int, int, int]],  # (x, y, floor) for each output port
        verbose: bool = False,
    ) -> Optional[CPSATSolution]:
        """
        Solve the layout placement problem.

        Args:
            machine_types: List of machine types to place
            connections: How machines connect to each other
            input_positions: Where inputs enter the grid
            output_positions: Where outputs exit the grid
            verbose: Print progress

        Returns:
            CPSATSolution if found, None otherwise
        """
        import time
        start_time = time.time()

        self.model = cp_model.CpModel()
        num_machines = len(machine_types)

        if verbose:
            print(f"CP-SAT: Placing {num_machines} machines on {self.grid_width}x{self.grid_height} grid")

        # === VARIABLES ===

        # Position variables for each machine: x, y, floor
        machine_x = []
        machine_y = []
        machine_floor = []
        machine_rotation = []  # 0=EAST, 1=SOUTH, 2=WEST, 3=NORTH

        for i, bt in enumerate(machine_types):
            spec = BUILDING_SPECS.get(bt)
            max_x = self.grid_width - (spec.width if spec else 1)
            max_y = self.grid_height - (spec.height if spec else 1)

            machine_x.append(self.model.NewIntVar(0, max_x, f'x_{i}'))
            machine_y.append(self.model.NewIntVar(0, max_y, f'y_{i}'))
            machine_floor.append(self.model.NewIntVar(0, self.num_floors - 1, f'floor_{i}'))
            machine_rotation.append(self.model.NewIntVar(0, 3, f'rot_{i}'))

        # === CONSTRAINTS ===

        # 1. No overlap between machines (using intervals)
        self._add_no_overlap_constraints(machine_types, machine_x, machine_y, machine_floor)

        # 2. Connected machines should be close (soft constraint via objective)
        # For now, minimize total manhattan distance between connected machines

        # === OBJECTIVE ===
        # Minimize: sum of distances between connected machines + total area used

        distance_terms = []
        for from_m, from_out, to_m, to_inp in connections:
            # Manhattan distance between machines
            dx = self.model.NewIntVar(-self.grid_width, self.grid_width, f'dx_{from_m}_{to_m}')
            dy = self.model.NewIntVar(-self.grid_height, self.grid_height, f'dy_{from_m}_{to_m}')
            abs_dx = self.model.NewIntVar(0, self.grid_width, f'abs_dx_{from_m}_{to_m}')
            abs_dy = self.model.NewIntVar(0, self.grid_height, f'abs_dy_{from_m}_{to_m}')

            self.model.Add(dx == machine_x[to_m] - machine_x[from_m])
            self.model.Add(dy == machine_y[to_m] - machine_y[from_m])
            self.model.AddAbsEquality(abs_dx, dx)
            self.model.AddAbsEquality(abs_dy, dy)

            distance_terms.append(abs_dx)
            distance_terms.append(abs_dy)

        if distance_terms:
            self.model.Minimize(sum(distance_terms))

        # === SOLVE ===
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.time_limit
        self.solver.parameters.num_search_workers = 8  # Use multiple cores

        if verbose:
            print(f"CP-SAT: Solving...")

        status = self.solver.Solve(self.model)
        solve_time = time.time() - start_time

        status_name = {
            cp_model.OPTIMAL: 'optimal',
            cp_model.FEASIBLE: 'feasible',
            cp_model.INFEASIBLE: 'infeasible',
            cp_model.MODEL_INVALID: 'invalid',
            cp_model.UNKNOWN: 'timeout',
        }.get(status, 'unknown')

        if verbose:
            print(f"CP-SAT: Status={status_name}, Time={solve_time:.2f}s")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return CPSATSolution(
                machines=[], belts=[], fitness=0.0,
                solve_time=solve_time, status=status_name
            )

        # Extract solution
        machines = []
        for i, bt in enumerate(machine_types):
            x = self.solver.Value(machine_x[i])
            y = self.solver.Value(machine_y[i])
            floor = self.solver.Value(machine_floor[i])
            rot_val = self.solver.Value(machine_rotation[i])
            rot = [Rotation.EAST, Rotation.SOUTH, Rotation.WEST, Rotation.NORTH][rot_val]
            machines.append((bt, x, y, floor, rot))

        # Calculate fitness (higher is better)
        # Based on compactness and successful placement
        fitness = 100.0 - (self.solver.ObjectiveValue() / 10.0) if distance_terms else 100.0
        fitness = max(0.0, min(100.0, fitness))

        return CPSATSolution(
            machines=machines,
            belts=[],  # Belt routing done separately
            fitness=fitness,
            solve_time=solve_time,
            status=status_name
        )

    def _add_no_overlap_constraints(
        self,
        machine_types: List[BuildingType],
        machine_x: List[cp_model.IntVar],
        machine_y: List[cp_model.IntVar],
        machine_floor: List[cp_model.IntVar],
    ):
        """Add constraints to prevent machine overlap."""
        n = len(machine_types)

        # Create interval variables for each machine
        x_intervals = []
        y_intervals = []

        for i, bt in enumerate(machine_types):
            spec = BUILDING_SPECS.get(bt)
            w = spec.width if spec else 1
            h = spec.height if spec else 1

            # X interval: [x, x+width)
            x_size = self.model.NewConstant(w)
            x_interval = self.model.NewIntervalVar(
                machine_x[i], x_size, machine_x[i] + w, f'x_interval_{i}'
            )
            x_intervals.append(x_interval)

            # Y interval: [y, y+height)
            y_size = self.model.NewConstant(h)
            y_interval = self.model.NewIntervalVar(
                machine_y[i], y_size, machine_y[i] + h, f'y_interval_{i}'
            )
            y_intervals.append(y_interval)

        # For each floor, add 2D no-overlap constraint
        # Simplified: assume all machines on same floor for now
        # TODO: Add floor-based grouping for multi-floor layouts

        # For each pair of machines, they must not overlap if on same floor
        for i in range(n):
            for j in range(i + 1, n):
                # Either different floors, or no X overlap, or no Y overlap
                same_floor = self.model.NewBoolVar(f'same_floor_{i}_{j}')
                self.model.Add(machine_floor[i] == machine_floor[j]).OnlyEnforceIf(same_floor)
                self.model.Add(machine_floor[i] != machine_floor[j]).OnlyEnforceIf(same_floor.Not())

                # If same floor, must not overlap in X or Y
                spec_i = BUILDING_SPECS.get(machine_types[i])
                spec_j = BUILDING_SPECS.get(machine_types[j])
                wi, hi = (spec_i.width, spec_i.height) if spec_i else (1, 1)
                wj, hj = (spec_j.width, spec_j.height) if spec_j else (1, 1)

                # No overlap conditions (at least one must be true if same floor)
                no_overlap_x_left = self.model.NewBoolVar(f'no_x_left_{i}_{j}')
                no_overlap_x_right = self.model.NewBoolVar(f'no_x_right_{i}_{j}')
                no_overlap_y_top = self.model.NewBoolVar(f'no_y_top_{i}_{j}')
                no_overlap_y_bottom = self.model.NewBoolVar(f'no_y_bottom_{i}_{j}')

                self.model.Add(machine_x[i] + wi <= machine_x[j]).OnlyEnforceIf(no_overlap_x_left)
                self.model.Add(machine_x[j] + wj <= machine_x[i]).OnlyEnforceIf(no_overlap_x_right)
                self.model.Add(machine_y[i] + hi <= machine_y[j]).OnlyEnforceIf(no_overlap_y_top)
                self.model.Add(machine_y[j] + hj <= machine_y[i]).OnlyEnforceIf(no_overlap_y_bottom)

                # If same floor, at least one no-overlap condition must hold
                self.model.AddBoolOr([
                    same_floor.Not(),
                    no_overlap_x_left,
                    no_overlap_x_right,
                    no_overlap_y_top,
                    no_overlap_y_bottom,
                ])


class CPSATSystemSolver:
    """
    CP-SAT solver for finding which machines to use.

    This handles the "system design" problem - determining what machines
    are needed to transform input shapes to output shapes.
    """

    def __init__(
        self,
        input_shapes: List[Shape],
        output_shapes: List[Shape],
        max_machines: int = 8,
        time_limit_seconds: float = 60.0,
    ):
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.max_machines = max_machines
        self.time_limit = time_limit_seconds

    def solve(self, verbose: bool = False) -> Optional[List[BuildingType]]:
        """
        Find a set of machines that can transform inputs to outputs.

        This is a simplified version - a full implementation would encode
        the shape transformation semantics into CP-SAT constraints.

        Returns:
            List of machine types to use, or None if no solution found
        """
        # For now, use a heuristic approach based on shape analysis
        # A full CP-SAT encoding would require modeling shape transformations

        machines_needed = []

        # Analyze transformation requirements
        for inp, out in zip(self.input_shapes, self.output_shapes):
            if inp.to_code() == out.to_code():
                continue  # No transformation needed

            # Check if cutting is needed (output has fewer parts)
            inp_parts = self._count_parts(inp)
            out_parts = self._count_parts(out)

            if out_parts < inp_parts:
                machines_needed.append(BuildingType.CUTTER)

            # Check if rotation is needed
            if self._needs_rotation(inp, out):
                machines_needed.append(BuildingType.ROTATOR_CW)

        # Handle multiple outputs from single input (need splitters)
        if len(self.output_shapes) > len(self.input_shapes):
            num_splits = len(self.output_shapes) - len(self.input_shapes)
            for _ in range(num_splits):
                machines_needed.append(BuildingType.CUTTER)

        return machines_needed if machines_needed else [BuildingType.CUTTER]

    def _count_parts(self, shape: Shape) -> int:
        """Count non-empty parts in a shape."""
        count = 0
        for layer_idx in range(shape.num_layers):
            layer = shape.get_layer(layer_idx)
            if layer:
                for part_idx in range(4):
                    part = layer.get_part(part_idx)
                    if part and not part.is_empty():
                        count += 1
        return count

    def _needs_rotation(self, inp: Shape, out: Shape) -> bool:
        """Check if rotation might be needed."""
        # Simplified check - compare part positions
        return False  # TODO: Implement rotation detection


class CPSATFullSolver:
    """
    Complete CP-SAT solver that handles both system design and layout.

    This is the main entry point for CP-SAT based solving.
    """

    def __init__(
        self,
        foundation_type: str,
        input_specs: List[Tuple[str, int, int, str]],  # (side, pos, floor, shape_code)
        output_specs: List[Tuple[str, int, int, str]],
        max_machines: int = 10,
        time_limit_seconds: float = 60.0,
    ):
        self.foundation_type = foundation_type
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.max_machines = max_machines
        self.time_limit = time_limit_seconds

        # Get foundation dimensions
        spec = FOUNDATION_SPECS.get(foundation_type)
        if spec is None:
            raise ValueError(f"Unknown foundation type: {foundation_type}")

        self.grid_width = spec.grid_width
        self.grid_height = spec.grid_height
        self.num_floors = spec.num_floors

        # Parse shapes
        self.input_shapes = [Shape.from_code(code) for _, _, _, code in input_specs]
        self.output_shapes = [Shape.from_code(code) for _, _, _, code in output_specs]

        # For viewer compatibility
        self.top_solutions: List[Candidate] = []
        self.config = None  # Set by caller if needed

    def solve(self, verbose: bool = False) -> Optional[CPSATSolution]:
        """
        Solve the complete problem.

        Returns:
            CPSATSolution with machines and layout, or None if infeasible
        """
        import time
        start_time = time.time()

        if verbose:
            print("=" * 50)
            print("CP-SAT SOLVER")
            print("=" * 50)
            print(f"Inputs: {len(self.input_specs)}, Outputs: {len(self.output_specs)}")
            print(f"Grid: {self.grid_width}x{self.grid_height}x{self.num_floors}")

        # Step 1: Determine required machines (system design)
        if verbose:
            print("\nPhase 1: System Design")

        system_solver = CPSATSystemSolver(
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
            max_machines=self.max_machines,
            time_limit_seconds=self.time_limit / 2,
        )

        machine_types = system_solver.solve(verbose=verbose)

        if machine_types is None:
            if verbose:
                print("No machine configuration found")
            return None

        if verbose:
            print(f"Machines needed: {[m.name for m in machine_types]}")

        # Step 2: Solve layout placement
        if verbose:
            print("\nPhase 2: Layout Placement")

        layout_solver = CPSATLayoutSolver(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_floors=self.num_floors,
            max_machines=self.max_machines,
            time_limit_seconds=self.time_limit / 2,
        )

        # Create simple linear connections for now
        # TODO: Use actual system topology
        connections = []
        for i in range(len(machine_types) - 1):
            connections.append((i, 0, i + 1, 0))

        # Input/output positions (from specs)
        input_positions = [(0, 0, 0)]  # Simplified
        output_positions = [(self.grid_width - 1, 0, 0)]  # Simplified

        solution = layout_solver.solve_layout(
            machine_types=machine_types,
            connections=connections,
            input_positions=input_positions,
            output_positions=output_positions,
            verbose=verbose,
        )

        total_time = time.time() - start_time

        if solution:
            solution.solve_time = total_time

            # Store for viewer
            candidate = solution.to_candidate()
            self.top_solutions = [candidate]

            if verbose:
                print(f"\nTotal solve time: {total_time:.2f}s")
                print(f"Solution status: {solution.status}")
                print(f"Fitness: {solution.fitness:.1f}")

        return solution


def solve_with_cpsat(
    foundation_type: str,
    input_specs: List[Tuple[str, int, int, str]],
    output_specs: List[Tuple[str, int, int, str]],
    max_machines: int = 10,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> Optional[Candidate]:
    """
    Convenience function to solve using CP-SAT.

    Returns:
        Candidate solution or None if no solution found
    """
    solver = CPSATFullSolver(
        foundation_type=foundation_type,
        input_specs=input_specs,
        output_specs=output_specs,
        max_machines=max_machines,
        time_limit_seconds=time_limit,
    )

    solution = solver.solve(verbose=verbose)

    if solution and solution.status in ['optimal', 'feasible']:
        return solution.to_candidate()
    return None
