"""
CP-SAT Solver for Shapez 2 - Constraint Programming approach.

Uses Google OR-Tools CP-SAT solver for optimal/near-optimal solutions.
This approach guarantees finding a solution if one exists.

Features:
- Optimal machine placement with no-overlap constraints
- Integration with A* router for belt connections
- Full input → machine → output routing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from ortools.sat.python import cp_model

from ..shapes.shape import Shape
from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS, BUILDING_PORTS
from .foundation_evolution import PlacedBuilding, Candidate
from .foundation_config import FOUNDATION_SPECS, FoundationSpec, Side
from .router import BeltRouter, Connection


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


@dataclass
class CPSATSolution:
    """Solution from CP-SAT solver."""
    machines: List[Tuple[BuildingType, int, int, int, Rotation]]  # (type, x, y, floor, rotation)
    belts: List[Tuple[int, int, int, BuildingType, Rotation]]  # (x, y, floor, belt_type, rotation)
    fitness: float
    solve_time: float
    status: str  # 'optimal', 'feasible', 'infeasible', 'timeout'
    routing_success: bool = False

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

        return Candidate(buildings=buildings, fitness=self.fitness, routing_success=self.routing_success)


def side_from_string(s: str) -> Side:
    """Convert side string to Side enum."""
    mapping = {'N': Side.NORTH, 'S': Side.SOUTH, 'E': Side.EAST, 'W': Side.WEST,
               'NORTH': Side.NORTH, 'SOUTH': Side.SOUTH, 'EAST': Side.EAST, 'WEST': Side.WEST}
    return mapping.get(s.upper(), Side.WEST)


def get_direction_for_side(side: Side, entering: bool = True) -> Rotation:
    """Get the rotation/direction for a belt entering or exiting a side."""
    if entering:
        # Items entering from this side need to face inward
        return {
            Side.NORTH: Rotation.SOUTH,  # Entering from north, going south
            Side.SOUTH: Rotation.NORTH,
            Side.EAST: Rotation.WEST,
            Side.WEST: Rotation.EAST,
        }[side]
    else:
        # Items exiting to this side need to face outward
        return {
            Side.NORTH: Rotation.NORTH,
            Side.SOUTH: Rotation.SOUTH,
            Side.EAST: Rotation.EAST,
            Side.WEST: Rotation.WEST,
        }[side]


class CPSATFullSolver:
    """
    Complete CP-SAT solver that handles system design, placement, and routing.

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
        self.spec = FOUNDATION_SPECS.get(foundation_type)
        if self.spec is None:
            raise ValueError(f"Unknown foundation type: {foundation_type}")

        self.grid_width = self.spec.grid_width
        self.grid_height = self.spec.grid_height
        self.num_floors = self.spec.num_floors

        # Parse shapes
        self.input_shapes = [Shape.from_code(code) for _, _, _, code in input_specs]
        self.output_shapes = [Shape.from_code(code) for _, _, _, code in output_specs]

        # Calculate port positions
        self.input_positions = []
        for side_str, pos, floor, _ in input_specs:
            side = side_from_string(side_str)
            gx, gy = self.spec.get_port_grid_position(side, pos)
            # Adjust to be inside the grid (ports are on the edge)
            if side == Side.NORTH:
                gy = 0
            elif side == Side.SOUTH:
                gy = self.grid_height - 1
            elif side == Side.WEST:
                gx = 0
            elif side == Side.EAST:
                gx = self.grid_width - 1
            self.input_positions.append((gx, gy, floor, side))

        self.output_positions = []
        for side_str, pos, floor, _ in output_specs:
            side = side_from_string(side_str)
            gx, gy = self.spec.get_port_grid_position(side, pos)
            if side == Side.NORTH:
                gy = 0
            elif side == Side.SOUTH:
                gy = self.grid_height - 1
            elif side == Side.WEST:
                gx = 0
            elif side == Side.EAST:
                gx = self.grid_width - 1
            self.output_positions.append((gx, gy, floor, side))

        # For viewer compatibility
        self.top_solutions: List[Candidate] = []
        self.config = None  # Set by caller if needed

    def solve(self, verbose: bool = False) -> Optional[CPSATSolution]:
        """
        Solve the complete problem: system design + placement + routing.

        Uses Benders-style lazy constraint generation:
        1. Solve placement with CP-SAT
        2. Try routing
        3. If routing fails, add nogood constraint and retry
        4. Repeat until success or timeout

        Returns:
            CPSATSolution with machines and belts, or None if infeasible
        """
        import time
        start_time = time.time()

        if verbose:
            print("=" * 50)
            print("CP-SAT SOLVER (with Lazy Constraints)")
            print("=" * 50)
            print(f"Inputs: {len(self.input_specs)}, Outputs: {len(self.output_specs)}")
            print(f"Grid: {self.grid_width}x{self.grid_height}x{self.num_floors}")
            print(f"Input positions: {self.input_positions}")
            print(f"Output positions: {self.output_positions}")

        # Step 1: Determine required machines (system design)
        if verbose:
            print("\nPhase 1: System Design")

        machine_types = self._determine_machines()

        if verbose:
            print(f"Machines needed: {[m.name for m in machine_types]}")

        # Step 2: Iteratively solve placement + routing with feedback
        if verbose:
            print("\nPhase 2: Iterative Placement + Routing")

        # No hard iteration limit - keep trying until timeout
        nogood_placements = []  # Store failed placements to avoid
        best_solution = None
        iteration = 0

        while True:
            iteration += 1

            # Check timeout
            remaining_time = self.time_limit - (time.time() - start_time)
            if remaining_time <= 0:
                if verbose:
                    print(f"\nIteration {iteration}: Timeout reached")
                break

            # Need at least 5 seconds for placement + routing
            if remaining_time < 5:
                if verbose:
                    print(f"\nIteration {iteration}: Insufficient time remaining ({remaining_time:.1f}s)")
                break

            if verbose:
                remaining_time = self.time_limit - (time.time() - start_time)
                print(f"\n--- Iteration {iteration} (time remaining: {remaining_time:.1f}s) ---")

            # Solve placement with CP-SAT (excluding nogood placements)
            # Give placement solver limited time to allow multiple attempts
            placement_time_limit = min(30.0, remaining_time * 0.3)  # 30% of remaining time, max 30s
            placement = self._solve_placement(
                machine_types, nogood_placements,
                time_limit=placement_time_limit,
                iteration=iteration,
                verbose=verbose
            )

            if placement is None:
                if verbose:
                    print("No valid placement found")
                break

            machines, placement_status = placement

            if verbose:
                print(f"Placement status: {placement_status}")
                for bt, x, y, f, rot in machines:
                    print(f"  {bt.name}: ({x}, {y}, floor {f}), rot={rot.name}")

            # Try routing with this placement
            if verbose:
                print("Attempting routing...")

            belts, routing_success = self._route_all_connections(machines, verbose)

            if verbose:
                print(f"Routing success: {routing_success}")
                print(f"Total belts placed: {len(belts)}")

            # Calculate fitness for this solution
            fitness = 0.0

            # Throughput score (0-50 points based on throughput)
            if machines:
                throughput_per_output = self._calculate_throughput(machines, verbose and iteration == 0)

                # For N outputs, theoretical max is 180/N items/min (perfect splitting)
                # But with cutters, max is limited by root machine max_rate / N
                # Scale: 0 items/min = 0 points, theoretical_max = 50 points
                num_outputs = len(self.output_positions)
                theoretical_max = 180.0 / max(1, num_outputs)  # Perfect splitting
                throughput_ratio = throughput_per_output / theoretical_max if theoretical_max > 0 else 0
                fitness += min(50.0, throughput_ratio * 50.0)

            # Routing success (hard constraint) - 50 points
            if routing_success:
                fitness += 50.0

                # Compactness bonus (fewer belts = better) - up to 20 points
                fitness += max(0, 20.0 - len(belts) * 0.5)

            # Store solution
            current_solution = CPSATSolution(
                machines=machines,
                belts=belts,
                fitness=fitness,
                solve_time=time.time() - start_time,
                status=placement_status,
                routing_success=routing_success,
            )

            # Keep best solution
            if best_solution is None or fitness > best_solution.fitness:
                best_solution = current_solution

            # If routing succeeded, we're done!
            if routing_success:
                if verbose:
                    print(f"\n✓ SUCCESS in {iteration} iteration(s)")
                break

            # Routing failed - add this placement to nogood list
            if verbose:
                print("Routing failed - adding nogood constraint")

            # Store placement as nogood (positions only, simplified)
            placement_tuple = tuple((x, y, f) for _, x, y, f, _ in machines)
            nogood_placements.append(placement_tuple)

        total_time = time.time() - start_time

        if best_solution is None:
            return CPSATSolution(
                machines=[], belts=[], fitness=0.0,
                solve_time=total_time, status='infeasible'
            )

        # Store for viewer
        candidate = best_solution.to_candidate()
        self.top_solutions = [candidate]

        if verbose:
            print(f"\n{'='*50}")
            print(f"FINAL RESULT:")
            print(f"Total solve time: {total_time:.2f}s")
            print(f"Solution status: {best_solution.status}")
            print(f"Routing success: {best_solution.routing_success}")
            print(f"Fitness: {best_solution.fitness:.1f}")
            print(f"Machines: {len(best_solution.machines)}")
            print(f"Belts: {len(best_solution.belts)}")

            # Show throughput analysis
            if best_solution.machines:
                throughput = self._calculate_throughput(best_solution.machines, verbose=False)
                print(f"Throughput per output: {throughput:.2f} items/min")

            print(f"{'='*50}")

        return best_solution

    def _determine_machines(self) -> List[BuildingType]:
        """
        Determine which machines are needed based on input/output shapes.

        For throughput optimization:
        - Use splitters (180 items/min) for pure splitting (no shape change)
        - Use cutters (45 items/min) only when shape transformation is needed

        This is a heuristic approach - a full CP-SAT encoding would model
        the shape transformations as constraints.
        """
        import math

        machines = []

        # Analyze transformation requirements
        num_outputs = len(self.output_shapes)
        num_inputs = len(self.input_shapes)

        # Check if shape transformation is needed
        needs_shape_transformation = False
        if num_inputs > 0 and num_outputs > 0:
            # For simplicity, check if any input differs from any output
            for inp in self.input_shapes:
                for out in self.output_shapes:
                    if inp.to_code() != out.to_code():
                        inp_parts = self._count_parts(inp)
                        out_parts = self._count_parts(out)
                        if inp_parts != out_parts:
                            needs_shape_transformation = True
                            break
                if needs_shape_transformation:
                    break

        # Analyze if we need splitting based on shape transformation
        # Even if num_outputs == num_inputs, we might need splitting if:
        # - Each input produces multiple different transformed outputs
        # - E.g., 1 full shape → 4 corners (needs 1→4 splitting per input)

        # Check if this is a "split each input into multiple outputs" scenario
        splitting_needed = False
        if needs_shape_transformation and num_inputs > 0 and num_outputs > 0:
            # Count how many unique output shapes exist
            unique_outputs = len(set(out.to_code() for out in self.output_shapes))

            # If we have multiple unique outputs per input on average, we need splitting
            if unique_outputs >= num_inputs:
                # Assume each input should produce all unique outputs for maximum throughput
                outputs_per_input = unique_outputs
                splitting_needed = True

        # Standard case: more outputs than inputs
        if not splitting_needed and num_outputs > num_inputs and num_inputs > 0:
            outputs_per_input = num_outputs / num_inputs
            splitting_needed = True

        # Create machines for splitting
        if splitting_needed and num_inputs > 0:
            if needs_shape_transformation:
                # Need cutters for shape transformation
                # For maximum throughput, create independent trees for each input
                # Each tree: 1 input -> K outputs requires (2^depth - 1) cutters
                # Total: num_inputs × cutters_per_tree

                # Calculate cutters needed per input tree
                depth = int(math.ceil(math.log2(outputs_per_input)))  # Tree depth for one input
                cutters_per_tree = (2 ** depth) - 1  # Cutters in one tree

                # Create independent trees for each input (maximizes throughput)
                total_cutters = num_inputs * cutters_per_tree

                for _ in range(total_cutters):
                    machines.append(BuildingType.CUTTER)
            else:
                # Just splitting, no transformation - use splitters for better throughput
                # Splitters: 180 items/min (full belt speed) vs Cutters: 45 items/min
                # Create independent splitter trees for each input

                depth = int(math.ceil(math.log2(outputs_per_input)))
                splitters_per_tree = (2 ** depth) - 1

                total_splitters = num_inputs * splitters_per_tree

                for _ in range(total_splitters):
                    machines.append(BuildingType.SPLITTER)

        # Check for additional transformations on individual outputs
        for i, (inp, out) in enumerate(zip(self.input_shapes, self.output_shapes)):
            if inp.to_code() != out.to_code():
                inp_parts = self._count_parts(inp)
                out_parts = self._count_parts(out)

                if out_parts < inp_parts:
                    # Need to cut away parts
                    if BuildingType.CUTTER not in machines:
                        machines.append(BuildingType.CUTTER)
                elif out_parts > inp_parts:
                    # Need to stack
                    machines.append(BuildingType.STACKER_BENT)  # Use bent for better throughput (45 vs 30)

        # Default: at least one machine for basic operations
        if not machines:
            machines.append(BuildingType.CUTTER)

        return machines

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

    def _solve_placement(
        self,
        machine_types: List[BuildingType],
        nogood_placements: List[Tuple[Tuple[int, int, int], ...]] = None,
        time_limit: float = 30.0,
        iteration: int = 1,
        verbose: bool = False
    ) -> Optional[Tuple[List[Tuple[BuildingType, int, int, int, Rotation]], str]]:
        """
        Solve machine placement using CP-SAT.

        Args:
            machine_types: List of machine types to place
            nogood_placements: List of placement tuples to exclude (failed routing)
            verbose: Print debug info

        Returns:
            Tuple of (machine placements, status) or None if infeasible
        """
        if nogood_placements is None:
            nogood_placements = []

        model = cp_model.CpModel()
        num_machines = len(machine_types)

        # Position variables for each machine
        machine_x = []
        machine_y = []
        machine_floor = []
        machine_rotation = []

        for i, bt in enumerate(machine_types):
            spec = BUILDING_SPECS.get(bt)
            w = spec.width if spec else 1
            h = spec.height if spec else 1

            # Leave room at edges for belts
            max_x = self.grid_width - w - 1
            max_y = self.grid_height - h - 1

            machine_x.append(model.NewIntVar(1, max(1, max_x), f'x_{i}'))
            machine_y.append(model.NewIntVar(1, max(1, max_y), f'y_{i}'))
            machine_floor.append(model.NewIntVar(0, self.num_floors - 1, f'floor_{i}'))
            machine_rotation.append(model.NewIntVar(0, 3, f'rot_{i}'))

        # No overlap constraints
        for i in range(num_machines):
            for j in range(i + 1, num_machines):
                spec_i = BUILDING_SPECS.get(machine_types[i])
                spec_j = BUILDING_SPECS.get(machine_types[j])
                wi, hi = (spec_i.width, spec_i.height) if spec_i else (1, 1)
                wj, hj = (spec_j.width, spec_j.height) if spec_j else (1, 1)

                # Same floor -> no overlap
                same_floor = model.NewBoolVar(f'same_floor_{i}_{j}')
                model.Add(machine_floor[i] == machine_floor[j]).OnlyEnforceIf(same_floor)
                model.Add(machine_floor[i] != machine_floor[j]).OnlyEnforceIf(same_floor.Not())

                # No overlap conditions
                no_x_overlap_left = model.NewBoolVar(f'no_x_left_{i}_{j}')
                no_x_overlap_right = model.NewBoolVar(f'no_x_right_{i}_{j}')
                no_y_overlap_top = model.NewBoolVar(f'no_y_top_{i}_{j}')
                no_y_overlap_bottom = model.NewBoolVar(f'no_y_bottom_{i}_{j}')

                model.Add(machine_x[i] + wi <= machine_x[j]).OnlyEnforceIf(no_x_overlap_left)
                model.Add(machine_x[j] + wj <= machine_x[i]).OnlyEnforceIf(no_x_overlap_right)
                model.Add(machine_y[i] + hi <= machine_y[j]).OnlyEnforceIf(no_y_overlap_top)
                model.Add(machine_y[j] + hj <= machine_y[i]).OnlyEnforceIf(no_y_overlap_bottom)

                # If same floor, at least one separation must exist
                model.AddBoolOr([
                    same_floor.Not(),
                    no_x_overlap_left,
                    no_x_overlap_right,
                    no_y_overlap_top,
                    no_y_overlap_bottom,
                ])

        # Add nogood constraints (exclude placements that failed routing)
        for nogood in nogood_placements:
            if len(nogood) != num_machines:
                continue  # Skip if different number of machines

            # Create constraint: NOT (all positions match this nogood)
            # This is equivalent to: at least one machine must be in a different position
            different_position_vars = []
            for i in range(num_machines):
                nogood_x, nogood_y, nogood_f = nogood[i]

                # Create boolean: this machine is in different position
                diff_x = model.NewBoolVar(f'nogood_{len(nogood_placements)}_m{i}_diff_x')
                diff_y = model.NewBoolVar(f'nogood_{len(nogood_placements)}_m{i}_diff_y')
                diff_f = model.NewBoolVar(f'nogood_{len(nogood_placements)}_m{i}_diff_f')

                model.Add(machine_x[i] != nogood_x).OnlyEnforceIf(diff_x)
                model.Add(machine_x[i] == nogood_x).OnlyEnforceIf(diff_x.Not())

                model.Add(machine_y[i] != nogood_y).OnlyEnforceIf(diff_y)
                model.Add(machine_y[i] == nogood_y).OnlyEnforceIf(diff_y.Not())

                model.Add(machine_floor[i] != nogood_f).OnlyEnforceIf(diff_f)
                model.Add(machine_floor[i] == nogood_f).OnlyEnforceIf(diff_f.Not())

                # At least one coordinate differs
                is_different = model.NewBoolVar(f'nogood_{len(nogood_placements)}_m{i}_diff')
                model.AddBoolOr([diff_x, diff_y, diff_f]).OnlyEnforceIf(is_different)
                model.AddBoolAnd([diff_x.Not(), diff_y.Not(), diff_f.Not()]).OnlyEnforceIf(is_different.Not())

                different_position_vars.append(is_different)

            # At least one machine must be in a different position
            model.AddBoolOr(different_position_vars)

        # Objective: minimize total distance while encouraging spatial separation
        # Prefer different X or Y coordinates for better routing
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2

        distance_terms = []
        overlap_penalty_terms = []

        for i in range(num_machines):
            dx = model.NewIntVar(-self.grid_width, self.grid_width, f'dx_{i}')
            dy = model.NewIntVar(-self.grid_height, self.grid_height, f'dy_{i}')
            abs_dx = model.NewIntVar(0, self.grid_width, f'abs_dx_{i}')
            abs_dy = model.NewIntVar(0, self.grid_height, f'abs_dy_{i}')

            model.Add(dx == machine_x[i] - center_x)
            model.Add(dy == machine_y[i] - center_y)
            model.AddAbsEquality(abs_dx, dx)
            model.AddAbsEquality(abs_dy, dy)

            distance_terms.extend([abs_dx, abs_dy])

        # Add penalty for machines at same X,Y (different floors)
        for i in range(num_machines):
            for j in range(i + 1, num_machines):
                same_x = model.NewBoolVar(f'same_x_{i}_{j}')
                same_y = model.NewBoolVar(f'same_y_{i}_{j}')

                model.Add(machine_x[i] == machine_x[j]).OnlyEnforceIf(same_x)
                model.Add(machine_x[i] != machine_x[j]).OnlyEnforceIf(same_x.Not())
                model.Add(machine_y[i] == machine_y[j]).OnlyEnforceIf(same_y)
                model.Add(machine_y[i] != machine_y[j]).OnlyEnforceIf(same_y.Not())

                # Both same X and same Y = vertical stacking (penalize heavily)
                both_same = model.NewBoolVar(f'both_same_{i}_{j}')
                model.AddBoolAnd([same_x, same_y]).OnlyEnforceIf(both_same)
                model.AddBoolOr([same_x.Not(), same_y.Not()]).OnlyEnforceIf(both_same.Not())

                # Add penalty variable (10 points if stacked vertically)
                penalty = model.NewIntVar(0, 10, f'penalty_{i}_{j}')
                model.Add(penalty == 10).OnlyEnforceIf(both_same)
                model.Add(penalty == 0).OnlyEnforceIf(both_same.Not())

                overlap_penalty_terms.append(penalty)

        # Objective changes based on iteration to encourage diversity
        # Early iterations: spread out (minimize overlap penalty more)
        # Later iterations: compact (minimize distance more)
        spread_weight = max(5, 20 - iteration * 2)  # Decreases from 20 to 5
        compact_weight = min(3, iteration // 2)     # Increases from 0 to 3

        objective_terms = []
        if overlap_penalty_terms:
            # Heavily penalize vertical stacking (harder to route)
            objective_terms.extend([p * spread_weight for p in overlap_penalty_terms])
        if distance_terms and compact_weight > 0:
            # Some compactness (but less important than spreading)
            objective_terms.extend([d * compact_weight for d in distance_terms])

        if objective_terms:
            model.Minimize(sum(objective_terms))

        # Solve with limited time per attempt
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 8

        # Encourage diversity through search parameters
        if iteration > 1:
            solver.parameters.random_seed = iteration * 42  # Different seed each iteration

        status = solver.Solve(model)

        status_name = {
            cp_model.OPTIMAL: 'optimal',
            cp_model.FEASIBLE: 'feasible',
            cp_model.INFEASIBLE: 'infeasible',
            cp_model.MODEL_INVALID: 'invalid',
            cp_model.UNKNOWN: 'timeout',
        }.get(status, 'unknown')

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return None

        # Extract solution
        machines = []
        for i, bt in enumerate(machine_types):
            x = solver.Value(machine_x[i])
            y = solver.Value(machine_y[i])
            floor = solver.Value(machine_floor[i])
            rot_val = solver.Value(machine_rotation[i])
            rot = [Rotation.EAST, Rotation.SOUTH, Rotation.WEST, Rotation.NORTH][rot_val]
            machines.append((bt, x, y, floor, rot))

        return machines, status_name

    def _route_all_connections(
        self,
        machines: List[Tuple[BuildingType, int, int, int, Rotation]],
        verbose: bool = False
    ) -> Tuple[List[Tuple[int, int, int, BuildingType, Rotation]], bool]:
        """
        Route all belt connections using A* pathfinding.

        Connections:
        1. Input ports → First machine input
        2. Machine outputs → Next machine inputs (or output ports)
        3. Final machine outputs → Output ports
        """
        all_belts = []
        all_success = True

        # Create router
        router = BeltRouter(
            self.grid_width, self.grid_height, self.num_floors,
            use_belt_ports=True, max_belt_ports=4
        )

        # Mark machine positions as occupied
        occupied = set()
        for bt, x, y, floor, rot in machines:
            spec = BUILDING_SPECS.get(bt)
            w = spec.width if spec else 1
            h = spec.height if spec else 1
            d = spec.depth if spec else 1

            for dx in range(w):
                for dy in range(h):
                    for dz in range(d):
                        occupied.add((x + dx, y + dy, floor + dz))

        router.set_occupied(occupied)

        # Get machine input/output positions
        machine_inputs = []  # List of (x, y, floor) for each machine's input
        machine_outputs = []  # List of (x, y, floor) for each machine's output

        for bt, mx, my, mfloor, rot in machines:
            ports = BUILDING_PORTS.get(bt, {'inputs': [(0, 0, 0)], 'outputs': [(0, 0, 0)]})

            # Get first input port position (relative to machine)
            if ports['inputs']:
                inp = ports['inputs'][0]
                # Adjust based on rotation (simplified - assume EAST facing)
                input_pos = (mx + inp[0], my + inp[1], mfloor + inp[2])
                # Move input position to be adjacent to machine (accessible)
                input_pos = (mx - 1, my, mfloor)  # Input from west
            else:
                input_pos = (mx - 1, my, mfloor)

            machine_inputs.append(input_pos)

            # Get first output port position
            if ports['outputs']:
                out = ports['outputs'][0]
                spec = BUILDING_SPECS.get(bt)
                w = spec.width if spec else 1
                output_pos = (mx + w, my + out[1], mfloor + out[2])
            else:
                spec = BUILDING_SPECS.get(bt)
                w = spec.width if spec else 1
                output_pos = (mx + w, my, mfloor)

            machine_outputs.append(output_pos)

        # Collect all output positions from all machines
        all_machine_outputs = []  # List of lists: machine_outputs[machine_idx][output_idx] = (x, y, floor)
        for bt, mx, my, mfloor, rot in machines:
            ports = BUILDING_PORTS.get(bt, {'outputs': [(0, 0, 0)]})
            spec = BUILDING_SPECS.get(bt)
            w = spec.width if spec else 1

            outputs_for_machine = []
            for out in ports['outputs']:
                # Output position relative to machine
                output_pos = (mx + w, my + out[1], mfloor + out[2])
                outputs_for_machine.append(output_pos)

            all_machine_outputs.append(outputs_for_machine)

        # Determine tree structure
        # With N inputs, M outputs, and K machines:
        # - We have N independent trees (one per input)
        # - Each tree has K/N machines
        # - Each tree serves M/N outputs

        num_machines = len(machines)
        num_inputs = len(self.input_positions)
        num_outputs_needed = len(self.output_positions)

        if num_inputs == 0 or num_machines == 0:
            return all_belts, all_success

        # Calculate outputs per input (even distribution)
        outputs_per_input = num_outputs_needed / num_inputs if num_inputs > 0 else num_outputs_needed
        machines_per_input = num_machines / num_inputs if num_inputs > 0 else num_machines

        if verbose:
            print(f"\n  Multi-tree routing:")
            print(f"    Inputs: {num_inputs}")
            print(f"    Machines: {num_machines} ({machines_per_input:.1f} per input)")
            print(f"    Outputs: {num_outputs_needed} ({outputs_per_input:.1f} per input)\n")

        # Route each independent tree
        for tree_idx in range(num_inputs):
            # Determine machine range for this tree
            machine_start = int(tree_idx * machines_per_input)
            machine_end = int((tree_idx + 1) * machines_per_input)

            # Determine output range for this tree
            output_start = int(tree_idx * outputs_per_input)
            output_end = int((tree_idx + 1) * outputs_per_input)

            tree_machines = list(range(machine_start, min(machine_end, num_machines)))
            tree_outputs = list(range(output_start, min(output_end, num_outputs_needed)))

            if not tree_machines or not tree_outputs:
                continue

            if verbose:
                print(f"  Tree {tree_idx}: Machines {machine_start}-{machine_end-1}, Outputs {output_start}-{output_end-1}")

            # Route input to root machine of this tree
            if tree_idx < len(self.input_positions):
                inp_x, inp_y, inp_floor, inp_side = self.input_positions[tree_idx]

                # Start position inside grid from port
                if inp_side == Side.WEST:
                    start = (1, inp_y, inp_floor)
                elif inp_side == Side.EAST:
                    start = (self.grid_width - 2, inp_y, inp_floor)
                elif inp_side == Side.NORTH:
                    start = (inp_x, 1, inp_floor)
                else:  # SOUTH
                    start = (inp_x, self.grid_height - 2, inp_floor)

                root_machine = tree_machines[0]
                goal = machine_inputs[root_machine]

                if verbose:
                    print(f"    Input {tree_idx} -> Machine {root_machine}")

                path = router.find_path(start, goal, allow_belt_ports=True)
                if path:
                    belts = router.path_to_belts(path)
                    all_belts.extend(belts)
                    for bx, by, bf, _, _ in belts:
                        router.add_occupied(bx, by, bf)
                    if verbose:
                        print(f"      Success: {len(belts)} belts")
                else:
                    all_success = False
                    if verbose:
                        print(f"      FAILED")

            # Route tree internals and outputs
            if len(tree_machines) == 1:
                # Single machine in tree: route outputs directly to this tree's output ports
                machine_idx = tree_machines[0]
                machine_outputs_list = all_machine_outputs[machine_idx]

                for local_out_idx, output_idx in enumerate(tree_outputs):
                    if local_out_idx >= len(machine_outputs_list):
                        break

                    start = machine_outputs_list[local_out_idx]
                    out_x, out_y, out_floor, out_side = self.output_positions[output_idx]

                    if out_side == Side.EAST:
                        goal = (self.grid_width - 2, out_y, out_floor)
                    elif out_side == Side.WEST:
                        goal = (1, out_y, out_floor)
                    elif out_side == Side.SOUTH:
                        goal = (out_x, self.grid_height - 2, out_floor)
                    else:  # NORTH
                        goal = (out_x, 1, out_floor)

                    if verbose:
                        print(f"    Machine {machine_idx} -> Output {output_idx}")

                    path = router.find_path(start, goal, allow_belt_ports=True)
                    if path:
                        belts = router.path_to_belts(path)
                        all_belts.extend(belts)
                        for bx, by, bf, _, _ in belts:
                            router.add_occupied(bx, by, bf)
                        if verbose:
                            print(f"      Success: {len(belts)} belts")
                    else:
                        all_success = False
                        if verbose:
                            print(f"      FAILED")

            elif len(tree_machines) >= 2:
                # Multiple machines: binary tree topology
                # Root machine (tree_machines[0]) → children (tree_machines[1:])
                root_machine = tree_machines[0]
                root_outputs = all_machine_outputs[root_machine]

                # Route root to child machines
                for i, child_idx in enumerate(tree_machines[1:], start=1):
                    if i - 1 >= len(root_outputs):
                        break

                    start = root_outputs[i - 1]
                    goal = machine_inputs[child_idx]

                    if verbose:
                        print(f"    Machine {root_machine} -> Machine {child_idx}")

                    path = router.find_path(start, goal, allow_belt_ports=True)
                    if path:
                        belts = router.path_to_belts(path)
                        all_belts.extend(belts)
                        for bx, by, bf, _, _ in belts:
                            router.add_occupied(bx, by, bf)
                        if verbose:
                            print(f"      Success: {len(belts)} belts")
                    else:
                        all_success = False
                        if verbose:
                            print(f"      FAILED")

                # Route child machines to output ports
                # Distribute outputs evenly among children
                outputs_per_child = len(tree_outputs) // max(1, len(tree_machines) - 1)

                for child_local_idx, child_machine_idx in enumerate(tree_machines[1:]):
                    child_outputs = all_machine_outputs[child_machine_idx]
                    child_output_start = child_local_idx * outputs_per_child
                    child_output_end = (child_local_idx + 1) * outputs_per_child

                    for local_out_idx, child_output_pos in enumerate(child_outputs):
                        output_global_idx = child_output_start + local_out_idx
                        if output_global_idx >= len(tree_outputs):
                            break

                        port_idx = tree_outputs[output_global_idx]
                        out_x, out_y, out_floor, out_side = self.output_positions[port_idx]

                        if out_side == Side.EAST:
                            goal = (self.grid_width - 2, out_y, out_floor)
                        elif out_side == Side.WEST:
                            goal = (1, out_y, out_floor)
                        elif out_side == Side.SOUTH:
                            goal = (out_x, self.grid_height - 2, out_floor)
                        else:  # NORTH
                            goal = (out_x, 1, out_floor)

                        if verbose:
                            print(f"    Machine {child_machine_idx} -> Output {port_idx}")

                        path = router.find_path(child_output_pos, goal, allow_belt_ports=True)
                        if path:
                            belts = router.path_to_belts(path)
                            all_belts.extend(belts)
                            for bx, by, bf, _, _ in belts:
                                router.add_occupied(bx, by, bf)
                            if verbose:
                                print(f"      Success: {len(belts)} belts")
                        else:
                            all_success = False
                            if verbose:
                                print(f"      FAILED")

        # Add edge connection belts from ports to the routing start/end positions
        if all_success:
            if verbose:
                print(f"\n  Adding edge connection belts...")

            # Add belts from input ports to first routing position
            for tree_idx, (inp_x, inp_y, inp_floor, inp_side) in enumerate(self.input_positions):
                # Determine the first routing position (one tile inside)
                if inp_side == Side.WEST:
                    inner_pos = (1, inp_y, inp_floor)
                    # Add belt from port to inner position
                    all_belts.append((inp_x, inp_y, inp_floor, BuildingType.BELT_FORWARD, Rotation.EAST))
                elif inp_side == Side.EAST:
                    inner_pos = (self.grid_width - 2, inp_y, inp_floor)
                    # Add belt from port to inner position
                    all_belts.append((inp_x, inp_y, inp_floor, BuildingType.BELT_FORWARD, Rotation.WEST))
                elif inp_side == Side.NORTH:
                    inner_pos = (inp_x, 1, inp_floor)
                    # Add belt from port to inner position
                    all_belts.append((inp_x, inp_y, inp_floor, BuildingType.BELT_FORWARD, Rotation.SOUTH))
                else:  # SOUTH
                    inner_pos = (inp_x, self.grid_height - 2, inp_floor)
                    # Add belt from port to inner position
                    all_belts.append((inp_x, inp_y, inp_floor, BuildingType.BELT_FORWARD, Rotation.NORTH))

            # Add belts from last routing position to output ports
            for out_x, out_y, out_floor, out_side in self.output_positions:
                # Determine the last routing position (one tile inside)
                if out_side == Side.EAST:
                    inner_pos = (self.grid_width - 2, out_y, out_floor)
                    # Add belt from inner position to port
                    all_belts.append((out_x, out_y, out_floor, BuildingType.BELT_FORWARD, Rotation.EAST))
                elif out_side == Side.WEST:
                    inner_pos = (1, out_y, out_floor)
                    # Add belt from inner position to port
                    all_belts.append((out_x, out_y, out_floor, BuildingType.BELT_FORWARD, Rotation.WEST))
                elif out_side == Side.SOUTH:
                    inner_pos = (out_x, self.grid_height - 2, out_floor)
                    # Add belt from inner position to port
                    all_belts.append((out_x, out_y, out_floor, BuildingType.BELT_FORWARD, Rotation.SOUTH))
                else:  # NORTH
                    inner_pos = (out_x, 1, out_floor)
                    # Add belt from inner position to port
                    all_belts.append((out_x, out_y, out_floor, BuildingType.BELT_FORWARD, Rotation.NORTH))

            if verbose:
                print(f"    Added {len(self.input_positions)} input edge belts")
                print(f"    Added {len(self.output_positions)} output edge belts")

        return all_belts, all_success

    def _calculate_throughput(
        self,
        machines: List[Tuple[BuildingType, int, int, int, Rotation]],
        verbose: bool = False
    ) -> float:
        """
        Calculate actual throughput (items/min) to output ports.

        For tree topology:
        - Input belt provides 180 items/min (max belt speed)
        - Each machine has a max_rate processing capacity
        - Splitters: 180 items/min (full belt speed, no bottleneck)
        - Cutters: 45 items/min (processing bottleneck)
        - With multiple inputs, we have multiple independent trees

        Returns:
            Minimum throughput per output port (items/min)
        """
        if not machines:
            return 0.0

        # Belt speed (max upgraded)
        BELT_SPEED = 180.0  # items/min

        # Get machine capacities (max upgraded tier 5)
        machine_types = []
        machine_capacities = []
        for bt, _, _, _, _ in machines:
            machine_types.append(bt)
            spec = BUILDING_SPECS.get(bt)
            if spec:
                machine_capacities.append(spec.max_rate)
            else:
                machine_capacities.append(30.0)  # Default

        num_machines = len(machines)
        num_outputs = len(self.output_positions)
        num_inputs = len(self.input_positions)

        if num_machines == 0:
            return 0.0

        # Single machine case: output = min(input belt, machine capacity) / num_outputs
        if num_machines == 1:
            throughput_in = min(BELT_SPEED, machine_capacities[0])
            return throughput_in / max(1, num_outputs)

        # Multi-tree topology case:
        # With N inputs and M outputs, we have N independent trees
        # Each tree processes outputs_per_tree outputs
        # Throughput per tree is limited by its root machine

        root_type = machine_types[0]
        root_capacity = machine_capacities[0]

        # Check if we're using splitters (high throughput) or cutters (bottleneck)
        if root_type == BuildingType.SPLITTER:
            # Splitters maintain full belt speed (no processing bottleneck)
            throughput_per_tree = BELT_SPEED
            machine_type_name = "Splitter"
        else:
            # Cutters or other processing machines create bottleneck
            throughput_per_tree = min(BELT_SPEED, root_capacity)
            machine_type_name = root_type.name

        # Calculate outputs per tree (assuming even distribution)
        outputs_per_tree = num_outputs / max(1, num_inputs)

        # Throughput per output = tree throughput / outputs per tree
        throughput_per_output = throughput_per_tree / max(1, outputs_per_tree)

        if verbose:
            print(f"\n=== Throughput Analysis ===")
            print(f"Machine type: {machine_type_name}")
            print(f"Input belts: {num_inputs}")
            print(f"Total outputs: {num_outputs}")
            print(f"Outputs per input tree: {outputs_per_tree:.1f}")
            print(f"Belt speed: {BELT_SPEED:.1f} items/min")
            print(f"Root machine capacity: {root_capacity:.1f} items/min")
            print(f"Throughput per tree: {throughput_per_tree:.1f} items/min")
            print(f"Throughput per output: {throughput_per_output:.1f} items/min")
            print(f"Total system throughput: {throughput_per_output * num_outputs:.1f} items/min")
            if root_type == BuildingType.SPLITTER:
                print(f"✓ Using splitters - NO processing bottleneck!")
            else:
                print(f"⚠ Using {machine_type_name} - processing bottleneck at {root_capacity:.1f} items/min")
            print(f"=========================")

        return throughput_per_output


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


# Keep old class names for backwards compatibility
CPSATLayoutSolver = CPSATFullSolver
CPSATSystemSolver = CPSATFullSolver
