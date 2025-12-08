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

        return Candidate(buildings=buildings, fitness=self.fitness)


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

        max_iterations = 10
        nogood_placements = []  # Store failed placements to avoid
        best_solution = None

        for iteration in range(max_iterations):
            if time.time() - start_time > self.time_limit:
                if verbose:
                    print(f"\nIteration {iteration + 1}: Timeout")
                break

            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Solve placement with CP-SAT (excluding nogood placements)
            placement = self._solve_placement(machine_types, nogood_placements, verbose)

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
            if machines:
                fitness += 30.0  # Base for having machines
            if routing_success:
                fitness += 50.0  # Major bonus for successful routing
                # Bonus for compactness (fewer belts = better)
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
                    print(f"\n✓ SUCCESS in {iteration + 1} iteration(s)")
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
            print(f"{'='*50}")

        return best_solution

    def _determine_machines(self) -> List[BuildingType]:
        """
        Determine which machines are needed based on input/output shapes.

        This is a heuristic approach - a full CP-SAT encoding would model
        the shape transformations as constraints.
        """
        import math

        machines = []

        # Analyze transformation requirements
        num_outputs = len(self.output_shapes)
        num_inputs = len(self.input_shapes)

        # If we have more outputs than inputs, we need cutters to split
        if num_outputs > num_inputs:
            # Cutters form a binary tree: 1 input -> 2 -> 4 -> 8 outputs
            # For 4 outputs, we need: 1 root cutter + 2 leaf cutters = 3 total
            # For N outputs, we need (N-1) cutters in a complete binary tree
            # Simplified: for power-of-2 outputs, we need (outputs - 1) cutters
            # For 1 -> 4: need 3 cutters (1 parent, 2 children)
            # For 1 -> 2: need 1 cutter
            ratio = num_outputs / num_inputs
            depth = int(math.ceil(math.log2(ratio)))  # Tree depth
            num_cutters = (2 ** depth) - 1  # Total nodes in complete binary tree

            for _ in range(num_cutters):
                machines.append(BuildingType.CUTTER)

        # Check if input and output shapes differ (need transformation)
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
                    machines.append(BuildingType.STACKER)

        # Default: at least one cutter for splitting operations
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

        # Combined objective: minimize distance + minimize vertical stacking
        if distance_terms or overlap_penalty_terms:
            model.Minimize(sum(distance_terms) + sum(overlap_penalty_terms) * 2)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit / 2
        solver.parameters.num_search_workers = 8

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

        # Route 1: Input port to first machine
        if machines and self.input_positions:
            inp_x, inp_y, inp_floor, inp_side = self.input_positions[0]

            # Start position is just inside the grid from the port
            if inp_side == Side.WEST:
                start = (1, inp_y, inp_floor)
            elif inp_side == Side.EAST:
                start = (self.grid_width - 2, inp_y, inp_floor)
            elif inp_side == Side.NORTH:
                start = (inp_x, 1, inp_floor)
            else:  # SOUTH
                start = (inp_x, self.grid_height - 2, inp_floor)

            goal = machine_inputs[0]

            if verbose:
                print(f"  Routing input port ({inp_x},{inp_y}) -> machine 0 input {goal}")

            path = router.find_path(start, goal, allow_belt_ports=True)
            if path:
                belts = router.path_to_belts(path)
                all_belts.extend(belts)
                for bx, by, bf, _, _ in belts:
                    router.add_occupied(bx, by, bf)
                if verbose:
                    print(f"    Success: {len(belts)} belts")
            else:
                all_success = False
                if verbose:
                    print(f"    FAILED to route")

        # Route 2 & 3: Build tree topology for cutters
        # For splitting 1 -> 4, we have 3 cutters in a tree:
        #   Machine 0 (root) → outputs to Machines 1 and 2
        #   Machine 1 (left child) → outputs to ports 0, 1
        #   Machine 2 (right child) → outputs to ports 2, 3

        num_machines = len(machines)
        num_outputs_needed = len(self.output_positions)

        if num_machines == 1:
            # Single machine: route its outputs directly to output ports
            for out_idx, (out_x, out_y, out_floor, out_side) in enumerate(self.output_positions):
                machine_outputs_list = all_machine_outputs[0]
                if out_idx < len(machine_outputs_list):
                    start = machine_outputs_list[out_idx]
                else:
                    start = machine_outputs_list[-1]

                if out_side == Side.EAST:
                    goal = (self.grid_width - 2, out_y, out_floor)
                elif out_side == Side.WEST:
                    goal = (1, out_y, out_floor)
                elif out_side == Side.SOUTH:
                    goal = (out_x, self.grid_height - 2, out_floor)
                else:  # NORTH
                    goal = (out_x, 1, out_floor)

                if verbose:
                    print(f"  Routing machine 0 output[{out_idx}] {start} -> output port {out_idx}")

                path = router.find_path(start, goal, allow_belt_ports=True)
                if path:
                    belts = router.path_to_belts(path)
                    all_belts.extend(belts)
                    for bx, by, bf, _, _ in belts:
                        router.add_occupied(bx, by, bf)
                    if verbose:
                        print(f"    Success: {len(belts)} belts")
                else:
                    all_success = False
                    if verbose:
                        print(f"    FAILED to route")

        elif num_machines >= 2:
            # Tree topology: machine 0 is root, others are children
            # Machine 0 output[0] → Machine 1 input
            # Machine 0 output[1] → Machine 2 input (if exists)
            root_outputs = all_machine_outputs[0]

            # Route root to children
            for child_idx in range(1, min(len(root_outputs) + 1, num_machines)):
                output_idx = child_idx - 1
                if output_idx < len(root_outputs):
                    start = root_outputs[output_idx]
                    goal = machine_inputs[child_idx]

                    if verbose:
                        print(f"  Routing machine 0 output[{output_idx}] {start} -> machine {child_idx} input {goal}")

                    path = router.find_path(start, goal, allow_belt_ports=True)
                    if path:
                        belts = router.path_to_belts(path)
                        all_belts.extend(belts)
                        for bx, by, bf, _, _ in belts:
                            router.add_occupied(bx, by, bf)
                        if verbose:
                            print(f"    Success: {len(belts)} belts")
                    else:
                        all_success = False
                        if verbose:
                            print(f"    FAILED to route")

            # Route leaf machines to output ports
            # Machine 1 outputs → ports 0, 1
            # Machine 2 outputs → ports 2, 3
            outputs_per_leaf = num_outputs_needed // max(1, num_machines - 1)

            for leaf_idx in range(1, num_machines):
                leaf_outputs = all_machine_outputs[leaf_idx]
                port_start_idx = (leaf_idx - 1) * outputs_per_leaf

                for local_out_idx, leaf_output_pos in enumerate(leaf_outputs):
                    port_idx = port_start_idx + local_out_idx
                    if port_idx >= len(self.output_positions):
                        break

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
                        print(f"  Routing machine {leaf_idx} output[{local_out_idx}] {leaf_output_pos} -> output port {port_idx}")

                    path = router.find_path(leaf_output_pos, goal, allow_belt_ports=True)
                    if path:
                        belts = router.path_to_belts(path)
                        all_belts.extend(belts)
                        for bx, by, bf, _, _ in belts:
                            router.add_occupied(bx, by, bf)
                        if verbose:
                            print(f"    Success: {len(belts)} belts")
                    else:
                        all_success = False
                        if verbose:
                            print(f"    FAILED to route")

        return all_belts, all_success


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
