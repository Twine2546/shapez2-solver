"""
Layout Search - Phase 2 of Two-Phase Search

Given a valid system design (machines and their logical connections),
this module searches for the optimal physical layout on the foundation.

The search space is positions and rotations for each machine.
Belt routing is handled deterministically by the router.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import random

from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS, BuildingSpec
from .system_search import SystemDesign, MachineNode, InputPort, OutputPort
from .router import BeltRouter, Connection, RouteResult


@dataclass
class PlacedMachine:
    """A machine with a physical position."""
    node_id: int
    building_type: BuildingType
    x: int
    y: int
    floor: int
    rotation: Rotation


@dataclass
class LayoutCandidate:
    """A candidate layout solution."""
    machines: List[PlacedMachine]
    belts: List[Tuple[int, int, int, BuildingType, Rotation]]  # (x, y, floor, type, rotation)
    fitness: float = 0.0
    routing_success: bool = False

    def copy(self) -> 'LayoutCandidate':
        return LayoutCandidate(
            machines=[PlacedMachine(m.node_id, m.building_type, m.x, m.y, m.floor, m.rotation)
                     for m in self.machines],
            belts=list(self.belts),
            fitness=self.fitness,
            routing_success=self.routing_success
        )


class LayoutSearch:
    """
    Searches for optimal physical layout given a system design.

    Uses evolutionary search over machine positions and rotations.
    Belt routing is handled deterministically after each layout is generated.
    """

    def __init__(
        self,
        system_design: SystemDesign,
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        population_size: int = 50,
        mutation_rate: float = 0.3,
    ):
        self.system = system_design
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_floors = num_floors
        self.population_size = population_size
        self.mutation_rate = mutation_rate

        # Build port position map
        self._build_port_positions()

        self.population: List[LayoutCandidate] = []
        self.best_layout: Optional[LayoutCandidate] = None
        self.best_fitness: float = 0.0

    def _build_port_positions(self):
        """Build mapping from port IDs to physical positions."""
        self.input_positions: Dict[int, Tuple[int, int, int, Rotation]] = {}
        self.output_positions: Dict[int, Tuple[int, int, int, Rotation]] = {}

        # Calculate port positions based on side
        # Simplified: assume ports are evenly distributed along edges
        for port in self.system.input_ports:
            pos = self._get_port_position(port.side, port.position, port.floor, is_input=True)
            self.input_positions[port.port_id] = pos

        for port in self.system.output_ports:
            pos = self._get_port_position(port.side, port.position, port.floor, is_input=False)
            self.output_positions[port.port_id] = pos

    def _get_port_position(self, side: str, position: int, floor: int, is_input: bool) -> Tuple[int, int, int, Rotation]:
        """
        Get the grid position and entry direction for a port.

        Returns:
            Tuple of (x, y, floor, direction_into_foundation)
        """
        # Port position along the edge (0-3 for first unit, 4-7 for second, etc.)
        unit_idx = position // 4
        pos_in_unit = position % 4

        # Calculate position
        if side == 'N':
            x = unit_idx * 20 + 3 + pos_in_unit * 3
            y = 0
            direction = Rotation.SOUTH if is_input else Rotation.NORTH
        elif side == 'S':
            x = unit_idx * 20 + 3 + pos_in_unit * 3
            y = self.grid_height - 1
            direction = Rotation.NORTH if is_input else Rotation.SOUTH
        elif side == 'W':
            x = 0
            y = unit_idx * 20 + 3 + pos_in_unit * 3
            direction = Rotation.EAST if is_input else Rotation.WEST
        else:  # E
            x = self.grid_width - 1
            y = unit_idx * 20 + 3 + pos_in_unit * 3
            direction = Rotation.WEST if is_input else Rotation.EAST

        return (x, y, floor, direction)

    def _get_building_cells(self, machine: PlacedMachine) -> Set[Tuple[int, int, int]]:
        """Get all grid cells occupied by a building."""
        spec = BUILDING_SPECS.get(machine.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
        cells = set()

        for dx in range(spec.width):
            for dy in range(spec.height):
                for dz in range(spec.depth):
                    cells.add((machine.x + dx, machine.y + dy, machine.floor + dz))

        return cells

    def _check_collision(self, machine: PlacedMachine, occupied: Set[Tuple[int, int, int]]) -> bool:
        """Check if a machine collides with occupied cells."""
        cells = self._get_building_cells(machine)
        return bool(cells & occupied)

    def _is_valid_position(self, machine: PlacedMachine) -> bool:
        """Check if a machine position is within grid bounds."""
        spec = BUILDING_SPECS.get(machine.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

        if machine.x < 0 or machine.x + spec.width > self.grid_width:
            return False
        if machine.y < 0 or machine.y + spec.height > self.grid_height:
            return False
        if machine.floor < 0 or machine.floor + spec.depth > self.num_floors:
            return False

        return True

    def _create_random_layout(self) -> LayoutCandidate:
        """Create a random layout for all machines."""
        machines = []
        occupied: Set[Tuple[int, int, int]] = set()

        # Get preferred floor from input ports
        input_floors = [p.floor for p in self.system.input_ports]
        prefer_floor = input_floors[0] if input_floors else 0

        for node in self.system.machines:
            # Try to place this machine
            placed = None
            for _ in range(100):  # Max attempts
                x = random.randint(1, self.grid_width - 3)
                y = random.randint(1, self.grid_height - 3)
                floor = prefer_floor if random.random() < 0.8 else random.randint(0, self.num_floors - 1)
                rotation = random.choice(list(Rotation))

                candidate = PlacedMachine(
                    node_id=node.node_id,
                    building_type=node.building_type,
                    x=x, y=y, floor=floor,
                    rotation=rotation
                )

                if self._is_valid_position(candidate) and not self._check_collision(candidate, occupied):
                    placed = candidate
                    occupied.update(self._get_building_cells(candidate))
                    break

            if placed:
                machines.append(placed)

        layout = LayoutCandidate(machines=machines, belts=[])
        self._route_layout(layout)
        return layout

    def _route_layout(self, layout: LayoutCandidate):
        """Route belts for a layout using A* pathfinding."""
        router = BeltRouter(self.grid_width, self.grid_height, self.num_floors)

        # Mark machine cells as occupied
        occupied = set()
        machine_map: Dict[int, PlacedMachine] = {}

        for m in layout.machines:
            machine_map[m.node_id] = m
            occupied.update(self._get_building_cells(m))

        router.set_occupied(occupied)

        # Build list of connections to route
        connections: List[Connection] = []

        # Route from input ports to machines
        for port in self.system.input_ports:
            # Find which machine receives this input
            for node in self.system.machines:
                for input_idx in range(node.num_inputs()):
                    conn_key = (node.node_id, input_idx)
                    if conn_key in self.system.connections:
                        src_type, src_id, _ = self.system.connections[conn_key]
                        if src_type == 'input_port' and src_id == port.port_id:
                            # Route from input port to machine
                            if node.node_id in machine_map:
                                machine = machine_map[node.node_id]
                                port_pos = self.input_positions.get(port.port_id)
                                if port_pos:
                                    from_pos = (port_pos[0], port_pos[1], port_pos[2])
                                    to_pos = (machine.x, machine.y, machine.floor)
                                    connections.append(Connection(
                                        from_pos=from_pos,
                                        to_pos=to_pos,
                                        from_direction=port_pos[3],
                                        to_direction=machine.rotation,
                                        priority=2
                                    ))

        # Route between machines
        for node in self.system.machines:
            if node.node_id not in machine_map:
                continue

            for input_idx in range(node.num_inputs()):
                conn_key = (node.node_id, input_idx)
                if conn_key in self.system.connections:
                    src_type, src_id, src_out = self.system.connections[conn_key]
                    if src_type == 'machine' and src_id in machine_map:
                        # Route from source machine to this machine
                        src_machine = machine_map[src_id]
                        dst_machine = machine_map[node.node_id]

                        from_pos = (src_machine.x + 1, src_machine.y, src_machine.floor)
                        to_pos = (dst_machine.x - 1, dst_machine.y, dst_machine.floor)

                        connections.append(Connection(
                            from_pos=from_pos,
                            to_pos=to_pos,
                            from_direction=src_machine.rotation,
                            to_direction=dst_machine.rotation,
                            priority=1
                        ))

        # Route from machines to output ports
        for port in self.system.output_ports:
            if port.source is not None:
                src_id, src_out = port.source
                if src_id in machine_map:
                    machine = machine_map[src_id]
                    port_pos = self.output_positions.get(port.port_id)
                    if port_pos:
                        from_pos = (machine.x + 1, machine.y, machine.floor)
                        to_pos = (port_pos[0], port_pos[1], port_pos[2])
                        connections.append(Connection(
                            from_pos=from_pos,
                            to_pos=to_pos,
                            from_direction=machine.rotation,
                            to_direction=port_pos[3],
                            priority=3
                        ))

        # Route all connections
        results = router.route_all(connections)

        # Collect belts and check success
        all_belts = []
        num_success = 0
        for result in results:
            if result.success:
                all_belts.extend(result.belts)
                num_success += 1

        layout.belts = all_belts
        layout.routing_success = (num_success == len(connections)) if connections else True

    def _evaluate_layout(self, layout: LayoutCandidate) -> float:
        """Evaluate a layout's fitness."""
        fitness = 0.0

        # Base score for having all machines placed
        if len(layout.machines) == len(self.system.machines):
            fitness += 30.0
        else:
            fitness += 30.0 * len(layout.machines) / max(1, len(self.system.machines))

        # Bonus for successful routing
        if layout.routing_success:
            fitness += 40.0
        else:
            # Partial credit for some routes
            expected_routes = len(self.system.machines) + len(self.system.output_ports)
            if expected_routes > 0:
                actual_routes = len(layout.belts) // 5  # Rough estimate
                fitness += 40.0 * min(1.0, actual_routes / expected_routes)

        # Compactness bonus (prefer layouts that use less space)
        if layout.machines:
            min_x = min(m.x for m in layout.machines)
            max_x = max(m.x for m in layout.machines)
            min_y = min(m.y for m in layout.machines)
            max_y = max(m.y for m in layout.machines)
            spread = (max_x - min_x + 1) * (max_y - min_y + 1)
            max_spread = self.grid_width * self.grid_height
            compactness = 1.0 - (spread / max_spread)
            fitness += 15.0 * compactness

        # Belt efficiency (fewer belts is better)
        if layout.belts:
            belt_penalty = min(15.0, len(layout.belts) * 0.1)
            fitness += 15.0 - belt_penalty
        else:
            fitness += 15.0

        layout.fitness = fitness
        return fitness

    def _mutate(self, layout: LayoutCandidate) -> LayoutCandidate:
        """Mutate a layout."""
        mutated = layout.copy()

        if not mutated.machines:
            return mutated

        # Choose a random machine to mutate
        idx = random.randint(0, len(mutated.machines) - 1)
        machine = mutated.machines[idx]

        # Get occupied cells (excluding this machine)
        occupied = set()
        for i, m in enumerate(mutated.machines):
            if i != idx:
                occupied.update(self._get_building_cells(m))

        mutation_type = random.random()

        if mutation_type < 0.4:
            # Move machine
            for _ in range(50):
                new_x = machine.x + random.randint(-3, 3)
                new_y = machine.y + random.randint(-3, 3)
                new_m = PlacedMachine(
                    machine.node_id, machine.building_type,
                    new_x, new_y, machine.floor, machine.rotation
                )
                if self._is_valid_position(new_m) and not self._check_collision(new_m, occupied):
                    mutated.machines[idx] = new_m
                    break

        elif mutation_type < 0.7:
            # Change rotation
            new_rotation = random.choice(list(Rotation))
            mutated.machines[idx] = PlacedMachine(
                machine.node_id, machine.building_type,
                machine.x, machine.y, machine.floor, new_rotation
            )

        else:
            # Change floor
            for _ in range(10):
                new_floor = random.randint(0, self.num_floors - 1)
                new_m = PlacedMachine(
                    machine.node_id, machine.building_type,
                    machine.x, machine.y, new_floor, machine.rotation
                )
                if self._is_valid_position(new_m) and not self._check_collision(new_m, occupied):
                    mutated.machines[idx] = new_m
                    break

        # Re-route
        self._route_layout(mutated)
        return mutated

    def _crossover(self, parent1: LayoutCandidate, parent2: LayoutCandidate) -> LayoutCandidate:
        """Crossover two layouts."""
        # Take some machines from each parent
        child_machines = []
        occupied = set()

        for node in self.system.machines:
            # Find this machine in each parent
            p1_machine = next((m for m in parent1.machines if m.node_id == node.node_id), None)
            p2_machine = next((m for m in parent2.machines if m.node_id == node.node_id), None)

            # Choose one
            if p1_machine and p2_machine:
                chosen = p1_machine if random.random() < 0.5 else p2_machine
            elif p1_machine:
                chosen = p1_machine
            elif p2_machine:
                chosen = p2_machine
            else:
                continue

            # Check for collision
            candidate = PlacedMachine(
                chosen.node_id, chosen.building_type,
                chosen.x, chosen.y, chosen.floor, chosen.rotation
            )
            if not self._check_collision(candidate, occupied):
                child_machines.append(candidate)
                occupied.update(self._get_building_cells(candidate))

        child = LayoutCandidate(machines=child_machines, belts=[])
        self._route_layout(child)
        return child

    def run(self, max_generations: int = 100, verbose: bool = False) -> Optional[LayoutCandidate]:
        """
        Run the layout search.

        Args:
            max_generations: Maximum generations
            verbose: Print progress

        Returns:
            Best layout found
        """
        # Initialize population
        self.population = []
        for _ in range(self.population_size):
            layout = self._create_random_layout()
            self._evaluate_layout(layout)
            self.population.append(layout)

        for generation in range(max_generations):
            # Sort by fitness
            self.population.sort(key=lambda x: -x.fitness)

            # Update best
            if self.population[0].fitness > self.best_fitness:
                self.best_fitness = self.population[0].fitness
                self.best_layout = self.population[0].copy()

                if verbose:
                    print(f"Gen {generation}: fitness={self.best_fitness:.2f}, "
                          f"machines={len(self.best_layout.machines)}, "
                          f"belts={len(self.best_layout.belts)}, "
                          f"routing_ok={self.best_layout.routing_success}")

            # Early termination if perfect
            if self.best_fitness >= 99.0 and self.best_layout.routing_success:
                break

            # Select top half
            survivors = self.population[:self.population_size // 2]

            # Create next generation
            new_population = [s.copy() for s in survivors]

            while len(new_population) < self.population_size:
                if random.random() < 0.3 and len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    child = self._crossover(p1, p2)
                else:
                    parent = random.choice(survivors)
                    child = self._mutate(parent)

                self._evaluate_layout(child)
                new_population.append(child)

            self.population = new_population

        return self.best_layout


class LayoutSearchSA(LayoutSearch):
    """
    Simulated Annealing variant of layout search.

    Uses temperature-based acceptance to escape local optima.
    """

    def __init__(
        self,
        system_design: SystemDesign,
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        initial_temp: float = 50.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.1,
    ):
        super().__init__(system_design, grid_width, grid_height, num_floors, population_size=1)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def run(self, max_generations: int = 100, verbose: bool = False) -> Optional[LayoutCandidate]:
        """Run simulated annealing layout search."""
        import math

        # Start with random layout
        current = self._create_random_layout()
        current_fitness = self._evaluate_layout(current)

        self.best_layout = current.copy()
        self.best_fitness = current_fitness

        temperature = self.initial_temp

        for iteration in range(max_generations):
            # Generate neighbor
            neighbor = self._mutate(current)
            neighbor_fitness = self._evaluate_layout(neighbor)

            delta = neighbor_fitness - current_fitness

            if delta > 0:
                # Better - always accept
                current = neighbor
                current_fitness = neighbor_fitness
            elif temperature > self.min_temp:
                # Worse - accept with probability
                prob = math.exp(delta / temperature)
                if random.random() < prob:
                    current = neighbor
                    current_fitness = neighbor_fitness

            # Update best
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.best_layout = current.copy()

                if verbose:
                    print(f"Iter {iteration}: fitness={self.best_fitness:.2f}, "
                          f"machines={len(self.best_layout.machines)}, "
                          f"routing_ok={self.best_layout.routing_success}, "
                          f"temp={temperature:.2f}")

            # Early termination
            if self.best_fitness >= 99.0 and self.best_layout.routing_success:
                break

            temperature *= self.cooling_rate

        return self.best_layout


class LayoutSearchHybrid(LayoutSearch):
    """
    Hybrid algorithm combining SA exploration with EA refinement.
    """

    def __init__(
        self,
        system_design: SystemDesign,
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        population_size: int = 50,
        sa_fraction: float = 0.5,
    ):
        super().__init__(system_design, grid_width, grid_height, num_floors, population_size)
        self.sa_fraction = sa_fraction

    def run(self, max_generations: int = 100, verbose: bool = False) -> Optional[LayoutCandidate]:
        """Run hybrid layout search."""
        sa_iterations = int(max_generations * self.sa_fraction)
        ea_generations = max_generations - sa_iterations

        if verbose:
            print(f"Layout Phase 1: SA ({sa_iterations} iterations)")

        # Phase 1: SA exploration
        sa = LayoutSearchSA(
            system_design=self.system,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_floors=self.num_floors,
        )
        sa_result = sa.run(max_generations=sa_iterations, verbose=verbose)

        if sa.best_fitness >= 99.0 and sa.best_layout and sa.best_layout.routing_success:
            self.best_layout = sa.best_layout
            self.best_fitness = sa.best_fitness
            return self.best_layout

        if verbose:
            print(f"\nLayout Phase 2: EA ({ea_generations} generations)")

        # Phase 2: EA refinement
        self.population = []
        if sa_result:
            self.population.append(sa_result.copy())
            for _ in range(self.population_size // 3):
                mutated = self._mutate(sa_result)
                self._evaluate_layout(mutated)
                self.population.append(mutated)

        while len(self.population) < self.population_size:
            layout = self._create_random_layout()
            self._evaluate_layout(layout)
            self.population.append(layout)

        self.best_layout = sa.best_layout
        self.best_fitness = sa.best_fitness

        for generation in range(ea_generations):
            self.population.sort(key=lambda x: -x.fitness)

            if self.population[0].fitness > self.best_fitness:
                self.best_fitness = self.population[0].fitness
                self.best_layout = self.population[0].copy()

                if verbose:
                    print(f"EA Gen {generation}: fitness={self.best_fitness:.2f}, "
                          f"routing_ok={self.best_layout.routing_success}")

            if self.best_fitness >= 99.0 and self.best_layout.routing_success:
                break

            survivors = self.population[:self.population_size // 2]
            new_population = [s.copy() for s in survivors]

            while len(new_population) < self.population_size:
                if random.random() < 0.3 and len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    child = self._crossover(p1, p2)
                else:
                    parent = random.choice(survivors)
                    child = self._mutate(parent)
                self._evaluate_layout(child)
                new_population.append(child)

            self.population = new_population

        return self.best_layout


def layout_system_design(
    system: SystemDesign,
    grid_width: int,
    grid_height: int,
    num_floors: int = 4,
    max_generations: int = 100,
    algorithm: str = 'evolution',
    verbose: bool = False
) -> Optional[LayoutCandidate]:
    """
    Find an optimal layout for a system design.

    Args:
        system: The system design to layout
        grid_width: Foundation grid width
        grid_height: Foundation grid height
        num_floors: Number of floors
        max_generations: Search generations
        algorithm: 'evolution', 'sa', or 'hybrid'
        verbose: Print progress

    Returns:
        Best layout found
    """
    if algorithm == 'sa':
        search = LayoutSearchSA(
            system_design=system,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
        )
    elif algorithm == 'hybrid':
        search = LayoutSearchHybrid(
            system_design=system,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
        )
    else:
        search = LayoutSearch(
            system_design=system,
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
        )
    return search.run(max_generations=max_generations, verbose=verbose)
