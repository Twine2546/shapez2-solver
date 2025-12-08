"""
Optimization algorithms for foundation evolution.

Provides multiple algorithms:
- Evolutionary Algorithm (genetic algorithm)
- Simulated Annealing
"""

import random
import math
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from enum import Enum

from .foundation_config import FoundationConfig, Side
from ..blueprint.building_types import (
    BuildingType, Rotation, BUILDING_SPECS, BuildingSpec, BUILDING_PORTS
)

# Import PlacedBuilding and Candidate from foundation_evolution to ensure compatibility
# These are imported after the module is defined to avoid circular imports
# We'll define a lazy import function


class AlgorithmType(Enum):
    """Available optimization algorithms."""
    EVOLUTIONARY = "evolutionary"
    SIMULATED_ANNEALING = "simulated_annealing"
    HYBRID = "hybrid"  # SA for initial exploration, then EA for refinement


def _get_foundation_classes():
    """Lazy import to avoid circular dependency."""
    from .foundation_evolution import PlacedBuilding, Candidate, OPERATION_BUILDINGS
    return PlacedBuilding, Candidate, OPERATION_BUILDINGS


def _get_router():
    """Lazy import router to avoid circular dependency."""
    from .router import BeltRouter, Connection
    return BeltRouter, Connection


class BaseAlgorithm(ABC):
    """Base class for optimization algorithms."""

    def __init__(
        self,
        config: FoundationConfig,
        evaluate_fn: Callable,  # Candidate -> float
        valid_buildings: List[BuildingType],
        max_buildings: int = 20,
        use_routing: bool = True,
    ):
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.valid_buildings = valid_buildings
        self.max_buildings = max_buildings
        self.use_routing = use_routing
        self.best_solution = None
        self.history: List[Dict[str, Any]] = []
        self._occupied: Set[Tuple[int, int, int]] = set()

    @abstractmethod
    def run(self, iterations: int, verbose: bool = True):
        """Run the algorithm for given iterations."""
        pass

    def _create_random_candidate(self):
        """Create a random candidate solution with collision detection and routing."""
        PlacedBuilding, Candidate, OPERATION_BUILDINGS = _get_foundation_classes()
        buildings = []
        self._occupied = set()

        if self.use_routing:
            # Only place machines, then route belts
            valid_machines = [bt for bt in OPERATION_BUILDINGS if bt in self.valid_buildings]
            num_machines = random.randint(1, min(6, self.max_buildings // 2))

            for i in range(num_machines):
                building = self._create_random_machine(i, valid_machines)
                if building:
                    buildings.append(building)
                    self._mark_occupied(building)

            candidate = Candidate(buildings=buildings)
            self._route_candidate(candidate)
            return candidate
        else:
            # Legacy mode: random buildings including belts
            num_buildings = random.randint(3, min(self.max_buildings, 12))

            for i in range(num_buildings):
                building = self._create_random_building_avoiding_collisions(i)
                if building:
                    buildings.append(building)
                    self._mark_occupied(building)

            return Candidate(buildings=buildings)

    def _create_random_machine(self, building_id: int, valid_machines: List[BuildingType]):
        """Create a random machine (not belt) avoiding collisions."""
        PlacedBuilding, Candidate, _ = _get_foundation_classes()
        if not valid_machines:
            return None

        for _ in range(10):
            building_type = random.choice(valid_machines)
            spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

            max_x = max(0, self.config.spec.grid_width - spec.width)
            max_y = max(0, self.config.spec.grid_height - spec.height)
            max_floor = max(0, self.config.spec.num_floors - spec.depth)

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            floor = random.randint(0, max_floor)

            # Check for collisions
            collision = False
            for dx in range(spec.width):
                for dy in range(spec.height):
                    for df in range(spec.depth):
                        if (x + dx, y + dy, floor + df) in self._occupied:
                            collision = True
                            break
                    if collision:
                        break
                if collision:
                    break

            if not collision:
                return PlacedBuilding(
                    building_id=building_id,
                    building_type=building_type,
                    x=x, y=y, floor=floor,
                    rotation=random.choice(list(Rotation))
                )

        return None

    def _route_candidate(self, candidate) -> None:
        """Route belts to connect machines to ports using A* pathfinding."""
        PlacedBuilding, Candidate, OPERATION_BUILDINGS = _get_foundation_classes()
        BeltRouter, Connection = _get_router()

        # Build occupied set from existing buildings
        occupied = set()
        machines = []

        for b in candidate.buildings:
            spec = BUILDING_SPECS.get(b.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
            for dx in range(spec.width):
                for dy in range(spec.height):
                    for df in range(spec.depth):
                        occupied.add((b.x + dx, b.y + dy, b.floor + df))
            if b.building_type in OPERATION_BUILDINGS:
                machines.append(b)

        if not machines:
            return

        # Create router
        router = BeltRouter(
            self.config.spec.grid_width,
            self.config.spec.grid_height,
            self.config.spec.num_floors
        )
        router.set_occupied(occupied)

        # Route from input ports to nearest machine
        connections = []
        for side, pos, floor, _ in self.config.get_all_inputs():
            gx, gy = self.config.spec.get_port_grid_position(side, pos)

            if side == Side.NORTH:
                entry = (gx, 0, floor)
                entry_dir = Rotation.SOUTH
            elif side == Side.SOUTH:
                entry = (gx, self.config.spec.grid_height - 1, floor)
                entry_dir = Rotation.NORTH
            elif side == Side.WEST:
                entry = (0, gy, floor)
                entry_dir = Rotation.EAST
            elif side == Side.EAST:
                entry = (self.config.spec.grid_width - 1, gy, floor)
                entry_dir = Rotation.WEST
            else:
                continue

            # Find nearest machine
            best_machine = min(
                (m for m in machines if m.floor == floor),
                key=lambda m: abs(m.x - entry[0]) + abs(m.y - entry[1]),
                default=None
            )

            if best_machine:
                spec = BUILDING_SPECS.get(best_machine.building_type)
                if spec:
                    if best_machine.rotation == Rotation.EAST:
                        target = (best_machine.x - 1, best_machine.y + spec.height // 2, floor)
                    elif best_machine.rotation == Rotation.WEST:
                        target = (best_machine.x + spec.width, best_machine.y + spec.height // 2, floor)
                    elif best_machine.rotation == Rotation.SOUTH:
                        target = (best_machine.x + spec.width // 2, best_machine.y - 1, floor)
                    else:
                        target = (best_machine.x + spec.width // 2, best_machine.y + spec.height, floor)

                    connections.append(Connection(entry, target, entry_dir, best_machine.rotation, 2))

        # Route from machines to output ports
        for side, pos, floor, _ in self.config.get_all_outputs():
            gx, gy = self.config.spec.get_port_grid_position(side, pos)

            if side == Side.NORTH:
                exit_pos = (gx, 0, floor)
                exit_dir = Rotation.NORTH
            elif side == Side.SOUTH:
                exit_pos = (gx, self.config.spec.grid_height - 1, floor)
                exit_dir = Rotation.SOUTH
            elif side == Side.WEST:
                exit_pos = (0, gy, floor)
                exit_dir = Rotation.WEST
            elif side == Side.EAST:
                exit_pos = (self.config.spec.grid_width - 1, gy, floor)
                exit_dir = Rotation.EAST
            else:
                continue

            best_machine = min(
                (m for m in machines if m.floor == floor),
                key=lambda m: abs(m.x - exit_pos[0]) + abs(m.y - exit_pos[1]),
                default=None
            )

            if best_machine:
                spec = BUILDING_SPECS.get(best_machine.building_type)
                if spec:
                    if best_machine.rotation == Rotation.EAST:
                        source = (best_machine.x + spec.width, best_machine.y + spec.height // 2, floor)
                    elif best_machine.rotation == Rotation.WEST:
                        source = (best_machine.x - 1, best_machine.y + spec.height // 2, floor)
                    elif best_machine.rotation == Rotation.SOUTH:
                        source = (best_machine.x + spec.width // 2, best_machine.y + spec.height, floor)
                    else:
                        source = (best_machine.x + spec.width // 2, best_machine.y - 1, floor)

                    connections.append(Connection(source, exit_pos, best_machine.rotation, exit_dir, 1))

        # Route and add belts
        results = router.route_all(connections)
        building_id = len(candidate.buildings)

        for result in results:
            if result.success:
                for x, y, floor, belt_type, rotation in result.belts:
                    belt = PlacedBuilding(
                        building_id=building_id,
                        building_type=belt_type,
                        x=x, y=y, floor=floor,
                        rotation=rotation
                    )
                    candidate.buildings.append(belt)
                    building_id += 1

    def _create_random_building_avoiding_collisions(self, building_id: int):
        """Create a single random building avoiding collisions."""
        PlacedBuilding, Candidate, _ = _get_foundation_classes()
        if not self.valid_buildings:
            return None

        for _ in range(10):  # Try up to 10 times
            building_type = random.choice(self.valid_buildings)
            spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

            max_x = max(0, self.config.spec.grid_width - spec.width)
            max_y = max(0, self.config.spec.grid_height - spec.height)
            max_floor = max(0, self.config.spec.num_floors - spec.depth)

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            floor = random.randint(0, max_floor)

            # Check for collisions
            collision = False
            for dx in range(spec.width):
                for dy in range(spec.height):
                    for df in range(spec.depth):
                        if (x + dx, y + dy, floor + df) in self._occupied:
                            collision = True
                            break
                    if collision:
                        break
                if collision:
                    break

            if not collision:
                return PlacedBuilding(
                    building_id=building_id,
                    building_type=building_type,
                    x=x, y=y, floor=floor,
                    rotation=random.choice(list(Rotation))
                )

        return None

    def _mark_occupied(self, building) -> None:
        """Mark cells as occupied by a building."""
        spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
        for dx in range(spec.width):
            for dy in range(spec.height):
                for df in range(spec.depth):
                    self._occupied.add((building.x + dx, building.y + dy, building.floor + df))

    def _create_seeded_candidate(self):
        """Create a candidate with machines and routed belts."""
        PlacedBuilding, Candidate, OPERATION_BUILDINGS = _get_foundation_classes()
        buildings = []
        building_id = 0
        self._occupied = set()

        if self.use_routing:
            # Only place machines, routing will add belts
            valid_machines = [bt for bt in OPERATION_BUILDINGS if bt in self.valid_buildings]
            num_machines = random.randint(1, min(4, len(valid_machines)))

            for _ in range(num_machines):
                building = self._create_random_machine(building_id, valid_machines)
                if building:
                    buildings.append(building)
                    self._mark_occupied(building)
                    building_id += 1

            candidate = Candidate(buildings=buildings)
            self._route_candidate(candidate)
            return candidate
        else:
            # Legacy mode: place belts at port positions
            for side, pos, floor, shape_code in self.config.get_all_inputs():
                gx, gy = self.config.spec.get_port_grid_position(side, pos)
                belt = self._create_port_belt(building_id, side, gx, gy, floor, is_input=True)
                if belt and (belt.x, belt.y, belt.floor) not in self._occupied:
                    buildings.append(belt)
                    self._occupied.add((belt.x, belt.y, belt.floor))
                    building_id += 1

            for side, pos, floor, shape_code in self.config.get_all_outputs():
                gx, gy = self.config.spec.get_port_grid_position(side, pos)
                belt = self._create_port_belt(building_id, side, gx, gy, floor, is_input=False)
                if belt and (belt.x, belt.y, belt.floor) not in self._occupied:
                    buildings.append(belt)
                    self._occupied.add((belt.x, belt.y, belt.floor))
                    building_id += 1

            num_extra = random.randint(2, min(self.max_buildings - len(buildings), 8))
            for _ in range(num_extra):
                building = self._create_random_building_avoiding_collisions(building_id)
                if building:
                    buildings.append(building)
                self._mark_occupied(building)
                building_id += 1

        return Candidate(buildings=buildings)

    def _create_port_belt(self, building_id: int, side: Side, gx: int, gy: int, floor: int, is_input: bool):
        """Create a belt at a port position."""
        PlacedBuilding, Candidate, _ = _get_foundation_classes()

        # Belt should be at the edge of the foundation
        if side == Side.NORTH:
            x, y = gx, 0
            rotation = Rotation.SOUTH if is_input else Rotation.NORTH
        elif side == Side.SOUTH:
            x, y = gx, self.config.spec.grid_height - 1
            rotation = Rotation.NORTH if is_input else Rotation.SOUTH
        elif side == Side.WEST:
            x, y = 0, gy
            rotation = Rotation.EAST if is_input else Rotation.WEST
        elif side == Side.EAST:
            x, y = self.config.spec.grid_width - 1, gy
            rotation = Rotation.WEST if is_input else Rotation.EAST
        else:
            return None

        return PlacedBuilding(
            building_id=building_id,
            building_type=BuildingType.BELT_FORWARD,
            x=x, y=y, floor=floor,
            rotation=rotation
        )


class EvolutionaryAlgorithm(BaseAlgorithm):
    """Genetic algorithm for optimization."""

    def __init__(
        self,
        config: FoundationConfig,
        evaluate_fn: Callable,
        valid_buildings: List[BuildingType],
        max_buildings: int = 20,
        population_size: int = 50,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elitism: int = 5,
        use_routing: bool = True,
    ):
        super().__init__(config, evaluate_fn, valid_buildings, max_buildings, use_routing)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.population = []

    def run(self, iterations: int, verbose: bool = True):
        """Run evolutionary algorithm."""
        # Initialize population - mix of seeded and random
        self.population = []
        for i in range(self.population_size):
            if i < self.population_size // 3:
                # Seed some with port belts
                candidate = self._create_seeded_candidate()
            else:
                candidate = self._create_random_candidate()
            candidate.fitness = self.evaluate_fn(candidate)
            self.population.append(candidate)

        self.population.sort(key=lambda c: c.fitness, reverse=True)
        self.best_solution = self.population[0].copy()

        for gen in range(iterations):
            # Evolve
            new_population = []

            # Elitism
            for i in range(min(self.elitism, len(self.population))):
                new_population.append(self.population[i].copy())

            # Fill rest with offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1 = self._tournament_select()
                    p2 = self._tournament_select()
                    child = self._crossover(p1, p2)
                else:
                    child = self._tournament_select().copy()

                if random.random() < self.mutation_rate:
                    self._mutate(child)

                child.fitness = self.evaluate_fn(child)
                new_population.append(child)

            self.population = new_population
            self.population.sort(key=lambda c: c.fitness, reverse=True)

            if self.population[0].fitness > self.best_solution.fitness:
                self.best_solution = self.population[0].copy()

            self.history.append({
                'generation': gen,
                'best_fitness': self.best_solution.fitness,
                'avg_fitness': sum(c.fitness for c in self.population) / len(self.population),
            })

            if verbose and (gen % 10 == 0 or gen == iterations - 1):
                print(f"Gen {gen}: Best={self.best_solution.fitness:.2f}, "
                      f"Avg={self.history[-1]['avg_fitness']:.2f}")

        return self.best_solution

    def _tournament_select(self, size: int = 3):
        """Tournament selection."""
        tournament = random.sample(self.population, min(size, len(self.population)))
        return max(tournament, key=lambda c: c.fitness)

    def _crossover(self, p1, p2):
        """Crossover two parents."""
        PlacedBuilding, Candidate, _ = _get_foundation_classes()
        # Take buildings from both parents
        split = random.randint(0, len(p1.buildings))
        buildings = [copy.copy(b) for b in p1.buildings[:split]]

        for b in p2.buildings[split:]:
            new_b = copy.copy(b)
            new_b.building_id = len(buildings)
            buildings.append(new_b)

        return Candidate(buildings=buildings)

    def _mutate(self, candidate) -> None:
        """Mutate a candidate with collision detection."""
        # Build occupied set from existing buildings
        self._occupied = set()
        for b in candidate.buildings:
            self._mark_occupied(b)

        mutation_type = random.choice(['add', 'remove', 'modify', 'add_port_belt', 'add_belt'])

        if mutation_type == 'add' and len(candidate.buildings) < self.max_buildings:
            building = self._create_random_building_avoiding_collisions(len(candidate.buildings))
            if building:
                candidate.buildings.append(building)

        elif mutation_type == 'add_belt' and len(candidate.buildings) < self.max_buildings:
            # Add a connecting belt
            self._add_connecting_belt(candidate)

        elif mutation_type == 'remove' and len(candidate.buildings) > 1:
            idx = random.randint(0, len(candidate.buildings) - 1)
            candidate.buildings.pop(idx)
            # Renumber
            for i, b in enumerate(candidate.buildings):
                b.building_id = i

        elif mutation_type == 'modify' and candidate.buildings:
            building = random.choice(candidate.buildings)
            mod = random.choice(['position', 'rotation', 'type'])

            if mod == 'position':
                spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
                building.x = random.randint(0, max(0, self.config.spec.grid_width - spec.width))
                building.y = random.randint(0, max(0, self.config.spec.grid_height - spec.height))
            elif mod == 'rotation':
                building.rotation = random.choice(list(Rotation))
            else:
                building.building_type = random.choice(self.valid_buildings)

        elif mutation_type == 'add_port_belt' and len(candidate.buildings) < self.max_buildings:
            # Add a belt at a random port position
            all_ports = list(self.config.get_all_inputs()) + list(self.config.get_all_outputs())
            if all_ports:
                side, pos, floor, _ = random.choice(all_ports)
                gx, gy = self.config.spec.get_port_grid_position(side, pos)
                is_input = any(p[0] == side and p[1] == pos and p[2] == floor
                              for p in self.config.get_all_inputs())
                belt = self._create_port_belt(len(candidate.buildings), side, gx, gy, floor, is_input)
                if belt and (belt.x, belt.y, belt.floor) not in self._occupied:
                    candidate.buildings.append(belt)

    def _add_connecting_belt(self, candidate) -> None:
        """Add a belt to connect two buildings."""
        PlacedBuilding, Candidate, _ = _get_foundation_classes()
        if len(candidate.buildings) < 2:
            return

        b1 = random.choice(candidate.buildings)
        b2 = random.choice(candidate.buildings)
        if b1.building_id == b2.building_id or b1.floor != b2.floor:
            return

        # Place a belt between them
        mid_x = (b1.x + b2.x) // 2
        mid_y = (b1.y + b2.y) // 2

        if (mid_x, mid_y, b1.floor) not in self._occupied:
            dx = b2.x - b1.x
            dy = b2.y - b1.y
            if abs(dx) > abs(dy):
                rotation = Rotation.EAST if dx > 0 else Rotation.WEST
            else:
                rotation = Rotation.SOUTH if dy > 0 else Rotation.NORTH

            belt = PlacedBuilding(
                building_id=len(candidate.buildings),
                building_type=BuildingType.BELT_FORWARD,
                x=mid_x, y=mid_y, floor=b1.floor,
                rotation=rotation
            )
            candidate.buildings.append(belt)
            self._occupied.add((mid_x, mid_y, b1.floor))


class SimulatedAnnealing(BaseAlgorithm):
    """Simulated annealing algorithm for optimization."""

    def __init__(
        self,
        config: FoundationConfig,
        evaluate_fn: Callable,
        valid_buildings: List[BuildingType],
        max_buildings: int = 20,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.1,
        use_routing: bool = True,
    ):
        super().__init__(config, evaluate_fn, valid_buildings, max_buildings, use_routing)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def run(self, iterations: int, verbose: bool = True):
        """Run simulated annealing."""
        # Start with a seeded candidate
        current = self._create_seeded_candidate()
        current.fitness = self.evaluate_fn(current)
        self.best_solution = current.copy()

        temperature = self.initial_temp

        for i in range(iterations):
            # Generate neighbor
            neighbor = self._get_neighbor(current)
            neighbor.fitness = self.evaluate_fn(neighbor)

            # Calculate acceptance probability
            delta = neighbor.fitness - current.fitness

            if delta > 0:
                # Better solution, always accept
                current = neighbor
            else:
                # Worse solution, accept with probability
                prob = math.exp(delta / temperature)
                if random.random() < prob:
                    current = neighbor

            # Update best
            if current.fitness > self.best_solution.fitness:
                self.best_solution = current.copy()

            # Cool down
            temperature = max(self.min_temp, temperature * self.cooling_rate)

            self.history.append({
                'iteration': i,
                'temperature': temperature,
                'current_fitness': current.fitness,
                'best_fitness': self.best_solution.fitness,
            })

            if verbose and (i % 50 == 0 or i == iterations - 1):
                print(f"Iter {i}: Temp={temperature:.2f}, Current={current.fitness:.2f}, "
                      f"Best={self.best_solution.fitness:.2f}")

        return self.best_solution

    def _get_neighbor(self, candidate):
        """Generate a neighboring solution with collision checking."""
        PlacedBuilding, Candidate, _ = _get_foundation_classes()
        neighbor = candidate.copy()

        # Build occupied set
        self._occupied = set()
        for b in neighbor.buildings:
            self._mark_occupied(b)

        # Choose a random modification
        moves = ['add', 'remove', 'move', 'rotate', 'change_type', 'add_port_belt', 'add_belt']
        move = random.choice(moves)

        if move == 'add' and len(neighbor.buildings) < self.max_buildings:
            building = self._create_random_building_avoiding_collisions(len(neighbor.buildings))
            if building:
                neighbor.buildings.append(building)

        elif move == 'add_belt' and len(neighbor.buildings) < self.max_buildings:
            # Add a connecting belt
            if len(neighbor.buildings) >= 2:
                b1 = random.choice(neighbor.buildings)
                b2 = random.choice(neighbor.buildings)
                if b1.building_id != b2.building_id and b1.floor == b2.floor:
                    mid_x = (b1.x + b2.x) // 2
                    mid_y = (b1.y + b2.y) // 2
                    if (mid_x, mid_y, b1.floor) not in self._occupied:
                        dx = b2.x - b1.x
                        dy = b2.y - b1.y
                        rotation = Rotation.EAST if abs(dx) > abs(dy) else Rotation.SOUTH
                        belt = PlacedBuilding(
                            building_id=len(neighbor.buildings),
                            building_type=BuildingType.BELT_FORWARD,
                            x=mid_x, y=mid_y, floor=b1.floor,
                            rotation=rotation
                        )
                        neighbor.buildings.append(belt)

        elif move == 'remove' and len(neighbor.buildings) > 1:
            idx = random.randint(0, len(neighbor.buildings) - 1)
            neighbor.buildings.pop(idx)
            for i, b in enumerate(neighbor.buildings):
                b.building_id = i

        elif move == 'move' and neighbor.buildings:
            building = random.choice(neighbor.buildings)
            spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
            # Small random move
            building.x = max(0, min(self.config.spec.grid_width - spec.width,
                                    building.x + random.randint(-3, 3)))
            building.y = max(0, min(self.config.spec.grid_height - spec.height,
                                    building.y + random.randint(-3, 3)))

        elif move == 'rotate' and neighbor.buildings:
            building = random.choice(neighbor.buildings)
            building.rotation = random.choice(list(Rotation))

        elif move == 'change_type' and neighbor.buildings:
            building = random.choice(neighbor.buildings)
            building.building_type = random.choice(self.valid_buildings)

        elif move == 'add_port_belt' and len(neighbor.buildings) < self.max_buildings:
            all_ports = list(self.config.get_all_inputs()) + list(self.config.get_all_outputs())
            if all_ports:
                side, pos, floor, _ = random.choice(all_ports)
                gx, gy = self.config.spec.get_port_grid_position(side, pos)
                is_input = any(p[0] == side and p[1] == pos and p[2] == floor
                              for p in self.config.get_all_inputs())
                belt = self._create_port_belt(len(neighbor.buildings), side, gx, gy, floor, is_input)
                if belt and (belt.x, belt.y, belt.floor) not in self._occupied:
                    neighbor.buildings.append(belt)

        return neighbor


class HybridAlgorithm(BaseAlgorithm):
    """Hybrid algorithm: SA for exploration, then EA for refinement."""

    def __init__(
        self,
        config: FoundationConfig,
        evaluate_fn: Callable,
        valid_buildings: List[BuildingType],
        max_buildings: int = 20,
        sa_iterations: int = 200,
        ea_iterations: int = 100,
        use_routing: bool = True,
    ):
        super().__init__(config, evaluate_fn, valid_buildings, max_buildings, use_routing)
        self.sa_iterations = sa_iterations
        self.ea_iterations = ea_iterations

    def run(self, iterations: int, verbose: bool = True):
        """Run hybrid algorithm."""
        # Phase 1: Simulated Annealing for exploration
        if verbose:
            print("Phase 1: Simulated Annealing exploration...")

        sa = SimulatedAnnealing(
            self.config, self.evaluate_fn, self.valid_buildings,
            self.max_buildings, initial_temp=100.0, cooling_rate=0.99
        )
        sa_result = sa.run(self.sa_iterations, verbose=verbose)

        # Phase 2: Evolutionary refinement starting from SA result
        if verbose:
            print("\nPhase 2: Evolutionary refinement...")

        ea = EvolutionaryAlgorithm(
            self.config, self.evaluate_fn, self.valid_buildings,
            self.max_buildings, population_size=30
        )
        # Seed EA population with SA result
        ea.population = [sa_result.copy() for _ in range(10)]
        for _ in range(20):
            candidate = ea._create_seeded_candidate()
            candidate.fitness = self.evaluate_fn(candidate)
            ea.population.append(candidate)

        ea_result = ea.run(self.ea_iterations, verbose=verbose)

        self.best_solution = ea_result if ea_result.fitness > sa_result.fitness else sa_result
        self.history = sa.history + ea.history

        return self.best_solution


def create_algorithm(
    algorithm_type: AlgorithmType,
    config: FoundationConfig,
    evaluate_fn: Callable,
    valid_buildings: List[BuildingType],
    **kwargs
) -> BaseAlgorithm:
    """Factory function to create an algorithm instance."""
    if algorithm_type == AlgorithmType.EVOLUTIONARY:
        return EvolutionaryAlgorithm(config, evaluate_fn, valid_buildings, **kwargs)
    elif algorithm_type == AlgorithmType.SIMULATED_ANNEALING:
        return SimulatedAnnealing(config, evaluate_fn, valid_buildings, **kwargs)
    elif algorithm_type == AlgorithmType.HYBRID:
        return HybridAlgorithm(config, evaluate_fn, valid_buildings, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
