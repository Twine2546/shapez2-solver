"""Foundation-aware evolution system for Shapez 2.

Evolves building layouts on foundations with configurable inputs/outputs per side/floor.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum, auto

from .foundation_config import (
    FoundationConfig, FoundationSpec, Side, PortType, PortConfig,
    FOUNDATION_SPECS
)
from ..blueprint.building_types import (
    BuildingType, Rotation, BUILDING_SPECS, BuildingSpec, BUILDING_PORTS
)
from ..blueprint.encoder import BlueprintEncoder
from ..shapes.shape import Shape
from ..shapes.parser import ShapeCodeParser
from ..simulation.grid_simulator import GridSimulator


class CellType(Enum):
    """Type of cell in the grid."""
    EMPTY = auto()
    BUILDING = auto()
    BELT = auto()
    INPUT_PORT = auto()
    OUTPUT_PORT = auto()


@dataclass
class GridCell:
    """A cell in the evolution grid."""
    cell_type: CellType = CellType.EMPTY
    building_type: Optional[BuildingType] = None
    rotation: Rotation = Rotation.EAST
    building_id: Optional[int] = None  # For multi-cell buildings


@dataclass
class PlacedBuilding:
    """A building placed on the grid."""
    building_id: int
    building_type: BuildingType
    x: int
    y: int
    floor: int
    rotation: Rotation
    # For belt ports, channel_id links sender/receiver pairs
    channel_id: Optional[int] = None


@dataclass
class Connection:
    """A logical connection between buildings."""
    from_building_id: int
    from_output_idx: int
    to_building_id: int
    to_input_idx: int


@dataclass
class Candidate:
    """A candidate solution in the evolution."""
    buildings: List[PlacedBuilding] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    fitness: float = 0.0
    output_shapes: Dict[Tuple[Side, int, int], Optional[Shape]] = field(default_factory=dict)

    def copy(self) -> "Candidate":
        """Create a deep copy."""
        new = Candidate()
        new.buildings = [copy.copy(b) for b in self.buildings]
        new.connections = [copy.copy(c) for c in self.connections]
        new.fitness = self.fitness
        new.output_shapes = dict(self.output_shapes)
        return new


# Operation buildings for evolution
OPERATION_BUILDINGS = [
    BuildingType.ROTATOR_CW,
    BuildingType.ROTATOR_CCW,
    BuildingType.ROTATOR_180,
    BuildingType.CUTTER,
    BuildingType.CUTTER_MIRRORED,
    BuildingType.HALF_CUTTER,
    BuildingType.SWAPPER,
    BuildingType.STACKER,
    BuildingType.UNSTACKER,
    BuildingType.PIN_PUSHER,
]

# Belt types for routing
BELT_TYPES = [
    BuildingType.BELT_FORWARD,
    BuildingType.BELT_LEFT,
    BuildingType.BELT_RIGHT,
    BuildingType.LIFT_UP,
    BuildingType.LIFT_DOWN,
]

# All buildings that can be evolved (operations + belts)
EVOLVABLE_BUILDINGS = OPERATION_BUILDINGS + BELT_TYPES


class FoundationEvolution:
    """
    Evolution system that works with foundation configurations.

    Key features:
    - Configurable foundation size and port layout
    - Multi-floor support
    - Tracks top N solutions
    - Visualizes each floor
    - Supports belt ports (teleporters) with channel pairing
    """

    def __init__(
        self,
        config: FoundationConfig,
        population_size: int = 50,
        max_buildings: int = 20,
        num_top_solutions: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
    ):
        """
        Initialize the evolution system.

        Args:
            config: Foundation configuration with inputs/outputs
            population_size: Number of candidates per generation
            max_buildings: Maximum buildings per candidate
            num_top_solutions: Number of top solutions to track
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.config = config
        self.population_size = population_size
        self.max_buildings = max_buildings
        self.num_top_solutions = num_top_solutions
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.parser = ShapeCodeParser()
        self.population: List[Candidate] = []
        self.top_solutions: List[Candidate] = []
        self.generation = 0

        # Grid dimensions (foundation internal grid + margins for belts)
        self.grid_margin = 2
        self.grid_width = config.spec.grid_width + 2 * self.grid_margin
        self.grid_height = config.spec.grid_height + 2 * self.grid_margin
        self.num_floors = config.spec.num_floors

        # Parse expected shapes
        self.input_shapes: Dict[Tuple[Side, int, int], Shape] = {}
        self.expected_outputs: Dict[Tuple[Side, int, int], Shape] = {}
        self._parse_port_shapes()

    def _parse_port_shapes(self) -> None:
        """Parse shape codes from configuration."""
        for side, pos, floor, shape_code in self.config.get_all_inputs():
            if shape_code:
                try:
                    self.input_shapes[(side, pos, floor)] = self.parser.parse(shape_code)
                except Exception as e:
                    print(f"Warning: Could not parse input shape '{shape_code}': {e}")

        for side, pos, floor, shape_code in self.config.get_all_outputs():
            if shape_code:
                try:
                    self.expected_outputs[(side, pos, floor)] = self.parser.parse(shape_code)
                except Exception as e:
                    print(f"Warning: Could not parse output shape '{shape_code}': {e}")

    def initialize_population(self) -> None:
        """Initialize the population with random candidates."""
        self.population = []
        for _ in range(self.population_size):
            candidate = self._create_random_candidate()
            self.population.append(candidate)

    def _create_random_candidate(self) -> Candidate:
        """Create a random candidate solution."""
        candidate = Candidate()

        # Filter buildings that fit on this foundation (use internal grid dimensions)
        valid_buildings = []
        for bt in EVOLVABLE_BUILDINGS:
            spec = BUILDING_SPECS.get(bt, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
            if (spec.width <= self.config.spec.grid_width and
                spec.height <= self.config.spec.grid_height and
                spec.depth <= self.num_floors):
                valid_buildings.append(bt)

        if not valid_buildings:
            return candidate  # No buildings fit

        # Add random number of buildings
        num_buildings = random.randint(1, min(self.max_buildings, 10))
        building_id = 0

        for _ in range(num_buildings):
            # Choose random building type that fits
            building_type = random.choice(valid_buildings)
            spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

            # Choose random position within foundation (internal grid)
            max_x = max(0, self.config.spec.grid_width - spec.width)
            max_y = max(0, self.config.spec.grid_height - spec.height)
            max_floor = max(0, self.num_floors - spec.depth)

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            floor = random.randint(0, max_floor)
            rotation = random.choice(list(Rotation))

            building = PlacedBuilding(
                building_id=building_id,
                building_type=building_type,
                x=x, y=y, floor=floor,
                rotation=rotation
            )
            candidate.buildings.append(building)
            building_id += 1

        # Create random connections
        self._create_random_connections(candidate)

        return candidate

    def _create_random_connections(self, candidate: Candidate) -> None:
        """Create random connections between buildings."""
        if len(candidate.buildings) < 2:
            return

        # Create some random connections
        num_connections = random.randint(1, len(candidate.buildings))
        for _ in range(num_connections):
            b1 = random.choice(candidate.buildings)
            b2 = random.choice(candidate.buildings)
            if b1.building_id != b2.building_id:
                spec1 = BUILDING_SPECS.get(b1.building_type)
                spec2 = BUILDING_SPECS.get(b2.building_type)
                if spec1 and spec2 and spec1.num_outputs > 0 and spec2.num_inputs > 0:
                    conn = Connection(
                        from_building_id=b1.building_id,
                        from_output_idx=random.randint(0, spec1.num_outputs - 1),
                        to_building_id=b2.building_id,
                        to_input_idx=random.randint(0, spec2.num_inputs - 1)
                    )
                    candidate.connections.append(conn)

    def evaluate_fitness(self, candidate: Candidate) -> float:
        """
        Evaluate the fitness of a candidate.

        Fitness is based on:
        1. How well outputs match expected shapes
        2. Connectivity (are all buildings connected?)
        3. Efficiency (fewer buildings is better)
        """
        # Simulate the candidate
        output_shapes = self._simulate(candidate)
        candidate.output_shapes = output_shapes

        fitness = 0.0

        # Score based on output matching
        for port_key, expected in self.expected_outputs.items():
            actual = output_shapes.get(port_key)
            if actual is not None:
                match_score = self._compare_shapes(expected, actual)
                fitness += match_score * 100  # Weight output matching heavily
            else:
                fitness -= 10  # Penalty for missing output

        # Small bonus for efficiency
        if len(candidate.buildings) > 0:
            efficiency_bonus = 10.0 / len(candidate.buildings)
            fitness += efficiency_bonus

        # Bonus for connectivity
        connected_ratio = self._calculate_connectivity(candidate)
        fitness += connected_ratio * 20

        candidate.fitness = max(0, fitness)
        return candidate.fitness

    def _simulate(self, candidate: Candidate) -> Dict[Tuple[Side, int, int], Optional[Shape]]:
        """Simulate the candidate and return output shapes.

        This creates a GridSimulator from the candidate's buildings and
        attempts to route shapes from inputs to outputs.
        """
        # Create simulator with foundation internal grid + margins for ports
        margin = 2  # Space for input/output belts
        sim = GridSimulator(
            width=self.config.spec.grid_width + 2 * margin,
            height=self.config.spec.grid_height + 2 * margin,
            num_floors=self.num_floors
        )

        # Track port-to-grid-position mappings
        input_grid_positions: Dict[Tuple[Side, int, int], Tuple[int, int, int]] = {}
        output_grid_positions: Dict[Tuple[Side, int, int], Tuple[int, int, int]] = {}

        # Calculate grid positions for ports and add input belts
        for side, pos, floor, shape_code in self.config.get_all_inputs():
            grid_pos = self._port_to_grid_pos(side, pos, floor, margin, is_input=True)
            input_grid_positions[(side, pos, floor)] = grid_pos

            # Add a belt at the input position pointing into the foundation
            belt_rotation = self._get_inward_rotation(side)
            sim.add_belt(grid_pos[0], grid_pos[1], grid_pos[2],
                        BuildingType.BELT_FORWARD, belt_rotation)

            # Set input shape
            input_shape = self.input_shapes.get((side, pos, floor))
            if input_shape:
                # Input position is one cell outside the belt
                in_dir = self._get_outward_direction(side)
                input_pos = (grid_pos[0] + in_dir[0], grid_pos[1] + in_dir[1], grid_pos[2])
                sim.set_input(input_pos[0], input_pos[1], input_pos[2], input_shape)

        # Calculate grid positions for output ports
        for side, pos, floor, shape_code in self.config.get_all_outputs():
            grid_pos = self._port_to_grid_pos(side, pos, floor, margin, is_input=False)
            output_grid_positions[(side, pos, floor)] = grid_pos
            sim.set_output(grid_pos[0], grid_pos[1], grid_pos[2])

        # Add buildings from candidate
        building_id_offset = 100  # Avoid conflicts with port belts
        for building in candidate.buildings:
            # Offset building positions to account for margin
            grid_x = building.x + margin
            grid_y = building.y + margin

            if building.building_type in [BuildingType.BELT_FORWARD,
                                          BuildingType.BELT_LEFT,
                                          BuildingType.BELT_RIGHT]:
                sim.add_belt(grid_x, grid_y, building.floor,
                            building.building_type, building.rotation)
            elif building.building_type in [BuildingType.LIFT_UP, BuildingType.LIFT_DOWN]:
                sim.add_lift(grid_x, grid_y, building.floor,
                            building.building_type, building.rotation)
            else:
                sim.add_building(building_id_offset + building.building_id,
                               building.building_type,
                               grid_x, grid_y, building.floor,
                               building.rotation)

        # Route belts between connected buildings
        self._route_connections(sim, candidate, margin)

        # Build expected outputs for simulation
        expected = {}
        for port_key, shape in self.expected_outputs.items():
            grid_pos = output_grid_positions.get(port_key)
            if grid_pos:
                expected[grid_pos] = shape

        # Run simulation
        result = sim.simulate(max_steps=50, expected_outputs=expected)

        # Map grid outputs back to port keys
        output_shapes: Dict[Tuple[Side, int, int], Optional[Shape]] = {}
        for port_key, grid_pos in output_grid_positions.items():
            output_shapes[port_key] = result.output_shapes.get(grid_pos)

        return output_shapes

    def _port_to_grid_pos(self, side: Side, pos: int, floor: int,
                          margin: int, is_input: bool) -> Tuple[int, int, int]:
        """Convert a port specification to simulation grid coordinates.

        Ports are centered on each 1x1 unit. Each unit has 4 ports.
        The simulation grid has a margin around the foundation.
        """
        # Get the position within the foundation's internal grid
        internal_x, internal_y = self.config.spec.get_port_grid_position(side, pos)

        # For input belts: position one cell outside the foundation
        # For output positions: position at the edge of the foundation
        if side == Side.NORTH:
            x = internal_x + margin
            y = margin - 1 if is_input else margin
        elif side == Side.SOUTH:
            x = internal_x + margin
            y = self.config.spec.grid_height + margin if is_input else self.config.spec.grid_height + margin - 1
        elif side == Side.WEST:
            x = margin - 1 if is_input else margin
            y = internal_y + margin
        elif side == Side.EAST:
            x = self.config.spec.grid_width + margin if is_input else self.config.spec.grid_width + margin - 1
            y = internal_y + margin
        else:
            x, y = 0, 0

        return (x, y, floor)

    def _get_inward_rotation(self, side: Side) -> Rotation:
        """Get the rotation for a belt pointing inward from a side."""
        return {
            Side.NORTH: Rotation.SOUTH,
            Side.SOUTH: Rotation.NORTH,
            Side.WEST: Rotation.EAST,
            Side.EAST: Rotation.WEST,
        }[side]

    def _get_outward_direction(self, side: Side) -> Tuple[int, int]:
        """Get the (dx, dy) direction pointing outward from a side."""
        return {
            Side.NORTH: (0, -1),
            Side.SOUTH: (0, 1),
            Side.WEST: (-1, 0),
            Side.EAST: (1, 0),
        }[side]

    def _route_connections(self, sim: GridSimulator, candidate: Candidate, margin: int) -> None:
        """Attempt to route belt connections between buildings.

        This is a simplified routing that creates direct paths where possible.
        """
        building_map = {b.building_id: b for b in candidate.buildings}

        for conn in candidate.connections:
            from_b = building_map.get(conn.from_building_id)
            to_b = building_map.get(conn.to_building_id)

            if not from_b or not to_b:
                continue

            # Get output position of from_building
            from_spec = BUILDING_SPECS.get(from_b.building_type)
            to_spec = BUILDING_SPECS.get(to_b.building_type)

            if not from_spec or not to_spec:
                continue

            # Simple direct routing: try to place belts between buildings
            from_x = from_b.x + margin
            from_y = from_b.y + margin
            to_x = to_b.x + margin
            to_y = to_b.y + margin

            # Only route if buildings are adjacent or close
            dx = to_x - from_x
            dy = to_y - from_y

            if abs(dx) <= 2 and abs(dy) <= 2 and from_b.floor == to_b.floor:
                # Try to place connecting belts
                if dx != 0:
                    step_x = 1 if dx > 0 else -1
                    for x in range(from_x + step_x, to_x, step_x):
                        rotation = Rotation.EAST if dx > 0 else Rotation.WEST
                        sim.add_belt(x, from_y, from_b.floor,
                                    BuildingType.BELT_FORWARD, rotation)
                if dy != 0:
                    step_y = 1 if dy > 0 else -1
                    for y in range(from_y + step_y, to_y, step_y):
                        rotation = Rotation.SOUTH if dy > 0 else Rotation.NORTH
                        sim.add_belt(to_x if dx != 0 else from_x, y, from_b.floor,
                                    BuildingType.BELT_FORWARD, rotation)

    def _compare_shapes(self, expected: Shape, actual: Shape) -> float:
        """Compare two shapes and return similarity score (0-1)."""
        if expected.to_code() == actual.to_code():
            return 1.0

        # Partial matching based on layers
        matching_layers = 0
        total_layers = max(expected.num_layers, actual.num_layers)

        for i in range(min(expected.num_layers, actual.num_layers)):
            exp_layer = expected.get_layer(i)
            act_layer = actual.get_layer(i)
            if exp_layer and act_layer:
                matching_parts = sum(
                    1 for j in range(4)
                    if exp_layer.get_part(j) == act_layer.get_part(j)
                )
                matching_layers += matching_parts / 4

        return matching_layers / total_layers if total_layers > 0 else 0

    def _calculate_connectivity(self, candidate: Candidate) -> float:
        """Calculate what fraction of buildings are connected."""
        if len(candidate.buildings) <= 1:
            return 1.0

        connected = set()
        for conn in candidate.connections:
            connected.add(conn.from_building_id)
            connected.add(conn.to_building_id)

        return len(connected) / len(candidate.buildings)

    def evolve_generation(self) -> None:
        """Evolve one generation."""
        # Evaluate all candidates
        for candidate in self.population:
            self.evaluate_fitness(candidate)

        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Update top solutions
        for candidate in self.population[:self.num_top_solutions]:
            self._update_top_solutions(candidate)

        # Selection and reproduction
        new_population = []

        # Elitism: keep top solutions
        for i in range(min(5, len(self.population))):
            new_population.append(self.population[i].copy())

        # Fill rest with crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                # Crossover
                p1 = self._select_parent()
                p2 = self._select_parent()
                child = self._crossover(p1, p2)
            else:
                # Clone and mutate
                parent = self._select_parent()
                child = parent.copy()

            if random.random() < self.mutation_rate:
                self._mutate(child)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def _select_parent(self) -> Candidate:
        """Tournament selection."""
        tournament_size = 3
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda c: c.fitness)

    def _crossover(self, p1: Candidate, p2: Candidate) -> Candidate:
        """Crossover two parents."""
        child = Candidate()

        # Take buildings from both parents
        split = random.randint(0, len(p1.buildings))
        child.buildings = [copy.copy(b) for b in p1.buildings[:split]]

        # Add some buildings from p2
        for b in p2.buildings[split:]:
            new_b = copy.copy(b)
            new_b.building_id = len(child.buildings)
            child.buildings.append(new_b)

        # Create new connections
        self._create_random_connections(child)

        return child

    def _mutate(self, candidate: Candidate) -> None:
        """Mutate a candidate."""
        # Get valid buildings for this foundation size (internal grid)
        valid_buildings = []
        for bt in EVOLVABLE_BUILDINGS:
            spec = BUILDING_SPECS.get(bt, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
            if (spec.width <= self.config.spec.grid_width and
                spec.height <= self.config.spec.grid_height and
                spec.depth <= self.num_floors):
                valid_buildings.append(bt)

        if not valid_buildings:
            return

        mutation_type = random.choice(['add', 'remove', 'modify', 'reconnect'])

        if mutation_type == 'add' and len(candidate.buildings) < self.max_buildings:
            # Add a building
            building_type = random.choice(valid_buildings)
            spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
            building = PlacedBuilding(
                building_id=len(candidate.buildings),
                building_type=building_type,
                x=random.randint(0, max(0, self.config.spec.grid_width - spec.width)),
                y=random.randint(0, max(0, self.config.spec.grid_height - spec.height)),
                floor=random.randint(0, max(0, self.num_floors - spec.depth)),
                rotation=random.choice(list(Rotation))
            )
            candidate.buildings.append(building)

        elif mutation_type == 'remove' and len(candidate.buildings) > 1:
            # Remove a random building
            idx = random.randint(0, len(candidate.buildings) - 1)
            removed_id = candidate.buildings[idx].building_id
            candidate.buildings.pop(idx)
            # Remove connections to/from removed building
            candidate.connections = [
                c for c in candidate.connections
                if c.from_building_id != removed_id and c.to_building_id != removed_id
            ]

        elif mutation_type == 'modify' and candidate.buildings:
            # Modify a random building
            building = random.choice(candidate.buildings)
            mod_type = random.choice(['position', 'rotation', 'type'])

            if mod_type == 'position':
                spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
                building.x = random.randint(0, max(0, self.config.spec.grid_width - spec.width))
                building.y = random.randint(0, max(0, self.config.spec.grid_height - spec.height))
            elif mod_type == 'rotation':
                building.rotation = random.choice(list(Rotation))
            else:
                # Change building type to one that fits
                building.building_type = random.choice(valid_buildings)
                spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
                # Ensure position is still valid
                building.x = min(building.x, max(0, self.config.spec.grid_width - spec.width))
                building.y = min(building.y, max(0, self.config.spec.grid_height - spec.height))
                building.floor = min(building.floor, max(0, self.num_floors - spec.depth))

        elif mutation_type == 'reconnect':
            # Modify connections
            if candidate.connections:
                if random.random() < 0.5:
                    # Remove a connection
                    idx = random.randint(0, len(candidate.connections) - 1)
                    candidate.connections.pop(idx)
                else:
                    # Add a connection
                    self._create_random_connections(candidate)

    def _update_top_solutions(self, candidate: Candidate) -> None:
        """Update the top solutions list."""
        # Check if this is a new unique solution
        for existing in self.top_solutions:
            if self._solutions_similar(candidate, existing):
                if candidate.fitness > existing.fitness:
                    existing.fitness = candidate.fitness
                    existing.buildings = [copy.copy(b) for b in candidate.buildings]
                    existing.connections = [copy.copy(c) for c in candidate.connections]
                return

        # Add new solution
        self.top_solutions.append(candidate.copy())
        self.top_solutions.sort(key=lambda c: c.fitness, reverse=True)
        self.top_solutions = self.top_solutions[:self.num_top_solutions]

    def _solutions_similar(self, c1: Candidate, c2: Candidate) -> bool:
        """Check if two solutions are similar."""
        if len(c1.buildings) != len(c2.buildings):
            return False

        # Compare building types and positions (use .value for sorting)
        types1 = sorted([(b.building_type.value, b.x, b.y, b.floor) for b in c1.buildings])
        types2 = sorted([(b.building_type.value, b.x, b.y, b.floor) for b in c2.buildings])
        return types1 == types2

    def run(self, num_generations: int, verbose: bool = True) -> List[Candidate]:
        """
        Run the evolution for a number of generations.

        Args:
            num_generations: Number of generations to run
            verbose: Whether to print progress

        Returns:
            Top solutions
        """
        self.initialize_population()

        for gen in range(num_generations):
            self.evolve_generation()

            if verbose and (gen % 10 == 0 or gen == num_generations - 1):
                self._print_progress()

        return self.top_solutions

    def _print_progress(self) -> None:
        """Print evolution progress."""
        print(f"\n{'=' * 70}")
        print(f"Generation {self.generation}")
        print(f"{'=' * 70}")

        if self.population:
            best = self.population[0]
            avg = sum(c.fitness for c in self.population) / len(self.population)
            print(f"Best Fitness: {best.fitness:.2f}")
            print(f"Avg Fitness: {avg:.2f}")
            print(f"Buildings: {len(best.buildings)}")

        print(f"\nTop {self.num_top_solutions} Solutions:")
        for i, sol in enumerate(self.top_solutions):
            print(f"  {i+1}. Fitness: {sol.fitness:.2f}, Buildings: {len(sol.buildings)}")

        # Visualize best solution
        if self.top_solutions:
            self._visualize_solution(self.top_solutions[0])

    def _visualize_solution(self, candidate: Candidate) -> None:
        """Visualize a solution showing each floor."""
        print(f"\nLayout Visualization:")

        # Symbol mapping
        symbols = {
            BuildingType.ROTATOR_CW: 'R',
            BuildingType.ROTATOR_CCW: 'r',
            BuildingType.ROTATOR_180: '⟳',
            BuildingType.CUTTER: 'C',
            BuildingType.CUTTER_MIRRORED: 'c',
            BuildingType.HALF_CUTTER: 'H',
            BuildingType.SWAPPER: 'X',
            BuildingType.STACKER: 'S',
            BuildingType.UNSTACKER: 'U',
            BuildingType.PIN_PUSHER: 'P',
            BuildingType.BELT_FORWARD: '→',
            BuildingType.BELT_LEFT: '↰',
            BuildingType.BELT_RIGHT: '↱',
            BuildingType.LIFT_UP: '↑',
            BuildingType.LIFT_DOWN: '↓',
            BuildingType.BELT_PORT_SENDER: '⊳',
            BuildingType.BELT_PORT_RECEIVER: '⊲',
            BuildingType.SPLITTER: '⊤',
            BuildingType.MERGER: '⊥',
        }

        for floor in range(self.num_floors):
            floor_buildings = [b for b in candidate.buildings if b.floor == floor]

            if not floor_buildings and floor > 0:
                continue

            print(f"\n  Floor {floor}:")

            # Create simplified grid (show 1x1 units, not full internal grid)
            width = self.config.spec.units_x + 4
            height = self.config.spec.units_y + 4
            grid = [['·' for _ in range(width)] for _ in range(height)]

            # Draw foundation outline (each 1x1 unit)
            for ux in range(self.config.spec.units_x):
                for uy in range(self.config.spec.units_y):
                    # Check if cell is present (for irregular foundations)
                    if self.config.spec.present_cells is not None:
                        if (ux, uy) not in self.config.spec.present_cells:
                            continue
                    grid[uy + 2][ux + 2] = '░'

            # Draw ports
            for side, pos, f, shape_code in self.config.get_all_inputs():
                if f == floor:
                    px, py = self._get_port_grid_pos(side, pos)
                    if 0 <= py < height and 0 <= px < width:
                        grid[py][px] = 'I'

            for side, pos, f, shape_code in self.config.get_all_outputs():
                if f == floor:
                    px, py = self._get_port_grid_pos(side, pos)
                    if 0 <= py < height and 0 <= px < width:
                        grid[py][px] = 'O'

            # Draw buildings
            for building in floor_buildings:
                x = building.x + 2
                y = building.y + 2
                symbol = symbols.get(building.building_type, '?')
                if 0 <= y < height and 0 <= x < width:
                    grid[y][x] = symbol

            # Print grid
            print("    " + "".join(str(i % 10) for i in range(width)))
            for row_idx, row in enumerate(grid):
                print(f"  {row_idx:2d} {''.join(row)}")

    def _get_port_grid_pos(self, side: Side, pos: int) -> Tuple[int, int]:
        """Get simplified grid position for a port (for visualization).

        In the simplified view, each 1x1 unit is one cell.
        Ports are shown at the edge of the unit they belong to.
        """
        unit_idx = pos // 4  # Which 1x1 unit this port belongs to

        if side == Side.NORTH:
            return (unit_idx + 2, 1)
        elif side == Side.SOUTH:
            return (unit_idx + 2, self.config.spec.units_y + 2)
        elif side == Side.WEST:
            return (1, unit_idx + 2)
        elif side == Side.EAST:
            return (self.config.spec.units_x + 2, unit_idx + 2)
        return (0, 0)

    def export_blueprint(self, candidate: Candidate) -> str:
        """Export a candidate as a blueprint code."""
        encoder = BlueprintEncoder()

        for building in candidate.buildings:
            entry = {
                "T": building.building_type.value,
                "X": building.x,
                "Y": building.y,
                "L": building.floor,
                "R": building.rotation.value,
            }
            encoder.entries.append(entry)

        return encoder.encode()

    def print_goal(self) -> None:
        """Print the evolution goal (inputs/outputs)."""
        print("\n" + "=" * 70)
        print("EVOLUTION GOAL")
        print("=" * 70)
        print(f"Foundation: {self.config.spec.name}")
        print(f"Units: {self.config.spec.units_x}x{self.config.spec.units_y}")
        print(f"Internal grid: {self.config.spec.grid_width}x{self.config.spec.grid_height} tiles")
        print(f"Floors: {self.num_floors}")

        print("\nINPUTS:")
        for side, pos, floor, shape_code in self.config.get_all_inputs():
            print(f"  {side.value}[{pos}] Floor {floor}: {shape_code}")

        print("\nEXPECTED OUTPUTS:")
        for side, pos, floor, shape_code in self.config.get_all_outputs():
            print(f"  {side.value}[{pos}] Floor {floor}: {shape_code}")

        print("\n" + self.config.print_config())


def create_evolution_from_spec(
    foundation_type: str,
    inputs: List[Tuple[str, int, int, str]],  # (side, pos, floor, shape_code)
    outputs: List[Tuple[str, int, int, str]],  # (side, pos, floor, shape_code)
    **kwargs
) -> FoundationEvolution:
    """
    Create an evolution instance from a specification.

    Args:
        foundation_type: Foundation type name (e.g., "2x2")
        inputs: List of (side, position, floor, shape_code) tuples
        outputs: List of (side, position, floor, shape_code) tuples
        **kwargs: Additional arguments for FoundationEvolution

    Returns:
        Configured FoundationEvolution instance
    """
    spec = FOUNDATION_SPECS.get(foundation_type)
    if not spec:
        raise ValueError(f"Unknown foundation type: {foundation_type}")

    config = FoundationConfig(spec)

    side_map = {"N": Side.NORTH, "E": Side.EAST, "S": Side.SOUTH, "W": Side.WEST}

    for side_str, pos, floor, shape_code in inputs:
        side = side_map.get(side_str.upper())
        if side:
            config.set_input(side, pos, floor, shape_code)

    for side_str, pos, floor, shape_code in outputs:
        side = side_map.get(side_str.upper())
        if side:
            config.set_output(side, pos, floor, shape_code)

    return FoundationEvolution(config, **kwargs)
