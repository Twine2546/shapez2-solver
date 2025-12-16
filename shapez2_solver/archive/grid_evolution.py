"""Grid-based evolutionary algorithm for layout optimization."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Type, Union
from enum import Enum

from ..shapes.shape import Shape
from ..operations.base import Operation
from ..operations.cutter import CutOperation
from ..operations.rotator import RotateOperation
from ..operations.stacker import StackOperation, UnstackOperation
from ..blueprint.building_types import (
    BuildingType, BUILDING_SPECS, BuildingSpec, Rotation,
    OPERATION_TO_BUILDING, BUILDING_PORTS
)
from ..blueprint.encoder import BlueprintEncoder


@dataclass
class GridBuilding:
    """A building placed on the grid."""
    building_type: BuildingType
    x: int
    y: int
    floor: int = 0
    rotation: Rotation = Rotation.EAST

    # For operation buildings, store the operation
    operation: Optional[Operation] = None

    def get_input_positions(self) -> List[Tuple[int, int, int]]:
        """Get absolute input positions."""
        ports = BUILDING_PORTS.get(self.building_type, {'inputs': [(-1, 0, 0)]})
        result = []
        for rel_x, rel_y, rel_floor in ports.get('inputs', []):
            # Rotate the offset based on building rotation
            rx, ry = self._rotate_offset(rel_x, rel_y)
            result.append((self.x + rx, self.y + ry, self.floor + rel_floor))
        return result

    def get_output_positions(self) -> List[Tuple[int, int, int]]:
        """Get absolute output positions."""
        ports = BUILDING_PORTS.get(self.building_type, {'outputs': [(1, 0, 0)]})
        result = []
        for rel_x, rel_y, rel_floor in ports.get('outputs', []):
            rx, ry = self._rotate_offset(rel_x, rel_y)
            result.append((self.x + rx, self.y + ry, self.floor + rel_floor))
        return result

    def _rotate_offset(self, dx: int, dy: int) -> Tuple[int, int]:
        """Rotate an offset based on building rotation."""
        if self.rotation == Rotation.EAST:
            return (dx, dy)
        elif self.rotation == Rotation.SOUTH:
            return (-dy, dx)
        elif self.rotation == Rotation.WEST:
            return (-dx, -dy)
        else:  # NORTH
            return (dy, -dx)

    def get_cells(self) -> List[Tuple[int, int, int]]:
        """Get all cells occupied by this building."""
        spec = BUILDING_SPECS.get(self.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
        cells = []
        for f in range(spec.depth):
            for dx in range(spec.width):
                for dy in range(spec.height):
                    rx, ry = self._rotate_offset(dx, dy)
                    cells.append((self.x + rx, self.y + ry, self.floor + f))
        return cells


@dataclass
class BeltPath:
    """A belt connection between two buildings."""
    from_building_idx: int
    from_output: int
    to_building_idx: int
    to_input: int
    # Belt segments: list of (x, y, floor, belt_type, rotation)
    segments: List[Tuple[int, int, int, BuildingType, Rotation]] = field(default_factory=list)


@dataclass
class GridCandidate:
    """A candidate solution with grid-based layout."""
    width: int
    height: int
    num_floors: int
    buildings: List[GridBuilding] = field(default_factory=list)
    belts: List[BeltPath] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0

    # Special indices
    input_building_idx: int = -1  # Index of input marker
    output_building_indices: List[int] = field(default_factory=list)

    def copy(self) -> 'GridCandidate':
        """Create a deep copy."""
        new = GridCandidate(
            width=self.width,
            height=self.height,
            num_floors=self.num_floors,
            fitness=self.fitness,
            generation=self.generation,
            input_building_idx=self.input_building_idx,
            output_building_indices=list(self.output_building_indices),
        )
        new.buildings = [
            GridBuilding(
                building_type=b.building_type,
                x=b.x, y=b.y, floor=b.floor,
                rotation=b.rotation,
                operation=b.operation,
            )
            for b in self.buildings
        ]
        new.belts = [
            BeltPath(
                from_building_idx=bp.from_building_idx,
                from_output=bp.from_output,
                to_building_idx=bp.to_building_idx,
                to_input=bp.to_input,
                segments=list(bp.segments),
            )
            for bp in self.belts
        ]
        return new

    def get_occupied_cells(self) -> Set[Tuple[int, int, int]]:
        """Get all occupied cells."""
        cells = set()
        for b in self.buildings:
            cells.update(b.get_cells())
        for bp in self.belts:
            for x, y, f, _, _ in bp.segments:
                cells.add((x, y, f))
        return cells

    def is_valid_placement(self, building: GridBuilding, exclude_idx: int = -1) -> bool:
        """Check if a building can be placed without overlapping."""
        cells = building.get_cells()

        # Check bounds
        for x, y, f in cells:
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return False
            if f < 0 or f >= self.num_floors:
                return False

        # Check overlaps with other buildings
        for i, other in enumerate(self.buildings):
            if i == exclude_idx:
                continue
            other_cells = set(other.get_cells())
            if any(c in other_cells for c in cells):
                return False

        return True

    def to_blueprint(self) -> str:
        """Convert to blueprint code."""
        encoder = BlueprintEncoder()

        for building in self.buildings:
            # Skip input/output markers (they're virtual)
            if building.building_type == BuildingType.BELT_FORWARD:
                continue

            entry = {
                'T': building.building_type.value,
                'X': building.x,
                'Y': building.y,
                'L': building.floor,
                'R': building.rotation.value,
            }
            encoder.entries.append(entry)

        # Add belt segments
        for bp in self.belts:
            for x, y, f, belt_type, rotation in bp.segments:
                entry = {
                    'T': belt_type.value,
                    'X': x,
                    'Y': y,
                    'L': f,
                    'R': rotation.value,
                }
                encoder.entries.append(entry)

        return encoder.encode()


class GridEvolution:
    """Evolutionary algorithm for grid-based layout optimization."""

    def __init__(
        self,
        width: int,
        height: int,
        num_floors: int,
        input_shape: Shape,
        expected_outputs: Dict[str, Optional[Shape]],
        allowed_operations: List[Type[Operation]] = None,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.3,
    ):
        self.width = width
        self.height = height
        self.num_floors = num_floors
        self.input_shape = input_shape
        self.expected_outputs = expected_outputs
        self.num_outputs = len(expected_outputs)

        self.allowed_operations = allowed_operations or [CutOperation, RotateOperation]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.population: List[GridCandidate] = []
        self.best_candidate: Optional[GridCandidate] = None
        self.generation = 0

    def _operation_to_building_type(self, op: Operation) -> BuildingType:
        """Convert operation to building type."""
        op_class = op.__class__.__name__

        if op_class == "RotateOperation":
            steps = getattr(op, 'steps', 1)
            if steps == 1:
                return BuildingType.ROTATOR_CW
            elif steps == 2:
                return BuildingType.ROTATOR_180
            elif steps == 3:
                return BuildingType.ROTATOR_CCW
            return BuildingType.ROTATOR_CW

        return OPERATION_TO_BUILDING.get(op_class, BuildingType.BELT_FORWARD)

    def _create_random_candidate(self) -> GridCandidate:
        """Create a random candidate layout."""
        candidate = GridCandidate(
            width=self.width,
            height=self.height,
            num_floors=self.num_floors,
            generation=self.generation,
        )

        # Add input marker on left side
        input_y = self.height // 2
        input_building = GridBuilding(
            building_type=BuildingType.BELT_FORWARD,
            x=0, y=input_y, floor=0,
            rotation=Rotation.EAST,
        )
        candidate.buildings.append(input_building)
        candidate.input_building_idx = 0

        # Add output markers on right side
        for i in range(self.num_outputs):
            output_y = (i + 1) * self.height // (self.num_outputs + 1)
            output_building = GridBuilding(
                building_type=BuildingType.BELT_FORWARD,
                x=self.width - 1, y=output_y, floor=0,
                rotation=Rotation.EAST,
            )
            candidate.buildings.append(output_building)
            candidate.output_building_indices.append(len(candidate.buildings) - 1)

        # Add random operations
        num_ops = random.randint(3, 8)
        for _ in range(num_ops):
            op_class = random.choice(self.allowed_operations)
            if op_class == RotateOperation:
                op = op_class(steps=random.randint(1, 3))
            else:
                op = op_class()

            building_type = self._operation_to_building_type(op)
            spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

            # Try to place at random position
            for _ in range(20):  # Max attempts
                x = random.randint(2, self.width - spec.width - 2)
                y = random.randint(0, self.height - spec.height)
                floor = random.randint(0, self.num_floors - spec.depth)
                rotation = random.choice(list(Rotation))

                building = GridBuilding(
                    building_type=building_type,
                    x=x, y=y, floor=floor,
                    rotation=rotation,
                    operation=op,
                )

                if candidate.is_valid_placement(building):
                    candidate.buildings.append(building)
                    break

        # Create random belt connections
        self._create_random_connections(candidate)

        return candidate

    def _create_random_connections(self, candidate: GridCandidate) -> None:
        """Create belt connections ensuring all multi-output ops are properly routed."""
        candidate.belts.clear()

        # Get operation buildings sorted by x position (left to right flow)
        op_indices = [i for i, b in enumerate(candidate.buildings) if b.operation is not None]
        op_indices.sort(key=lambda i: candidate.buildings[i].x)

        if not op_indices:
            # Connect input directly to outputs
            for out_idx in candidate.output_building_indices:
                bp = BeltPath(
                    from_building_idx=candidate.input_building_idx,
                    from_output=0,
                    to_building_idx=out_idx,
                    to_input=0,
                )
                self._route_belt(candidate, bp)
                candidate.belts.append(bp)
            return

        # Track available outputs: list of (building_idx, output_idx)
        # and which outputs have been used
        available_outputs: List[Tuple[int, int]] = [(candidate.input_building_idx, 0)]
        used_outputs: Set[Tuple[int, int]] = set()

        # Process operations left to right
        for op_idx in op_indices:
            building = candidate.buildings[op_idx]
            spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

            # Connect each input from an available output
            for inp in range(spec.num_inputs):
                if not available_outputs:
                    break

                # Prefer unused outputs from multi-output buildings
                unused = [o for o in available_outputs if o not in used_outputs]
                if unused:
                    source = random.choice(unused)
                else:
                    source = random.choice(available_outputs)

                bp = BeltPath(
                    from_building_idx=source[0],
                    from_output=source[1],
                    to_building_idx=op_idx,
                    to_input=inp,
                )
                self._route_belt(candidate, bp)
                candidate.belts.append(bp)
                used_outputs.add(source)

            # Add this operation's outputs to available pool
            for out in range(spec.num_outputs):
                available_outputs.append((op_idx, out))

        # Connect to output markers - try to use all unique outputs
        remaining_outputs = [o for o in available_outputs if o not in used_outputs]

        # Shuffle to get variety
        random.shuffle(remaining_outputs)

        for i, out_idx in enumerate(candidate.output_building_indices):
            if remaining_outputs:
                source = remaining_outputs.pop(0)
            elif available_outputs:
                # Fall back to any available output
                source = available_outputs[i % len(available_outputs)]
            else:
                continue

            bp = BeltPath(
                from_building_idx=source[0],
                from_output=source[1],
                to_building_idx=out_idx,
                to_input=0,
            )
            self._route_belt(candidate, bp)
            candidate.belts.append(bp)
            used_outputs.add(source)

    def _route_belt(self, candidate: GridCandidate, belt_path: BeltPath) -> None:
        """Route a belt between two buildings using simple L-routing."""
        belt_path.segments.clear()

        from_building = candidate.buildings[belt_path.from_building_idx]
        to_building = candidate.buildings[belt_path.to_building_idx]

        # Get connection points
        from_outputs = from_building.get_output_positions()
        to_inputs = to_building.get_input_positions()

        if belt_path.from_output >= len(from_outputs):
            return
        if belt_path.to_input >= len(to_inputs):
            return

        start = from_outputs[belt_path.from_output]
        end = to_inputs[belt_path.to_input]

        # Simple L-routing
        x, y, floor = start
        end_x, end_y, end_floor = end

        occupied = candidate.get_occupied_cells()

        # Horizontal first
        dx = 1 if end_x > x else -1
        rotation = Rotation.EAST if dx > 0 else Rotation.WEST
        while x != end_x:
            if (x, y, floor) not in occupied:
                belt_path.segments.append((x, y, floor, BuildingType.BELT_FORWARD, rotation))
            x += dx

        # Then vertical
        dy = 1 if end_y > y else -1
        rotation = Rotation.SOUTH if dy > 0 else Rotation.NORTH
        while y != end_y:
            if (x, y, floor) not in occupied:
                belt_path.segments.append((x, y, floor, BuildingType.BELT_FORWARD, rotation))
            y += dy

        # Floor change if needed
        while floor != end_floor:
            if floor < end_floor:
                if (x, y, floor) not in occupied:
                    belt_path.segments.append((x, y, floor, BuildingType.LIFT_UP, Rotation.EAST))
                floor += 1
            else:
                floor -= 1
                if (x, y, floor) not in occupied:
                    belt_path.segments.append((x, y, floor, BuildingType.LIFT_DOWN, Rotation.EAST))

    def _evaluate_candidate(self, candidate: GridCandidate) -> float:
        """Evaluate a candidate's fitness."""
        # Simulate the design
        try:
            results = self._simulate(candidate)
        except Exception:
            return 0.0

        if not results:
            return 0.0

        # Compare outputs
        total_score = 0.0
        for i, (name, expected) in enumerate(self.expected_outputs.items()):
            out_key = f"out_{i}"
            actual = results.get(out_key)

            if expected is None:
                if actual is None:
                    total_score += 1.0
            elif actual is not None:
                # Compare shapes
                score = self._compare_shapes(actual, expected)
                total_score += score

        fitness = total_score / len(self.expected_outputs)

        # Bonus for using fewer buildings
        num_ops = sum(1 for b in candidate.buildings if b.operation is not None)
        if fitness > 0.5:
            fitness += 0.01 * (10 - min(num_ops, 10)) / 10

        return min(fitness, 1.0)

    def _simulate(self, candidate: GridCandidate) -> Dict[str, Optional[Shape]]:
        """Simulate the candidate to get output shapes."""
        # Build signal flow graph
        signals: Dict[Tuple[int, int], Optional[Shape]] = {}

        # Initialize input
        signals[(candidate.input_building_idx, 0)] = self.input_shape

        # Process buildings in connection order
        # First, build adjacency from belts
        connections: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for bp in candidate.belts:
            key = (bp.from_building_idx, bp.from_output)
            if key not in connections:
                connections[key] = []
            connections[key].append((bp.to_building_idx, bp.to_input))

        # Process each operation building
        processed = set()
        max_iterations = 100

        for _ in range(max_iterations):
            progress = False

            for i, building in enumerate(candidate.buildings):
                if i in processed:
                    continue
                if building.operation is None:
                    continue

                # Check if all inputs are available
                spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
                inputs_ready = True
                input_shapes = []

                for inp_idx in range(spec.num_inputs):
                    # Find connection to this input
                    found = False
                    for (from_b, from_o), targets in connections.items():
                        for to_b, to_i in targets:
                            if to_b == i and to_i == inp_idx:
                                if (from_b, from_o) in signals:
                                    input_shapes.append(signals[(from_b, from_o)])
                                    found = True
                                    break
                        if found:
                            break

                    if not found:
                        inputs_ready = False
                        break

                if not inputs_ready:
                    continue

                # Execute operation
                try:
                    if spec.num_inputs == 1:
                        result = building.operation.execute(input_shapes[0])
                    else:
                        result = building.operation.execute(*input_shapes)

                    # Store outputs
                    if isinstance(result, tuple):
                        for out_idx, out_shape in enumerate(result):
                            signals[(i, out_idx)] = out_shape
                    else:
                        signals[(i, 0)] = result

                    processed.add(i)
                    progress = True
                except Exception:
                    processed.add(i)

            if not progress:
                break

        # Collect outputs
        results = {}
        for out_num, out_idx in enumerate(candidate.output_building_indices):
            # Find what's connected to this output marker
            for (from_b, from_o), targets in connections.items():
                for to_b, to_i in targets:
                    if to_b == out_idx:
                        results[f"out_{out_num}"] = signals.get((from_b, from_o))
                        break

        return results

    def _compare_shapes(self, actual: Shape, expected: Shape) -> float:
        """Compare two shapes and return similarity score."""
        if actual is None or expected is None:
            return 0.0

        actual_code = actual.to_code()
        expected_code = expected.to_code()

        if actual_code == expected_code:
            return 1.0

        # Partial matching based on quadrants
        score = 0.0
        min_len = min(len(actual_code), len(expected_code))

        for i in range(0, min_len, 2):
            if i + 1 < min_len:
                if actual_code[i:i+2] == expected_code[i:i+2]:
                    score += 0.25
                elif actual_code[i:i+2] == '--' or expected_code[i:i+2] == '--':
                    if actual_code[i:i+2] == expected_code[i:i+2]:
                        score += 0.25

        return score

    def _mutate(self, candidate: GridCandidate) -> GridCandidate:
        """Mutate a candidate."""
        new_candidate = candidate.copy()

        mutation_type = random.choice(['move', 'rotate', 'add', 'remove', 'rewire'])

        op_indices = [i for i, b in enumerate(new_candidate.buildings) if b.operation is not None]

        if mutation_type == 'move' and op_indices:
            # Move a random operation building
            idx = random.choice(op_indices)
            building = new_candidate.buildings[idx]
            spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

            for _ in range(10):
                new_x = building.x + random.randint(-3, 3)
                new_y = building.y + random.randint(-3, 3)
                new_x = max(1, min(new_x, self.width - spec.width - 1))
                new_y = max(0, min(new_y, self.height - spec.height))

                test_building = GridBuilding(
                    building_type=building.building_type,
                    x=new_x, y=new_y, floor=building.floor,
                    rotation=building.rotation,
                    operation=building.operation,
                )

                if new_candidate.is_valid_placement(test_building, exclude_idx=idx):
                    new_candidate.buildings[idx] = test_building
                    break

        elif mutation_type == 'rotate' and op_indices:
            # Rotate a random building
            idx = random.choice(op_indices)
            new_candidate.buildings[idx].rotation = random.choice(list(Rotation))

        elif mutation_type == 'add':
            # Add a new operation
            op_class = random.choice(self.allowed_operations)
            if op_class == RotateOperation:
                op = op_class(steps=random.randint(1, 3))
            else:
                op = op_class()

            building_type = self._operation_to_building_type(op)
            spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

            for _ in range(20):
                x = random.randint(2, self.width - spec.width - 2)
                y = random.randint(0, self.height - spec.height)
                floor = 0

                building = GridBuilding(
                    building_type=building_type,
                    x=x, y=y, floor=floor,
                    rotation=random.choice(list(Rotation)),
                    operation=op,
                )

                if new_candidate.is_valid_placement(building):
                    new_candidate.buildings.append(building)
                    break

        elif mutation_type == 'remove' and len(op_indices) > 1:
            # Remove a random operation
            idx = random.choice(op_indices)
            new_candidate.buildings.pop(idx)

            # Update input index
            if new_candidate.input_building_idx > idx:
                new_candidate.input_building_idx -= 1

            # Update output indices
            for i in range(len(new_candidate.output_building_indices)):
                if new_candidate.output_building_indices[i] > idx:
                    new_candidate.output_building_indices[i] -= 1

            # Remove invalid belts and update indices
            valid_belts = []
            for bp in new_candidate.belts:
                if bp.from_building_idx == idx or bp.to_building_idx == idx:
                    continue  # Remove belts connected to deleted building

                if bp.from_building_idx > idx:
                    bp.from_building_idx -= 1
                if bp.to_building_idx > idx:
                    bp.to_building_idx -= 1
                valid_belts.append(bp)

            new_candidate.belts = valid_belts

            # Rewire after removal
            self._create_random_connections(new_candidate)

        elif mutation_type == 'rewire':
            # Rewire connections
            self._create_random_connections(new_candidate)

        # Re-route all belts (check validity first)
        valid_belts = []
        for bp in new_candidate.belts:
            if (0 <= bp.from_building_idx < len(new_candidate.buildings) and
                0 <= bp.to_building_idx < len(new_candidate.buildings)):
                self._route_belt(new_candidate, bp)
                valid_belts.append(bp)
        new_candidate.belts = valid_belts

        return new_candidate

    def _crossover(self, parent1: GridCandidate, parent2: GridCandidate) -> GridCandidate:
        """Crossover two candidates."""
        child = GridCandidate(
            width=self.width,
            height=self.height,
            num_floors=self.num_floors,
            generation=self.generation,
        )

        # Copy input/output structure from parent1
        child.buildings.append(parent1.buildings[parent1.input_building_idx])
        child.input_building_idx = 0

        for out_idx in parent1.output_building_indices:
            child.buildings.append(parent1.buildings[out_idx])
            child.output_building_indices.append(len(child.buildings) - 1)

        # Mix operation buildings from both parents
        ops1 = [b for b in parent1.buildings if b.operation is not None]
        ops2 = [b for b in parent2.buildings if b.operation is not None]

        # Take some from each parent
        for op in ops1:
            if random.random() < 0.5:
                new_building = GridBuilding(
                    building_type=op.building_type,
                    x=op.x, y=op.y, floor=op.floor,
                    rotation=op.rotation,
                    operation=op.operation,
                )
                if child.is_valid_placement(new_building):
                    child.buildings.append(new_building)

        for op in ops2:
            if random.random() < 0.5:
                new_building = GridBuilding(
                    building_type=op.building_type,
                    x=op.x, y=op.y, floor=op.floor,
                    rotation=op.rotation,
                    operation=op.operation,
                )
                if child.is_valid_placement(new_building):
                    child.buildings.append(new_building)

        # Create connections
        self._create_random_connections(child)

        return child

    def run(self, callback=None) -> Optional[GridCandidate]:
        """Run the evolutionary algorithm."""
        # Initialize population
        self.population = [self._create_random_candidate() for _ in range(self.population_size)]

        # Evaluate initial population
        for candidate in self.population:
            candidate.fitness = self._evaluate_candidate(candidate)

        self.best_candidate = max(self.population, key=lambda c: c.fitness)

        # Evolution loop
        for gen in range(self.generations):
            self.generation = gen + 1

            # Sort by fitness
            self.population.sort(key=lambda c: c.fitness, reverse=True)

            # Update best
            if self.population[0].fitness > self.best_candidate.fitness:
                self.best_candidate = self.population[0].copy()

            # Callback
            if callback:
                if not callback(self.generation, self.best_candidate.fitness):
                    break

            # Early stopping
            if self.best_candidate.fitness >= 1.0:
                break

            # Create new population
            new_population = []

            # Elitism
            new_population.append(self.population[0].copy())
            new_population.append(self.population[1].copy())

            # Fill rest
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament = random.sample(self.population, min(3, len(self.population)))
                parent1 = max(tournament, key=lambda c: c.fitness)
                tournament = random.sample(self.population, min(3, len(self.population)))
                parent2 = max(tournament, key=lambda c: c.fitness)

                # Crossover
                if random.random() < 0.7:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                # Evaluate
                child.fitness = self._evaluate_candidate(child)
                child.generation = self.generation
                new_population.append(child)

            self.population = new_population

        return self.best_candidate
