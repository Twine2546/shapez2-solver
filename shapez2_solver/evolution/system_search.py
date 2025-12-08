"""
System Search - Phase 1 of Two-Phase Search

This module searches for a logical system design (which machines and how they connect)
that transforms the input shapes to the desired output shapes, without worrying about
physical layout.

The search space is machine topologies (graph of machines and connections).
Fitness is based on whether the system produces correct outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from copy import deepcopy
import random

from ..shapes.shape import Shape
from ..operations import (
    Operation, OperationType,
    CutOperation, HalfDestroyerOperation, SwapperOperation,
    RotateOperation, StackOperation, UnstackOperation,
    PaintOperation, PinPusherOperation
)
from ..blueprint.building_types import BuildingType


# Mapping from BuildingType to Operation class
BUILDING_TO_OPERATION: Dict[BuildingType, type] = {
    BuildingType.CUTTER: CutOperation,
    BuildingType.HALF_CUTTER: HalfDestroyerOperation,
    BuildingType.SWAPPER: SwapperOperation,
    BuildingType.ROTATOR_CW: lambda: RotateOperation(1),
    BuildingType.ROTATOR_CCW: lambda: RotateOperation(3),
    BuildingType.ROTATOR_180: lambda: RotateOperation(2),
    BuildingType.STACKER: StackOperation,
    BuildingType.UNSTACKER: UnstackOperation,
    BuildingType.PAINTER: PaintOperation,
    BuildingType.PIN_PUSHER: PinPusherOperation,
}


def create_operation(building_type: BuildingType) -> Optional[Operation]:
    """Create an operation instance from a building type."""
    op_class = BUILDING_TO_OPERATION.get(building_type)
    if op_class is None:
        return None
    if callable(op_class):
        result = op_class()
        if isinstance(result, Operation):
            return result
        return result  # Lambda returned Operation
    return op_class()


@dataclass
class MachineNode:
    """A machine node in the system graph."""
    node_id: int
    building_type: BuildingType
    # Connections: list of (source_node_id, source_output_idx) for each input
    # None means unconnected input
    input_connections: List[Optional[Tuple[int, int]]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize input connections list based on building type."""
        op = create_operation(self.building_type)
        if op and not self.input_connections:
            self.input_connections = [None] * op.num_inputs

    def num_inputs(self) -> int:
        """Get number of inputs for this machine."""
        op = create_operation(self.building_type)
        return op.num_inputs if op else 1

    def num_outputs(self) -> int:
        """Get number of outputs for this machine."""
        op = create_operation(self.building_type)
        return op.num_outputs if op else 1


@dataclass
class InputPort:
    """An input port on the foundation."""
    port_id: int
    side: str  # N, E, S, W
    position: int
    floor: int
    shape: Shape


@dataclass
class OutputPort:
    """An output port on the foundation."""
    port_id: int
    side: str  # N, E, S, W
    position: int
    floor: int
    expected_shape: Shape
    # Connection: (source_node_id, source_output_idx) or None
    source: Optional[Tuple[int, int]] = None


@dataclass
class SystemDesign:
    """
    A complete system design specifying machines and their connections.

    This is the output of Phase 1 (system search) and input to Phase 2 (layout search).
    """
    input_ports: List[InputPort]
    output_ports: List[OutputPort]
    machines: List[MachineNode]

    # Track which output each machine input gets its shape from
    # Key: (node_id, input_idx), Value: (source_type, source_id, output_idx)
    # source_type: 'input_port' or 'machine'
    connections: Dict[Tuple[int, int], Tuple[str, int, int]] = field(default_factory=dict)

    def copy(self) -> 'SystemDesign':
        """Create a deep copy of this system design."""
        return SystemDesign(
            input_ports=[InputPort(p.port_id, p.side, p.position, p.floor, p.shape.copy())
                        for p in self.input_ports],
            output_ports=[OutputPort(p.port_id, p.side, p.position, p.floor, p.expected_shape.copy(), p.source)
                         for p in self.output_ports],
            machines=[MachineNode(m.node_id, m.building_type, list(m.input_connections))
                     for m in self.machines],
            connections=dict(self.connections)
        )


class SystemSimulator:
    """Simulates a system design to compute outputs."""

    def __init__(self, design: SystemDesign):
        self.design = design
        self._cache: Dict[Tuple[str, int, int], Optional[Shape]] = {}
        self._computing: Set[Tuple[str, int, int]] = set()  # Track nodes being computed (cycle detection)

    def get_output(self, source_type: str, source_id: int, output_idx: int) -> Optional[Shape]:
        """
        Get the shape output from a source.

        Args:
            source_type: 'input_port' or 'machine'
            source_id: The port_id or node_id
            output_idx: Which output index (0 for input ports)

        Returns:
            The output shape, or None if not available
        """
        cache_key = (source_type, source_id, output_idx)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Cycle detection
        if cache_key in self._computing:
            return None  # Cycle detected, return None to break it
        self._computing.add(cache_key)

        try:
            result = self._compute_output(source_type, source_id, output_idx)
            self._cache[cache_key] = result
            return result
        finally:
            self._computing.discard(cache_key)

    def _compute_output(self, source_type: str, source_id: int, output_idx: int) -> Optional[Shape]:
        """Actually compute the output (called by get_output with cycle detection)."""
        if source_type == 'input_port':
            for port in self.design.input_ports:
                if port.port_id == source_id:
                    return port.shape
            return None

        elif source_type == 'machine':
            # Find the machine
            machine = None
            for m in self.design.machines:
                if m.node_id == source_id:
                    machine = m
                    break

            if machine is None:
                return None

            # Get inputs for this machine
            op = create_operation(machine.building_type)
            if op is None:
                return None

            inputs = []
            for input_idx in range(op.num_inputs):
                conn_key = (machine.node_id, input_idx)
                if conn_key in self.design.connections:
                    src_type, src_id, src_out = self.design.connections[conn_key]
                    input_shape = self.get_output(src_type, src_id, src_out)
                    inputs.append(input_shape)
                else:
                    inputs.append(None)

            # Execute the operation
            try:
                outputs = op.execute(*inputs)
                # Cache all outputs from this machine
                for i, out_shape in enumerate(outputs):
                    self._cache[('machine', source_id, i)] = out_shape

                if output_idx < len(outputs):
                    return outputs[output_idx]
            except Exception:
                pass

            return None

        return None

    def simulate(self) -> Dict[int, Optional[Shape]]:
        """
        Simulate the system and return output shapes for each output port.

        Returns:
            Dict mapping output port_id to the shape at that port
        """
        self._cache.clear()
        results = {}

        for port in self.design.output_ports:
            if port.source is not None:
                src_id, src_out = port.source
                # Determine source type
                src_type = 'machine'
                for inp in self.design.input_ports:
                    if inp.port_id == src_id:
                        src_type = 'input_port'
                        break
                shape = self.get_output(src_type, src_id, src_out)
                results[port.port_id] = shape
            else:
                results[port.port_id] = None

        return results


def evaluate_system(design: SystemDesign) -> Tuple[float, Dict[int, Optional[Shape]]]:
    """
    Evaluate how well a system design meets the output requirements.

    Returns:
        Tuple of (fitness_score, actual_outputs)
        fitness_score: 0.0 to 1.0 (1.0 = all outputs match)
    """
    simulator = SystemSimulator(design)
    actual_outputs = simulator.simulate()

    total_score = 0.0
    num_outputs = len(design.output_ports)

    if num_outputs == 0:
        return 0.0, actual_outputs

    for port in design.output_ports:
        actual = actual_outputs.get(port.port_id)
        expected = port.expected_shape

        if actual is None:
            # No output at all
            score = 0.0
        elif expected.is_empty():
            # Expected empty, got something
            score = 0.0 if not actual.is_empty() else 1.0
        else:
            # Compare shapes
            actual_code = actual.to_code()
            expected_code = expected.to_code()

            if actual_code == expected_code:
                score = 1.0
            else:
                # Partial match - compare quadrant by quadrant
                score = _compare_shapes(actual, expected)

        total_score += score

    return total_score / num_outputs, actual_outputs


def _compare_shapes(actual: Shape, expected: Shape) -> float:
    """Compare two shapes and return a similarity score (0.0 to 1.0)."""
    if actual.is_empty() and expected.is_empty():
        return 1.0
    if actual.is_empty() or expected.is_empty():
        return 0.0

    # Compare layer by layer, part by part
    max_layers = max(actual.num_layers, expected.num_layers)
    total_parts = max_layers * 4
    matching_parts = 0

    for layer_idx in range(max_layers):
        actual_layer = actual.get_layer(layer_idx)
        expected_layer = expected.get_layer(layer_idx)

        for part_idx in range(4):
            actual_part = actual_layer.get_part(part_idx) if actual_layer else None
            expected_part = expected_layer.get_part(part_idx) if expected_layer else None

            if actual_part is None and expected_part is None:
                matching_parts += 1
            elif actual_part is not None and expected_part is not None:
                if actual_part.is_empty() and expected_part.is_empty():
                    matching_parts += 1
                elif not actual_part.is_empty() and not expected_part.is_empty():
                    # Compare shape type and color
                    if (actual_part.shape_type == expected_part.shape_type and
                        actual_part.color == expected_part.color):
                        matching_parts += 1
                    elif actual_part.shape_type == expected_part.shape_type:
                        matching_parts += 0.5  # Shape matches but color doesn't

    return matching_parts / total_parts


class SystemSearch:
    """
    Searches for a system design that produces the desired outputs.

    Uses evolutionary search over machine topologies.
    """

    # Machine types that can be used in systems
    AVAILABLE_MACHINES = [
        BuildingType.CUTTER,
        BuildingType.HALF_CUTTER,
        BuildingType.ROTATOR_CW,
        BuildingType.ROTATOR_CCW,
        BuildingType.ROTATOR_180,
        BuildingType.STACKER,
        BuildingType.UNSTACKER,
        BuildingType.SWAPPER,
    ]

    def __init__(
        self,
        input_specs: List[Tuple[str, int, int, str]],  # (side, pos, floor, shape_code)
        output_specs: List[Tuple[str, int, int, str]],  # (side, pos, floor, shape_code)
        population_size: int = 30,
        max_machines: int = 10,
        mutation_rate: float = 0.3,
    ):
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.population_size = population_size
        self.max_machines = max_machines
        self.mutation_rate = mutation_rate

        # Parse input/output shapes
        self.input_ports = [
            InputPort(i, side, pos, floor, Shape.from_code(code))
            for i, (side, pos, floor, code) in enumerate(input_specs)
        ]
        self.output_ports = [
            OutputPort(i + 100, side, pos, floor, Shape.from_code(code))
            for i, (side, pos, floor, code) in enumerate(output_specs)
        ]

        # Analyze required transformations
        self._analyze_transformations()

        self.population: List[SystemDesign] = []
        self.best_design: Optional[SystemDesign] = None
        self.best_fitness: float = 0.0

    def _analyze_transformations(self):
        """Analyze what transformations might be needed."""
        # Check if any cutting is needed (output has fewer filled parts than input)
        self.needs_cutting = False
        self.needs_stacking = False
        self.needs_rotation = False

        # Count input parts
        input_parts = set()
        for port in self.input_ports:
            for layer_idx in range(port.shape.num_layers):
                layer = port.shape.get_layer(layer_idx)
                if layer:
                    for part_idx in range(4):
                        if not layer.get_part(part_idx).is_empty():
                            input_parts.add((layer_idx, part_idx))

        # Count output parts
        output_parts = set()
        for port in self.output_ports:
            for layer_idx in range(port.expected_shape.num_layers):
                layer = port.expected_shape.get_layer(layer_idx)
                if layer:
                    for part_idx in range(4):
                        if not layer.get_part(part_idx).is_empty():
                            output_parts.add((layer_idx, part_idx))

        # If outputs need fewer parts, we need cutting
        if len(output_parts) < len(input_parts):
            self.needs_cutting = True

        # If we have more outputs than inputs, we might need splitting
        if len(self.output_ports) > len(self.input_ports):
            self.needs_cutting = True

    def _create_random_design(self) -> SystemDesign:
        """Create a random system design."""
        design = SystemDesign(
            input_ports=self.input_ports.copy(),
            output_ports=[OutputPort(p.port_id, p.side, p.position, p.floor, p.expected_shape.copy())
                         for p in self.output_ports],
            machines=[],
            connections={}
        )

        # Add some random machines
        num_machines = random.randint(1, min(5, self.max_machines))

        # Prefer machines that match our analysis
        preferred_machines = list(self.AVAILABLE_MACHINES)
        if self.needs_cutting:
            preferred_machines = [BuildingType.CUTTER] * 3 + preferred_machines

        for i in range(num_machines):
            machine_type = random.choice(preferred_machines)
            machine = MachineNode(
                node_id=i,
                building_type=machine_type,
            )
            design.machines.append(machine)

        # Create random connections
        self._create_random_connections(design)

        return design

    def _create_seeded_design(self) -> SystemDesign:
        """Create a design seeded with likely-needed machines."""
        design = SystemDesign(
            input_ports=self.input_ports.copy(),
            output_ports=[OutputPort(p.port_id, p.side, p.position, p.floor, p.expected_shape.copy())
                         for p in self.output_ports],
            machines=[],
            connections={}
        )

        node_id = 0

        # If we need to split input to multiple outputs, add cutters
        if len(self.output_ports) > 1:
            # One cutter splits 1 -> 2
            # For 4 outputs from 1 input, need: 1 -> 2 -> 4 (two cutters)
            num_cutters_needed = 0
            outputs_needed = len(self.output_ports)
            sources = len(self.input_ports)
            while sources < outputs_needed:
                num_cutters_needed += 1
                sources *= 2

            for _ in range(min(num_cutters_needed, self.max_machines // 2)):
                machine = MachineNode(
                    node_id=node_id,
                    building_type=BuildingType.CUTTER,
                )
                design.machines.append(machine)
                node_id += 1

        # Connect machines in a logical way
        self._create_logical_connections(design)

        return design

    def _create_random_connections(self, design: SystemDesign):
        """Create random connections between components."""
        design.connections.clear()

        # Collect all available outputs
        # Format: list of (source_type, source_id, output_idx)
        available_outputs: List[Tuple[str, int, int]] = []

        # Input ports are sources
        for port in design.input_ports:
            available_outputs.append(('input_port', port.port_id, 0))

        # Connect machines in order (each machine can use outputs from previous machines)
        for machine in design.machines:
            op = create_operation(machine.building_type)
            if op is None:
                continue

            # Connect each input
            for input_idx in range(op.num_inputs):
                if available_outputs and random.random() < 0.8:
                    source = random.choice(available_outputs)
                    design.connections[(machine.node_id, input_idx)] = source

            # Add this machine's outputs to available
            for out_idx in range(op.num_outputs):
                available_outputs.append(('machine', machine.node_id, out_idx))

        # Connect output ports to available sources
        for port in design.output_ports:
            if available_outputs and random.random() < 0.9:
                source = random.choice(available_outputs)
                port.source = (source[1], source[2])
                # Determine if it's a machine or input port
                if source[0] == 'input_port':
                    port.source = (source[1], 0)
                else:
                    port.source = (source[1], source[2])

    def _create_logical_connections(self, design: SystemDesign):
        """Create logical connections based on machine topology."""
        design.connections.clear()

        # Track available outputs at each stage
        # Stage 0: input ports
        # Stage n: machines at that depth
        available_outputs: List[Tuple[str, int, int]] = []

        for port in design.input_ports:
            available_outputs.append(('input_port', port.port_id, 0))

        # Connect machines in chain (first cutter to input, second cutter to first cutter's outputs, etc.)
        cutters = [m for m in design.machines if m.building_type == BuildingType.CUTTER]

        if cutters and available_outputs:
            # First cutter connects to first input
            design.connections[(cutters[0].node_id, 0)] = available_outputs[0]

            # Add first cutter's outputs
            cutter_outputs = [
                ('machine', cutters[0].node_id, 0),
                ('machine', cutters[0].node_id, 1),
            ]

            # Second cutter (if exists) connects to first cutter's output
            if len(cutters) > 1:
                design.connections[(cutters[1].node_id, 0)] = cutter_outputs[0]
                # Add second cutter's outputs
                cutter_outputs = [
                    ('machine', cutters[1].node_id, 0),
                    ('machine', cutters[1].node_id, 1),
                    cutter_outputs[1],  # Keep the other output from first cutter
                ]

            available_outputs = cutter_outputs

        # Connect output ports
        for i, port in enumerate(design.output_ports):
            if i < len(available_outputs):
                src = available_outputs[i]
                port.source = (src[1], src[2])

    def _mutate(self, design: SystemDesign) -> SystemDesign:
        """Mutate a system design."""
        mutated = design.copy()

        mutation_type = random.random()

        if mutation_type < 0.2 and len(mutated.machines) < self.max_machines:
            # Add a machine
            new_id = max([m.node_id for m in mutated.machines] + [-1]) + 1
            machine_type = random.choice(self.AVAILABLE_MACHINES)
            mutated.machines.append(MachineNode(new_id, machine_type))

        elif mutation_type < 0.4 and len(mutated.machines) > 0:
            # Remove a machine
            to_remove = random.choice(mutated.machines)
            mutated.machines = [m for m in mutated.machines if m.node_id != to_remove.node_id]
            # Clean up connections
            mutated.connections = {
                k: v for k, v in mutated.connections.items()
                if k[0] != to_remove.node_id and
                   (v[0] != 'machine' or v[1] != to_remove.node_id)
            }
            for port in mutated.output_ports:
                if port.source and port.source[0] == to_remove.node_id:
                    port.source = None

        elif mutation_type < 0.6 and len(mutated.machines) > 0:
            # Change a machine type
            machine = random.choice(mutated.machines)
            machine.building_type = random.choice(self.AVAILABLE_MACHINES)

        elif mutation_type < 0.8:
            # Change a connection
            self._mutate_connection(mutated)

        else:
            # Reconnect randomly
            self._create_random_connections(mutated)

        return mutated

    def _mutate_connection(self, design: SystemDesign):
        """Mutate a single connection."""
        # Collect all available outputs
        available_outputs: List[Tuple[str, int, int]] = []
        for port in design.input_ports:
            available_outputs.append(('input_port', port.port_id, 0))
        for machine in design.machines:
            op = create_operation(machine.building_type)
            if op:
                for out_idx in range(op.num_outputs):
                    available_outputs.append(('machine', machine.node_id, out_idx))

        if not available_outputs:
            return

        # Either mutate a machine input or an output port
        if random.random() < 0.5 and design.machines:
            machine = random.choice(design.machines)
            op = create_operation(machine.building_type)
            if op and op.num_inputs > 0:
                input_idx = random.randint(0, op.num_inputs - 1)
                design.connections[(machine.node_id, input_idx)] = random.choice(available_outputs)
        else:
            port = random.choice(design.output_ports)
            src = random.choice(available_outputs)
            port.source = (src[1], src[2])

    def _crossover(self, parent1: SystemDesign, parent2: SystemDesign) -> SystemDesign:
        """Crossover two system designs."""
        # Take machines from both parents
        all_machines = parent1.machines + parent2.machines
        if len(all_machines) > self.max_machines:
            all_machines = random.sample(all_machines, self.max_machines)

        # Reassign node IDs
        new_machines = []
        for i, m in enumerate(all_machines):
            new_machines.append(MachineNode(i, m.building_type))

        child = SystemDesign(
            input_ports=self.input_ports.copy(),
            output_ports=[OutputPort(p.port_id, p.side, p.position, p.floor, p.expected_shape.copy())
                         for p in self.output_ports],
            machines=new_machines,
            connections={}
        )

        # Create new connections
        self._create_random_connections(child)

        return child

    def run(self, max_generations: int = 100, target_fitness: float = 1.0, verbose: bool = False) -> Optional[SystemDesign]:
        """
        Run the system search.

        Args:
            max_generations: Maximum number of generations
            target_fitness: Stop early if this fitness is achieved
            verbose: Print progress

        Returns:
            The best system design found, or None if no valid design found
        """
        # Initialize population
        self.population = []

        # Add seeded design
        self.population.append(self._create_seeded_design())

        # Fill rest with random
        while len(self.population) < self.population_size:
            self.population.append(self._create_random_design())

        for generation in range(max_generations):
            # Evaluate population
            fitness_scores = []
            for design in self.population:
                fitness, _ = evaluate_system(design)
                fitness_scores.append((fitness, design))

            # Sort by fitness
            fitness_scores.sort(key=lambda x: -x[0])

            # Update best
            if fitness_scores[0][0] > self.best_fitness:
                self.best_fitness = fitness_scores[0][0]
                self.best_design = fitness_scores[0][1].copy()

                if verbose:
                    print(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}")

            # Check for early termination
            if self.best_fitness >= target_fitness:
                if verbose:
                    print(f"Target fitness reached at generation {generation}")
                break

            # Select top half for reproduction
            survivors = [design for _, design in fitness_scores[:self.population_size // 2]]

            # Create next generation
            new_population = survivors.copy()

            while len(new_population) < self.population_size:
                if random.random() < 0.3 and len(survivors) >= 2:
                    # Crossover
                    p1, p2 = random.sample(survivors, 2)
                    child = self._crossover(p1, p2)
                else:
                    # Mutation
                    parent = random.choice(survivors)
                    child = self._mutate(parent)

                new_population.append(child)

            self.population = new_population

        return self.best_design


class SystemSearchSA(SystemSearch):
    """
    Simulated Annealing variant of system search.

    Uses temperature-based acceptance of worse solutions to escape local optima.
    """

    def __init__(
        self,
        input_specs: List[Tuple[str, int, int, str]],
        output_specs: List[Tuple[str, int, int, str]],
        max_machines: int = 10,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.01,
    ):
        super().__init__(input_specs, output_specs, population_size=1, max_machines=max_machines)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def run(self, max_generations: int = 100, target_fitness: float = 1.0, verbose: bool = False) -> Optional[SystemDesign]:
        """Run simulated annealing search."""
        import math

        # Start with seeded design
        current = self._create_seeded_design()
        current_fitness, _ = evaluate_system(current)

        self.best_design = current.copy()
        self.best_fitness = current_fitness

        temperature = self.initial_temp

        for iteration in range(max_generations):
            # Generate neighbor
            neighbor = self._mutate(current)
            neighbor_fitness, _ = evaluate_system(neighbor)

            # Calculate acceptance probability
            delta = neighbor_fitness - current_fitness

            if delta > 0:
                # Better solution - always accept
                current = neighbor
                current_fitness = neighbor_fitness
            elif temperature > self.min_temp:
                # Worse solution - accept with probability
                prob = math.exp(delta / temperature)
                if random.random() < prob:
                    current = neighbor
                    current_fitness = neighbor_fitness

            # Update best
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.best_design = current.copy()

                if verbose:
                    print(f"Iteration {iteration}: fitness={self.best_fitness:.4f}, temp={temperature:.4f}")

            # Check early termination
            if self.best_fitness >= target_fitness:
                if verbose:
                    print(f"Target reached at iteration {iteration}")
                break

            # Cool down
            temperature *= self.cooling_rate

        return self.best_design


class SystemSearchHybrid(SystemSearch):
    """
    Hybrid algorithm combining SA exploration with EA refinement.

    Phase 1: Use SA to explore broadly
    Phase 2: Use EA to refine best solutions
    """

    def __init__(
        self,
        input_specs: List[Tuple[str, int, int, str]],
        output_specs: List[Tuple[str, int, int, str]],
        population_size: int = 30,
        max_machines: int = 10,
        sa_fraction: float = 0.5,
    ):
        super().__init__(input_specs, output_specs, population_size, max_machines)
        self.sa_fraction = sa_fraction

    def run(self, max_generations: int = 100, target_fitness: float = 1.0, verbose: bool = False) -> Optional[SystemDesign]:
        """Run hybrid search."""
        sa_iterations = int(max_generations * self.sa_fraction)
        ea_generations = max_generations - sa_iterations

        if verbose:
            print(f"Phase 1: SA exploration ({sa_iterations} iterations)")

        # Phase 1: SA exploration
        sa = SystemSearchSA(
            input_specs=self.input_specs,
            output_specs=self.output_specs,
            max_machines=self.max_machines,
        )
        sa_result = sa.run(max_generations=sa_iterations, target_fitness=target_fitness, verbose=verbose)

        if sa.best_fitness >= target_fitness:
            self.best_design = sa.best_design
            self.best_fitness = sa.best_fitness
            return self.best_design

        if verbose:
            print(f"\nPhase 2: EA refinement ({ea_generations} generations)")

        # Phase 2: EA refinement
        # Seed population with SA result
        self.population = []
        if sa_result:
            self.population.append(sa_result.copy())
            # Add mutations of best
            for _ in range(self.population_size // 3):
                self.population.append(self._mutate(sa_result))

        # Fill rest with random
        while len(self.population) < self.population_size:
            self.population.append(self._create_random_design())

        self.best_design = sa.best_design
        self.best_fitness = sa.best_fitness

        # Run EA
        for generation in range(ea_generations):
            fitness_scores = []
            for design in self.population:
                fitness, _ = evaluate_system(design)
                fitness_scores.append((fitness, design))

            fitness_scores.sort(key=lambda x: -x[0])

            if fitness_scores[0][0] > self.best_fitness:
                self.best_fitness = fitness_scores[0][0]
                self.best_design = fitness_scores[0][1].copy()

                if verbose:
                    print(f"EA Gen {generation}: fitness={self.best_fitness:.4f}")

            if self.best_fitness >= target_fitness:
                break

            survivors = [d for _, d in fitness_scores[:self.population_size // 2]]
            new_population = survivors.copy()

            while len(new_population) < self.population_size:
                if random.random() < 0.3 and len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    child = self._crossover(p1, p2)
                else:
                    parent = random.choice(survivors)
                    child = self._mutate(parent)
                new_population.append(child)

            self.population = new_population

        return self.best_design


def find_system_for_transformation(
    input_specs: List[Tuple[str, int, int, str]],
    output_specs: List[Tuple[str, int, int, str]],
    max_generations: int = 100,
    algorithm: str = 'evolution',
    verbose: bool = False
) -> Tuple[Optional[SystemDesign], float]:
    """
    Find a system design that transforms inputs to outputs.

    Args:
        input_specs: List of (side, position, floor, shape_code) for inputs
        output_specs: List of (side, position, floor, shape_code) for outputs
        max_generations: Maximum search generations
        algorithm: 'evolution', 'sa', or 'hybrid'
        verbose: Print progress

    Returns:
        Tuple of (best_design, fitness)
    """
    if algorithm == 'sa':
        search = SystemSearchSA(
            input_specs=input_specs,
            output_specs=output_specs,
        )
    elif algorithm == 'hybrid':
        search = SystemSearchHybrid(
            input_specs=input_specs,
            output_specs=output_specs,
        )
    else:
        search = SystemSearch(
            input_specs=input_specs,
            output_specs=output_specs,
        )

    design = search.run(max_generations=max_generations, verbose=verbose)
    return design, search.best_fitness
