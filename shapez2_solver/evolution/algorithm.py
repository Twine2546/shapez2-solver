"""Main evolutionary algorithm implementation."""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..shapes.shape import Shape, Color
from ..simulator.design import Design
from ..foundations.foundation import Foundation, Port, PortDirection
from ..operations.base import Operation
from ..operations.cutter import CutOperation, HalfDestroyerOperation, SwapperOperation
from ..operations.rotator import RotateOperation
from ..operations.stacker import StackOperation, UnstackOperation
from ..operations.painter import PaintOperation
from .candidate import Candidate
from .fitness import FitnessFunction, ShapeMatchFitness
from .operators import (
    MutationOperator,
    CrossoverOperator,
    CompositeMutation,
    AddOperationMutation,
    RemoveOperationMutation,
    ModifyConnectionMutation,
    AddConnectionMutation,
    UniformCrossover,
    AVAILABLE_OPERATIONS,
)

# Operations that only need shape inputs (no color required)
SHAPE_ONLY_OPERATIONS: List[Type[Operation]] = [
    HalfDestroyerOperation,
    CutOperation,
    RotateOperation,
    UnstackOperation,
]

# Operations that need 2 shape inputs
TWO_INPUT_OPERATIONS: List[Type[Operation]] = [
    SwapperOperation,
    StackOperation,
]

# All available colors for painting
ALL_COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.CYAN,
              Color.MAGENTA, Color.YELLOW, Color.WHITE]


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary algorithm."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 3
    max_operations: int = 10
    parallel_evaluation: bool = True
    allowed_operations: List[Type[Operation]] = field(default_factory=lambda: AVAILABLE_OPERATIONS.copy())
    enable_painting: bool = False  # Enable painting operations with color inputs
    available_colors: List[Color] = field(default_factory=lambda: ALL_COLORS.copy())

    def __post_init__(self):
        if self.elitism_count >= self.population_size:
            self.elitism_count = max(1, self.population_size // 10)


class EvolutionaryAlgorithm:
    """Evolutionary algorithm for finding shape transformation solutions."""

    def __init__(
        self,
        foundation: Foundation,
        input_shapes: Dict[str, Union[Shape, Color]],
        expected_outputs: Dict[str, Optional[Union[Shape, Color]]],
        config: Optional[EvolutionConfig] = None,
        fitness_function: Optional[FitnessFunction] = None,
    ):
        """
        Initialize the evolutionary algorithm.

        Args:
            foundation: The foundation to use for designs
            input_shapes: Input shapes/colors
            expected_outputs: Expected output shapes/colors
            config: Evolution configuration
            fitness_function: Fitness function to use
        """
        self.foundation = foundation
        self.input_shapes = input_shapes
        self.expected_outputs = expected_outputs
        self.config = config or EvolutionConfig()
        self.fitness_function = fitness_function or ShapeMatchFitness()

        # Setup operators - weight towards adding operations and connections
        self.mutation_operator = CompositeMutation(
            operators=[
                AddOperationMutation(self.config.allowed_operations),
                RemoveOperationMutation(),
                ModifyConnectionMutation(),
                AddConnectionMutation(),
            ],
            weights=[0.4, 0.1, 0.2, 0.3]  # Favor adding operations and connections
        )
        self.crossover_operator = UniformCrossover()

        # State
        self.population: List[Candidate] = []
        self.generation = 0
        self.best_candidate: Optional[Candidate] = None
        self.history: List[Dict] = []

        # Callbacks
        self.on_generation: Optional[Callable[[int, List[Candidate]], None]] = None
        self.on_new_best: Optional[Callable[[Candidate], None]] = None

    def initialize_population(self) -> None:
        """Initialize the population with random designs."""
        self.population = []

        for _ in range(self.config.population_size):
            design = self._create_random_design()
            candidate = Candidate(design=design, generation=0)
            self.population.append(candidate)

        self._evaluate_population()

    def _create_random_design(self) -> Design:
        """Create a random design."""
        design = Design(self.foundation)

        # Add shape input ports
        shape_input_ids = []
        for name in self.input_shapes.keys():
            port = Port(PortDirection.WEST, floor=0, position=len(shape_input_ids))
            input_id = design.add_input(port)
            shape_input_ids.append(input_id)

        # Add color input ports if painting is enabled
        color_input_ids = []
        if self.config.enable_painting and self.config.available_colors:
            # Add one color input for each available color
            for i, color in enumerate(self.config.available_colors):
                port = Port(PortDirection.SOUTH, floor=0, position=i, is_input=True)
                color_id = design.add_input(port)
                color_input_ids.append(color_id)

        # Add output ports
        output_ids = []
        for name in self.expected_outputs.keys():
            port = Port(PortDirection.EAST, floor=0, position=len(output_ids))
            output_id = design.add_output(port)
            output_ids.append(output_id)

        # Build list of available operations
        if self.config.enable_painting:
            # Include all operations including painter
            shape_ops = SHAPE_ONLY_OPERATIONS + TWO_INPUT_OPERATIONS + [PaintOperation]
        else:
            # Filter to shape-only operations (exclude painter which needs color)
            shape_ops = [op for op in self.config.allowed_operations
                         if op in SHAPE_ONLY_OPERATIONS or op in TWO_INPUT_OPERATIONS]

        if not shape_ops:
            shape_ops = SHAPE_ONLY_OPERATIONS.copy()

        # Add random operations - use wider range to explore more complex solutions
        num_ops = random.randint(1, min(5, self.config.max_operations))
        op_ids = []
        painter_ops = []  # Track painter operations for color connections

        for _ in range(num_ops):
            op_class = random.choice(shape_ops)
            if op_class == RotateOperation:
                operation = op_class(steps=random.randint(1, 3))
            else:
                operation = op_class()
            op_id = design.add_operation(operation)
            op_ids.append(op_id)
            if op_class == PaintOperation:
                painter_ops.append(op_id)

        # Create valid connections
        self._create_valid_connections_with_colors(
            design, shape_input_ids, color_input_ids, op_ids, painter_ops, output_ids
        )

        return design

    def _create_random_connections(
        self,
        design: Design,
        input_ids: List[str],
        op_ids: List[str],
        output_ids: List[str]
    ) -> None:
        """Create random connections in a design."""
        # Build list of possible sources (input ports and operation outputs)
        sources = [(inp_id, 0) for inp_id in input_ids]
        for op_id in op_ids:
            op_node = design.get_node(op_id)
            if op_node:
                for i in range(op_node.operation.num_outputs):
                    sources.append((op_id, i))

        # Connect each operation's inputs
        for op_id in op_ids:
            op_node = design.get_node(op_id)
            if not op_node:
                continue

            for input_idx in range(op_node.operation.num_inputs):
                if sources:
                    source = random.choice(sources)
                    design.connect(source[0], source[1], op_id, input_idx)

        # Connect outputs
        for out_id in output_ids:
            if sources:
                source = random.choice(sources)
                design.connect(source[0], source[1], out_id, 0)

    def _create_valid_connections(
        self,
        design: Design,
        input_ids: List[str],
        op_ids: List[str],
        output_ids: List[str]
    ) -> None:
        """Create valid connections ensuring proper data flow."""
        # Track available signal sources (node_id, output_index)
        available_sources = [(inp_id, 0) for inp_id in input_ids]

        # Process operations in order, connecting inputs and adding outputs
        for op_id in op_ids:
            op_node = design.get_node(op_id)
            if not op_node:
                continue

            op = op_node.operation

            # Connect required inputs from available sources
            for input_idx in range(op.num_inputs):
                if available_sources:
                    # Prefer using the most recent source
                    source = random.choice(available_sources)
                    design.connect(source[0], source[1], op_id, input_idx)

            # Add this operation's outputs to available sources
            for output_idx in range(op.num_outputs):
                available_sources.append((op_id, output_idx))

        # Connect to output ports - use the last available source for direct path
        for out_id in output_ids:
            if available_sources:
                # Try to use the most recently added source (likely operation output)
                # or fall back to input if no operations
                source = available_sources[-1] if len(available_sources) > len(input_ids) else available_sources[0]
                design.connect(source[0], source[1], out_id, 0)

    def _create_valid_connections_with_colors(
        self,
        design: Design,
        shape_input_ids: List[str],
        color_input_ids: List[str],
        op_ids: List[str],
        painter_ops: List[str],
        output_ids: List[str]
    ) -> None:
        """Create valid connections including color inputs for painters."""
        # Track available shape sources (node_id, output_index)
        shape_sources = [(inp_id, 0) for inp_id in shape_input_ids]

        # Process operations in order
        for op_id in op_ids:
            op_node = design.get_node(op_id)
            if not op_node:
                continue

            op = op_node.operation

            if op_id in painter_ops:
                # Painter needs shape input (index 0) and color input (index 1)
                if shape_sources:
                    source = random.choice(shape_sources)
                    design.connect(source[0], source[1], op_id, 0)  # Shape input

                if color_input_ids:
                    color_source = random.choice(color_input_ids)
                    design.connect(color_source, 0, op_id, 1)  # Color input
            else:
                # Regular operation - connect shape inputs
                for input_idx in range(op.num_inputs):
                    if shape_sources:
                        source = random.choice(shape_sources)
                        design.connect(source[0], source[1], op_id, input_idx)

            # Add operation's outputs to available shape sources
            for output_idx in range(op.num_outputs):
                shape_sources.append((op_id, output_idx))

        # Connect to output ports
        for out_id in output_ids:
            if shape_sources:
                source = shape_sources[-1] if len(shape_sources) > len(shape_input_ids) else shape_sources[0]
                design.connect(source[0], source[1], out_id, 0)

    def _evaluate_population(self) -> None:
        """Evaluate fitness for all candidates in the population."""
        # Create input/output mappings
        inputs = {}
        for i, (name, value) in enumerate(self.input_shapes.items()):
            inputs[f"in_{i}"] = value

        # Add color inputs if painting is enabled
        if self.config.enable_painting and self.config.available_colors:
            shape_input_count = len(self.input_shapes)
            for j, color in enumerate(self.config.available_colors):
                inputs[f"in_{shape_input_count + j}"] = color

        expected = {}
        for i, (name, value) in enumerate(self.expected_outputs.items()):
            expected[f"out_{i}"] = value

        if self.config.parallel_evaluation:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self.fitness_function.evaluate,
                        candidate.design,
                        inputs,
                        expected
                    ): candidate
                    for candidate in self.population
                }

                for future in as_completed(futures):
                    candidate = futures[future]
                    try:
                        candidate.fitness = future.result()
                    except Exception:
                        candidate.fitness = 0.0
        else:
            for candidate in self.population:
                try:
                    candidate.fitness = self.fitness_function.evaluate(
                        candidate.design, inputs, expected
                    )
                except Exception:
                    candidate.fitness = 0.0

        # Update best candidate
        sorted_pop = sorted(self.population, reverse=True)
        if sorted_pop and (
            self.best_candidate is None or sorted_pop[0].fitness > self.best_candidate.fitness
        ):
            self.best_candidate = sorted_pop[0].copy()
            if self.on_new_best:
                self.on_new_best(self.best_candidate)

    def evolve_generation(self) -> None:
        """Evolve one generation."""
        self.generation += 1

        # Sort population by fitness
        sorted_pop = sorted(self.population, reverse=True)

        # Elitism: keep best candidates
        new_population = [c.copy() for c in sorted_pop[:self.config.elitism_count]]

        # Inject fresh diversity every 10 generations (10% of population)
        diversity_count = 0
        if self.generation % 10 == 0:
            diversity_count = max(1, self.config.population_size // 10)
            for _ in range(diversity_count):
                design = self._create_random_design()
                candidate = Candidate(design=design, generation=self.generation)
                new_population.append(candidate)

        # Fill rest of population
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover
            if random.random() < self.config.crossover_rate:
                child_design = self.crossover_operator.crossover(
                    parent1.design, parent2.design
                )
            else:
                child_design = parent1.design.copy()

            # Mutation
            child_design = self.mutation_operator.mutate(
                child_design, self.config.mutation_rate
            )

            child = Candidate(design=child_design, generation=self.generation)
            new_population.append(child)

        self.population = new_population
        self._evaluate_population()

        # Record history
        best_fitness = max(c.fitness for c in self.population)
        avg_fitness = sum(c.fitness for c in self.population) / len(self.population)
        self.history.append({
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
        })

        # Callback
        if self.on_generation:
            self.on_generation(self.generation, self.population)

    def _tournament_select(self) -> Candidate:
        """Select a candidate using tournament selection."""
        tournament = random.sample(
            self.population,
            min(self.config.tournament_size, len(self.population))
        )
        return max(tournament, key=lambda c: c.fitness)

    def run(self, callback: Optional[Callable[[int, float], bool]] = None) -> Candidate:
        """
        Run the evolutionary algorithm.

        Args:
            callback: Optional callback called each generation with (generation, best_fitness).
                     Return False to stop early.

        Returns:
            The best candidate found
        """
        self.initialize_population()

        for _ in range(self.config.generations):
            self.evolve_generation()

            if callback:
                should_continue = callback(self.generation, self.best_candidate.fitness if self.best_candidate else 0.0)
                if not should_continue:
                    break

            # Early stopping if perfect solution found
            if self.best_candidate and self.best_candidate.fitness >= 1.0:
                break

        return self.best_candidate

    def get_statistics(self) -> Dict:
        """Get statistics about the evolution run."""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_candidate.fitness if self.best_candidate else 0.0,
            'history': self.history,
        }
