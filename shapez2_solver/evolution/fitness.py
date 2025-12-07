"""Fitness functions for evaluating candidate solutions."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from ..shapes.shape import Shape, Color
from ..simulator.design import Design
from ..simulator.simulator import Simulator


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""

    @abstractmethod
    def evaluate(
        self,
        design: Design,
        inputs: Dict[str, Union[Shape, Color]],
        expected_outputs: Dict[str, Optional[Union[Shape, Color]]]
    ) -> float:
        """
        Evaluate the fitness of a design.

        Args:
            design: The design to evaluate
            inputs: Input values for the design
            expected_outputs: Expected output values

        Returns:
            Fitness score (higher is better, 1.0 = perfect)
        """
        pass


class ShapeMatchFitness(FitnessFunction):
    """Fitness function that measures how well output shapes match expected shapes."""

    def __init__(self, partial_match_weight: float = 0.5):
        """
        Initialize the fitness function.

        Args:
            partial_match_weight: Weight for partial matches (0.0 to 1.0)
        """
        self.partial_match_weight = partial_match_weight

    def evaluate(
        self,
        design: Design,
        inputs: Dict[str, Union[Shape, Color]],
        expected_outputs: Dict[str, Optional[Union[Shape, Color]]]
    ) -> float:
        """Evaluate fitness based on shape matching."""
        # Run simulation
        simulator = Simulator(design)
        result = simulator.execute(inputs)

        if not result.success:
            return 0.0

        # Compare outputs
        total_score = 0.0
        num_outputs = len(expected_outputs)

        if num_outputs == 0:
            return 0.0

        for output_id, expected in expected_outputs.items():
            actual = result.get_output(output_id)
            score = self._compare_values(actual, expected)
            total_score += score

        return total_score / num_outputs

    def _compare_values(
        self,
        actual: Optional[Union[Shape, Color]],
        expected: Optional[Union[Shape, Color]]
    ) -> float:
        """Compare two values and return a similarity score."""
        if actual is None and expected is None:
            return 1.0

        if actual is None or expected is None:
            return 0.0

        if isinstance(expected, Shape) and isinstance(actual, Shape):
            return self._compare_shapes(actual, expected)

        if isinstance(expected, Color) and isinstance(actual, Color):
            return 1.0 if actual == expected else 0.0

        return 0.0

    def _compare_shapes(self, actual: Shape, expected: Shape) -> float:
        """Compare two shapes and return a similarity score."""
        # Exact match
        if actual.to_code() == expected.to_code():
            return 1.0

        # Partial matching
        if not self.partial_match_weight:
            return 0.0

        # Compare layer by layer
        max_layers = max(actual.num_layers, expected.num_layers)
        if max_layers == 0:
            return 1.0

        layer_scores = []
        for i in range(max_layers):
            actual_layer = actual.get_layer(i)
            expected_layer = expected.get_layer(i)

            if actual_layer is None and expected_layer is None:
                layer_scores.append(1.0)
            elif actual_layer is None or expected_layer is None:
                layer_scores.append(0.0)
            else:
                layer_scores.append(self._compare_layers(actual_layer, expected_layer))

        avg_layer_score = sum(layer_scores) / len(layer_scores)
        return self.partial_match_weight * avg_layer_score

    def _compare_layers(self, actual, expected) -> float:
        """Compare two layers and return a similarity score."""
        if actual.num_parts != expected.num_parts:
            return 0.0

        matching_parts = 0
        for i in range(actual.num_parts):
            actual_part = actual.get_part(i)
            expected_part = expected.get_part(i)

            if actual_part.to_code() == expected_part.to_code():
                matching_parts += 1

        return matching_parts / actual.num_parts


class ComplexityPenaltyFitness(FitnessFunction):
    """Fitness function that penalizes complex designs."""

    def __init__(
        self,
        base_fitness: FitnessFunction,
        operation_penalty: float = 0.01,
        connection_penalty: float = 0.005
    ):
        """
        Initialize the fitness function.

        Args:
            base_fitness: The base fitness function to use
            operation_penalty: Penalty per operation (subtracted from fitness)
            connection_penalty: Penalty per connection
        """
        self.base_fitness = base_fitness
        self.operation_penalty = operation_penalty
        self.connection_penalty = connection_penalty

    def evaluate(
        self,
        design: Design,
        inputs: Dict[str, Union[Shape, Color]],
        expected_outputs: Dict[str, Optional[Union[Shape, Color]]]
    ) -> float:
        """Evaluate fitness with complexity penalty."""
        base_score = self.base_fitness.evaluate(design, inputs, expected_outputs)

        # Apply penalties
        op_penalty = len(design.operations) * self.operation_penalty
        conn_penalty = len(design.connections) * self.connection_penalty

        # Ensure fitness stays positive
        penalized = max(0.0, base_score - op_penalty - conn_penalty)

        return penalized


class ParsimonyFitness(FitnessFunction):
    """
    Fitness function with parsimony pressure.

    Correct solutions are always preferred over incorrect ones.
    Among equally correct solutions, simpler ones score higher.
    """

    def __init__(
        self,
        base_fitness: Optional[FitnessFunction] = None,
        max_operations: int = 10,
        simplicity_weight: float = 0.1
    ):
        """
        Initialize the fitness function.

        Args:
            base_fitness: The base fitness function (default: ShapeMatchFitness)
            max_operations: Maximum expected operations (for normalization)
            simplicity_weight: Weight for simplicity bonus (0.0 to 1.0)
        """
        self.base_fitness = base_fitness or ShapeMatchFitness()
        self.max_operations = max_operations
        self.simplicity_weight = simplicity_weight

    def evaluate(
        self,
        design: Design,
        inputs: Dict[str, Union[Shape, Color]],
        expected_outputs: Dict[str, Optional[Union[Shape, Color]]]
    ) -> float:
        """
        Evaluate fitness with parsimony pressure.

        Score = correctness + (simplicity_bonus * simplicity_weight)

        Correctness is in range [0, 1].
        Simplicity bonus is in range [0, 1] based on operation count.

        This ensures correct solutions always beat incorrect ones,
        but simpler correct solutions beat complex correct ones.
        """
        correctness = self.base_fitness.evaluate(design, inputs, expected_outputs)

        # Calculate simplicity bonus (fewer ops = higher bonus)
        num_ops = len(design.operations)
        if num_ops >= self.max_operations:
            simplicity_bonus = 0.0
        else:
            simplicity_bonus = 1.0 - (num_ops / self.max_operations)

        # Combine: correctness is primary, simplicity is secondary
        # Scale simplicity to be a small fraction so it doesn't override correctness
        score = correctness + (simplicity_bonus * self.simplicity_weight)

        return score


class SolutionMinimizer:
    """
    Post-processing to find minimal solutions.

    After finding a working solution, tries to remove operations
    one by one while maintaining correctness.
    """

    def __init__(self, fitness_function: Optional[FitnessFunction] = None):
        self.fitness_function = fitness_function or ShapeMatchFitness()

    def minimize(
        self,
        design: Design,
        inputs: Dict[str, Union[Shape, Color]],
        expected_outputs: Dict[str, Optional[Union[Shape, Color]]],
        threshold: float = 0.999
    ) -> Design:
        """
        Minimize a solution by removing unnecessary operations.

        Args:
            design: The solution design to minimize
            inputs: Input values
            expected_outputs: Expected outputs
            threshold: Fitness threshold for valid solution (default: 0.999)

        Returns:
            Minimized design
        """
        current = design.copy()
        current_fitness = self.fitness_function.evaluate(current, inputs, expected_outputs)

        if current_fitness < threshold:
            return current  # Not a valid solution to minimize

        improved = True
        while improved:
            improved = False

            # Try removing each operation
            for i in range(len(current.operations) - 1, -1, -1):
                candidate = self._remove_operation(current, i)
                if candidate is None:
                    continue

                candidate_fitness = self.fitness_function.evaluate(
                    candidate, inputs, expected_outputs
                )

                if candidate_fitness >= threshold:
                    current = candidate
                    improved = True
                    break  # Restart from beginning

        return current

    def _remove_operation(self, design: Design, op_index: int) -> Optional[Design]:
        """
        Remove an operation and rewire connections.

        Returns None if removal would break the design.
        """
        if op_index >= len(design.operations):
            return None

        new_design = design.copy()
        removed_op = new_design.operations[op_index]
        removed_id = removed_op.node_id

        # Find incoming connections to this operation
        incoming = [c for c in new_design.connections if c.target_id == removed_id]

        # Find outgoing connections from this operation
        outgoing = [c for c in new_design.connections if c.source_id == removed_id]

        # If no outgoing connections, just remove
        if not outgoing:
            new_design.connections = [
                c for c in new_design.connections
                if c.source_id != removed_id and c.target_id != removed_id
            ]
            new_design.operations.pop(op_index)
            new_design._rebuild_lookup()
            return new_design

        # If no incoming connections, can't rewire properly
        if not incoming:
            new_design.connections = [
                c for c in new_design.connections
                if c.source_id != removed_id and c.target_id != removed_id
            ]
            new_design.operations.pop(op_index)
            new_design._rebuild_lookup()
            return new_design

        # Rewire: connect each outgoing target to the first incoming source
        primary_source = incoming[0]
        new_connections = []

        for conn in new_design.connections:
            if conn.source_id == removed_id:
                # Rewire to primary source
                from ..simulator.design import Connection
                new_conn = Connection(
                    primary_source.source_id,
                    primary_source.source_output_idx,
                    conn.target_id,
                    conn.target_input_idx
                )
                new_connections.append(new_conn)
            elif conn.target_id == removed_id:
                # Skip incoming connections to removed op
                pass
            else:
                new_connections.append(conn)

        new_design.connections = new_connections
        new_design.operations.pop(op_index)
        new_design._rebuild_lookup()

        return new_design
