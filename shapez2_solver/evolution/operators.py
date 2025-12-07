"""Genetic operators for mutation and crossover."""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from ..simulator.design import Design, Connection, OperationNode
from ..operations.base import Operation, OperationType
from ..operations.cutter import CutOperation, HalfDestroyerOperation, SwapperOperation
from ..operations.rotator import RotateOperation
from ..operations.stacker import StackOperation, UnstackOperation
from ..operations.painter import PaintOperation
from ..foundations.foundation import Foundation, Port


# Available operation types for random generation
AVAILABLE_OPERATIONS: List[Type[Operation]] = [
    HalfDestroyerOperation,
    CutOperation,
    SwapperOperation,
    RotateOperation,
    StackOperation,
    UnstackOperation,
    PaintOperation,
]


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""

    @abstractmethod
    def mutate(self, design: Design, mutation_rate: float) -> Design:
        """
        Apply mutation to a design.

        Args:
            design: The design to mutate
            mutation_rate: Probability of mutation (0.0 to 1.0)

        Returns:
            A mutated copy of the design
        """
        pass


class AddOperationMutation(MutationOperator):
    """Mutation that adds a random operation and connects it."""

    def __init__(self, allowed_operations: Optional[List[Type[Operation]]] = None):
        self.allowed_operations = allowed_operations or AVAILABLE_OPERATIONS

    def mutate(self, design: Design, mutation_rate: float) -> Design:
        if random.random() > mutation_rate:
            return design.copy()

        new_design = design.copy()

        # Choose a random operation type
        op_class = random.choice(self.allowed_operations)

        # Handle operations with parameters
        if op_class == RotateOperation:
            operation = op_class(steps=random.randint(1, 3))
        else:
            operation = op_class()

        # Add the operation
        op_id = new_design.add_operation(operation)

        # Connect the new operation to existing graph
        # Get all possible sources for inputs
        possible_sources = []
        for inp in new_design.inputs:
            possible_sources.append((inp.node_id, 0))
        for op in new_design.operations:
            if op.node_id != op_id:  # Don't connect to self
                for i in range(op.operation.num_outputs):
                    possible_sources.append((op.node_id, i))

        # Connect inputs of new operation
        if possible_sources:
            op_node = new_design.get_node(op_id)
            if op_node:
                for input_idx in range(op_node.operation.num_inputs):
                    source = random.choice(possible_sources)
                    new_design.connect(source[0], source[1], op_id, input_idx)

        # Randomly connect this operation's output to something
        possible_targets = []
        for op in new_design.operations:
            if op.node_id != op_id:
                for i in range(op.operation.num_inputs):
                    possible_targets.append((op.node_id, i))
        for out in new_design.outputs:
            possible_targets.append((out.node_id, 0))

        if possible_targets and random.random() < 0.5:
            target = random.choice(possible_targets)
            op_node = new_design.get_node(op_id)
            if op_node:
                output_idx = random.randint(0, op_node.operation.num_outputs - 1)
                new_design.connect(op_id, output_idx, target[0], target[1])

        return new_design


class RemoveOperationMutation(MutationOperator):
    """Mutation that removes a random operation."""

    def mutate(self, design: Design, mutation_rate: float) -> Design:
        if random.random() > mutation_rate:
            return design.copy()

        if not design.operations:
            return design.copy()

        new_design = design.copy()

        # Choose a random operation to remove
        op_idx = random.randint(0, len(new_design.operations) - 1)
        removed_op = new_design.operations[op_idx]

        # Remove connections involving this operation
        new_design.connections = [
            c for c in new_design.connections
            if c.source_id != removed_op.node_id and c.target_id != removed_op.node_id
        ]

        # Remove the operation
        new_design.operations.pop(op_idx)
        new_design._rebuild_lookup()

        return new_design


class ModifyConnectionMutation(MutationOperator):
    """Mutation that modifies a random connection."""

    def mutate(self, design: Design, mutation_rate: float) -> Design:
        if random.random() > mutation_rate:
            return design.copy()

        if not design.connections:
            return design.copy()

        new_design = design.copy()

        # Choose a random connection to modify
        conn_idx = random.randint(0, len(new_design.connections) - 1)
        conn = new_design.connections[conn_idx]

        # Get all possible sources and targets
        possible_sources = []
        for inp in new_design.inputs:
            possible_sources.append((inp.node_id, 0))
        for op in new_design.operations:
            for i in range(op.operation.num_outputs):
                possible_sources.append((op.node_id, i))

        possible_targets = []
        for op in new_design.operations:
            for i in range(op.operation.num_inputs):
                possible_targets.append((op.node_id, i))
        for out in new_design.outputs:
            possible_targets.append((out.node_id, 0))

        # Randomly change source or target
        if random.random() < 0.5 and possible_sources:
            new_source = random.choice(possible_sources)
            new_design.connections[conn_idx] = Connection(
                new_source[0], new_source[1], conn.target_id, conn.target_input_idx
            )
        elif possible_targets:
            new_target = random.choice(possible_targets)
            new_design.connections[conn_idx] = Connection(
                conn.source_id, conn.source_output_idx, new_target[0], new_target[1]
            )

        return new_design


class AddConnectionMutation(MutationOperator):
    """Mutation that adds a random connection."""

    def mutate(self, design: Design, mutation_rate: float) -> Design:
        if random.random() > mutation_rate:
            return design.copy()

        new_design = design.copy()

        # Get all possible sources
        possible_sources = []
        for inp in new_design.inputs:
            possible_sources.append((inp.node_id, 0))
        for op in new_design.operations:
            for i in range(op.operation.num_outputs):
                possible_sources.append((op.node_id, i))

        # Get all possible targets
        possible_targets = []
        for op in new_design.operations:
            for i in range(op.operation.num_inputs):
                possible_targets.append((op.node_id, i))
        for out in new_design.outputs:
            possible_targets.append((out.node_id, 0))

        if possible_sources and possible_targets:
            source = random.choice(possible_sources)
            target = random.choice(possible_targets)
            new_design.connect(source[0], source[1], target[0], target[1])

        return new_design


class CompositeMutation(MutationOperator):
    """Combines multiple mutation operators."""

    def __init__(self, operators: List[MutationOperator], weights: Optional[List[float]] = None):
        self.operators = operators
        self.weights = weights or [1.0 / len(operators)] * len(operators)

    def mutate(self, design: Design, mutation_rate: float) -> Design:
        # Choose an operator based on weights
        operator = random.choices(self.operators, weights=self.weights)[0]
        return operator.mutate(design, mutation_rate)


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def crossover(self, parent1: Design, parent2: Design) -> Design:
        """
        Create a child design from two parents.

        Args:
            parent1: First parent design
            parent2: Second parent design

        Returns:
            A new child design
        """
        pass


class UniformCrossover(CrossoverOperator):
    """Crossover that uniformly selects operations from both parents."""

    def crossover(self, parent1: Design, parent2: Design) -> Design:
        # Use parent1's foundation
        child = Design(parent1.foundation)

        # Copy inputs and outputs from parent1
        for inp in parent1.inputs:
            child.add_input(inp.port)
        for out in parent1.outputs:
            child.add_output(out.port)

        # Combine operations from both parents
        all_operations = []
        for op in parent1.operations:
            all_operations.append(op.operation)
        for op in parent2.operations:
            all_operations.append(op.operation)

        # Select operations - prefer at least as many as the larger parent
        if all_operations:
            min_ops = max(len(parent1.operations), len(parent2.operations))
            max_ops = len(all_operations)
            # Bias towards keeping more operations
            num_ops = random.randint(min_ops, max_ops) if min_ops <= max_ops else max_ops
            num_ops = max(1, num_ops)  # At least 1 operation
            selected = random.sample(all_operations, min(num_ops, len(all_operations)))
            for op in selected:
                child.add_operation(op)

        # Generate random connections
        self._generate_random_connections(child)

        return child

    def _generate_random_connections(self, design: Design) -> None:
        """Generate random valid connections for a design."""
        # Connect inputs to operations
        for inp in design.inputs:
            if design.operations:
                target_op = random.choice(design.operations)
                target_input = random.randint(0, target_op.operation.num_inputs - 1)
                design.connect(inp.node_id, 0, target_op.node_id, target_input)

        # Connect operations to outputs
        for out in design.outputs:
            sources = []
            for inp in design.inputs:
                sources.append((inp.node_id, 0))
            for op in design.operations:
                for i in range(op.operation.num_outputs):
                    sources.append((op.node_id, i))

            if sources:
                source = random.choice(sources)
                design.connect(source[0], source[1], out.node_id, 0)
