"""Simulator for executing designs."""

from typing import Dict, List, Optional, Tuple, Union
from collections import deque

from ..shapes.shape import Shape, Color
from .design import Design, OperationNode, InputNode, OutputNode, Connection


class SimulatorResult:
    """Result of a simulation run."""

    def __init__(self):
        self.outputs: Dict[str, Optional[Union[Shape, Color]]] = {}
        self.node_values: Dict[str, List[Optional[Union[Shape, Color]]]] = {}
        self.success: bool = True
        self.error: Optional[str] = None

    def get_output(self, output_id: str) -> Optional[Union[Shape, Color]]:
        """Get the value at an output port."""
        return self.outputs.get(output_id)

    def get_all_outputs(self) -> List[Optional[Union[Shape, Color]]]:
        """Get all output values in order."""
        return list(self.outputs.values())


class Simulator:
    """Simulator for executing designs with input shapes."""

    def __init__(self, design: Design):
        """
        Initialize the simulator with a design.

        Args:
            design: The design to simulate
        """
        self.design = design
        self._execution_order: Optional[List[str]] = None

    def execute(
        self,
        inputs: Dict[str, Union[Shape, Color]]
    ) -> SimulatorResult:
        """
        Execute the design with the given inputs.

        Args:
            inputs: Dictionary mapping input node IDs to their values

        Returns:
            SimulatorResult containing output values
        """
        result = SimulatorResult()

        # Validate design first
        valid, errors = self.design.validate()
        if not valid:
            result.success = False
            result.error = "; ".join(errors)
            return result

        # Build execution order if not cached
        if self._execution_order is None:
            self._execution_order = self._topological_sort()

        # Initialize node values
        node_values: Dict[str, List[Optional[Union[Shape, Color]]]] = {}

        # Set input values
        for input_node in self.design.inputs:
            value = inputs.get(input_node.node_id)
            node_values[input_node.node_id] = [value]

        # Execute nodes in order
        try:
            for node_id in self._execution_order:
                node = self.design.get_node(node_id)

                if isinstance(node, InputNode):
                    continue  # Already set

                if isinstance(node, OperationNode):
                    # Gather inputs for this operation
                    op_inputs = self._gather_inputs(node_id, node_values)

                    # Execute operation
                    op_outputs = node.operation.execute(*op_inputs)

                    # Store outputs
                    node_values[node_id] = list(op_outputs)

                elif isinstance(node, OutputNode):
                    # Get value from incoming connection
                    incoming = self.design.get_incoming_connections(node_id)
                    if incoming:
                        conn = incoming[0]
                        source_values = node_values.get(conn.source_id, [])
                        if conn.source_output_idx < len(source_values):
                            node_values[node_id] = [source_values[conn.source_output_idx]]
                        else:
                            node_values[node_id] = [None]
                    else:
                        node_values[node_id] = [None]

        except Exception as e:
            result.success = False
            result.error = str(e)
            return result

        # Collect output values
        for output_node in self.design.outputs:
            values = node_values.get(output_node.node_id, [None])
            result.outputs[output_node.node_id] = values[0] if values else None

        result.node_values = node_values
        return result

    def _gather_inputs(
        self,
        node_id: str,
        node_values: Dict[str, List[Optional[Union[Shape, Color]]]]
    ) -> List[Optional[Union[Shape, Color]]]:
        """Gather input values for a node from its incoming connections."""
        node = self.design.get_node(node_id)
        if not isinstance(node, OperationNode):
            return []

        # Get expected number of inputs
        num_inputs = node.operation.num_inputs
        inputs: List[Optional[Union[Shape, Color]]] = [None] * num_inputs

        # Get incoming connections
        for conn in self.design.get_incoming_connections(node_id):
            source_values = node_values.get(conn.source_id, [])
            if conn.source_output_idx < len(source_values):
                value = source_values[conn.source_output_idx]
            else:
                value = None

            if conn.target_input_idx < num_inputs:
                inputs[conn.target_input_idx] = value

        return inputs

    def _topological_sort(self) -> List[str]:
        """
        Topologically sort nodes for execution order.

        Returns a list of node IDs in execution order.
        """
        # Build adjacency and in-degree
        in_degree: Dict[str, int] = {}
        adjacency: Dict[str, List[str]] = {}

        # Initialize all nodes
        for node in self.design.inputs:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []
        for node in self.design.operations:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []
        for node in self.design.outputs:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []

        # Build graph from connections
        for conn in self.design.connections:
            if conn.source_id in adjacency:
                adjacency[conn.source_id].append(conn.target_id)
            if conn.target_id in in_degree:
                in_degree[conn.target_id] += 1

        # Kahn's algorithm
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node_id = queue.popleft()
            order.append(node_id)

            for neighbor in adjacency.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def reset(self) -> None:
        """Reset the simulator state."""
        self._execution_order = None
