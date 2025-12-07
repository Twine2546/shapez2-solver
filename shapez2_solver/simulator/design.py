"""Design representation for the simulator."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum, auto

from ..shapes.shape import Shape, Color
from ..operations.base import Operation
from ..foundations.foundation import Foundation, Port


class NodeType(Enum):
    """Types of nodes in a design."""
    INPUT = auto()
    OUTPUT = auto()
    OPERATION = auto()


@dataclass
class OperationNode:
    """A node representing an operation in the design."""
    node_id: str
    operation: Operation
    position: Tuple[int, int] = (0, 0)  # Grid position on foundation
    floor: int = 0

    def __hash__(self):
        return hash(self.node_id)


@dataclass
class InputNode:
    """A node representing an input port."""
    node_id: str
    port: Port

    def __hash__(self):
        return hash(self.node_id)


@dataclass
class OutputNode:
    """A node representing an output port."""
    node_id: str
    port: Port

    def __hash__(self):
        return hash(self.node_id)


@dataclass
class Connection:
    """A connection between nodes."""
    source_id: str
    source_output_idx: int  # Which output of the source
    target_id: str
    target_input_idx: int  # Which input of the target

    def __hash__(self):
        return hash((self.source_id, self.source_output_idx,
                    self.target_id, self.target_input_idx))


@dataclass
class Design:
    """A complete design representing a solution."""
    foundation: Foundation
    operations: List[OperationNode] = field(default_factory=list)
    inputs: List[InputNode] = field(default_factory=list)
    outputs: List[OutputNode] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)

    def __post_init__(self):
        """Build lookup tables."""
        self._node_lookup: Dict[str, Any] = {}
        self._rebuild_lookup()

    def _rebuild_lookup(self):
        """Rebuild the node lookup table."""
        self._node_lookup = {}
        for node in self.operations:
            self._node_lookup[node.node_id] = node
        for node in self.inputs:
            self._node_lookup[node.node_id] = node
        for node in self.outputs:
            self._node_lookup[node.node_id] = node

    def add_operation(self, operation: Operation, position: Tuple[int, int] = (0, 0), floor: int = 0) -> str:
        """Add an operation to the design."""
        node_id = f"op_{len(self.operations)}"
        node = OperationNode(node_id, operation, position, floor)
        self.operations.append(node)
        self._node_lookup[node_id] = node
        return node_id

    def add_input(self, port: Port) -> str:
        """Add an input port to the design."""
        node_id = f"in_{len(self.inputs)}"
        node = InputNode(node_id, port)
        self.inputs.append(node)
        self._node_lookup[node_id] = node
        return node_id

    def add_output(self, port: Port) -> str:
        """Add an output port to the design."""
        node_id = f"out_{len(self.outputs)}"
        node = OutputNode(node_id, port)
        self.outputs.append(node)
        self._node_lookup[node_id] = node
        return node_id

    def connect(
        self,
        source_id: str,
        source_output: int,
        target_id: str,
        target_input: int
    ) -> None:
        """Create a connection between nodes."""
        conn = Connection(source_id, source_output, target_id, target_input)
        self.connections.append(conn)

    def get_node(self, node_id: str) -> Optional[Any]:
        """Get a node by ID."""
        return self._node_lookup.get(node_id)

    def get_incoming_connections(self, node_id: str) -> List[Connection]:
        """Get all connections targeting a node."""
        return [c for c in self.connections if c.target_id == node_id]

    def get_outgoing_connections(self, node_id: str) -> List[Connection]:
        """Get all connections from a node."""
        return [c for c in self.connections if c.source_id == node_id]

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the design for consistency."""
        errors = []

        # Check all connection endpoints exist
        for conn in self.connections:
            if conn.source_id not in self._node_lookup:
                errors.append(f"Connection source not found: {conn.source_id}")
            if conn.target_id not in self._node_lookup:
                errors.append(f"Connection target not found: {conn.target_id}")

        # Check operation input counts
        for op_node in self.operations:
            incoming = self.get_incoming_connections(op_node.node_id)
            if len(incoming) != op_node.operation.num_inputs:
                errors.append(
                    f"Operation {op_node.node_id} expects {op_node.operation.num_inputs} "
                    f"inputs but has {len(incoming)} connections"
                )

        return len(errors) == 0, errors

    def copy(self) -> "Design":
        """Create a deep copy of the design."""
        new_design = Design(self.foundation)
        new_design.operations = [
            OperationNode(op.node_id, op.operation, op.position, op.floor)
            for op in self.operations
        ]
        new_design.inputs = [
            InputNode(inp.node_id, inp.port)
            for inp in self.inputs
        ]
        new_design.outputs = [
            OutputNode(out.node_id, out.port)
            for out in self.outputs
        ]
        new_design.connections = [
            Connection(c.source_id, c.source_output_idx, c.target_id, c.target_input_idx)
            for c in self.connections
        ]
        new_design._rebuild_lookup()
        return new_design

    def __repr__(self) -> str:
        return (
            f"Design(foundation={self.foundation.foundation_type.value}, "
            f"ops={len(self.operations)}, "
            f"conns={len(self.connections)})"
        )
