"""Solution visualization showing the processing pipeline."""

from typing import Dict, List, Optional, Tuple, Union
from ..shapes.shape import Shape, Color
from ..shapes.encoder import ShapeCodeEncoder
from ..simulator.design import Design, OperationNode, InputNode, OutputNode
from ..simulator.simulator import Simulator

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class SolutionDisplay:
    """Displays a solution with floors, operations, and shape states."""

    def __init__(self, design: Design, inputs: Dict[str, Union[Shape, Color]]):
        """
        Initialize the solution display.

        Args:
            design: The solution design
            inputs: Input values for simulation
        """
        self.design = design
        self.inputs = inputs
        self.node_values: Dict[str, List[Optional[Union[Shape, Color]]]] = {}
        self._simulate()

    def _simulate(self) -> None:
        """Run simulation to get shape values at each node."""
        simulator = Simulator(self.design)
        result = simulator.execute(self.inputs)
        if result.success:
            self.node_values = result.node_values

    def to_ascii(self) -> str:
        """
        Generate ASCII representation of the solution.

        Shows:
        - Each floor/level
        - Operations with their inputs/outputs
        - Shape state at each point
        - Connections as arrows
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  SOLUTION PIPELINE")
        lines.append("=" * 60)

        # Get execution order
        execution_order = self._get_execution_order()

        # Group by floor
        floors: Dict[int, List[str]] = {}
        for node_id in execution_order:
            node = self.design.get_node(node_id)
            floor = 0
            if isinstance(node, OperationNode):
                floor = node.floor
            if floor not in floors:
                floors[floor] = []
            floors[floor].append(node_id)

        # Display each floor
        for floor in sorted(floors.keys()):
            lines.append("")
            lines.append(f"{'─' * 20} FLOOR {floor} {'─' * 20}")
            lines.append("")

            for node_id in floors[floor]:
                node = self.design.get_node(node_id)
                node_lines = self._render_node(node_id, node)
                lines.extend(node_lines)
                lines.append("")

        # Show final output summary
        lines.append("=" * 60)
        lines.append("  OUTPUTS")
        lines.append("=" * 60)
        for output_node in self.design.outputs:
            values = self.node_values.get(output_node.node_id, [None])
            value = values[0] if values else None
            if isinstance(value, Shape):
                lines.append(f"  {output_node.node_id}: {value.to_code()}")
            else:
                lines.append(f"  {output_node.node_id}: {value}")

        return "\n".join(lines)

    def _render_node(self, node_id: str, node) -> List[str]:
        """Render a single node with its inputs, operation, and outputs."""
        lines = []

        if isinstance(node, InputNode):
            values = self.node_values.get(node_id, [None])
            value = values[0] if values else None
            shape_code = value.to_code() if isinstance(value, Shape) else str(value)

            lines.append(f"  ┌─────────────────────────────────────┐")
            lines.append(f"  │  INPUT: {node_id:<28} │")
            lines.append(f"  │  Port: {node.port.direction.name:<29} │")
            lines.append(f"  │  Shape: {shape_code:<28} │")
            lines.append(f"  └─────────────────────────────────────┘")
            lines.append(f"                    │")
            lines.append(f"                    ▼")

        elif isinstance(node, OperationNode):
            op_name = node.operation.__class__.__name__

            # Get input shapes
            incoming = self.design.get_incoming_connections(node_id)
            input_shapes = []
            for conn in incoming:
                source_values = self.node_values.get(conn.source_id, [])
                if conn.source_output_idx < len(source_values):
                    val = source_values[conn.source_output_idx]
                    if isinstance(val, Shape):
                        input_shapes.append(val.to_code())
                    else:
                        input_shapes.append(str(val))
                else:
                    input_shapes.append("None")

            # Get output shapes
            output_values = self.node_values.get(node_id, [])
            output_shapes = []
            for val in output_values:
                if isinstance(val, Shape):
                    output_shapes.append(val.to_code())
                elif val is None:
                    output_shapes.append("None")
                else:
                    output_shapes.append(str(val))

            lines.append(f"  ┌─────────────────────────────────────┐")
            lines.append(f"  │  OPERATION: {op_name:<24} │")
            lines.append(f"  │  Node: {node_id:<29} │")
            lines.append(f"  ├─────────────────────────────────────┤")

            # Input section
            lines.append(f"  │  INPUTS:                            │")
            for i, shape in enumerate(input_shapes):
                if len(shape) > 30:
                    shape = shape[:27] + "..."
                lines.append(f"  │    [{i}] {shape:<30} │")

            lines.append(f"  ├─────────────────────────────────────┤")

            # Output section
            lines.append(f"  │  OUTPUTS:                           │")
            for i, shape in enumerate(output_shapes):
                if len(shape) > 30:
                    shape = shape[:27] + "..."
                lines.append(f"  │    [{i}] {shape:<30} │")

            lines.append(f"  └─────────────────────────────────────┘")
            lines.append(f"                    │")
            lines.append(f"                    ▼")

        elif isinstance(node, OutputNode):
            values = self.node_values.get(node_id, [None])
            value = values[0] if values else None
            shape_code = value.to_code() if isinstance(value, Shape) else str(value)

            lines.append(f"  ┌─────────────────────────────────────┐")
            lines.append(f"  │  OUTPUT: {node_id:<27} │")
            lines.append(f"  │  Port: {node.port.direction.name:<29} │")
            lines.append(f"  │  Shape: {shape_code:<28} │")
            lines.append(f"  └─────────────────────────────────────┘")

        return lines

    def _get_execution_order(self) -> List[str]:
        """Get nodes in execution order."""
        from collections import deque

        in_degree: Dict[str, int] = {}
        adjacency: Dict[str, List[str]] = {}

        # Initialize
        for node in self.design.inputs:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []
        for node in self.design.operations:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []
        for node in self.design.outputs:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []

        # Build graph
        for conn in self.design.connections:
            if conn.source_id in adjacency:
                adjacency[conn.source_id].append(conn.target_id)
            if conn.target_id in in_degree:
                in_degree[conn.target_id] += 1

        # Topological sort
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

    def to_compact(self) -> str:
        """Generate a compact one-line representation."""
        execution_order = self._get_execution_order()
        parts = []

        for node_id in execution_order:
            node = self.design.get_node(node_id)
            values = self.node_values.get(node_id, [])

            if isinstance(node, InputNode):
                val = values[0] if values else None
                shape = val.to_code() if isinstance(val, Shape) else "?"
                parts.append(f"[IN:{shape}]")

            elif isinstance(node, OperationNode):
                op_name = node.operation.__class__.__name__.replace("Operation", "")
                out_vals = []
                for v in values:
                    if isinstance(v, Shape):
                        out_vals.append(v.to_code())
                    elif v is None:
                        out_vals.append("∅")
                    else:
                        out_vals.append(str(v))
                parts.append(f"→ {op_name} → [{','.join(out_vals)}]")

            elif isinstance(node, OutputNode):
                val = values[0] if values else None
                shape = val.to_code() if isinstance(val, Shape) else "?"
                parts.append(f"→ [OUT:{shape}]")

        return " ".join(parts)


def display_solution(design: Design, inputs: Dict[str, Union[Shape, Color]]) -> None:
    """
    Display a solution in the console.

    Args:
        design: The solution design
        inputs: Input values
    """
    display = SolutionDisplay(design, inputs)
    print(display.to_ascii())


def display_solution_compact(design: Design, inputs: Dict[str, Union[Shape, Color]]) -> None:
    """
    Display a compact solution in the console.

    Args:
        design: The solution design
        inputs: Input values
    """
    display = SolutionDisplay(design, inputs)
    print(display.to_compact())
