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

    def to_grid(self) -> str:
        """
        Generate a grid-based layout showing the factory floor.

        Shows:
        - Grid with operations placed
        - Conveyor belt connections (arrows)
        - Shape values at each step
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  FACTORY LAYOUT")
        lines.append("=" * 70)

        execution_order = self._get_execution_order()

        # Build connection map
        connections_from: Dict[str, List[Tuple[str, int, int]]] = {}  # source -> [(target, src_out, tgt_in)]
        for conn in self.design.connections:
            if conn.source_id not in connections_from:
                connections_from[conn.source_id] = []
            connections_from[conn.source_id].append((conn.target_id, conn.source_output_idx, conn.target_input_idx))

        # Group nodes by floor
        floors: Dict[int, List] = {0: []}
        for node_id in execution_order:
            node = self.design.get_node(node_id)
            floor = node.floor if isinstance(node, OperationNode) else 0
            if floor not in floors:
                floors[floor] = []
            floors[floor].append((node_id, node))

        # Render each floor
        for floor_num in sorted(floors.keys()):
            floor_nodes = floors[floor_num]

            lines.append("")
            lines.append(f"╔{'═' * 68}╗")
            lines.append(f"║  FLOOR {floor_num:<59}║")
            lines.append(f"╠{'═' * 68}╣")

            # Render nodes on this floor
            for node_id, node in floor_nodes:
                values = self.node_values.get(node_id, [None])

                if isinstance(node, InputNode):
                    val = values[0] if values else None
                    shape_str = val.to_code() if isinstance(val, Shape) else str(val)
                    symbol = "📥"
                    label = f"INPUT [{node.port.direction.name}]"

                elif isinstance(node, OperationNode):
                    op_name = node.operation.__class__.__name__.replace("Operation", "")
                    symbol = self._get_operation_symbol(op_name)
                    label = op_name

                elif isinstance(node, OutputNode):
                    val = values[0] if values else None
                    shape_str = val.to_code() if isinstance(val, Shape) else str(val)
                    symbol = "📤"
                    label = f"OUTPUT [{node.port.direction.name}]"
                else:
                    continue

                # Render the node
                lines.append(f"║                                                                    ║")
                lines.append(f"║   {symbol} {label:<20} ({node_id})                        ║"[:71] + "║")

                # Show input values (what comes in)
                incoming = self.design.get_incoming_connections(node_id)
                if incoming:
                    for conn in incoming:
                        src_values = self.node_values.get(conn.source_id, [])
                        if conn.source_output_idx < len(src_values):
                            src_val = src_values[conn.source_output_idx]
                            if isinstance(src_val, Shape):
                                val_str = src_val.to_code()[:25]
                            else:
                                val_str = str(src_val)[:25]
                        else:
                            val_str = "None"
                        line = f"║      ◀── from {conn.source_id}[{conn.source_output_idx}]: {val_str}"
                        lines.append(f"{line:<69}║")

                # Show output values (what comes out)
                if isinstance(node, OperationNode):
                    output_vals = self.node_values.get(node_id, [])
                    for i, val in enumerate(output_vals):
                        if isinstance(val, Shape):
                            val_str = val.to_code()[:25]
                        elif val is None:
                            val_str = "∅ (empty)"
                        else:
                            val_str = str(val)[:25]
                        line = f"║      ──▶ output[{i}]: {val_str}"
                        lines.append(f"{line:<69}║")

                # Show where outputs go
                if node_id in connections_from:
                    for target_id, src_out, tgt_in in connections_from[node_id]:
                        line = f"║          └─► {target_id}[{tgt_in}]"
                        lines.append(f"{line:<69}║")

            lines.append(f"╚{'═' * 68}╝")

        # Summary
        lines.append("")
        lines.append("=" * 70)
        lines.append("  SIGNAL FLOW")
        lines.append("=" * 70)
        for node_id in execution_order:
            node = self.design.get_node(node_id)
            values = self.node_values.get(node_id, [])

            if isinstance(node, InputNode):
                val = values[0] if values else None
                shape = val.to_code() if isinstance(val, Shape) else "?"
                lines.append(f"  {node_id}: {shape}")
            elif isinstance(node, OperationNode):
                op_name = node.operation.__class__.__name__.replace("Operation", "")
                out_strs = []
                for v in values:
                    if isinstance(v, Shape):
                        out_strs.append(v.to_code())
                    elif v is None:
                        out_strs.append("∅")
                    else:
                        out_strs.append(str(v))
                lines.append(f"  {node_id} ({op_name}): {' | '.join(out_strs)}")
            elif isinstance(node, OutputNode):
                val = values[0] if values else None
                shape = val.to_code() if isinstance(val, Shape) else "?"
                lines.append(f"  {node_id}: {shape}")

        return "\n".join(lines)

    def _get_operation_symbol(self, op_name: str) -> str:
        """Get a symbol for an operation type."""
        symbols = {
            "Rotate": "🔄",
            "Cut": "✂️",
            "HalfDestroyer": "💥",
            "Stack": "📚",
            "Unstack": "📖",
            "Swap": "🔀",
            "Swapper": "🔀",
            "Paint": "🎨",
        }
        return symbols.get(op_name, "⚙️")


def display_solution(design: Design, inputs: Dict[str, Union[Shape, Color]]) -> None:
    """
    Display a solution in the console with factory grid layout.

    Args:
        design: The solution design
        inputs: Input values
    """
    display = SolutionDisplay(design, inputs)
    print(display.to_grid())


def display_solution_ascii(design: Design, inputs: Dict[str, Union[Shape, Color]]) -> None:
    """
    Display a solution in simple ASCII format.

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
