"""Grid-based placement for operations on a foundation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from ..simulator.design import Design, OperationNode, InputNode, OutputNode, Connection
from ..operations.rotator import RotateOperation
from .building_types import (
    BuildingType, OPERATION_TO_BUILDING, BUILDING_SPECS, BUILDING_SIZES,
    Rotation, BuildingSpec, BUILDING_PORTS, get_building_ports
)


@dataclass
class PlacedBuilding:
    """A building placed on the grid."""
    node_id: str
    building_type: BuildingType
    x: int
    y: int
    layer: int = 0  # Base floor (for multi-floor buildings, this is the bottom)
    rotation: Rotation = Rotation.EAST

    # Input/output positions: (rel_x, rel_y, floor) relative to building origin
    input_positions: List[Tuple[int, int, int]] = field(default_factory=list)
    output_positions: List[Tuple[int, int, int]] = field(default_factory=list)


@dataclass
class GridCell:
    """A cell in the placement grid."""
    x: int
    y: int
    layer: int
    occupied: bool = False
    building_id: Optional[str] = None
    is_belt: bool = False


class GridPlacer:
    """Places operations on a grid for blueprint generation."""

    def __init__(self, width: int = 32, height: int = 32, num_floors: int = 3,
                 optimize_throughput: bool = True):
        """
        Initialize the grid placer.

        Args:
            width: Grid width
            height: Grid height
            num_floors: Number of floors available
            optimize_throughput: If True, place multiple buildings to match belt throughput
        """
        self.width = width
        self.height = height
        self.num_floors = num_floors
        self.optimize_throughput = optimize_throughput
        self.grid: Dict[Tuple[int, int, int], GridCell] = {}  # (x, y, layer) -> cell
        self.placements: Dict[str, PlacedBuilding] = {}  # node_id -> placement

    def place_design(self, design: Design) -> Dict[str, PlacedBuilding]:
        """
        Place all operations from a design on the grid.

        Uses a left-to-right flow layout:
        - Inputs on the left
        - Operations in the middle (in execution order)
        - Outputs on the right

        Args:
            design: The design to place

        Returns:
            Dictionary mapping node IDs to placements
        """
        self.grid.clear()
        self.placements.clear()

        # Get execution order (topological sort)
        execution_order = self._get_execution_order(design)

        # Calculate layout
        num_inputs = len(design.inputs)
        num_outputs = len(design.outputs)

        # Starting positions
        input_x = 0
        current_x = 2  # Start operations after inputs with belt space

        # Place inputs on the left edge, spaced vertically
        for i, inp in enumerate(design.inputs):
            y = i * 2  # Space inputs apart

            placement = PlacedBuilding(
                node_id=inp.node_id,
                building_type=BuildingType.BELT_FORWARD,
                x=input_x,
                y=y,
                layer=0,
                rotation=Rotation.EAST,
                input_positions=[],  # External input
                output_positions=[(1, 0, 0)]  # Output goes right on floor 0
            )
            self.placements[inp.node_id] = placement
            self._mark_occupied(input_x, y, 0, inp.node_id)

        # Place operations in execution order
        op_column = current_x
        ops_in_column = 0
        max_ops_per_column = max(3, num_inputs)

        for node_id in execution_order:
            node = design.get_node(node_id)

            if isinstance(node, OperationNode):
                building_type = self._get_building_type(node)
                spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
                ports = get_building_ports(building_type)

                # How many buildings do we need for full throughput?
                num_buildings = spec.per_belt if self.optimize_throughput else 1

                # For now, place just one building (throughput optimization TODO)
                # Find free position
                placed = False
                for attempt_y in range(self.height - spec.height + 1):
                    if self._can_place_building(op_column, attempt_y, 0, spec):
                        rotation = Rotation.EAST

                        placement = PlacedBuilding(
                            node_id=node_id,
                            building_type=building_type,
                            x=op_column,
                            y=attempt_y,
                            layer=0,
                            rotation=rotation,
                            input_positions=ports.get('inputs', []),
                            output_positions=ports.get('outputs', []),
                        )
                        self.placements[node_id] = placement

                        # Mark all cells as occupied (including multi-floor)
                        self._mark_building_occupied(op_column, attempt_y, 0, spec, node_id)

                        placed = True
                        ops_in_column += 1

                        # Move to next column if getting full
                        if ops_in_column >= max_ops_per_column:
                            op_column += max(3, spec.width + 2)
                            ops_in_column = 0

                        break

                if not placed:
                    # Move to next column and try again
                    op_column += 3
                    ops_in_column = 0

                    for attempt_y in range(self.height - spec.height + 1):
                        if self._can_place_building(op_column, attempt_y, 0, spec):
                            placement = PlacedBuilding(
                                node_id=node_id,
                                building_type=building_type,
                                x=op_column,
                                y=attempt_y,
                                layer=0,
                                rotation=Rotation.EAST,
                                input_positions=ports.get('inputs', []),
                                output_positions=ports.get('outputs', []),
                            )
                            self.placements[node_id] = placement
                            self._mark_building_occupied(op_column, attempt_y, 0, spec, node_id)
                            break

        # Place outputs on the right edge
        output_x = op_column + 3
        for i, out in enumerate(design.outputs):
            y = i * 2

            placement = PlacedBuilding(
                node_id=out.node_id,
                building_type=BuildingType.BELT_FORWARD,
                x=output_x,
                y=y,
                layer=0,
                rotation=Rotation.EAST,
                input_positions=[(-1, 0, 0)],  # Input from left on floor 0
                output_positions=[]  # External output
            )
            self.placements[out.node_id] = placement
            self._mark_occupied(output_x, y, 0, out.node_id)

        return self.placements

    def _get_building_type(self, node: OperationNode) -> BuildingType:
        """Get the building type for an operation."""
        op_class = node.operation.__class__.__name__

        # Special handling for rotator with steps
        if op_class == "RotateOperation":
            if hasattr(node.operation, 'steps'):
                steps = node.operation.steps
                if steps == 1:
                    return BuildingType.ROTATOR_CW
                elif steps == 2:
                    return BuildingType.ROTATOR_180
                elif steps == 3:
                    return BuildingType.ROTATOR_CCW
            return BuildingType.ROTATOR_CW

        return OPERATION_TO_BUILDING.get(op_class, BuildingType.BELT_FORWARD)

    def _can_place_building(self, x: int, y: int, base_layer: int, spec: BuildingSpec) -> bool:
        """Check if a building can be placed at the given position (including all floors)."""
        if x < 0 or y < 0 or x + spec.width > self.width or y + spec.height > self.height:
            return False

        # Check all cells on all floors the building occupies
        for floor in range(spec.depth):
            layer = base_layer + floor
            if layer >= self.num_floors:
                return False
            for dx in range(spec.width):
                for dy in range(spec.height):
                    cell = self.grid.get((x + dx, y + dy, layer))
                    if cell and cell.occupied:
                        return False

        return True

    def _mark_building_occupied(self, x: int, y: int, base_layer: int, spec: BuildingSpec,
                                 building_id: str) -> None:
        """Mark all cells occupied by a building (including multi-floor)."""
        for floor in range(spec.depth):
            layer = base_layer + floor
            for dx in range(spec.width):
                for dy in range(spec.height):
                    self._mark_occupied(x + dx, y + dy, layer, building_id)

    def _mark_occupied(self, x: int, y: int, layer: int, building_id: Optional[str] = None) -> None:
        """Mark a cell as occupied."""
        self.grid[(x, y, layer)] = GridCell(x, y, layer, occupied=True, building_id=building_id)

    def _get_execution_order(self, design: Design) -> List[str]:
        """Get nodes in execution order (topological sort)."""
        from collections import deque

        in_degree: Dict[str, int] = {}
        adjacency: Dict[str, List[str]] = {}

        for node in design.operations:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []

        for conn in design.connections:
            if conn.source_id in adjacency and conn.target_id in adjacency:
                in_degree[conn.target_id] += 1
                adjacency[conn.source_id].append(conn.target_id)

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

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounding box of all placements (min_x, min_y, max_x, max_y)."""
        if not self.placements:
            return (0, 0, 0, 0)

        min_x = min(p.x for p in self.placements.values())
        min_y = min(p.y for p in self.placements.values())
        max_x = max(p.x for p in self.placements.values())
        max_y = max(p.y for p in self.placements.values())

        return (min_x, min_y, max_x, max_y)

    def get_max_floor(self) -> int:
        """Get the highest floor used."""
        max_floor = 0
        for p in self.placements.values():
            spec = BUILDING_SPECS.get(p.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
            max_floor = max(max_floor, p.layer + spec.depth - 1)
        return max_floor

    def print_grid(self, floor: int = 0) -> str:
        """Print ASCII representation of the grid for a specific floor."""
        bounds = self.get_bounds()
        min_x, min_y, max_x, max_y = bounds

        # Build symbol map
        symbols = {
            BuildingType.BELT_FORWARD: "→",
            BuildingType.BELT_LEFT: "↰",
            BuildingType.BELT_RIGHT: "↱",
            BuildingType.ROTATOR_CW: "R",
            BuildingType.ROTATOR_CCW: "r",
            BuildingType.ROTATOR_180: "⟳",
            BuildingType.CUTTER: "C",
            BuildingType.HALF_CUTTER: "H",
            BuildingType.STACKER: "S",
            BuildingType.UNSTACKER: "U",
            BuildingType.SWAPPER: "X",
            BuildingType.PAINTER: "P",
            BuildingType.TRASH: "T",
        }

        lines = []
        max_floor = self.get_max_floor()

        for f in range(max_floor + 1):
            lines.append(f"Floor {f}:")
            lines.append(f"  Grid ({min_x},{min_y}) to ({max_x},{max_y}):")

            # Create grid
            width = max_x - min_x + 3
            height = max_y - min_y + 3
            grid = [['.' for _ in range(width)] for _ in range(height)]

            # Place buildings that exist on this floor
            for node_id, placement in self.placements.items():
                spec = BUILDING_SPECS.get(placement.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

                # Check if this building is on this floor
                if placement.layer <= f < placement.layer + spec.depth:
                    x = placement.x - min_x + 1
                    y = placement.y - min_y + 1
                    symbol = symbols.get(placement.building_type, "?")

                    # Mark input/output markers
                    if node_id.startswith("in_"):
                        symbol = "I"
                    elif node_id.startswith("out_"):
                        symbol = "O"

                    if 0 <= y < height and 0 <= x < width:
                        grid[y][x] = symbol

            # Build output
            for row in grid:
                lines.append("  " + "".join(row))
            lines.append("")

        return "\n".join(lines)

    def print_throughput_info(self) -> str:
        """Print information about throughput for each building."""
        lines = []
        lines.append("Throughput Information:")
        lines.append("-" * 40)

        for node_id, placement in self.placements.items():
            if node_id.startswith("in_") or node_id.startswith("out_"):
                continue

            spec = BUILDING_SPECS.get(placement.building_type)
            if spec:
                lines.append(f"  {node_id} ({placement.building_type.name}):")
                lines.append(f"    Size: {spec.width}x{spec.height}x{spec.depth}")
                lines.append(f"    Rate: {spec.base_rate}-{spec.max_rate} ops/min")
                lines.append(f"    Per belt: {spec.per_belt} buildings needed")
                if spec.per_belt > 1:
                    lines.append(f"    Note: Single building = {100//spec.per_belt}% belt throughput")

        return "\n".join(lines)
