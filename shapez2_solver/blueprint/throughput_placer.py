"""Throughput-optimized placement with splitter/merger networks."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from math import ceil, log2

from ..simulator.design import Design, OperationNode, InputNode, OutputNode
from .building_types import (
    BuildingType, BUILDING_SPECS, BuildingSpec, Rotation,
    get_building_ports, BUILDING_PORTS
)
from .placer import PlacedBuilding, GridCell


@dataclass
class ParallelGroup:
    """A group of parallel buildings for throughput matching."""
    operation_id: str
    building_type: BuildingType
    count: int  # Number of parallel buildings needed
    placements: List[PlacedBuilding] = field(default_factory=list)
    splitter_placements: List[PlacedBuilding] = field(default_factory=list)
    merger_placements: List[PlacedBuilding] = field(default_factory=list)


class ThroughputPlacer:
    """Places operations with throughput-optimized parallel buildings."""

    def __init__(self, width: int = 64, height: int = 64, num_floors: int = 3):
        self.width = width
        self.height = height
        self.num_floors = num_floors
        self.grid: Dict[Tuple[int, int, int], GridCell] = {}
        self.placements: Dict[str, PlacedBuilding] = {}
        self.parallel_groups: Dict[str, ParallelGroup] = {}

    def place_design(self, design: Design) -> Dict[str, PlacedBuilding]:
        """
        Place all operations with throughput optimization.

        Creates splitter trees before slow operations and merger trees after.
        """
        self.grid.clear()
        self.placements.clear()
        self.parallel_groups.clear()

        # Get execution order
        execution_order = self._get_execution_order(design)

        # Calculate parallel requirements for each operation
        for node_id in execution_order:
            node = design.get_node(node_id)
            if isinstance(node, OperationNode):
                building_type = self._get_building_type(node)
                spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

                self.parallel_groups[node_id] = ParallelGroup(
                    operation_id=node_id,
                    building_type=building_type,
                    count=spec.per_belt
                )

        # Place inputs
        input_x = 0
        for i, inp in enumerate(design.inputs):
            y = i * 4  # More space for splitter networks
            self._place_io(inp.node_id, input_x, y, 0, is_input=True)

        # Place operations with parallel groups
        current_x = 4
        for node_id in execution_order:
            if node_id in self.parallel_groups:
                group = self.parallel_groups[node_id]
                current_x = self._place_parallel_group(group, current_x, design)
                current_x += 4  # Space between operation groups

        # Place outputs
        output_x = current_x + 2
        for i, out in enumerate(design.outputs):
            y = i * 4
            self._place_io(out.node_id, output_x, y, 0, is_input=False)

        return self.placements

    def _place_parallel_group(self, group: ParallelGroup, start_x: int, design: Design) -> int:
        """
        Place a parallel group of buildings with splitters and mergers.

        Returns the x position after the group.
        """
        spec = BUILDING_SPECS.get(group.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
        n = group.count

        if n == 1:
            # Single building, no splitters/mergers needed
            y = 0
            for attempt_y in range(self.height - spec.height + 1):
                if self._can_place_building(start_x, attempt_y, 0, spec):
                    placement = self._create_placement(
                        group.operation_id, group.building_type,
                        start_x, attempt_y, 0, Rotation.EAST
                    )
                    self.placements[group.operation_id] = placement
                    group.placements.append(placement)
                    self._mark_building_occupied(start_x, attempt_y, 0, spec, group.operation_id)
                    break
            return start_x + spec.width + 1

        # Multi-building parallel setup
        # Layout: splitter tree -> N buildings -> merger tree

        # Calculate splitter tree depth (log2 rounded up)
        splitter_depth = ceil(log2(n))
        splitter_width = splitter_depth + 1

        # Place splitter tree
        splitter_x = start_x
        splitter_y = 0
        self._place_splitter_tree(group, splitter_x, splitter_y, n)

        # Place parallel buildings
        building_x = splitter_x + splitter_width + 1
        building_spacing = max(2, spec.height + 1)

        for i in range(n):
            building_y = i * building_spacing
            building_id = f"{group.operation_id}_p{i}"

            # Handle multi-floor buildings
            base_floor = 0
            if spec.depth > 1 and self._needs_floor_separation(group.building_type, i, n):
                # Alternate floors for multi-floor buildings to avoid conflicts
                base_floor = (i % 2) * spec.depth

            if self._can_place_building(building_x, building_y, base_floor, spec):
                placement = self._create_placement(
                    building_id, group.building_type,
                    building_x, building_y, base_floor, Rotation.EAST
                )
                self.placements[building_id] = placement
                group.placements.append(placement)
                self._mark_building_occupied(building_x, building_y, base_floor, spec, building_id)

        # Place merger tree
        merger_x = building_x + spec.width + 1
        self._place_merger_tree(group, merger_x, 0, n)

        # Store the main placement reference
        if group.placements:
            self.placements[group.operation_id] = group.placements[0]

        return merger_x + splitter_depth + 2

    def _place_splitter_tree(self, group: ParallelGroup, start_x: int, start_y: int, n: int) -> None:
        """Place a binary splitter tree to divide input into n outputs."""
        if n <= 1:
            return

        depth = ceil(log2(n))
        current_level = [(start_x, start_y, 0)]  # (x, y, floor)

        for level in range(depth):
            next_level = []
            spacing = 2 ** (depth - level - 1)

            for i, (x, y, floor) in enumerate(current_level):
                splitter_id = f"{group.operation_id}_split_{level}_{i}"

                if self._can_place_building(x, y, floor, BuildingSpec(1, 1, 1, 1, 2, 1, 180, 180)):
                    placement = self._create_placement(
                        splitter_id, BuildingType.SPLITTER,
                        x, y, floor, Rotation.EAST
                    )
                    self.placements[splitter_id] = placement
                    group.splitter_placements.append(placement)
                    self._mark_occupied(x, y, floor, splitter_id)

                    # Add output positions for next level
                    next_level.append((x + 1, y - spacing, floor))
                    next_level.append((x + 1, y + spacing, floor))

            current_level = next_level[:n]  # Limit to needed count

    def _place_merger_tree(self, group: ParallelGroup, start_x: int, start_y: int, n: int) -> None:
        """Place a binary merger tree to combine n inputs into one output."""
        if n <= 1:
            return

        depth = ceil(log2(n))
        spec = BUILDING_SPECS.get(group.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

        # Start positions match the building outputs
        building_spacing = max(2, spec.height + 1)
        current_level = [(start_x, i * building_spacing, 0) for i in range(n)]

        for level in range(depth):
            next_level = []
            pairs = len(current_level) // 2

            for i in range(pairs):
                pos1 = current_level[i * 2]
                pos2 = current_level[i * 2 + 1] if i * 2 + 1 < len(current_level) else None

                if pos2:
                    # Place merger between the two inputs
                    merger_y = (pos1[1] + pos2[1]) // 2
                    merger_x = pos1[0] + 1
                    merger_id = f"{group.operation_id}_merge_{level}_{i}"

                    if self._can_place_building(merger_x, merger_y, 0, BuildingSpec(1, 1, 1, 2, 1, 1, 180, 180)):
                        placement = self._create_placement(
                            merger_id, BuildingType.MERGER,
                            merger_x, merger_y, 0, Rotation.EAST
                        )
                        self.placements[merger_id] = placement
                        group.merger_placements.append(placement)
                        self._mark_occupied(merger_x, merger_y, 0, merger_id)

                        next_level.append((merger_x + 1, merger_y, 0))
                else:
                    # Odd one out, pass through
                    next_level.append((pos1[0] + 1, pos1[1], pos1[2]))

            # Handle odd element at end
            if len(current_level) % 2 == 1:
                next_level.append(current_level[-1])

            current_level = next_level

    def _needs_floor_separation(self, building_type: BuildingType, index: int, total: int) -> bool:
        """Check if this building needs floor separation for multi-floor layouts."""
        # For now, don't separate floors - place all on ground level
        return False

    def _place_io(self, node_id: str, x: int, y: int, floor: int, is_input: bool) -> None:
        """Place an input or output marker."""
        placement = PlacedBuilding(
            node_id=node_id,
            building_type=BuildingType.BELT_FORWARD,
            x=x,
            y=y,
            layer=floor,
            rotation=Rotation.EAST,
            input_positions=[] if is_input else [(-1, 0, 0)],
            output_positions=[(1, 0, 0)] if is_input else []
        )
        self.placements[node_id] = placement
        self._mark_occupied(x, y, floor, node_id)

    def _create_placement(self, node_id: str, building_type: BuildingType,
                          x: int, y: int, floor: int, rotation: Rotation) -> PlacedBuilding:
        """Create a placed building with port positions."""
        ports = get_building_ports(building_type, rotation)
        return PlacedBuilding(
            node_id=node_id,
            building_type=building_type,
            x=x,
            y=y,
            layer=floor,
            rotation=rotation,
            input_positions=ports.get('inputs', []),
            output_positions=ports.get('outputs', [])
        )

    def _get_building_type(self, node: OperationNode) -> BuildingType:
        """Get the building type for an operation."""
        from .building_types import OPERATION_TO_BUILDING

        op_class = node.operation.__class__.__name__

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
        """Check if a building can be placed at the given position."""
        if x < 0 or y < 0 or x + spec.width > self.width or y + spec.height > self.height:
            return False

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

    def _mark_building_occupied(self, x: int, y: int, base_layer: int,
                                 spec: BuildingSpec, building_id: str) -> None:
        """Mark all cells occupied by a building."""
        for floor in range(spec.depth):
            layer = base_layer + floor
            for dx in range(spec.width):
                for dy in range(spec.height):
                    self._mark_occupied(x + dx, y + dy, layer, building_id)

    def _mark_occupied(self, x: int, y: int, layer: int, building_id: str) -> None:
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
        """Get the bounding box of all placements."""
        if not self.placements:
            return (0, 0, 0, 0)

        min_x = min(p.x for p in self.placements.values())
        min_y = min(p.y for p in self.placements.values())
        max_x = max(p.x for p in self.placements.values())
        max_y = max(p.y for p in self.placements.values())

        return (min_x, min_y, max_x, max_y)

    def print_grid(self) -> str:
        """Print ASCII representation of all floors."""
        bounds = self.get_bounds()
        min_x, min_y, max_x, max_y = bounds

        symbols = {
            BuildingType.BELT_FORWARD: "→",
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
            BuildingType.SPLITTER: "Y",
            BuildingType.MERGER: "λ",
            BuildingType.LIFT_UP: "↑",
            BuildingType.LIFT_DOWN: "↓",
        }

        lines = []
        max_floor = max((p.layer for p in self.placements.values()), default=0)

        for f in range(max_floor + 1):
            lines.append(f"Floor {f}:")
            width = max_x - min_x + 3
            height = max_y - min_y + 3
            grid = [['.' for _ in range(width)] for _ in range(height)]

            for node_id, placement in self.placements.items():
                spec = BUILDING_SPECS.get(placement.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

                if placement.layer <= f < placement.layer + spec.depth:
                    x = placement.x - min_x + 1
                    y = placement.y - min_y + 1
                    symbol = symbols.get(placement.building_type, "?")

                    if node_id.startswith("in_"):
                        symbol = "I"
                    elif node_id.startswith("out_"):
                        symbol = "O"

                    if 0 <= y < height and 0 <= x < width:
                        grid[y][x] = symbol

            for row in grid:
                lines.append("  " + "".join(row))
            lines.append("")

        return "\n".join(lines)

    def print_throughput_summary(self) -> str:
        """Print throughput optimization summary."""
        lines = ["Throughput Optimization Summary:", "-" * 50]

        for op_id, group in self.parallel_groups.items():
            spec = BUILDING_SPECS.get(group.building_type)
            if spec:
                actual = len(group.placements)
                needed = spec.per_belt
                pct = (actual / needed) * 100 if needed > 0 else 100

                lines.append(f"  {op_id} ({group.building_type.name}):")
                lines.append(f"    Placed: {actual}/{needed} buildings")
                lines.append(f"    Throughput: {pct:.0f}% of belt capacity")
                lines.append(f"    Splitters: {len(group.splitter_placements)}")
                lines.append(f"    Mergers: {len(group.merger_placements)}")

        return "\n".join(lines)
