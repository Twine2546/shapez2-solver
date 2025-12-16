"""Conveyor belt routing between buildings with A* pathfinding."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import deque
import heapq

from ..simulator.design import Design, Connection
from .placer import PlacedBuilding, GridPlacer
from .building_types import BuildingType, Rotation, BUILDING_SPECS, BuildingSpec


@dataclass
class BeltSegment:
    """A segment of conveyor belt."""
    x: int
    y: int
    layer: int
    belt_type: BuildingType  # BELT_FORWARD, BELT_LEFT, BELT_RIGHT, LIFT_UP, LIFT_DOWN
    rotation: Rotation


@dataclass
class Route:
    """A complete route between two buildings."""
    source_id: str
    source_output: int
    target_id: str
    target_input: int
    segments: List[BeltSegment]


@dataclass(order=True)
class AStarNode:
    """Node for A* pathfinding."""
    priority: float
    x: int = field(compare=False)
    y: int = field(compare=False)
    layer: int = field(compare=False)
    direction: Tuple[int, int] = field(compare=False)  # (dx, dy) of incoming direction
    g_cost: float = field(compare=False)  # Cost from start
    parent: Optional['AStarNode'] = field(compare=False, default=None)


class ConveyorRouter:
    """Routes conveyor belts between placed buildings using A* pathfinding."""

    # Movement costs
    STRAIGHT_COST = 1.0
    TURN_COST = 1.5
    LIFT_COST = 2.0  # Floor changes cost more

    def __init__(self, placer: GridPlacer, num_floors: int = 3):
        """
        Initialize the router.

        Args:
            placer: The grid placer with building placements
            num_floors: Number of floors available for routing
        """
        self.placer = placer
        self.num_floors = num_floors
        self.routes: List[Route] = []
        self.belt_grid: Dict[Tuple[int, int, int], BeltSegment] = {}
        self.occupied: Set[Tuple[int, int, int]] = set()

        # Initialize occupied set from placer
        self._init_occupied()

    def _init_occupied(self) -> None:
        """Initialize the occupied set from placer grid."""
        for pos, cell in self.placer.grid.items():
            if cell.occupied:
                self.occupied.add(pos)

    def route_connections(self, design: Design) -> List[Route]:
        """
        Route all connections in the design.

        Args:
            design: The design with connections to route

        Returns:
            List of routes with belt segments
        """
        self.routes.clear()
        self.belt_grid.clear()

        for conn in design.connections:
            route = self._route_connection(conn, design)
            if route:
                self.routes.append(route)

        return self.routes

    def _route_connection(self, conn: Connection, design: Design) -> Optional[Route]:
        """Route a single connection with multi-floor support."""
        source_placement = self.placer.placements.get(conn.source_id)
        target_placement = self.placer.placements.get(conn.target_id)

        if not source_placement or not target_placement:
            return None

        # Get source output position (using source_output_idx field)
        source_output_idx = conn.source_output_idx
        if source_output_idx < len(source_placement.output_positions):
            src_offset = source_placement.output_positions[source_output_idx]
            # src_offset is (rel_x, rel_y, floor) - 3D
            if len(src_offset) == 3:
                src_x = source_placement.x + src_offset[0]
                src_y = source_placement.y + src_offset[1]
                src_layer = source_placement.layer + src_offset[2]
            else:
                src_x = source_placement.x + src_offset[0]
                src_y = source_placement.y + src_offset[1]
                src_layer = source_placement.layer
        else:
            src_x = source_placement.x + 1
            src_y = source_placement.y
            src_layer = source_placement.layer

        # Get target input position (using target_input_idx field)
        target_input_idx = conn.target_input_idx
        if target_input_idx < len(target_placement.input_positions):
            tgt_offset = target_placement.input_positions[target_input_idx]
            # tgt_offset is (rel_x, rel_y, floor) - 3D
            if len(tgt_offset) == 3:
                tgt_x = target_placement.x + tgt_offset[0]
                tgt_y = target_placement.y + tgt_offset[1]
                tgt_layer = target_placement.layer + tgt_offset[2]
            else:
                tgt_x = target_placement.x + tgt_offset[0]
                tgt_y = target_placement.y + tgt_offset[1]
                tgt_layer = target_placement.layer
        else:
            tgt_x = target_placement.x - 1
            tgt_y = target_placement.y
            tgt_layer = target_placement.layer

        # Find path using A* with multi-floor support
        segments = self._find_path(src_x, src_y, src_layer, tgt_x, tgt_y, tgt_layer)

        return Route(
            source_id=conn.source_id,
            source_output=source_output_idx,
            target_id=conn.target_id,
            target_input=target_input_idx,
            segments=segments
        )

    def _find_path(self, start_x: int, start_y: int, start_layer: int,
                    end_x: int, end_y: int, end_layer: int) -> List[BeltSegment]:
        """
        Find a path from start to end using A* pathfinding.

        Supports multi-floor routing with lifts.
        """
        # A* with 3D positions (x, y, layer) and direction tracking
        # Direction is tracked to properly cost turns

        # Cardinal directions: (dx, dy)
        DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, S, W, N

        def heuristic(x: int, y: int, layer: int) -> float:
            """Manhattan distance heuristic including floor changes."""
            return abs(x - end_x) + abs(y - end_y) + abs(layer - end_layer) * 2

        def get_neighbors(node: AStarNode) -> List[Tuple[int, int, int, Tuple[int, int], float]]:
            """Get valid neighbors with movement costs."""
            neighbors = []

            # Horizontal/vertical moves on same floor
            for dx, dy in DIRECTIONS:
                nx, ny = node.x + dx, node.y + dy

                # Check bounds
                if nx < -10 or ny < -10 or nx > self.placer.width + 10 or ny > self.placer.height + 10:
                    continue

                # Check if occupied (but destination is always valid)
                if (nx, ny, node.layer) in self.occupied and (nx, ny, node.layer) != (end_x, end_y, end_layer):
                    continue

                # Check if belt already placed there
                if (nx, ny, node.layer) in self.belt_grid:
                    continue

                # Calculate cost
                if node.direction == (dx, dy):
                    cost = self.STRAIGHT_COST
                else:
                    cost = self.TURN_COST

                neighbors.append((nx, ny, node.layer, (dx, dy), cost))

            # Vertical moves (lifts) - only at certain positions
            if node.layer < self.num_floors - 1:
                # Can go up
                up_layer = node.layer + 1
                if (node.x, node.y, up_layer) not in self.occupied:
                    if (node.x, node.y, up_layer) not in self.belt_grid:
                        neighbors.append((node.x, node.y, up_layer, node.direction, self.LIFT_COST))

            if node.layer > 0:
                # Can go down
                down_layer = node.layer - 1
                if (node.x, node.y, down_layer) not in self.occupied:
                    if (node.x, node.y, down_layer) not in self.belt_grid:
                        neighbors.append((node.x, node.y, down_layer, node.direction, self.LIFT_COST))

            return neighbors

        # Initial direction: towards destination
        init_dx = 1 if end_x > start_x else -1 if end_x < start_x else 0
        init_dy = 1 if end_y > start_y else -1 if end_y < start_y else 0
        if init_dx == 0 and init_dy == 0:
            init_dx = 1  # Default direction

        start_node = AStarNode(
            priority=heuristic(start_x, start_y, start_layer),
            x=start_x, y=start_y, layer=start_layer,
            direction=(init_dx, init_dy),
            g_cost=0
        )

        open_set = [start_node]
        closed_set: Set[Tuple[int, int, int, int, int]] = set()  # (x, y, layer, dx, dy)

        max_iterations = 10000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            current = heapq.heappop(open_set)

            # Check if reached destination
            if current.x == end_x and current.y == end_y and current.layer == end_layer:
                return self._reconstruct_path(current)

            state = (current.x, current.y, current.layer, current.direction[0], current.direction[1])
            if state in closed_set:
                continue
            closed_set.add(state)

            # Explore neighbors
            for nx, ny, nl, direction, move_cost in get_neighbors(current):
                new_g = current.g_cost + move_cost
                new_h = heuristic(nx, ny, nl)

                neighbor = AStarNode(
                    priority=new_g + new_h,
                    x=nx, y=ny, layer=nl,
                    direction=direction,
                    g_cost=new_g,
                    parent=current
                )

                neighbor_state = (nx, ny, nl, direction[0], direction[1])
                if neighbor_state not in closed_set:
                    heapq.heappush(open_set, neighbor)

        # No path found, fall back to simple routing
        return self._simple_path(start_x, start_y, start_layer, end_x, end_y, end_layer)

    def _reconstruct_path(self, end_node: AStarNode) -> List[BeltSegment]:
        """Reconstruct the path from A* result."""
        path = []
        current = end_node
        nodes = []

        # Collect all nodes
        while current.parent is not None:
            nodes.append(current)
            current = current.parent

        nodes.reverse()

        # Convert to belt segments
        prev_direction = None
        prev_layer = None

        for i, node in enumerate(nodes):
            # Skip the last node (destination)
            if i == len(nodes) - 1:
                continue

            # Check for floor change
            if prev_layer is not None and node.layer != prev_layer:
                if node.layer > prev_layer:
                    # Going up
                    segment = BeltSegment(
                        x=node.x, y=node.y, layer=prev_layer,
                        belt_type=BuildingType.LIFT_UP,
                        rotation=Rotation.EAST
                    )
                else:
                    # Going down
                    segment = BeltSegment(
                        x=node.x, y=node.y, layer=node.layer,
                        belt_type=BuildingType.LIFT_DOWN,
                        rotation=Rotation.EAST
                    )
                path.append(segment)
                self.belt_grid[(node.x, node.y, segment.layer)] = segment
                prev_layer = node.layer
                continue

            # Determine belt type based on direction change
            belt_type, rotation = self._get_belt_type(prev_direction, node.direction)

            segment = BeltSegment(
                x=node.x, y=node.y, layer=node.layer,
                belt_type=belt_type,
                rotation=rotation
            )
            path.append(segment)
            self.belt_grid[(node.x, node.y, node.layer)] = segment

            prev_direction = node.direction
            prev_layer = node.layer

        return path

    def _get_belt_type(self, prev_dir: Optional[Tuple[int, int]],
                       curr_dir: Tuple[int, int]) -> Tuple[BuildingType, Rotation]:
        """Determine belt type and rotation based on direction."""
        dx, dy = curr_dir

        # Map direction to rotation
        if dx > 0:
            rotation = Rotation.EAST
        elif dx < 0:
            rotation = Rotation.WEST
        elif dy > 0:
            rotation = Rotation.SOUTH
        else:
            rotation = Rotation.NORTH

        # Check for turn
        if prev_dir is not None and prev_dir != curr_dir:
            # Determine turn direction
            # Cross product to determine turn direction
            cross = prev_dir[0] * curr_dir[1] - prev_dir[1] * curr_dir[0]

            if cross > 0:
                return BuildingType.BELT_RIGHT, rotation
            elif cross < 0:
                return BuildingType.BELT_LEFT, rotation

        return BuildingType.BELT_FORWARD, rotation

    def _simple_path(self, start_x: int, start_y: int, start_layer: int,
                     end_x: int, end_y: int, end_layer: int) -> List[BeltSegment]:
        """Simple L-shaped fallback routing."""
        segments = []
        x, y, layer = start_x, start_y, start_layer

        # Handle floor change first if needed
        while layer != end_layer:
            if layer < end_layer:
                segment = BeltSegment(x, y, layer, BuildingType.LIFT_UP, Rotation.EAST)
                self.belt_grid[(x, y, layer)] = segment
                segments.append(segment)
                layer += 1
            else:
                layer -= 1
                segment = BeltSegment(x, y, layer, BuildingType.LIFT_DOWN, Rotation.EAST)
                self.belt_grid[(x, y, layer)] = segment
                segments.append(segment)

        # Horizontal segment
        dx = 1 if end_x > x else -1
        rotation = Rotation.EAST if dx > 0 else Rotation.WEST
        while x != end_x:
            segment = BeltSegment(x, y, layer, BuildingType.BELT_FORWARD, rotation)
            self.belt_grid[(x, y, layer)] = segment
            segments.append(segment)
            x += dx

        # Vertical segment
        dy = 1 if end_y > y else -1
        rotation = Rotation.SOUTH if dy > 0 else Rotation.NORTH
        while y != end_y:
            segment = BeltSegment(x, y, layer, BuildingType.BELT_FORWARD, rotation)
            self.belt_grid[(x, y, layer)] = segment
            segments.append(segment)
            y += dy

        return segments

    def _get_belt_for_direction(
        self,
        prev_dx: int, prev_dy: int,
        curr_dx: int, curr_dy: int
    ) -> Tuple[BuildingType, Rotation]:
        """Get belt type and rotation for a direction change."""
        # Going straight
        if prev_dx == curr_dx and prev_dy == curr_dy:
            if curr_dx > 0:
                return BuildingType.BELT_FORWARD, Rotation.EAST
            elif curr_dx < 0:
                return BuildingType.BELT_FORWARD, Rotation.WEST
            elif curr_dy > 0:
                return BuildingType.BELT_FORWARD, Rotation.SOUTH
            else:
                return BuildingType.BELT_FORWARD, Rotation.NORTH

        # Default
        if curr_dx > 0:
            return BuildingType.BELT_FORWARD, Rotation.EAST
        elif curr_dx < 0:
            return BuildingType.BELT_FORWARD, Rotation.WEST
        elif curr_dy > 0:
            return BuildingType.BELT_FORWARD, Rotation.SOUTH
        else:
            return BuildingType.BELT_FORWARD, Rotation.NORTH

    def get_all_belt_segments(self) -> List[BeltSegment]:
        """Get all belt segments from all routes."""
        return list(self.belt_grid.values())
