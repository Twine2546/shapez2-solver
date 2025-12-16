"""
A* Pathfinding Router for connecting buildings with belts.

This module provides deterministic belt routing between buildings using A* pathfinding.
The evolution algorithm determines machine positions and connections, then the router
places the actual belt paths.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS, BuildingSpec


@dataclass
class GridNode:
    """A node in the pathfinding grid."""
    x: int
    y: int
    floor: int

    def __hash__(self):
        return hash((self.x, self.y, self.floor))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.floor == other.floor

    def __lt__(self, other):
        # For heap comparison
        return (self.x, self.y, self.floor) < (other.x, other.y, other.floor)


@dataclass
class Connection:
    """A connection between two points that needs to be routed."""
    from_pos: Tuple[int, int, int]  # (x, y, floor)
    to_pos: Tuple[int, int, int]    # (x, y, floor)
    from_direction: Rotation        # Direction items exit from source
    to_direction: Rotation          # Direction items should enter target
    priority: int = 0               # Higher priority routes first


@dataclass
class RouteResult:
    """Result of routing a connection."""
    path: List[Tuple[int, int, int]]  # List of (x, y, floor) positions
    belts: List[Tuple[int, int, int, BuildingType, Rotation]]  # (x, y, floor, type, rotation)
    success: bool
    cost: float = 0.0


class BeltRouter:
    """Routes belts between buildings using A* pathfinding."""

    def __init__(self, grid_width: int, grid_height: int, num_floors: int = 4,
                 use_belt_ports: bool = True, max_belt_ports: int = 4):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_floors = num_floors
        self.use_belt_ports = use_belt_ports
        self.max_belt_ports = max_belt_ports
        self.belt_ports_used = 0
        self.occupied: Set[Tuple[int, int, int]] = set()
        self.belt_positions: Dict[Tuple[int, int, int], Tuple[BuildingType, Rotation]] = {}
        self.belt_port_pairs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []  # (sender, receiver)

    def set_occupied(self, positions: Set[Tuple[int, int, int]]) -> None:
        """Set which grid positions are occupied by buildings."""
        self.occupied = positions.copy()

    def add_occupied(self, x: int, y: int, floor: int) -> None:
        """Mark a position as occupied."""
        self.occupied.add((x, y, floor))

    def is_valid(self, x: int, y: int, floor: int) -> bool:
        """Check if a position is valid and not occupied."""
        if x < 0 or x >= self.grid_width:
            return False
        if y < 0 or y >= self.grid_height:
            return False
        if floor < 0 or floor >= self.num_floors:
            return False
        if (x, y, floor) in self.occupied:
            return False
        return True

    def get_neighbors(self, x: int, y: int, floor: int,
                       allow_belt_port: bool = False) -> List[Tuple[int, int, int, Rotation, str]]:
        """
        Get valid neighboring positions with the direction to reach them.

        Returns list of (x, y, floor, rotation, move_type) where move_type is:
        - 'belt': regular belt movement
        - 'lift_up': lift going up
        - 'lift_down': lift going down
        - 'belt_port': teleporter jump
        """
        neighbors = []

        # Cardinal directions on same floor
        directions = [
            (1, 0, 0, Rotation.EAST),
            (-1, 0, 0, Rotation.WEST),
            (0, 1, 0, Rotation.SOUTH),
            (0, -1, 0, Rotation.NORTH),
        ]

        for dx, dy, df, direction in directions:
            nx, ny, nf = x + dx, y + dy, floor + df
            if self.is_valid(nx, ny, nf):
                neighbors.append((nx, ny, nf, direction, 'belt'))

        # Floor changes (lifts)
        if floor < self.num_floors - 1 and self.is_valid(x, y, floor + 1):
            neighbors.append((x, y, floor + 1, Rotation.EAST, 'lift_up'))
        if floor > 0 and self.is_valid(x, y, floor - 1):
            neighbors.append((x, y, floor - 1, Rotation.EAST, 'lift_down'))

        # Belt port jumps (teleporters) - only if enabled and we have ports available
        # Constraints: max 4 squares, straight line only (no diagonal), same floor
        if allow_belt_port and self.use_belt_ports and self.belt_ports_used < self.max_belt_ports:
            max_jump = 4
            # Check each cardinal direction for valid jump destinations
            for dx, dy, direction in [(1, 0, Rotation.EAST), (-1, 0, Rotation.WEST),
                                       (0, 1, Rotation.SOUTH), (0, -1, Rotation.NORTH)]:
                for dist in range(2, max_jump + 1):  # Min 2 squares to be useful
                    tx, ty = x + dx * dist, y + dy * dist
                    if self.is_valid(tx, ty, floor):
                        neighbors.append((tx, ty, floor, direction, 'belt_port'))

        return neighbors

    def heuristic(self, pos: Tuple[int, int, int], goal: Tuple[int, int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) + abs(pos[2] - goal[2]) * 2

    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int],
                   allow_belt_ports: bool = True) -> Optional[List[Tuple[int, int, int, str]]]:
        """
        Find a path from start to goal using A*.

        Returns list of (x, y, floor, move_type) where move_type indicates how to reach that position.
        """
        if start == goal:
            return [(start[0], start[1], start[2], 'start')]

        if not self.is_valid(goal[0], goal[1], goal[2]):
            # Goal is occupied, try adjacent cells
            return None

        # A* algorithm
        open_set = [(0, start)]
        came_from: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], str]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int, int], float] = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path with move types
                path = []
                pos = current
                while pos in came_from:
                    prev_pos, move_type = came_from[pos]
                    path.append((pos[0], pos[1], pos[2], move_type))
                    pos = prev_pos
                path.append((pos[0], pos[1], pos[2], 'start'))
                return list(reversed(path))

            # Check if we should allow belt ports for this step
            allow_port = allow_belt_ports and self.belt_ports_used < self.max_belt_ports

            for nx, ny, nf, _, move_type in self.get_neighbors(current[0], current[1], current[2], allow_port):
                neighbor = (nx, ny, nf)

                # Belt ports have a cost penalty to prefer regular belts
                if move_type == 'belt_port':
                    move_cost = 3  # Higher cost to discourage overuse
                else:
                    move_cost = 1

                tentative_g = g_score.get(current, float('inf')) + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, move_type)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def path_to_belts(self, path: List[Tuple[int, int, int, str]]) -> List[Tuple[int, int, int, BuildingType, Rotation]]:
        """
        Convert a path to belt placements.

        Path format: list of (x, y, floor, move_type)
        """
        if len(path) < 2:
            return []

        belts = []
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            move_type = next_pos[3]  # How we got to next_pos

            curr_pos = (curr[0], curr[1], curr[2])
            next_xyz = (next_pos[0], next_pos[1], next_pos[2])

            dx = next_xyz[0] - curr_pos[0]
            dy = next_xyz[1] - curr_pos[1]
            dz = next_xyz[2] - curr_pos[2]

            # Handle belt port teleportation
            if move_type == 'belt_port':
                # Determine direction of the jump (straight line only)
                if dx > 0:
                    rotation = Rotation.EAST
                elif dx < 0:
                    rotation = Rotation.WEST
                elif dy > 0:
                    rotation = Rotation.SOUTH
                else:
                    rotation = Rotation.NORTH

                # Place sender at current position (input from behind, output into teleport)
                belts.append((curr_pos[0], curr_pos[1], curr_pos[2],
                             BuildingType.BELT_PORT_SENDER, rotation))
                # Place receiver at next position (receives teleport, outputs forward)
                belts.append((next_xyz[0], next_xyz[1], next_xyz[2],
                             BuildingType.BELT_PORT_RECEIVER, rotation))
                # Track the pair
                self.belt_port_pairs.append((curr_pos, next_xyz))
                self.belt_ports_used += 1
                continue

            # Determine belt type and rotation for regular moves
            if move_type == 'lift_up' or dz > 0:
                belt_type = BuildingType.LIFT_UP
                rotation = Rotation.EAST
            elif move_type == 'lift_down' or dz < 0:
                belt_type = BuildingType.LIFT_DOWN
                rotation = Rotation.EAST
            elif dx > 0:
                belt_type = BuildingType.BELT_FORWARD
                rotation = Rotation.EAST
            elif dx < 0:
                belt_type = BuildingType.BELT_FORWARD
                rotation = Rotation.WEST
            elif dy > 0:
                belt_type = BuildingType.BELT_FORWARD
                rotation = Rotation.SOUTH
            else:
                belt_type = BuildingType.BELT_FORWARD
                rotation = Rotation.NORTH

            # Check if we need a turn belt
            if i > 0:
                prev = path[i - 1]
                prev_dx = curr_pos[0] - prev[0]
                prev_dy = curr_pos[1] - prev[1]

                # If direction changed, might need a turn belt
                if (prev_dx != 0 and dy != 0) or (prev_dy != 0 and dx != 0):
                    # Determine turn direction
                    if (prev_dx > 0 and dy > 0) or (prev_dy < 0 and dx > 0):
                        belt_type = BuildingType.BELT_RIGHT
                    elif (prev_dx > 0 and dy < 0) or (prev_dy > 0 and dx > 0):
                        belt_type = BuildingType.BELT_LEFT
                    elif (prev_dx < 0 and dy > 0) or (prev_dy < 0 and dx < 0):
                        belt_type = BuildingType.BELT_LEFT
                    elif (prev_dx < 0 and dy < 0) or (prev_dy > 0 and dx < 0):
                        belt_type = BuildingType.BELT_RIGHT

            belts.append((curr_pos[0], curr_pos[1], curr_pos[2], belt_type, rotation))

        return belts

    def route_connection(self, connection: Connection) -> RouteResult:
        """Route a single connection."""
        # Find adjacent cell to start from (in the exit direction)
        start = connection.from_pos
        goal = connection.to_pos

        path_with_types = self.find_path(start, goal)

        if path_with_types is None:
            return RouteResult(path=[], belts=[], success=False)

        belts = self.path_to_belts(path_with_types)

        # Extract simple path (x, y, floor) for result
        simple_path = [(p[0], p[1], p[2]) for p in path_with_types]

        # Mark path as occupied
        for x, y, floor, _, _ in belts:
            self.add_occupied(x, y, floor)
            self.belt_positions[(x, y, floor)] = (belts[0][3], belts[0][4])  # Store belt info

        return RouteResult(
            path=simple_path,
            belts=belts,
            success=True,
            cost=len(simple_path)
        )

    def route_all(self, connections: List[Connection]) -> List[RouteResult]:
        """Route all connections, prioritizing by priority value."""
        # Sort by priority (higher first)
        sorted_connections = sorted(connections, key=lambda c: -c.priority)

        results = []
        for conn in sorted_connections:
            result = self.route_connection(conn)
            results.append(result)

        return results

    def clear(self) -> None:
        """Clear all routing state."""
        self.belt_positions.clear()
        self.belt_port_pairs.clear()
        self.belt_ports_used = 0
        # Note: doesn't clear occupied - call set_occupied to reset


def route_candidate_connections(
    candidate,  # Candidate from foundation_evolution
    config,     # FoundationConfig
    connections: List[Tuple[int, int, int, int]]  # (from_building_id, from_output, to_building_id, to_input)
) -> List[Tuple[int, int, int, BuildingType, Rotation]]:
    """
    Route connections for a candidate solution.

    Args:
        candidate: The candidate with building placements
        config: Foundation configuration
        connections: List of (from_building_id, from_output_idx, to_building_id, to_input_idx)

    Returns:
        List of belt placements (x, y, floor, building_type, rotation)
    """
    from .core import PlacedBuilding

    router = BeltRouter(config.spec.grid_width, config.spec.grid_height, config.spec.num_floors)

    # Mark all building cells as occupied
    occupied = set()
    building_map = {}
    for building in candidate.buildings:
        building_map[building.building_id] = building
        spec = BUILDING_SPECS.get(building.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
        for dx in range(spec.width):
            for dy in range(spec.height):
                for df in range(spec.depth):
                    occupied.add((building.x + dx, building.y + dy, building.floor + df))

    router.set_occupied(occupied)

    # Convert connections to routing format
    route_connections = []
    for from_id, from_out, to_id, to_in in connections:
        from_b = building_map.get(from_id)
        to_b = building_map.get(to_id)

        if not from_b or not to_b:
            continue

        from_spec = BUILDING_SPECS.get(from_b.building_type)
        to_spec = BUILDING_SPECS.get(to_b.building_type)

        if not from_spec or not to_spec:
            continue

        # Get output position of from building
        # Simplified: use building edge based on rotation
        if from_b.rotation == Rotation.EAST:
            from_pos = (from_b.x + from_spec.width, from_b.y + from_spec.height // 2, from_b.floor)
            from_dir = Rotation.EAST
        elif from_b.rotation == Rotation.WEST:
            from_pos = (from_b.x - 1, from_b.y + from_spec.height // 2, from_b.floor)
            from_dir = Rotation.WEST
        elif from_b.rotation == Rotation.SOUTH:
            from_pos = (from_b.x + from_spec.width // 2, from_b.y + from_spec.height, from_b.floor)
            from_dir = Rotation.SOUTH
        else:  # NORTH
            from_pos = (from_b.x + from_spec.width // 2, from_b.y - 1, from_b.floor)
            from_dir = Rotation.NORTH

        # Get input position of to building
        if to_b.rotation == Rotation.EAST:
            to_pos = (to_b.x - 1, to_b.y + to_spec.height // 2, to_b.floor)
            to_dir = Rotation.EAST
        elif to_b.rotation == Rotation.WEST:
            to_pos = (to_b.x + to_spec.width, to_b.y + to_spec.height // 2, to_b.floor)
            to_dir = Rotation.WEST
        elif to_b.rotation == Rotation.SOUTH:
            to_pos = (to_b.x + to_spec.width // 2, to_b.y - 1, to_b.floor)
            to_dir = Rotation.SOUTH
        else:  # NORTH
            to_pos = (to_b.x + to_spec.width // 2, to_b.y + to_spec.height, to_b.floor)
            to_dir = Rotation.NORTH

        route_connections.append(Connection(
            from_pos=from_pos,
            to_pos=to_pos,
            from_direction=from_dir,
            to_direction=to_dir,
            priority=1
        ))

    # Route all connections
    results = router.route_all(route_connections)

    # Collect all belt placements
    all_belts = []
    for result in results:
        if result.success:
            all_belts.extend(result.belts)

    return all_belts
