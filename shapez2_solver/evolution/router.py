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

from ..blueprint.building_types import (
    BuildingType, Rotation, BUILDING_SPECS, BuildingSpec,
    BELT_THROUGHPUT_TIER5, get_throughput_per_second
)


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
    shape_code: Optional[str] = None  # Shape being transported (for merge compatibility)
    throughput: float = 0.75        # Items/second - default assumes tier 5 machine (45 ops/min)
    # Common tier 5 throughputs: rotator=1.5, cutter=0.75, stacker=0.5, belt=3.0


def get_machine_throughput(building_type: BuildingType) -> float:
    """Get tier 5 output throughput for a machine type (items/second)."""
    return get_throughput_per_second(building_type, tier=5)


@dataclass
class RouteResult:
    """Result of routing a connection."""
    path: List[Tuple[int, int, int]]  # List of (x, y, floor) positions
    belts: List[Tuple[int, int, int, BuildingType, Rotation]]  # (x, y, floor, type, rotation)
    success: bool
    cost: float = 0.0
    # A* search statistics for ML training
    nodes_explored: int = 0  # Number of nodes expanded during A* search
    nodes_in_open_set: int = 0  # Size of open set when search ended
    blocked_positions: List[Tuple[int, int, int]] = field(default_factory=list)  # Positions that blocked expansion


class BeltRouter:
    """Routes belts between buildings using A* pathfinding."""

    def __init__(self, grid_width: int, grid_height: int, num_floors: int = 4,
                 use_belt_ports: bool = True, max_belt_ports: int = 4,
                 allow_shape_merging: bool = False,
                 max_belt_throughput: float = None):
        """
        Initialize the belt router.

        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            num_floors: Number of floors
            use_belt_ports: Whether to allow teleporter jumps
            max_belt_ports: Max number of teleporter pairs
            allow_shape_merging: Allow routes with same shape to share belts
            max_belt_throughput: Max items/second per belt (default: 3.0 for tier 5)
        """
        # Use actual belt throughput from game data (180 ops/min = 3 items/sec)
        if max_belt_throughput is None:
            max_belt_throughput = BELT_THROUGHPUT_TIER5  # 3.0 items/second
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_floors = num_floors
        self.use_belt_ports = use_belt_ports
        self.max_belt_ports = max_belt_ports
        self.belt_ports_used = 0
        self.allow_shape_merging = allow_shape_merging
        self.occupied: Set[Tuple[int, int, int]] = set()
        self.belt_positions: Dict[Tuple[int, int, int], Tuple[BuildingType, Rotation]] = {}
        self.belt_port_pairs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []  # (sender, receiver)
        # Shape tracking for merge compatibility
        self.shape_at_cell: Dict[Tuple[int, int, int], str] = {}  # cell -> shape_code
        # Destination reachability - which cells lead to which destinations
        # Maps destination -> set of cells that are on a path to that destination
        self.cells_reaching_dest: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        # Belt capacity tracking - throughput (items/second) per cell
        self.belt_load: Dict[Tuple[int, int, int], float] = {}  # cell -> cumulative throughput
        self.max_belt_capacity = max_belt_throughput  # Max throughput (items/second) per belt

    def set_occupied(self, positions: Set[Tuple[int, int, int]]) -> None:
        """Set which grid positions are occupied by buildings."""
        self.occupied = positions.copy()

    def add_occupied(self, x: int, y: int, floor: int) -> None:
        """Mark a position as occupied."""
        self.occupied.add((x, y, floor))

    def is_valid(self, x: int, y: int, floor: int, shape_code: Optional[str] = None,
                  throughput: float = 1.0) -> bool:
        """
        Check if a position is valid for routing.

        Args:
            x, y, floor: Position to check
            shape_code: Shape being routed (for merge compatibility check)
            throughput: Throughput (items/second) this route needs

        Returns:
            True if position can be used for routing
        """
        if x < 0 or x >= self.grid_width:
            return False
        if y < 0 or y >= self.grid_height:
            return False
        if floor < 0 or floor >= self.num_floors:
            return False
        if (x, y, floor) in self.occupied:
            return False

        # Check for existing belt at this position
        cell = (x, y, floor)
        if cell in self.belt_positions:
            # Cell has a belt - can we merge?
            if not self.allow_shape_merging:
                return False  # No merging allowed

            if shape_code is None:
                return False  # Can't merge without knowing shape

            existing_shape = self.shape_at_cell.get(cell)
            if existing_shape is None:
                return False  # Unknown shape on existing belt

            # Check shape match
            if existing_shape != shape_code:
                return False

            # Check belt has capacity for this throughput
            if not self.has_belt_capacity(x, y, floor, throughput):
                return False  # Belt would exceed capacity

            return True

        return True

    def can_merge_at(self, x: int, y: int, floor: int, shape_code: str, throughput: float = 1.0) -> bool:
        """Check if we can merge onto an existing belt at this position."""
        cell = (x, y, floor)
        if cell not in self.belt_positions:
            return False  # No belt to merge onto

        # Check shape compatibility
        existing_shape = self.shape_at_cell.get(cell)
        if existing_shape != shape_code:
            return False

        # Check belt capacity - is there room for this throughput?
        if not self.has_belt_capacity(x, y, floor, throughput):
            return False  # Belt would exceed capacity

        return True

    def get_belt_load(self, x: int, y: int, floor: int) -> float:
        """Get current throughput load on a belt (items/second)."""
        return self.belt_load.get((x, y, floor), 0.0)

    def has_belt_capacity(self, x: int, y: int, floor: int, throughput: float = 1.0) -> bool:
        """Check if belt has remaining capacity for additional throughput."""
        return self.get_belt_load(x, y, floor) + throughput <= self.max_belt_capacity

    def get_remaining_capacity(self, x: int, y: int, floor: int) -> float:
        """Get remaining throughput capacity on a belt."""
        return self.max_belt_capacity - self.get_belt_load(x, y, floor)

    def cell_reaches_destination(self, cell: Tuple[int, int, int], destination: Tuple[int, int, int]) -> bool:
        """Check if a cell is on an existing path to the destination."""
        if destination not in self.cells_reaching_dest:
            return False
        return cell in self.cells_reaching_dest[destination]

    def register_path_to_destination(self, path: List[Tuple[int, int, int]], destination: Tuple[int, int, int]) -> None:
        """Register all cells in a path as reaching the destination."""
        if destination not in self.cells_reaching_dest:
            self.cells_reaching_dest[destination] = set()

        for cell in path:
            self.cells_reaching_dest[destination].add(cell)

    def get_neighbors(self, x: int, y: int, floor: int,
                       allow_belt_port: bool = False,
                       shape_code: Optional[str] = None,
                       throughput: float = 1.0) -> List[Tuple[int, int, int, Rotation, str]]:
        """
        Get valid neighboring positions with the direction to reach them.

        Args:
            x, y, floor: Current position
            allow_belt_port: Whether to include belt port jumps
            shape_code: Shape being routed (for merge compatibility)
            throughput: Throughput (items/second) this route needs

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
            if self.is_valid(nx, ny, nf, shape_code, throughput):
                neighbors.append((nx, ny, nf, direction, 'belt'))

        # Floor changes (lifts)
        if floor < self.num_floors - 1 and self.is_valid(x, y, floor + 1, shape_code, throughput):
            neighbors.append((x, y, floor + 1, Rotation.EAST, 'lift_up'))
        if floor > 0 and self.is_valid(x, y, floor - 1, shape_code, throughput):
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
                    if self.is_valid(tx, ty, floor, shape_code, throughput):
                        neighbors.append((tx, ty, floor, direction, 'belt_port'))

        return neighbors

    def heuristic(self, pos: Tuple[int, int, int], goal: Tuple[int, int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) + abs(pos[2] - goal[2]) * 2

    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int],
                   allow_belt_ports: bool = True,
                   shape_code: Optional[str] = None,
                   throughput: float = 1.0) -> Optional[List[Tuple[int, int, int, str]]]:
        """
        Find a path from start to goal using A*.

        Args:
            start: Starting position (x, y, floor)
            goal: Target position (x, y, floor)
            allow_belt_ports: Whether to allow teleporter jumps
            shape_code: Shape being routed (for merge compatibility)
            throughput: Throughput (items/second) this route needs

        Returns list of (x, y, floor, move_type) where move_type indicates how to reach that position.
        """
        if start == goal:
            return [(start[0], start[1], start[2], 'start')]

        if not self.is_valid(goal[0], goal[1], goal[2], shape_code, throughput):
            # Goal is occupied (and can't merge), try adjacent cells
            return None

        # A* algorithm
        open_set = [(0, start)]
        came_from: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], str]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int, int], float] = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            # Check if we reached the goal OR merged onto a path that reaches goal
            reached_goal = (current == goal)
            merged_to_goal = (
                self.allow_shape_merging and
                shape_code is not None and
                current != start and  # Don't count start as merge point
                self.cell_reaches_destination(current, goal) and
                self.can_merge_at(current[0], current[1], current[2], shape_code, throughput)
            )

            if reached_goal or merged_to_goal:
                # Reconstruct path with move types
                path = []
                pos = current
                while pos in came_from:
                    prev_pos, move_type = came_from[pos]
                    # Mark merge point with special move type
                    if pos == current and merged_to_goal:
                        path.append((pos[0], pos[1], pos[2], 'merge'))
                    else:
                        path.append((pos[0], pos[1], pos[2], move_type))
                    pos = prev_pos
                path.append((pos[0], pos[1], pos[2], 'start'))
                return list(reversed(path))

            # Check if we should allow belt ports for this step
            allow_port = allow_belt_ports and self.belt_ports_used < self.max_belt_ports

            for nx, ny, nf, _, move_type in self.get_neighbors(current[0], current[1], current[2],
                                                                allow_port, shape_code, throughput):
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

    def find_path_with_stats(self, start: Tuple[int, int, int], goal: Tuple[int, int, int],
                              allow_belt_ports: bool = True,
                              shape_code: Optional[str] = None,
                              throughput: float = 1.0) -> Tuple[Optional[List[Tuple[int, int, int, str]]], Dict]:
        """
        Find a path from start to goal using A*, returning search statistics for ML training.

        Returns:
            (path, stats) where stats contains:
                - nodes_explored: Number of nodes expanded
                - nodes_in_open_set: Final size of open set
                - blocked_positions: List of positions that were invalid/blocked
        """
        stats = {
            'nodes_explored': 0,
            'nodes_in_open_set': 0,
            'blocked_positions': [],
        }

        if start == goal:
            return [(start[0], start[1], start[2], 'start')], stats

        if not self.is_valid(goal[0], goal[1], goal[2], shape_code, throughput):
            stats['blocked_positions'].append(goal)
            return None, stats

        # A* algorithm with statistics tracking
        open_set = [(0, start)]
        came_from: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], str]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int, int], float] = {start: self.heuristic(start, goal)}
        closed_set: Set[Tuple[int, int, int]] = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            stats['nodes_explored'] += 1

            if current in closed_set:
                continue
            closed_set.add(current)

            # Check if we reached the goal OR merged onto a path that reaches goal
            reached_goal = (current == goal)
            merged_to_goal = (
                self.allow_shape_merging and
                shape_code is not None and
                current != start and
                self.cell_reaches_destination(current, goal) and
                self.can_merge_at(current[0], current[1], current[2], shape_code, throughput)
            )

            if reached_goal or merged_to_goal:
                stats['nodes_in_open_set'] = len(open_set)
                # Reconstruct path
                path = []
                pos = current
                while pos in came_from:
                    prev_pos, move_type = came_from[pos]
                    if pos == current and merged_to_goal:
                        path.append((pos[0], pos[1], pos[2], 'merge'))
                    else:
                        path.append((pos[0], pos[1], pos[2], move_type))
                    pos = prev_pos
                path.append((pos[0], pos[1], pos[2], 'start'))
                return list(reversed(path)), stats

            # Check if we should allow belt ports for this step
            allow_port = allow_belt_ports and self.belt_ports_used < self.max_belt_ports

            # Get neighbors and track blocked positions
            for nx, ny, nf, _, move_type in self.get_neighbors(current[0], current[1], current[2],
                                                                allow_port, shape_code, throughput):
                neighbor = (nx, ny, nf)

                if neighbor in closed_set:
                    continue

                # Belt ports have a cost penalty
                move_cost = 3 if move_type == 'belt_port' else 1
                tentative_g = g_score.get(current, float('inf')) + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, move_type)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

            # Track blocked neighbors (positions we couldn't expand to)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = current[0] + dx, current[1] + dy
                nf = current[2]
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if not self.is_valid(nx, ny, nf, shape_code, throughput):
                        blocked = (nx, ny, nf)
                        if blocked not in stats['blocked_positions'] and len(stats['blocked_positions']) < 50:
                            stats['blocked_positions'].append(blocked)

        stats['nodes_in_open_set'] = len(open_set)
        return None, stats  # No path found

    def route_connection_with_stats(self, connection: Connection) -> RouteResult:
        """Route a single connection, capturing A* search statistics for ML training."""
        start = connection.from_pos
        goal = connection.to_pos
        shape_code = connection.shape_code
        throughput = connection.throughput

        path_with_types, stats = self.find_path_with_stats(start, goal, shape_code=shape_code, throughput=throughput)

        if path_with_types is None:
            return RouteResult(
                path=[], belts=[], success=False,
                nodes_explored=stats['nodes_explored'],
                nodes_in_open_set=stats['nodes_in_open_set'],
                blocked_positions=stats['blocked_positions']
            )

        belts = self.path_to_belts(path_with_types)
        simple_path = [(p[0], p[1], p[2]) for p in path_with_types]

        # Mark new belt positions as occupied
        for x, y, floor, belt_type, rotation in belts:
            cell = (x, y, floor)
            if cell not in self.belt_positions:
                self.add_occupied(x, y, floor)
            self.belt_positions[cell] = (belt_type, rotation)
            if shape_code:
                self.shape_at_cell[cell] = shape_code
            self.belt_load[cell] = self.belt_load.get(cell, 0.0) + throughput

        self.register_path_to_destination(simple_path, goal)

        return RouteResult(
            path=simple_path,
            belts=belts,
            success=True,
            cost=len(simple_path),
            nodes_explored=stats['nodes_explored'],
            nodes_in_open_set=stats['nodes_in_open_set'],
            blocked_positions=stats['blocked_positions']
        )

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
        start = connection.from_pos
        goal = connection.to_pos
        shape_code = connection.shape_code
        throughput = connection.throughput

        path_with_types = self.find_path(start, goal, shape_code=shape_code, throughput=throughput)

        if path_with_types is None:
            return RouteResult(path=[], belts=[], success=False)

        # Check if path ends with merge (no new belt needed at merge point)
        ends_with_merge = path_with_types[-1][3] == 'merge' if path_with_types else False

        belts = self.path_to_belts(path_with_types)

        # Extract simple path (x, y, floor) for result
        simple_path = [(p[0], p[1], p[2]) for p in path_with_types]

        # Mark new belt positions as occupied, track shapes, and update load
        for x, y, floor, belt_type, rotation in belts:
            cell = (x, y, floor)
            # Only mark as occupied if not merging onto existing belt
            if cell not in self.belt_positions:
                self.add_occupied(x, y, floor)
            self.belt_positions[cell] = (belt_type, rotation)
            # Track shape at this cell
            if shape_code:
                self.shape_at_cell[cell] = shape_code
            # Add this connection's throughput to the belt load
            self.belt_load[cell] = self.belt_load.get(cell, 0.0) + throughput

        # Register this path as reaching the goal (for future merge detection)
        self.register_path_to_destination(simple_path, goal)

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
        self.shape_at_cell.clear()
        self.cells_reaching_dest.clear()
        self.belt_load.clear()
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
    from .foundation_evolution import PlacedBuilding

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


# =============================================================================
# ML-Enhanced Router
# =============================================================================

class MLEnhancedRouter(BeltRouter):
    """
    A* router enhanced with ML direction predictions.

    Uses trained direction predictor to bias pathfinding toward
    likely-good directions, improving solve speed and success rate.
    """

    # Direction mapping: ML model output -> Rotation
    ML_DIR_TO_ROTATION = {
        0: Rotation.NORTH,
        1: Rotation.EAST,
        2: Rotation.SOUTH,
        3: Rotation.WEST,
        4: None,  # none
    }

    # Rotation to delta for matching
    ROTATION_TO_DELTA = {
        Rotation.NORTH: (0, -1, 0),
        Rotation.SOUTH: (0, 1, 0),
        Rotation.EAST: (1, 0, 0),
        Rotation.WEST: (-1, 0, 0),
    }

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        use_belt_ports: bool = True,
        max_belt_ports: int = 4,
        ml_router=None,
        ml_bias_strength: float = 0.5,
    ):
        super().__init__(grid_width, grid_height, num_floors, use_belt_ports, max_belt_ports)
        self.ml_router = ml_router  # MLGuidedRouter instance
        self.ml_bias_strength = ml_bias_strength  # How much to favor ML predictions (0-1)
        self._direction_cache = {}  # Cache for ML predictions
        self._inputs_2d = []
        self._outputs_2d = []

    def set_io_positions(
        self,
        inputs: List[Tuple[int, int]],
        outputs: List[Tuple[int, int]],
    ):
        """Set input/output positions for ML direction prediction."""
        self._inputs_2d = inputs
        self._outputs_2d = outputs
        self._direction_cache.clear()
        self._precomputed_grid = None

    def precompute_directions(self):
        """Pre-compute direction predictions for entire grid (faster)."""
        if self.ml_router is None or not self._outputs_2d:
            return

        occupied_2d = {(ox, oy) for (ox, oy, _) in self.occupied}

        # Use batch prediction for entire grid
        if hasattr(self.ml_router, 'direction_predictor') and self.ml_router.direction_predictor:
            self._precomputed_grid = self.ml_router.direction_predictor.predict_grid(
                self.grid_width, self.grid_height,
                occupied_2d, self._inputs_2d, self._outputs_2d
            )

    def _get_ml_direction(self, x: int, y: int) -> Tuple[int, float]:
        """
        Get ML-predicted direction for a cell.

        Returns:
            (direction_idx, confidence) where direction_idx is 0-4
        """
        if self.ml_router is None:
            return 4, 0.0  # none, no confidence

        # Use precomputed grid if available (much faster)
        if hasattr(self, '_precomputed_grid') and self._precomputed_grid is not None:
            if 0 <= y < self._precomputed_grid.shape[0] and 0 <= x < self._precomputed_grid.shape[1]:
                dir_idx = int(self._precomputed_grid[y, x])
                return dir_idx, 0.8  # Fixed confidence for batch prediction
            return 4, 0.0

        # Fall back to per-cell prediction with caching
        cache_key = (x, y)
        if cache_key in self._direction_cache:
            return self._direction_cache[cache_key]

        occupied_2d = {(ox, oy) for (ox, oy, _) in self.occupied}

        direction, confidence = self.ml_router.get_direction_at(
            x, y,
            self.grid_width, self.grid_height,
            occupied_2d,
            self._inputs_2d,
            self._outputs_2d,
        )

        self._direction_cache[cache_key] = (direction, confidence)
        return direction, confidence

    def heuristic(self, pos: Tuple[int, int, int], goal: Tuple[int, int, int]) -> float:
        """
        Enhanced heuristic that incorporates ML direction predictions.

        Base: Manhattan distance
        ML bias: Penalty for moving against predicted direction
        """
        base_h = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) + abs(pos[2] - goal[2]) * 2

        if self.ml_router is None or self.ml_bias_strength == 0:
            return base_h

        # Get ML-predicted direction at this position
        ml_dir, confidence = self._get_ml_direction(pos[0], pos[1])

        if ml_dir == 4 or confidence < 0.3:  # none or low confidence
            return base_h

        # Calculate direction to goal
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]

        # Get the ML-predicted rotation
        ml_rotation = self.ML_DIR_TO_ROTATION.get(ml_dir)
        if ml_rotation is None:
            return base_h

        # Check if goal direction aligns with ML prediction
        ml_delta = self.ROTATION_TO_DELTA.get(ml_rotation, (0, 0, 0))

        # Calculate alignment: positive if aligned, negative if opposite
        alignment = 0
        if ml_delta[0] != 0:
            alignment += ml_delta[0] * (1 if dx > 0 else -1 if dx < 0 else 0)
        if ml_delta[1] != 0:
            alignment += ml_delta[1] * (1 if dy > 0 else -1 if dy < 0 else 0)

        # Apply bias: reduce heuristic if aligned, increase if misaligned
        bias = -alignment * self.ml_bias_strength * confidence * 0.5

        return max(0, base_h + bias)

    def find_path_ml_guided(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        allow_belt_ports: bool = True,
    ) -> Optional[List[Tuple[int, int, int, str]]]:
        """
        Find path with ML-guided neighbor ordering.

        Explores ML-predicted directions first to find paths faster.
        """
        if start == goal:
            return [(start[0], start[1], start[2], 'start')]

        if not self.is_valid(goal[0], goal[1], goal[2]):
            return None

        import heapq

        # A* with ML-guided tie-breaking
        open_set = [(0, 0, start)]  # (f_score, -ml_score, position)
        came_from: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], str]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start: 0}

        counter = 0  # For stable sorting

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                pos = current
                while pos in came_from:
                    prev_pos, move_type = came_from[pos]
                    path.append((pos[0], pos[1], pos[2], move_type))
                    pos = prev_pos
                path.append((pos[0], pos[1], pos[2], 'start'))
                return list(reversed(path))

            allow_port = allow_belt_ports and self.belt_ports_used < self.max_belt_ports

            # Get neighbors with ML scoring for ordering
            neighbors = self.get_neighbors(current[0], current[1], current[2], allow_port)

            # Score neighbors by ML prediction alignment
            scored_neighbors = []
            for nx, ny, nf, _, move_type in neighbors:
                neighbor = (nx, ny, nf)

                # Get ML direction preference for current cell
                ml_dir, confidence = self._get_ml_direction(current[0], current[1])

                # Calculate how well this move aligns with ML prediction
                dx = nx - current[0]
                dy = ny - current[1]

                ml_score = 0
                if ml_dir < 4 and confidence > 0.3:
                    ml_rotation = self.ML_DIR_TO_ROTATION.get(ml_dir)
                    if ml_rotation:
                        ml_delta = self.ROTATION_TO_DELTA.get(ml_rotation, (0, 0, 0))
                        if (ml_delta[0] == dx and ml_delta[1] == dy):
                            ml_score = confidence

                scored_neighbors.append((neighbor, move_type, ml_score))

            # Sort by ML score (higher first) for exploration priority
            scored_neighbors.sort(key=lambda x: -x[2])

            for neighbor, move_type, ml_score in scored_neighbors:
                if move_type == 'belt_port':
                    move_cost = 3
                else:
                    move_cost = 1

                # Slight bonus for following ML prediction
                if ml_score > 0.5:
                    move_cost *= (1 - 0.1 * ml_score)

                tentative_g = g_score.get(current, float('inf')) + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, move_type)
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f, -ml_score, neighbor))

        return None

    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int],
                   allow_belt_ports: bool = True) -> Optional[List[Tuple[int, int, int, str]]]:
        """
        Override find_path to use ML guidance when available.
        """
        if self.ml_router is not None:
            return self.find_path_ml_guided(start, goal, allow_belt_ports)
        return super().find_path(start, goal, allow_belt_ports)


def create_ml_router(
    grid_width: int,
    grid_height: int,
    num_floors: int = 4,
    model_dir: str = None,
) -> MLEnhancedRouter:
    """
    Create an ML-enhanced router with trained models.

    Args:
        grid_width: Width of the grid
        grid_height: Height of the grid
        num_floors: Number of floors
        model_dir: Directory containing trained models (default: shapez2_solver/learning/models)

    Returns:
        MLEnhancedRouter instance
    """
    from pathlib import Path

    if model_dir is None:
        # Default to learning/models directory
        model_dir = Path(__file__).parent.parent / "learning" / "models"
    else:
        model_dir = Path(model_dir)

    ml_router = None

    try:
        from ..learning.ml_models import MLGuidedRouter

        solvability_path = model_dir / "solvability_classifier.pkl"
        direction_path = model_dir / "direction_predictor.pkl"

        ml_router = MLGuidedRouter(
            solvability_model_path=str(solvability_path) if solvability_path.exists() else None,
            direction_model_path=str(direction_path) if direction_path.exists() else None,
        )

        print(f"ML-enhanced router loaded:")
        print(f"  Solvability model: {solvability_path.exists()}")
        print(f"  Direction model: {direction_path.exists()}")

    except ImportError as e:
        print(f"ML models not available: {e}")

    return MLEnhancedRouter(
        grid_width=grid_width,
        grid_height=grid_height,
        num_floors=num_floors,
        ml_router=ml_router,
    )
