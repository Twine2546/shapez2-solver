"""
A* Pathfinding Router for connecting buildings with belts.

This module provides deterministic belt routing between buildings using A* pathfinding.
The evolution algorithm determines machine positions and connections, then the router
places the actual belt paths.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from enum import Enum

from ..blueprint.building_types import (
    BuildingType, Rotation, BUILDING_SPECS, BuildingSpec,
    BELT_THROUGHPUT_TIER5, get_throughput_per_second
)

# Type aliases for pluggable functions
HeuristicFn = Callable[[Tuple[int, int, int], Tuple[int, int, int], Optional[Dict[str, Any]]], float]
MoveCostFn = Callable[[Tuple[int, int, int], Tuple[int, int, int], str, Optional[Dict[str, Any]]], float]


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
                 max_belt_throughput: float = None,
                 valid_cells: Optional[Set[Tuple[int, int]]] = None,
                 heuristic_fn: Optional[HeuristicFn] = None,
                 move_cost_fn: Optional[MoveCostFn] = None):
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
            valid_cells: For irregular foundations, set of valid (x, y) positions
            heuristic_fn: Custom heuristic function for A* (default: Manhattan distance)
            move_cost_fn: Custom move cost function (default: 1.0 for horizontal, 1.5 for lifts)
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
        self.debug = False  # Set via set_debug()
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
        # For irregular foundations (L, T, Cross, etc.)
        self.valid_cells = valid_cells

        # Pluggable evaluation functions
        self._custom_heuristic = heuristic_fn
        self._custom_move_cost = move_cost_fn

        # Routing context for ML functions (set during route_all)
        self._all_connections: List[Connection] = []
        self._remaining_connections: List[Connection] = []
        self._current_connection_index: int = 0

        # === Conflict tracking for ML analysis ===
        # Track which connection placed which belts (for conflict analysis)
        self.belt_owner: Dict[Tuple[int, int, int], int] = {}  # cell -> connection_index
        self.connection_belts: Dict[int, List[Tuple[int, int, int]]] = {}  # conn_idx -> list of cells
        self.connection_paths: Dict[int, List[Tuple[int, int, int]]] = {}  # conn_idx -> path
        self.failed_connections: List[Dict] = []  # List of failed connection info with blockers

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug logging."""
        self.debug = enabled

    def _debug(self, msg: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"      [A*] {msg}")

    def set_occupied(self, positions: Set[Tuple[int, int, int]]) -> None:
        """Set which grid positions are occupied by buildings."""
        self.occupied = positions.copy()
        if self.debug:
            print(f"      [A*] Occupied cells: {len(self.occupied)}")

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
        # For irregular foundations, check if (x, y) is in valid area
        if self.valid_cells is not None and (x, y) not in self.valid_cells:
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
        # LIFT_UP: occupies (x, y, floor) and (x, y, floor+1), outputs EAST on upper floor
        # So the destination is (x+1, y, floor+1) - one step east on the upper floor
        # LIFT_DOWN: occupies (x, y, floor) and (x, y, floor-1), outputs EAST on lower floor
        # So the destination is (x+1, y, floor-1)
        if floor < self.num_floors - 1 and self.is_valid(x + 1, y, floor + 1, shape_code, throughput):
            # Check that the lift positions are also valid (lift occupies both floors)
            if self.is_valid(x, y, floor, shape_code, throughput) and self.is_valid(x, y, floor + 1, shape_code, throughput):
                neighbors.append((x + 1, y, floor + 1, Rotation.EAST, 'lift_up'))
        if floor > 0 and self.is_valid(x + 1, y, floor - 1, shape_code, throughput):
            if self.is_valid(x, y, floor, shape_code, throughput) and self.is_valid(x, y, floor - 1, shape_code, throughput):
                neighbors.append((x + 1, y, floor - 1, Rotation.EAST, 'lift_down'))

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

    def heuristic(self, pos: Tuple[int, int, int], goal: Tuple[int, int, int],
                  context: Optional[Dict[str, Any]] = None) -> float:
        """
        A* heuristic function.

        Uses custom heuristic if provided, otherwise defaults to Manhattan distance.
        """
        if self._custom_heuristic is not None:
            return self._custom_heuristic(pos, goal, context)
        # Default: Manhattan distance with floor weight of 2
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) + abs(pos[2] - goal[2]) * 2

    def move_cost(self, current: Tuple[int, int, int], neighbor: Tuple[int, int, int],
                  move_type: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate the cost of moving from current to neighbor.

        Uses custom move cost if provided, otherwise defaults to standard costs.
        """
        if self._custom_move_cost is not None:
            return self._custom_move_cost(current, neighbor, move_type, context)
        # Default costs
        if move_type in ('lift_up', 'lift_down'):
            return 1.5
        elif move_type == 'belt_port':
            distance = abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1])
            return 2.0 + distance * 0.1
        else:
            return 1.0

    def _build_context(self, goal: Tuple[int, int, int]) -> Dict[str, Any]:
        """Build context dict for ML heuristic/cost functions."""
        # Get all connection endpoints for global awareness
        all_goals = [c.to_pos for c in self._all_connections]

        return {
            'occupied': self.occupied,
            'belt_positions': self.belt_positions,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'num_floors': self.num_floors,
            'remaining_connections': self._remaining_connections,
            'connection_index': self._current_connection_index,
            'all_goals': all_goals,
            'current_goal': goal,
        }

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
        if self.debug:
            self._debug(f"find_path: {start} -> {goal}")

        if start == goal:
            if self.debug:
                self._debug(f"  start == goal, trivial path")
            return [(start[0], start[1], start[2], 'start')]

        if not self.is_valid(goal[0], goal[1], goal[2], shape_code, throughput):
            if self.debug:
                self._debug(f"  FAIL: goal {goal} is blocked/invalid")
                if goal in self.occupied:
                    self._debug(f"    (goal is in occupied set)")
            return None

        # Check if start is valid (not inside a machine)
        if not self.is_valid(start[0], start[1], start[2], shape_code, throughput):
            if self.debug:
                self._debug(f"  FAIL: start {start} is blocked/invalid")
                if start in self.occupied:
                    self._debug(f"    (start is in occupied set)")
            return None

        # A* algorithm
        open_set = [(0, start)]
        nodes_explored = 0
        came_from: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], str]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start: 0}
        context = self._build_context(goal)
        f_score: Dict[Tuple[int, int, int], float] = {start: self.heuristic(start, goal, context)}

        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_explored += 1

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
                result = list(reversed(path))
                if self.debug:
                    method = "merged" if merged_to_goal else "direct"
                    self._debug(f"  SUCCESS ({method}): {len(result)} steps, {nodes_explored} nodes explored")
                    self._debug(f"  Path: {' -> '.join(f'({p[0]},{p[1]})' for p in result)}")
                return result

            # Check if we should allow belt ports for this step
            # Don't allow belt_port if we arrived here via belt_port (can't place sender on receiver)
            arrived_via_belt_port = current in came_from and came_from[current][1] == 'belt_port'
            allow_port = allow_belt_ports and self.belt_ports_used < self.max_belt_ports and not arrived_via_belt_port

            for nx, ny, nf, _, move_type in self.get_neighbors(current[0], current[1], current[2],
                                                                allow_port, shape_code, throughput):
                neighbor = (nx, ny, nf)

                # Use pluggable move cost function
                cost = self.move_cost(current, neighbor, move_type, context)

                tentative_g = g_score.get(current, float('inf')) + cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, move_type)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal, context)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        if self.debug:
            self._debug(f"  FAIL: no path found after {nodes_explored} nodes")
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
        context = self._build_context(goal)
        f_score: Dict[Tuple[int, int, int], float] = {start: self.heuristic(start, goal, context)}
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

                # Use pluggable move cost function
                cost = self.move_cost(current, neighbor, move_type, context)
                tentative_g = g_score.get(current, float('inf')) + cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, move_type)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal, context)
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

    def route_connection_indexed(
        self,
        connection: Connection,
        connection_index: int,
    ) -> RouteResult:
        """
        Route a connection while tracking ownership for conflict analysis.

        Args:
            connection: The connection to route
            connection_index: Index of this connection (for tracking)

        Returns:
            RouteResult with conflict analysis data
        """
        start = connection.from_pos
        goal = connection.to_pos
        shape_code = connection.shape_code
        throughput = connection.throughput

        path_with_types, stats = self.find_path_with_stats(
            start, goal, shape_code=shape_code, throughput=throughput
        )

        if path_with_types is None:
            # Analyze which connections are blocking this one
            blocking_connections = self._analyze_blockers(
                start, goal, stats['blocked_positions']
            )

            self.failed_connections.append({
                'index': connection_index,
                'src': start,
                'dst': goal,
                'blocked_positions': stats['blocked_positions'],
                'blocking_connections': blocking_connections,
                'nodes_explored': stats['nodes_explored'],
            })

            return RouteResult(
                path=[], belts=[], success=False,
                nodes_explored=stats['nodes_explored'],
                nodes_in_open_set=stats['nodes_in_open_set'],
                blocked_positions=stats['blocked_positions']
            )

        belts = self.path_to_belts(path_with_types)
        simple_path = [(p[0], p[1], p[2]) for p in path_with_types]

        # Track belt ownership
        belt_cells = []
        for x, y, floor, belt_type, rotation in belts:
            cell = (x, y, floor)
            belt_cells.append(cell)

            if cell not in self.belt_positions:
                self.add_occupied(x, y, floor)
                # Track ownership (only for new belts, not merged)
                self.belt_owner[cell] = connection_index

            self.belt_positions[cell] = (belt_type, rotation)
            if shape_code:
                self.shape_at_cell[cell] = shape_code
            self.belt_load[cell] = self.belt_load.get(cell, 0.0) + throughput

        # Store connection info
        self.connection_belts[connection_index] = belt_cells
        self.connection_paths[connection_index] = simple_path

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

    def _analyze_blockers(
        self,
        src: Tuple[int, int, int],
        dst: Tuple[int, int, int],
        blocked_positions: List[Tuple[int, int, int]],
    ) -> Dict[int, int]:
        """
        Analyze which connections are blocking a failed route.

        Returns:
            Dict mapping connection_index -> number of blocking cells from that connection
        """
        blocking_counts: Dict[int, int] = {}

        for pos in blocked_positions:
            if pos in self.belt_owner:
                owner = self.belt_owner[pos]
                blocking_counts[owner] = blocking_counts.get(owner, 0) + 1

        return blocking_counts

    def get_conflict_analysis(self) -> Dict:
        """
        Get analysis of routing conflicts for ML training.

        Returns dict with:
            - failed_connections: List of failed connections with blocking info
            - blocking_scores: Score for each successful connection (how many others it blocks)
            - reroute_suggestions: Which connections should be rerouted
        """
        # Calculate blocking scores for each successful connection
        blocking_scores: Dict[int, int] = {}
        for conn_idx in self.connection_belts:
            blocking_scores[conn_idx] = 0

        for failed in self.failed_connections:
            for blocker_idx, count in failed['blocking_connections'].items():
                blocking_scores[blocker_idx] = blocking_scores.get(blocker_idx, 0) + count

        # Identify connections that should be reconsidered
        reroute_suggestions = []
        for conn_idx, score in sorted(blocking_scores.items(), key=lambda x: -x[1]):
            if score > 0:
                reroute_suggestions.append({
                    'connection_index': conn_idx,
                    'blocking_score': score,
                    'num_belts': len(self.connection_belts.get(conn_idx, [])),
                    'path_length': len(self.connection_paths.get(conn_idx, [])),
                    'blocks_connections': [
                        f['index'] for f in self.failed_connections
                        if conn_idx in f['blocking_connections']
                    ],
                })

        return {
            'failed_connections': self.failed_connections,
            'blocking_scores': blocking_scores,
            'reroute_suggestions': reroute_suggestions,
            'total_failed': len(self.failed_connections),
            'total_routed': len(self.connection_belts),
        }

    def get_reroute_order(self) -> List[int]:
        """
        Get suggested order for re-routing to resolve conflicts.

        Returns list of connection indices to reroute, starting with
        the ones that block the most failed connections.
        """
        analysis = self.get_conflict_analysis()

        # Sort by blocking score (highest first - these should be rerouted)
        return [s['connection_index'] for s in analysis['reroute_suggestions']]

    def remove_connection_belts(self, connection_index: int) -> List[Tuple[int, int, int]]:
        """
        Remove all belts placed by a specific connection.

        Args:
            connection_index: Index of the connection to remove

        Returns:
            List of cells that were cleared
        """
        if connection_index not in self.connection_belts:
            return []

        cleared_cells = []
        for cell in self.connection_belts[connection_index]:
            # Remove from occupied
            if cell in self.occupied:
                self.occupied.discard(cell)
            # Remove from belt positions
            if cell in self.belt_positions:
                del self.belt_positions[cell]
            # Remove from belt owner
            if cell in self.belt_owner:
                del self.belt_owner[cell]
            # Clear shape tracking
            if cell in self.shape_at_cell:
                del self.shape_at_cell[cell]
            # Clear belt load
            if cell in self.belt_load:
                del self.belt_load[cell]

            cleared_cells.append(cell)

        # Clear connection tracking
        del self.connection_belts[connection_index]
        if connection_index in self.connection_paths:
            del self.connection_paths[connection_index]

        return cleared_cells

    def route_with_reroute_loop(
        self,
        connections: List[Connection],
        max_retries: int = 3,
    ) -> Tuple[List[RouteResult], int]:
        """
        Route all connections with automatic rerouting on failure.

        When routing fails, identifies blocking routes, removes them,
        routes the failed connection first, then re-routes the removed ones.

        Args:
            connections: List of connections to route
            max_retries: Maximum number of reroute attempts

        Returns:
            Tuple of (results, num_reroutes_performed)
        """
        num_connections = len(connections)
        results: List[Optional[RouteResult]] = [None] * num_connections
        reroute_count = 0

        # Initial routing order
        routing_order = list(range(num_connections))

        # Capture initial machine occupancy BEFORE the loop
        initial_occupied = self.occupied.copy()

        for attempt in range(max_retries + 1):
            # Reset belt-specific state
            self.belt_positions.clear()
            self.belt_port_pairs.clear()
            self.belt_ports_used = 0
            self.shape_at_cell.clear()
            self.cells_reaching_dest.clear()
            self.belt_load.clear()
            self.belt_owner.clear()
            self.connection_belts.clear()
            self.connection_paths.clear()
            self.failed_connections.clear()

            # Restore initial occupied (machines only)
            self.occupied = initial_occupied.copy()

            # Route in current order
            failed_indices = []
            for i in routing_order:
                conn = connections[i]
                result = self.route_connection_indexed(conn, i)
                results[i] = result
                if not result.success:
                    failed_indices.append(i)

            # If all succeeded, we're done
            if not failed_indices:
                return [r for r in results if r is not None], reroute_count

            # If this was the last attempt, return what we have
            if attempt >= max_retries:
                break

            # Get conflict analysis to find what to reroute
            analysis = self.get_conflict_analysis()

            if not analysis['reroute_suggestions']:
                # No suggestions, nothing we can do
                break

            # Find the connection(s) blocking the most failed routes
            blockers_to_remove = set()
            for suggestion in analysis['reroute_suggestions'][:2]:  # Top 2 blockers
                blocker_idx = suggestion['connection_index']
                if blocker_idx not in failed_indices:
                    blockers_to_remove.add(blocker_idx)

            if not blockers_to_remove:
                # The blockers are themselves failed - can't help
                break

            # Build new routing order:
            # 1. Failed connections first (they need space)
            # 2. Then the blockers (reroute around the new paths)
            # 3. Then everything else
            new_order = []

            # Failed connections first
            for i in failed_indices:
                new_order.append(i)

            # Then blockers
            for i in blockers_to_remove:
                if i not in new_order:
                    new_order.append(i)

            # Then the rest
            for i in routing_order:
                if i not in new_order:
                    new_order.append(i)

            routing_order = new_order
            reroute_count += 1

        return [r for r in results if r is not None], reroute_count

    def route_all_with_retry(
        self,
        connections: List[Connection],
        max_retries: int = 3,
        use_smart_ports: bool = True,
    ) -> List[RouteResult]:
        """
        Route all connections with rerouting and smart belt ports.

        Combines:
        - Reroute loop (when fails, reorder and retry)
        - Smart belt ports (use jumps to avoid congestion)

        Args:
            connections: List of connections to route
            max_retries: Maximum reroute attempts
            use_smart_ports: Use smart belt port routing

        Returns:
            List of RouteResults
        """
        num_connections = len(connections)
        results: List[Optional[RouteResult]] = [None] * num_connections
        reroute_count = 0

        # Start with simple order
        routing_order = list(range(num_connections))

        # Capture initial machine occupancy BEFORE the loop
        initial_occupied = self.occupied.copy()

        for attempt in range(max_retries + 1):
            # Clear routing state
            self.belt_positions.clear()
            self.belt_port_pairs.clear()
            self.belt_ports_used = 0
            self.shape_at_cell.clear()
            self.cells_reaching_dest.clear()
            self.belt_load.clear()
            self.belt_owner.clear()
            self.connection_belts.clear()
            self.connection_paths.clear()
            self.failed_connections.clear()
            self.occupied = initial_occupied.copy()

            # Route in current order
            failed_indices = []
            for i in routing_order:
                conn = connections[i]

                # Use smart routing if enabled
                if use_smart_ports:
                    result = self.route_connection_smart(conn)
                else:
                    result = self.route_connection(conn)

                # Track ownership for conflict analysis
                if result.success:
                    self.connection_belts[i] = [(b[0], b[1], b[2]) for b in result.belts]
                    self.connection_paths[i] = result.path
                    for cell in self.connection_belts[i]:
                        self.belt_owner[cell] = i
                else:
                    # Analyze what blocked this
                    blocked_by = {}
                    for cell in self.occupied:
                        if cell in self.belt_owner:
                            owner = self.belt_owner[cell]
                            blocked_by[owner] = blocked_by.get(owner, 0) + 1

                    self.failed_connections.append({
                        'index': i,
                        'src': conn.from_pos,
                        'dst': conn.to_pos,
                        'blocking_connections': blocked_by,
                    })
                    failed_indices.append(i)

                results[i] = result

            # All succeeded?
            if not failed_indices:
                return [r for r in results if r is not None]

            # Last attempt?
            if attempt >= max_retries:
                break

            # Find blockers
            analysis = self.get_conflict_analysis()
            if not analysis['reroute_suggestions']:
                break

            # Reorder: failed first, then top blockers, then rest
            blockers = set()
            for sug in analysis['reroute_suggestions'][:2]:
                if sug['connection_index'] not in failed_indices:
                    blockers.add(sug['connection_index'])

            if not blockers:
                break

            new_order = list(failed_indices)
            for b in blockers:
                if b not in new_order:
                    new_order.append(b)
            for i in routing_order:
                if i not in new_order:
                    new_order.append(i)

            routing_order = new_order
            reroute_count += 1

        return [r for r in results if r is not None]

    def path_to_belts(self, path: List[Tuple[int, int, int, str]]) -> List[Tuple[int, int, int, BuildingType, Rotation]]:
        """
        Convert a path to belt placements.

        Path format: list of (x, y, floor, move_type)
        """
        if len(path) < 2:
            return []

        belts = []
        # Track positions that have belt port receivers (to avoid overwriting them)
        receiver_positions = set()
        # Track positions that already have belts
        belt_positions = set()

        # First pass: identify all belt port receiver positions
        for i in range(len(path) - 1):
            next_pos = path[i + 1]
            move_type = next_pos[3]
            if move_type == 'belt_port':
                receiver_positions.add((next_pos[0], next_pos[1], next_pos[2]))

        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            curr_type = curr[3] if len(curr) > 3 else None
            move_type = next_pos[3]  # How we got to next_pos

            # Skip input_hint and output_hint - they're just for direction calculation
            if curr_type in ('input_hint', 'output_hint'):
                continue

            curr_pos = (curr[0], curr[1], curr[2])
            next_xyz = (next_pos[0], next_pos[1], next_pos[2])

            # Skip if this position already has a belt port receiver
            if curr_pos in receiver_positions:
                continue

            dx = next_xyz[0] - curr_pos[0]
            dy = next_xyz[1] - curr_pos[1]
            dz = next_xyz[2] - curr_pos[2]

            # Handle belt port teleportation
            if move_type == 'belt_port':
                # Sender rotation determines teleport direction - where the receiver is
                # The flow simulator looks for receivers in the sender's facing direction
                if dx > 0:
                    sender_rotation = Rotation.EAST
                elif dx < 0:
                    sender_rotation = Rotation.WEST
                elif dy > 0:
                    sender_rotation = Rotation.SOUTH
                else:
                    sender_rotation = Rotation.NORTH

                # Check if there's an input_hint that would be blocked by this sender
                # A sender facing direction D cannot receive from direction D
                # If the input_hint is in the same direction as teleport, we need an intermediate belt
                need_intermediate_belt = False
                if i > 0:
                    prev = path[i - 1]
                    prev_type = prev[3] if len(prev) > 3 else None
                    if prev_type == 'input_hint':
                        hint_x, hint_y = prev[0], prev[1]
                        # Direction from sender to input_hint
                        hint_dx = hint_x - curr_pos[0]
                        hint_dy = hint_y - curr_pos[1]
                        # Check if input direction matches teleport direction
                        # (sender would block input from machine)
                        if (sender_rotation == Rotation.EAST and hint_dx > 0) or \
                           (sender_rotation == Rotation.WEST and hint_dx < 0) or \
                           (sender_rotation == Rotation.SOUTH and hint_dy > 0) or \
                           (sender_rotation == Rotation.NORTH and hint_dy < 0):
                            need_intermediate_belt = True

                # If input is blocked, we need to place a turn belt first to redirect flow
                # before using the belt_port
                if need_intermediate_belt:
                    # The sender would block input from the machine
                    # We need to:
                    # 1. Place a turn belt at curr_pos that receives from machine and outputs perpendicular
                    # 2. Place sender at adjacent perpendicular cell
                    # 3. Receiver stays at next_pos

                    # Determine input direction from machine (hint)
                    prev = path[i - 1]  # input_hint
                    hint_x, hint_y = prev[0], prev[1]
                    hint_dx = hint_x - curr_pos[0]
                    hint_dy = hint_y - curr_pos[1]

                    # Machine is in direction of hint, belt receives from opposite
                    # hint_dy > 0 means machine is SOUTH, belt receives from SOUTH (faces NORTH)
                    # hint_dy < 0 means machine is NORTH, belt receives from NORTH (faces SOUTH)
                    # etc.

                    # Find a valid perpendicular direction for the turn belt output
                    # The turn belt outputs perpendicular to the machine direction
                    perpendicular_candidates = []
                    if hint_dx != 0:  # Machine is E/W, perpendicular is N/S
                        perpendicular_candidates = [
                            ((curr_pos[0], curr_pos[1] - 1, curr_pos[2]), Rotation.NORTH),  # Output north
                            ((curr_pos[0], curr_pos[1] + 1, curr_pos[2]), Rotation.SOUTH),  # Output south
                        ]
                    else:  # Machine is N/S, perpendicular is E/W
                        perpendicular_candidates = [
                            ((curr_pos[0] + 1, curr_pos[1], curr_pos[2]), Rotation.EAST),   # Output east
                            ((curr_pos[0] - 1, curr_pos[1], curr_pos[2]), Rotation.WEST),   # Output west
                        ]

                    # Find a valid perpendicular neighbor that's:
                    # 1. In bounds
                    # 2. Not occupied
                    # 3. Aligned with the teleport destination (same x or same y as next_pos)
                    perp_pos = None
                    perp_dir = None
                    for (px, py, pz), pdir in perpendicular_candidates:
                        if not self.is_valid(px, py, pz):
                            continue
                        # Check alignment with receiver for teleport
                        if px == next_xyz[0] or py == next_xyz[1]:
                            perp_pos = (px, py, pz)
                            perp_dir = pdir
                            break

                    if perp_pos is not None:
                        # Determine turn belt rotation: faces toward perpendicular output
                        # This also determines input (opposite side)
                        # For turn belt: rotation = input direction, belt_left/right based on output
                        if hint_dy > 0:  # Machine SOUTH of us, receive from SOUTH, face NORTH
                            input_rot = Rotation.NORTH
                        elif hint_dy < 0:  # Machine NORTH of us, receive from NORTH, face SOUTH
                            input_rot = Rotation.SOUTH
                        elif hint_dx > 0:  # Machine EAST of us, receive from EAST, face WEST
                            input_rot = Rotation.WEST
                        else:  # Machine WEST of us, receive from WEST, face EAST
                            input_rot = Rotation.EAST

                        # Determine turn direction (left or right from input to output)
                        # Turn belt rotation = input direction (where items come FROM)
                        # BELT_LEFT outputs 90° CCW from input, BELT_RIGHT outputs 90° CW
                        input_dirs = [Rotation.EAST, Rotation.SOUTH, Rotation.WEST, Rotation.NORTH]
                        input_idx = input_dirs.index(input_rot)
                        output_idx = input_dirs.index(perp_dir)
                        turn = (output_idx - input_idx) % 4

                        if turn == 1:  # 90° CW = right turn
                            turn_belt_type = BuildingType.BELT_RIGHT
                        elif turn == 3:  # 90° CCW = left turn
                            turn_belt_type = BuildingType.BELT_LEFT
                        else:
                            # Shouldn't happen for perpendicular
                            turn_belt_type = BuildingType.BELT_FORWARD

                        # Place turn belt at curr_pos
                        belts.append((curr_pos[0], curr_pos[1], curr_pos[2],
                                     turn_belt_type, input_rot))

                        # Now place sender at perpendicular position, facing toward receiver
                        pdx = next_xyz[0] - perp_pos[0]
                        pdy = next_xyz[1] - perp_pos[1]
                        if pdx > 0:
                            new_sender_rot = Rotation.EAST
                        elif pdx < 0:
                            new_sender_rot = Rotation.WEST
                        elif pdy > 0:
                            new_sender_rot = Rotation.SOUTH
                        else:
                            new_sender_rot = Rotation.NORTH

                        # Place sender at perpendicular position
                        belts.append((perp_pos[0], perp_pos[1], perp_pos[2],
                                     BuildingType.BELT_PORT_SENDER, new_sender_rot))

                        # Determine receiver rotation
                        receiver_rotation = new_sender_rot
                        if i + 2 < len(path):
                            after_receiver = path[i + 2]
                            after_xyz = (after_receiver[0], after_receiver[1], after_receiver[2])
                            dx_next = after_xyz[0] - next_xyz[0]
                            dy_next = after_xyz[1] - next_xyz[1]
                            if dx_next > 0:
                                receiver_rotation = Rotation.EAST
                            elif dx_next < 0:
                                receiver_rotation = Rotation.WEST
                            elif dy_next > 0:
                                receiver_rotation = Rotation.SOUTH
                            elif dy_next < 0:
                                receiver_rotation = Rotation.NORTH

                        # Place receiver at original destination
                        belts.append((next_xyz[0], next_xyz[1], next_xyz[2],
                                     BuildingType.BELT_PORT_RECEIVER, receiver_rotation))

                        # Track the pair and mark perpendicular as occupied
                        self.belt_port_pairs.append((perp_pos, next_xyz))
                        self.belt_ports_used += 1
                        self.occupied.add(perp_pos)
                        receiver_positions.add(next_xyz)
                        continue
                    else:
                        # No aligned perpendicular found - need to redirect with turn belts
                        # and use regular belts instead of teleport
                        # This happens when start is at grid edge and teleport direction
                        # would require receiving from off-grid

                        # Determine input direction from machine
                        if hint_dy > 0:  # Machine SOUTH
                            input_rot = Rotation.NORTH  # Face north to receive from south
                        elif hint_dy < 0:  # Machine NORTH
                            input_rot = Rotation.SOUTH
                        elif hint_dx > 0:  # Machine EAST
                            input_rot = Rotation.WEST
                        else:  # Machine WEST
                            input_rot = Rotation.EAST

                        # Find any valid perpendicular direction (not aligned requirement)
                        perp_pos_any = None
                        perp_dir_any = None
                        for (px, py, pz), pdir in perpendicular_candidates:
                            if self.is_valid(px, py, pz):
                                perp_pos_any = (px, py, pz)
                                perp_dir_any = pdir
                                break

                        if perp_pos_any is not None:
                            # Determine turn belt type
                            input_dirs = [Rotation.EAST, Rotation.SOUTH, Rotation.WEST, Rotation.NORTH]
                            input_idx = input_dirs.index(input_rot)
                            output_idx = input_dirs.index(perp_dir_any)
                            turn = (output_idx - input_idx) % 4

                            if turn == 1:
                                turn_belt_type = BuildingType.BELT_RIGHT
                            elif turn == 3:
                                turn_belt_type = BuildingType.BELT_LEFT
                            else:
                                turn_belt_type = BuildingType.BELT_FORWARD

                            # Place turn belt at curr_pos
                            belts.append((curr_pos[0], curr_pos[1], curr_pos[2],
                                         turn_belt_type, input_rot))
                            self.occupied.add(curr_pos)

                            # Now we need to route from perp_pos_any to next_xyz using regular belts
                            # For simplicity, place a series of belts along the path
                            # This is a simplified approach - ideally we'd re-route

                            # Place second turn belt at perp_pos to redirect toward destination
                            # perp_pos receives from curr_pos direction
                            # Belt rotation = facing direction = opposite of input direction
                            perp_x, perp_y, perp_z = perp_pos_any
                            # Items came from curr_pos in direction perp_dir_any
                            # So belt faces perp_dir_any to receive from opposite
                            if perp_dir_any == Rotation.EAST:
                                perp_belt_facing = Rotation.EAST  # Faces EAST, receives from WEST
                            elif perp_dir_any == Rotation.WEST:
                                perp_belt_facing = Rotation.WEST  # Faces WEST, receives from EAST
                            elif perp_dir_any == Rotation.SOUTH:
                                perp_belt_facing = Rotation.SOUTH  # Faces SOUTH, receives from NORTH
                            else:
                                perp_belt_facing = Rotation.NORTH  # Faces NORTH, receives from SOUTH

                            # Determine direction toward teleport destination
                            dest_dx = next_xyz[0] - perp_x
                            dest_dy = next_xyz[1] - perp_y
                            if abs(dest_dy) >= abs(dest_dx):
                                # Go vertically toward destination
                                if dest_dy > 0:
                                    dest_dir = Rotation.SOUTH
                                else:
                                    dest_dir = Rotation.NORTH
                            else:
                                # Go horizontally toward destination
                                if dest_dx > 0:
                                    dest_dir = Rotation.EAST
                                else:
                                    dest_dir = Rotation.WEST

                            # Determine turn belt type at perp_pos
                            # Belt faces perp_belt_facing, outputs toward dest_dir
                            facing_idx = input_dirs.index(perp_belt_facing)
                            dest_idx = input_dirs.index(dest_dir)
                            perp_turn = (dest_idx - facing_idx) % 4

                            if perp_turn == 0:
                                perp_belt_type = BuildingType.BELT_FORWARD
                            elif perp_turn == 1:
                                perp_belt_type = BuildingType.BELT_RIGHT
                            elif perp_turn == 3:
                                perp_belt_type = BuildingType.BELT_LEFT
                            else:  # 180 degree turn, shouldn't happen
                                perp_belt_type = BuildingType.BELT_FORWARD

                            belts.append((perp_x, perp_y, perp_z, perp_belt_type, perp_belt_facing))
                            self.occupied.add(perp_pos_any)

                            # Now place belts from perp_pos + 1 step toward destination
                            # until we reach the teleport receiver position (next_xyz)
                            cur_x, cur_y, cur_z = perp_x, perp_y, perp_z
                            if dest_dir == Rotation.SOUTH:
                                cur_y += 1
                            elif dest_dir == Rotation.NORTH:
                                cur_y -= 1
                            elif dest_dir == Rotation.EAST:
                                cur_x += 1
                            else:
                                cur_x -= 1

                            # Place regular belts until we reach the column/row of destination
                            # then turn toward it
                            steps = 0
                            max_steps = 20  # Safety limit
                            while steps < max_steps:
                                if (cur_x, cur_y, cur_z) == next_xyz:
                                    # Reached destination - place receiver-like belt
                                    # (actually just a forward belt since we're not teleporting)
                                    receiver_rotation = dest_dir
                                    if i + 2 < len(path):
                                        after_receiver = path[i + 2]
                                        after_xyz = (after_receiver[0], after_receiver[1], after_receiver[2])
                                        dx_next = after_xyz[0] - next_xyz[0]
                                        dy_next = after_xyz[1] - next_xyz[1]
                                        if dx_next > 0:
                                            receiver_rotation = Rotation.EAST
                                        elif dx_next < 0:
                                            receiver_rotation = Rotation.WEST
                                        elif dy_next > 0:
                                            receiver_rotation = Rotation.SOUTH
                                        elif dy_next < 0:
                                            receiver_rotation = Rotation.NORTH

                                    # Determine belt type (might need turn)
                                    cur_input_idx = input_dirs.index(dest_dir)
                                    out_idx = input_dirs.index(receiver_rotation)
                                    belt_turn = (out_idx - cur_input_idx) % 4
                                    if belt_turn == 0:
                                        final_belt_type = BuildingType.BELT_FORWARD
                                    elif belt_turn == 1:
                                        final_belt_type = BuildingType.BELT_RIGHT
                                    elif belt_turn == 3:
                                        final_belt_type = BuildingType.BELT_LEFT
                                    else:
                                        final_belt_type = BuildingType.BELT_FORWARD

                                    belts.append((cur_x, cur_y, cur_z, final_belt_type, dest_dir))
                                    self.occupied.add((cur_x, cur_y, cur_z))
                                    break

                                if not self.is_valid(cur_x, cur_y, cur_z):
                                    self._debug(f"Hit invalid cell at ({cur_x}, {cur_y}, {cur_z})")
                                    break

                                # Check if we need to turn toward destination
                                dx_to_dest = next_xyz[0] - cur_x
                                dy_to_dest = next_xyz[1] - cur_y

                                need_turn = False
                                new_dir = dest_dir
                                if dest_dir in (Rotation.NORTH, Rotation.SOUTH):
                                    # Moving vertically, check if we're at destination column
                                    if cur_x == next_xyz[0]:
                                        # Just continue straight
                                        pass
                                    elif dy_to_dest == 0:
                                        # Same row, turn toward dest column
                                        need_turn = True
                                        new_dir = Rotation.EAST if dx_to_dest > 0 else Rotation.WEST
                                else:
                                    # Moving horizontally
                                    if cur_y == next_xyz[1]:
                                        pass
                                    elif dx_to_dest == 0:
                                        need_turn = True
                                        new_dir = Rotation.SOUTH if dy_to_dest > 0 else Rotation.NORTH

                                if need_turn:
                                    # Turn belts: rotation = facing direction = direction we came FROM
                                    # Items traveling in dest_dir direction came from opposite of dest_dir
                                    # Belt faces dest_dir (old direction) to receive from that opposite
                                    old_dir = dest_dir  # Direction items were traveling
                                    old_idx = input_dirs.index(old_dir)
                                    new_idx = input_dirs.index(new_dir)
                                    belt_turn = (new_idx - old_idx) % 4
                                    if belt_turn == 1:
                                        belt_type = BuildingType.BELT_RIGHT
                                    elif belt_turn == 3:
                                        belt_type = BuildingType.BELT_LEFT
                                    else:
                                        belt_type = BuildingType.BELT_FORWARD
                                    # Turn belt faces the OLD direction (to receive from its opposite)
                                    belt_rotation = old_dir
                                    dest_dir = new_dir  # Update for next cells
                                else:
                                    belt_type = BuildingType.BELT_FORWARD
                                    belt_rotation = dest_dir  # Forward belt faces output direction

                                belts.append((cur_x, cur_y, cur_z, belt_type, belt_rotation))
                                self.occupied.add((cur_x, cur_y, cur_z))

                                # Move to next cell
                                if dest_dir == Rotation.SOUTH:
                                    cur_y += 1
                                elif dest_dir == Rotation.NORTH:
                                    cur_y -= 1
                                elif dest_dir == Rotation.EAST:
                                    cur_x += 1
                                else:
                                    cur_x -= 1
                                steps += 1

                            # Skip the belt_port processing for the receiver
                            receiver_positions.add(next_xyz)
                            continue
                        else:
                            # Completely stuck - no valid perpendicular at all
                            self._debug(f"No valid perpendicular for blocked belt_port at {curr_pos}")
                            # Fall through to regular belt handling (will likely fail)
                            pass
                else:
                    # Determine receiver rotation based on the NEXT step in the path
                    # The receiver should point toward where the path continues
                    receiver_rotation = sender_rotation  # Default to same as sender
                    if i + 2 < len(path):
                        # There's a next step after the receiver
                        after_receiver = path[i + 2]
                        after_xyz = (after_receiver[0], after_receiver[1], after_receiver[2])
                        dx_next = after_xyz[0] - next_xyz[0]
                        dy_next = after_xyz[1] - next_xyz[1]

                        if dx_next > 0:
                            receiver_rotation = Rotation.EAST
                        elif dx_next < 0:
                            receiver_rotation = Rotation.WEST
                        elif dy_next > 0:
                            receiver_rotation = Rotation.SOUTH
                        elif dy_next < 0:
                            receiver_rotation = Rotation.NORTH
                        # else: dz change (lift) - keep default

                    # Place sender at current position (input from behind, output into teleport)
                    belts.append((curr_pos[0], curr_pos[1], curr_pos[2],
                                 BuildingType.BELT_PORT_SENDER, sender_rotation))
                    # Place receiver at next position (receives teleport, outputs in direction of path continuation)
                    belts.append((next_xyz[0], next_xyz[1], next_xyz[2],
                                 BuildingType.BELT_PORT_RECEIVER, receiver_rotation))
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
                # Turn belts: rotation = input direction (opposite of where flow comes from)
                # The belt receives from the opposite of rotation
                if (prev_dx != 0 and dy != 0) or (prev_dy != 0 and dx != 0):
                    # Determine input direction (from prev to curr)
                    # prev_dx > 0 means came from WEST, prev_dx < 0 means came from EAST
                    # prev_dy > 0 means came from NORTH, prev_dy < 0 means came from SOUTH

                    # Set rotation based on input (belts receive from opposite of rotation)
                    if prev_dx > 0:  # Came from WEST, so rotation = EAST to receive from WEST
                        rotation = Rotation.EAST
                    elif prev_dx < 0:  # Came from EAST, so rotation = WEST to receive from EAST
                        rotation = Rotation.WEST
                    elif prev_dy > 0:  # Came from NORTH, so rotation = SOUTH to receive from NORTH
                        rotation = Rotation.SOUTH
                    else:  # prev_dy < 0, came from SOUTH, so rotation = NORTH to receive from SOUTH
                        rotation = Rotation.NORTH

                    # Determine turn type based on input and output directions
                    # LEFT turn: EAST->NORTH, SOUTH->EAST, WEST->SOUTH, NORTH->WEST
                    # RIGHT turn: EAST->SOUTH, SOUTH->WEST, WEST->NORTH, NORTH->EAST
                    if (prev_dx > 0 and dy < 0) or (prev_dx < 0 and dy > 0) or \
                       (prev_dy > 0 and dx > 0) or (prev_dy < 0 and dx < 0):
                        belt_type = BuildingType.BELT_LEFT
                    else:
                        belt_type = BuildingType.BELT_RIGHT

            belts.append((curr_pos[0], curr_pos[1], curr_pos[2], belt_type, rotation))
            belt_positions.add(curr_pos)

        # Final position handling: always add a belt at the final position if it doesn't have one.
        # The final position is the goal (e.g., one cell inside from the edge), and there's
        # usually an edge belt at the actual edge. We need a belt at the final position to
        # connect to the edge belt.
        # If path ends with 'output_hint', use the second-to-last as final position
        # and determine direction from final->hint (not prev->final)
        if len(path) >= 2:
            if path[-1][3] == 'output_hint':
                # Final position is second-to-last, direction is towards the hint
                if len(path) >= 2:
                    final_pos = path[-2]
                    hint_pos = path[-1]
                    final_xyz = (final_pos[0], final_pos[1], final_pos[2])
                    hint_xyz = (hint_pos[0], hint_pos[1], hint_pos[2])

                    # Add belt at final position if it doesn't already have one
                    if final_xyz not in belt_positions and final_xyz not in receiver_positions:
                        # Direction from final to hint
                        dx = hint_xyz[0] - final_xyz[0]
                        dy = hint_xyz[1] - final_xyz[1]

                        if dx > 0:
                            rotation = Rotation.EAST
                        elif dx < 0:
                            rotation = Rotation.WEST
                        elif dy > 0:
                            rotation = Rotation.SOUTH
                        elif dy < 0:
                            rotation = Rotation.NORTH
                        else:
                            # Same position (shouldn't happen), default to EAST
                            rotation = Rotation.EAST

                        belts.append((final_xyz[0], final_xyz[1], final_xyz[2],
                                     BuildingType.BELT_FORWARD, rotation))
                        belt_positions.add(final_xyz)
            else:
                # No hint, use prev->final direction (original behavior)
                final_pos = path[-1]
                final_xyz = (final_pos[0], final_pos[1], final_pos[2])
                prev_pos = path[-2]
                prev_xyz = (prev_pos[0], prev_pos[1], prev_pos[2])

                # Add belt at final position if it doesn't already have one
                if final_xyz not in belt_positions and final_xyz not in receiver_positions:
                    dx = final_xyz[0] - prev_xyz[0]
                    dy = final_xyz[1] - prev_xyz[1]

                    if dx > 0:
                        rotation = Rotation.EAST
                    elif dx < 0:
                        rotation = Rotation.WEST
                    elif dy > 0:
                        rotation = Rotation.SOUTH
                    elif dy < 0:
                        rotation = Rotation.NORTH
                    else:
                        # Same position (shouldn't happen), default to EAST
                        rotation = Rotation.EAST

                    belts.append((final_xyz[0], final_xyz[1], final_xyz[2],
                                 BuildingType.BELT_FORWARD, rotation))
                    belt_positions.add(final_xyz)

        return belts

    def route_connection_smart(self, connection: Connection) -> RouteResult:
        """
        Route a connection using smart belt port decisions.

        Uses find_path_with_retry which first tries without belt ports,
        then with smart congestion-aware belt ports if needed.
        """
        start = connection.from_pos
        goal = connection.to_pos
        shape_code = connection.shape_code
        throughput = connection.throughput

        path_with_types = self.find_path_with_retry(start, goal, shape_code, throughput)

        if path_with_types is None:
            return RouteResult(path=[], belts=[], success=False)

        belts = self.path_to_belts(path_with_types)
        simple_path = [(p[0], p[1], p[2]) for p in path_with_types]

        # Mark belt positions as occupied
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
            cost=len(simple_path)
        )

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

    def route_all(self, connections: List[Connection], track_paths: bool = True) -> List[RouteResult]:
        """Route all connections, prioritizing by priority value.

        Args:
            connections: List of connections to route
            track_paths: If True, use indexed routing to track paths and ownership
                        for ML training. Default True.

        Returns:
            List of RouteResult for each connection
        """
        # Sort by priority (higher first)
        sorted_connections = sorted(connections, key=lambda c: -c.priority)

        # Store for context building (ML functions can use this)
        self._all_connections = sorted_connections
        self._remaining_connections = list(sorted_connections)

        results = []
        for i, conn in enumerate(sorted_connections):
            self._current_connection_index = i
            # Use indexed routing to track paths/ownership for ML training
            if track_paths:
                result = self.route_connection_indexed(conn, i)
            else:
                result = self.route_connection(conn)
            results.append(result)
            # Remove from remaining after routing
            if conn in self._remaining_connections:
                self._remaining_connections.remove(conn)

        return results

    def clear(self) -> None:
        """Clear all routing state."""
        self.belt_positions.clear()
        self.belt_port_pairs.clear()
        self.belt_ports_used = 0
        self.shape_at_cell.clear()
        self.cells_reaching_dest.clear()
        self.belt_load.clear()
        # Clear conflict tracking
        self.belt_owner.clear()
        self.connection_belts.clear()
        # Clear ML context
        self._all_connections = []
        self._remaining_connections = []
        self._current_connection_index = 0
        self.connection_paths.clear()
        self.failed_connections.clear()
        # Note: doesn't clear occupied - call set_occupied to reset

    def export_routing_outcome(self) -> Dict[str, Any]:
        """
        Export complete routing outcome for ML training.

        Call this after route_all() or route_all_with_retry() to get
        detailed routing data for ML training.

        Returns:
            Dict with:
            - grid_size: (width, height, floors)
            - connections: List of (from_pos, to_pos) tuples
            - paths: Dict mapping connection_index -> path
            - belts: Dict mapping connection_index -> belt cells
            - failed_connections: List of failed connection info
            - conflict_analysis: Detailed conflict analysis
            - success: Overall routing success
            - cells_used: Set of all cells used by belts
        """
        # Collect all cells used
        cells_used = set()
        for cells in self.connection_belts.values():
            cells_used.update(cells)

        # Build connection list from stored data
        connections = []
        for conn in self._all_connections:
            connections.append((conn.from_pos, conn.to_pos))

        # Determine overall success
        all_success = len(self.failed_connections) == 0 and len(self.connection_paths) > 0

        return {
            'grid_size': (self.grid_width, self.grid_height, self.num_floors),
            'connections': connections,
            'paths': dict(self.connection_paths),
            'belts': dict(self.connection_belts),
            'belt_owner': dict(self.belt_owner),
            'failed_connections': list(self.failed_connections),
            'conflict_analysis': self.get_conflict_analysis(),
            'success': all_success,
            'cells_used': cells_used,
            'num_routed': len(self.connection_paths),
            'num_failed': len(self.failed_connections),
        }

    # =========================================================================
    # Smart Belt Port Methods
    # =========================================================================

    def get_corridor_congestion(
        self,
        start: Tuple[int, int, int],
        end: Tuple[int, int, int],
        width: int = 3,
    ) -> float:
        """
        Calculate congestion level in the corridor between two points.

        Args:
            start: Start position
            end: End position
            width: Corridor width to check (cells on each side)

        Returns:
            Congestion ratio (0 = empty, 1 = fully blocked)
        """
        x1, y1, f1 = start
        x2, y2, f2 = end

        # Only check same floor
        if f1 != f2:
            return 0.5  # Unknown for floor changes

        # Get bounding box with width
        min_x = min(x1, x2) - width
        max_x = max(x1, x2) + width
        min_y = min(y1, y2) - width
        max_y = max(y1, y2) + width

        total_cells = 0
        blocked_cells = 0

        for x in range(max(0, min_x), min(self.grid_width, max_x + 1)):
            for y in range(max(0, min_y), min(self.grid_height, max_y + 1)):
                total_cells += 1
                if (x, y, f1) in self.occupied:
                    blocked_cells += 1

        return blocked_cells / max(1, total_cells)

    def get_smart_belt_port_cost(
        self,
        current: Tuple[int, int, int],
        jump_target: Tuple[int, int, int],
        goal: Tuple[int, int, int],
    ) -> float:
        """
        Calculate smart cost for a belt port jump based on congestion.

        Lower cost when:
        - Jumping over congested areas
        - Jump is toward the goal
        - Direct path is blocked

        Args:
            current: Current position
            jump_target: Where the belt port lands
            goal: Final destination

        Returns:
            Cost for this belt port jump (1.5 to 3.0)
        """
        # Base cost
        base_cost = 2.5

        # Check congestion in the jump corridor
        congestion = self.get_corridor_congestion(current, jump_target, width=1)

        # Reduce cost based on congestion (jumping over blocked areas is good)
        congestion_bonus = congestion * 1.5  # Up to 1.5 reduction

        # Check if jump is toward goal
        curr_dist_to_goal = self.heuristic(current, goal)
        target_dist_to_goal = self.heuristic(jump_target, goal)

        if target_dist_to_goal < curr_dist_to_goal:
            # Jumping toward goal - bonus
            direction_bonus = 0.3
        else:
            # Jumping away from goal - penalty
            direction_bonus = -0.5

        # Check if immediate neighbors are blocked (encourages jumping when stuck)
        blocked_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if not self.is_valid(nx, ny, current[2]):
                blocked_neighbors += 1

        stuck_bonus = blocked_neighbors * 0.2  # Up to 0.8 reduction if surrounded

        # Calculate final cost (minimum 1.5 to still prefer direct paths when available)
        final_cost = max(1.5, base_cost - congestion_bonus - direction_bonus - stuck_bonus)

        return final_cost

    def find_path_with_smart_ports(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        shape_code: Optional[str] = None,
        throughput: float = 1.0,
    ) -> Optional[List[Tuple[int, int, int, str]]]:
        """
        Find path with intelligent belt port usage.

        Uses dynamic cost for belt ports based on congestion, making jumps
        more attractive when the direct path is blocked.

        Args:
            start: Start position
            goal: Goal position
            shape_code: Shape for merge checking
            throughput: Required throughput

        Returns:
            Path with move types, or None if no path found
        """
        if start == goal:
            return [(start[0], start[1], start[2], 'start')]

        if not self.is_valid(goal[0], goal[1], goal[2], shape_code, throughput):
            return None

        # A* with smart belt port costs
        open_set = [(0, start)]
        came_from: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], str]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start: 0}
        context = self._build_context(goal)
        f_score: Dict[Tuple[int, int, int], float] = {start: self.heuristic(start, goal, context)}

        while open_set:
            _, current = heapq.heappop(open_set)

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

            # Always consider belt ports when available
            allow_port = self.use_belt_ports and self.belt_ports_used < self.max_belt_ports

            for nx, ny, nf, _, move_type in self.get_neighbors(
                current[0], current[1], current[2],
                allow_port, shape_code, throughput
            ):
                neighbor = (nx, ny, nf)

                # Smart cost calculation - use custom move cost or smart belt port cost
                if move_type == 'belt_port':
                    # For belt ports, use smart cost calculation if no custom function
                    if self._custom_move_cost is not None:
                        cost = self._custom_move_cost(current, neighbor, move_type, context)
                    else:
                        cost = self.get_smart_belt_port_cost(current, neighbor, goal)
                else:
                    cost = self.move_cost(current, neighbor, move_type, context)

                tentative_g = g_score.get(current, float('inf')) + cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, move_type)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal, context)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def find_path_with_retry(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        shape_code: Optional[str] = None,
        throughput: float = 1.0,
    ) -> Optional[List[Tuple[int, int, int, str]]]:
        """
        Find path with automatic retry using belt ports if direct path fails.

        First tries without belt ports (cheaper), then with smart belt ports
        if the direct path fails.

        Args:
            start: Start position
            goal: Goal position
            shape_code: Shape for merge checking
            throughput: Required throughput

        Returns:
            Path with move types, or None if no path found
        """
        # First try without belt ports (usually faster and cheaper)
        path = self.find_path(start, goal, allow_belt_ports=False,
                             shape_code=shape_code, throughput=throughput)
        if path:
            return path

        # If that failed and we have belt ports available, try with smart ports
        if self.use_belt_ports and self.belt_ports_used < self.max_belt_ports:
            return self.find_path_with_smart_ports(start, goal, shape_code, throughput)

        return None


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
    from .core_types import PlacedBuilding

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
                # Use pluggable move cost function
                cost = self.move_cost(current, neighbor, move_type)

                # Slight bonus for following ML prediction
                if ml_score > 0.5:
                    cost *= (1 - 0.1 * ml_score)

                tentative_g = g_score.get(current, float('inf')) + cost

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
