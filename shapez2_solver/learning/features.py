"""
Feature extraction for routing quality prediction.

All features are designed to be foundation-agnostic by using:
- Normalized values (ratios, percentages)
- Local/relative measurements
- Topology-based features (not absolute coordinates)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Set, Dict, Optional
import math
from collections import defaultdict


@dataclass
class SolutionFeatures:
    """
    Foundation-agnostic features extracted from a complete solution.

    All spatial features are normalized to [0, 1] range to generalize
    across different foundation sizes.
    """

    # === Foundation Info (context, not learned) ===
    foundation_type: str = ""
    grid_width: int = 0
    grid_height: int = 0
    num_floors: int = 3
    total_cells: int = 0  # available cells (excluding blocked)

    # === Machine Placement Features ===
    num_machines: int = 0
    machine_density: float = 0.0           # machines / available_cells
    machine_spread: float = 0.0            # std dev of machine positions (normalized)
    center_of_mass_x: float = 0.0          # normalized [0, 1]
    center_of_mass_y: float = 0.0          # normalized [0, 1]
    machine_clustering: float = 0.0        # avg distance between machines (normalized)

    # === Belt/Routing Features ===
    num_belts: int = 0
    belt_density: float = 0.0              # belt_cells / available_cells
    total_path_length: int = 0
    avg_path_length: float = 0.0
    max_path_length: int = 0
    min_path_length: int = 0
    path_length_variance: float = 0.0

    # === Path Efficiency Features ===
    avg_path_stretch: float = 0.0          # actual_length / manhattan_distance
    max_path_stretch: float = 0.0
    min_path_stretch: float = 0.0

    # === Congestion Features ===
    max_local_density_3x3: float = 0.0     # worst 3x3 region
    avg_local_density_3x3: float = 0.0     # average 3x3 region
    max_local_density_5x5: float = 0.0     # worst 5x5 region
    congestion_variance: float = 0.0       # uniformity of space usage

    # === Floor Utilization ===
    floor_utilization: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    floor_balance: float = 0.0             # how evenly distributed (0=all one floor, 1=even)
    num_floor_transitions: int = 0         # vertical belt changes

    # === Port/Connection Features ===
    num_inputs: int = 0
    num_outputs: int = 0
    num_connections: int = 0
    input_output_separation: float = 0.0   # avg normalized distance
    port_spread: float = 0.0               # how spread out are ports

    # === Topology Features ===
    crossing_potential: float = 0.0        # estimated path conflicts
    bottleneck_score: float = 0.0          # narrowest corridor usage

    # === Outcome Labels ===
    routing_success: bool = False
    routing_success_rate: float = 0.0      # connections_routed / total_connections
    throughput: float = 0.0
    solve_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_feature_vector(self) -> List[float]:
        """
        Convert to numeric feature vector for ML models.
        Excludes non-numeric and label fields.
        """
        return [
            # Machine features
            self.num_machines,
            self.machine_density,
            self.machine_spread,
            self.center_of_mass_x,
            self.center_of_mass_y,
            self.machine_clustering,
            # Belt features
            self.num_belts,
            self.belt_density,
            self.total_path_length,
            self.avg_path_length,
            self.max_path_length,
            self.min_path_length,
            self.path_length_variance,
            # Efficiency
            self.avg_path_stretch,
            self.max_path_stretch,
            self.min_path_stretch,
            # Congestion
            self.max_local_density_3x3,
            self.avg_local_density_3x3,
            self.max_local_density_5x5,
            self.congestion_variance,
            # Floors
            *self.floor_utilization,
            self.floor_balance,
            self.num_floor_transitions,
            # Ports
            self.num_inputs,
            self.num_outputs,
            self.num_connections,
            self.input_output_separation,
            self.port_spread,
            # Topology
            self.crossing_potential,
            self.bottleneck_score,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Get names for feature vector elements."""
        return [
            'num_machines', 'machine_density', 'machine_spread',
            'center_of_mass_x', 'center_of_mass_y', 'machine_clustering',
            'num_belts', 'belt_density', 'total_path_length',
            'avg_path_length', 'max_path_length', 'min_path_length',
            'path_length_variance',
            'avg_path_stretch', 'max_path_stretch', 'min_path_stretch',
            'max_local_density_3x3', 'avg_local_density_3x3',
            'max_local_density_5x5', 'congestion_variance',
            'floor_util_0', 'floor_util_1', 'floor_util_2', 'floor_util_3',
            'floor_balance', 'num_floor_transitions',
            'num_inputs', 'num_outputs', 'num_connections',
            'input_output_separation', 'port_spread',
            'crossing_potential', 'bottleneck_score',
        ]


@dataclass
class ConnectionFeatures:
    """
    Features for a single connection (source -> dest).
    Used for learning routing difficulty/priority.
    """

    # === Distance Features ===
    manhattan_distance: int = 0
    euclidean_distance: float = 0.0
    normalized_distance: float = 0.0       # distance / grid_diagonal

    # === Position Features (normalized) ===
    src_x_norm: float = 0.0
    src_y_norm: float = 0.0
    src_floor: int = 0
    dst_x_norm: float = 0.0
    dst_y_norm: float = 0.0
    dst_floor: int = 0
    floor_change: int = 0

    # === Local Congestion ===
    src_local_density: float = 0.0         # density around source
    dst_local_density: float = 0.0         # density around dest
    path_corridor_density: float = 0.0     # density in bounding box

    # === Geometric Features ===
    crosses_center: bool = False           # path likely crosses grid center
    direction_complexity: float = 0.0      # how much direction change needed
    min_corridor_width: int = 0            # narrowest passage

    # === Context ===
    connection_index: int = 0              # order in connection list
    total_connections: int = 0
    connections_already_routed: int = 0

    # === Outcome ===
    routed_successfully: bool = False
    actual_path_length: int = 0
    path_stretch: float = 0.0              # actual / manhattan

    def to_feature_vector(self) -> List[float]:
        """Convert to numeric vector for ML."""
        return [
            self.manhattan_distance,
            self.euclidean_distance,
            self.normalized_distance,
            self.src_x_norm,
            self.src_y_norm,
            self.src_floor,
            self.dst_x_norm,
            self.dst_y_norm,
            self.dst_floor,
            self.floor_change,
            self.src_local_density,
            self.dst_local_density,
            self.path_corridor_density,
            float(self.crosses_center),
            self.direction_complexity,
            self.min_corridor_width,
            self.connection_index,
            self.total_connections,
            self.connections_already_routed,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'manhattan_distance', 'euclidean_distance', 'normalized_distance',
            'src_x_norm', 'src_y_norm', 'src_floor',
            'dst_x_norm', 'dst_y_norm', 'dst_floor', 'floor_change',
            'src_local_density', 'dst_local_density', 'path_corridor_density',
            'crosses_center', 'direction_complexity', 'min_corridor_width',
            'connection_index', 'total_connections', 'connections_already_routed',
        ]


@dataclass
class LocalRegionFeatures:
    """Features for a local region (e.g., 3x3 or 5x5 area)."""

    center_x: int = 0
    center_y: int = 0
    center_floor: int = 0
    region_size: int = 3

    # === Density ===
    occupied_ratio: float = 0.0            # occupied / total cells in region
    machine_count: int = 0
    belt_count: int = 0

    # === Flow ===
    inflow_count: int = 0                  # belts entering region
    outflow_count: int = 0                 # belts leaving region

    # === Accessibility ===
    open_edges: int = 0                    # edges with free cells


def extract_solution_features(
    grid_width: int,
    grid_height: int,
    num_floors: int,
    machines: List[Tuple],              # (type, x, y, floor, rotation)
    belts: List[Tuple],                 # (x, y, floor, belt_type, rotation)
    input_positions: List[Tuple],       # (x, y, floor)
    output_positions: List[Tuple],      # (x, y, floor)
    connections: List[Tuple],           # ((src_x, src_y, src_z), (dst_x, dst_y, dst_z))
    paths: Optional[List[List[Tuple]]] = None,  # actual routed paths
    occupied: Optional[Set[Tuple]] = None,      # pre-occupied cells
    routing_success: bool = False,
    throughput: float = 0.0,
    solve_time: float = 0.0,
    foundation_type: str = "",
) -> SolutionFeatures:
    """
    Extract all features from a solution.

    This is the main entry point for feature extraction.
    """
    features = SolutionFeatures()

    # Basic info
    features.foundation_type = foundation_type
    features.grid_width = grid_width
    features.grid_height = grid_height
    features.num_floors = num_floors
    features.total_cells = grid_width * grid_height * num_floors

    if occupied:
        features.total_cells -= len(occupied)

    # === Machine Features ===
    features.num_machines = len(machines)
    if features.total_cells > 0:
        # Estimate machine cell usage (assume avg 2x2 = 4 cells per machine)
        machine_cells = len(machines) * 4
        features.machine_density = machine_cells / features.total_cells

    if machines:
        # Center of mass
        xs = [m[1] for m in machines]
        ys = [m[2] for m in machines]
        features.center_of_mass_x = (sum(xs) / len(xs)) / max(1, grid_width - 1)
        features.center_of_mass_y = (sum(ys) / len(ys)) / max(1, grid_height - 1)

        # Machine spread (std dev)
        if len(machines) > 1:
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            var = sum((x - mean_x)**2 + (y - mean_y)**2 for x, y in zip(xs, ys)) / len(machines)
            features.machine_spread = math.sqrt(var) / math.sqrt(grid_width**2 + grid_height**2)

        # Machine clustering (average pairwise distance)
        if len(machines) > 1:
            total_dist = 0
            count = 0
            for i, m1 in enumerate(machines):
                for m2 in machines[i+1:]:
                    dist = abs(m1[1] - m2[1]) + abs(m1[2] - m2[2])
                    total_dist += dist
                    count += 1
            avg_dist = total_dist / count if count > 0 else 0
            max_possible = grid_width + grid_height
            features.machine_clustering = avg_dist / max_possible if max_possible > 0 else 0

    # === Belt Features ===
    features.num_belts = len(belts)
    if features.total_cells > 0:
        features.belt_density = len(belts) / features.total_cells

    # Path length features
    if paths:
        path_lengths = [len(p) for p in paths if p]
        if path_lengths:
            features.total_path_length = sum(path_lengths)
            features.avg_path_length = sum(path_lengths) / len(path_lengths)
            features.max_path_length = max(path_lengths)
            features.min_path_length = min(path_lengths)

            if len(path_lengths) > 1:
                mean = features.avg_path_length
                features.path_length_variance = sum((l - mean)**2 for l in path_lengths) / len(path_lengths)

    # === Path Efficiency ===
    if paths and connections:
        stretches = []
        for path, conn in zip(paths, connections):
            if path and len(conn) == 2:
                src, dst = conn
                manhattan = abs(src[0] - dst[0]) + abs(src[1] - dst[1]) + abs(src[2] - dst[2])
                if manhattan > 0:
                    stretch = len(path) / manhattan
                    stretches.append(stretch)

        if stretches:
            features.avg_path_stretch = sum(stretches) / len(stretches)
            features.max_path_stretch = max(stretches)
            features.min_path_stretch = min(stretches)

    # === Congestion Features ===
    all_occupied = set()
    for m in machines:
        # Add machine cells (simplified - actual size depends on type)
        x, y, floor = m[1], m[2], m[3]
        for dx in range(2):
            for dy in range(2):
                all_occupied.add((x + dx, y + dy, floor))

    for b in belts:
        all_occupied.add((b[0], b[1], b[2]))

    if occupied:
        all_occupied.update(occupied)

    # Calculate local densities
    densities_3x3 = []
    densities_5x5 = []

    for cx in range(0, grid_width, 3):
        for cy in range(0, grid_height, 3):
            for cz in range(num_floors):
                # 3x3 density
                count_3x3 = 0
                total_3x3 = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < grid_width and 0 <= ny < grid_height:
                            total_3x3 += 1
                            if (nx, ny, cz) in all_occupied:
                                count_3x3 += 1
                if total_3x3 > 0:
                    densities_3x3.append(count_3x3 / total_3x3)

                # 5x5 density
                count_5x5 = 0
                total_5x5 = 0
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < grid_width and 0 <= ny < grid_height:
                            total_5x5 += 1
                            if (nx, ny, cz) in all_occupied:
                                count_5x5 += 1
                if total_5x5 > 0:
                    densities_5x5.append(count_5x5 / total_5x5)

    if densities_3x3:
        features.max_local_density_3x3 = max(densities_3x3)
        features.avg_local_density_3x3 = sum(densities_3x3) / len(densities_3x3)
        mean = features.avg_local_density_3x3
        features.congestion_variance = sum((d - mean)**2 for d in densities_3x3) / len(densities_3x3)

    if densities_5x5:
        features.max_local_density_5x5 = max(densities_5x5)

    # === Floor Utilization ===
    floor_counts = [0] * num_floors
    for cell in all_occupied:
        if len(cell) >= 3 and 0 <= cell[2] < num_floors:
            floor_counts[cell[2]] += 1

    cells_per_floor = grid_width * grid_height
    features.floor_utilization = [c / cells_per_floor if cells_per_floor > 0 else 0 for c in floor_counts]

    # Floor balance (entropy-like measure)
    total_used = sum(floor_counts)
    if total_used > 0:
        probs = [c / total_used for c in floor_counts if c > 0]
        if len(probs) > 1:
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            max_entropy = math.log(num_floors)
            features.floor_balance = entropy / max_entropy if max_entropy > 0 else 0

    # Count floor transitions in belts
    if belts:
        sorted_belts = sorted(belts, key=lambda b: (b[0], b[1]))
        for i in range(1, len(sorted_belts)):
            if sorted_belts[i][2] != sorted_belts[i-1][2]:
                features.num_floor_transitions += 1

    # === Port Features ===
    features.num_inputs = len(input_positions)
    features.num_outputs = len(output_positions)
    features.num_connections = len(connections)

    # Input-output separation
    if input_positions and output_positions:
        total_dist = 0
        count = 0
        for inp in input_positions:
            for out in output_positions:
                dist = abs(inp[0] - out[0]) + abs(inp[1] - out[1])
                total_dist += dist
                count += 1
        if count > 0:
            avg_dist = total_dist / count
            max_dist = grid_width + grid_height
            features.input_output_separation = avg_dist / max_dist if max_dist > 0 else 0

    # Port spread
    all_ports = list(input_positions) + list(output_positions)
    if len(all_ports) > 1:
        xs = [p[0] for p in all_ports]
        ys = [p[1] for p in all_ports]
        spread_x = (max(xs) - min(xs)) / max(1, grid_width - 1)
        spread_y = (max(ys) - min(ys)) / max(1, grid_height - 1)
        features.port_spread = (spread_x + spread_y) / 2

    # === Topology Features ===
    # Crossing potential: estimate based on bounding box overlaps
    if connections:
        overlap_count = 0
        for i, c1 in enumerate(connections):
            if len(c1) != 2:
                continue
            src1, dst1 = c1
            box1 = (min(src1[0], dst1[0]), min(src1[1], dst1[1]),
                    max(src1[0], dst1[0]), max(src1[1], dst1[1]))

            for c2 in connections[i+1:]:
                if len(c2) != 2:
                    continue
                src2, dst2 = c2
                box2 = (min(src2[0], dst2[0]), min(src2[1], dst2[1]),
                        max(src2[0], dst2[0]), max(src2[1], dst2[1]))

                # Check bounding box overlap
                if not (box1[2] < box2[0] or box2[2] < box1[0] or
                        box1[3] < box2[1] or box2[3] < box1[1]):
                    overlap_count += 1

        max_overlaps = len(connections) * (len(connections) - 1) / 2
        features.crossing_potential = overlap_count / max_overlaps if max_overlaps > 0 else 0

    # === Outcome Labels ===
    features.routing_success = routing_success
    if connections:
        successful = sum(1 for p in (paths or []) if p)
        features.routing_success_rate = successful / len(connections)
    features.throughput = throughput
    features.solve_time = solve_time

    return features


def extract_connection_features(
    connection: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    grid_width: int,
    grid_height: int,
    occupied: Set[Tuple[int, int, int]],
    connection_index: int = 0,
    total_connections: int = 1,
    connections_routed: int = 0,
    actual_path: Optional[List[Tuple]] = None,
) -> ConnectionFeatures:
    """
    Extract features for a single connection.
    Used for learning routing difficulty/priority.
    """
    src, dst = connection
    features = ConnectionFeatures()

    # Distance features
    features.manhattan_distance = abs(src[0] - dst[0]) + abs(src[1] - dst[1]) + abs(src[2] - dst[2])
    features.euclidean_distance = math.sqrt(
        (src[0] - dst[0])**2 + (src[1] - dst[1])**2 + (src[2] - dst[2])**2
    )
    grid_diagonal = math.sqrt(grid_width**2 + grid_height**2)
    features.normalized_distance = features.euclidean_distance / grid_diagonal if grid_diagonal > 0 else 0

    # Position features (normalized)
    features.src_x_norm = src[0] / max(1, grid_width - 1)
    features.src_y_norm = src[1] / max(1, grid_height - 1)
    features.src_floor = src[2]
    features.dst_x_norm = dst[0] / max(1, grid_width - 1)
    features.dst_y_norm = dst[1] / max(1, grid_height - 1)
    features.dst_floor = dst[2]
    features.floor_change = abs(src[2] - dst[2])

    # Local congestion
    def local_density(pos: Tuple, radius: int = 2) -> float:
        count = 0
        total = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    total += 1
                    if (nx, ny, pos[2]) in occupied:
                        count += 1
        return count / total if total > 0 else 0

    features.src_local_density = local_density(src)
    features.dst_local_density = local_density(dst)

    # Corridor density (bounding box)
    min_x, max_x = min(src[0], dst[0]), max(src[0], dst[0])
    min_y, max_y = min(src[1], dst[1]), max(src[1], dst[1])
    corridor_count = 0
    corridor_total = 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            corridor_total += 1
            if (x, y, src[2]) in occupied or (x, y, dst[2]) in occupied:
                corridor_count += 1
    features.path_corridor_density = corridor_count / corridor_total if corridor_total > 0 else 0

    # Geometric features
    center_x, center_y = grid_width / 2, grid_height / 2

    # Check if path likely crosses center
    if (min_x <= center_x <= max_x) and (min_y <= center_y <= max_y):
        features.crosses_center = True

    # Direction complexity (how much turning needed)
    dx = abs(dst[0] - src[0])
    dy = abs(dst[1] - src[1])
    dz = abs(dst[2] - src[2])
    total_dist = dx + dy + dz
    if total_dist > 0:
        # More balanced = more complex (needs turns)
        dims_used = sum([dx > 0, dy > 0, dz > 0])
        features.direction_complexity = dims_used / 3.0

    # Context
    features.connection_index = connection_index
    features.total_connections = total_connections
    features.connections_already_routed = connections_routed

    # Outcome (if path provided)
    if actual_path is not None:
        features.routed_successfully = len(actual_path) > 0
        features.actual_path_length = len(actual_path)
        if features.manhattan_distance > 0:
            features.path_stretch = len(actual_path) / features.manhattan_distance

    return features


def calculate_crossing_potential(
    connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
) -> float:
    """
    Estimate how likely paths are to cross/conflict.
    Higher value = more potential conflicts.
    """
    if len(connections) < 2:
        return 0.0

    crossings = 0
    total_pairs = 0

    for i, (src1, dst1) in enumerate(connections):
        for src2, dst2 in connections[i+1:]:
            total_pairs += 1

            # Check if line segments potentially intersect
            # Using bounding box overlap as approximation
            box1 = (
                min(src1[0], dst1[0]), min(src1[1], dst1[1]),
                max(src1[0], dst1[0]), max(src1[1], dst1[1])
            )
            box2 = (
                min(src2[0], dst2[0]), min(src2[1], dst2[1]),
                max(src2[0], dst2[0]), max(src2[1], dst2[1])
            )

            # Boxes overlap?
            if not (box1[2] < box2[0] or box2[2] < box1[0] or
                    box1[3] < box2[1] or box2[3] < box1[1]):
                crossings += 1

    return crossings / total_pairs if total_pairs > 0 else 0.0
