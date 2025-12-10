"""Foundation configuration for evolution system.

Defines how foundations are laid out with ports on each side, position, and floor.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import copy


class Side(Enum):
    """Sides of a foundation."""
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"

    @property
    def opposite(self) -> "Side":
        return {
            Side.NORTH: Side.SOUTH,
            Side.SOUTH: Side.NORTH,
            Side.EAST: Side.WEST,
            Side.WEST: Side.EAST,
        }[self]


class PortType(Enum):
    """Type of port."""
    UNUSED = auto()
    INPUT = auto()
    OUTPUT = auto()


@dataclass
class PortConfig:
    """Configuration for a single port."""
    port_type: PortType = PortType.UNUSED
    shape_code: Optional[str] = None  # e.g., "CuCuCuCu" for input, expected output for output

    def __repr__(self):
        if self.port_type == PortType.UNUSED:
            return "·"
        elif self.port_type == PortType.INPUT:
            return f"I:{self.shape_code or '?'}"
        else:
            return f"O:{self.shape_code or '?'}"


@dataclass
class FoundationSpec:
    """
    Specification for a foundation type.

    Foundation dimensions in Shapez 2:
    - Each 1x1 unit is 14×14 internal grid tiles
    - Additional units add 20 tiles per axis (14 + overlap/connection zone)
    - Each 1x1 unit has 4 ports centered on each of its edges
    - 4 floors per foundation

    Internal grid formula:
    - grid_width = 14 + (units_x - 1) * 20
    - grid_height = 14 + (units_y - 1) * 20

    Area verification:
    - 1x1: 14×14 = 196 ✓
    - 2x1: 34×14 = 476 ✓
    - 2x2: 34×34 = 1156 ✓
    - 3x3: 54×54 = 2916 ✓
    """
    name: str
    units_x: int  # Number of 1x1 units in X dimension
    units_y: int  # Number of 1x1 units in Y dimension
    num_floors: int = 4

    # For irregular foundations, which cells are present
    # None means all cells present (rectangular), otherwise list of (x, y) tuples
    present_cells: Optional[List[Tuple[int, int]]] = None

    @property
    def grid_width(self) -> int:
        """Internal grid width in tiles."""
        return 14 + (self.units_x - 1) * 20

    @property
    def grid_height(self) -> int:
        """Internal grid height in tiles."""
        return 14 + (self.units_y - 1) * 20

    @property
    def total_area(self) -> int:
        """Total buildable area in tiles."""
        if self.present_cells is None:
            return self.grid_width * self.grid_height
        # For irregular foundations, calculate based on present cells
        # Each 1x1 cell contributes 196 tiles, connections add more
        return len(self.present_cells) * 196 + self._connection_area()

    def _connection_area(self) -> int:
        """Calculate extra area from connections between cells."""
        if self.present_cells is None:
            return 0
        # Count horizontal and vertical connections
        connections = 0
        cells_set = set(self.present_cells)
        for x, y in self.present_cells:
            if (x + 1, y) in cells_set:
                connections += 1  # Horizontal connection
            if (x, y + 1) in cells_set:
                connections += 1  # Vertical connection
        # Each connection adds area (20-14)*14 = 84 tiles approximately
        return connections * 140  # Adjusted based on actual game data

    def get_valid_grid_cells(self) -> Optional[Set[Tuple[int, int]]]:
        """Get set of valid (x, y) grid positions for this foundation.

        Returns None for rectangular foundations (all cells valid).
        For irregular foundations, returns set of valid tile coordinates.
        """
        if self.present_cells is None:
            return None  # All cells are valid

        valid_cells = set()
        cells_set = set(self.present_cells)

        for ux, uy in self.present_cells:
            # Each 1x1 unit is 14x14 tiles
            # First unit at (0,0), next at (20,0), etc.
            start_x = ux * 20 if ux > 0 else 0
            start_y = uy * 20 if uy > 0 else 0

            # Add the 14x14 core of this unit
            for dx in range(14):
                for dy in range(14):
                    valid_cells.add((start_x + dx, start_y + dy))

            # Add connection strips to adjacent units
            # Horizontal connection (to the right)
            if (ux + 1, uy) in cells_set:
                for dy in range(14):
                    for dx in range(6):
                        valid_cells.add((start_x + 14 + dx, start_y + dy))

            # Vertical connection (downward)
            if (ux, uy + 1) in cells_set:
                for dx in range(14):
                    for dy in range(6):
                        valid_cells.add((start_x + dx, start_y + 14 + dy))

            # Diagonal connection area (if both right and down neighbors exist)
            if (ux + 1, uy) in cells_set and (ux, uy + 1) in cells_set and (ux + 1, uy + 1) in cells_set:
                for dx in range(6):
                    for dy in range(6):
                        valid_cells.add((start_x + 14 + dx, start_y + 14 + dy))

        return valid_cells

    @property
    def ports_per_side(self) -> Dict[Side, int]:
        """Get number of port positions per side (4 per 1x1 unit on that edge).

        For irregular foundations, counts ALL exposed faces including internal gaps.
        """
        # For rectangular foundations (no gaps)
        if self.present_cells is None:
            return {
                Side.NORTH: self.units_x * 4,
                Side.SOUTH: self.units_x * 4,
                Side.EAST: self.units_y * 4,
                Side.WEST: self.units_y * 4,
            }

        # For irregular foundations: find all exposed faces (perimeter + internal gaps)
        cells = set(self.present_cells)

        # Count exposed faces in each direction
        north_faces = []  # (x, y) of units with exposed north face
        south_faces = []
        east_faces = []
        west_faces = []

        for x, y in cells:
            # North face (y-1): exposed if (x, y-1) not present
            if (x, y - 1) not in cells:
                north_faces.append((x, y))

            # South face (y+1): exposed if (x, y+1) not present
            if (x, y + 1) not in cells:
                south_faces.append((x, y))

            # West face (x-1): exposed if (x-1, y) not present
            if (x - 1, y) not in cells:
                west_faces.append((x, y))

            # East face (x+1): exposed if (x+1, y) not present
            if (x + 1, y) not in cells:
                east_faces.append((x, y))

        return {
            Side.NORTH: len(north_faces) * 4,
            Side.SOUTH: len(south_faces) * 4,
            Side.EAST: len(east_faces) * 4,
            Side.WEST: len(west_faces) * 4,
        }

    @property
    def total_ports_per_floor(self) -> int:
        """Total port positions per floor."""
        return 2 * (self.units_x + self.units_y) * 4

    def get_port_grid_position(self, side: Side, port_index: int) -> Tuple[int, int]:
        """
        Get internal grid coordinates for a port position.

        Ports are centered on each 1x1 unit. Each unit has 4 ports.
        Port indices 0-3 are on unit 0, 4-7 on unit 1, etc.

        For irregular foundations, port indices map to exposed faces sorted by position.
        """
        unit_index = port_index // 4
        port_in_unit = port_index % 4  # 0-3 within the unit

        # For irregular foundations, need to find which unit this port belongs to
        if self.present_cells is not None:
            cells = set(self.present_cells)
            exposed_units = []

            # Find units with exposed faces in this direction
            for x, y in cells:
                is_exposed = False
                if side == Side.NORTH and (x, y - 1) not in cells:
                    is_exposed = True
                elif side == Side.SOUTH and (x, y + 1) not in cells:
                    is_exposed = True
                elif side == Side.WEST and (x - 1, y) not in cells:
                    is_exposed = True
                elif side == Side.EAST and (x + 1, y) not in cells:
                    is_exposed = True

                if is_exposed:
                    exposed_units.append((x, y))

            # Sort units by position (x for N/S, y for E/W)
            if side in (Side.NORTH, Side.SOUTH):
                exposed_units.sort(key=lambda u: u[0])  # Sort by x
            else:
                exposed_units.sort(key=lambda u: u[1])  # Sort by y

            # Get the unit for this port index
            if unit_index < len(exposed_units):
                unit_x, unit_y = exposed_units[unit_index]
                unit_center_x = 7 + unit_x * 20
                unit_center_y = 7 + unit_y * 20
            else:
                # Fallback if index out of range
                unit_center_x = 7 + unit_index * 20
                unit_center_y = 7 + unit_index * 20
        else:
            # Rectangular foundation
            unit_center_x = 7 + unit_index * 20
            unit_center_y = 7 + unit_index * 20

        # Calculate port position within the unit
        # Ports spread around center: center+offset where offset in {-1.5, -0.5, 0.5, 1.5}
        # Approximated as integer offsets: -2, -1, 1, 2 (skip 0 to spread evenly)
        offsets = [-2, -1, 1, 2]
        port_offset = offsets[port_in_unit] if port_in_unit < len(offsets) else 0

        if side == Side.NORTH:
            return (unit_center_x + port_offset, 0)
        elif side == Side.SOUTH:
            return (unit_center_x + port_offset, self.grid_height - 1)
        elif side == Side.WEST:
            return (0, unit_center_y + port_offset)
        elif side == Side.EAST:
            return (self.grid_width - 1, unit_center_y + port_offset)
        return (0, 0)


# Standard foundation specifications
# Format: FoundationSpec(name, units_x, units_y, num_floors)
FOUNDATION_SPECS = {
    # Line foundations
    "1x1": FoundationSpec("1x1", 1, 1, 4),
    "2x1": FoundationSpec("2x1", 2, 1, 4),
    "3x1": FoundationSpec("3x1", 3, 1, 4),
    "4x1": FoundationSpec("4x1", 4, 1, 4),
    "1x2": FoundationSpec("1x2", 1, 2, 4),
    "1x3": FoundationSpec("1x3", 1, 3, 4),
    "1x4": FoundationSpec("1x4", 1, 4, 4),

    # Rectangular foundations
    "2x2": FoundationSpec("2x2", 2, 2, 4),
    "3x2": FoundationSpec("3x2", 3, 2, 4),
    "4x2": FoundationSpec("4x2", 4, 2, 4),
    "2x3": FoundationSpec("2x3", 2, 3, 4),
    "2x4": FoundationSpec("2x4", 2, 4, 4),
    "3x3": FoundationSpec("3x3", 3, 3, 4),

    # Irregular foundations (with present_cells defining shape)
    # T-shape: 3 on top, 1 centered below
    "T": FoundationSpec("T", 3, 2, 4, present_cells=[(0, 0), (1, 0), (2, 0), (1, 1)]),

    # L-shape (small): 2x2 with one corner missing
    "L": FoundationSpec("L", 2, 2, 4, present_cells=[(0, 0), (1, 0), (0, 1)]),

    # L-shape (large): 2 down, 3 across
    "L4": FoundationSpec("L4", 3, 2, 4, present_cells=[(0, 0), (0, 1), (1, 1), (2, 1)]),

    # S/Z-shape: 2 across top-left, 2 across bottom-right
    "S4": FoundationSpec("S4", 3, 2, 4, present_cells=[(0, 0), (1, 0), (1, 1), (2, 1)]),

    # Cross: 3x3 with corners missing
    "Cross": FoundationSpec("Cross", 3, 3, 4, present_cells=[
        (1, 0),          # Top
        (0, 1), (1, 1), (2, 1),  # Middle row
        (1, 2)           # Bottom
    ]),
}


@dataclass
class FoundationConfig:
    """
    Complete configuration for a foundation's ports.

    Ports are organized as:
    - ports[side][position][floor] = PortConfig

    For a 1x1 foundation (14×14 internal grid):
    - Each side has 4 port positions (centered on the 1x1 unit)
    - Each position has 4 floors
    - Total: 4 sides × 4 positions × 4 floors = 64 I/O points

    For a 2x1 foundation (34×14 internal grid):
    - N/S sides have 8 positions (4 per 1x1 unit)
    - E/W sides have 4 positions
    - Each position has 4 floors
    """
    spec: FoundationSpec
    ports: Dict[Side, List[List[PortConfig]]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize port grid if not provided."""
        if not self.ports:
            self.ports = {}
            for side in Side:
                num_positions = self.spec.ports_per_side[side]
                self.ports[side] = [
                    [PortConfig() for _ in range(self.spec.num_floors)]
                    for _ in range(num_positions)
                ]

    def set_port(self, side: Side, position: int, floor: int,
                 port_type: PortType, shape_code: Optional[str] = None) -> None:
        """Set a port configuration."""
        if position >= len(self.ports[side]):
            raise ValueError(f"Position {position} out of range for side {side}")
        if floor >= self.spec.num_floors:
            raise ValueError(f"Floor {floor} out of range (max {self.spec.num_floors - 1})")
        self.ports[side][position][floor] = PortConfig(port_type, shape_code)

    def get_port(self, side: Side, position: int, floor: int) -> PortConfig:
        """Get a port configuration."""
        return self.ports[side][position][floor]

    def set_input(self, side: Side, position: int, floor: int, shape_code: str) -> None:
        """Convenience method to set an input port."""
        self.set_port(side, position, floor, PortType.INPUT, shape_code)

    def set_output(self, side: Side, position: int, floor: int, shape_code: str) -> None:
        """Convenience method to set an output port."""
        self.set_port(side, position, floor, PortType.OUTPUT, shape_code)

    def get_all_inputs(self) -> List[Tuple[Side, int, int, str]]:
        """Get all input ports as (side, position, floor, shape_code)."""
        inputs = []
        for side in Side:
            for pos in range(len(self.ports[side])):
                for floor in range(self.spec.num_floors):
                    port = self.ports[side][pos][floor]
                    if port.port_type == PortType.INPUT:
                        inputs.append((side, pos, floor, port.shape_code or ""))
        return inputs

    def get_all_outputs(self) -> List[Tuple[Side, int, int, str]]:
        """Get all output ports as (side, position, floor, shape_code)."""
        outputs = []
        for side in Side:
            for pos in range(len(self.ports[side])):
                for floor in range(self.spec.num_floors):
                    port = self.ports[side][pos][floor]
                    if port.port_type == PortType.OUTPUT:
                        outputs.append((side, pos, floor, port.shape_code or ""))
        return outputs

    def copy(self) -> "FoundationConfig":
        """Create a deep copy."""
        new_config = FoundationConfig(self.spec)
        new_config.ports = copy.deepcopy(self.ports)
        return new_config

    def print_config(self) -> str:
        """Print a visual representation of the configuration."""
        lines = []
        lines.append(f"Foundation: {self.spec.name} ({self.spec.units_x}x{self.spec.units_y} units)")
        lines.append(f"Internal grid: {self.spec.grid_width}x{self.spec.grid_height} tiles")
        lines.append(f"Floors: {self.spec.num_floors}")
        lines.append(f"Ports per side: N={self.spec.ports_per_side[Side.NORTH]}, "
                     f"E={self.spec.ports_per_side[Side.EAST]}, "
                     f"S={self.spec.ports_per_side[Side.SOUTH]}, "
                     f"W={self.spec.ports_per_side[Side.WEST]}")
        lines.append("")

        for floor in range(self.spec.num_floors):
            # Only show floors that have configured ports
            has_ports = False
            for side in Side:
                for pos in range(len(self.ports[side])):
                    if self.ports[side][pos][floor].port_type != PortType.UNUSED:
                        has_ports = True
                        break
                if has_ports:
                    break

            if not has_ports and floor > 0:
                continue

            lines.append(f"Floor {floor}:")

            # Build visual grid - simplified view showing 1x1 units
            units_x = self.spec.units_x
            units_y = self.spec.units_y
            grid_width = units_x + 4  # Margins for ports
            grid_height = units_y + 4

            grid = [['·' for _ in range(grid_width * 5)] for _ in range(grid_height)]

            # Draw foundation outline (each 1x1 unit as a block)
            for ux in range(units_x):
                for uy in range(units_y):
                    # Check if cell is present (for irregular foundations)
                    if self.spec.present_cells is not None:
                        if (ux, uy) not in self.spec.present_cells:
                            continue
                    gx = (ux + 2) * 5
                    gy = uy + 2
                    grid[gy][gx:gx+5] = ['[', '1', 'x', '1', ']']

            # Draw ports - group by 1x1 unit
            # North side
            for pos in range(self.spec.ports_per_side[Side.NORTH]):
                port = self.ports[Side.NORTH][pos][floor]
                unit_idx = pos // 4
                port_in_unit = pos % 4
                gx = (unit_idx + 2) * 5 + port_in_unit + 1
                gy = 1
                if port.port_type == PortType.INPUT:
                    grid[gy][gx] = 'I'
                elif port.port_type == PortType.OUTPUT:
                    grid[gy][gx] = 'O'

            # South side
            for pos in range(self.spec.ports_per_side[Side.SOUTH]):
                port = self.ports[Side.SOUTH][pos][floor]
                unit_idx = pos // 4
                port_in_unit = pos % 4
                gx = (unit_idx + 2) * 5 + port_in_unit + 1
                gy = units_y + 2
                if port.port_type == PortType.INPUT:
                    grid[gy][gx] = 'I'
                elif port.port_type == PortType.OUTPUT:
                    grid[gy][gx] = 'O'

            # West side
            for pos in range(self.spec.ports_per_side[Side.WEST]):
                port = self.ports[Side.WEST][pos][floor]
                unit_idx = pos // 4
                port_in_unit = pos % 4
                gx = 5 + port_in_unit
                gy = unit_idx + 2
                if port.port_type == PortType.INPUT:
                    grid[gy][gx] = 'I'
                elif port.port_type == PortType.OUTPUT:
                    grid[gy][gx] = 'O'

            # East side
            for pos in range(self.spec.ports_per_side[Side.EAST]):
                port = self.ports[Side.EAST][pos][floor]
                unit_idx = pos // 4
                port_in_unit = pos % 4
                gx = (units_x + 2) * 5 + port_in_unit
                gy = unit_idx + 2
                if port.port_type == PortType.INPUT:
                    grid[gy][gx] = 'I'
                elif port.port_type == PortType.OUTPUT:
                    grid[gy][gx] = 'O'

            for row in grid:
                lines.append("  " + "".join(row))
            lines.append("")

        # Print port details
        lines.append("Port Details:")
        for side in Side:
            for pos in range(len(self.ports[side])):
                for floor in range(self.spec.num_floors):
                    port = self.ports[side][pos][floor]
                    if port.port_type != PortType.UNUSED:
                        unit_idx = pos // 4
                        port_in_unit = pos % 4
                        lines.append(f"  {side.value}[unit{unit_idx}:port{port_in_unit}][F{floor}]: {port}")

        return "\n".join(lines)


def create_corner_splitter_config() -> FoundationConfig:
    """
    Create a configuration for a corner splitter on a 2x2 foundation.

    Input: 1 shape on West side
    Outputs: 4 corners on East side
    """
    config = FoundationConfig(FOUNDATION_SPECS["2x2"])

    # Input on west side, position 0, floor 0
    config.set_input(Side.WEST, 0, 0, "CuCuCuCu")

    # Outputs on east side - 4 corners
    config.set_output(Side.EAST, 0, 0, "Cu------")  # NE corner
    config.set_output(Side.EAST, 0, 1, "--Cu----")  # NW corner
    config.set_output(Side.EAST, 1, 0, "----Cu--")  # SW corner
    config.set_output(Side.EAST, 1, 1, "------Cu")  # SE corner

    return config


def create_3floor_corner_splitter_config() -> FoundationConfig:
    """
    Create a configuration for a 3-floor corner splitter.

    3 inputs (one per floor), 12 outputs (4 per floor)
    """
    config = FoundationConfig(FOUNDATION_SPECS["2x2"])

    for floor in range(3):
        # Input on west side
        config.set_input(Side.WEST, 0, floor, "CuCuCuCu")

        # 4 corner outputs on east side
        config.set_output(Side.EAST, 0, floor, "Cu------")
        config.set_output(Side.EAST, 1, floor, "--Cu----")
        # Use south side for other 2 outputs
        config.set_output(Side.SOUTH, 0, floor, "----Cu--")
        config.set_output(Side.SOUTH, 1, floor, "------Cu")

    return config
