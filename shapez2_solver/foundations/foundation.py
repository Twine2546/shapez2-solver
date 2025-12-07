"""Foundation data structures."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class FoundationType(Enum):
    """Types of foundations available in Shapez 2."""
    SIZE_1X1 = "1x1"
    SIZE_2X1 = "2x1"
    SIZE_3X1 = "3x1"
    SIZE_4X1 = "4x1"
    SIZE_2X2 = "2x2"
    SIZE_3X2 = "3x2"
    SIZE_4X2 = "4x2"
    SIZE_3X3 = "3x3"
    T_SHAPE = "T"
    L_SHAPE = "L"
    S_SHAPE = "S"
    CROSS = "Cross"


class PortDirection(Enum):
    """Direction of a port on a foundation."""
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    @property
    def opposite(self) -> "PortDirection":
        """Get the opposite direction."""
        opposites = {
            PortDirection.NORTH: PortDirection.SOUTH,
            PortDirection.SOUTH: PortDirection.NORTH,
            PortDirection.EAST: PortDirection.WEST,
            PortDirection.WEST: PortDirection.EAST,
        }
        return opposites[self]


@dataclass
class Port:
    """A port (notch) on a foundation edge."""
    direction: PortDirection
    floor: int  # 0, 1, or 2
    position: int  # Position along the edge (0-based)
    is_input: bool = True  # True for input, False for output

    @property
    def is_output(self) -> bool:
        return not self.is_input

    def __hash__(self):
        return hash((self.direction, self.floor, self.position, self.is_input))


@dataclass
class Foundation:
    """A foundation/platform that operations are placed on."""
    foundation_type: FoundationType
    width: int
    height: int
    num_floors: int = 3
    ports: List[Port] = field(default_factory=list)

    @property
    def tile_cost(self) -> int:
        """Get the tile cost for this foundation."""
        return self.width * self.height

    @property
    def area(self) -> int:
        """Get the total building area."""
        # Based on wiki data, area scales with size
        base_areas = {
            (1, 1): 196,
            (2, 1): 476,
            (3, 1): 756,
            (4, 1): 1036,
            (2, 2): 1156,
            (3, 2): 1836,
            (4, 2): 2516,
            (3, 3): 2916,
        }
        return base_areas.get((self.width, self.height), self.width * self.height * 196)

    def get_ports_by_direction(self, direction: PortDirection) -> List[Port]:
        """Get all ports on a specific edge."""
        return [p for p in self.ports if p.direction == direction]

    def get_ports_by_floor(self, floor: int) -> List[Port]:
        """Get all ports on a specific floor."""
        return [p for p in self.ports if p.floor == floor]

    def get_input_ports(self) -> List[Port]:
        """Get all input ports."""
        return [p for p in self.ports if p.is_input]

    def get_output_ports(self) -> List[Port]:
        """Get all output ports."""
        return [p for p in self.ports if p.is_output]

    def add_port(self, port: Port) -> None:
        """Add a port to the foundation."""
        self.ports.append(port)

    @classmethod
    def create_1x1(cls) -> "Foundation":
        """Create a 1x1 foundation with standard ports."""
        foundation = cls(
            foundation_type=FoundationType.SIZE_1X1,
            width=1,
            height=1,
        )
        # Add ports on all edges, all floors
        for direction in PortDirection:
            for floor in range(3):
                foundation.add_port(Port(direction, floor, 0, is_input=True))
        return foundation

    @classmethod
    def create_2x1(cls) -> "Foundation":
        """Create a 2x1 foundation."""
        foundation = cls(
            foundation_type=FoundationType.SIZE_2X1,
            width=2,
            height=1,
        )
        # Add multiple ports per long edge
        for floor in range(3):
            for pos in range(2):
                foundation.add_port(Port(PortDirection.NORTH, floor, pos, is_input=True))
                foundation.add_port(Port(PortDirection.SOUTH, floor, pos, is_input=True))
            foundation.add_port(Port(PortDirection.EAST, floor, 0, is_input=True))
            foundation.add_port(Port(PortDirection.WEST, floor, 0, is_input=True))
        return foundation

    @classmethod
    def create_3x3(cls) -> "Foundation":
        """Create a 3x3 foundation (maximum size)."""
        foundation = cls(
            foundation_type=FoundationType.SIZE_3X3,
            width=3,
            height=3,
        )
        # 3 ports per edge, 3 floors = 12 connections per edge
        for direction in PortDirection:
            for floor in range(3):
                for pos in range(3):
                    foundation.add_port(Port(direction, floor, pos, is_input=True))
        return foundation

    def __repr__(self) -> str:
        return f"Foundation({self.foundation_type.value}, ports={len(self.ports)})"
