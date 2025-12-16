"""Core types for the solver system.

This module contains the essential data structures used across the solver.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, auto

from ..blueprint.building_types import BuildingType, Rotation
from ..shapes.shape import Shape


class CellType(Enum):
    """Type of cell in the grid."""
    EMPTY = auto()
    BUILDING = auto()
    BELT = auto()
    INPUT_PORT = auto()
    OUTPUT_PORT = auto()


@dataclass
class GridCell:
    """A cell in the grid."""
    cell_type: CellType = CellType.EMPTY
    building_type: Optional[BuildingType] = None
    rotation: Rotation = Rotation.EAST
    building_id: Optional[int] = None  # For multi-cell buildings


@dataclass
class PlacedBuilding:
    """A building placed on the grid."""
    building_id: int
    building_type: BuildingType
    x: int
    y: int
    floor: int
    rotation: Rotation
    # For belt ports, channel_id links sender/receiver pairs
    channel_id: Optional[int] = None


@dataclass
class LogicalConnection:
    """A logical connection between buildings (what connects to what)."""
    from_building_id: int
    from_output_idx: int
    to_building_id: int
    to_input_idx: int


# Import Side for type hints in Candidate
from .foundation_config import Side


@dataclass
class Candidate:
    """A candidate solution."""
    buildings: List[PlacedBuilding] = field(default_factory=list)
    connections: List[LogicalConnection] = field(default_factory=list)
    fitness: float = 0.0
    output_shapes: Dict[Tuple[Side, int, int], Optional[Shape]] = field(default_factory=dict)
    routing_success: bool = False

    def copy(self) -> "Candidate":
        """Create a deep copy."""
        new = Candidate()
        new.buildings = [copy.copy(b) for b in self.buildings]
        new.connections = [copy.copy(c) for c in self.connections]
        new.fitness = self.fitness
        new.output_shapes = dict(self.output_shapes)
        new.routing_success = self.routing_success
        return new


# Operation buildings
OPERATION_BUILDINGS = [
    BuildingType.ROTATOR_CW,
    BuildingType.ROTATOR_CCW,
    BuildingType.ROTATOR_180,
    BuildingType.CUTTER,
    BuildingType.CUTTER_MIRRORED,
    BuildingType.HALF_CUTTER,
    BuildingType.SWAPPER,
    BuildingType.STACKER,
    BuildingType.UNSTACKER,
    BuildingType.PIN_PUSHER,
    BuildingType.PAINTER,
    BuildingType.TRASH,
]

# Belt types for routing
BELT_TYPES = [
    BuildingType.BELT_FORWARD,
    BuildingType.BELT_LEFT,
    BuildingType.BELT_RIGHT,
    BuildingType.LIFT_UP,
    BuildingType.LIFT_DOWN,
]

# Splitters and mergers for routing
ROUTING_BUILDINGS = [
    BuildingType.SPLITTER,
    BuildingType.MERGER,
]

# All buildings that can be used (operations + belts + routing)
EVOLVABLE_BUILDINGS = OPERATION_BUILDINGS + BELT_TYPES + ROUTING_BUILDINGS

# Buildings that are belts or routing (for connectivity checking)
CONVEYOR_TYPES = set(BELT_TYPES + ROUTING_BUILDINGS)

# Buildings that require special input handling
DUAL_INPUT_BUILDINGS = {
    BuildingType.PAINTER,   # Needs shape + paint inputs
    BuildingType.STACKER,   # Needs bottom + top inputs
    BuildingType.SWAPPER,   # Needs two inputs to swap
}
