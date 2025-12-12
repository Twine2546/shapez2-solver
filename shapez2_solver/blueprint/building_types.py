"""Shapez 2 building type mappings with accurate game data."""

from enum import Enum
from typing import Dict, Tuple, NamedTuple, List


class BuildingType(Enum):
    """Building types used in Shapez 2 blueprints."""
    # Belts
    BELT_FORWARD = "BeltDefaultForwardInternalVariant"
    BELT_LEFT = "BeltDefaultLeftInternalVariant"
    BELT_RIGHT = "BeltDefaultRightInternalVariant"

    # Splitters and Mergers (for load balancing)
    SPLITTER = "SplitterInternalVariant"           # 1 in -> 2 out (alternating)
    MERGER = "MergerInternalVariant"               # 2 in -> 1 out (alternating)
    SPLITTER_LEFT = "SplitterLeftInternalVariant"  # 1 in -> 2 out (priority left)
    SPLITTER_RIGHT = "SplitterRightInternalVariant" # 1 in -> 2 out (priority right)

    # Lifts (for multi-floor connections)
    LIFT_UP = "LiftUpInternalVariant"              # Move items up one floor
    LIFT_DOWN = "LiftDownInternalVariant"          # Move items down one floor

    # Belt Ports (teleporters - sender/receiver pairs)
    BELT_PORT_SENDER = "BeltPortSenderInternalVariant"      # Teleport items (sender)
    BELT_PORT_RECEIVER = "BeltPortReceiverInternalVariant"  # Teleport items (receiver)

    # Rotators
    ROTATOR_CW = "RotatorOneQuadInternalVariant"
    ROTATOR_CCW = "RotatorOneQuadCCWInternalVariant"
    ROTATOR_180 = "RotatorHalfInternalVariant"

    # Cutters
    HALF_CUTTER = "CutterHalfInternalVariant"
    CUTTER = "CutterDefaultInternalVariant"
    CUTTER_MIRRORED = "CutterDefaultInternalVariantMirrored"
    SWAPPER = "HalvesSwapperDefaultInternalVariant"

    # Stackers
    STACKER = "StackerStraightInternalVariant"          # Straight stacker - 6 per belt, slower
    STACKER_BENT = "StackerDefaultInternalVariant"      # Bent stacker - 4 per belt, faster
    STACKER_BENT_MIRRORED = "StackerDefaultInternalVariantMirrored"  # Mirrored bent stacker
    UNSTACKER = "UnstackerDefaultInternalVariant"

    # Painters
    PAINTER = "PainterInternalVariant"
    PAINTER_MIRRORED = "PainterInternalVariantMirrored"

    # Other
    TRASH = "TrashInternalVariant"
    PIN_PUSHER = "PinPusherInternalVariant"


# Map from our operation classes to building types
OPERATION_TO_BUILDING: Dict[str, BuildingType] = {
    "RotateOperation": BuildingType.ROTATOR_CW,  # Will need to check steps
    "RotateCCWOperation": BuildingType.ROTATOR_CCW,
    "HalfDestroyerOperation": BuildingType.HALF_CUTTER,
    "CutOperation": BuildingType.CUTTER,
    "SwapperOperation": BuildingType.SWAPPER,
    "StackOperation": BuildingType.STACKER,
    "UnstackOperation": BuildingType.UNSTACKER,
    "PaintOperation": BuildingType.PAINTER,
}


class BuildingSpec(NamedTuple):
    """Specification for a building type."""
    width: int          # Grid cells wide (X)
    height: int         # Grid cells deep (Y)
    depth: int          # Grid cells tall (Z/floors) - 1=single floor, 2=two floors
    num_inputs: int     # Number of input ports
    num_outputs: int    # Number of output ports
    per_belt: int       # Buildings needed per full belt (at equal upgrades)
    base_rate: int      # Base operations per minute (tier 1)
    max_rate: int       # Max operations per minute (tier 5)

    # Input/output floor configuration for multi-floor buildings
    # Format: list of (relative_x, relative_y, floor) for each port
    # None means use default (all on floor 0)


# Accurate building specifications from Shapez 2 wiki
# Format: (width, height, depth, inputs, outputs, per_belt, base_rate, max_rate)
BUILDING_SPECS: Dict[BuildingType, BuildingSpec] = {
    # Belts: 1x1, throughput matches belt speed
    BuildingType.BELT_FORWARD: BuildingSpec(1, 1, 1, 1, 1, 1, 180, 180),
    BuildingType.BELT_LEFT: BuildingSpec(1, 1, 1, 1, 1, 1, 180, 180),
    BuildingType.BELT_RIGHT: BuildingSpec(1, 1, 1, 1, 1, 1, 180, 180),

    # Splitters: 1x1, 1 in -> 2 out, full belt speed
    BuildingType.SPLITTER: BuildingSpec(1, 1, 1, 1, 2, 1, 180, 180),
    BuildingType.SPLITTER_LEFT: BuildingSpec(1, 1, 1, 1, 2, 1, 180, 180),
    BuildingType.SPLITTER_RIGHT: BuildingSpec(1, 1, 1, 1, 2, 1, 180, 180),

    # Mergers: 1x1, 2 in -> 1 out, full belt speed
    BuildingType.MERGER: BuildingSpec(1, 1, 1, 2, 1, 1, 180, 180),

    # Lifts: 1x1x2, move items between floors
    BuildingType.LIFT_UP: BuildingSpec(1, 1, 2, 1, 1, 1, 180, 180),
    BuildingType.LIFT_DOWN: BuildingSpec(1, 1, 2, 1, 1, 1, 180, 180),

    # Rotators: 1x1x1, 2 per belt, 30-90 ops/min
    BuildingType.ROTATOR_CW: BuildingSpec(1, 1, 1, 1, 1, 2, 30, 90),
    BuildingType.ROTATOR_CCW: BuildingSpec(1, 1, 1, 1, 1, 2, 30, 90),
    BuildingType.ROTATOR_180: BuildingSpec(1, 1, 1, 1, 1, 2, 30, 90),

    # Cutters:
    # Half-cutter: 1x1, destroys one half
    BuildingType.HALF_CUTTER: BuildingSpec(1, 1, 1, 1, 1, 4, 15, 45),
    # Cutter: 1x2, outputs both halves to separate outputs (second output goes south/y+1)
    BuildingType.CUTTER: BuildingSpec(1, 2, 1, 1, 2, 4, 15, 45),
    # Cutter Mirrored: 1x2, outputs both halves (second output goes north/y-1, mirror of CUTTER)
    BuildingType.CUTTER_MIRRORED: BuildingSpec(1, 2, 1, 1, 2, 4, 15, 45),

    # Swapper: 1x2, 2 in -> 2 out (swaps halves between two parallel items)
    # Width=1, Height=2 so it spans two Y positions for side-by-side inputs
    BuildingType.SWAPPER: BuildingSpec(1, 2, 1, 2, 2, 4, 15, 45),

    # Stacker (Straight): 1x1x2 (2 floors tall), 6 per belt, 10-30 ops/min
    # Input 0: bottom floor from west (the "bottom" shape)
    # Input 1: top floor from west (the "top" shape that stacks on bottom)
    # Output: bottom floor to east
    BuildingType.STACKER: BuildingSpec(1, 1, 2, 2, 1, 6, 10, 30),

    # Stacker (Bent): 1x1x2 (2 floors tall), 4 per belt, 15-45 ops/min (faster but needs more per belt)
    # Same port layout as straight stacker, but output is bent (perpendicular to input)
    BuildingType.STACKER_BENT: BuildingSpec(1, 1, 2, 2, 1, 4, 15, 45),
    BuildingType.STACKER_BENT_MIRRORED: BuildingSpec(1, 1, 2, 2, 1, 4, 15, 45),

    # Unstacker: 1x1x2 (2 floors tall), 6 per belt
    # Input: bottom floor from west
    # Output 0: bottom floor to east (bottom layer)
    # Output 1: top floor to east (top layer)
    BuildingType.UNSTACKER: BuildingSpec(1, 1, 2, 1, 2, 6, 10, 30),

    # Painter: 1x2, shape + color inputs (color from north)
    BuildingType.PAINTER: BuildingSpec(1, 2, 1, 2, 1, 4, 15, 45),
    # Painter Mirrored: 1x2, color from south instead of north
    BuildingType.PAINTER_MIRRORED: BuildingSpec(1, 2, 1, 2, 1, 4, 15, 45),

    # Trash: 1x1
    BuildingType.TRASH: BuildingSpec(1, 1, 1, 1, 0, 1, 180, 180),

    # Pin pusher: 1x1
    BuildingType.PIN_PUSHER: BuildingSpec(1, 1, 1, 1, 1, 2, 30, 90),

    # Belt ports (teleporters): 1x1, sender/receiver pairs
    # Sender takes input and teleports to paired receiver
    # Receiver outputs items from paired sender
    BuildingType.BELT_PORT_SENDER: BuildingSpec(1, 1, 1, 1, 0, 1, 180, 180),
    BuildingType.BELT_PORT_RECEIVER: BuildingSpec(1, 1, 1, 0, 1, 1, 180, 180),
}


# Port configurations for buildings
# FORMAT: (cell_x, cell_y, cell_z, direction)
#   - cell_x, cell_y, cell_z: Position relative to building origin (INTERNAL to footprint)
#   - direction: Which edge the port is on ('W'=west, 'E'=east, 'N'=north, 'S'=south)
#     For inputs: direction items come FROM
#     For outputs: direction items go TO
# For EAST rotation (default): inputs from west, outputs to east
BUILDING_PORTS: Dict[BuildingType, Dict[str, List[Tuple[int, int, int, str]]]] = {
    # Belts - 1x1, pass-through
    BuildingType.BELT_FORWARD: {
        'inputs': [(0, 0, 0, 'W')],   # Input from west edge
        'outputs': [(0, 0, 0, 'E')],  # Output to east edge
    },
    BuildingType.BELT_LEFT: {
        'inputs': [(0, 0, 0, 'W')],   # Input from west edge
        'outputs': [(0, 0, 0, 'N')],  # Output to north edge (left turn)
    },
    BuildingType.BELT_RIGHT: {
        'inputs': [(0, 0, 0, 'W')],   # Input from west edge
        'outputs': [(0, 0, 0, 'S')],  # Output to south edge (right turn)
    },

    # Rotators - 1x1
    BuildingType.ROTATOR_CW: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'E')],
    },
    BuildingType.ROTATOR_CCW: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'E')],
    },
    BuildingType.ROTATOR_180: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'E')],
    },
    BuildingType.HALF_CUTTER: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'E')],
    },
    # Cutter: 1x2, input on top cell (0,0), outputs from both cells to east
    BuildingType.CUTTER: {
        'inputs': [(0, 0, 0, 'W')],  # Input on top cell from west
        'outputs': [(0, 0, 0, 'E'), (0, 1, 0, 'E')],  # Output from top and bottom cells to east
    },
    # Cutter Mirrored: 1x2, input on bottom cell (0,1), outputs from both cells
    BuildingType.CUTTER_MIRRORED: {
        'inputs': [(0, 1, 0, 'W')],  # Input on bottom cell from west (mirrored)
        'outputs': [(0, 1, 0, 'E'), (0, 0, 0, 'E')],  # Output from bottom and top cells
    },
    # Swapper: 1x2, inputs on both cells from west, outputs from both to east
    BuildingType.SWAPPER: {
        'inputs': [(0, 0, 0, 'W'), (0, 1, 0, 'W')],
        'outputs': [(0, 0, 0, 'E'), (0, 1, 0, 'E')],
    },
    # Stacker (Straight): inputs on floor 0 and 1, output from floor 0 to east
    BuildingType.STACKER: {
        'inputs': [(0, 0, 0, 'W'), (0, 0, 1, 'W')],  # Floor 0 and floor 1 inputs
        'outputs': [(0, 0, 0, 'E')],
    },
    # Stacker (Bent): inputs same, output to south
    BuildingType.STACKER_BENT: {
        'inputs': [(0, 0, 0, 'W'), (0, 0, 1, 'W')],
        'outputs': [(0, 0, 0, 'S')],  # Output to south
    },
    # Stacker (Bent Mirrored): output to north
    BuildingType.STACKER_BENT_MIRRORED: {
        'inputs': [(0, 0, 0, 'W'), (0, 0, 1, 'W')],
        'outputs': [(0, 0, 0, 'N')],  # Output to north
    },
    # Unstacker: input floor 0, outputs on floor 0 and 1
    BuildingType.UNSTACKER: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'E'), (0, 0, 1, 'E')],
    },
    # Painter: shape from west, color from north
    BuildingType.PAINTER: {
        'inputs': [(0, 0, 0, 'W'), (0, 0, 0, 'N')],
        'outputs': [(0, 0, 0, 'E')],
    },
    # Painter Mirrored: color from south (at bottom cell of 1x2 building)
    BuildingType.PAINTER_MIRRORED: {
        'inputs': [(0, 0, 0, 'W'), (0, 1, 0, 'S')],
        'outputs': [(0, 0, 0, 'E')],
    },
    BuildingType.TRASH: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [],
    },
    BuildingType.PIN_PUSHER: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'E')],
    },
    # Splitters: input from west, outputs to north and south
    BuildingType.SPLITTER: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'N'), (0, 0, 0, 'S')],
    },
    BuildingType.SPLITTER_LEFT: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'N'), (0, 0, 0, 'S')],
    },
    BuildingType.SPLITTER_RIGHT: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'N'), (0, 0, 0, 'S')],
    },
    # Mergers: inputs from north and south, output to east
    BuildingType.MERGER: {
        'inputs': [(0, 0, 0, 'N'), (0, 0, 0, 'S')],
        'outputs': [(0, 0, 0, 'E')],
    },
    # Lifts: vertical movement between floors
    BuildingType.LIFT_UP: {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 1, 'E')],
    },
    BuildingType.LIFT_DOWN: {
        'inputs': [(0, 0, 1, 'W')],
        'outputs': [(0, 0, 0, 'E')],
    },
    # Belt ports (teleporters) - sender/receiver pairs
    # Note: sender has no physical output, receiver has no physical input
    # They are linked by channel ID in the game
    BuildingType.BELT_PORT_SENDER: {
        'inputs': [(0, 0, 0, 'W')],   # Input from west edge
        'outputs': [],                 # No physical output (teleports)
    },
    BuildingType.BELT_PORT_RECEIVER: {
        'inputs': [],                  # No physical input (receives teleport)
        'outputs': [(0, 0, 0, 'E')],   # Output to east edge
    },
}


# Legacy compatibility - building sizes (width, height) in grid cells
BUILDING_SIZES: Dict[BuildingType, Tuple[int, int]] = {
    bt: (spec.width, spec.height) for bt, spec in BUILDING_SPECS.items()
}


# Rotation codes: 0=East, 1=South, 2=West, 3=North
class Rotation(Enum):
    EAST = 0   # Right (default) - input from west, output to east
    SOUTH = 1  # Down - input from north, output to south
    WEST = 2   # Left - input from east, output to west
    NORTH = 3  # Up - input from south, output to north

    def get_input_direction(self) -> Tuple[int, int]:
        """Get the direction items come FROM (dx, dy)."""
        dirs = {
            Rotation.EAST: (-1, 0),   # Input from west
            Rotation.SOUTH: (0, -1),  # Input from north
            Rotation.WEST: (1, 0),    # Input from east
            Rotation.NORTH: (0, 1),   # Input from south
        }
        return dirs[self]

    def get_output_direction(self) -> Tuple[int, int]:
        """Get the direction items go TO (dx, dy)."""
        dirs = {
            Rotation.EAST: (1, 0),    # Output to east
            Rotation.SOUTH: (0, 1),   # Output to south
            Rotation.WEST: (-1, 0),   # Output to west
            Rotation.NORTH: (0, -1),  # Output to north
        }
        return dirs[self]


def _rotate_position(x: int, y: int, rotation: Rotation) -> Tuple[int, int]:
    """Rotate a relative position by the given rotation."""
    if rotation == Rotation.EAST:
        return (x, y)
    elif rotation == Rotation.SOUTH:
        return (-y, x)
    elif rotation == Rotation.WEST:
        return (-x, -y)
    else:  # NORTH
        return (y, -x)


def _rotate_direction(direction: str, rotation: Rotation) -> str:
    """Rotate a direction by the given rotation."""
    directions = ['E', 'S', 'W', 'N']  # Clockwise order
    rotations = {
        Rotation.EAST: 0,
        Rotation.SOUTH: 1,
        Rotation.WEST: 2,
        Rotation.NORTH: 3,
    }
    idx = directions.index(direction)
    new_idx = (idx + rotations[rotation]) % 4
    return directions[new_idx]


def get_building_ports(building_type: BuildingType, rotation: Rotation = Rotation.EAST) -> Dict[str, List[Tuple[int, int, int, str]]]:
    """Get input/output port positions for a building type and rotation.

    Returns ports in format: (cell_x, cell_y, cell_z, direction)
    All positions are INTERNAL to the building footprint.
    """
    default_ports = {
        'inputs': [(0, 0, 0, 'W')],
        'outputs': [(0, 0, 0, 'E')],
    }
    ports = BUILDING_PORTS.get(building_type, default_ports)

    if rotation == Rotation.EAST:
        return ports  # Default orientation

    # Rotate ports for other orientations
    rotated = {'inputs': [], 'outputs': []}

    for port_type in ['inputs', 'outputs']:
        for port in ports.get(port_type, []):
            if len(port) == 4:
                x, y, z, direction = port
                rot_x, rot_y = _rotate_position(x, y, rotation)
                rot_dir = _rotate_direction(direction, rotation)
                rotated[port_type].append((rot_x, rot_y, z, rot_dir))
            else:
                # Legacy 3-tuple format - shouldn't happen anymore
                x, y, z = port
                rot_x, rot_y = _rotate_position(x, y, rotation)
                rotated[port_type].append((rot_x, rot_y, z, 'W' if port_type == 'inputs' else 'E'))

    return rotated


def get_throughput_ratio(building_type: BuildingType) -> int:
    """Get how many of this building are needed per full belt."""
    return BUILDING_SPECS.get(building_type, BuildingSpec(1,1,1,1,1,1,30,90)).per_belt


def get_building_rate(building_type: BuildingType, tier: int = 1) -> float:
    """Get operations per minute for a building at given tier (1-5)."""
    spec = BUILDING_SPECS.get(building_type)
    if not spec:
        return 30.0
    # Linear interpolation between base and max rate
    tier = max(1, min(5, tier))
    rate_range = spec.max_rate - spec.base_rate
    return spec.base_rate + (rate_range * (tier - 1) / 4)


def get_throughput_per_second(building_type: BuildingType, tier: int = 1) -> float:
    """Get items per second for a building at given tier (1-5)."""
    ops_per_min = get_building_rate(building_type, tier)
    return ops_per_min / 60.0


def get_belt_max_throughput(tier: int = 5) -> float:
    """Get max belt throughput in items/second at given tier."""
    return get_throughput_per_second(BuildingType.BELT_FORWARD, tier)


# Belt throughput constants (items per second)
BELT_THROUGHPUT_TIER1 = 3.0   # 180 ops/min = 3 items/sec
BELT_THROUGHPUT_TIER5 = 3.0   # Belts don't upgrade speed, always 180 ops/min
