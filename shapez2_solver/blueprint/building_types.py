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

    # Cutter: 1x2x1, 4 per belt, 15-45 ops/min, 1 in -> 2 out
    BuildingType.HALF_CUTTER: BuildingSpec(1, 1, 1, 1, 1, 4, 15, 45),
    BuildingType.CUTTER: BuildingSpec(1, 2, 1, 1, 2, 4, 15, 45),

    # Swapper: 2x2, 2 in -> 2 out
    BuildingType.SWAPPER: BuildingSpec(2, 2, 1, 2, 2, 4, 15, 45),

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

    # Painter: 1x2, shape + color inputs
    BuildingType.PAINTER: BuildingSpec(1, 2, 1, 2, 1, 4, 15, 45),

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
# Format: {BuildingType: {'inputs': [(rel_x, rel_y, floor), ...], 'outputs': [(rel_x, rel_y, floor), ...]}}
# rel_x, rel_y are relative to building origin, floor is 0-indexed
# For EAST rotation (default): inputs from west (-x), outputs to east (+x)
BUILDING_PORTS: Dict[BuildingType, Dict[str, List[Tuple[int, int, int]]]] = {
    # Single floor buildings - all ports on floor 0
    BuildingType.BELT_FORWARD: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(1, 0, 0)],
    },
    BuildingType.ROTATOR_CW: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(1, 0, 0)],
    },
    BuildingType.ROTATOR_CCW: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(1, 0, 0)],
    },
    BuildingType.ROTATOR_180: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(1, 0, 0)],
    },
    BuildingType.HALF_CUTTER: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(1, 0, 0)],
    },
    # Cutter: 1x2, input on one side, two outputs
    BuildingType.CUTTER: {
        'inputs': [(-1, 0, 0)],  # Input from west
        'outputs': [(1, 0, 0), (1, 1, 0)],  # Two outputs to east (left half, right half)
    },
    # Swapper: 2x2, two inputs, two outputs
    BuildingType.SWAPPER: {
        'inputs': [(-1, 0, 0), (-1, 1, 0)],  # Two inputs from west
        'outputs': [(2, 0, 0), (2, 1, 0)],   # Two outputs to east
    },
    # Stacker (Straight): 2 floors - bottom input floor 0, top input floor 1, output floor 0
    BuildingType.STACKER: {
        'inputs': [(-1, 0, 0), (-1, 0, 1)],  # Bottom shape from floor 0, top shape from floor 1
        'outputs': [(1, 0, 0)],               # Output on floor 0 (straight ahead)
    },
    # Stacker (Bent): 2 floors - same inputs, but output is perpendicular (to south)
    BuildingType.STACKER_BENT: {
        'inputs': [(-1, 0, 0), (-1, 0, 1)],  # Bottom shape from floor 0, top shape from floor 1
        'outputs': [(0, 1, 0)],               # Output on floor 0 (bent to south)
    },
    # Stacker (Bent Mirrored): 2 floors - same inputs, output bent to north
    BuildingType.STACKER_BENT_MIRRORED: {
        'inputs': [(-1, 0, 0), (-1, 0, 1)],  # Bottom shape from floor 0, top shape from floor 1
        'outputs': [(0, -1, 0)],              # Output on floor 0 (bent to north)
    },
    # Unstacker: 2 floors - input floor 0, outputs on both floors
    BuildingType.UNSTACKER: {
        'inputs': [(-1, 0, 0)],               # Input on floor 0
        'outputs': [(1, 0, 0), (1, 0, 1)],    # Bottom layer to floor 0, top layer to floor 1
    },
    # Painter: shape input + color input
    BuildingType.PAINTER: {
        'inputs': [(-1, 0, 0), (0, -1, 0)],  # Shape from west, color from north
        'outputs': [(1, 0, 0)],
    },
    BuildingType.TRASH: {
        'inputs': [(-1, 0, 0)],
        'outputs': [],
    },
    BuildingType.PIN_PUSHER: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(1, 0, 0)],
    },
    # Splitters: 1 input -> 2 outputs (left and right side)
    BuildingType.SPLITTER: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(0, -1, 0), (0, 1, 0)],  # Split to north and south
    },
    BuildingType.SPLITTER_LEFT: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(0, -1, 0), (0, 1, 0)],  # Priority to north
    },
    BuildingType.SPLITTER_RIGHT: {
        'inputs': [(-1, 0, 0)],
        'outputs': [(0, -1, 0), (0, 1, 0)],  # Priority to south
    },
    # Mergers: 2 inputs -> 1 output
    BuildingType.MERGER: {
        'inputs': [(0, -1, 0), (0, 1, 0)],  # From north and south
        'outputs': [(1, 0, 0)],
    },
    # Lifts: vertical movement between floors
    BuildingType.LIFT_UP: {
        'inputs': [(-1, 0, 0)],   # Input on floor 0
        'outputs': [(1, 0, 1)],   # Output on floor 1
    },
    BuildingType.LIFT_DOWN: {
        'inputs': [(-1, 0, 1)],   # Input on floor 1
        'outputs': [(1, 0, 0)],   # Output on floor 0
    },
    # Belt ports (teleporters) - sender/receiver pairs
    # Note: sender has no physical output, receiver has no physical input
    # They are linked by channel ID in the game
    BuildingType.BELT_PORT_SENDER: {
        'inputs': [(-1, 0, 0)],   # Input from west
        'outputs': [],            # No physical output (teleports)
    },
    BuildingType.BELT_PORT_RECEIVER: {
        'inputs': [],             # No physical input (receives teleport)
        'outputs': [(1, 0, 0)],   # Output to east
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


def get_building_ports(building_type: BuildingType, rotation: Rotation = Rotation.EAST) -> Dict[str, List[Tuple[int, int, int]]]:
    """Get input/output port positions for a building type and rotation."""
    ports = BUILDING_PORTS.get(building_type, {
        'inputs': [(-1, 0, 0)],
        'outputs': [(1, 0, 0)],
    })

    if rotation == Rotation.EAST:
        return ports  # Default orientation

    # Rotate ports for other orientations
    # TODO: Implement rotation transformation
    return ports


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
