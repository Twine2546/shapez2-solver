"""
Grid-based simulation for Shapez 2 layouts.

Simulates shape flow through buildings connected by belts on a 3D grid.
Belts must physically connect for shapes to flow - no teleportation!
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum, auto
import copy

from ..shapes.shape import Shape, ShapeLayer, ShapePart
from ..shapes.parser import ShapeCodeParser
from ..blueprint.building_types import (
    BuildingType, Rotation, BUILDING_SPECS, BuildingSpec, BUILDING_PORTS
)
from ..operations.cutter import CutOperation, HalfDestroyerOperation, SwapperOperation
from ..operations.rotator import RotateOperation
from ..operations.stacker import StackOperation, UnstackOperation


class Direction(Enum):
    """Cardinal directions for belt flow."""
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)
    NORTH = (0, -1)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    @property
    def opposite(self) -> "Direction":
        opposites = {
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
        }
        return opposites[self]

    @classmethod
    def from_rotation(cls, rotation: Rotation) -> "Direction":
        """Convert Rotation enum to Direction."""
        mapping = {
            Rotation.EAST: Direction.EAST,
            Rotation.SOUTH: Direction.SOUTH,
            Rotation.WEST: Direction.WEST,
            Rotation.NORTH: Direction.NORTH,
        }
        return mapping[rotation]


@dataclass
class GridCell:
    """A cell in the simulation grid."""
    x: int
    y: int
    floor: int

    # What's in this cell
    building_type: Optional[BuildingType] = None
    building_id: Optional[int] = None
    rotation: Rotation = Rotation.EAST

    # For belts: flow direction
    flow_direction: Optional[Direction] = None

    # Shape currently on this cell (for belts)
    shape: Optional[Shape] = None

    def is_belt(self) -> bool:
        return self.building_type in (
            BuildingType.BELT_FORWARD,
            BuildingType.BELT_LEFT,
            BuildingType.BELT_RIGHT,
        )

    def is_lift(self) -> bool:
        return self.building_type in (
            BuildingType.LIFT_UP,
            BuildingType.LIFT_DOWN,
        )

    def is_building(self) -> bool:
        return self.building_type is not None and not self.is_belt() and not self.is_lift()


@dataclass
class SimBuilding:
    """A building in the simulation."""
    building_id: int
    building_type: BuildingType
    x: int
    y: int
    floor: int
    rotation: Rotation

    # Input/output buffers
    input_shapes: List[Optional[Shape]] = field(default_factory=list)
    output_shapes: List[Optional[Shape]] = field(default_factory=list)

    # Port positions (absolute grid coordinates)
    input_ports: List[Tuple[int, int, int]] = field(default_factory=list)
    output_ports: List[Tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        spec = BUILDING_SPECS.get(self.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
        self.input_shapes = [None] * spec.num_inputs
        self.output_shapes = [None] * spec.num_outputs
        self._calculate_ports()

    def _calculate_ports(self):
        """Calculate absolute port positions based on rotation."""
        ports = BUILDING_PORTS.get(self.building_type, {
            'inputs': [(-1, 0, 0)],
            'outputs': [(1, 0, 0)],
        })

        # Rotate port positions based on building rotation
        def rotate_offset(dx: int, dy: int, rotation: Rotation) -> Tuple[int, int]:
            """Rotate a relative offset by the building rotation."""
            if rotation == Rotation.EAST:
                return (dx, dy)
            elif rotation == Rotation.SOUTH:
                return (-dy, dx)
            elif rotation == Rotation.WEST:
                return (-dx, -dy)
            elif rotation == Rotation.NORTH:
                return (dy, -dx)
            return (dx, dy)

        self.input_ports = []
        for rel_x, rel_y, rel_floor in ports.get('inputs', []):
            rot_x, rot_y = rotate_offset(rel_x, rel_y, self.rotation)
            self.input_ports.append((
                self.x + rot_x,
                self.y + rot_y,
                self.floor + rel_floor
            ))

        self.output_ports = []
        for rel_x, rel_y, rel_floor in ports.get('outputs', []):
            rot_x, rot_y = rotate_offset(rel_x, rel_y, self.rotation)
            self.output_ports.append((
                self.x + rot_x,
                self.y + rot_y,
                self.floor + rel_floor
            ))


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    success: bool
    steps: int
    input_shapes: Dict[Tuple[int, int, int], Shape]  # Position -> shape fed in
    output_shapes: Dict[Tuple[int, int, int], Optional[Shape]]  # Position -> shape received
    expected_outputs: Dict[Tuple[int, int, int], Shape]  # What we expected
    errors: List[str] = field(default_factory=list)
    debug_log: List[str] = field(default_factory=list)

    def get_accuracy(self) -> float:
        """Calculate accuracy of outputs vs expected."""
        if not self.expected_outputs:
            return 0.0

        correct = 0
        total = len(self.expected_outputs)

        for pos, expected in self.expected_outputs.items():
            actual = self.output_shapes.get(pos)
            if actual is not None and expected is not None:
                if actual.to_code() == expected.to_code():
                    correct += 1

        return correct / total if total > 0 else 0.0

    def print_summary(self) -> str:
        """Print a summary of the simulation results."""
        lines = []
        lines.append("=" * 60)
        lines.append("SIMULATION RESULTS")
        lines.append("=" * 60)
        lines.append(f"Success: {self.success}")
        lines.append(f"Steps: {self.steps}")
        lines.append(f"Accuracy: {self.get_accuracy() * 100:.1f}%")

        lines.append("\nInputs:")
        for pos, shape in self.input_shapes.items():
            lines.append(f"  {pos}: {shape.to_code() if shape else 'None'}")

        lines.append("\nExpected Outputs:")
        for pos, shape in self.expected_outputs.items():
            lines.append(f"  {pos}: {shape.to_code() if shape else 'None'}")

        lines.append("\nActual Outputs:")
        for pos, shape in self.output_shapes.items():
            expected = self.expected_outputs.get(pos)
            match = "✓" if expected and shape and expected.to_code() == shape.to_code() else "✗"
            lines.append(f"  {pos}: {shape.to_code() if shape else 'None'} {match}")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  - {err}")

        return "\n".join(lines)


class GridSimulator:
    """
    Simulates shape flow through a grid of buildings and belts.

    Key principles:
    1. Belts must physically connect to transfer shapes
    2. Buildings process shapes when all inputs are available
    3. Shapes move one cell per simulation step
    """

    def __init__(self, width: int = 32, height: int = 32, num_floors: int = 4):
        self.width = width
        self.height = height
        self.num_floors = num_floors

        # 3D grid: (x, y, floor) -> GridCell
        self.grid: Dict[Tuple[int, int, int], GridCell] = {}

        # Buildings by ID
        self.buildings: Dict[int, SimBuilding] = {}

        # Input/output positions
        self.input_positions: Dict[Tuple[int, int, int], Shape] = {}
        self.output_positions: Set[Tuple[int, int, int]] = set()

        # Shapes waiting at output positions
        self.collected_outputs: Dict[Tuple[int, int, int], Optional[Shape]] = {}

        self.parser = ShapeCodeParser()
        self.debug = False

    def clear(self):
        """Clear the simulation grid."""
        self.grid.clear()
        self.buildings.clear()
        self.input_positions.clear()
        self.output_positions.clear()
        self.collected_outputs.clear()

    def add_building(
        self,
        building_id: int,
        building_type: BuildingType,
        x: int, y: int, floor: int,
        rotation: Rotation = Rotation.EAST
    ) -> SimBuilding:
        """Add a building to the grid."""
        building = SimBuilding(
            building_id=building_id,
            building_type=building_type,
            x=x, y=y, floor=floor,
            rotation=rotation
        )
        self.buildings[building_id] = building

        # Mark grid cells as occupied
        spec = BUILDING_SPECS.get(building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))
        for dx in range(spec.width):
            for dy in range(spec.height):
                for df in range(spec.depth):
                    pos = (x + dx, y + dy, floor + df)
                    self.grid[pos] = GridCell(
                        x=x + dx, y=y + dy, floor=floor + df,
                        building_type=building_type,
                        building_id=building_id,
                        rotation=rotation
                    )

        return building

    def add_belt(
        self,
        x: int, y: int, floor: int,
        belt_type: BuildingType,
        rotation: Rotation = Rotation.EAST
    ):
        """Add a belt segment to the grid.

        For belts, 'rotation' indicates the OUTPUT direction for forward belts,
        or the INPUT direction for turn belts.

        BELT_FORWARD: rotation = output direction
        BELT_LEFT: rotation = input direction, output turns left (CCW)
        BELT_RIGHT: rotation = input direction, output turns right (CW)
        """
        # For turn belts, rotation is the INPUT direction
        # For forward belts, rotation is both input and output direction
        input_dir = Direction.from_rotation(rotation)

        if belt_type == BuildingType.BELT_FORWARD:
            output_dir = input_dir
        elif belt_type == BuildingType.BELT_LEFT:
            # Left turn: output goes CCW from input direction
            turns_ccw = {
                Direction.EAST: Direction.NORTH,
                Direction.SOUTH: Direction.EAST,
                Direction.WEST: Direction.SOUTH,
                Direction.NORTH: Direction.WEST,
            }
            output_dir = turns_ccw[input_dir]
        elif belt_type == BuildingType.BELT_RIGHT:
            # Right turn: output goes CW from input direction
            turns_cw = {
                Direction.EAST: Direction.SOUTH,
                Direction.SOUTH: Direction.WEST,
                Direction.WEST: Direction.NORTH,
                Direction.NORTH: Direction.EAST,
            }
            output_dir = turns_cw[input_dir]
        else:
            output_dir = input_dir

        pos = (x, y, floor)
        self.grid[pos] = GridCell(
            x=x, y=y, floor=floor,
            building_type=belt_type,
            rotation=rotation,
            flow_direction=output_dir  # Store output direction
        )

    def add_lift(
        self,
        x: int, y: int, floor: int,
        lift_type: BuildingType,
        rotation: Rotation = Rotation.EAST
    ):
        """Add a lift to the grid."""
        pos = (x, y, floor)
        self.grid[pos] = GridCell(
            x=x, y=y, floor=floor,
            building_type=lift_type,
            rotation=rotation,
            flow_direction=Direction.from_rotation(rotation)
        )

        # Lifts occupy 2 floors
        if lift_type == BuildingType.LIFT_UP:
            pos2 = (x, y, floor + 1)
            self.grid[pos2] = GridCell(
                x=x, y=y, floor=floor + 1,
                building_type=lift_type,
                rotation=rotation,
                flow_direction=Direction.from_rotation(rotation)
            )
        elif lift_type == BuildingType.LIFT_DOWN:
            if floor > 0:
                pos2 = (x, y, floor - 1)
                self.grid[pos2] = GridCell(
                    x=x, y=y, floor=floor - 1,
                    building_type=lift_type,
                    rotation=rotation,
                    flow_direction=Direction.from_rotation(rotation)
                )

    def set_input(self, x: int, y: int, floor: int, shape: Shape):
        """Set an input position with a shape to feed in."""
        self.input_positions[(x, y, floor)] = shape

    def set_output(self, x: int, y: int, floor: int):
        """Set a position as an output collection point."""
        self.output_positions.add((x, y, floor))
        self.collected_outputs[(x, y, floor)] = None

    def _get_belt_input_position(self, cell: GridCell) -> Tuple[int, int, int]:
        """Get where a belt receives input from.

        For ALL belt types (forward, left, right), the input comes from the
        opposite of the rotation (facing) direction. The rotation indicates
        which way the belt points, so input is from behind.
        """
        if cell.is_belt() or cell.is_lift():
            # Input comes from opposite of the rotation/facing direction
            facing_dir = Direction.from_rotation(cell.rotation)
            in_dir = facing_dir.opposite
            return (cell.x + in_dir.dx, cell.y + in_dir.dy, cell.floor)
        return (cell.x - 1, cell.y, cell.floor)

    def _get_belt_output_position(self, cell: GridCell) -> Tuple[int, int, int]:
        """Get where a belt outputs to."""
        if cell.flow_direction:
            return (
                cell.x + cell.flow_direction.dx,
                cell.y + cell.flow_direction.dy,
                cell.floor
            )
        return (cell.x + 1, cell.y, cell.floor)

    def _check_belt_connection(
        self,
        from_pos: Tuple[int, int, int],
        to_pos: Tuple[int, int, int]
    ) -> bool:
        """Check if two belt positions are properly connected."""
        from_cell = self.grid.get(from_pos)
        to_cell = self.grid.get(to_pos)

        if not from_cell or not to_cell:
            return False

        # Check if from_cell outputs to to_pos
        if from_cell.is_belt() or from_cell.is_lift():
            out_pos = self._get_belt_output_position(from_cell)
            if out_pos != to_pos:
                return False

        # Check if to_cell can receive from from_pos direction
        if to_cell.is_belt() or to_cell.is_lift():
            in_pos = self._get_belt_input_position(to_cell)
            # Allow connection if the input position matches from_pos
            if in_pos != from_pos:
                return False

        return True

    def _can_belt_receive_from(self, belt_cell: GridCell, from_pos: Tuple[int, int, int]) -> bool:
        """Check if a belt can receive input from a specific position."""
        if not belt_cell.is_belt() and not belt_cell.is_lift():
            return False

        in_pos = self._get_belt_input_position(belt_cell)
        return in_pos == from_pos

    def _process_building(self, building: SimBuilding) -> bool:
        """
        Process a building - transform inputs to outputs.
        Returns True if processing occurred.
        """
        # Check if all required inputs are available
        spec = BUILDING_SPECS.get(building.building_type)
        if not spec:
            return False

        # Skip if not all inputs received
        if spec.num_inputs > 0:
            if any(s is None for s in building.input_shapes[:spec.num_inputs]):
                return False

        # Process based on building type
        outputs = self._execute_operation(building)

        if outputs:
            building.output_shapes = list(outputs)
            # Clear inputs after processing
            building.input_shapes = [None] * spec.num_inputs
            return True

        return False

    def _execute_operation(self, building: SimBuilding) -> Optional[Tuple[Optional[Shape], ...]]:
        """Execute the building's operation on its inputs."""
        bt = building.building_type
        inputs = building.input_shapes

        try:
            if bt == BuildingType.ROTATOR_CW:
                op = RotateOperation(steps=1)
                return op.execute(inputs[0] if inputs else None)

            elif bt == BuildingType.ROTATOR_CCW:
                op = RotateOperation(steps=3)  # CCW = 3 CW steps
                return op.execute(inputs[0] if inputs else None)

            elif bt == BuildingType.ROTATOR_180:
                op = RotateOperation(steps=2)
                return op.execute(inputs[0] if inputs else None)

            elif bt == BuildingType.CUTTER:
                op = CutOperation()
                return op.execute(inputs[0] if inputs else None)

            elif bt == BuildingType.CUTTER_MIRRORED:
                # Mirrored cutter swaps outputs
                op = CutOperation()
                result = op.execute(inputs[0] if inputs else None)
                if result and len(result) == 2:
                    return (result[1], result[0])
                return result

            elif bt == BuildingType.HALF_CUTTER:
                op = HalfDestroyerOperation()
                return op.execute(inputs[0] if inputs else None)

            elif bt == BuildingType.SWAPPER:
                op = SwapperOperation()
                return op.execute(inputs[0] if len(inputs) > 0 else None,
                                  inputs[1] if len(inputs) > 1 else None)

            elif bt == BuildingType.STACKER:
                op = StackOperation()
                return op.execute(inputs[0] if len(inputs) > 0 else None,
                                  inputs[1] if len(inputs) > 1 else None)

            elif bt == BuildingType.UNSTACKER:
                op = UnstackOperation()
                return op.execute(inputs[0] if inputs else None)

            elif bt == BuildingType.PIN_PUSHER:
                # Pin pusher passes through (simplified)
                return (inputs[0] if inputs else None,)

            elif bt == BuildingType.TRASH:
                # Trash consumes input, no output
                return ()

            else:
                # Unknown building, pass through
                return (inputs[0] if inputs else None,)

        except Exception as e:
            if self.debug:
                print(f"Operation error for {bt}: {e}")
            return None

    def simulate(
        self,
        max_steps: int = 100,
        expected_outputs: Optional[Dict[Tuple[int, int, int], Shape]] = None
    ) -> SimulationResult:
        """
        Run the simulation.

        Args:
            max_steps: Maximum simulation steps
            expected_outputs: Expected shapes at output positions

        Returns:
            SimulationResult with actual outputs and comparison
        """
        result = SimulationResult(
            success=False,
            steps=0,
            input_shapes=dict(self.input_positions),
            output_shapes={},
            expected_outputs=expected_outputs or {}
        )

        # Reset building states
        for building in self.buildings.values():
            spec = BUILDING_SPECS.get(building.building_type)
            if spec:
                building.input_shapes = [None] * spec.num_inputs
                building.output_shapes = [None] * spec.num_outputs

        # Reset collected outputs
        for pos in self.output_positions:
            self.collected_outputs[pos] = None

        # Shapes currently on belts: pos -> shape
        belt_shapes: Dict[Tuple[int, int, int], Shape] = {}

        # Feed initial shapes from input positions to adjacent belts
        for input_pos, shape in self.input_positions.items():
            # Find adjacent belt that can receive input
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                adj_pos = (input_pos[0] + dx, input_pos[1] + dy, input_pos[2])
                adj_cell = self.grid.get(adj_pos)
                if adj_cell and (adj_cell.is_belt() or adj_cell.is_lift()):
                    # Check if this belt receives from input direction
                    in_pos = self._get_belt_input_position(adj_cell)
                    if in_pos == input_pos:
                        belt_shapes[adj_pos] = shape
                        if self.debug:
                            result.debug_log.append(f"Fed input to belt at {adj_pos}: {shape.to_code()}")
                        break

        for step in range(max_steps):
            result.steps = step + 1
            changed = False

            if self.debug:
                result.debug_log.append(f"\n--- Step {step + 1} ---")

            # 2. Move shapes on belts
            new_belt_shapes: Dict[Tuple[int, int, int], Shape] = {}

            for pos, shape in belt_shapes.items():
                cell = self.grid.get(pos)

                if cell is None:
                    # Shape fell off the grid
                    if self.debug:
                        result.debug_log.append(f"  Shape lost at {pos} (no cell)")
                    continue

                if cell.is_belt():
                    # Move shape along belt
                    out_pos = self._get_belt_output_position(cell)
                    out_cell = self.grid.get(out_pos)

                    if out_pos in self.output_positions:
                        # Reached output
                        self.collected_outputs[out_pos] = shape
                        if self.debug:
                            result.debug_log.append(f"  Collected at {out_pos}: {shape.to_code()}")
                        changed = True

                    elif out_cell is not None:
                        if out_cell.is_belt() or out_cell.is_lift():
                            # Check proper connection
                            if self._check_belt_connection(pos, out_pos):
                                if out_pos not in new_belt_shapes:
                                    new_belt_shapes[out_pos] = shape
                                    changed = True
                                    if self.debug:
                                        result.debug_log.append(f"  Moved {pos} -> {out_pos}")
                                else:
                                    # Collision - shape lost
                                    if self.debug:
                                        result.debug_log.append(f"  Collision at {out_pos}")
                            else:
                                if self.debug:
                                    result.debug_log.append(f"  Belt not connected: {pos} -> {out_pos}")

                        elif out_cell.is_building():
                            # Deliver to building input
                            # The belt at 'pos' outputs to 'out_pos' which is inside the building
                            # Check if 'pos' matches one of the building's input ports
                            building = self.buildings.get(out_cell.building_id)
                            if building:
                                # Input ports are positions where items come FROM
                                # So check if current belt position matches an input port
                                delivered = False
                                for i, in_port in enumerate(building.input_ports):
                                    if in_port == pos:
                                        if i < len(building.input_shapes) and building.input_shapes[i] is None:
                                            building.input_shapes[i] = shape
                                            changed = True
                                            delivered = True
                                            if self.debug:
                                                result.debug_log.append(
                                                    f"  Delivered to building {building.building_id} input {i}"
                                                )
                                        break
                                if not delivered and self.debug:
                                    result.debug_log.append(
                                        f"  Cannot deliver to building {building.building_id}: pos={pos}, ports={building.input_ports}"
                                    )
                    else:
                        # No destination
                        if self.debug:
                            result.debug_log.append(f"  Shape stuck at {pos} (no connection)")

                elif cell.is_lift():
                    # Handle lift movement
                    # Lifts occupy 2 floors. When a shape enters at the entry floor,
                    # it moves to the exit floor. At the exit floor, it outputs
                    # to an adjacent belt in the facing direction.

                    if cell.building_type == BuildingType.LIFT_UP:
                        # Check if we're at the bottom (entry) or top (exit) of the lift
                        top_floor_pos = (cell.x, cell.y, cell.floor + 1)
                        top_cell = self.grid.get(top_floor_pos)

                        if top_cell and top_cell.building_type == BuildingType.LIFT_UP:
                            # We're at the bottom, move to top
                            new_belt_shapes[top_floor_pos] = shape
                            changed = True
                            if self.debug:
                                result.debug_log.append(f"  Lift moved {pos} -> {top_floor_pos}")
                        else:
                            # We're at the top, exit to adjacent belt in facing direction
                            facing_dir = Direction.from_rotation(cell.rotation)
                            exit_pos = (cell.x + facing_dir.dx, cell.y + facing_dir.dy, cell.floor)

                            # Check if exit pos is an output
                            if exit_pos in self.output_positions:
                                self.collected_outputs[exit_pos] = shape
                                changed = True
                                if self.debug:
                                    result.debug_log.append(f"  Lift output to {exit_pos}: {shape.to_code()}")
                            else:
                                exit_cell = self.grid.get(exit_pos)
                                if exit_cell and (exit_cell.is_belt() or exit_cell.is_lift()):
                                    new_belt_shapes[exit_pos] = shape
                                    changed = True
                                    if self.debug:
                                        result.debug_log.append(f"  Lift exit {pos} -> {exit_pos}")
                    else:
                        # LIFT_DOWN: similar but reversed
                        bottom_floor_pos = (cell.x, cell.y, cell.floor - 1)
                        bottom_cell = self.grid.get(bottom_floor_pos)

                        if bottom_cell and bottom_cell.building_type == BuildingType.LIFT_DOWN:
                            # We're at the top, move to bottom
                            new_belt_shapes[bottom_floor_pos] = shape
                            changed = True
                            if self.debug:
                                result.debug_log.append(f"  Lift moved {pos} -> {bottom_floor_pos}")
                        else:
                            # We're at the bottom, exit to adjacent belt
                            facing_dir = Direction.from_rotation(cell.rotation)
                            exit_pos = (cell.x + facing_dir.dx, cell.y + facing_dir.dy, cell.floor)

                            if exit_pos in self.output_positions:
                                self.collected_outputs[exit_pos] = shape
                                changed = True
                            else:
                                exit_cell = self.grid.get(exit_pos)
                                if exit_cell and (exit_cell.is_belt() or exit_cell.is_lift()):
                                    new_belt_shapes[exit_pos] = shape
                                    changed = True

            belt_shapes = new_belt_shapes

            # 3. Process buildings
            for building in self.buildings.values():
                if self._process_building(building):
                    changed = True
                    if self.debug:
                        result.debug_log.append(
                            f"  Building {building.building_id} processed: "
                            f"{[s.to_code() if s else 'None' for s in building.output_shapes]}"
                        )

                    # Put outputs onto output belts
                    for i, out_port in enumerate(building.output_ports):
                        if i < len(building.output_shapes) and building.output_shapes[i]:
                            output_shape = building.output_shapes[i]
                            placed = False

                            # out_port is where the output goes TO
                            # Check if there's a belt at that position
                            belt_cell = self.grid.get(out_port)
                            if belt_cell and (belt_cell.is_belt() or belt_cell.is_lift()):
                                # Check if the belt receives from the building direction
                                belt_shapes[out_port] = output_shape
                                building.output_shapes[i] = None
                                placed = True
                                if self.debug:
                                    result.debug_log.append(
                                        f"    Output {i} placed on belt at {out_port}"
                                    )

                            # Also check if output port is directly an output position
                            if not placed and out_port in self.output_positions:
                                self.collected_outputs[out_port] = output_shape
                                building.output_shapes[i] = None
                                placed = True

                            if not placed and self.debug:
                                result.debug_log.append(
                                    f"    Output {i} stuck at {out_port}: {output_shape.to_code()}"
                                )

            # 4. Check if simulation is complete
            all_outputs_collected = all(
                self.collected_outputs.get(pos) is not None
                for pos in self.output_positions
            )

            if all_outputs_collected and not belt_shapes:
                result.success = True
                break

            if not changed and not belt_shapes:
                # Simulation stalled
                result.errors.append("Simulation stalled - no shapes moving")
                break

        # Collect final results
        result.output_shapes = dict(self.collected_outputs)

        return result

    def _adjacent_to(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> bool:
        """Check if two positions are adjacent (including same position)."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        dz = abs(pos1[2] - pos2[2])
        return dx + dy + dz <= 1

    def print_grid(self, floor: int = 0) -> str:
        """Print ASCII representation of a grid floor."""
        # Find bounds
        min_x = min((p[0] for p in self.grid.keys() if p[2] == floor), default=0)
        max_x = max((p[0] for p in self.grid.keys() if p[2] == floor), default=10)
        min_y = min((p[1] for p in self.grid.keys() if p[2] == floor), default=0)
        max_y = max((p[1] for p in self.grid.keys() if p[2] == floor), default=10)

        # Add margin
        min_x -= 1
        min_y -= 1
        max_x += 2
        max_y += 2

        symbols = {
            BuildingType.BELT_FORWARD: '→',
            BuildingType.BELT_LEFT: '↰',
            BuildingType.BELT_RIGHT: '↱',
            BuildingType.LIFT_UP: '↑',
            BuildingType.LIFT_DOWN: '↓',
            BuildingType.ROTATOR_CW: 'R',
            BuildingType.ROTATOR_CCW: 'r',
            BuildingType.ROTATOR_180: '⟳',
            BuildingType.CUTTER: 'C',
            BuildingType.CUTTER_MIRRORED: 'c',
            BuildingType.HALF_CUTTER: 'H',
            BuildingType.SWAPPER: 'X',
            BuildingType.STACKER: 'S',
            BuildingType.UNSTACKER: 'U',
            BuildingType.PIN_PUSHER: 'P',
            BuildingType.TRASH: 'T',
        }

        lines = [f"Floor {floor}:"]
        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                pos = (x, y, floor)
                cell = self.grid.get(pos)

                if pos in self.input_positions:
                    row += "I"
                elif pos in self.output_positions:
                    row += "O"
                elif cell:
                    row += symbols.get(cell.building_type, "?")
                else:
                    row += "·"
            lines.append(row)

        return "\n".join(lines)
