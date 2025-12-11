"""
Flow Simulator for Shapez 2 Layouts

Simulates item flow through a factory:
- Tracks shape at every cell
- Calculates throughput and utilization
- Shows what shape comes out of each machine output
- Detects backed up ports and bottlenecks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shapez2_solver.blueprint.building_types import (
    BuildingType, Rotation, BUILDING_SPECS, BUILDING_PORTS
)


# Throughput constants (items per minute)
BELT_THROUGHPUT = 180.0
MACHINE_THROUGHPUT = {
    BuildingType.CUTTER: 45.0,
    BuildingType.CUTTER_MIRRORED: 45.0,
    BuildingType.HALF_CUTTER: 45.0,
    BuildingType.ROTATOR_CW: 90.0,
    BuildingType.ROTATOR_CCW: 90.0,
    BuildingType.ROTATOR_180: 90.0,
    BuildingType.STACKER: 30.0,
    BuildingType.STACKER_BENT: 30.0,
    BuildingType.STACKER_BENT_MIRRORED: 30.0,
    BuildingType.UNSTACKER: 30.0,
    BuildingType.SWAPPER: 45.0,
    BuildingType.PAINTER: 45.0,
    BuildingType.PAINTER_MIRRORED: 45.0,
    BuildingType.SPLITTER: 180.0,
    BuildingType.MERGER: 180.0,
}


def direction_delta(rotation: Rotation) -> Tuple[int, int]:
    """Get (dx, dy) for a direction."""
    return {
        Rotation.EAST: (1, 0),
        Rotation.WEST: (-1, 0),
        Rotation.SOUTH: (0, 1),
        Rotation.NORTH: (0, -1),
    }[rotation]


def delta_to_direction(dx: int, dy: int) -> Optional[Rotation]:
    """Convert (dx, dy) delta to a Rotation direction."""
    mapping = {
        (1, 0): Rotation.EAST,
        (-1, 0): Rotation.WEST,
        (0, 1): Rotation.SOUTH,
        (0, -1): Rotation.NORTH,
    }
    return mapping.get((dx, dy))


def get_belt_input_direction(building_type: BuildingType, rotation: Rotation) -> Rotation:
    """Get the direction a belt receives input FROM."""
    # All belt types receive from behind their facing direction
    # BELT_FORWARD facing EAST receives from WEST
    # BELT_LEFT facing EAST receives from WEST (turns left to north)
    # BELT_RIGHT facing EAST receives from WEST (turns right to south)
    opposite = {
        Rotation.EAST: Rotation.WEST,
        Rotation.WEST: Rotation.EAST,
        Rotation.NORTH: Rotation.SOUTH,
        Rotation.SOUTH: Rotation.NORTH,
    }
    return opposite[rotation]


def are_directions_opposite(dir1: Rotation, dir2: Rotation) -> bool:
    """Check if two directions are 180° apart (opposite)."""
    opposites = {
        (Rotation.EAST, Rotation.WEST), (Rotation.WEST, Rotation.EAST),
        (Rotation.NORTH, Rotation.SOUTH), (Rotation.SOUTH, Rotation.NORTH),
    }
    return (dir1, dir2) in opposites


def rotate_offset(dx: int, dy: int, rotation: Rotation) -> Tuple[int, int]:
    """Rotate a relative offset by rotation."""
    if rotation == Rotation.EAST:
        return (dx, dy)
    elif rotation == Rotation.SOUTH:
        return (-dy, dx)
    elif rotation == Rotation.WEST:
        return (-dx, -dy)
    else:  # NORTH
        return (dy, -dx)


@dataclass
class FlowCell:
    """Flow state at a single cell."""
    position: Tuple[int, int, int]
    building_type: Optional[BuildingType] = None
    rotation: Rotation = Rotation.EAST
    
    # Flow state
    shape: Optional[str] = None
    throughput: float = 0.0  # items/min flowing through
    max_throughput: float = BELT_THROUGHPUT
    
    # Connections
    source: Optional['FlowCell'] = None  # Where flow comes from
    destinations: List['FlowCell'] = field(default_factory=list)  # Where flow goes
    
    @property
    def utilization(self) -> float:
        """Belt/machine utilization as percentage."""
        if self.max_throughput <= 0:
            return 0.0
        return min(100.0, 100.0 * self.throughput / self.max_throughput)
    
    @property
    def is_saturated(self) -> bool:
        """Is this cell at max capacity?"""
        return self.throughput >= self.max_throughput * 0.99


@dataclass
class MachineState:
    """State of a machine during simulation."""
    origin: Tuple[int, int, int]
    building_type: BuildingType
    rotation: Rotation
    
    # Port info
    input_ports: List[Dict] = field(default_factory=list)  # [{position, shape, throughput}]
    output_ports: List[Dict] = field(default_factory=list)
    
    throughput: float = 0.0
    max_throughput: float = 45.0
    
    @property
    def utilization(self) -> float:
        if self.max_throughput <= 0:
            return 0.0
        return min(100.0, 100.0 * self.throughput / self.max_throughput)


class FlowSimulator:
    """
    Simulates item flow through a factory layout.
    
    Tracks:
    - Shape at every cell
    - Throughput and utilization
    - Shape transformations through machines
    - Final output shapes
    """
    
    def __init__(self, width: int = 14, height: int = 14, num_floors: int = 4):
        self.width = width
        self.height = height
        self.num_floors = num_floors
        
        # Grid of flow cells
        self.cells: Dict[Tuple[int, int, int], FlowCell] = {}
        
        # Buildings by origin position
        self.buildings: Dict[Tuple[int, int, int], Dict] = {}
        
        # Machine states
        self.machines: Dict[Tuple[int, int, int], MachineState] = {}
        
        # Edge I/O
        self.inputs: List[Dict] = []
        self.outputs: List[Dict] = []
        
        # Errors and warnings
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def _get_cell(self, x: int, y: int, floor: int) -> FlowCell:
        """Get or create a flow cell."""
        pos = (x, y, floor)
        if pos not in self.cells:
            self.cells[pos] = FlowCell(position=pos)
        return self.cells[pos]
    
    def place_building(self, building_type: BuildingType, x: int, y: int, floor: int, rotation: Rotation):
        """Place a building and set up its flow cell."""
        spec = BUILDING_SPECS.get(building_type)
        base_w = spec.width if spec else 1
        base_h = spec.height if spec else 1
        depth = spec.depth if spec else 1
        
        # Effective dimensions
        if rotation in (Rotation.SOUTH, Rotation.NORTH):
            eff_w, eff_h = base_h, base_w
        else:
            eff_w, eff_h = base_w, base_h
        
        origin = (x, y, floor)
        
        # Mark all cells occupied
        for dx in range(eff_w):
            for dy in range(eff_h):
                for dz in range(depth):
                    cell = self._get_cell(x + dx, y + dy, floor + dz)
                    cell.building_type = building_type
                    cell.rotation = rotation
                    if building_type in MACHINE_THROUGHPUT:
                        cell.max_throughput = MACHINE_THROUGHPUT[building_type]
        
        # Store building info
        self.buildings[origin] = {
            'type': building_type,
            'rotation': rotation,
            'cells': [(x + dx, y + dy, floor + dz) 
                     for dx in range(eff_w) for dy in range(eff_h) for dz in range(depth)]
        }
        
        # For machines, create state and calculate ports
        if building_type in MACHINE_THROUGHPUT:
            ports_def = BUILDING_PORTS.get(building_type, {'inputs': [(0,0,0,'W')], 'outputs': [(0,0,0,'E')]})

            input_ports = []
            for i, port_info in enumerate(ports_def.get('inputs', [])):
                # New format: (cell_x, cell_y, cell_z, direction) - all internal
                if len(port_info) == 4:
                    rel_x, rel_y, rel_z, direction = port_info
                else:
                    # Old format fallback
                    rel_x, rel_y, rel_z = port_info
                    direction = 'W'  # Default input from west

                rot_x, rot_y = rotate_offset(rel_x, rel_y, rotation)
                # Port position is ON the machine cell
                machine_cell_pos = (x + rot_x, y + rot_y, floor + rel_z)
                # Rotate direction based on machine rotation
                rot_dir = self._rotate_direction(direction, rotation)
                input_ports.append({
                    'index': i,
                    'position': machine_cell_pos,  # Position ON the machine
                    'direction': rot_dir,  # Which edge receives input
                    'shape': None,
                    'throughput': 0.0,
                    'connected': False,
                })

            output_ports = []
            for i, port_info in enumerate(ports_def.get('outputs', [])):
                if len(port_info) == 4:
                    rel_x, rel_y, rel_z, direction = port_info
                else:
                    rel_x, rel_y, rel_z = port_info
                    direction = 'E'  # Default output to east

                rot_x, rot_y = rotate_offset(rel_x, rel_y, rotation)
                machine_cell_pos = (x + rot_x, y + rot_y, floor + rel_z)
                rot_dir = self._rotate_direction(direction, rotation)
                output_ports.append({
                    'index': i,
                    'position': machine_cell_pos,  # Position ON the machine
                    'direction': rot_dir,  # Which edge outputs
                    'shape': None,
                    'throughput': 0.0,
                    'connected': False,
                    'backed_up': False,
                })

            self.machines[origin] = MachineState(
                origin=origin,
                building_type=building_type,
                rotation=rotation,
                input_ports=input_ports,
                output_ports=output_ports,
                max_throughput=MACHINE_THROUGHPUT[building_type],
            )

    def _rotate_direction(self, direction: str, rotation: Rotation) -> str:
        """Rotate a direction (W/E/N/S) by the building rotation."""
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

    def _direction_to_delta(self, direction: str) -> Tuple[int, int]:
        """Convert direction to (dx, dy) for finding adjacent cell."""
        return {
            'E': (1, 0),
            'W': (-1, 0),
            'N': (0, -1),
            'S': (0, 1),
        }[direction]
    
    def set_input(self, x: int, y: int, floor: int, shape: str, throughput: float = 180.0):
        """Set an input source."""
        self.inputs.append({
            'position': (x, y, floor),
            'shape': shape,
            'throughput': throughput,
        })
    
    def set_output(self, x: int, y: int, floor: int, expected_shape: Optional[str] = None):
        """Set an output sink."""
        self.outputs.append({
            'position': (x, y, floor),
            'expected_shape': expected_shape,
            'actual_shape': None,
            'throughput': 0.0,
        })
    
    def _normalize_shape(self, shape: str) -> str:
        """Normalize shape to 8-character format."""
        if not shape:
            return "--------"
        if len(shape) == 4:
            # Expand: "ABCD" -> "A-B-C-D-" (single layer)
            return "".join(c + "-" for c in shape)
        if len(shape) < 8:
            return shape + "-" * (8 - len(shape))
        return shape[:8]

    def _stack_shapes(self, bottom_shape: str, top_shape: str) -> str:
        """
        Stack two shapes (stacker operation).
        Top shape goes on top layer, bottom shape on bottom layer.
        In Shapez 2, stacking creates a 2-layer shape.
        """
        bottom = self._normalize_shape(bottom_shape)
        top = self._normalize_shape(top_shape)

        # Result: combine non-empty parts, top overwrites bottom in same positions
        result = []
        for i in range(0, 8, 2):
            bot_quarter = bottom[i:i+2]
            top_quarter = top[i:i+2]
            # If top has content, use it; otherwise use bottom
            if top_quarter != "--":
                result.append(top_quarter)
            else:
                result.append(bot_quarter)
        return "".join(result)

    def _unstack_shape(self, shape: str, output_index: int) -> str:
        """
        Unstack a shape (unstacker operation).
        Output 0 = top layer, Output 1 = bottom layer.
        For simple shapes, output 0 gets shape, output 1 gets empty.
        """
        norm = self._normalize_shape(shape)
        # For now, treat as: output 0 = shape, output 1 = empty (simplified)
        # Full implementation would track layers
        if output_index == 0:
            return norm
        return "--------"

    def _swap_quadrants(self, shape1: str, shape2: str, output_index: int) -> str:
        """
        Swapper: swaps right halves between two shapes.
        Input 0's left + Input 1's right -> Output 0
        Input 1's left + Input 0's right -> Output 1
        """
        s1 = self._normalize_shape(shape1)
        s2 = self._normalize_shape(shape2)

        # Extract quarters: TL, TR, BL, BR
        s1_tl, s1_tr, s1_bl, s1_br = s1[0:2], s1[2:4], s1[4:6], s1[6:8]
        s2_tl, s2_tr, s2_bl, s2_br = s2[0:2], s2[2:4], s2[4:6], s2[6:8]

        if output_index == 0:
            # Shape 1's left half + Shape 2's right half
            return f"{s1_tl}{s2_tr}{s1_bl}{s2_br}"
        else:
            # Shape 2's left half + Shape 1's right half
            return f"{s2_tl}{s1_tr}{s2_bl}{s1_br}"

    def _paint_shape(self, shape: str, color: str) -> str:
        """
        Painter: applies color to non-empty quarters.
        Color codes: Cu=copper, Ru=ruby, Gr=green, Bl=blue, Cy=cyan, Pu=purple, Ye=yellow, Wh=white
        """
        norm = self._normalize_shape(shape)
        if len(color) < 2:
            color = color + "-" if len(color) == 1 else "--"

        color_code = color[:2]
        result = []
        for i in range(0, 8, 2):
            quarter = norm[i:i+2]
            if quarter != "--":
                # Apply color to this quarter
                result.append(color_code)
            else:
                result.append("--")
        return "".join(result)

    def _transform_shape(self, input_shapes: List[str], building_type: BuildingType, output_index: int) -> str:
        """
        Transform shapes through a machine.

        Args:
            input_shapes: List of input shapes (1 or 2 depending on machine)
            building_type: The type of machine
            output_index: Which output port (for multi-output machines)

        Shape format: 8 chars "TLTRBLBR" where each pair is a quarter (or "--" for empty)
        Example: "CuCuCuCu" = full copper, "Cu--Cu--" = left half copper
        """
        if not input_shapes or not input_shapes[0]:
            return None

        input_shape = self._normalize_shape(input_shapes[0])
        input_shape2 = self._normalize_shape(input_shapes[1]) if len(input_shapes) > 1 else "--------"

        if building_type in (BuildingType.CUTTER, BuildingType.CUTTER_MIRRORED):
            # Cutter: splits into left half (TL, BL) and right half (TR, BR)
            tl, tr, bl, br = input_shape[0:2], input_shape[2:4], input_shape[4:6], input_shape[6:8]
            if output_index == 0:  # Left half
                return f"{tl}--{bl}--"
            else:  # Right half
                return f"--{tr}--{br}"

        elif building_type == BuildingType.HALF_CUTTER:
            # Half cutter: outputs left half, destroys right
            tl, bl = input_shape[0:2], input_shape[4:6]
            return f"{tl}--{bl}--"

        elif building_type == BuildingType.ROTATOR_CW:
            # Rotate clockwise: BL->TL, TL->TR, TR->BR, BR->BL
            tl, tr, bl, br = input_shape[0:2], input_shape[2:4], input_shape[4:6], input_shape[6:8]
            return f"{bl}{tl}{br}{tr}"

        elif building_type == BuildingType.ROTATOR_CCW:
            # Rotate counter-clockwise: TR->TL, BR->TR, BL->BR, TL->BL
            tl, tr, bl, br = input_shape[0:2], input_shape[2:4], input_shape[4:6], input_shape[6:8]
            return f"{tr}{br}{tl}{bl}"

        elif building_type == BuildingType.ROTATOR_180:
            # Rotate 180: swap TL<->BR and TR<->BL
            tl, tr, bl, br = input_shape[0:2], input_shape[2:4], input_shape[4:6], input_shape[6:8]
            return f"{br}{bl}{tr}{tl}"

        elif building_type in (BuildingType.STACKER, BuildingType.STACKER_BENT, BuildingType.STACKER_BENT_MIRRORED):
            # Stacker: combine two inputs (input 0 = bottom, input 1 = top)
            return self._stack_shapes(input_shape, input_shape2)

        elif building_type == BuildingType.UNSTACKER:
            # Unstacker: split into layers
            return self._unstack_shape(input_shape, output_index)

        elif building_type == BuildingType.SWAPPER:
            # Swapper: swap right halves between two inputs
            return self._swap_quadrants(input_shape, input_shape2, output_index)

        elif building_type in (BuildingType.PAINTER, BuildingType.PAINTER_MIRRORED):
            # Painter: apply color from input 1 to shape from input 0
            # Color input is treated as the color code
            return self._paint_shape(input_shape, input_shape2[:2] if input_shape2 else "Cu")

        elif building_type in (BuildingType.SPLITTER, BuildingType.SPLITTER_LEFT, BuildingType.SPLITTER_RIGHT):
            # Splitter: same shape to both outputs (throughput split handled separately)
            return input_shape

        elif building_type == BuildingType.MERGER:
            # Merger: same shape passes through (assumes compatible shapes)
            return input_shape

        # Default: pass through
        return input_shape
    
    def _get_adjacent_pos_for_direction(self, pos: Tuple[int, int, int], direction: str) -> Tuple[int, int, int]:
        """Get the adjacent position in the given direction."""
        x, y, z = pos
        deltas = {'E': (1, 0), 'W': (-1, 0), 'N': (0, -1), 'S': (0, 1)}
        dx, dy = deltas[direction]
        return (x + dx, y + dy, z)

    def _opposite_direction(self, direction: str) -> str:
        """Get the opposite direction."""
        return {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}[direction]

    def _find_belt_branch_outputs(self, pos: Tuple[int, int, int], primary_direction: Rotation) -> List[Tuple[Tuple[int, int, int], Rotation, bool]]:
        """
        Find all valid output positions for a belt, supporting branching.

        A belt can branch to multiple destinations as long as:
        1. The adjacent position accepts input from this belt's direction
        2. The output directions are not 180° apart

        Returns list of (position, direction_from_source, is_machine_input) tuples.
        """
        x, y, z = pos
        valid_outputs = []

        # Check all 4 adjacent positions
        adjacent = [
            ((x + 1, y, z), Rotation.EAST, 'E'),
            ((x - 1, y, z), Rotation.WEST, 'W'),
            ((x, y + 1, z), Rotation.SOUTH, 'S'),
            ((x, y - 1, z), Rotation.NORTH, 'N'),
        ]

        for adj_pos, direction_to_adj, dir_char in adjacent:
            adj_cell = self.cells.get(adj_pos)

            # Direction from adj back to us (opposite)
            dir_from_adj_to_us = self._opposite_direction(dir_char)

            # Check if adjacent position is a machine cell with an input facing us
            # A port with direction 'W' means it receives from the west (our position if we're west of it)
            is_machine_input = False
            for origin, machine in self.machines.items():
                for port in machine.input_ports:
                    # Port is ON the machine cell at port['position']
                    # Port direction tells us which edge receives input
                    # If port direction is 'W', it receives from the cell to its west
                    if port['position'] == adj_pos and port.get('direction') == dir_from_adj_to_us:
                        # Machine at adj_pos has an input that receives from our direction
                        valid_outputs.append((adj_pos, direction_to_adj, True))
                        is_machine_input = True
                        break
                if is_machine_input:
                    break

            if is_machine_input:
                continue  # Already added as machine input

            # Check for edge outputs at adjacent position
            for out in self.outputs:
                if out['position'] == adj_pos:
                    valid_outputs.append((adj_pos, direction_to_adj, False))
                    break
            else:
                # Not an edge output, check for belt connection
                if not adj_cell or not adj_cell.building_type:
                    continue

                # Check if adjacent cell is a belt-type building
                if adj_cell.building_type not in (BuildingType.BELT_FORWARD, BuildingType.BELT_LEFT,
                                                   BuildingType.BELT_RIGHT, BuildingType.BELT_PORT_SENDER,
                                                   BuildingType.BELT_PORT_RECEIVER, BuildingType.LIFT_UP,
                                                   BuildingType.LIFT_DOWN):
                    # Could be a machine body cell - skip
                    continue

                # Check if the adjacent belt accepts input from our direction
                adj_input_dir = get_belt_input_direction(adj_cell.building_type, adj_cell.rotation)

                # From adjacent's perspective, we are in the opposite direction
                our_direction_from_adj = {
                    Rotation.EAST: Rotation.WEST,
                    Rotation.WEST: Rotation.EAST,
                    Rotation.NORTH: Rotation.SOUTH,
                    Rotation.SOUTH: Rotation.NORTH,
                }[direction_to_adj]

                # Adjacent belt accepts from us if its input direction matches where we are
                if adj_input_dir == our_direction_from_adj:
                    valid_outputs.append((adj_pos, direction_to_adj, False))

        # If we have multiple outputs, check they're not 180° apart
        if len(valid_outputs) > 1:
            directions = [d for _, d, _ in valid_outputs]
            has_conflict = False
            for i in range(len(directions)):
                for j in range(i + 1, len(directions)):
                    if are_directions_opposite(directions[i], directions[j]):
                        has_conflict = True
                        break
                if has_conflict:
                    break

            if has_conflict:
                # Only use primary direction output
                valid_outputs = [(p, d, m) for p, d, m in valid_outputs if d == primary_direction]

        return valid_outputs

    def _find_belt_port_receiver(self, sender_pos: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        """Find the receiver position for a belt port sender."""
        # Belt ports can jump up to 4 cells in their facing direction
        cell = self.cells.get(sender_pos)
        if not cell or cell.building_type != BuildingType.BELT_PORT_SENDER:
            return None

        dx, dy = direction_delta(cell.rotation)
        for dist in range(1, 5):  # Belt ports can reach 1-4 cells
            check_pos = (sender_pos[0] + dx * dist, sender_pos[1] + dy * dist, sender_pos[2])
            check_cell = self.cells.get(check_pos)
            if check_cell and check_cell.building_type == BuildingType.BELT_PORT_RECEIVER:
                return check_pos
        return None

    def _trace_from_position(self, start: Tuple[int, int, int], shape: str, throughput: float,
                              visited: Set, path: List[Tuple[int, int, int]] = None):
        """
        Trace flow from a position through belts to destinations.

        Args:
            start: Starting position
            shape: Shape code being transported
            throughput: Items per minute
            visited: Set of already-visited positions (prevents infinite loops)
            path: List to collect the path for visualization
        """
        if start in visited:
            return
        visited.add(start)

        if path is not None:
            path.append(start)

        cell = self.cells.get(start)
        if not cell or not cell.building_type:
            return

        # Update this cell's flow
        cell.shape = shape
        cell.throughput += throughput

        # Handle different building types
        if cell.building_type in (BuildingType.BELT_FORWARD, BuildingType.BELT_LEFT,
                                   BuildingType.BELT_RIGHT):
            # Get primary output direction based on belt type
            if cell.building_type == BuildingType.BELT_FORWARD:
                primary_dir = cell.rotation
            elif cell.building_type == BuildingType.BELT_LEFT:
                # Left turn: EAST->NORTH, SOUTH->EAST, WEST->SOUTH, NORTH->WEST
                left_turn = {
                    Rotation.EAST: Rotation.NORTH,
                    Rotation.SOUTH: Rotation.EAST,
                    Rotation.WEST: Rotation.SOUTH,
                    Rotation.NORTH: Rotation.WEST,
                }
                primary_dir = left_turn[cell.rotation]
            else:  # BELT_RIGHT
                # Right turn: EAST->SOUTH, SOUTH->WEST, WEST->NORTH, NORTH->EAST
                right_turn = {
                    Rotation.EAST: Rotation.SOUTH,
                    Rotation.SOUTH: Rotation.WEST,
                    Rotation.WEST: Rotation.NORTH,
                    Rotation.NORTH: Rotation.EAST,
                }
                primary_dir = right_turn[cell.rotation]

            # Find all valid branch outputs
            branch_outputs = self._find_belt_branch_outputs(start, primary_dir)

            if branch_outputs:
                # Split throughput among branches
                branch_throughput = throughput / len(branch_outputs)

                for branch_pos, branch_dir, is_machine_input in branch_outputs:
                    handled = False

                    # Handle machine input ports directly
                    if is_machine_input:
                        for origin, machine in self.machines.items():
                            for port in machine.input_ports:
                                if port['position'] == branch_pos:
                                    # Update belt at branch position if there is one
                                    branch_cell = self.cells.get(branch_pos)
                                    if branch_cell and branch_cell.building_type:
                                        branch_cell.shape = shape
                                        branch_cell.throughput += branch_throughput
                                    port['shape'] = shape
                                    port['throughput'] += branch_throughput
                                    port['connected'] = True
                                    handled = True
                                    break
                            if handled:
                                break
                        continue

                    # Check for edge outputs at branch position
                    for out in self.outputs:
                        if out['position'] == branch_pos:
                            branch_cell = self.cells.get(branch_pos)
                            if branch_cell and branch_cell.building_type:
                                branch_cell.shape = shape
                                branch_cell.throughput += branch_throughput
                            out['actual_shape'] = shape
                            out['throughput'] += branch_throughput
                            handled = True
                            break

                    # Continue tracing if it's a belt
                    if not handled and branch_pos in self.cells:
                        branch_path = [] if path is not None else None
                        self._trace_from_position(branch_pos, shape, branch_throughput, visited, branch_path)
                        if path is not None and branch_path:
                            path.extend(branch_path)

                return  # Already handled all branches

            # No valid branches found, use primary direction
            dx, dy = direction_delta(primary_dir)
            next_pos = (start[0] + dx, start[1] + dy, start[2])

        elif cell.building_type == BuildingType.BELT_PORT_SENDER:
            # Belt port - find receiver
            receiver_pos = self._find_belt_port_receiver(start)
            if receiver_pos:
                # Mark receiver cell with flow too
                recv_cell = self.cells.get(receiver_pos)
                if recv_cell:
                    recv_cell.shape = shape
                    recv_cell.throughput += throughput
                    if path is not None:
                        path.append(receiver_pos)
                # Continue from after the receiver
                dx, dy = direction_delta(recv_cell.rotation if recv_cell else cell.rotation)
                next_pos = (receiver_pos[0] + dx, receiver_pos[1] + dy, receiver_pos[2])
            else:
                self.warnings.append(f"Belt port sender at {start} has no receiver")
                return

        elif cell.building_type == BuildingType.BELT_PORT_RECEIVER:
            # Flow continues in facing direction
            dx, dy = direction_delta(cell.rotation)
            next_pos = (start[0] + dx, start[1] + dy, start[2])

        elif cell.building_type == BuildingType.LIFT_UP:
            # Move to next floor up
            next_pos = (start[0], start[1], start[2] + 1)

        elif cell.building_type == BuildingType.LIFT_DOWN:
            # Move to floor below
            next_pos = (start[0], start[1], start[2] - 1)

        else:
            # Non-belt building - stop tracing
            return

        # Check if next position is a machine input
        # Machine inputs are ON the machine cell - belt must be adjacent in the input direction
        for origin, machine in self.machines.items():
            for port in machine.input_ports:
                port_pos = port['position']
                port_dir = port.get('direction', 'W')
                # The belt position that can feed this port is adjacent in the input direction
                # e.g., if port direction is 'W', the belt must be to the west of the port
                expected_belt_pos = self._get_adjacent_pos_for_direction(port_pos, port_dir)

                if start == expected_belt_pos:
                    # This belt is in the right position to feed this machine input
                    port['shape'] = shape
                    port['throughput'] += throughput
                    port['connected'] = True
                    return

        # Check if next is an edge output, or if belt is at output position
        for out in self.outputs:
            if out['position'] == next_pos or out['position'] == start:
                out['actual_shape'] = shape
                out['throughput'] += throughput
                return

        # Continue tracing if there's a belt at next position
        if next_pos in self.cells:
            self._trace_from_position(next_pos, shape, throughput, visited, path)
    
    def _is_splitter(self, bt: BuildingType) -> bool:
        """Check if building type is a splitter."""
        return bt in (BuildingType.SPLITTER, BuildingType.SPLITTER_LEFT,
                      BuildingType.SPLITTER_RIGHT)

    def _is_merger(self, bt: BuildingType) -> bool:
        """Check if building type is a merger."""
        return bt == BuildingType.MERGER

    def _count_connected_outputs(self, machine: MachineState) -> int:
        """Count how many output ports have connected destinations."""
        count = 0
        for port in machine.output_ports:
            out_pos = port['position']
            out_dir = port.get('direction', 'E')
            # Belt should be adjacent to machine in the output direction
            belt_pos = self._get_adjacent_pos_for_direction(out_pos, out_dir)

            # Check for belt at expected position
            if belt_pos in self.cells and self.cells[belt_pos].building_type:
                count += 1
            else:
                # Check for direct machine-to-machine connection
                for other_origin, other_m in self.machines.items():
                    if other_origin == machine.origin:
                        continue
                    for other_port in other_m.input_ports:
                        other_port_pos = other_port['position']
                        other_port_dir = other_port.get('direction', 'W')
                        expected_feed_pos = self._get_adjacent_pos_for_direction(other_port_pos, other_port_dir)
                        if expected_feed_pos == out_pos and self._opposite_direction(out_dir) == other_port_dir:
                            count += 1
                            break
                # Check for edge output at belt position
                for out in self.outputs:
                    if out['position'] == belt_pos:
                        count += 1
        return count

    def simulate(self) -> 'FlowReport':
        """Run the flow simulation."""
        self.errors.clear()
        self.warnings.clear()

        # Store traced paths for visualization
        self.traced_paths: List[List[Tuple[int, int, int]]] = []

        # Reset flow state
        for cell in self.cells.values():
            cell.shape = None
            cell.throughput = 0.0
            cell.source = None
            cell.destinations.clear()

        for machine in self.machines.values():
            machine.throughput = 0.0
            for port in machine.input_ports:
                port['shape'] = None
                port['throughput'] = 0.0
                port['connected'] = False
            for port in machine.output_ports:
                port['shape'] = None
                port['throughput'] = 0.0
                port['connected'] = False
                port['backed_up'] = False

        for out in self.outputs:
            out['actual_shape'] = None
            out['throughput'] = 0.0

        # Phase 1: Trace from inputs through belts to first machines
        # Use per-input visited sets to allow merging (multiple inputs into one belt)
        # But track globally visited to prevent infinite loops within same trace
        for inp in self.inputs:
            visited = set()  # Fresh visited set for each input to allow merging
            pos = inp['position']
            shape = inp['shape']
            throughput = inp['throughput']

            path = []

            # Check if input position has a belt
            if pos in self.cells:
                cell = self.cells[pos]
                cell.shape = shape
                cell.throughput += throughput  # ADD to existing throughput for merging
                visited.add(pos)  # Mark as visited so trace doesn't double-count
                path.append(pos)

                # Find next position based on belt type and trace from there
                if cell.building_type in (BuildingType.BELT_FORWARD, BuildingType.BELT_LEFT,
                                          BuildingType.BELT_RIGHT):
                    # Get primary output direction
                    if cell.building_type == BuildingType.BELT_FORWARD:
                        primary_dir = cell.rotation
                    elif cell.building_type == BuildingType.BELT_LEFT:
                        left_turn = {Rotation.EAST: Rotation.NORTH, Rotation.SOUTH: Rotation.EAST,
                                     Rotation.WEST: Rotation.SOUTH, Rotation.NORTH: Rotation.WEST}
                        primary_dir = left_turn[cell.rotation]
                    else:
                        right_turn = {Rotation.EAST: Rotation.SOUTH, Rotation.SOUTH: Rotation.WEST,
                                      Rotation.WEST: Rotation.NORTH, Rotation.NORTH: Rotation.EAST}
                        primary_dir = right_turn[cell.rotation]

                    # Find branch outputs and trace from each
                    branch_outputs = self._find_belt_branch_outputs(pos, primary_dir)
                    if branch_outputs:
                        branch_throughput = throughput / len(branch_outputs)
                        for branch_pos, _, is_machine_input in branch_outputs:
                            if is_machine_input:
                                # Directly feed machine input port
                                for origin, machine in self.machines.items():
                                    for port in machine.input_ports:
                                        if port['position'] == branch_pos:
                                            port['shape'] = shape
                                            port['throughput'] += branch_throughput
                                            port['connected'] = True
                                            break
                            else:
                                self._trace_from_position(branch_pos, shape, branch_throughput, visited, path)
                    else:
                        # No branches, trace in primary direction
                        dx, dy = direction_delta(primary_dir)
                        next_pos = (pos[0] + dx, pos[1] + dy, pos[2])
                        self._trace_from_position(next_pos, shape, throughput, visited, path)
                else:
                    # Non-standard belt at input, just trace forward
                    self._trace_from_position(pos, shape, throughput, visited, path)
            else:
                # Check if directly feeding a machine input
                for origin, machine in self.machines.items():
                    for port in machine.input_ports:
                        if port['position'] == pos:
                            port['shape'] = shape
                            port['throughput'] = throughput
                            port['connected'] = True

            if path:
                self.traced_paths.append(path)

        # Phase 2: Process machines iteratively (some machines feed others)
        # Process in multiple passes until no more changes
        max_iterations = 10
        for iteration in range(max_iterations):
            changes = False

            for origin, machine in self.machines.items():
                # Skip if already processed with non-zero throughput
                if machine.throughput > 0:
                    continue

                # Collect input shapes and throughputs
                input_shapes = []
                input_throughputs = []
                for port in machine.input_ports:
                    if port['throughput'] > 0:
                        input_shapes.append(port['shape'])
                        input_throughputs.append(port['throughput'])
                    else:
                        input_shapes.append(None)
                        input_throughputs.append(0)

                # Check if we have enough inputs
                active_inputs = [t for t in input_throughputs if t > 0]
                if not active_inputs:
                    continue

                # For multi-input machines, need all inputs
                min_required_inputs = 1
                if machine.building_type in (BuildingType.STACKER, BuildingType.STACKER_BENT,
                                               BuildingType.STACKER_BENT_MIRRORED, BuildingType.SWAPPER):
                    min_required_inputs = 2

                if len(active_inputs) < min_required_inputs:
                    # Not enough inputs yet - wait for more
                    continue

                changes = True

                # Calculate effective throughput
                if self._is_merger(machine.building_type):
                    # Merger: combines all inputs, output throughput = sum of inputs (capped)
                    total_input = sum(active_inputs)
                    effective_tp = min(total_input, machine.max_throughput)
                else:
                    # Other machines: limited by slowest input and capacity
                    effective_tp = min(min(active_inputs), machine.max_throughput)

                machine.throughput = effective_tp

                # Get shapes for transformation (filter out None)
                valid_shapes = [s for s in input_shapes if s]

                # Calculate output shapes and throughput
                num_outputs = len(machine.output_ports)
                connected_outputs = self._count_connected_outputs(machine)

                for port in machine.output_ports:
                    port['shape'] = self._transform_shape(valid_shapes, machine.building_type, port['index'])

                    # Handle throughput for splitters (split between outputs)
                    if self._is_splitter(machine.building_type) and connected_outputs > 0:
                        # Splitter divides throughput among connected outputs
                        port['throughput'] = effective_tp / connected_outputs
                    else:
                        port['throughput'] = effective_tp

                    # Check if output is connected and trace from there
                    # Output port is ON the machine cell, direction tells where output goes
                    out_pos = port['position']
                    out_dir = port.get('direction', 'E')
                    # Belt should be adjacent to machine in the output direction
                    belt_pos = self._get_adjacent_pos_for_direction(out_pos, out_dir)

                    # Look for belt at the expected position
                    if belt_pos in self.cells and self.cells[belt_pos].building_type:
                        port['connected'] = True
                        # Trace from the belt
                        path = []
                        self._trace_from_position(belt_pos, port['shape'], port['throughput'], visited, path)
                        if path:
                            self.traced_paths.append(path)
                    else:
                        # Check if output goes to another machine input (direct connection)
                        found_dest = False
                        for other_origin, other_machine in self.machines.items():
                            if other_origin == origin:
                                continue
                            for other_port in other_machine.input_ports:
                                other_port_pos = other_port['position']
                                other_port_dir = other_port.get('direction', 'W')
                                # Check if this machine's output feeds directly to other machine's input
                                # Output at out_pos going out_dir should match input expecting from opposite direction
                                expected_feed_pos = self._get_adjacent_pos_for_direction(other_port_pos, other_port_dir)
                                if expected_feed_pos == out_pos and self._opposite_direction(out_dir) == other_port_dir:
                                    port['connected'] = True
                                    other_port['shape'] = port['shape']
                                    other_port['throughput'] += port['throughput']
                                    other_port['connected'] = True
                                    found_dest = True
                                    break

                        # Check for edge outputs at belt position
                        for out in self.outputs:
                            if out['position'] == belt_pos:
                                port['connected'] = True
                                out['actual_shape'] = port['shape']
                                out['throughput'] += port['throughput']
                                found_dest = True

                        if not found_dest:
                            port['backed_up'] = True
                            self.errors.append(
                                f"{machine.building_type.name} @ {origin}: "
                                f"output[{port['index']}] at {out_pos} dir={out_dir} has no destination "
                                f"(belt expected at {belt_pos}, shape={port['shape']}, {port['throughput']:.0f}/min will back up!)"
                            )

            if not changes:
                break

        return FlowReport(self)
    
    def print_grid(self, floor: int = 0, show_flow: bool = True):
        """Print ASCII grid with optional flow info."""
        symbols = {
            BuildingType.BELT_FORWARD: {
                Rotation.EAST: '→', Rotation.WEST: '←',
                Rotation.SOUTH: '↓', Rotation.NORTH: '↑'
            },
            BuildingType.BELT_LEFT: 'L',
            BuildingType.BELT_RIGHT: 'R',
            BuildingType.CUTTER: 'C',
            BuildingType.CUTTER_MIRRORED: 'c',
            BuildingType.HALF_CUTTER: 'H',
            BuildingType.ROTATOR_CW: '↻',
            BuildingType.ROTATOR_CCW: '↺',
            BuildingType.SPLITTER: 'Y',
            BuildingType.MERGER: 'M',
            BuildingType.STACKER: 'S',
            BuildingType.BELT_PORT_SENDER: '⊳',
            BuildingType.BELT_PORT_RECEIVER: '⊲',
        }
        
        print(f"\n{'='*60}")
        print(f"FLOOR {floor} - Grid View")
        print(f"{'='*60}")
        print("    " + "".join(f"{x%10}" for x in range(self.width)))
        print("    " + "-" * self.width)
        
        for y in range(self.height):
            row = f"{y:3}|"
            for x in range(self.width):
                pos = (x, y, floor)
                
                is_input = any(i['position'] == pos for i in self.inputs)
                is_output = any(o['position'] == pos for o in self.outputs)
                
                if pos in self.cells:
                    cell = self.cells[pos]
                    if cell.building_type:
                        sym = symbols.get(cell.building_type)
                        if isinstance(sym, dict):
                            row += sym.get(cell.rotation, '?')
                        elif sym:
                            # Check if this is origin or extended cell
                            for orig, bld in self.buildings.items():
                                if pos in bld['cells'] and pos != orig:
                                    row += '#'
                                    break
                            else:
                                row += sym
                        else:
                            row += '?'
                    elif is_input:
                        row += 'I'
                    elif is_output:
                        row += 'O'
                    else:
                        row += '.'
                elif is_input:
                    row += 'I'
                elif is_output:
                    row += 'O'
                else:
                    row += '.'
            row += "|"
            print(row)
        
        print("    " + "-" * self.width)
        
        if show_flow:
            print(f"\n{'='*60}")
            print("FLOW DETAILS")
            print(f"{'='*60}")
            
            # Show belt flows
            belt_flows = []
            for pos, cell in self.cells.items():
                if pos[2] != floor:
                    continue
                if cell.building_type in (BuildingType.BELT_FORWARD, BuildingType.BELT_LEFT,
                                          BuildingType.BELT_RIGHT) and cell.throughput > 0:
                    belt_flows.append((pos, cell))
            
            if belt_flows:
                print("\nBelt Utilization:")
                for pos, cell in sorted(belt_flows):
                    util_bar = "█" * int(cell.utilization / 10) + "░" * (10 - int(cell.utilization / 10))
                    print(f"  ({pos[0]:2},{pos[1]:2}): {cell.shape or '---':8} "
                          f"{cell.throughput:5.0f}/{cell.max_throughput:.0f}/min "
                          f"[{util_bar}] {cell.utilization:.0f}%")


@dataclass
class FlowReport:
    """Detailed flow simulation report."""
    sim: FlowSimulator
    
    def __str__(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("FLOW SIMULATION REPORT")
        lines.append("=" * 70)
        
        # Errors
        if self.sim.errors:
            lines.append("\n🔴 CRITICAL ERRORS (will cause backup):")
            for err in self.sim.errors:
                lines.append(f"   ❌ {err}")
        
        # Warnings
        if self.sim.warnings:
            lines.append("\n🟡 WARNINGS:")
            for warn in self.sim.warnings:
                lines.append(f"   ⚠ {warn}")
        
        # Inputs
        lines.append("\n" + "─" * 70)
        lines.append("📥 INPUTS")
        lines.append("─" * 70)
        for i, inp in enumerate(self.sim.inputs):
            lines.append(f"   [{i}] Position: {inp['position']}")
            lines.append(f"       Shape:    {inp['shape']}")
            lines.append(f"       Rate:     {inp['throughput']:.0f} items/min")
        
        # Machines with full flow details
        lines.append("\n" + "─" * 70)
        lines.append("🏭 MACHINES (with shape transformations)")
        lines.append("─" * 70)
        
        for origin, machine in self.sim.machines.items():
            util_bar = "█" * int(machine.utilization / 10) + "░" * (10 - int(machine.utilization / 10))
            lines.append(f"\n   {machine.building_type.name} @ {origin}")
            lines.append(f"   Utilization: [{util_bar}] {machine.utilization:.0f}% "
                        f"({machine.throughput:.0f}/{machine.max_throughput:.0f} items/min)")
            
            lines.append("   ┌─ INPUTS:")
            for port in machine.input_ports:
                status = "✅" if port['connected'] and port['throughput'] > 0 else "❌ STARVED"
                lines.append(f"   │  [{port['index']}] @ {port['position']}: "
                           f"{port['shape'] or '(empty)':10} @ {port['throughput']:5.0f}/min {status}")
            
            lines.append("   │")
            lines.append(f"   │  ══► {machine.building_type.name} PROCESSING ══►")
            lines.append("   │")
            
            lines.append("   └─ OUTPUTS:")
            for port in machine.output_ports:
                if port['backed_up']:
                    status = "🔴 BACKED UP - NO DESTINATION!"
                elif port['connected']:
                    status = "✅ connected"
                else:
                    status = "⚪ idle"
                lines.append(f"      [{port['index']}] @ {port['position']}: "
                           f"{port['shape'] or '(empty)':10} @ {port['throughput']:5.0f}/min {status}")
        
        # Outputs
        lines.append("\n" + "─" * 70)
        lines.append("📤 OUTPUTS")
        lines.append("─" * 70)
        
        for i, out in enumerate(self.sim.outputs):
            expected = out.get('expected_shape', 'any')
            actual = out.get('actual_shape') or '(nothing)'
            throughput = out.get('throughput', 0)
            
            if actual == '(nothing)':
                status = "❌ NO FLOW"
            elif expected == 'any' or actual == expected:
                status = "✅ CORRECT"
            else:
                status = f"❌ WRONG (want {expected})"
            
            lines.append(f"   [{i}] Position: {out['position']}")
            lines.append(f"       Shape:    {actual} {status}")
            lines.append(f"       Rate:     {throughput:.0f} items/min")
        
        # Summary
        lines.append("\n" + "=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        
        total_in = sum(i['throughput'] for i in self.sim.inputs)
        total_out = sum(o.get('throughput', 0) for o in self.sim.outputs)
        efficiency = 100 * total_out / total_in if total_in > 0 else 0
        
        backed_up = sum(1 for m in self.sim.machines.values() 
                       for p in m.output_ports if p['backed_up'])
        
        lines.append(f"   Total Input:     {total_in:.0f} items/min")
        lines.append(f"   Total Output:    {total_out:.0f} items/min")
        lines.append(f"   Efficiency:      {efficiency:.1f}%")
        lines.append(f"   Backed Up Ports: {backed_up}")
        lines.append(f"   Errors:          {len(self.sim.errors)}")
        
        if self.sim.errors or backed_up > 0:
            lines.append("\n   ❌ LAYOUT INVALID - Items will back up!")
        elif total_out == 0:
            lines.append("\n   ⚠ NO OUTPUT - Check connections!")
        else:
            lines.append("\n   ✅ LAYOUT VALID - Flow looks good!")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def is_valid(self) -> bool:
        if self.sim.errors:
            return False
        for m in self.sim.machines.values():
            for p in m.output_ports:
                if p['backed_up']:
                    return False
        return True


def demo():
    """Demo showing flow simulation with shape tracking."""
    print("=" * 70)
    print("FLOW SIMULATOR DEMO - Shape Transformation Tracking")
    print("=" * 70)
    
    # Test 1: Cutter with both outputs
    print("\n\n>>> TEST 1: Cutter with BOTH outputs connected")
    print("    Input: CuCuCuCu (full copper square)")
    print("    Expected: Left half -> Output 0, Right half -> Output 1")
    
    sim = FlowSimulator(14, 14, 4)
    sim.place_building(BuildingType.CUTTER, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.EAST)
    
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0, "Cu--Cu--")
    sim.set_output(6, 6, 0, "--Cu--Cu")
    
    report = sim.simulate()
    sim.print_grid(0)
    print(report)
    
    # Test 2: Missing output connection
    print("\n\n>>> TEST 2: Cutter with ONE output MISSING")
    print("    This will cause backup!")
    
    sim2 = FlowSimulator(14, 14, 4)
    sim2.place_building(BuildingType.CUTTER, 3, 5, 0, Rotation.EAST)
    sim2.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim2.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    # NOT placing belt at (4,6) - right half output will back up!
    
    sim2.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim2.set_output(5, 5, 0, "Cu--Cu--")
    
    report2 = sim2.simulate()
    sim2.print_grid(0)
    print(report2)
    
    # Test 3: Rotator
    print("\n\n>>> TEST 3: Rotator CW")
    print("    Input: CuCu---- (top half only)")
    print("    Expected: --Cu--Cu (rotated clockwise to right half)")

    sim3 = FlowSimulator(14, 14, 4)
    sim3.place_building(BuildingType.ROTATOR_CW, 3, 5, 0, Rotation.EAST)
    sim3.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim3.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)

    sim3.set_input(2, 5, 0, "CuCu----", 180.0)
    sim3.set_output(5, 5, 0, "--Cu--Cu")

    report3 = sim3.simulate()
    sim3.print_grid(0)
    print(report3)

    # Test 4: Splitter (throughput division)
    print("\n\n>>> TEST 4: Splitter (throughput division)")
    print("    Input: CuCuCuCu at 180/min")
    print("    Expected: 90/min to each output")

    sim4 = FlowSimulator(14, 14, 4)
    sim4.place_building(BuildingType.SPLITTER, 4, 5, 0, Rotation.EAST)
    sim4.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    sim4.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.EAST)  # Output 0
    sim4.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)  # Output 1
    sim4.place_building(BuildingType.BELT_FORWARD, 5, 4, 0, Rotation.EAST)
    sim4.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.EAST)

    sim4.set_input(3, 5, 0, "CuCuCuCu", 180.0)
    sim4.set_output(6, 4, 0)
    sim4.set_output(6, 6, 0)

    report4 = sim4.simulate()
    sim4.print_grid(0)
    print(report4)

    # Test 5: Merger (throughput combining)
    print("\n\n>>> TEST 5: Merger (throughput combining)")
    print("    Two inputs: 90/min each")
    print("    Expected: 180/min output")

    sim5 = FlowSimulator(14, 14, 4)
    sim5.place_building(BuildingType.MERGER, 5, 5, 0, Rotation.EAST)
    sim5.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.SOUTH)  # Into merger
    sim5.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.NORTH)  # Into merger
    sim5.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.EAST)
    sim5.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.EAST)
    sim5.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim5.set_input(3, 4, 0, "CuCuCuCu", 90.0)
    sim5.set_input(3, 6, 0, "CuCuCuCu", 90.0)
    sim5.set_output(7, 5, 0)

    report5 = sim5.simulate()
    sim5.print_grid(0)
    print(report5)

    # Test 6: Belt branching (90° allowed)
    print("\n\n>>> TEST 6: Belt Branching (90° - ALLOWED)")
    print("    One input belt branching to two belts at 90°")
    print("    Expected: 90/min to each branch")

    sim6 = FlowSimulator(14, 14, 4)
    # Input belt going east
    sim6.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Branch belt going east (continues forward)
    sim6.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    # Branch belt going north (takes from south - 90° branch)
    sim6.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.NORTH)
    # Continue east branch
    sim6.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    # Continue north branch
    sim6.place_building(BuildingType.BELT_FORWARD, 4, 3, 0, Rotation.NORTH)

    sim6.set_input(3, 5, 0, "CuCuCuCu", 180.0)
    sim6.set_output(6, 5, 0)  # East output
    sim6.set_output(4, 2, 0)  # North output

    report6 = sim6.simulate()
    sim6.print_grid(0)
    print(report6)

    # Test 7: Belt branching (180° NOT allowed)
    print("\n\n>>> TEST 7: Belt Branching (180° - NOT ALLOWED)")
    print("    One input belt with opposite direction branches")
    print("    Expected: Only primary direction gets flow")

    sim7 = FlowSimulator(14, 14, 4)
    # Input belt going east
    sim7.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Branch belt going east (forward)
    sim7.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    # Branch belt going west (opposite - should be blocked)
    sim7.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.WEST)
    # Continue east
    sim7.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)

    sim7.set_input(3, 5, 0, "CuCuCuCu", 180.0)
    sim7.set_output(6, 5, 0)  # East output
    sim7.set_output(1, 5, 0)  # West output (should get nothing)

    report7 = sim7.simulate()
    sim7.print_grid(0)
    print(report7)

    # Test 8: Belt merging (multiple inputs to one belt)
    print("\n\n>>> TEST 8: Belt Merging")
    print("    Two input belts merging into one")
    print("    Expected: Combined throughput on merged belt")

    sim8 = FlowSimulator(14, 14, 4)
    # Two input belts
    sim8.place_building(BuildingType.BELT_FORWARD, 2, 4, 0, Rotation.SOUTH)  # Goes south to merge point
    sim8.place_building(BuildingType.BELT_FORWARD, 2, 6, 0, Rotation.NORTH)  # Goes north to merge point
    # Merge point
    sim8.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)  # Receives from both, outputs east
    # Continue east
    sim8.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    sim8.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)

    sim8.set_input(2, 4, 0, "CuCuCuCu", 90.0)  # 90 from north
    sim8.set_input(2, 6, 0, "CuCuCuCu", 90.0)  # 90 from south
    sim8.set_output(5, 5, 0)  # Should receive 180 combined

    report8 = sim8.simulate()
    sim8.print_grid(0)
    print(report8)


if __name__ == "__main__":
    demo()
