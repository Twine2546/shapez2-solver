"""
Pygame-based Flow Simulator Viewer

Interactive interface to:
- Place buildings on a grid
- Set inputs/outputs
- Run simulation and see flow at each step
- Switch between floors
"""

import pygame
import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, List, Optional, Tuple
from shapez2_solver.simulation.flow_simulator import FlowSimulator, MACHINE_THROUGHPUT, BELT_THROUGHPUT
from shapez2_solver.blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS, BUILDING_PORTS
from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS
from shapez2_solver.simulation.test_scenarios import ALL_SCENARIOS, get_scenario_count, get_scenario


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (80, 80, 255)
YELLOW = (255, 255, 80)
ORANGE = (255, 165, 0)
CYAN = (80, 255, 255)
PURPLE = (200, 80, 200)

# Building colors
BUILDING_COLORS = {
    BuildingType.BELT_FORWARD: (100, 100, 100),
    BuildingType.BELT_LEFT: (100, 100, 120),
    BuildingType.BELT_RIGHT: (100, 100, 120),
    BuildingType.CUTTER: (200, 100, 100),
    BuildingType.CUTTER_MIRRORED: (200, 100, 100),
    BuildingType.HALF_CUTTER: (180, 80, 80),
    BuildingType.ROTATOR_CW: (100, 200, 100),
    BuildingType.ROTATOR_CCW: (100, 200, 100),
    BuildingType.ROTATOR_180: (80, 180, 80),
    BuildingType.STACKER: (100, 100, 200),
    BuildingType.STACKER_BENT: (100, 100, 200),
    BuildingType.STACKER_BENT_MIRRORED: (100, 100, 200),
    BuildingType.UNSTACKER: (80, 80, 180),
    BuildingType.SWAPPER: (180, 180, 100),
    BuildingType.SPLITTER: (200, 200, 100),
    BuildingType.SPLITTER_LEFT: (200, 200, 100),
    BuildingType.SPLITTER_RIGHT: (200, 200, 100),
    BuildingType.MERGER: (200, 100, 200),
    BuildingType.PAINTER: (100, 180, 180),
    BuildingType.PAINTER_MIRRORED: (100, 180, 180),
    BuildingType.BELT_PORT_SENDER: (150, 100, 200),
    BuildingType.BELT_PORT_RECEIVER: (200, 100, 150),
    BuildingType.LIFT_UP: (150, 150, 200),
    BuildingType.LIFT_DOWN: (200, 150, 150),
}

# Shape colors for visualization
SHAPE_COLORS = {
    "Cu": (200, 150, 100),   # Copper
    "Ru": (200, 80, 80),     # Ruby/Red
    "Gr": (80, 200, 80),     # Green
    "Bl": (80, 80, 200),     # Blue
    "Cy": (80, 200, 200),    # Cyan
    "Pu": (200, 80, 200),    # Purple
    "Ye": (200, 200, 80),    # Yellow
    "Wh": (240, 240, 240),   # White
    "--": (40, 40, 40),      # Empty
}


class FlowViewer:
    def __init__(self, width: int = 14, height: int = 14, num_floors: int = 4):
        pygame.init()

        self.grid_width = width
        self.grid_height = height
        self.num_floors = num_floors

        self.cell_size = 40
        self.sidebar_width = 350
        self.toolbar_height = 60
        self.grid_offset_x = 30  # Left margin for west edge I/O
        self.grid_sidebar_gap = 20  # Gap between grid and sidebar

        # Fixed window size - grid scales to fit
        self.screen_width = 1200
        self.screen_height = 900

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flow Simulator - Interactive")

        # Calculate initial cell size to fit grid
        available_width = self.screen_width - self.sidebar_width - 20
        available_height = self.screen_height - self.toolbar_height - 240
        cell_size_for_width = available_width // self.grid_width
        cell_size_for_height = available_height // self.grid_height
        self.cell_size = max(8, min(cell_size_for_width, cell_size_for_height, 40))

        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)

        # State
        self.current_floor = 0
        self.selected_building = BuildingType.BELT_FORWARD
        self.current_rotation = Rotation.EAST
        self.mode = "place"  # "place", "input", "output", "delete"
        self.input_shape = "CuCuCuCu"

        # Mouse state for preview and hover
        self.mouse_pos = (0, 0)
        self.hovered_palette_idx = -1

        # Foundation options - use actual FOUNDATION_SPECS (including irregular shapes)
        self.foundation_names = [
            "1x1", "2x1", "1x2", "2x2", "3x2", "3x3",  # Rectangular
            "L", "L4", "T", "S4", "Cross"  # Irregular
        ]
        self.current_foundation_idx = 0
        self.current_foundation_name = "1x1"

        # Simulator
        self.sim = FlowSimulator(width, height, num_floors)
        self.last_report = None

        # Building palette - expanded with all useful building types
        self.building_palette = [
            BuildingType.BELT_FORWARD,
            BuildingType.BELT_LEFT,
            BuildingType.BELT_RIGHT,
            BuildingType.BELT_PORT_SENDER,
            BuildingType.BELT_PORT_RECEIVER,
            BuildingType.LIFT_UP,
            BuildingType.LIFT_DOWN,
            BuildingType.CUTTER,
            BuildingType.CUTTER_MIRRORED,
            BuildingType.HALF_CUTTER,
            BuildingType.ROTATOR_CW,
            BuildingType.ROTATOR_CCW,
            BuildingType.ROTATOR_180,
            BuildingType.STACKER,
            BuildingType.STACKER_BENT,
            BuildingType.STACKER_BENT_MIRRORED,
            BuildingType.UNSTACKER,
            BuildingType.SWAPPER,
            BuildingType.SPLITTER,
            BuildingType.SPLITTER_LEFT,
            BuildingType.SPLITTER_RIGHT,
            BuildingType.MERGER,
            BuildingType.PAINTER,
            BuildingType.PAINTER_MIRRORED,
        ]

        # Palette scroll offset (for when list is too long)
        self.palette_scroll = 0
        self.palette_visible_count = 12  # Number of items visible at once

        # Toggle for showing shapes on belts
        self.show_shapes = True
        # Toggle for showing port labels
        self.show_ports = True

        # Test scenario navigation
        self.scenario_mode = False
        self.current_scenario_idx = -1
        self.scenario_name = ""
        self.scenario_description = ""
        self.total_scenarios = get_scenario_count()

        # Sample save/load
        self.samples_dir = Path(__file__).parent / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        self.current_sample_slot = 0  # 0-9 slots
        self.sample_message = ""  # Status message for save/load
        self.sample_message_time = 0  # Time when message was set

        self.clock = pygame.time.Clock()
    
    def run(self):
        """Main loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos, event.button)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = event.pos
                    self.update_hover()
                elif event.type == pygame.MOUSEWHEEL:
                    # Scroll building palette
                    if event.y > 0:
                        self.palette_scroll = max(0, self.palette_scroll - 1)
                    else:
                        max_scroll = max(0, len(self.building_palette) - self.palette_visible_count)
                        self.palette_scroll = min(max_scroll, self.palette_scroll + 1)

            self.draw()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    def _can_place_building(self, x: int, y: int, floor: int) -> bool:
        """Check if a building can be placed at this position (no overlap, valid cells)."""
        spec = BUILDING_SPECS.get(self.selected_building)
        if not spec:
            return True  # Unknown building, allow placement

        base_w = spec.width
        base_h = spec.height
        depth = spec.depth

        # Adjust for rotation
        if self.current_rotation in (Rotation.SOUTH, Rotation.NORTH):
            eff_w, eff_h = base_h, base_w
        else:
            eff_w, eff_h = base_w, base_h

        # Get valid cells for irregular foundations
        foundation_spec = FOUNDATION_SPECS[self.current_foundation_name]
        valid_cells = foundation_spec.get_valid_grid_cells()

        # Check all cells the building would occupy
        for dx in range(eff_w):
            for dy in range(eff_h):
                for dz in range(depth):
                    check_pos = (x + dx, y + dy, floor + dz)
                    # Check if position is out of bounds
                    if check_pos[0] >= self.grid_width or check_pos[1] >= self.grid_height:
                        return False
                    if check_pos[2] >= self.num_floors:
                        return False
                    # Check if position is on invalid cell (irregular foundations)
                    if valid_cells is not None and (check_pos[0], check_pos[1]) not in valid_cells:
                        return False
                    # Check if already occupied
                    if check_pos in self.sim.cells:
                        cell = self.sim.cells[check_pos]
                        if cell.building_type is not None:
                            return False
        return True

    def update_hover(self):
        """Update hover state based on mouse position."""
        x, y = self.mouse_pos
        sidebar_x = self.grid_offset_x + self.grid_width * self.cell_size + self.grid_sidebar_gap

        self.hovered_palette_idx = -1

        if x > sidebar_x:
            rel_x = x - sidebar_x
            # Calculate dynamic offset based on foundation rows
            cols = 4
            num_rows = (len(self.foundation_names) + cols - 1) // cols
            foundation_area_end = 18 + num_rows * 26
            palette_y_start = foundation_area_end + 25 + 60 + 25
            rel_y = y - self.toolbar_height - palette_y_start

            if 0 <= rel_x < self.sidebar_width - 20:
                idx = rel_y // 25 + self.palette_scroll
                if 0 <= idx < len(self.building_palette):
                    self.hovered_palette_idx = idx
    
    def handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse click."""
        x, y = pos

        # Check if click is on grid (including external I/O positions)
        # Account for grid offset
        grid_x = (x - self.grid_offset_x) // self.cell_size
        grid_y = (y - self.toolbar_height) // self.cell_size

        # For I/O mode, allow clicking on external edge positions
        # External positions: x=-1 (west), x=grid_width (east), y=-1 (north), y=grid_height (south)
        is_in_grid = 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height
        is_external_io = False

        if self.mode in ("input", "output"):
            # Check for external west edge (x=-1) - click in left margin
            if x < self.grid_offset_x and 0 <= grid_y < self.grid_height:
                grid_x = -1
                is_external_io = True
            # Check for external east edge (x=grid_width)
            elif grid_x == self.grid_width and 0 <= grid_y < self.grid_height:
                is_external_io = True
            # Check for external north edge (y=-1)
            elif y - self.toolbar_height < self.cell_size // 2 and 0 <= grid_x < self.grid_width:
                grid_y = -1
                is_external_io = True
            # Check for external south edge (y=grid_height)
            elif grid_y == self.grid_height and 0 <= grid_x < self.grid_width:
                is_external_io = True

        if is_in_grid or is_external_io:
            if button == 1:  # Left click
                if self.mode == "place" and is_in_grid:
                    # Check if cells are already occupied
                    if not self._can_place_building(grid_x, grid_y, self.current_floor):
                        return  # Cell occupied, don't place
                    self.sim.place_building(
                        self.selected_building,
                        grid_x, grid_y, self.current_floor,
                        self.current_rotation
                    )
                elif self.mode == "input":
                    try:
                        self.sim.set_input(grid_x, grid_y, self.current_floor, self.input_shape, 180.0)
                    except ValueError as e:
                        print(f"Invalid input position: {e}")
                elif self.mode == "output":
                    try:
                        self.sim.set_output(grid_x, grid_y, self.current_floor)
                    except ValueError as e:
                        print(f"Invalid output position: {e}")
                elif self.mode == "delete" and is_in_grid:
                    # Remove building at position (only for grid positions)
                    pos_key = (grid_x, grid_y, self.current_floor)
                    if pos_key in self.sim.cells:
                        del self.sim.cells[pos_key]
                    if pos_key in self.sim.machines:
                        del self.sim.machines[pos_key]
                    if pos_key in self.sim.buildings:
                        del self.sim.buildings[pos_key]
                    # Also remove any inputs/outputs at this position
                    self.sim.inputs = [i for i in self.sim.inputs if i['position'] != pos_key]
                    self.sim.outputs = [o for o in self.sim.outputs if o['position'] != pos_key]
                elif self.mode == "delete" and is_external_io:
                    # Remove external I/O at position
                    pos_key = (grid_x, grid_y, self.current_floor)
                    self.sim.inputs = [i for i in self.sim.inputs if i['position'] != pos_key]
                    self.sim.outputs = [o for o in self.sim.outputs if o['position'] != pos_key]

                # Auto-simulate after changes
                self.last_report = self.sim.simulate()

            elif button == 3:  # Right click - rotate
                rotations = [Rotation.EAST, Rotation.SOUTH, Rotation.WEST, Rotation.NORTH]
                idx = rotations.index(self.current_rotation)
                self.current_rotation = rotations[(idx + 1) % 4]

        # Check sidebar clicks
        sidebar_x = self.grid_offset_x + self.grid_width * self.cell_size + self.grid_sidebar_gap
        if x > sidebar_x:
            rel_x = x - sidebar_x
            rel_y = y - self.toolbar_height

            # Foundation buttons (4 per row, starting at y=18)
            btn_width = 45
            cols = 4
            num_rows = (len(self.foundation_names) + cols - 1) // cols
            foundation_area_end = 18 + num_rows * 26
            if 18 <= rel_y <= foundation_area_end:
                for i, name in enumerate(self.foundation_names):
                    row = i // cols
                    col = i % cols
                    btn_x = 10 + col * (btn_width + 5)
                    btn_y = 18 + row * 26
                    if btn_x <= rel_x <= btn_x + btn_width and btn_y <= rel_y <= btn_y + 22:
                        self.set_foundation(name)
                        self.current_foundation_idx = i
                        return

            # Floor buttons (after foundation buttons)
            floor_y_start = foundation_area_end + 25
            if floor_y_start <= rel_y <= floor_y_start + 50:
                btn_width = 40
                for f in range(self.num_floors):
                    btn_x = 10 + f * (btn_width + 5)
                    if btn_x <= rel_x <= btn_x + btn_width:
                        self.current_floor = f
                        self.last_report = self.sim.simulate()
                        return

            # Building palette (dynamically offset based on foundation rows)
            palette_y_start = foundation_area_end + 25 + 60 + 25  # floor section + buildings title
            palette_y = rel_y - palette_y_start
            if palette_y >= 0 and button == 1:
                idx = palette_y // 25 + self.palette_scroll
                if 0 <= idx < len(self.building_palette):
                    self.selected_building = self.building_palette[idx]
                    self.mode = "place"
    
    def handle_key(self, key: int):
        """Handle keyboard input."""
        # Check modifiers FIRST before bare key checks
        mods = pygame.key.get_mods()

        # Ctrl+number to quick load sample slot
        if mods & pygame.KMOD_CTRL and pygame.K_0 <= key <= pygame.K_9:
            slot = key - pygame.K_0
            self.load_sample(slot)
            return
        # Shift+number to quick save to sample slot
        if mods & pygame.KMOD_SHIFT and pygame.K_0 <= key <= pygame.K_9:
            slot = key - pygame.K_0
            self.save_sample(slot)
            return

        if key == pygame.K_1:
            self.mode = "place"
        elif key == pygame.K_2:
            self.mode = "input"
        elif key == pygame.K_3:
            self.mode = "output"
        elif key == pygame.K_4 or key == pygame.K_DELETE:
            self.mode = "delete"
        elif key == pygame.K_r:
            # Rotate
            rotations = [Rotation.EAST, Rotation.SOUTH, Rotation.WEST, Rotation.NORTH]
            idx = rotations.index(self.current_rotation)
            self.current_rotation = rotations[(idx + 1) % 4]
        elif key == pygame.K_UP:
            self.current_floor = min(self.current_floor + 1, self.num_floors - 1)
        elif key == pygame.K_DOWN:
            self.current_floor = max(self.current_floor - 1, 0)
        elif key == pygame.K_SPACE:
            # Run simulation
            self.last_report = self.sim.simulate()
        elif key == pygame.K_c:
            # Clear all
            self.sim = FlowSimulator(self.grid_width, self.grid_height, self.num_floors)
            self.last_report = None
        elif key == pygame.K_s:
            # Cycle through shapes
            shapes = ["CuCuCuCu", "Cu------", "CuCu----", "----CuCu", "RuRuRuRu",
                      "GrGrGrGr", "BlBlBlBl", "Cu--Cu--", "--Cu--Cu"]
            idx = shapes.index(self.input_shape) if self.input_shape in shapes else 0
            self.input_shape = shapes[(idx + 1) % len(shapes)]
        elif key == pygame.K_v:
            # Toggle shape visualization
            self.show_shapes = not self.show_shapes
        elif key == pygame.K_p:
            # Toggle port labels
            self.show_ports = not self.show_ports
        elif key == pygame.K_PAGEUP:
            # Scroll building palette up
            current_idx = self.building_palette.index(self.selected_building)
            self.selected_building = self.building_palette[max(0, current_idx - 1)]
        elif key == pygame.K_PAGEDOWN:
            # Scroll building palette down
            current_idx = self.building_palette.index(self.selected_building)
            self.selected_building = self.building_palette[min(len(self.building_palette) - 1, current_idx + 1)]
        elif key == pygame.K_f:
            # Cycle through foundations
            self.cycle_foundation()
        elif key == pygame.K_t:
            # Enter test scenario mode - load first scenario
            if not self.scenario_mode:
                self.load_scenario(0)
            else:
                self.exit_scenario_mode()
        elif key == pygame.K_LEFTBRACKET or key == pygame.K_COMMA:
            # Previous scenario
            if self.scenario_mode:
                self.prev_scenario()
            else:
                self.load_scenario(self.total_scenarios - 1)  # Start from last
        elif key == pygame.K_RIGHTBRACKET or key == pygame.K_PERIOD:
            # Next scenario
            if self.scenario_mode:
                self.next_scenario()
            else:
                self.load_scenario(0)  # Start from first
        elif key == pygame.K_ESCAPE:
            # Exit scenario mode
            if self.scenario_mode:
                self.exit_scenario_mode()
        elif key == pygame.K_F5:
            # Save to current slot
            self.save_sample(self.current_sample_slot)
        elif key == pygame.K_F6:
            # Load from current slot
            self.load_sample(self.current_sample_slot)
        elif key == pygame.K_F7:
            # Previous sample slot
            self.current_sample_slot = (self.current_sample_slot - 1) % 10
            self.sample_message = f"Slot {self.current_sample_slot}"
            self.sample_message_time = pygame.time.get_ticks()
        elif key == pygame.K_F8:
            # Next sample slot
            self.current_sample_slot = (self.current_sample_slot + 1) % 10
            self.sample_message = f"Slot {self.current_sample_slot}"
            self.sample_message_time = pygame.time.get_ticks()

    def cycle_foundation(self):
        """Switch to next foundation size."""
        self.current_foundation_idx = (self.current_foundation_idx + 1) % len(self.foundation_names)
        self.set_foundation(self.foundation_names[self.current_foundation_idx])

    def load_scenario(self, index: int):
        """Load a test scenario by index."""
        if index < 0:
            index = self.total_scenarios - 1
        elif index >= self.total_scenarios:
            index = 0

        self.current_scenario_idx = index
        scenario = get_scenario(index)
        if scenario:
            self.scenario_mode = True
            self.scenario_name = scenario['name']
            self.scenario_description = scenario['description']
            self.sim = scenario['sim']
            self.grid_width = self.sim.width
            self.grid_height = self.sim.height
            self.num_floors = self.sim.num_floors

            # Recalculate cell size
            available_width = self.screen_width - self.sidebar_width - 20
            available_height = self.screen_height - self.toolbar_height - 240
            cell_size_for_width = available_width // self.grid_width
            cell_size_for_height = available_height // self.grid_height
            self.cell_size = max(8, min(cell_size_for_width, cell_size_for_height, 40))

            # Run simulation
            self.last_report = self.sim.simulate()
            pygame.display.set_caption(f"Flow Simulator - Test {index + 1}/{self.total_scenarios}: {self.scenario_name}")

    def next_scenario(self):
        """Load the next test scenario."""
        self.load_scenario(self.current_scenario_idx + 1)

    def prev_scenario(self):
        """Load the previous test scenario."""
        self.load_scenario(self.current_scenario_idx - 1)

    def exit_scenario_mode(self):
        """Exit scenario mode and return to normal editing."""
        self.scenario_mode = False
        self.current_scenario_idx = -1
        self.scenario_name = ""
        self.scenario_description = ""
        # Reset to default foundation
        self.set_foundation("1x1")
        pygame.display.set_caption("Flow Simulator - Interactive")

    def save_sample(self, slot: int = None, name: str = None):
        """Save current layout to a sample file."""
        if slot is None:
            slot = self.current_sample_slot

        # Build sample data
        sample_data = {
            "name": name or f"Sample {slot}",
            "created": datetime.now().isoformat(),
            "foundation": self.current_foundation_name,
            "buildings": [],
            "inputs": [],
            "outputs": []
        }

        # Save buildings
        for pos, bld in self.sim.buildings.items():
            sample_data["buildings"].append({
                "type": bld["type"].name,
                "x": pos[0],
                "y": pos[1],
                "floor": pos[2],
                "rotation": bld["rotation"].name
            })

        # Save inputs
        for inp in self.sim.inputs:
            sample_data["inputs"].append({
                "x": inp["position"][0],
                "y": inp["position"][1],
                "floor": inp["position"][2],
                "shape": inp["shape"],
                "throughput": inp["throughput"]
            })

        # Save outputs
        for out in self.sim.outputs:
            sample_data["outputs"].append({
                "x": out["position"][0],
                "y": out["position"][1],
                "floor": out["position"][2],
                "expected_shape": out.get("expected_shape")
            })

        # Write to file
        filename = self.samples_dir / f"sample_{slot:02d}.json"
        with open(filename, "w") as f:
            json.dump(sample_data, f, indent=2)

        self.sample_message = f"Saved to slot {slot}"
        self.sample_message_time = pygame.time.get_ticks()
        print(f"Saved sample to {filename}")

    def load_sample(self, slot: int = None):
        """Load a sample from file."""
        if slot is None:
            slot = self.current_sample_slot

        filename = self.samples_dir / f"sample_{slot:02d}.json"
        if not filename.exists():
            self.sample_message = f"Slot {slot} empty"
            self.sample_message_time = pygame.time.get_ticks()
            print(f"No sample at slot {slot}")
            return False

        try:
            with open(filename, "r") as f:
                sample_data = json.load(f)

            # Set foundation first
            foundation = sample_data.get("foundation", "1x1")
            self.set_foundation(foundation)

            # Clear and rebuild
            spec = FOUNDATION_SPECS.get(foundation)
            self.sim = FlowSimulator(foundation_spec=spec, validate_io=True)

            # Place buildings
            for bld in sample_data.get("buildings", []):
                building_type = BuildingType[bld["type"]]
                rotation = Rotation[bld["rotation"]]
                self.sim.place_building(building_type, bld["x"], bld["y"], bld["floor"], rotation)

            # Set inputs
            for inp in sample_data.get("inputs", []):
                try:
                    self.sim.set_input(inp["x"], inp["y"], inp["floor"], inp["shape"], inp["throughput"])
                except ValueError as e:
                    print(f"Warning: Could not set input: {e}")

            # Set outputs
            for out in sample_data.get("outputs", []):
                try:
                    self.sim.set_output(out["x"], out["y"], out["floor"], out.get("expected_shape"))
                except ValueError as e:
                    print(f"Warning: Could not set output: {e}")

            # Run simulation
            self.last_report = self.sim.simulate()

            # Exit scenario mode if we were in it
            self.scenario_mode = False
            self.scenario_name = sample_data.get("name", f"Sample {slot}")
            self.scenario_description = f"Loaded from slot {slot}"

            self.sample_message = f"Loaded slot {slot}"
            self.sample_message_time = pygame.time.get_ticks()
            pygame.display.set_caption(f"Flow Simulator - {self.scenario_name}")
            print(f"Loaded sample from {filename}")
            return True

        except Exception as e:
            self.sample_message = f"Load error: {e}"
            self.sample_message_time = pygame.time.get_ticks()
            print(f"Error loading sample: {e}")
            return False

    def list_samples(self) -> List[Tuple[int, str]]:
        """List all available samples."""
        samples = []
        for i in range(10):
            filename = self.samples_dir / f"sample_{i:02d}.json"
            if filename.exists():
                try:
                    with open(filename, "r") as f:
                        data = json.load(f)
                    samples.append((i, data.get("name", f"Sample {i}")))
                except:
                    samples.append((i, f"Sample {i} (corrupt)"))
        return samples

    def set_foundation(self, name: str):
        """Set foundation by name and scale grid to fit window."""
        if name not in FOUNDATION_SPECS:
            return

        spec = FOUNDATION_SPECS[name]
        self.current_foundation_name = name
        self.grid_width = spec.grid_width
        self.grid_height = spec.grid_height
        self.num_floors = spec.num_floors

        # Recreate simulator with new dimensions
        self.sim = FlowSimulator(self.grid_width, self.grid_height, self.num_floors)
        self.last_report = None

        # Calculate cell size to fit grid in fixed window area
        # Keep window size fixed, scale cell_size to fit the grid
        available_width = self.screen_width - self.sidebar_width - 20  # 20px margin
        available_height = self.screen_height - self.toolbar_height - 240  # Space for info panel

        # Calculate cell size that fits both dimensions
        cell_size_for_width = available_width // self.grid_width
        cell_size_for_height = available_height // self.grid_height
        self.cell_size = max(8, min(cell_size_for_width, cell_size_for_height, 40))  # Clamp between 8 and 40

        pygame.display.set_caption(f"Flow Simulator - {name} Foundation ({self.grid_width}x{self.grid_height})")
    
    def draw(self):
        """Draw everything."""
        self.screen.fill(DARK_GRAY)

        # Draw toolbar
        self.draw_toolbar()

        # Draw foundation outline and I/O zones first (behind buildings)
        self.draw_foundation_features()

        # Draw grid
        self.draw_grid()

        # Draw placement preview
        self.draw_preview()

        # Draw machine port labels
        if self.show_ports:
            self.draw_port_labels()

        # Draw sidebar
        self.draw_sidebar()

        # Draw info panel
        self.draw_info_panel()

    def _draw_hatched_rect(self, rect: pygame.Rect, color: Tuple[int, int, int],
                            line_spacing: int = 6, line_width: int = 1, diagonal: bool = True):
        """Draw a rectangle with a hatched pattern."""
        # Draw background first (darker version of color)
        bg_color = tuple(max(0, c - 30) for c in color)
        pygame.draw.rect(self.screen, bg_color, rect)

        if diagonal:
            # Diagonal hatching (/ pattern)
            for i in range(-rect.height, rect.width + rect.height, line_spacing):
                start_x = rect.x + i
                start_y = rect.y
                end_x = rect.x + i + rect.height
                end_y = rect.y + rect.height

                # Clip to rect bounds
                if start_x < rect.x:
                    start_y += rect.x - start_x
                    start_x = rect.x
                if end_x > rect.x + rect.width:
                    end_y -= end_x - (rect.x + rect.width)
                    end_x = rect.x + rect.width
                if start_y < rect.y:
                    start_x += rect.y - start_y
                    start_y = rect.y
                if end_y > rect.y + rect.height:
                    end_x -= end_y - (rect.y + rect.height)
                    end_y = rect.y + rect.height

                if start_x < end_x and start_y < end_y:
                    pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), line_width)
        else:
            # Horizontal hatching
            for y in range(rect.y, rect.y + rect.height, line_spacing):
                pygame.draw.line(self.screen, color, (rect.x, y), (rect.x + rect.width, y), line_width)

    def _draw_blocked_cell(self, x: int, y: int, offset_y: int):
        """Draw a blocked/invalid cell with X pattern - highly visible."""
        rect = pygame.Rect(
            self.grid_offset_x + x * self.cell_size,
            offset_y + y * self.cell_size,
            self.cell_size - 1,
            self.cell_size - 1
        )
        # Brighter red/maroon background for visibility
        pygame.draw.rect(self.screen, (80, 30, 30), rect)
        # Bright X pattern
        pygame.draw.line(self.screen, (150, 60, 60), rect.topleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, (150, 60, 60), rect.topright, rect.bottomleft, 2)
        # Border to make it stand out
        pygame.draw.rect(self.screen, (120, 50, 50), rect, 1)

    def draw_foundation_features(self):
        """Draw foundation outline, invalid squares, and valid I/O zones on edges."""
        offset_y = self.toolbar_height
        spec = FOUNDATION_SPECS[self.current_foundation_name]

        # Draw foundation background (slightly lighter than dark gray)
        offset_x = self.grid_offset_x
        foundation_rect = pygame.Rect(
            offset_x,
            offset_y,
            self.grid_width * self.cell_size,
            self.grid_height * self.cell_size
        )
        pygame.draw.rect(self.screen, (50, 55, 60), foundation_rect)  # Foundation area

        # For irregular foundations, block out invalid cells
        valid_cells = spec.get_valid_grid_cells()
        if valid_cells is not None:
            # Draw blocked cells (cells not in valid_cells set)
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    if (x, y) not in valid_cells:
                        self._draw_blocked_cell(x, y, offset_y)

        # Draw foundation outline (bright green border around the valid build area)
        pygame.draw.rect(self.screen, (100, 200, 100), foundation_rect, 4)  # Thick bright border

        # Draw I/O port zones on edges with hatched pattern
        # Each edge has ports centered on each 1x1 unit (4 ports per unit)
        # Ports are at positions 3, 5, 8, 10 within each 14-tile unit

        # Port zone colors
        io_zone_color = (60, 100, 60)  # Dark green for I/O zones
        io_highlight_color = (80, 140, 80)  # Lighter when in input/output mode

        # Highlight color based on mode
        if self.mode in ("input", "output"):
            zone_color = io_highlight_color
        else:
            zone_color = io_zone_color

        # Calculate port positions for each side
        # Ports are at fixed offsets within each 14-tile unit section
        # Unit center is at 7, offsets are [-2, -1, 0, 1] giving positions 5, 6, 7, 8
        port_offsets = [5, 6, 7, 8]  # Grid positions within each unit

        # For irregular foundations, need to check which unit cells are present
        cells_set = set(spec.present_cells) if spec.present_cells else None

        # Collect all exposed edges for irregular foundations
        # Each exposed edge is identified by (unit_x, unit_y, side)
        exposed_west = []  # List of unit_y values with exposed west edge
        exposed_east = []  # List of unit_y values with exposed east edge
        exposed_north = []  # List of unit_x values with exposed north edge
        exposed_south = []  # List of unit_x values with exposed south edge

        if cells_set is not None:
            for ux, uy in cells_set:
                # West edge: exposed if no cell to the left
                if (ux - 1, uy) not in cells_set:
                    exposed_west.append((ux, uy))
                # East edge: exposed if no cell to the right
                if (ux + 1, uy) not in cells_set:
                    exposed_east.append((ux, uy))
                # North edge: exposed if no cell above
                if (ux, uy - 1) not in cells_set:
                    exposed_north.append((ux, uy))
                # South edge: exposed if no cell below
                if (ux, uy + 1) not in cells_set:
                    exposed_south.append((ux, uy))

        # WEST edge - draw ports for all exposed west faces
        if cells_set is None:
            # Rectangular: all units on west side
            west_units = [(0, unit) for unit in range(spec.units_y)]
        else:
            west_units = exposed_west

        for ux, uy in west_units:
            unit_start_y = uy * 14
            unit_start_x = ux * 14
            for offset in port_offsets:
                port_y = unit_start_y + offset
                if port_y < self.grid_height:
                    rect = pygame.Rect(
                        offset_x + unit_start_x,  # x position at left edge of this unit
                        offset_y + port_y * self.cell_size,
                        self.cell_size // 4,
                        self.cell_size
                    )
                    self._draw_hatched_rect(rect, zone_color, line_spacing=4)

        # EAST edge - draw ports for all exposed east faces
        if cells_set is None:
            # Rectangular: all units on east side
            east_units = [(spec.units_x - 1, unit) for unit in range(spec.units_y)]
        else:
            east_units = exposed_east

        for ux, uy in east_units:
            unit_start_y = uy * 14
            unit_start_x = ux * 14
            edge_x = unit_start_x + 14  # Right edge of 14-tile unit
            if edge_x > self.grid_width:
                edge_x = self.grid_width
            for offset in port_offsets:
                port_y = unit_start_y + offset
                if port_y < self.grid_height:
                    rect = pygame.Rect(
                        offset_x + edge_x * self.cell_size - self.cell_size // 4,
                        offset_y + port_y * self.cell_size,
                        self.cell_size // 4,
                        self.cell_size
                    )
                    self._draw_hatched_rect(rect, zone_color, line_spacing=4)

        # NORTH edge - draw ports for all exposed north faces
        if cells_set is None:
            # Rectangular: all units on north side
            north_units = [(unit, 0) for unit in range(spec.units_x)]
        else:
            north_units = exposed_north

        for ux, uy in north_units:
            unit_start_x = ux * 14
            unit_start_y = uy * 14
            for offset in port_offsets:
                port_x = unit_start_x + offset
                if port_x < self.grid_width:
                    rect = pygame.Rect(
                        offset_x + port_x * self.cell_size,
                        offset_y + unit_start_y * self.cell_size,  # Top edge of this unit
                        self.cell_size,
                        self.cell_size // 4
                    )
                    self._draw_hatched_rect(rect, zone_color, line_spacing=4)

        # SOUTH edge - draw ports for all exposed south faces
        if cells_set is None:
            # Rectangular: all units on south side
            south_units = [(unit, spec.units_y - 1) for unit in range(spec.units_x)]
        else:
            south_units = exposed_south

        for ux, uy in south_units:
            unit_start_x = ux * 14
            unit_start_y = uy * 14
            edge_y = unit_start_y + 14  # Bottom edge of 14-tile unit
            if edge_y > self.grid_height:
                edge_y = self.grid_height
            for offset in port_offsets:
                port_x = unit_start_x + offset
                if port_x < self.grid_width:
                    rect = pygame.Rect(
                        offset_x + port_x * self.cell_size,
                        offset_y + edge_y * self.cell_size - self.cell_size // 4,
                        self.cell_size,
                        self.cell_size // 4
                    )
                    self._draw_hatched_rect(rect, zone_color, line_spacing=4)

        # Draw corner markers to show foundation extent
        corner_size = 8
        corner_color = (150, 200, 150)
        corners = [
            (offset_x, offset_y),  # Top-left
            (offset_x + self.grid_width * self.cell_size - corner_size, offset_y),  # Top-right
            (offset_x, offset_y + self.grid_height * self.cell_size - corner_size),  # Bottom-left
            (offset_x + self.grid_width * self.cell_size - corner_size,
             offset_y + self.grid_height * self.cell_size - corner_size),  # Bottom-right
        ]
        for cx, cy in corners:
            pygame.draw.rect(self.screen, corner_color, (cx, cy, corner_size, corner_size))

    def draw_preview(self):
        """Draw a ghost preview of the building at cursor position with full size."""
        if self.mode != "place":
            return

        x, y = self.mouse_pos
        offset_x = self.grid_offset_x
        grid_x = (x - offset_x) // self.cell_size
        grid_y = (y - self.toolbar_height) // self.cell_size

        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return

        offset_y = self.toolbar_height

        # Get building dimensions
        spec = BUILDING_SPECS.get(self.selected_building)
        base_w = spec.width if spec else 1
        base_h = spec.height if spec else 1
        depth = spec.depth if spec else 1

        # Adjust for rotation
        if self.current_rotation in (Rotation.SOUTH, Rotation.NORTH):
            eff_w, eff_h = base_h, base_w
        else:
            eff_w, eff_h = base_w, base_h

        # Check if placement is valid
        can_place = self._can_place_building(grid_x, grid_y, self.current_floor)
        preview_color = BUILDING_COLORS.get(self.selected_building, GRAY)
        outline_color = GREEN if can_place else RED

        # Draw all cells the building will occupy
        for dx in range(eff_w):
            for dy in range(eff_h):
                cell_x = grid_x + dx
                cell_y = grid_y + dy

                if cell_x >= self.grid_width or cell_y >= self.grid_height:
                    continue

                rect = pygame.Rect(
                    offset_x + cell_x * self.cell_size,
                    offset_y + cell_y * self.cell_size,
                    self.cell_size - 1,
                    self.cell_size - 1
                )

                # Draw semi-transparent preview
                preview_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                alpha = 100 if can_place else 60
                preview_surface.fill((*preview_color, alpha))
                self.screen.blit(preview_surface, rect.topleft)

                # Draw building symbol only on origin cell
                if dx == 0 and dy == 0:
                    self.draw_building_symbol(rect, self.selected_building, self.current_rotation)

                # Draw cell outline
                pygame.draw.rect(self.screen, outline_color, rect, 2)

        # Draw multi-floor indicator if building spans floors
        if depth > 1:
            text = self.font_small.render(f"+{depth-1}F", True, YELLOW)
            self.screen.blit(text, (offset_x + grid_x * self.cell_size + 2, offset_y + grid_y * self.cell_size + 2))

        # Show input/output ports as edge markers ON the machine cells
        # Ports show which edge of the machine belts connect to
        if spec and self.selected_building in BUILDING_PORTS:
            ports_def = BUILDING_PORTS[self.selected_building]
            bar_thickness = 4

            # Show input ports as cyan edge bars on the machine cell
            for port_info in ports_def.get('inputs', []):
                if len(port_info) == 4:
                    rel_x, rel_y, rel_z, direction = port_info
                else:
                    rel_x, rel_y, rel_z = port_info
                    direction = 'W'

                # Rotate position and direction
                rot_x, rot_y = self._rotate_offset(rel_x, rel_y, base_w, base_h)
                rot_dir = self._rotate_direction_char(direction)

                # Cell position is internal to machine footprint
                cell_x = grid_x + rot_x
                cell_y = grid_y + rot_y

                if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                    cx = offset_x + cell_x * self.cell_size
                    cy = offset_y + cell_y * self.cell_size
                    # Draw bar on the edge indicated by direction
                    bar_rect = self._get_edge_bar_rect(cx, cy, rot_dir, bar_thickness)
                    pygame.draw.rect(self.screen, CYAN, bar_rect)

            # Show output ports as orange edge bars on the machine cell
            for port_info in ports_def.get('outputs', []):
                if len(port_info) == 4:
                    rel_x, rel_y, rel_z, direction = port_info
                else:
                    rel_x, rel_y, rel_z = port_info
                    direction = 'E'

                rot_x, rot_y = self._rotate_offset(rel_x, rel_y, base_w, base_h)
                rot_dir = self._rotate_direction_char(direction)

                cell_x = grid_x + rot_x
                cell_y = grid_y + rot_y

                if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                    cx = offset_x + cell_x * self.cell_size
                    cy = offset_y + cell_y * self.cell_size
                    bar_rect = self._get_edge_bar_rect(cx, cy, rot_dir, bar_thickness)
                    pygame.draw.rect(self.screen, ORANGE, bar_rect)

    def _rotate_offset(self, dx: int, dy: int, width: int = 1, height: int = 1) -> Tuple[int, int]:
        """Rotate a relative offset by current rotation.

        Args:
            dx, dy: Relative offset within original bounding box
            width, height: Original building dimensions (before rotation)
        """
        if self.current_rotation == Rotation.EAST:
            return (dx, dy)
        elif self.current_rotation == Rotation.SOUTH:
            # 90° CW: map (x, y) to (y, width-1-x)
            return (dy, width - 1 - dx)
        elif self.current_rotation == Rotation.WEST:
            # 180°: map (x, y) to (width-1-x, height-1-y)
            return (width - 1 - dx, height - 1 - dy)
        else:  # NORTH
            # 270° CW: map (x, y) to (height-1-y, x)
            return (height - 1 - dy, dx)

    def _rotate_direction_char(self, direction: str) -> str:
        """Rotate a direction character by current rotation."""
        directions = ['E', 'S', 'W', 'N']  # Clockwise order
        rotations = {
            Rotation.EAST: 0,
            Rotation.SOUTH: 1,
            Rotation.WEST: 2,
            Rotation.NORTH: 3,
        }
        idx = directions.index(direction)
        new_idx = (idx + rotations[self.current_rotation]) % 4
        return directions[new_idx]

    def _get_edge_bar_rect(self, cx: int, cy: int, direction: str, thickness: int) -> Tuple[int, int, int, int]:
        """Get rectangle for drawing a bar on a cell edge."""
        if direction == 'W':
            return (cx, cy, thickness, self.cell_size - 1)
        elif direction == 'E':
            return (cx + self.cell_size - thickness - 1, cy, thickness, self.cell_size - 1)
        elif direction == 'N':
            return (cx, cy, self.cell_size - 1, thickness)
        else:  # S
            return (cx, cy + self.cell_size - thickness - 1, self.cell_size - 1, thickness)

    def draw_port_labels(self):
        """Draw input/output port labels for machines on the machine edges."""
        offset_x = self.grid_offset_x
        offset_y = self.toolbar_height

        # Direction to edge offset mapping
        dir_to_offset = {
            'W': (-0.4, 0),
            'E': (0.4, 0),
            'N': (0, -0.4),
            'S': (0, 0.4),
        }

        for origin, machine in self.sim.machines.items():
            ox, oy, oz = origin

            # Draw input ports
            for port in machine.input_ports:
                px, py, pz = port['position']
                if pz != self.current_floor:
                    continue

                port_dir = port.get('direction', 'W')

                # Port is ON the machine cell, draw label on the appropriate edge
                port_cx = offset_x + px * self.cell_size + self.cell_size // 2
                port_cy = offset_y + py * self.cell_size + self.cell_size // 2

                # Offset label toward the edge based on direction
                dx, dy = dir_to_offset.get(port_dir, (0, 0))
                label_x = port_cx + dx * self.cell_size
                label_y = port_cy + dy * self.cell_size

                # Draw small "I" label (input)
                pygame.draw.circle(self.screen, CYAN, (int(label_x), int(label_y)), 7)
                label = "I"
                if pz != oz:  # Port on different floor than machine origin
                    label = f"I{pz}"
                text = self.font_small.render(label, True, BLACK)
                self.screen.blit(text, (label_x - 3, label_y - 5))

            # Draw output ports
            for port in machine.output_ports:
                px, py, pz = port['position']
                if pz != self.current_floor:
                    continue

                port_dir = port.get('direction', 'E')

                port_cx = offset_x + px * self.cell_size + self.cell_size // 2
                port_cy = offset_y + py * self.cell_size + self.cell_size // 2

                dx, dy = dir_to_offset.get(port_dir, (0, 0))
                label_x = port_cx + dx * self.cell_size
                label_y = port_cy + dy * self.cell_size

                # Draw small "O" label (output)
                color = RED if port.get('backed_up') else ORANGE
                pygame.draw.circle(self.screen, color, (int(label_x), int(label_y)), 7)
                label = "O"
                if pz != oz:
                    label = f"O{pz}"
                text = self.font_small.render(label, True, BLACK)
                self.screen.blit(text, (label_x - 3, label_y - 5))
    
    def draw_toolbar(self):
        """Draw top toolbar."""
        pygame.draw.rect(self.screen, GRAY, (0, 0, self.screen_width, self.toolbar_height))
        
        # Mode buttons
        modes = [("1:Place", "place"), ("2:Input", "input"), ("3:Output", "output"), ("4:Delete", "delete")]
        for i, (label, mode) in enumerate(modes):
            color = GREEN if self.mode == mode else LIGHT_GRAY
            pygame.draw.rect(self.screen, color, (10 + i * 80, 10, 70, 25))
            text = self.font.render(label, True, BLACK)
            self.screen.blit(text, (15 + i * 80, 15))
        
        # Floor indicator
        floor_text = self.font_large.render(f"Floor: {self.current_floor}", True, WHITE)
        self.screen.blit(floor_text, (350, 15))
        
        # Rotation indicator
        rot_arrows = {Rotation.EAST: "→", Rotation.SOUTH: "↓", Rotation.WEST: "←", Rotation.NORTH: "↑"}
        rot_text = self.font_large.render(f"Rot: {rot_arrows[self.current_rotation]} (R)", True, WHITE)
        self.screen.blit(rot_text, (480, 15))
        
        # Input shape
        shape_text = self.font.render(f"Shape(S): {self.input_shape}", True, YELLOW)
        self.screen.blit(shape_text, (350, 40))
    
    def draw_shape_mini(self, rect: pygame.Rect, shape: str):
        """Draw a mini shape visualization in the cell."""
        if not shape or shape == "--------":
            return

        # Shape format: TLTRBLBR (8 chars, pairs for each quarter)
        # Draw 4 small squares representing the shape quarters
        qw = rect.width // 4
        qh = rect.height // 4
        margin = 2

        quarters = []
        if len(shape) >= 8:
            quarters = [shape[0:2], shape[2:4], shape[4:6], shape[6:8]]
        elif len(shape) >= 4:
            quarters = [shape[0:1] + "-", shape[1:2] + "-", shape[2:3] + "-", shape[3:4] + "-"]

        positions = [
            (rect.x + margin, rect.y + margin),           # TL
            (rect.x + rect.width - qw - margin, rect.y + margin),  # TR
            (rect.x + margin, rect.y + rect.height - qh - margin),  # BL
            (rect.x + rect.width - qw - margin, rect.y + rect.height - qh - margin),  # BR
        ]

        for i, (q, (qx, qy)) in enumerate(zip(quarters, positions)):
            color = SHAPE_COLORS.get(q, SHAPE_COLORS.get("--", (40, 40, 40)))
            if q != "--":
                pygame.draw.rect(self.screen, color, (qx, qy, qw - 1, qh - 1))

    def draw_grid(self):
        """Draw the placement grid."""
        offset_x = self.grid_offset_x
        offset_y = self.toolbar_height

        # Get valid cells for irregular foundations (to skip invalid cells)
        foundation_spec = FOUNDATION_SPECS[self.current_foundation_name]
        valid_cells = foundation_spec.get_valid_grid_cells()

        # Draw cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Skip invalid cells (they're drawn by draw_foundation_features)
                if valid_cells is not None and (x, y) not in valid_cells:
                    continue

                rect = pygame.Rect(
                    offset_x + x * self.cell_size,
                    offset_y + y * self.cell_size,
                    self.cell_size - 1,
                    self.cell_size - 1
                )

                pos = (x, y, self.current_floor)
                cell = self.sim.cells.get(pos)

                # Background color
                if cell and cell.building_type:
                    color = BUILDING_COLORS.get(cell.building_type, GRAY)
                    # Tint based on throughput if simulated
                    if cell.throughput > 0:
                        # Green tint for flowing
                        color = tuple(min(255, c + 50) for c in color)
                else:
                    color = (40, 40, 40)

                pygame.draw.rect(self.screen, color, rect)

                # Draw building symbol
                if cell and cell.building_type:
                    self.draw_building_symbol(rect, cell.building_type, cell.rotation)

                    # Draw shape visualization if flowing
                    if cell.throughput > 0 and cell.shape and self.show_shapes:
                        self.draw_shape_mini(rect, cell.shape)

                    # Draw throughput number
                    if cell.throughput > 0:
                        tp_text = self.font_small.render(f"{cell.throughput:.0f}", True, YELLOW)
                        self.screen.blit(tp_text, (rect.x + 2, rect.y + rect.height - 12))

                        # Draw utilization bar
                        bar_width = int((rect.width - 4) * min(1.0, cell.utilization / 100))
                        bar_color = GREEN if cell.utilization < 80 else (YELLOW if cell.utilization < 100 else RED)
                        pygame.draw.rect(self.screen, bar_color,
                                        (rect.x + 2, rect.y + rect.height - 4, bar_width, 3))

                # Check for inputs/outputs
                for inp in self.sim.inputs:
                    if inp['position'] == pos:
                        pygame.draw.circle(self.screen, GREEN, rect.center, 10, 2)
                        # Draw input shape
                        if self.show_shapes:
                            self.draw_shape_mini(rect, inp['shape'])
                        text = self.font_small.render("IN", True, GREEN)
                        self.screen.blit(text, (rect.centerx - 8, rect.y + 2))

                for out in self.sim.outputs:
                    if out['position'] == pos:
                        pygame.draw.circle(self.screen, RED, rect.center, 10, 2)
                        # Draw output shape if received
                        if self.show_shapes and out.get('actual_shape'):
                            self.draw_shape_mini(rect, out['actual_shape'])
                        text = self.font_small.render("OUT", True, RED)
                        self.screen.blit(text, (rect.centerx - 10, rect.y + 2))

        # Draw external I/O indicators (outside the grid)
        for inp in self.sim.inputs:
            ix, iy, iz = inp['position']
            if iz != self.current_floor:
                continue
            # Check if external position
            if ix == -1 or ix == self.grid_width or iy == -1 or iy == self.grid_height:
                # Calculate pixel position for external I/O
                if ix == -1:
                    px = offset_x // 2  # Center in left margin
                elif ix == self.grid_width:
                    px = offset_x + self.grid_width * self.cell_size + self.cell_size // 4
                else:
                    px = offset_x + ix * self.cell_size + self.cell_size // 2

                if iy == -1:
                    py = offset_y - self.cell_size // 2
                elif iy == self.grid_height:
                    py = offset_y + self.grid_height * self.cell_size + self.cell_size // 4
                else:
                    py = offset_y + iy * self.cell_size + self.cell_size // 2

                # Draw input marker (green arrow pointing into grid)
                pygame.draw.circle(self.screen, GREEN, (px, py), 8, 2)
                text = self.font_small.render("IN", True, GREEN)
                self.screen.blit(text, (px - 8, py - 16))

        for out in self.sim.outputs:
            ox, oy, oz = out['position']
            if oz != self.current_floor:
                continue
            # Check if external position
            if ox == -1 or ox == self.grid_width or oy == -1 or oy == self.grid_height:
                # Calculate pixel position for external I/O
                if ox == -1:
                    px = offset_x // 2  # Center in left margin
                elif ox == self.grid_width:
                    px = offset_x + self.grid_width * self.cell_size + self.cell_size // 4
                else:
                    px = offset_x + ox * self.cell_size + self.cell_size // 2

                if oy == -1:
                    py = offset_y - self.cell_size // 2
                elif oy == self.grid_height:
                    py = offset_y + self.grid_height * self.cell_size + self.cell_size // 4
                else:
                    py = offset_y + oy * self.cell_size + self.cell_size // 2

                # Draw output marker (red)
                pygame.draw.circle(self.screen, RED, (px, py), 8, 2)
                text = self.font_small.render("OUT", True, RED)
                self.screen.blit(text, (px - 10, py - 16))
                # Show throughput if available
                if out.get('throughput', 0) > 0:
                    tp_text = self.font_small.render(f"{out['throughput']:.0f}", True, YELLOW)
                    self.screen.blit(tp_text, (px - 8, py + 4))

        # Draw flow paths (lines connecting traced paths)
        if hasattr(self.sim, 'traced_paths') and self.sim.traced_paths:
            for path in self.sim.traced_paths:
                if len(path) < 2:
                    continue
                # Only draw paths on current floor
                path_on_floor = [(p[0], p[1]) for p in path if p[2] == self.current_floor]
                if len(path_on_floor) >= 2:
                    points = [
                        (offset_x + p[0] * self.cell_size + self.cell_size // 2,
                         offset_y + p[1] * self.cell_size + self.cell_size // 2)
                        for p in path_on_floor
                    ]
                    pygame.draw.lines(self.screen, (100, 255, 100, 128), False, points, 2)

        # Draw grid lines
        for x in range(self.grid_width + 1):
            pygame.draw.line(
                self.screen, GRAY,
                (offset_x + x * self.cell_size, offset_y),
                (offset_x + x * self.cell_size, offset_y + self.grid_height * self.cell_size)
            )
        for y in range(self.grid_height + 1):
            pygame.draw.line(
                self.screen, GRAY,
                (offset_x, offset_y + y * self.cell_size),
                (offset_x + self.grid_width * self.cell_size, offset_y + y * self.cell_size)
            )
    
    def draw_building_symbol(self, rect: pygame.Rect, bt: BuildingType, rotation: Rotation):
        """Draw a building's symbol."""
        cx, cy = rect.center

        if bt == BuildingType.BELT_FORWARD:
            # Clear directional arrow
            # Draw shaft and arrowhead based on direction
            if rotation == Rotation.EAST:  # →
                # Shaft
                pygame.draw.line(self.screen, WHITE, (cx - 12, cy), (cx + 8, cy), 3)
                # Arrowhead
                pygame.draw.polygon(self.screen, WHITE, [
                    (cx + 14, cy), (cx + 4, cy - 7), (cx + 4, cy + 7)
                ])
            elif rotation == Rotation.WEST:  # ←
                pygame.draw.line(self.screen, WHITE, (cx + 12, cy), (cx - 8, cy), 3)
                pygame.draw.polygon(self.screen, WHITE, [
                    (cx - 14, cy), (cx - 4, cy - 7), (cx - 4, cy + 7)
                ])
            elif rotation == Rotation.SOUTH:  # ↓
                pygame.draw.line(self.screen, WHITE, (cx, cy - 12), (cx, cy + 8), 3)
                pygame.draw.polygon(self.screen, WHITE, [
                    (cx, cy + 14), (cx - 7, cy + 4), (cx + 7, cy + 4)
                ])
            else:  # NORTH ↑
                pygame.draw.line(self.screen, WHITE, (cx, cy + 12), (cx, cy - 8), 3)
                pygame.draw.polygon(self.screen, WHITE, [
                    (cx, cy - 14), (cx - 7, cy - 4), (cx + 7, cy - 4)
                ])

        elif bt == BuildingType.BELT_LEFT:
            # Curved arrow turning left based on rotation
            # Draw a curved arrow with proper rotation
            if rotation == Rotation.EAST:  # Enter from west, exit north
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 3.14, 4.71, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx, cy - 14), (cx - 5, cy - 6), (cx + 5, cy - 6)])
            elif rotation == Rotation.SOUTH:  # Enter from north, exit east
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 4.71, 6.28, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx + 14, cy), (cx + 6, cy - 5), (cx + 6, cy + 5)])
            elif rotation == Rotation.WEST:  # Enter from east, exit south
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 0, 1.57, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx, cy + 14), (cx - 5, cy + 6), (cx + 5, cy + 6)])
            else:  # NORTH - Enter from south, exit west
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 1.57, 3.14, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx - 14, cy), (cx - 6, cy - 5), (cx - 6, cy + 5)])
            text = self.font_small.render("L", True, WHITE)
            self.screen.blit(text, (cx - 3, cy - 5))

        elif bt == BuildingType.BELT_RIGHT:
            # Curved arrow turning right based on rotation
            if rotation == Rotation.EAST:  # Enter from west, exit south
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 4.71, 6.28, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx, cy + 14), (cx - 5, cy + 6), (cx + 5, cy + 6)])
            elif rotation == Rotation.SOUTH:  # Enter from north, exit west
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 0, 1.57, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx - 14, cy), (cx - 6, cy - 5), (cx - 6, cy + 5)])
            elif rotation == Rotation.WEST:  # Enter from east, exit north
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 1.57, 3.14, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx, cy - 14), (cx - 5, cy - 6), (cx + 5, cy - 6)])
            else:  # NORTH - Enter from south, exit east
                pygame.draw.arc(self.screen, WHITE, (cx - 14, cy - 14, 28, 28), 3.14, 4.71, 3)
                pygame.draw.polygon(self.screen, WHITE, [(cx + 14, cy), (cx + 6, cy - 5), (cx + 6, cy + 5)])
            text = self.font_small.render("R", True, WHITE)
            self.screen.blit(text, (cx - 3, cy - 5))

        elif bt == BuildingType.BELT_PORT_SENDER:
            # Triangle pointing in direction (sender)
            pygame.draw.polygon(self.screen, CYAN, [(cx - 8, cy - 8), (cx + 10, cy), (cx - 8, cy + 8)])
            text = self.font_small.render("⊳", True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 6))

        elif bt == BuildingType.BELT_PORT_RECEIVER:
            # Triangle pointing inward (receiver)
            pygame.draw.polygon(self.screen, PURPLE, [(cx + 8, cy - 8), (cx - 10, cy), (cx + 8, cy + 8)])
            text = self.font_small.render("⊲", True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 6))

        elif bt == BuildingType.LIFT_UP:
            # Up arrow with vertical bar
            pygame.draw.line(self.screen, WHITE, (cx, cy + 8), (cx, cy - 8), 2)
            pygame.draw.polygon(self.screen, WHITE, [(cx, cy - 12), (cx - 6, cy - 4), (cx + 6, cy - 4)])
            text = self.font_small.render("↑", True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 5))

        elif bt == BuildingType.LIFT_DOWN:
            # Down arrow with vertical bar
            pygame.draw.line(self.screen, WHITE, (cx, cy - 8), (cx, cy + 8), 2)
            pygame.draw.polygon(self.screen, WHITE, [(cx, cy + 12), (cx - 6, cy + 4), (cx + 6, cy + 4)])
            text = self.font_small.render("↓", True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 5))

        elif bt == BuildingType.CUTTER:
            # CUTTER: 1x2, input from west, outputs go east at y=0 and y=1 (down)
            # Draw box with vertical cutting line
            pygame.draw.rect(self.screen, WHITE, (cx - 10, cy - 10, 20, 20), 2)
            # Vertical cutting line (blade)
            pygame.draw.line(self.screen, WHITE, (cx, cy - 8), (cx, cy + 8), 2)
            # Input indicator (left arrow pointing in)
            pygame.draw.line(self.screen, CYAN, (cx - 14, cy), (cx - 10, cy), 2)
            # Output indicators - both go right, show split going down
            pygame.draw.line(self.screen, ORANGE, (cx + 10, cy - 3), (cx + 14, cy - 3), 2)
            pygame.draw.line(self.screen, ORANGE, (cx + 10, cy + 3), (cx + 14, cy + 3), 2)
            # Down arrow showing second output goes to y+1
            pygame.draw.polygon(self.screen, ORANGE, [
                (cx + 5, cy + 8), (cx + 2, cy + 4), (cx + 8, cy + 4)
            ])

        elif bt == BuildingType.CUTTER_MIRRORED:
            # CUTTER_MIRRORED: 1x2, input at bottom (y+1), outputs go east with second going north
            # Mirror of CUTTER - input at bottom, second output goes up instead of down
            pygame.draw.rect(self.screen, WHITE, (cx - 10, cy - 10, 20, 20), 2)
            # Vertical cutting line (blade)
            pygame.draw.line(self.screen, WHITE, (cx, cy - 8), (cx, cy + 8), 2)
            # Input indicator at BOTTOM (y+1 relative to origin, shown with down offset)
            pygame.draw.line(self.screen, CYAN, (cx - 14, cy + 3), (cx - 10, cy + 3), 2)
            # Output indicators - both go right, but second output goes UP (north)
            pygame.draw.line(self.screen, ORANGE, (cx + 10, cy + 3), (cx + 14, cy + 3), 2)
            pygame.draw.line(self.screen, ORANGE, (cx + 10, cy - 3), (cx + 14, cy - 3), 2)
            # Up arrow showing second output goes to y-1 (north)
            pygame.draw.polygon(self.screen, ORANGE, [
                (cx + 5, cy - 8), (cx + 2, cy - 4), (cx + 8, cy - 4)
            ])
            # "M" marker for mirrored
            text = self.font_small.render("M", True, WHITE)
            self.screen.blit(text, (cx - 3, cy - 3))

        elif bt == BuildingType.HALF_CUTTER:
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            text = self.font_small.render("HC", True, WHITE)
            self.screen.blit(text, (cx - 7, cy - 5))

        elif bt in (BuildingType.ROTATOR_CW, BuildingType.ROTATOR_CCW, BuildingType.ROTATOR_180):
            pygame.draw.circle(self.screen, WHITE, (cx, cy), 10, 2)
            if bt == BuildingType.ROTATOR_CW:
                symbol = "↻"
            elif bt == BuildingType.ROTATOR_CCW:
                symbol = "↺"
            else:
                symbol = "⟲"
            text = self.font.render(symbol, True, WHITE)
            self.screen.blit(text, (cx - 6, cy - 8))

        elif bt in (BuildingType.STACKER, BuildingType.STACKER_BENT, BuildingType.STACKER_BENT_MIRRORED):
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            # Stack symbol - two horizontal lines
            pygame.draw.line(self.screen, WHITE, (cx - 5, cy - 3), (cx + 5, cy - 3), 2)
            pygame.draw.line(self.screen, WHITE, (cx - 5, cy + 3), (cx + 5, cy + 3), 2)

        elif bt == BuildingType.UNSTACKER:
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            text = self.font_small.render("US", True, WHITE)
            self.screen.blit(text, (cx - 7, cy - 5))

        elif bt == BuildingType.SWAPPER:
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            # X symbol for swap
            pygame.draw.line(self.screen, WHITE, (cx - 5, cy - 5), (cx + 5, cy + 5), 2)
            pygame.draw.line(self.screen, WHITE, (cx + 5, cy - 5), (cx - 5, cy + 5), 2)

        elif bt in (BuildingType.SPLITTER, BuildingType.SPLITTER_LEFT, BuildingType.SPLITTER_RIGHT):
            pygame.draw.polygon(self.screen, WHITE, [(cx - 8, cy - 8), (cx + 8, cy), (cx - 8, cy + 8)], 2)

        elif bt == BuildingType.MERGER:
            pygame.draw.polygon(self.screen, WHITE, [(cx + 8, cy - 8), (cx - 8, cy), (cx + 8, cy + 8)], 2)

        elif bt in (BuildingType.PAINTER, BuildingType.PAINTER_MIRRORED):
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            pygame.draw.circle(self.screen, CYAN, (cx, cy), 5)
            if bt == BuildingType.PAINTER_MIRRORED:
                # Add M marker for mirrored
                text = self.font_small.render("M", True, WHITE)
                self.screen.blit(text, (cx + 4, cy - 10))

        else:
            # Default: just draw first letter
            text = self.font.render(bt.name[0], True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 7))
    
    def draw_sidebar(self):
        """Draw building palette and info."""
        x = self.grid_offset_x + self.grid_width * self.cell_size + self.grid_sidebar_gap + 10
        y = self.toolbar_height + 10

        # Foundation selector
        found_label = self.font.render("Foundation (F):", True, WHITE)
        self.screen.blit(found_label, (x, y))
        y += 18

        # Foundation buttons (4 per row)
        btn_width = 45
        cols = 4
        btn_x = x
        start_y = y
        for i, name in enumerate(self.foundation_names):
            row = i // cols
            col = i % cols
            btn_x = x + col * (btn_width + 5)
            btn_y = start_y + row * 26
            color = GREEN if name == self.current_foundation_name else LIGHT_GRAY
            pygame.draw.rect(self.screen, color, (btn_x, btn_y, btn_width, 22))
            pygame.draw.rect(self.screen, WHITE, (btn_x, btn_y, btn_width, 22), 1)
            text = self.font_small.render(name, True, BLACK)
            self.screen.blit(text, (btn_x + 3, btn_y + 5))
        num_rows = (len(self.foundation_names) + cols - 1) // cols
        y = start_y + num_rows * 26 + 5

        # Floor buttons
        floor_label = self.font.render("Floor:", True, WHITE)
        self.screen.blit(floor_label, (x, y))

        btn_width = 40
        btn_y = y + 20
        for f in range(self.num_floors):
            btn_x = x + f * (btn_width + 5)
            color = GREEN if f == self.current_floor else LIGHT_GRAY
            pygame.draw.rect(self.screen, color, (btn_x, btn_y, btn_width, 25))
            pygame.draw.rect(self.screen, WHITE, (btn_x, btn_y, btn_width, 25), 1)
            text = self.font.render(str(f), True, BLACK)
            self.screen.blit(text, (btn_x + 15, btn_y + 5))

        y += 55

        # Building palette title
        title = self.font_large.render("Buildings", True, WHITE)
        self.screen.blit(title, (x, y))
        # Show current foundation info
        spec = FOUNDATION_SPECS[self.current_foundation_name]
        info = self.font_small.render(f"Grid: {spec.grid_width}x{spec.grid_height}", True, LIGHT_GRAY)
        self.screen.blit(info, (x + 90, y + 4))
        y += 25

        # Building list with scroll
        visible_start = self.palette_scroll
        visible_end = min(visible_start + self.palette_visible_count, len(self.building_palette))

        # Scroll indicator
        if self.palette_scroll > 0:
            text = self.font_small.render("▲ scroll up", True, LIGHT_GRAY)
            self.screen.blit(text, (x, y - 12))

        for display_idx, palette_idx in enumerate(range(visible_start, visible_end)):
            bt = self.building_palette[palette_idx]

            # Determine color based on selection and hover
            if bt == self.selected_building:
                color = GREEN
            elif palette_idx == self.hovered_palette_idx:
                color = (180, 180, 180)  # Highlight on hover
            else:
                color = LIGHT_GRAY

            btn_rect = pygame.Rect(x, y + display_idx * 25, self.sidebar_width - 30, 22)
            pygame.draw.rect(self.screen, color, btn_rect)

            # Border for selected
            if bt == self.selected_building:
                pygame.draw.rect(self.screen, WHITE, btn_rect, 2)

            name = bt.name.replace("_", " ").title()
            if bt in MACHINE_THROUGHPUT:
                name += f" ({MACHINE_THROUGHPUT[bt]:.0f}/m)"

            text = self.font_small.render(name[:28], True, BLACK)
            self.screen.blit(text, (x + 3, y + display_idx * 25 + 4))

        # Scroll indicator
        if visible_end < len(self.building_palette):
            text = self.font_small.render("▼ scroll down", True, LIGHT_GRAY)
            self.screen.blit(text, (x, y + self.palette_visible_count * 25 + 2))

        # Controls help
        y += self.palette_visible_count * 25 + 25
        controls = [
            "─── CONTROLS ───",
            "LClick: Place/Select",
            "RClick: Rotate",
            "1-4: Mode select",
            "R: Rotate",
            "S: Cycle shape",
            "F: Next foundation",
            "V: Shapes " + ("ON" if self.show_shapes else "OFF"),
            "P: Ports " + ("ON" if self.show_ports else "OFF"),
            "─── TESTS ───",
            "T: Toggle tests",
            ",/.: Prev/Next test",
            "Esc: Exit tests",
            "─── SAMPLES ───",
            f"Slot: {self.current_sample_slot}",
            "F5: Save  F6: Load",
            "F7/F8: Prev/Next slot",
            "Ctrl+0-9: Quick load",
            "Shift+0-9: Quick save",
            "─── OTHER ───",
            "Space: Simulate",
            "C: Clear all",
        ]
        for line in controls:
            text = self.font_small.render(line, True, LIGHT_GRAY)
            self.screen.blit(text, (x, y))
            y += 14
    
    def draw_info_panel(self):
        """Draw simulation results."""
        y = self.toolbar_height + self.grid_height * self.cell_size + 10
        x = 10

        pygame.draw.rect(self.screen, (30, 30, 30), (0, y - 5, self.screen_width, 200))

        # Show sample save/load message (fades after 3 seconds)
        if self.sample_message and pygame.time.get_ticks() - self.sample_message_time < 3000:
            elapsed = pygame.time.get_ticks() - self.sample_message_time
            alpha = max(0, 255 - int(elapsed / 3000 * 255)) if elapsed > 2000 else 255
            msg_color = (100, 255, 100) if "Saved" in self.sample_message or "Loaded" in self.sample_message else (255, 200, 100)
            msg_text = self.font_large.render(self.sample_message, True, msg_color)
            # Position in top-right of grid area
            self.screen.blit(msg_text, (self.grid_offset_x + self.grid_width * self.cell_size - msg_text.get_width() - 5, self.toolbar_height + 5))

        # Show scenario info if in scenario mode
        if self.scenario_mode:
            scenario_title = self.font_large.render(
                f"Test {self.current_scenario_idx + 1}/{self.total_scenarios}: {self.scenario_name}",
                True, CYAN
            )
            self.screen.blit(scenario_title, (x, y))
            y += 22
            # Wrap description if too long
            desc = self.scenario_description
            max_chars = 100
            while desc:
                line = desc[:max_chars]
                if len(desc) > max_chars:
                    # Find last space to break at
                    space_idx = line.rfind(' ')
                    if space_idx > 0:
                        line = line[:space_idx]
                desc_text = self.font_small.render(line, True, LIGHT_GRAY)
                self.screen.blit(desc_text, (x, y))
                y += 14
                desc = desc[len(line):].lstrip()
            y += 5
        else:
            title = self.font_large.render("Simulation Results", True, WHITE)
            self.screen.blit(title, (x, y))
            y += 25

        if self.last_report:
            # Summary
            total_in = sum(i['throughput'] for i in self.sim.inputs)
            total_out = sum(o.get('throughput', 0) for o in self.sim.outputs)
            
            text = self.font.render(f"Input: {total_in:.0f}/min  Output: {total_out:.0f}/min  "
                                   f"Efficiency: {100*total_out/total_in if total_in else 0:.0f}%", 
                                   True, WHITE)
            self.screen.blit(text, (x, y))
            y += 20
            
            # Errors
            if self.sim.errors:
                for err in self.sim.errors[:3]:
                    text = self.font_small.render(f"❌ {err[:80]}", True, RED)
                    self.screen.blit(text, (x, y))
                    y += 15
            
            # Machine flows
            y += 10
            text = self.font.render("Machine Flows:", True, CYAN)
            self.screen.blit(text, (x, y))
            y += 18
            
            for origin, machine in list(self.sim.machines.items())[:4]:
                line = f"  {machine.building_type.name}@{origin}: "
                for p in machine.output_ports:
                    status = "🔴" if p['backed_up'] else "✓"
                    line += f"[{p['index']}]{p['shape'] or '---'}@{p['throughput']:.0f}{status} "
                text = self.font_small.render(line[:90], True, WHITE)
                self.screen.blit(text, (x, y))
                y += 14
        else:
            text = self.font.render("Press SPACE to simulate", True, LIGHT_GRAY)
            self.screen.blit(text, (x, y))


def main():
    # Start with 1x1 foundation (14x14 grid)
    spec = FOUNDATION_SPECS["1x1"]
    viewer = FlowViewer(spec.grid_width, spec.grid_height, spec.num_floors)
    pygame.display.set_caption("Flow Simulator - 1x1 Foundation")
    viewer.run()


if __name__ == "__main__":
    main()
