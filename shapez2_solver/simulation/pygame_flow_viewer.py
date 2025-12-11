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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, List, Optional, Tuple
from shapez2_solver.simulation.flow_simulator import FlowSimulator, MACHINE_THROUGHPUT, BELT_THROUGHPUT
from shapez2_solver.blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS


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
    BuildingType.UNSTACKER: (80, 80, 180),
    BuildingType.SWAPPER: (180, 180, 100),
    BuildingType.SPLITTER: (200, 200, 100),
    BuildingType.SPLITTER_LEFT: (200, 200, 100),
    BuildingType.SPLITTER_RIGHT: (200, 200, 100),
    BuildingType.MERGER: (200, 100, 200),
    BuildingType.PAINTER: (100, 180, 180),
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
        self.sidebar_width = 300
        self.toolbar_height = 60
        
        self.screen_width = self.grid_width * self.cell_size + self.sidebar_width
        self.screen_height = self.grid_height * self.cell_size + self.toolbar_height + 200  # Extra for info
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flow Simulator - Interactive")
        
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)
        
        # State
        self.current_floor = 0
        self.selected_building = BuildingType.BELT_FORWARD
        self.current_rotation = Rotation.EAST
        self.mode = "place"  # "place", "input", "output", "delete"
        self.input_shape = "CuCuCuCu"
        
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
            BuildingType.HALF_CUTTER,
            BuildingType.ROTATOR_CW,
            BuildingType.ROTATOR_CCW,
            BuildingType.ROTATOR_180,
            BuildingType.STACKER,
            BuildingType.UNSTACKER,
            BuildingType.SWAPPER,
            BuildingType.SPLITTER,
            BuildingType.MERGER,
            BuildingType.PAINTER,
        ]

        # Toggle for showing shapes on belts
        self.show_shapes = True
        
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
            
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
    
    def handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse click."""
        x, y = pos
        
        # Check if click is on grid
        grid_x = x // self.cell_size
        grid_y = (y - self.toolbar_height) // self.cell_size
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            if button == 1:  # Left click
                if self.mode == "place":
                    self.sim.place_building(
                        self.selected_building,
                        grid_x, grid_y, self.current_floor,
                        self.current_rotation
                    )
                elif self.mode == "input":
                    self.sim.set_input(grid_x, grid_y, self.current_floor, self.input_shape, 180.0)
                elif self.mode == "output":
                    self.sim.set_output(grid_x, grid_y, self.current_floor)
                elif self.mode == "delete":
                    # Remove building at position
                    pos_key = (grid_x, grid_y, self.current_floor)
                    if pos_key in self.sim.cells:
                        del self.sim.cells[pos_key]
                    if pos_key in self.sim.machines:
                        del self.sim.machines[pos_key]
                    if pos_key in self.sim.buildings:
                        del self.sim.buildings[pos_key]
                
                # Auto-simulate after changes
                self.last_report = self.sim.simulate()
            
            elif button == 3:  # Right click - rotate
                rotations = [Rotation.EAST, Rotation.SOUTH, Rotation.WEST, Rotation.NORTH]
                idx = rotations.index(self.current_rotation)
                self.current_rotation = rotations[(idx + 1) % 4]
        
        # Check sidebar clicks
        sidebar_x = self.grid_width * self.cell_size
        if x > sidebar_x:
            rel_y = y - self.toolbar_height
            
            # Building palette
            for i, bt in enumerate(self.building_palette):
                btn_y = 100 + i * 30
                if btn_y <= rel_y < btn_y + 25:
                    self.selected_building = bt
                    self.mode = "place"
                    break
    
    def handle_key(self, key: int):
        """Handle keyboard input."""
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
        elif key == pygame.K_PAGEUP:
            # Scroll building palette up
            current_idx = self.building_palette.index(self.selected_building)
            self.selected_building = self.building_palette[max(0, current_idx - 1)]
        elif key == pygame.K_PAGEDOWN:
            # Scroll building palette down
            current_idx = self.building_palette.index(self.selected_building)
            self.selected_building = self.building_palette[min(len(self.building_palette) - 1, current_idx + 1)]
    
    def draw(self):
        """Draw everything."""
        self.screen.fill(DARK_GRAY)
        
        # Draw toolbar
        self.draw_toolbar()
        
        # Draw grid
        self.draw_grid()
        
        # Draw sidebar
        self.draw_sidebar()
        
        # Draw info panel
        self.draw_info_panel()
    
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
        offset_y = self.toolbar_height

        # Draw cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                rect = pygame.Rect(
                    x * self.cell_size,
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

        # Draw flow paths (lines connecting traced paths)
        if hasattr(self.sim, 'traced_paths') and self.sim.traced_paths:
            for path in self.sim.traced_paths:
                if len(path) < 2:
                    continue
                # Only draw paths on current floor
                path_on_floor = [(p[0], p[1]) for p in path if p[2] == self.current_floor]
                if len(path_on_floor) >= 2:
                    points = [
                        (p[0] * self.cell_size + self.cell_size // 2,
                         offset_y + p[1] * self.cell_size + self.cell_size // 2)
                        for p in path_on_floor
                    ]
                    pygame.draw.lines(self.screen, (100, 255, 100, 128), False, points, 2)

        # Draw grid lines
        for x in range(self.grid_width + 1):
            pygame.draw.line(
                self.screen, GRAY,
                (x * self.cell_size, offset_y),
                (x * self.cell_size, offset_y + self.grid_height * self.cell_size)
            )
        for y in range(self.grid_height + 1):
            pygame.draw.line(
                self.screen, GRAY,
                (0, offset_y + y * self.cell_size),
                (self.grid_width * self.cell_size, offset_y + y * self.cell_size)
            )
    
    def draw_building_symbol(self, rect: pygame.Rect, bt: BuildingType, rotation: Rotation):
        """Draw a building's symbol."""
        cx, cy = rect.center

        if bt == BuildingType.BELT_FORWARD:
            # Arrow
            if rotation == Rotation.EAST:
                points = [(cx - 10, cy), (cx + 5, cy), (cx + 5, cy - 5), (cx + 12, cy), (cx + 5, cy + 5), (cx + 5, cy)]
            elif rotation == Rotation.WEST:
                points = [(cx + 10, cy), (cx - 5, cy), (cx - 5, cy - 5), (cx - 12, cy), (cx - 5, cy + 5), (cx - 5, cy)]
            elif rotation == Rotation.SOUTH:
                points = [(cx, cy - 10), (cx, cy + 5), (cx - 5, cy + 5), (cx, cy + 12), (cx + 5, cy + 5), (cx, cy + 5)]
            else:
                points = [(cx, cy + 10), (cx, cy - 5), (cx - 5, cy - 5), (cx, cy - 12), (cx + 5, cy - 5), (cx, cy - 5)]
            pygame.draw.polygon(self.screen, WHITE, points)

        elif bt == BuildingType.BELT_LEFT:
            # Curved arrow left
            pygame.draw.arc(self.screen, WHITE, (cx - 12, cy - 12, 24, 24), 0, 1.57, 2)
            text = self.font_small.render("L", True, WHITE)
            self.screen.blit(text, (cx - 3, cy - 5))

        elif bt == BuildingType.BELT_RIGHT:
            # Curved arrow right
            pygame.draw.arc(self.screen, WHITE, (cx - 12, cy - 12, 24, 24), 1.57, 3.14, 2)
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

        elif bt in (BuildingType.CUTTER, BuildingType.CUTTER_MIRRORED):
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            text = self.font.render("C", True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 7))

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

        elif bt in (BuildingType.STACKER, BuildingType.STACKER_BENT):
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

        elif bt == BuildingType.PAINTER:
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            pygame.draw.circle(self.screen, CYAN, (cx, cy), 5)

        else:
            # Default: just draw first letter
            text = self.font.render(bt.name[0], True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 7))
    
    def draw_sidebar(self):
        """Draw building palette and info."""
        x = self.grid_width * self.cell_size + 10
        y = self.toolbar_height + 10
        
        # Title
        title = self.font_large.render("Buildings", True, WHITE)
        self.screen.blit(title, (x, y))
        y += 30
        
        # Building list
        for i, bt in enumerate(self.building_palette):
            color = GREEN if bt == self.selected_building else LIGHT_GRAY
            pygame.draw.rect(self.screen, color, (x, y + i * 30, self.sidebar_width - 20, 25))
            
            name = bt.name.replace("_", " ").title()
            if bt in MACHINE_THROUGHPUT:
                name += f" ({MACHINE_THROUGHPUT[bt]:.0f}/min)"
            
            text = self.font.render(name[:25], True, BLACK)
            self.screen.blit(text, (x + 5, y + i * 30 + 5))
        
        # Controls help
        y += len(self.building_palette) * 30 + 20
        controls = [
            "─── CONTROLS ───",
            "Left Click: Place",
            "Right Click: Rotate",
            "1-4: Mode select",
            "R: Rotate",
            "S: Cycle input shape",
            "V: Toggle shape view",
            "↑↓: Change floor",
            "PgUp/Dn: Scroll bldgs",
            "Space: Simulate",
            "C: Clear all",
            "",
            f"Shapes: {'ON' if self.show_shapes else 'OFF'}",
        ]
        for line in controls:
            text = self.font_small.render(line, True, LIGHT_GRAY)
            self.screen.blit(text, (x, y))
            y += 15
    
    def draw_info_panel(self):
        """Draw simulation results."""
        y = self.toolbar_height + self.grid_height * self.cell_size + 10
        x = 10
        
        pygame.draw.rect(self.screen, (30, 30, 30), (0, y - 5, self.screen_width, 200))
        
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
    viewer = FlowViewer(14, 14, 4)
    viewer.run()


if __name__ == "__main__":
    main()
