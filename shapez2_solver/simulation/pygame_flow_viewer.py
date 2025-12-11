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
    BuildingType.ROTATOR_CW: (100, 200, 100),
    BuildingType.ROTATOR_CCW: (100, 200, 100),
    BuildingType.STACKER: (100, 100, 200),
    BuildingType.SPLITTER: (200, 200, 100),
    BuildingType.MERGER: (200, 100, 200),
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
        
        # Building palette
        self.building_palette = [
            BuildingType.BELT_FORWARD,
            BuildingType.BELT_LEFT,
            BuildingType.BELT_RIGHT,
            BuildingType.CUTTER,
            BuildingType.ROTATOR_CW,
            BuildingType.ROTATOR_CCW,
            BuildingType.SPLITTER,
            BuildingType.MERGER,
            BuildingType.STACKER,
        ]
        
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
            shapes = ["CuCuCuCu", "Cu------", "CuCu----", "----CuCu", "RuRuRuRu"]
            idx = shapes.index(self.input_shape) if self.input_shape in shapes else 0
            self.input_shape = shapes[(idx + 1) % len(shapes)]
    
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
                    
                    # Draw throughput if flowing
                    if cell.throughput > 0:
                        tp_text = self.font_small.render(f"{cell.throughput:.0f}", True, YELLOW)
                        self.screen.blit(tp_text, (rect.x + 2, rect.y + rect.height - 12))
                
                # Check for inputs/outputs
                for inp in self.sim.inputs:
                    if inp['position'] == pos:
                        pygame.draw.circle(self.screen, GREEN, rect.center, 8)
                        text = self.font_small.render("IN", True, BLACK)
                        self.screen.blit(text, (rect.centerx - 8, rect.centery - 5))
                
                for out in self.sim.outputs:
                    if out['position'] == pos:
                        pygame.draw.circle(self.screen, RED, rect.center, 8)
                        text = self.font_small.render("OUT", True, BLACK)
                        self.screen.blit(text, (rect.centerx - 10, rect.centery - 5))
        
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
        
        elif bt == BuildingType.CUTTER:
            pygame.draw.rect(self.screen, WHITE, (cx - 8, cy - 8, 16, 16), 2)
            text = self.font.render("C", True, WHITE)
            self.screen.blit(text, (cx - 4, cy - 7))
        
        elif bt in (BuildingType.ROTATOR_CW, BuildingType.ROTATOR_CCW):
            pygame.draw.circle(self.screen, WHITE, (cx, cy), 10, 2)
            symbol = "↻" if bt == BuildingType.ROTATOR_CW else "↺"
            text = self.font.render(symbol, True, WHITE)
            self.screen.blit(text, (cx - 6, cy - 8))
        
        elif bt == BuildingType.SPLITTER:
            pygame.draw.polygon(self.screen, WHITE, [(cx - 8, cy - 8), (cx + 8, cy), (cx - 8, cy + 8)], 2)
        
        elif bt == BuildingType.MERGER:
            pygame.draw.polygon(self.screen, WHITE, [(cx + 8, cy - 8), (cx - 8, cy), (cx + 8, cy + 8)], 2)
        
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
            "S: Cycle shape",
            "↑↓: Change floor",
            "Space: Simulate",
            "C: Clear all",
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
