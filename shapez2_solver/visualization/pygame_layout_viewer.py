"""
Pygame-based Layout Viewer for Shapez 2 Foundation Evolution Solutions.

Displays foundation layouts with buildings, belts, and ports using pygame.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..evolution.foundation_config import FoundationConfig, FoundationSpec, Side, PortType
from ..evolution.foundation_evolution import Candidate, PlacedBuilding, FoundationEvolution
from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS


# Color schemes for buildings (RGB tuples)
BUILDING_COLORS = {
    # Operations
    BuildingType.ROTATOR_CW: (76, 175, 80),       # Green
    BuildingType.ROTATOR_CCW: (139, 195, 74),     # Light green
    BuildingType.ROTATOR_180: (205, 220, 57),     # Lime
    BuildingType.CUTTER: (33, 150, 243),          # Blue
    BuildingType.CUTTER_MIRRORED: (3, 169, 244),  # Light blue
    BuildingType.HALF_CUTTER: (0, 188, 212),      # Cyan
    BuildingType.SWAPPER: (156, 39, 176),         # Purple
    BuildingType.STACKER: (255, 152, 0),          # Orange
    BuildingType.UNSTACKER: (255, 87, 34),        # Deep orange
    BuildingType.PIN_PUSHER: (121, 85, 72),       # Brown
    BuildingType.TRASH: (96, 125, 139),           # Blue grey

    # Belts
    BuildingType.BELT_FORWARD: (255, 235, 59),    # Yellow
    BuildingType.BELT_LEFT: (255, 193, 7),        # Amber
    BuildingType.BELT_RIGHT: (255, 213, 79),      # Light amber
    BuildingType.LIFT_UP: (233, 30, 99),          # Pink
    BuildingType.LIFT_DOWN: (240, 98, 146),       # Light pink

    # Other
    BuildingType.SPLITTER: (0, 150, 136),         # Teal
    BuildingType.MERGER: (0, 121, 107),           # Dark teal
}

BUILDING_SYMBOLS = {
    BuildingType.ROTATOR_CW: "R>",
    BuildingType.ROTATOR_CCW: "R<",
    BuildingType.ROTATOR_180: "R2",
    BuildingType.CUTTER: "CUT",
    BuildingType.CUTTER_MIRRORED: "CUm",
    BuildingType.HALF_CUTTER: "HCU",
    BuildingType.SWAPPER: "SWP",
    BuildingType.STACKER: "STK",
    BuildingType.UNSTACKER: "UST",
    BuildingType.PIN_PUSHER: "PIN",
    BuildingType.TRASH: "TRS",
    BuildingType.BELT_FORWARD: ">",
    BuildingType.BELT_LEFT: "<",
    BuildingType.BELT_RIGHT: ">",
    BuildingType.LIFT_UP: "^",
    BuildingType.LIFT_DOWN: "v",
    BuildingType.SPLITTER: "SPL",
    BuildingType.MERGER: "MRG",
}

ROTATION_SYMBOLS = {
    Rotation.EAST: ">",
    Rotation.SOUTH: "v",
    Rotation.WEST: "<",
    Rotation.NORTH: "^",
}


class PygameLayoutViewer:
    """Pygame-based viewer for foundation layouts."""

    def __init__(self, evolution: FoundationEvolution, screen_width: int = 1000, screen_height: int = 700):
        self.evolution = evolution
        self.config = evolution.config
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.current_floor = 0
        self.current_solution_idx = 0
        self.cell_size = 12  # Pixels per grid cell

        # Calculate view offset for centering/scrolling
        self.view_offset_x = 50
        self.view_offset_y = 50

        # Dragging state
        self.dragging = False
        self.drag_start = (0, 0)

        self.screen = None
        self.font = None
        self.small_font = None
        self.running = False

    def show(self):
        """Display the viewer window."""
        if not PYGAME_AVAILABLE:
            print("Error: pygame is required for the layout viewer")
            return

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Shapez 2 Layout Viewer")
        self.font = pygame.font.SysFont("monospace", 14)
        self.small_font = pygame.font.SysFont("monospace", 10)
        self.running = True

        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                self._handle_event(event)

            self._render()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def _handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.QUIT:
            self.running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                self.view_offset_y += 20
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                self.view_offset_y -= 20
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                self.view_offset_x += 20
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                self.view_offset_x -= 20
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.cell_size = min(30, self.cell_size + 2)
            elif event.key == pygame.K_MINUS:
                self.cell_size = max(4, self.cell_size - 2)
            elif event.key == pygame.K_PAGEUP:
                self.current_floor = min(self.config.spec.num_floors - 1, self.current_floor + 1)
            elif event.key == pygame.K_PAGEDOWN:
                self.current_floor = max(0, self.current_floor - 1)
            elif event.key == pygame.K_TAB:
                # Cycle through solutions
                if self.evolution.top_solutions:
                    self.current_solution_idx = (self.current_solution_idx + 1) % len(self.evolution.top_solutions)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.dragging = True
                self.drag_start = event.pos
            elif event.button == 4:  # Scroll up
                self.cell_size = min(30, self.cell_size + 1)
            elif event.button == 5:  # Scroll down
                self.cell_size = max(4, self.cell_size - 1)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                dx = event.pos[0] - self.drag_start[0]
                dy = event.pos[1] - self.drag_start[1]
                self.view_offset_x += dx
                self.view_offset_y += dy
                self.drag_start = event.pos

    def _render(self):
        """Render the layout."""
        self.screen.fill((30, 30, 40))

        # Draw grid
        self._draw_grid()

        # Draw ports
        self._draw_ports()

        # Draw buildings
        if self.evolution.top_solutions and self.current_solution_idx < len(self.evolution.top_solutions):
            candidate = self.evolution.top_solutions[self.current_solution_idx]
            self._draw_buildings(candidate)

        # Draw UI overlay
        self._draw_ui()

    def _draw_grid(self):
        """Draw the foundation grid."""
        grid_w = self.config.spec.grid_width
        grid_h = self.config.spec.grid_height
        cell = self.cell_size
        ox, oy = self.view_offset_x, self.view_offset_y

        # Draw grid cells
        for x in range(grid_w):
            for y in range(grid_h):
                x1 = ox + x * cell
                y1 = oy + y * cell

                # Check if visible
                if x1 + cell < 0 or x1 > self.screen_width or y1 + cell < 0 or y1 > self.screen_height - 100:
                    continue

                pygame.draw.rect(self.screen, (50, 50, 60), (x1, y1, cell, cell))
                pygame.draw.rect(self.screen, (60, 60, 70), (x1, y1, cell, cell), 1)

        # Draw 1x1 unit boundaries
        for ux in range(self.config.spec.units_x + 1):
            if ux == 0:
                x = 0
            else:
                x = 14 + (ux - 1) * 20
            if x <= grid_w:
                x1 = ox + x * cell
                pygame.draw.line(self.screen, (100, 100, 120), (x1, oy), (x1, oy + grid_h * cell), 2)

        for uy in range(self.config.spec.units_y + 1):
            if uy == 0:
                y = 0
            else:
                y = 14 + (uy - 1) * 20
            if y <= grid_h:
                y1 = oy + y * cell
                pygame.draw.line(self.screen, (100, 100, 120), (ox, y1), (ox + grid_w * cell, y1), 2)

    def _draw_ports(self):
        """Draw input/output ports."""
        floor = self.current_floor
        cell = self.cell_size
        ox, oy = self.view_offset_x, self.view_offset_y
        grid_w = self.config.spec.grid_width
        grid_h = self.config.spec.grid_height

        # Draw inputs (green)
        for side, pos, f, shape_code in self.config.get_all_inputs():
            if f != floor:
                continue
            gx, gy = self.config.spec.get_port_grid_position(side, pos)

            # Position outside the grid
            if side == Side.NORTH:
                gy = -1
            elif side == Side.SOUTH:
                gy = grid_h
            elif side == Side.WEST:
                gx = -1
            elif side == Side.EAST:
                gx = grid_w

            x1 = ox + gx * cell
            y1 = oy + gy * cell

            pygame.draw.rect(self.screen, (0, 150, 0), (x1, y1, cell, cell))
            pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, cell, cell), 2)

            if cell >= 10:
                text = self.small_font.render("I", True, (255, 255, 255))
                self.screen.blit(text, (x1 + cell//2 - 3, y1 + cell//2 - 5))

        # Draw outputs (red)
        for side, pos, f, shape_code in self.config.get_all_outputs():
            if f != floor:
                continue
            gx, gy = self.config.spec.get_port_grid_position(side, pos)

            # Position outside the grid
            if side == Side.NORTH:
                gy = -1
            elif side == Side.SOUTH:
                gy = grid_h
            elif side == Side.WEST:
                gx = -1
            elif side == Side.EAST:
                gx = grid_w

            x1 = ox + gx * cell
            y1 = oy + gy * cell

            pygame.draw.rect(self.screen, (150, 0, 0), (x1, y1, cell, cell))
            pygame.draw.rect(self.screen, (255, 0, 0), (x1, y1, cell, cell), 2)

            if cell >= 10:
                text = self.small_font.render("O", True, (255, 255, 255))
                self.screen.blit(text, (x1 + cell//2 - 3, y1 + cell//2 - 5))

    def _draw_buildings(self, candidate: Candidate):
        """Draw buildings from a candidate solution."""
        cell = self.cell_size
        ox, oy = self.view_offset_x, self.view_offset_y

        for building in candidate.buildings:
            if building.floor != self.current_floor:
                continue

            x = building.x
            y = building.y

            # Get building spec for size
            spec = BUILDING_SPECS.get(building.building_type)
            w = spec.width if spec else 1
            h = spec.height if spec else 1

            x1 = ox + x * cell
            y1 = oy + y * cell
            bw = w * cell
            bh = h * cell

            # Get color
            color = BUILDING_COLORS.get(building.building_type, (128, 128, 128))

            # Draw building rectangle
            pygame.draw.rect(self.screen, color, (x1 + 1, y1 + 1, bw - 2, bh - 2))
            pygame.draw.rect(self.screen, (255, 255, 255), (x1 + 1, y1 + 1, bw - 2, bh - 2), 1)

            # Draw symbol
            if cell >= 8:
                symbol = BUILDING_SYMBOLS.get(building.building_type, "?")

                # For belts, show rotation direction
                if building.building_type in [BuildingType.BELT_FORWARD, BuildingType.LIFT_UP, BuildingType.LIFT_DOWN]:
                    symbol = ROTATION_SYMBOLS.get(building.rotation, ">")

                text = self.small_font.render(symbol, True, (0, 0, 0))
                text_rect = text.get_rect(center=(x1 + bw//2, y1 + bh//2))
                self.screen.blit(text, text_rect)

    def _draw_ui(self):
        """Draw UI overlay."""
        # Bottom info panel
        panel_y = self.screen_height - 90
        pygame.draw.rect(self.screen, (40, 40, 50), (0, panel_y, self.screen_width, 90))
        pygame.draw.line(self.screen, (80, 80, 90), (0, panel_y), (self.screen_width, panel_y), 2)

        y = panel_y + 10

        # Solution info
        if self.evolution.top_solutions and self.current_solution_idx < len(self.evolution.top_solutions):
            candidate = self.evolution.top_solutions[self.current_solution_idx]

            info_text = f"Solution {self.current_solution_idx + 1}/{len(self.evolution.top_solutions)} | "
            info_text += f"Floor {self.current_floor}/{self.config.spec.num_floors - 1} | "
            info_text += f"Fitness: {candidate.fitness:.2f} | "
            info_text += f"Buildings: {len(candidate.buildings)}"

            text = self.font.render(info_text, True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 20

            # Building counts on this floor
            floor_buildings = [b for b in candidate.buildings if b.floor == self.current_floor]
            type_counts = {}
            for b in floor_buildings:
                name = b.building_type.name[:8]
                type_counts[name] = type_counts.get(name, 0) + 1

            if type_counts:
                counts_text = "This floor: " + ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
                text = self.font.render(counts_text, True, (150, 150, 150))
                self.screen.blit(text, (10, y))
        else:
            text = self.font.render("No solutions available", True, (200, 100, 100))
            self.screen.blit(text, (10, y))

        y += 25

        # Controls help
        controls = "Controls: Arrow/WASD=Pan | +/-/Scroll=Zoom | PgUp/PgDn=Floor | Tab=Next Solution | Esc=Close"
        text = self.small_font.render(controls, True, (120, 120, 130))
        self.screen.blit(text, (10, y))

        # Foundation info (top right)
        spec = self.config.spec
        info = f"{spec.name}: {spec.grid_width}x{spec.grid_height} grid"
        text = self.font.render(info, True, (150, 150, 160))
        self.screen.blit(text, (self.screen_width - text.get_width() - 10, 10))


def show_layout_pygame(evolution: FoundationEvolution):
    """Convenience function to show the pygame layout viewer."""
    viewer = PygameLayoutViewer(evolution)
    viewer.show()
