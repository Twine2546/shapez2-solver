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
from ..blueprint.encoder import BlueprintEncoder


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
    BuildingType.STACKER_BENT: (255, 167, 38),    # Light orange
    BuildingType.STACKER_BENT_MIRRORED: (255, 167, 38),
    BuildingType.UNSTACKER: (255, 87, 34),        # Deep orange
    BuildingType.PIN_PUSHER: (121, 85, 72),       # Brown
    BuildingType.TRASH: (96, 125, 139),           # Blue grey
    BuildingType.PAINTER: (244, 67, 54),          # Red

    # Belts
    BuildingType.BELT_FORWARD: (255, 235, 59),    # Yellow
    BuildingType.BELT_LEFT: (255, 193, 7),        # Amber
    BuildingType.BELT_RIGHT: (255, 213, 79),      # Light amber
    BuildingType.LIFT_UP: (233, 30, 99),          # Pink
    BuildingType.LIFT_DOWN: (240, 98, 146),       # Light pink

    # Belt Ports (teleporters)
    BuildingType.BELT_PORT_SENDER: (103, 58, 183),    # Deep purple
    BuildingType.BELT_PORT_RECEIVER: (149, 117, 205), # Light purple

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
    BuildingType.STACKER_BENT: "STb",
    BuildingType.STACKER_BENT_MIRRORED: "STb",
    BuildingType.UNSTACKER: "UST",
    BuildingType.PIN_PUSHER: "PIN",
    BuildingType.TRASH: "TRS",
    BuildingType.PAINTER: "PNT",
    BuildingType.BELT_FORWARD: ">",
    BuildingType.BELT_LEFT: "<",
    BuildingType.BELT_RIGHT: ">",
    BuildingType.LIFT_UP: "^",
    BuildingType.LIFT_DOWN: "v",
    BuildingType.BELT_PORT_SENDER: "S>",
    BuildingType.BELT_PORT_RECEIVER: ">R",
    BuildingType.SPLITTER: "SPL",
    BuildingType.MERGER: "MRG",
}

# Building descriptions for the legend (category, name, description, dimensions)
LEGEND_BUILDINGS = [
    # Operations
    ("Operations", BuildingType.ROTATOR_CW, "Rotator CW", "1x1", "Rotate 90° clockwise"),
    ("Operations", BuildingType.ROTATOR_CCW, "Rotator CCW", "1x1", "Rotate 90° counter-clockwise"),
    ("Operations", BuildingType.ROTATOR_180, "Rotator 180", "1x1", "Rotate 180°"),
    ("Operations", BuildingType.CUTTER, "Cutter", "1x2", "Cut shape in half vertically"),
    ("Operations", BuildingType.HALF_CUTTER, "Half Cutter", "1x1", "Cut and keep half"),
    ("Operations", BuildingType.STACKER, "Stacker (Straight)", "1x1 (2 floors)", "Stack shapes (6/belt)"),
    ("Operations", BuildingType.STACKER_BENT, "Stacker (Bent)", "1x1 (2 floors)", "Stack shapes (4/belt, faster)"),
    ("Operations", BuildingType.UNSTACKER, "Unstacker", "1x1 (2 floors)", "Unstack shapes"),
    ("Operations", BuildingType.SWAPPER, "Swapper", "1x1", "Swap quadrants"),
    ("Operations", BuildingType.PAINTER, "Painter", "1x2", "Paint shapes"),
    ("Operations", BuildingType.TRASH, "Trash", "1x1", "Delete shapes"),
    # Belts
    ("Belts", BuildingType.BELT_FORWARD, "Belt Forward", "1x1", "Move items forward"),
    ("Belts", BuildingType.BELT_LEFT, "Belt Left", "1x1", "Turn items left"),
    ("Belts", BuildingType.BELT_RIGHT, "Belt Right", "1x1", "Turn items right"),
    ("Belts", BuildingType.LIFT_UP, "Lift Up", "1x1", "Move items up a floor"),
    ("Belts", BuildingType.LIFT_DOWN, "Lift Down", "1x1", "Move items down a floor"),
    # Belt Ports
    ("Teleport", BuildingType.BELT_PORT_SENDER, "Belt Port Sender", "1x1", "Teleport send (max 4 tiles)"),
    ("Teleport", BuildingType.BELT_PORT_RECEIVER, "Belt Port Receiver", "1x1", "Teleport receive"),
    # Other
    ("Other", BuildingType.SPLITTER, "Splitter", "1x1", "Split into 2 outputs"),
    ("Other", BuildingType.MERGER, "Merger", "1x1", "Merge from 2 inputs"),
]

ROTATION_SYMBOLS = {
    Rotation.EAST: ">",
    Rotation.SOUTH: "v",
    Rotation.WEST: "<",
    Rotation.NORTH: "^",
}


class PygameLayoutViewer:
    """Pygame-based viewer for foundation layouts."""

    def __init__(self, evolution: FoundationEvolution, screen_width: int = 1280, screen_height: int = 800):
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

        # Legend panel width
        self.legend_width = 280

        # Dragging state
        self.dragging = False
        self.drag_start = (0, 0)

        # Blueprint string state
        self.blueprint_string = ""
        self.blueprint_copied = False
        self.copy_message_timer = 0

        self.screen = None
        self.font = None
        self.small_font = None
        self.tiny_font = None
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
        self.tiny_font = pygame.font.SysFont("monospace", 9)
        self.running = True

        # Generate initial blueprint
        self._update_blueprint()

        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                self._handle_event(event)

            # Update copy message timer
            if self.copy_message_timer > 0:
                self.copy_message_timer -= 1

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
                    self._update_blueprint()
            elif event.key == pygame.K_c:
                # Copy blueprint to clipboard
                self._copy_blueprint_to_clipboard()

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

        # Draw legend panel on the right
        self._draw_legend()

        # Draw UI overlay at bottom
        self._draw_ui()

        # Draw blueprint panel
        self._draw_blueprint_panel()

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
        controls = "Arrow/WASD=Pan | +/-/Scroll=Zoom | PgUp/PgDn=Floor | Tab=Next | C=Copy Blueprint | Esc=Close"
        text = self.small_font.render(controls, True, (120, 120, 130))
        self.screen.blit(text, (10, y))

        # Foundation info (top left)
        spec = self.config.spec
        info = f"{spec.name}: {spec.grid_width}x{spec.grid_height} grid"
        text = self.font.render(info, True, (150, 150, 160))
        self.screen.blit(text, (10, 10))

    def _draw_legend(self):
        """Draw the building legend panel on the right side."""
        panel_x = self.screen_width - self.legend_width
        panel_y = 0
        panel_height = self.screen_height - 90  # Leave room for bottom panel

        # Draw panel background
        pygame.draw.rect(self.screen, (35, 35, 45), (panel_x, panel_y, self.legend_width, panel_height))
        pygame.draw.line(self.screen, (60, 60, 70), (panel_x, panel_y), (panel_x, panel_y + panel_height), 2)

        y = panel_y + 10

        # Title
        title = self.font.render("Building Legend", True, (220, 220, 220))
        self.screen.blit(title, (panel_x + 10, y))
        y += 25

        # Draw separator
        pygame.draw.line(self.screen, (60, 60, 70), (panel_x + 10, y), (panel_x + self.legend_width - 10, y), 1)
        y += 8

        # Group by category
        current_category = None

        for category, building_type, name, dims, desc in LEGEND_BUILDINGS:
            # Check if we have room for more
            if y > panel_height - 30:
                break

            # Category header
            if category != current_category:
                current_category = category
                if y > panel_y + 50:  # Add spacing between categories
                    y += 5
                cat_text = self.small_font.render(f"-- {category} --", True, (140, 140, 160))
                self.screen.blit(cat_text, (panel_x + 10, y))
                y += 14

            # Draw color box
            color = BUILDING_COLORS.get(building_type, (128, 128, 128))
            box_size = 12
            pygame.draw.rect(self.screen, color, (panel_x + 12, y + 1, box_size, box_size))
            pygame.draw.rect(self.screen, (200, 200, 200), (panel_x + 12, y + 1, box_size, box_size), 1)

            # Draw symbol in box
            symbol = BUILDING_SYMBOLS.get(building_type, "?")[:2]
            sym_text = self.tiny_font.render(symbol, True, (0, 0, 0))
            sym_rect = sym_text.get_rect(center=(panel_x + 12 + box_size//2, y + 1 + box_size//2))
            self.screen.blit(sym_text, sym_rect)

            # Draw name and dimensions
            info_text = f"{name} ({dims})"
            text = self.tiny_font.render(info_text, True, (180, 180, 180))
            self.screen.blit(text, (panel_x + 30, y + 2))

            y += 16

    def _draw_blueprint_panel(self):
        """Draw the blueprint string panel."""
        if not self.blueprint_string:
            return

        # Show truncated blueprint at the very top of the legend panel
        panel_x = self.screen_width - self.legend_width
        panel_y = self.screen_height - 90 - 60  # Above bottom panel

        # Draw background
        pygame.draw.rect(self.screen, (45, 45, 55), (panel_x, panel_y, self.legend_width, 60))
        pygame.draw.line(self.screen, (60, 60, 70), (panel_x, panel_y), (panel_x + self.legend_width, panel_y), 1)

        y = panel_y + 5

        # Title
        title_text = "Blueprint (C to copy):"
        title = self.small_font.render(title_text, True, (180, 180, 180))
        self.screen.blit(title, (panel_x + 10, y))
        y += 14

        # Truncated blueprint string
        max_chars = 35
        display_str = self.blueprint_string[:max_chars]
        if len(self.blueprint_string) > max_chars:
            display_str += "..."

        bp_text = self.tiny_font.render(display_str, True, (120, 180, 120))
        self.screen.blit(bp_text, (panel_x + 10, y))
        y += 12

        # Show length
        len_text = f"Length: {len(self.blueprint_string)} chars"
        length = self.tiny_font.render(len_text, True, (120, 120, 130))
        self.screen.blit(length, (panel_x + 10, y))

        # Show copy confirmation if recently copied
        if self.copy_message_timer > 0:
            msg = self.small_font.render("Copied!", True, (100, 255, 100))
            self.screen.blit(msg, (panel_x + 150, y))

    def _update_blueprint(self):
        """Update the blueprint string for the current solution."""
        if not self.evolution.top_solutions or self.current_solution_idx >= len(self.evolution.top_solutions):
            self.blueprint_string = ""
            return

        candidate = self.evolution.top_solutions[self.current_solution_idx]
        if not candidate.buildings:
            self.blueprint_string = ""
            return

        try:
            encoder = BlueprintEncoder()

            for building in candidate.buildings:
                # Convert PlacedBuilding from foundation evolution to encoder format
                entry = {
                    "T": building.building_type.value,
                    "X": building.x,
                    "Y": building.y,
                    "L": building.floor,  # floor maps to layer
                    "R": building.rotation.value,
                }
                encoder.entries.append(entry)

            self.blueprint_string = encoder.encode()
        except Exception as e:
            self.blueprint_string = f"Error: {e}"

    def _copy_blueprint_to_clipboard(self):
        """Copy the blueprint string to the system clipboard."""
        if not self.blueprint_string:
            return

        try:
            # Try using pygame's scrap module
            pygame.scrap.init()
            pygame.scrap.put(pygame.SCRAP_TEXT, self.blueprint_string.encode('utf-8'))
            self.copy_message_timer = 60  # Show message for ~2 seconds
            self.blueprint_copied = True
        except Exception:
            # Fallback: try pyperclip if available
            try:
                import pyperclip
                pyperclip.copy(self.blueprint_string)
                self.copy_message_timer = 60
                self.blueprint_copied = True
            except ImportError:
                # Last resort: print to console
                print(f"\nBlueprint string:\n{self.blueprint_string}\n")
                self.copy_message_timer = 60


def show_layout_pygame(evolution: FoundationEvolution):
    """Convenience function to show the pygame layout viewer."""
    viewer = PygameLayoutViewer(evolution)
    viewer.show()
