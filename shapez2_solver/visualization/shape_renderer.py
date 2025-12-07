"""Shape rendering utilities."""

from typing import Tuple, Dict, Optional
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..shapes.shape import Shape, ShapeLayer, ShapePart, ShapeType, Color


# Color mappings for rendering
COLOR_MAP: Dict[Color, Tuple[int, int, int]] = {
    Color.UNCOLORED: (180, 180, 180),
    Color.RED: (255, 50, 50),
    Color.GREEN: (50, 255, 50),
    Color.BLUE: (50, 50, 255),
    Color.CYAN: (50, 255, 255),
    Color.MAGENTA: (255, 50, 255),
    Color.YELLOW: (255, 255, 50),
    Color.WHITE: (255, 255, 255),
    Color.NONE: (100, 100, 100),
}


class ShapeRenderer:
    """Renders shapes to various outputs."""

    def __init__(self, cell_size: int = 40, padding: int = 5):
        """
        Initialize the renderer.

        Args:
            cell_size: Size of each shape quadrant in pixels
            padding: Padding between layers
        """
        self.cell_size = cell_size
        self.padding = padding

    def to_ascii(self, shape: Shape) -> str:
        """
        Render a shape to ASCII art.

        Args:
            shape: The shape to render

        Returns:
            ASCII representation of the shape
        """
        if shape.is_empty():
            return "[Empty]"

        lines = []
        lines.append(f"Shape: {shape.to_code()}")
        lines.append("-" * 20)

        # Render layers from top to bottom
        for layer_idx in range(shape.num_layers - 1, -1, -1):
            layer = shape.get_layer(layer_idx)
            if layer:
                lines.append(f"Layer {layer_idx}:")
                lines.extend(self._render_layer_ascii(layer))
                lines.append("")

        return "\n".join(lines)

    def _render_layer_ascii(self, layer: ShapeLayer) -> list:
        """Render a single layer to ASCII."""
        if layer.num_parts == 4:
            # Standard 4-quadrant layout
            #   [1] [0]
            #   [2] [3]
            top = f"  {self._part_to_ascii(layer.parts[1])}  {self._part_to_ascii(layer.parts[0])}"
            bottom = f"  {self._part_to_ascii(layer.parts[2])}  {self._part_to_ascii(layer.parts[3])}"
            return [top, bottom]
        else:
            # Hexagonal or other
            return ["  " + "  ".join(self._part_to_ascii(p) for p in layer.parts)]

    def _part_to_ascii(self, part: ShapePart) -> str:
        """Convert a part to ASCII representation."""
        if part.is_empty():
            return ".."
        return part.to_code()

    def get_shape_size(self, shape: Shape) -> Tuple[int, int]:
        """Get the pixel size needed to render a shape."""
        if shape.is_empty():
            return (self.cell_size * 2, self.cell_size * 2)

        # Width: 2 cells for quadrants
        width = self.cell_size * 2 + self.padding * 2

        # Height: 2 cells per layer plus spacing
        height = (self.cell_size * 2) * shape.num_layers + self.padding * (shape.num_layers + 1)

        return (width, height)

    def render_to_surface(self, shape: Shape, surface=None) -> Optional[object]:
        """
        Render a shape to a pygame surface.

        Args:
            shape: The shape to render
            surface: Optional existing surface to render to

        Returns:
            The pygame surface, or None if pygame is not available
        """
        if not PYGAME_AVAILABLE:
            return None

        width, height = self.get_shape_size(shape)

        if surface is None:
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            surface.fill((40, 40, 40))

        # Render layers from bottom to top
        y_offset = height - self.cell_size * 2 - self.padding

        for layer_idx in range(shape.num_layers):
            layer = shape.get_layer(layer_idx)
            if layer:
                self._render_layer(surface, layer, self.padding, y_offset)
            y_offset -= self.cell_size * 2 + self.padding

        return surface

    def _render_layer(self, surface, layer: ShapeLayer, x: int, y: int) -> None:
        """Render a single layer to a pygame surface."""
        if not PYGAME_AVAILABLE:
            return

        # Quadrant positions:
        #   [1] [0]
        #   [2] [3]
        positions = [
            (x + self.cell_size, y),                    # 0: top-right
            (x, y),                                      # 1: top-left
            (x, y + self.cell_size),                    # 2: bottom-left
            (x + self.cell_size, y + self.cell_size),   # 3: bottom-right
        ]

        for i, (px, py) in enumerate(positions):
            if i < len(layer.parts):
                self._render_part(surface, layer.parts[i], px, py)

    def _render_part(self, surface, part: ShapePart, x: int, y: int) -> None:
        """Render a single part to a pygame surface."""
        if not PYGAME_AVAILABLE or part.is_empty():
            return

        color = COLOR_MAP.get(part.color, (100, 100, 100))
        center = (x + self.cell_size // 2, y + self.cell_size // 2)
        radius = self.cell_size // 2 - 2

        if part.shape_type == ShapeType.CIRCLE:
            pygame.draw.circle(surface, color, center, radius)
            pygame.draw.circle(surface, (0, 0, 0), center, radius, 2)

        elif part.shape_type == ShapeType.SQUARE:
            rect = pygame.Rect(x + 2, y + 2, self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (0, 0, 0), rect, 2)

        elif part.shape_type == ShapeType.STAR:
            self._draw_star(surface, center, radius, color)

        elif part.shape_type == ShapeType.DIAMOND:
            self._draw_diamond(surface, center, radius, color)

        elif part.shape_type == ShapeType.PIN:
            # Draw pin as small circle
            pygame.draw.circle(surface, (150, 150, 150), center, radius // 3)
            pygame.draw.circle(surface, (0, 0, 0), center, radius // 3, 1)

        elif part.shape_type == ShapeType.CRYSTAL:
            self._draw_crystal(surface, center, radius, color)

    def _draw_star(self, surface, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
        """Draw a star shape."""
        if not PYGAME_AVAILABLE:
            return

        points = []
        for i in range(4):
            angle = i * math.pi / 2 - math.pi / 2
            # Outer point
            outer_x = center[0] + radius * math.cos(angle)
            outer_y = center[1] + radius * math.sin(angle)
            points.append((outer_x, outer_y))
            # Inner point
            inner_angle = angle + math.pi / 4
            inner_x = center[0] + radius * 0.4 * math.cos(inner_angle)
            inner_y = center[1] + radius * 0.4 * math.sin(inner_angle)
            points.append((inner_x, inner_y))

        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)

    def _draw_diamond(self, surface, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
        """Draw a diamond shape."""
        if not PYGAME_AVAILABLE:
            return

        points = [
            (center[0], center[1] - radius),
            (center[0] + radius, center[1]),
            (center[0], center[1] + radius),
            (center[0] - radius, center[1]),
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)

    def _draw_crystal(self, surface, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
        """Draw a crystal shape."""
        if not PYGAME_AVAILABLE:
            return

        # Draw as hexagon
        points = []
        for i in range(6):
            angle = i * math.pi / 3 - math.pi / 2
            px = center[0] + radius * 0.8 * math.cos(angle)
            py = center[1] + radius * 0.8 * math.sin(angle)
            points.append((px, py))

        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)
