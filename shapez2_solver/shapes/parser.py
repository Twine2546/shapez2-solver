"""Shape code parsing utilities."""

from typing import Optional
from .shape import Shape, ShapeLayer, ShapePart, ShapeType, Color


class ShapeCodeParser:
    """Parser for Shapez 2 shape codes."""

    @staticmethod
    def parse(code: str) -> Shape:
        """
        Parse a shape code string into a Shape object.

        Format: LayerCode:LayerCode:... (bottom to top)
        Layer format: PartCode PartCode ... (starting top-right, clockwise)
        Part format: ShapeTypeColor (e.g., "Cr" = red circle)

        Args:
            code: The shape code string

        Returns:
            The parsed Shape object

        Raises:
            ValueError: If the code is invalid
        """
        if not code or code.strip() == "":
            return Shape.empty()

        code = code.strip()

        # Handle empty shape markers
        if code in ("--", "-", ""):
            return Shape.empty()

        # Split into layers
        layer_codes = code.split(":")
        layers = []

        for layer_code in layer_codes:
            layer = ShapeCodeParser._parse_layer(layer_code)
            layers.append(layer)

        return Shape(layers)

    @staticmethod
    def _parse_layer(layer_code: str) -> ShapeLayer:
        """Parse a single layer code."""
        if len(layer_code) % 2 != 0:
            raise ValueError(f"Layer code must have even length: {layer_code}")

        parts = []
        for i in range(0, len(layer_code), 2):
            part_code = layer_code[i:i+2]
            part = ShapeCodeParser._parse_part(part_code)
            parts.append(part)

        return ShapeLayer(parts)

    @staticmethod
    def _parse_part(part_code: str) -> ShapePart:
        """Parse a single part code."""
        if len(part_code) != 2:
            raise ValueError(f"Part code must be 2 characters: {part_code}")

        type_char = part_code[0]
        color_char = part_code[1]

        try:
            shape_type = ShapeType.from_code(type_char)
        except ValueError:
            raise ValueError(f"Unknown shape type: {type_char}")

        try:
            color = Color.from_code(color_char)
        except ValueError:
            raise ValueError(f"Unknown color: {color_char}")

        return ShapePart(shape_type, color)

    @staticmethod
    def validate(code: str) -> tuple[bool, Optional[str]]:
        """
        Validate a shape code without fully parsing it.

        Returns:
            A tuple of (is_valid, error_message)
        """
        try:
            ShapeCodeParser.parse(code)
            return True, None
        except ValueError as e:
            return False, str(e)

    @staticmethod
    def normalize(code: str) -> str:
        """
        Normalize a shape code (parse and re-encode).

        This ensures consistent formatting.
        """
        shape = ShapeCodeParser.parse(code)
        return shape.to_code()
