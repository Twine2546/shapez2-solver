"""Shape code encoding utilities."""

from .shape import Shape, ShapeLayer, ShapePart


class ShapeCodeEncoder:
    """Encoder for Shapez 2 shape codes."""

    @staticmethod
    def encode(shape: Shape) -> str:
        """
        Encode a Shape object into a shape code string.

        Args:
            shape: The Shape object to encode

        Returns:
            The encoded shape code string
        """
        if shape.is_empty():
            return "--"

        layer_codes = []
        for layer in shape.layers:
            layer_code = ShapeCodeEncoder._encode_layer(layer)
            layer_codes.append(layer_code)

        return ":".join(layer_codes)

    @staticmethod
    def _encode_layer(layer: ShapeLayer) -> str:
        """Encode a single layer."""
        return "".join(
            ShapeCodeEncoder._encode_part(part)
            for part in layer.parts
        )

    @staticmethod
    def _encode_part(part: ShapePart) -> str:
        """Encode a single part."""
        return f"{part.shape_type.value}{part.color.value}"

    @staticmethod
    def format_for_display(shape: Shape, multiline: bool = False) -> str:
        """
        Format a shape for human-readable display.

        Args:
            shape: The shape to format
            multiline: If True, show each layer on a separate line

        Returns:
            Formatted string representation
        """
        if shape.is_empty():
            return "[Empty Shape]"

        code = ShapeCodeEncoder.encode(shape)

        if not multiline:
            return code

        # Format with layers labeled
        lines = []
        for i, layer in enumerate(reversed(shape.layers)):
            layer_idx = len(shape.layers) - 1 - i
            layer_code = ShapeCodeEncoder._encode_layer(layer)
            lines.append(f"Layer {layer_idx}: {layer_code}")

        return "\n".join(lines)

    @staticmethod
    def to_visual_grid(shape: Shape) -> list[list[str]]:
        """
        Convert a shape to a 2D grid representation for visualization.

        For a 4-quadrant shape, returns a 2x2 grid per layer.
        Parts are arranged:
          [1] [0]   (top)
          [2] [3]   (bottom)

        Returns:
            List of grids, one per layer (bottom to top)
        """
        grids = []

        for layer in shape.layers:
            if len(layer.parts) == 4:
                grid = [
                    [layer.parts[1].to_code(), layer.parts[0].to_code()],
                    [layer.parts[2].to_code(), layer.parts[3].to_code()],
                ]
            else:
                # For hex shapes, just list them
                grid = [[p.to_code() for p in layer.parts]]
            grids.append(grid)

        return grids
