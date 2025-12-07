"""Core shape data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from copy import deepcopy


class ShapeType(Enum):
    """Shape part types available in Shapez 2."""
    CIRCLE = "C"
    SQUARE = "R"
    STAR = "S"
    DIAMOND = "W"
    HEXAGON = "H"  # Hexagonal scenario only
    FLOWER = "F"   # Hexagonal scenario only
    GEAR = "G"     # Hexagonal scenario only
    PIN = "P"      # No color
    CRYSTAL = "c"  # Special handling
    EMPTY = "-"    # No color

    @classmethod
    def from_code(cls, code: str) -> "ShapeType":
        """Parse a shape type from its code character."""
        for shape_type in cls:
            if shape_type.value == code:
                return shape_type
        raise ValueError(f"Unknown shape type code: {code}")

    @property
    def is_colorable(self) -> bool:
        """Check if this shape type can have a color."""
        return self not in (ShapeType.PIN, ShapeType.EMPTY)

    @property
    def is_regular(self) -> bool:
        """Check if this is a regular scenario shape."""
        return self in (
            ShapeType.CIRCLE,
            ShapeType.SQUARE,
            ShapeType.STAR,
            ShapeType.DIAMOND,
        )

    @property
    def is_hexagonal(self) -> bool:
        """Check if this is a hexagonal scenario shape."""
        return self in (ShapeType.HEXAGON, ShapeType.FLOWER, ShapeType.GEAR)


class Color(Enum):
    """Colors available in Shapez 2."""
    UNCOLORED = "u"
    RED = "r"
    GREEN = "g"
    BLUE = "b"
    CYAN = "c"
    MAGENTA = "m"
    YELLOW = "y"
    WHITE = "w"
    NONE = "-"  # For non-colorable shapes

    @classmethod
    def from_code(cls, code: str) -> "Color":
        """Parse a color from its code character."""
        for color in cls:
            if color.value == code:
                return color
        raise ValueError(f"Unknown color code: {code}")

    @classmethod
    def mix(cls, color1: "Color", color2: "Color") -> "Color":
        """Mix two colors together."""
        if color1 == cls.NONE or color2 == cls.NONE:
            return cls.NONE
        if color1 == cls.UNCOLORED:
            return color2
        if color2 == cls.UNCOLORED:
            return color1
        if color1 == color2:
            return color1

        # Primary color mixing
        primaries = {cls.RED, cls.GREEN, cls.BLUE}
        if color1 in primaries and color2 in primaries:
            mixed = {color1, color2}
            if mixed == {cls.RED, cls.GREEN}:
                return cls.YELLOW
            elif mixed == {cls.RED, cls.BLUE}:
                return cls.MAGENTA
            elif mixed == {cls.GREEN, cls.BLUE}:
                return cls.CYAN

        # Secondary + primary = white
        return cls.WHITE


@dataclass
class ShapePart:
    """A single part of a shape layer (one quadrant)."""
    shape_type: ShapeType
    color: Color

    def __post_init__(self):
        """Validate the shape part."""
        if not self.shape_type.is_colorable and self.color != Color.NONE:
            self.color = Color.NONE

    @classmethod
    def empty(cls) -> "ShapePart":
        """Create an empty shape part."""
        return cls(ShapeType.EMPTY, Color.NONE)

    @classmethod
    def from_code(cls, code: str) -> "ShapePart":
        """Parse a shape part from its two-character code."""
        if len(code) != 2:
            raise ValueError(f"Shape part code must be 2 characters: {code}")
        shape_type = ShapeType.from_code(code[0])
        color = Color.from_code(code[1])
        return cls(shape_type, color)

    def to_code(self) -> str:
        """Encode this shape part to its two-character code."""
        return f"{self.shape_type.value}{self.color.value}"

    def is_empty(self) -> bool:
        """Check if this part is empty."""
        return self.shape_type == ShapeType.EMPTY

    def is_pin(self) -> bool:
        """Check if this part is a pin."""
        return self.shape_type == ShapeType.PIN

    def is_crystal(self) -> bool:
        """Check if this part is a crystal."""
        return self.shape_type == ShapeType.CRYSTAL

    def copy(self) -> "ShapePart":
        """Create a copy of this shape part."""
        return ShapePart(self.shape_type, self.color)


@dataclass
class ShapeLayer:
    """A single layer of a shape (4 parts in regular, 6 in hexagonal)."""
    parts: List[ShapePart] = field(default_factory=list)

    def __post_init__(self):
        """Validate the layer."""
        if not self.parts:
            self.parts = [ShapePart.empty() for _ in range(4)]

    @classmethod
    def empty(cls, num_parts: int = 4) -> "ShapeLayer":
        """Create an empty layer."""
        return cls([ShapePart.empty() for _ in range(num_parts)])

    @classmethod
    def from_code(cls, code: str) -> "ShapeLayer":
        """Parse a layer from its code string."""
        if len(code) % 2 != 0:
            raise ValueError(f"Layer code must have even length: {code}")
        parts = []
        for i in range(0, len(code), 2):
            parts.append(ShapePart.from_code(code[i:i+2]))
        return cls(parts)

    def to_code(self) -> str:
        """Encode this layer to its code string."""
        return "".join(part.to_code() for part in self.parts)

    @property
    def num_parts(self) -> int:
        """Get the number of parts in this layer."""
        return len(self.parts)

    def is_empty(self) -> bool:
        """Check if all parts are empty."""
        return all(part.is_empty() for part in self.parts)

    def get_part(self, index: int) -> ShapePart:
        """Get a part by index (0 = top-right, going clockwise)."""
        return self.parts[index % self.num_parts]

    def set_part(self, index: int, part: ShapePart) -> None:
        """Set a part by index."""
        self.parts[index % self.num_parts] = part

    def rotate(self, steps: int = 1) -> "ShapeLayer":
        """Rotate the layer clockwise by the given number of steps."""
        n = self.num_parts
        steps = steps % n
        new_parts = self.parts[-steps:] + self.parts[:-steps]
        return ShapeLayer(new_parts)

    def copy(self) -> "ShapeLayer":
        """Create a deep copy of this layer."""
        return ShapeLayer([part.copy() for part in self.parts])


@dataclass
class Shape:
    """A complete shape with multiple layers."""
    layers: List[ShapeLayer] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "Shape":
        """Create an empty shape."""
        return cls([])

    @classmethod
    def from_code(cls, code: str) -> "Shape":
        """Parse a shape from its full code string."""
        if not code or code == "--":
            return cls.empty()
        layer_codes = code.split(":")
        layers = [ShapeLayer.from_code(lc) for lc in layer_codes]
        return cls(layers)

    def to_code(self) -> str:
        """Encode this shape to its full code string."""
        if not self.layers:
            return "--"
        return ":".join(layer.to_code() for layer in self.layers)

    @property
    def num_layers(self) -> int:
        """Get the number of layers."""
        return len(self.layers)

    def is_empty(self) -> bool:
        """Check if the shape is empty."""
        return not self.layers or all(layer.is_empty() for layer in self.layers)

    def get_layer(self, index: int) -> Optional[ShapeLayer]:
        """Get a layer by index (0 = bottom)."""
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None

    def add_layer(self, layer: ShapeLayer) -> None:
        """Add a layer to the top of the shape."""
        self.layers.append(layer)

    def pop_layer(self) -> Optional[ShapeLayer]:
        """Remove and return the top layer."""
        if self.layers:
            return self.layers.pop()
        return None

    def rotate(self, steps: int = 1) -> "Shape":
        """Rotate the entire shape clockwise by the given number of steps."""
        new_layers = [layer.rotate(steps) for layer in self.layers]
        return Shape(new_layers)

    def copy(self) -> "Shape":
        """Create a deep copy of this shape."""
        return Shape([layer.copy() for layer in self.layers])

    def __eq__(self, other: object) -> bool:
        """Check equality based on shape code."""
        if not isinstance(other, Shape):
            return False
        return self.to_code() == other.to_code()

    def __hash__(self) -> int:
        """Hash based on shape code."""
        return hash(self.to_code())

    def __repr__(self) -> str:
        """String representation."""
        return f"Shape({self.to_code()})"
