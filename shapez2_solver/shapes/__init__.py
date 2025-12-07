"""Shape representation and parsing module."""

from .shape import Shape, ShapePart, ShapeLayer
from .parser import ShapeCodeParser
from .encoder import ShapeCodeEncoder
from .validator import ShapeValidator

__all__ = [
    "Shape",
    "ShapePart",
    "ShapeLayer",
    "ShapeCodeParser",
    "ShapeCodeEncoder",
    "ShapeValidator",
]
