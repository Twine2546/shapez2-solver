"""Painting operations for shapes."""

from typing import Optional, Tuple, Union
from .base import Operation, OperationType
from ..shapes.shape import Shape, ShapeLayer, ShapePart, Color


class PaintOperation(Operation):
    """
    Paints the topmost layer of a shape with a color.

    Input: 1 shape, 1 color
    Output: 1 shape (painted)
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.PAINTER

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def input_types(self) -> list:
        return ["shape", "color"]

    def execute(
        self, *inputs: Union[Shape, Color, None]
    ) -> Tuple[Optional[Shape]]:
        if len(inputs) < 2:
            return (None,)

        shape, color = inputs[0], inputs[1]

        if shape is None or color is None:
            return (None,)

        if not isinstance(shape, Shape) or not isinstance(color, Color):
            return (None,)

        if shape.is_empty():
            return (None,)

        # Paint the topmost layer
        result = shape.copy()
        if result.layers:
            top_layer = result.layers[-1]
            new_parts = []
            for part in top_layer.parts:
                if part.shape_type.is_colorable and not part.is_empty():
                    new_parts.append(ShapePart(part.shape_type, color))
                else:
                    new_parts.append(part.copy())
            result.layers[-1] = ShapeLayer(new_parts)

        return (result,)


class FullPaintOperation(Operation):
    """
    Paints all layers of a shape with a color.

    Input: 1 shape, 1 color
    Output: 1 shape (fully painted)
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.PAINTER

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def input_types(self) -> list:
        return ["shape", "color"]

    def execute(
        self, *inputs: Union[Shape, Color, None]
    ) -> Tuple[Optional[Shape]]:
        if len(inputs) < 2:
            return (None,)

        shape, color = inputs[0], inputs[1]

        if shape is None or color is None:
            return (None,)

        if not isinstance(shape, Shape) or not isinstance(color, Color):
            return (None,)

        if shape.is_empty():
            return (None,)

        # Paint all layers
        new_layers = []
        for layer in shape.layers:
            new_parts = []
            for part in layer.parts:
                if part.shape_type.is_colorable and not part.is_empty():
                    new_parts.append(ShapePart(part.shape_type, color))
                else:
                    new_parts.append(part.copy())
            new_layers.append(ShapeLayer(new_parts))

        return (Shape(new_layers),)
