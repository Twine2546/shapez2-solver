"""Crystal-related operations for shapes."""

from typing import Optional, Tuple, Union
from .base import Operation, OperationType
from ..shapes.shape import Shape, ShapeLayer, ShapePart, ShapeType, Color


class CrystalGeneratorOperation(Operation):
    """
    Generates crystals in empty quadrants and pins.

    Input: 1 shape, 1 color
    Output: 1 shape (with crystals added)

    Replaces empty parts and pins with crystals of the given color.
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.CRYSTAL_GENERATOR

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

        # Add crystals to empty parts and pins
        new_layers = []
        for layer in shape.layers:
            new_parts = []
            for part in layer.parts:
                if part.is_empty() or part.is_pin():
                    # Replace with crystal
                    new_parts.append(ShapePart(ShapeType.CRYSTAL, color))
                else:
                    new_parts.append(part.copy())
            new_layers.append(ShapeLayer(new_parts))

        return (Shape(new_layers),)
