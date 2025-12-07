"""Pin pusher operation for shapes."""

from typing import Optional, Tuple
from .base import Operation, OperationType
from ..shapes.shape import Shape, ShapeLayer, ShapePart, ShapeType
from ..shapes.validator import ShapeValidator


class PinPusherOperation(Operation):
    """
    Pushes pins beneath the shape.

    Input: 1 shape
    Output: 1 shape (with pins pushed down)

    Pins are moved to form the bottom layer support structure.
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.PIN_PUSHER

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1

    def execute(
        self, *inputs: Optional[Shape]
    ) -> Tuple[Optional[Shape]]:
        if not inputs or inputs[0] is None or inputs[0].is_empty():
            return (None,)

        shape = inputs[0]

        # Find all pins in the shape
        pin_positions = set()
        for layer in shape.layers:
            for i, part in enumerate(layer.parts):
                if part.is_pin():
                    pin_positions.add(i)

        if not pin_positions:
            # No pins to push
            return (shape.copy(),)

        # Create new shape with pins pushed to bottom
        new_layers = []

        # Create bottom layer with pins
        pin_layer_parts = []
        for i in range(4):
            if i in pin_positions:
                pin_layer_parts.append(ShapePart(ShapeType.PIN, Color.NONE))
            else:
                pin_layer_parts.append(ShapePart.empty())
        new_layers.append(ShapeLayer(pin_layer_parts))

        # Add remaining layers without pins
        for layer in shape.layers:
            new_parts = []
            for i, part in enumerate(layer.parts):
                if part.is_pin():
                    new_parts.append(ShapePart.empty())
                else:
                    new_parts.append(part.copy())
            new_layer = ShapeLayer(new_parts)
            if not new_layer.is_empty():
                new_layers.append(new_layer)

        result = Shape(new_layers)

        # Apply gravity
        result = ShapeValidator.apply_gravity(result)

        return (result if not result.is_empty() else None,)
