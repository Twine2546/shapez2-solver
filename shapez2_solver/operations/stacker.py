"""Stacking operations for shapes."""

from typing import Optional, Tuple
from .base import Operation, OperationType
from ..shapes.shape import Shape, ShapeLayer, ShapePart
from ..shapes.validator import ShapeValidator


class StackOperation(Operation):
    """
    Stacks two shapes on top of each other.

    Input: 2 shapes (bottom, top)
    Output: 1 shape (stacked)

    The top shape is placed on top of the bottom shape.
    If parts overlap, they merge into the same layer.
    If parts don't overlap, they stay at the same height.
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.STACKER

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1

    def execute(
        self, *inputs: Optional[Shape]
    ) -> Tuple[Optional[Shape]]:
        if len(inputs) < 2:
            return (None,)

        bottom, top = inputs[0], inputs[1]

        if bottom is None or top is None:
            return (None,)

        if bottom.is_empty() and top.is_empty():
            return (None,)

        if bottom.is_empty():
            return (top.copy(),)

        if top.is_empty():
            return (bottom.copy(),)

        # Stack the shapes
        result = self._stack_shapes(bottom, top)

        # Apply gravity
        result = ShapeValidator.apply_gravity(result)

        # Handle crystal collision
        result = self._handle_crystal_collision(result)

        return (result if not result.is_empty() else None,)

    def _stack_shapes(self, bottom: Shape, top: Shape) -> Shape:
        """Stack two shapes together."""
        result_layers = []

        # Start with all bottom layers
        for layer in bottom.layers:
            result_layers.append(layer.copy())

        # Add top layers
        for top_layer in top.layers:
            placed = False

            # Try to place in existing layers (for non-overlapping parts)
            for i, result_layer in enumerate(result_layers):
                if self._can_merge_layers(result_layer, top_layer):
                    result_layers[i] = self._merge_layers(result_layer, top_layer)
                    placed = True
                    break

            if not placed:
                # Add as new layer on top
                result_layers.append(top_layer.copy())

        return Shape(result_layers)

    def _can_merge_layers(self, layer1: ShapeLayer, layer2: ShapeLayer) -> bool:
        """Check if two layers can be merged (no overlapping non-empty parts)."""
        for i in range(len(layer1.parts)):
            if not layer1.parts[i].is_empty() and not layer2.parts[i].is_empty():
                return False
        return True

    def _merge_layers(self, layer1: ShapeLayer, layer2: ShapeLayer) -> ShapeLayer:
        """Merge two non-overlapping layers."""
        new_parts = []
        for i in range(len(layer1.parts)):
            if not layer1.parts[i].is_empty():
                new_parts.append(layer1.parts[i].copy())
            else:
                new_parts.append(layer2.parts[i].copy())
        return ShapeLayer(new_parts)

    def _handle_crystal_collision(self, shape: Shape) -> Shape:
        """Handle crystal collision/shattering during stacking."""
        # In Shapez 2, crystals can shatter when dropped onto each other
        # This is a simplified implementation
        return shape


class UnstackOperation(Operation):
    """
    Separates the top layer from a shape.

    Input: 1 shape
    Output: 2 shapes (top layer, remaining layers)
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.UNSTACKER

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 2

    def execute(
        self, *inputs: Optional[Shape]
    ) -> Tuple[Optional[Shape], Optional[Shape]]:
        if not inputs or inputs[0] is None or inputs[0].is_empty():
            return (None, None)

        shape = inputs[0]

        if shape.num_layers == 0:
            return (None, None)

        if shape.num_layers == 1:
            # Only one layer - it goes to top output, bottom is empty
            return (shape.copy(), None)

        # Separate top layer from rest
        top_layer = shape.layers[-1].copy()
        remaining_layers = [layer.copy() for layer in shape.layers[:-1]]

        top_shape = Shape([top_layer])
        remaining_shape = Shape(remaining_layers)

        # Apply gravity to remaining shape
        remaining_shape = ShapeValidator.apply_gravity(remaining_shape)

        return (
            top_shape if not top_shape.is_empty() else None,
            remaining_shape if not remaining_shape.is_empty() else None,
        )
