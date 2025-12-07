"""Cutting operations for shapes."""

from typing import Optional, Tuple
from .base import Operation, OperationType
from ..shapes.shape import Shape, ShapeLayer, ShapePart, ShapeType
from ..shapes.validator import ShapeValidator


class HalfDestroyerOperation(Operation):
    """
    Destroys the west half of shapes.

    Input: 1 shape
    Output: 1 shape (east half only)
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.HALF_DESTROYER

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
        new_layers = []

        for layer in shape.layers:
            # For 4-part shapes: keep parts 0 (NE) and 3 (SE), destroy 1 (NW) and 2 (SW)
            new_parts = []
            for i, part in enumerate(layer.parts):
                if i in (0, 3):  # East half
                    new_parts.append(part.copy())
                else:
                    new_parts.append(ShapePart.empty())
            new_layers.append(ShapeLayer(new_parts))

        result = Shape(new_layers)

        # Check for crystal shattering
        result = self._handle_crystals(shape, result, destroyed_indices={1, 2})

        # Apply gravity
        result = ShapeValidator.apply_gravity(result)

        return (result if not result.is_empty() else None,)

    def _handle_crystals(
        self, original: Shape, result: Shape, destroyed_indices: set
    ) -> Shape:
        """Handle crystal shattering when cutting."""
        # For each layer, check if any crystals were connected across the cut
        for layer_idx, (orig_layer, res_layer) in enumerate(
            zip(original.layers, result.layers)
        ):
            # Find crystals in destroyed region
            destroyed_crystals = []
            for i in destroyed_indices:
                part = orig_layer.get_part(i)
                if part.is_crystal():
                    destroyed_crystals.append(i)

            # If we destroyed crystals, check if they were connected to kept crystals
            if destroyed_crystals:
                kept_crystals = []
                for i in range(len(res_layer.parts)):
                    if i not in destroyed_indices and orig_layer.get_part(i).is_crystal():
                        kept_crystals.append(i)

                # Check adjacency - if connected, shatter all connected crystals
                if self._are_crystals_connected(destroyed_crystals, kept_crystals):
                    # Shatter all connected crystals in the result
                    for i in kept_crystals:
                        res_layer.set_part(i, ShapePart.empty())

        return result

    def _are_crystals_connected(
        self, group1: list, group2: list
    ) -> bool:
        """Check if any crystals in group1 are adjacent to crystals in group2."""
        adjacency = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
        for i in group1:
            for adj in adjacency.get(i, []):
                if adj in group2:
                    return True
        return False


class CutOperation(Operation):
    """
    Cuts a shape in half vertically.

    Input: 1 shape
    Output: 2 shapes (east half, west half)
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.CUTTER

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
        east_layers = []
        west_layers = []

        for layer in shape.layers:
            # East half: parts 0 (NE) and 3 (SE)
            east_parts = []
            west_parts = []
            for i, part in enumerate(layer.parts):
                if i in (0, 3):
                    east_parts.append(part.copy())
                    west_parts.append(ShapePart.empty())
                else:
                    east_parts.append(ShapePart.empty())
                    west_parts.append(part.copy())

            east_layers.append(ShapeLayer(east_parts))
            west_layers.append(ShapeLayer(west_parts))

        east_shape = Shape(east_layers)
        west_shape = Shape(west_layers)

        # Handle crystal shattering for both halves
        east_shape = self._handle_crystals(shape, east_shape, {1, 2})
        west_shape = self._handle_crystals(shape, west_shape, {0, 3})

        # Apply gravity
        east_shape = ShapeValidator.apply_gravity(east_shape)
        west_shape = ShapeValidator.apply_gravity(west_shape)

        east_result = east_shape if not east_shape.is_empty() else None
        west_result = west_shape if not west_shape.is_empty() else None

        return (east_result, west_result)

    def _handle_crystals(
        self, original: Shape, result: Shape, destroyed_indices: set
    ) -> Shape:
        """Handle crystal shattering when cutting."""
        for layer_idx, (orig_layer, res_layer) in enumerate(
            zip(original.layers, result.layers)
        ):
            destroyed_crystals = []
            for i in destroyed_indices:
                part = orig_layer.get_part(i)
                if part.is_crystal():
                    destroyed_crystals.append(i)

            if destroyed_crystals:
                kept_crystals = []
                for i in range(len(res_layer.parts)):
                    if i not in destroyed_indices and orig_layer.get_part(i).is_crystal():
                        kept_crystals.append(i)

                if self._are_crystals_connected(destroyed_crystals, kept_crystals):
                    for i in kept_crystals:
                        res_layer.set_part(i, ShapePart.empty())

        return result

    def _are_crystals_connected(self, group1: list, group2: list) -> bool:
        adjacency = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
        for i in group1:
            for adj in adjacency.get(i, []):
                if adj in group2:
                    return True
        return False


class SwapperOperation(Operation):
    """
    Swaps the west halves of two shapes.

    Input: 2 shapes
    Output: 2 shapes (with swapped west halves)
    """

    @property
    def operation_type(self) -> OperationType:
        return OperationType.SWAPPER

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 2

    def execute(
        self, *inputs: Optional[Shape]
    ) -> Tuple[Optional[Shape], Optional[Shape]]:
        if len(inputs) < 2:
            return (None, None)

        shape1, shape2 = inputs[0], inputs[1]

        if shape1 is None or shape2 is None:
            return (None, None)

        if shape1.is_empty() or shape2.is_empty():
            return (None, None)

        # Ensure both shapes have the same number of layers
        max_layers = max(shape1.num_layers, shape2.num_layers)

        result1_layers = []
        result2_layers = []

        for i in range(max_layers):
            layer1 = shape1.get_layer(i) or ShapeLayer.empty()
            layer2 = shape2.get_layer(i) or ShapeLayer.empty()

            # Swap west halves (indices 1 and 2)
            new_layer1_parts = []
            new_layer2_parts = []

            for j in range(4):
                if j in (1, 2):  # West half
                    new_layer1_parts.append(layer2.get_part(j).copy())
                    new_layer2_parts.append(layer1.get_part(j).copy())
                else:  # East half
                    new_layer1_parts.append(layer1.get_part(j).copy())
                    new_layer2_parts.append(layer2.get_part(j).copy())

            result1_layers.append(ShapeLayer(new_layer1_parts))
            result2_layers.append(ShapeLayer(new_layer2_parts))

        result1 = Shape(result1_layers)
        result2 = Shape(result2_layers)

        # Apply gravity (crystals don't shatter in swapper since halves are placed side by side)
        result1 = ShapeValidator.apply_gravity(result1)
        result2 = ShapeValidator.apply_gravity(result2)

        return (
            result1 if not result1.is_empty() else None,
            result2 if not result2.is_empty() else None,
        )
