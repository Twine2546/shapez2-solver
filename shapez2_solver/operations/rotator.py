"""Rotation operations for shapes."""

from typing import Optional, Tuple
from .base import Operation, OperationType
from ..shapes.shape import Shape


class RotateOperation(Operation):
    """
    Rotates a shape clockwise by 90 degrees (one quadrant position).

    Input: 1 shape
    Output: 1 shape (rotated)
    """

    def __init__(self, steps: int = 1):
        """
        Initialize the rotator.

        Args:
            steps: Number of 90-degree clockwise rotations (default: 1)
        """
        self._steps = steps % 4  # Normalize to 0-3

    @property
    def operation_type(self) -> OperationType:
        return OperationType.ROTATOR

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def steps(self) -> int:
        """Get the number of rotation steps."""
        return self._steps

    def execute(
        self, *inputs: Optional[Shape]
    ) -> Tuple[Optional[Shape]]:
        if not inputs or inputs[0] is None or inputs[0].is_empty():
            return (None,)

        shape = inputs[0]
        result = shape.rotate(self._steps)

        return (result,)

    def __repr__(self) -> str:
        return f"RotateOperation(steps={self._steps})"


class RotateCCWOperation(RotateOperation):
    """
    Rotates a shape counter-clockwise by 90 degrees.

    This is equivalent to rotating clockwise by 3 steps.
    """

    def __init__(self):
        super().__init__(steps=3)
