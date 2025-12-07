"""Base operation interface."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
from ..shapes.shape import Shape, Color


class OperationType(Enum):
    """Types of operations available in Shapez 2."""
    HALF_DESTROYER = auto()
    CUTTER = auto()
    SWAPPER = auto()
    ROTATOR = auto()
    STACKER = auto()
    UNSTACKER = auto()
    PAINTER = auto()
    CRYSTAL_GENERATOR = auto()
    PIN_PUSHER = auto()


class Operation(ABC):
    """Abstract base class for all shape operations."""

    @property
    @abstractmethod
    def operation_type(self) -> OperationType:
        """Get the type of this operation."""
        pass

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        """Get the number of inputs this operation requires."""
        pass

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """Get the number of outputs this operation produces."""
        pass

    @property
    def input_types(self) -> List[str]:
        """Get the types of inputs (shape, color, etc.)."""
        return ["shape"] * self.num_inputs

    @property
    def output_types(self) -> List[str]:
        """Get the types of outputs."""
        return ["shape"] * self.num_outputs

    @abstractmethod
    def execute(
        self, *inputs: Union[Shape, Color, None]
    ) -> Tuple[Optional[Union[Shape, Color]], ...]:
        """
        Execute the operation on the given inputs.

        Args:
            *inputs: The input shapes/colors

        Returns:
            Tuple of output shapes/colors (may contain None for null outputs)
        """
        pass

    def validate_inputs(
        self, *inputs: Union[Shape, Color, None]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that the inputs are correct for this operation.

        Returns:
            A tuple of (is_valid, error_message)
        """
        if len(inputs) != self.num_inputs:
            return False, f"Expected {self.num_inputs} inputs, got {len(inputs)}"
        return True, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
