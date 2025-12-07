"""Predefined foundation layouts."""

from .foundation import Foundation, FoundationType, Port, PortDirection


def _create_foundation(
    ftype: FoundationType,
    width: int,
    height: int
) -> Foundation:
    """Create a foundation with automatic port generation."""
    foundation = Foundation(
        foundation_type=ftype,
        width=width,
        height=height,
    )

    # Generate ports for all edges based on dimensions
    for floor in range(3):
        # North and South edges (width ports each)
        for pos in range(width):
            foundation.add_port(Port(PortDirection.NORTH, floor, pos, is_input=True))
            foundation.add_port(Port(PortDirection.SOUTH, floor, pos, is_input=True))

        # East and West edges (height ports each)
        for pos in range(height):
            foundation.add_port(Port(PortDirection.EAST, floor, pos, is_input=True))
            foundation.add_port(Port(PortDirection.WEST, floor, pos, is_input=True))

    return foundation


# Predefined foundation layouts
FOUNDATION_LAYOUTS = {
    FoundationType.SIZE_1X1: lambda: _create_foundation(FoundationType.SIZE_1X1, 1, 1),
    FoundationType.SIZE_2X1: lambda: _create_foundation(FoundationType.SIZE_2X1, 2, 1),
    FoundationType.SIZE_3X1: lambda: _create_foundation(FoundationType.SIZE_3X1, 3, 1),
    FoundationType.SIZE_4X1: lambda: _create_foundation(FoundationType.SIZE_4X1, 4, 1),
    FoundationType.SIZE_2X2: lambda: _create_foundation(FoundationType.SIZE_2X2, 2, 2),
    FoundationType.SIZE_3X2: lambda: _create_foundation(FoundationType.SIZE_3X2, 3, 2),
    FoundationType.SIZE_4X2: lambda: _create_foundation(FoundationType.SIZE_4X2, 4, 2),
    FoundationType.SIZE_3X3: lambda: _create_foundation(FoundationType.SIZE_3X3, 3, 3),
}


def get_foundation(foundation_type: FoundationType) -> Foundation:
    """Get a foundation of the specified type."""
    if foundation_type not in FOUNDATION_LAYOUTS:
        raise ValueError(f"Unknown foundation type: {foundation_type}")
    return FOUNDATION_LAYOUTS[foundation_type]()


def list_available_foundations() -> list:
    """List all available foundation types."""
    return list(FOUNDATION_LAYOUTS.keys())
