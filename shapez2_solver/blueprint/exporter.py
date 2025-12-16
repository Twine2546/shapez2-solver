"""High-level blueprint export functionality."""

from typing import Optional

from ..simulator.design import Design
from .placer import GridPlacer
from .throughput_placer import ThroughputPlacer
from .router import ConveyorRouter
from .encoder import BlueprintEncoder


def export_blueprint(
    design: Design,
    grid_width: int = 32,
    grid_height: int = 32,
    include_belts: bool = True
) -> str:
    """
    Export a Design to a Shapez 2 blueprint code.

    Args:
        design: The design to export
        grid_width: Width of the placement grid
        grid_height: Height of the placement grid
        include_belts: Whether to include conveyor belt routing

    Returns:
        Blueprint code string (SHAPEZ2-1-xxxxx$)
    """
    # Step 1: Place buildings on the grid
    placer = GridPlacer(width=grid_width, height=grid_height)
    placements = placer.place_design(design)

    # Step 2: Create the encoder
    encoder = BlueprintEncoder()

    # Step 3: Add all buildings
    encoder.add_placements(placements)

    # Step 4: Route conveyor belts if requested
    if include_belts:
        router = ConveyorRouter(placer)
        routes = router.route_connections(design)
        encoder.add_routes(routes)

    # Step 5: Encode and return
    return encoder.encode()


def export_blueprint_verbose(
    design: Design,
    grid_width: int = 32,
    grid_height: int = 32,
    include_belts: bool = True
) -> dict:
    """
    Export a Design with detailed placement information.

    Returns a dict with:
        - blueprint_code: The encoded blueprint string
        - placements: Dict of node_id -> PlacedBuilding
        - routes: List of Route objects (if include_belts)
        - bounds: (min_x, min_y, max_x, max_y) of the layout
    """
    # Step 1: Place buildings on the grid
    placer = GridPlacer(width=grid_width, height=grid_height)
    placements = placer.place_design(design)

    # Step 2: Create the encoder
    encoder = BlueprintEncoder()

    # Step 3: Add all buildings
    encoder.add_placements(placements)

    routes = []
    if include_belts:
        router = ConveyorRouter(placer)
        routes = router.route_connections(design)
        encoder.add_routes(routes)

    return {
        "blueprint_code": encoder.encode(),
        "placements": placements,
        "routes": routes,
        "bounds": placer.get_bounds(),
        "entry_count": len(encoder.entries),
    }


def print_blueprint_layout(design: Design, grid_width: int = 32, grid_height: int = 32) -> str:
    """
    Print a visual representation of the blueprint layout.

    Returns a string with ASCII art of the layout.
    """
    placer = GridPlacer(width=grid_width, height=grid_height)
    placements = placer.place_design(design)

    router = ConveyorRouter(placer)
    routes = router.route_connections(design)

    bounds = placer.get_bounds()
    min_x, min_y, max_x, max_y = bounds

    # Create a grid for visualization
    vis_width = max_x - min_x + 3
    vis_height = max_y - min_y + 3
    grid = [[' ' for _ in range(vis_width)] for _ in range(vis_height)]

    # Building type abbreviations
    abbrevs = {
        "BeltDefaultForwardInternalVariant": "→",
        "BeltDefaultLeftInternalVariant": "↰",
        "BeltDefaultRightInternalVariant": "↱",
        "RotatorClockwiseInternalVariant": "↻",
        "RotatorCounterClockwiseInternalVariant": "↺",
        "Rotator180InternalVariant": "↔",
        "HalfCutterInternalVariant": "½",
        "CutterInternalVariant": "✂",
        "SwapperInternalVariant": "⇄",
        "StackerInternalVariant": "⊞",
        "UnstackerInternalVariant": "⊟",
        "PainterInternalVariant": "🎨",
        "TrashInternalVariant": "🗑",
    }

    # Place buildings
    for node_id, placement in placements.items():
        x = placement.x - min_x + 1
        y = placement.y - min_y + 1
        if 0 <= x < vis_width and 0 <= y < vis_height:
            symbol = abbrevs.get(placement.building_type.value, "?")
            grid[y][x] = symbol

    # Place belts
    for route in routes:
        for segment in route.segments:
            x = segment.x - min_x + 1
            y = segment.y - min_y + 1
            if 0 <= x < vis_width and 0 <= y < vis_height:
                if grid[y][x] == ' ':
                    symbol = abbrevs.get(segment.belt_type.value, "-")
                    grid[y][x] = symbol

    # Build output string
    lines = []
    lines.append("Blueprint Layout:")
    lines.append("=" * (vis_width + 2))
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("=" * (vis_width + 2))

    # Legend
    lines.append("\nLegend:")
    lines.append("  → = Belt Forward")
    lines.append("  ↻ = Rotator CW")
    lines.append("  ↺ = Rotator CCW")
    lines.append("  ✂ = Cutter")
    lines.append("  ⊞ = Stacker")
    lines.append("  ⊟ = Unstacker")

    return "\n".join(lines)


def export_throughput_blueprint(
    design: Design,
    grid_width: int = 64,
    grid_height: int = 64,
    num_floors: int = 3,
    include_belts: bool = True
) -> str:
    """
    Export a Design to a throughput-optimized Shapez 2 blueprint.

    This uses the ThroughputPlacer which creates parallel building groups
    with splitter/merger networks to match belt throughput.

    Args:
        design: The design to export
        grid_width: Width of the placement grid
        grid_height: Height of the placement grid
        num_floors: Number of floors available
        include_belts: Whether to include conveyor belt routing

    Returns:
        Blueprint code string (SHAPEZ2-1-xxxxx$)
    """
    # Step 1: Place buildings with throughput optimization
    placer = ThroughputPlacer(width=grid_width, height=grid_height, num_floors=num_floors)
    placements = placer.place_design(design)

    # Step 2: Create the encoder
    encoder = BlueprintEncoder()

    # Step 3: Add all buildings (including splitters/mergers)
    encoder.add_placements(placements)

    # Step 4: Route conveyor belts if requested
    if include_belts:
        router = ConveyorRouter(placer, num_floors=num_floors)
        routes = router.route_connections(design)
        encoder.add_routes(routes)

    # Step 5: Encode and return
    return encoder.encode()


def export_throughput_blueprint_verbose(
    design: Design,
    grid_width: int = 64,
    grid_height: int = 64,
    num_floors: int = 3,
    include_belts: bool = True
) -> dict:
    """
    Export a throughput-optimized blueprint with detailed information.

    Returns a dict with:
        - blueprint_code: The encoded blueprint string
        - placements: Dict of node_id -> PlacedBuilding
        - routes: List of Route objects (if include_belts)
        - bounds: (min_x, min_y, max_x, max_y) of the layout
        - throughput_summary: Summary of parallel building groups
        - grid_visualization: ASCII representation of the grid
    """
    # Step 1: Place buildings with throughput optimization
    placer = ThroughputPlacer(width=grid_width, height=grid_height, num_floors=num_floors)
    placements = placer.place_design(design)

    # Step 2: Create the encoder
    encoder = BlueprintEncoder()

    # Step 3: Add all buildings
    encoder.add_placements(placements)

    routes = []
    if include_belts:
        router = ConveyorRouter(placer, num_floors=num_floors)
        routes = router.route_connections(design)
        encoder.add_routes(routes)

    return {
        "blueprint_code": encoder.encode(),
        "placements": placements,
        "routes": routes,
        "bounds": placer.get_bounds(),
        "entry_count": len(encoder.entries),
        "throughput_summary": placer.print_throughput_summary(),
        "grid_visualization": placer.print_grid(),
    }
