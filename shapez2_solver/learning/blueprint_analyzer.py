"""
Blueprint analyzer for extracting features from Shapez 2 blueprints.

Parses blueprint codes and extracts building placements, connections,
and routing information for ML training.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict

from ..blueprint.encoder import BlueprintEncoder
from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS


@dataclass
class BuildingPlacement:
    """A building placement from a decoded blueprint."""
    building_type: str  # Raw type string from blueprint
    x: int
    y: int
    floor: int  # Layer in blueprint
    rotation: int  # 0=E, 1=S, 2=W, 3=N

    @property
    def position(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.floor)

    def get_building_enum(self) -> Optional[BuildingType]:
        """Try to map to BuildingType enum."""
        for bt in BuildingType:
            if bt.value == self.building_type:
                return bt
        return None


@dataclass
class BlueprintAnalysis:
    """Analysis results from a blueprint."""
    # Basic info
    blueprint_code: str
    version: int
    total_entries: int

    # Building counts by category
    belt_count: int = 0
    machine_count: int = 0
    splitter_count: int = 0
    merger_count: int = 0
    lift_count: int = 0

    # Detailed placements
    belts: List[BuildingPlacement] = field(default_factory=list)
    machines: List[BuildingPlacement] = field(default_factory=list)
    splitters: List[BuildingPlacement] = field(default_factory=list)
    mergers: List[BuildingPlacement] = field(default_factory=list)
    lifts: List[BuildingPlacement] = field(default_factory=list)
    other: List[BuildingPlacement] = field(default_factory=list)

    # Spatial info
    min_x: int = 0
    max_x: int = 0
    min_y: int = 0
    max_y: int = 0
    min_floor: int = 0
    max_floor: int = 0

    # Derived metrics
    grid_width: int = 0
    grid_height: int = 0
    num_floors: int = 1

    # Building type counts
    building_counts: Dict[str, int] = field(default_factory=dict)

    # Errors during parsing
    parse_errors: List[str] = field(default_factory=list)

    @property
    def complexity_score(self) -> float:
        """Estimate blueprint complexity for routing difficulty."""
        score = 0.0
        # More machines = more routing needed
        score += self.machine_count * 2.0
        # Splitters/mergers add routing complexity
        score += (self.splitter_count + self.merger_count) * 1.5
        # Multi-floor adds vertical routing complexity
        score += (self.num_floors - 1) * 3.0
        # Belt density suggests more constrained routing
        if self.grid_width > 0 and self.grid_height > 0:
            density = self.belt_count / (self.grid_width * self.grid_height * self.num_floors)
            score += density * 10.0
        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'version': self.version,
            'total_entries': self.total_entries,
            'belt_count': self.belt_count,
            'machine_count': self.machine_count,
            'splitter_count': self.splitter_count,
            'merger_count': self.merger_count,
            'lift_count': self.lift_count,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'num_floors': self.num_floors,
            'complexity_score': self.complexity_score,
            'building_counts': self.building_counts,
            'parse_errors': self.parse_errors,
        }


# Building type categories - use prefix/pattern matching
def is_belt_type(building_type: str) -> bool:
    """Check if building type is a belt."""
    return (building_type.startswith("BeltDefault") and
            not building_type.startswith("BeltPort"))

def is_splitter_type(building_type: str) -> bool:
    """Check if building type is a splitter."""
    return building_type.startswith("Splitter")

def is_merger_type(building_type: str) -> bool:
    """Check if building type is a merger."""
    return building_type.startswith("Merger")

def is_lift_type(building_type: str) -> bool:
    """Check if building type is a lift."""
    return building_type.startswith("Lift")

def is_belt_port_type(building_type: str) -> bool:
    """Check if building type is a belt port (teleporter)."""
    return building_type.startswith("BeltPort")

def is_pipe_type(building_type: str) -> bool:
    """Check if building type is a fluid pipe."""
    return building_type.startswith("Pipe") or building_type.startswith("FluidPort")

MACHINE_TYPES = {
    # Rotators
    "RotatorOneQuadInternalVariant",
    "RotatorOneQuadCCWInternalVariant",
    "RotatorHalfInternalVariant",
    # Cutters
    "CutterHalfInternalVariant",
    "CutterDefaultInternalVariant",
    "CutterDefaultInternalVariantMirrored",
    # Swapper
    "HalvesSwapperDefaultInternalVariant",
    # Stackers
    "StackerStraightInternalVariant",
    "StackerDefaultInternalVariant",
    "StackerDefaultInternalVariantMirrored",
    # Unstacker
    "UnstackerDefaultInternalVariant",
    # Painters
    "PainterInternalVariant",
    "PainterDefaultInternalVariant",
    "PainterDefaultInternalVariantMirrored",
    # Other
    "PinPusherInternalVariant",
    "TrashInternalVariant",
    "TrashDefaultInternalVariant",
}


def analyze_blueprint(blueprint_code: str) -> BlueprintAnalysis:
    """
    Analyze a blueprint code and extract features.

    Args:
        blueprint_code: The SHAPEZ2-X-... blueprint code

    Returns:
        BlueprintAnalysis with extracted features
    """
    analysis = BlueprintAnalysis(
        blueprint_code=blueprint_code,
        version=0,
        total_entries=0,
    )

    try:
        decoded = BlueprintEncoder.decode(blueprint_code)
    except Exception as e:
        analysis.parse_errors.append(f"Failed to decode blueprint: {e}")
        return analysis

    analysis.version = decoded.get('V', 0)

    bp = decoded.get('BP', {})
    entries = bp.get('Entries', [])
    analysis.total_entries = len(entries)

    # Track bounds
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_floor, max_floor = float('inf'), float('-inf')

    building_counts = defaultdict(int)

    for entry in entries:
        try:
            building_type = entry.get('T', '')
            x = entry.get('X', 0)
            y = entry.get('Y', 0)
            floor = entry.get('L', 0)
            rotation = entry.get('R', 0)

            placement = BuildingPlacement(
                building_type=building_type,
                x=x,
                y=y,
                floor=floor,
                rotation=rotation,
            )

            # Update bounds
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_floor = min(min_floor, floor)
            max_floor = max(max_floor, floor)

            # Count building type
            building_counts[building_type] += 1

            # Categorize using pattern matching
            if is_belt_type(building_type):
                analysis.belts.append(placement)
                analysis.belt_count += 1
            elif is_splitter_type(building_type):
                analysis.splitters.append(placement)
                analysis.splitter_count += 1
            elif is_merger_type(building_type):
                analysis.mergers.append(placement)
                analysis.merger_count += 1
            elif is_lift_type(building_type):
                analysis.lifts.append(placement)
                analysis.lift_count += 1
            elif building_type in MACHINE_TYPES:
                analysis.machines.append(placement)
                analysis.machine_count += 1
            elif is_belt_port_type(building_type) or is_pipe_type(building_type):
                # Belt ports and pipes are routing elements but handled differently
                analysis.other.append(placement)
            else:
                analysis.other.append(placement)

        except Exception as e:
            analysis.parse_errors.append(f"Entry parse error: {e}")

    # Set bounds
    if min_x != float('inf'):
        analysis.min_x = int(min_x)
        analysis.max_x = int(max_x)
        analysis.min_y = int(min_y)
        analysis.max_y = int(max_y)
        analysis.min_floor = int(min_floor)
        analysis.max_floor = int(max_floor)

        analysis.grid_width = analysis.max_x - analysis.min_x + 1
        analysis.grid_height = analysis.max_y - analysis.min_y + 1
        analysis.num_floors = analysis.max_floor - analysis.min_floor + 1

    analysis.building_counts = dict(building_counts)

    return analysis


def extract_connection_graph(analysis: BlueprintAnalysis) -> Dict[str, Any]:
    """
    Extract the connection graph from a blueprint analysis.

    This attempts to trace how items flow through the blueprint
    by following belt directions.

    Returns:
        Dict with 'nodes', 'edges', 'inputs', 'outputs'
    """
    # Build occupancy map: (x, y, floor) -> placement
    occupancy = {}
    for placement in (analysis.belts + analysis.splitters + analysis.mergers +
                      analysis.lifts + analysis.machines + analysis.other):
        occupancy[placement.position] = placement

    # Direction deltas based on rotation
    # Rotation: 0=E, 1=S, 2=W, 3=N
    ROTATION_TO_DELTA = {
        0: (1, 0, 0),   # East
        1: (0, 1, 0),   # South
        2: (-1, 0, 0),  # West
        3: (0, -1, 0),  # North
    }

    # Find belt endpoints (potential inputs/outputs)
    all_positions = set(occupancy.keys())
    has_incoming = set()
    has_outgoing = set()

    # Track belt flow edges
    edges = []

    for pos, placement in occupancy.items():
        if is_belt_type(placement.building_type):
            # Get output direction
            delta = ROTATION_TO_DELTA.get(placement.rotation, (0, 0, 0))
            next_pos = (pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2])

            if next_pos in occupancy:
                edges.append((pos, next_pos))
                has_outgoing.add(pos)
                has_incoming.add(next_pos)

        elif is_lift_type(placement.building_type):
            # Lifts move between floors
            if "Up" in placement.building_type:
                next_pos = (pos[0], pos[1], pos[2] + 1)
            else:
                next_pos = (pos[0], pos[1], pos[2] - 1)

            if next_pos in occupancy:
                edges.append((pos, next_pos))
                has_outgoing.add(pos)
                has_incoming.add(next_pos)

    # Inputs are belt cells with no incoming connection
    inputs = [p for p in all_positions if p not in has_incoming
              and is_belt_type(occupancy[p].building_type)]

    # Outputs are cells flowing to nothing
    outputs = []
    for pos, placement in occupancy.items():
        if is_belt_type(placement.building_type):
            delta = ROTATION_TO_DELTA.get(placement.rotation, (0, 0, 0))
            next_pos = (pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2])
            if next_pos not in occupancy:
                outputs.append(pos)

    return {
        'nodes': list(all_positions),
        'edges': edges,
        'inputs': inputs,
        'outputs': outputs,
        'machine_positions': [m.position for m in analysis.machines],
        'splitter_positions': [s.position for s in analysis.splitters],
        'merger_positions': [m.position for m in analysis.mergers],
    }


def extract_routing_problem(analysis: BlueprintAnalysis) -> Optional[Dict[str, Any]]:
    """
    Try to extract a routing problem from the blueprint.

    For ML training, we want to extract:
    - The machines (fixed positions)
    - The input/output ports
    - The ground truth routing (existing belts)

    Returns:
        Dict with routing problem definition, or None if can't extract
    """
    if analysis.machine_count == 0:
        return None  # No machines, can't define a routing problem

    graph = extract_connection_graph(analysis)

    # Normalize positions to start at (0, 0, 0)
    offset_x = analysis.min_x
    offset_y = analysis.min_y
    offset_z = analysis.min_floor

    def normalize(pos):
        return (pos[0] - offset_x, pos[1] - offset_y, pos[2] - offset_z)

    machines = []
    for m in analysis.machines:
        machines.append({
            'type': m.building_type,
            'position': normalize(m.position),
            'rotation': m.rotation,
        })

    # The existing belt layout is our "ground truth"
    belt_layout = []
    for b in analysis.belts:
        belt_layout.append({
            'type': b.building_type,
            'position': normalize(b.position),
            'rotation': b.rotation,
        })

    return {
        'grid_width': analysis.grid_width,
        'grid_height': analysis.grid_height,
        'num_floors': analysis.num_floors,
        'machines': machines,
        'splitters': [{'position': normalize(s.position), 'rotation': s.rotation}
                      for s in analysis.splitters],
        'mergers': [{'position': normalize(m.position), 'rotation': m.rotation}
                    for m in analysis.mergers],
        'lifts': [{'type': l.building_type, 'position': normalize(l.position)}
                  for l in analysis.lifts],
        'belt_layout': belt_layout,  # Ground truth
        'inputs': [normalize(p) for p in graph['inputs']],
        'outputs': [normalize(p) for p in graph['outputs']],
        'complexity': analysis.complexity_score,
    }


def is_suitable_for_ml(analysis: BlueprintAnalysis, max_buildings: int = 200) -> Tuple[bool, str]:
    """
    Check if a blueprint is suitable for ML training.

    Args:
        analysis: The blueprint analysis
        max_buildings: Maximum building count to consider

    Returns:
        Tuple of (is_suitable, reason)
    """
    if analysis.parse_errors:
        return False, f"Parse errors: {analysis.parse_errors[0]}"

    if analysis.total_entries == 0:
        return False, "Empty blueprint"

    if analysis.total_entries > max_buildings:
        return False, f"Too many buildings ({analysis.total_entries} > {max_buildings})"

    if analysis.machine_count == 0:
        return False, "No machines (trivial routing)"

    if analysis.belt_count == 0:
        return False, "No belts (no routing example)"

    if analysis.grid_width > 100 or analysis.grid_height > 100:
        return False, f"Grid too large ({analysis.grid_width}x{analysis.grid_height})"

    return True, "OK"


if __name__ == "__main__":
    # Test with a sample blueprint
    import sys

    if len(sys.argv) > 1:
        code = sys.argv[1]
    else:
        # Simple test blueprint
        code = "SHAPEZ2-1-H4sIAAAAAAAAA6tWKkktLlGyUlAqS8wpTtVRSs7PS0nNBQBQUjt5FQAAAA==$"

    print(f"Analyzing blueprint...")
    analysis = analyze_blueprint(code)

    print(f"\nBlueprint Analysis:")
    print(f"  Version: {analysis.version}")
    print(f"  Total entries: {analysis.total_entries}")
    print(f"  Grid size: {analysis.grid_width}x{analysis.grid_height}x{analysis.num_floors}")
    print(f"\nBuilding counts:")
    print(f"  Belts: {analysis.belt_count}")
    print(f"  Machines: {analysis.machine_count}")
    print(f"  Splitters: {analysis.splitter_count}")
    print(f"  Mergers: {analysis.merger_count}")
    print(f"  Lifts: {analysis.lift_count}")
    print(f"\nComplexity score: {analysis.complexity_score:.2f}")

    suitable, reason = is_suitable_for_ml(analysis)
    print(f"\nSuitable for ML: {suitable} ({reason})")

    if analysis.parse_errors:
        print(f"\nParse errors:")
        for err in analysis.parse_errors:
            print(f"  - {err}")
