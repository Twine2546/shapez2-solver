"""
Test scenarios for the flow simulator.
Run with: python -m shapez2_solver.simulation.test_scenarios

I/O positions must be on external walls at valid port positions:
- For 14x14 (1x1 foundation): ports at positions 5, 6, 8, 9 on each side
- West: (-1, y) where y in [5, 6, 8, 9]
- East: (14, y) where y in [5, 6, 8, 9]
- North: (x, -1) where x in [5, 6, 8, 9]
- South: (x, 14) where x in [5, 6, 8, 9]
"""

from shapez2_solver.simulation.flow_simulator import FlowSimulator, BuildingType, Rotation
from shapez2_solver.evolution.foundation_config import FOUNDATION_SPECS


def create_scenario(name: str, description: str, setup_func, foundation: str = "1x1") -> dict:
    """Create a test scenario with proper foundation and I/O validation."""
    spec = FOUNDATION_SPECS.get(foundation)
    sim = FlowSimulator(foundation_spec=spec, validate_io=True)
    setup_func(sim)
    return {
        'name': name,
        'description': description,
        'sim': sim,
    }


# Valid I/O positions for 1x1 foundation (14x14 grid):
# West:  (-1, 5), (-1, 6), (-1, 8), (-1, 9)
# East:  (14, 5), (14, 6), (14, 8), (14, 9)
# North: (5, -1), (6, -1), (8, -1), (9, -1)
# South: (5, 14), (6, 14), (8, 14), (9, 14)


# =============================================================================
# SCENARIO 1: Simple belt chain
# =============================================================================
def setup_simple_belt(sim):
    """Simple belt chain - west to east."""
    for x in range(14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)

SCENARIO_1 = lambda: create_scenario(
    "Simple Belt Chain",
    "EXPECTED: 180/min flows straight through all belts from west to east.",
    setup_simple_belt
)


# =============================================================================
# SCENARIO 2: Belt splitting (T-junction)
# =============================================================================
def setup_belt_split_no_conflict(sim):
    """Belt splits to EAST and NORTH."""
    # Main belt from west to east at y=5
    for x in range(14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Branch north from (5, 5) to north edge at x=5
    for y in range(4, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.NORTH)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)  # East
    sim.set_output(5, -1, 0)  # North

SCENARIO_2 = lambda: create_scenario(
    "Belt Split (T-junction)",
    "EXPECTED: 180/min splits to 90/min EAST and 90/min NORTH.",
    setup_belt_split_no_conflict
)


# =============================================================================
# SCENARIO 3: Belt splitting with 180° conflict
# =============================================================================
def setup_belt_split_180_conflict(sim):
    """Belt has NORTH and SOUTH branches - 180° conflict."""
    # Main belt at y=6 (has valid ports at both north and south)
    for x in range(14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    # Branch north from (6, 6)
    for y in range(5, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 6, y, 0, Rotation.NORTH)
    # Branch south from (6, 6)
    for y in range(7, 14):
        sim.place_building(BuildingType.BELT_FORWARD, 6, y, 0, Rotation.SOUTH)
    sim.set_input(-1, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 6, 0)   # East
    sim.set_output(6, -1, 0)   # North
    sim.set_output(6, 14, 0)   # South

SCENARIO_3 = lambda: create_scenario(
    "Belt Split (180° conflict)",
    "EXPECTED: Only one perpendicular branch gets flow (N or S, not both). "
    "EAST gets 90, one of N/S gets 90, other gets 0.",
    setup_belt_split_180_conflict
)


# =============================================================================
# SCENARIO 4: Belt merging (2 inputs to 1 belt)
# =============================================================================
def setup_belt_merge(sim):
    """Two belts merge into one."""
    # Input from west at y=5
    for x in range(6):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Input from west at y=6, turns north to merge
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.NORTH)
    # Continue merged flow east
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 90.0)
    sim.set_input(-1, 6, 0, "CuCuCuCu", 90.0)
    sim.set_output(14, 5, 0)

SCENARIO_4 = lambda: create_scenario(
    "Belt Merge (2 to 1)",
    "EXPECTED: Two 90/min inputs merge to 180/min at merge point.",
    setup_belt_merge
)


# =============================================================================
# SCENARIO 5: Cutter (both outputs)
# =============================================================================
def setup_cutter_both(sim):
    """Cutter with both outputs connected."""
    # Input from west
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Cutter at (5, 5) - 1x2, occupies (5,5) and (5,6)
    sim.place_building(BuildingType.CUTTER, 5, 5, 0, Rotation.EAST)
    # Output 1 at (5, 5) goes east
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Output 2 at (5, 6) goes east then south to valid output
    for x in range(6, 9):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    for y in range(7, 14):
        sim.place_building(BuildingType.BELT_FORWARD, 8, y, 0, Rotation.SOUTH)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)  # Left half
    sim.set_output(8, 14, 0)  # Right half

SCENARIO_5 = lambda: create_scenario(
    "Cutter (both outputs)",
    "EXPECTED: Cutter splits shape. Each output gets 45/min (cutter limit). "
    "Left half to east, right half to south.",
    setup_cutter_both
)


# =============================================================================
# SCENARIO 6: Cutter (missing output - should error)
# =============================================================================
def setup_cutter_missing(sim):
    """Cutter with only one output connected."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.CUTTER, 5, 5, 0, Rotation.EAST)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Second output at (5, 6) NOT connected
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)

SCENARIO_6 = lambda: create_scenario(
    "Cutter (missing output - ERROR)",
    "EXPECTED: Should report error for backed up output port at (5, 6). "
    "One output gets 45/min, the other backs up.",
    setup_cutter_missing
)


# =============================================================================
# SCENARIO 7: Cutter outputs merge
# =============================================================================
def setup_cutter_merge(sim):
    """Cutter outputs merge back together."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.CUTTER, 5, 5, 0, Rotation.EAST)
    # Both outputs go east
    for x in range(6, 10):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    # Merge at x=10
    sim.place_building(BuildingType.BELT_FORWARD, 10, 6, 0, Rotation.NORTH)
    for x in range(10, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)

SCENARIO_7 = lambda: create_scenario(
    "Cutter outputs merge",
    "EXPECTED: Both cutter outputs merge back to 90/min total "
    "(45+45 limited by cutter).",
    setup_cutter_merge
)


# =============================================================================
# SCENARIO 8: Splitter machine
# =============================================================================
def setup_splitter(sim):
    """Splitter splits flow to two outputs."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.SPLITTER, 5, 6, 0, Rotation.EAST)
    # North output at (5, 5)
    for y in range(5, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.NORTH)
    # South output at (5, 7)
    for y in range(7, 14):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.SOUTH)
    sim.set_input(-1, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, -1, 0)  # North
    sim.set_output(5, 14, 0)  # South

SCENARIO_8 = lambda: create_scenario(
    "Splitter machine",
    "EXPECTED: 180/min in -> 90/min north, 90/min south.",
    setup_splitter
)


# =============================================================================
# SCENARIO 9: Two cutters, merge all outputs
# =============================================================================
def setup_two_cutters(sim):
    """Two cutters in series, all outputs merge."""
    # First input line at y=5
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.CUTTER, 4, 5, 0, Rotation.EAST)
    # Second input line at y=8
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 8, 0, Rotation.EAST)
    sim.place_building(BuildingType.CUTTER, 4, 8, 0, Rotation.EAST)
    # Merge all 4 outputs toward east
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    # Route y=8 and y=9 outputs to valid outputs
    for x in range(5, 9):
        sim.place_building(BuildingType.BELT_FORWARD, x, 8, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 9, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 90.0)
    sim.set_input(-1, 8, 0, "RuRuRuRu", 90.0)
    sim.set_output(14, 5, 0)
    sim.set_output(14, 6, 0)
    sim.set_output(9, 14, 0)  # Route to south
    sim.set_output(8, 14, 0)

SCENARIO_9 = lambda: create_scenario(
    "Two cutters, merge all outputs",
    "EXPECTED: 4 output streams from 2 cutters. Total 180/min in, 90/min out "
    "(limited by cutter throughput 45*4=180 but each cutter only does 45).",
    setup_two_cutters
)


# =============================================================================
# SCENARIO 10: Rotator CW
# =============================================================================
def setup_rotator_cw(sim):
    """Rotator CW rotates shape clockwise."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.ROTATOR_CW, 5, 5, 0, Rotation.EAST)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCu----", 180.0)  # Top half
    sim.set_output(14, 5, 0)  # Rotated CW to right half

SCENARIO_10 = lambda: create_scenario(
    "Rotator CW",
    "EXPECTED: Top half 'CuCu----' rotates CW to right half '--Cu--Cu'. "
    "Throughput limited to 90/min.",
    setup_rotator_cw
)


# =============================================================================
# SCENARIO 11: Belt turns (left and right)
# =============================================================================
def setup_belt_turns(sim):
    """Test belt turning left and right."""
    # Start from west
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Turn south (right turn)
    sim.place_building(BuildingType.BELT_RIGHT, 5, 5, 0, Rotation.EAST)
    for y in range(6, 9):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.SOUTH)
    # Turn east (left turn)
    sim.place_building(BuildingType.BELT_LEFT, 5, 9, 0, Rotation.SOUTH)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 9, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 9, 0)

SCENARIO_11 = lambda: create_scenario(
    "Belt turns (left/right)",
    "EXPECTED: 180/min navigates through right and left turns to output.",
    setup_belt_turns
)


# =============================================================================
# SCENARIO 12: Stacker (two inputs)
# =============================================================================
def setup_stacker(sim):
    """Stacker combines two shapes from different floors."""
    # Floor 0 input at y=5
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Floor 1 input at y=5
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 1, Rotation.EAST)
    # Stacker at (4, 5)
    sim.place_building(BuildingType.STACKER, 4, 5, 0, Rotation.EAST)
    # Output on floor 0
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "Cu------", 30.0)  # Floor 0 - bottom layer
    sim.set_input(-1, 5, 1, "--Cu----", 30.0)  # Floor 1 - top layer
    sim.set_output(14, 5, 0)

SCENARIO_12 = lambda: create_scenario(
    "Stacker (two inputs)",
    "EXPECTED: Stacker combines floor 0 and floor 1 inputs. "
    "Output at 30/min (stacker speed limit).",
    setup_stacker
)


# =============================================================================
# SCENARIO 13: Rotated cutter (facing SOUTH)
# =============================================================================
def setup_rotated_cutter(sim):
    """Cutter facing SOUTH (rotated 90° CW)."""
    # Input from north at x=6
    for y in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, 6, y, 0, Rotation.SOUTH)
    # Cutter at (6, 5) facing south - occupies (6,5) and (7,5)
    sim.place_building(BuildingType.CUTTER, 6, 5, 0, Rotation.SOUTH)
    # Outputs go south
    for y in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, 6, y, 0, Rotation.SOUTH)
        sim.place_building(BuildingType.BELT_FORWARD, 7, y, 0, Rotation.SOUTH)
    sim.set_input(6, -1, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 14, 0)
    sim.set_output(8, 14, 0)  # Route second output

SCENARIO_13 = lambda: create_scenario(
    "Rotated cutter (facing SOUTH)",
    "EXPECTED: Cutter facing south works correctly. "
    "Each output gets 45/min.",
    setup_rotated_cutter
)


# =============================================================================
# SCENARIO 14: Merger machine (2 in -> 1 out)
# =============================================================================
def setup_merger(sim):
    """Merger combines two inputs into one output."""
    # Input from north at x=6 - belts from y=0 to y=5 (inclusive)
    for y in range(6):  # 0, 1, 2, 3, 4, 5
        sim.place_building(BuildingType.BELT_FORWARD, 6, y, 0, Rotation.SOUTH)
    # Input from south at x=6 - belts from y=13 to y=7 (inclusive)
    for y in range(13, 6, -1):  # 13, 12, 11, 10, 9, 8, 7
        sim.place_building(BuildingType.BELT_FORWARD, 6, y, 0, Rotation.NORTH)
    # Merger at (6, 6) - receives from N (y=5) and S (y=7)
    sim.place_building(BuildingType.MERGER, 6, 6, 0, Rotation.EAST)
    # Output east
    for x in range(7, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.set_input(6, -1, 0, "CuCuCuCu", 90.0)
    sim.set_input(6, 14, 0, "RuRuRuRu", 90.0)
    sim.set_output(14, 6, 0)

SCENARIO_14 = lambda: create_scenario(
    "Merger (2 to 1)",
    "EXPECTED: Two 90/min inputs merge to 180/min output. "
    "Shapes alternate CuCuCuCu and RuRuRuRu.",
    setup_merger
)


# =============================================================================
# SCENARIO 15: Rotator CCW
# =============================================================================
def setup_rotator_ccw(sim):
    """Rotator CCW rotates shape counter-clockwise."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.ROTATOR_CCW, 5, 5, 0, Rotation.EAST)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCu----", 90.0)
    sim.set_output(14, 5, 0)

SCENARIO_15 = lambda: create_scenario(
    "Rotator CCW",
    "EXPECTED: Top half 'CuCu----' rotates CCW to left half 'Cu--Cu--'.",
    setup_rotator_ccw
)


# =============================================================================
# SCENARIO 16: Rotator 180
# =============================================================================
def setup_rotator_180(sim):
    """Rotator 180 rotates shape 180 degrees."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.ROTATOR_180, 5, 5, 0, Rotation.EAST)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCu----", 90.0)
    sim.set_output(14, 5, 0)

SCENARIO_16 = lambda: create_scenario(
    "Rotator 180",
    "EXPECTED: Top half 'CuCu----' rotates 180° to bottom half '----CuCu'.",
    setup_rotator_180
)


# =============================================================================
# SCENARIO 17: Half Cutter
# =============================================================================
def setup_half_cutter(sim):
    """Half cutter destroys one half of the shape."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.HALF_CUTTER, 5, 5, 0, Rotation.EAST)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)

SCENARIO_17 = lambda: create_scenario(
    "Half Cutter",
    "EXPECTED: Full shape 'CuCuCuCu' becomes left half only 'Cu--Cu--'. "
    "Right half is destroyed. Throughput 45/min.",
    setup_half_cutter
)


# =============================================================================
# SCENARIO 18: Cutter Mirrored
# =============================================================================
def setup_cutter_mirrored(sim):
    """Cutter mirrored - input at bottom cell."""
    # Input at y=6 (bottom cell of mirrored cutter)
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    # Cutter mirrored at (5, 5) - occupies (5,5) and (5,6)
    sim.place_building(BuildingType.CUTTER_MIRRORED, 5, 5, 0, Rotation.EAST)
    # Outputs east
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.set_input(-1, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)
    sim.set_output(14, 6, 0)

SCENARIO_18 = lambda: create_scenario(
    "Cutter Mirrored",
    "EXPECTED: Input at bottom (y=6), outputs split. "
    "Mirror of regular cutter.",
    setup_cutter_mirrored
)


# =============================================================================
# SCENARIO 19: Swapper
# =============================================================================
def setup_swapper(sim):
    """Swapper swaps halves between two parallel items."""
    # Two input lines at y=5 and y=6
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    # Swapper at (4, 5) - occupies (4,5) and (4,6)
    sim.place_building(BuildingType.SWAPPER, 4, 5, 0, Rotation.EAST)
    # Outputs
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "Cu--Cu--", 45.0)
    sim.set_input(-1, 6, 0, "--Ru--Ru", 45.0)
    sim.set_output(14, 5, 0)
    sim.set_output(14, 6, 0)

SCENARIO_19 = lambda: create_scenario(
    "Swapper",
    "EXPECTED: Swapper swaps left/right halves between two shapes. "
    "Cu--Cu-- + --Ru--Ru become CuRuCuRu and RuCuRuCu.",
    setup_swapper
)


# =============================================================================
# SCENARIO 20: Stacker Bent
# =============================================================================
def setup_stacker_bent(sim):
    """Bent stacker - output perpendicular to input."""
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 1, Rotation.EAST)
    sim.place_building(BuildingType.STACKER_BENT, 4, 6, 0, Rotation.EAST)
    # Output goes south
    for y in range(7, 14):
        sim.place_building(BuildingType.BELT_FORWARD, 4, y, 0, Rotation.SOUTH)
    # Route to valid south output
    sim.place_building(BuildingType.BELT_FORWARD, 5, 13, 0, Rotation.EAST)
    sim.set_input(-1, 6, 0, "Cu------", 30.0)
    sim.set_input(-1, 6, 1, "--Ru----", 30.0)
    sim.set_output(5, 14, 0)

SCENARIO_20 = lambda: create_scenario(
    "Stacker Bent",
    "EXPECTED: Bent stacker combines floor 0 and floor 1 inputs. "
    "Output goes SOUTH (perpendicular to input).",
    setup_stacker_bent
)


# =============================================================================
# SCENARIO 21: Stacker Bent Mirrored
# =============================================================================
def setup_stacker_bent_mirrored(sim):
    """Bent stacker mirrored - output to north."""
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 1, Rotation.EAST)
    sim.place_building(BuildingType.STACKER_BENT_MIRRORED, 4, 6, 0, Rotation.EAST)
    # Output goes north
    for y in range(5, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 4, y, 0, Rotation.NORTH)
    # Route to valid north output
    sim.place_building(BuildingType.BELT_FORWARD, 5, 0, 0, Rotation.EAST)
    sim.set_input(-1, 6, 0, "Cu------", 30.0)
    sim.set_input(-1, 6, 1, "--Gr----", 30.0)
    sim.set_output(5, -1, 0)

SCENARIO_21 = lambda: create_scenario(
    "Stacker Bent Mirrored",
    "EXPECTED: Bent stacker mirrored - output goes NORTH instead of south.",
    setup_stacker_bent_mirrored
)


# =============================================================================
# SCENARIO 22: Unstacker
# =============================================================================
def setup_unstacker(sim):
    """Unstacker splits stacked shape into two layers."""
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.UNSTACKER, 4, 5, 0, Rotation.EAST)
    # Floor 0 output
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Floor 1 output
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 1, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuRuCuRu", 30.0)
    sim.set_output(14, 5, 0)  # Bottom layer
    sim.set_output(14, 5, 1)  # Top layer

SCENARIO_22 = lambda: create_scenario(
    "Unstacker",
    "EXPECTED: Stacked shape 'CuRuCuRu' splits into bottom layer (floor 0) "
    "and top layer (floor 1).",
    setup_unstacker
)


# =============================================================================
# SCENARIO 23: Painter
# =============================================================================
def setup_painter(sim):
    """Painter paints shapes with color from second input."""
    # Shape input from west at y=5
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Color input from north
    for y in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, 4, y, 0, Rotation.SOUTH)
    # Painter at (4, 5)
    sim.place_building(BuildingType.PAINTER, 4, 5, 0, Rotation.EAST)
    # Output
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 45.0)
    sim.set_input(5, -1, 0, "RuRuRuRu", 45.0)  # Color (north input)
    sim.set_output(14, 5, 0)

SCENARIO_23 = lambda: create_scenario(
    "Painter",
    "EXPECTED: Shape from west painted with color from north. "
    "Output at 45/min (painter limit).",
    setup_painter
)


# =============================================================================
# SCENARIO 24: Painter Mirrored
# =============================================================================
def setup_painter_mirrored(sim):
    """Painter mirrored - color input from south."""
    # Shape input from west at y=8
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 8, 0, Rotation.EAST)
    # Color input from south - route from x=5 (valid port) to x=4 (painter paint port)
    # Vertical belts at x=5 from y=13 to y=11, going NORTH
    for y in range(13, 10, -1):  # y=13, 12, 11
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.NORTH)
    # Turn west at y=10: belt at (5,10) going WEST outputs to (4,10)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 10, 0, Rotation.WEST)
    # Final belt going NORTH into painter paint port
    sim.place_building(BuildingType.BELT_FORWARD, 4, 10, 0, Rotation.NORTH)
    # Painter mirrored at (4, 8) - occupies (4, 8) and (4, 9), paint port at (4, 9) receives from S
    sim.place_building(BuildingType.PAINTER_MIRRORED, 4, 8, 0, Rotation.EAST)
    # Output
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 8, 0, Rotation.EAST)
    sim.set_input(-1, 8, 0, "CuCuCuCu", 45.0)
    sim.set_input(5, 14, 0, "GrGrGrGr", 45.0)  # Color (south input at valid port x=5)
    sim.set_output(14, 8, 0)

SCENARIO_24 = lambda: create_scenario(
    "Painter Mirrored",
    "EXPECTED: Color input from SOUTH instead of north. "
    "Shape painted with green color.",
    setup_painter_mirrored
)


# =============================================================================
# SCENARIO 25: Trash
# =============================================================================
def setup_trash(sim):
    """Trash destroys all incoming items."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.TRASH, 5, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    # No output - items are destroyed

SCENARIO_25 = lambda: create_scenario(
    "Trash",
    "EXPECTED: All 180/min items destroyed. No output. "
    "Trash shows 180 throughput but no backed up errors.",
    setup_trash
)


# =============================================================================
# SCENARIO 26: Lift Up
# =============================================================================
def setup_lift_up(sim):
    """Lift moves items from floor 0 to floor 1."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.LIFT_UP, 5, 5, 0, Rotation.EAST)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 1, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 1)

SCENARIO_26 = lambda: create_scenario(
    "Lift Up",
    "EXPECTED: Items move from floor 0 to floor 1. "
    "Full 180/min throughput.",
    setup_lift_up
)


# =============================================================================
# SCENARIO 27: Lift Down
# =============================================================================
def setup_lift_down(sim):
    """Lift moves items from floor 1 to floor 0."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.LIFT_DOWN, 5, 5, 0, Rotation.EAST)
    for x in range(6, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 1, "RuRuRuRu", 180.0)
    sim.set_output(14, 5, 0)

SCENARIO_27 = lambda: create_scenario(
    "Lift Down",
    "EXPECTED: Items move from floor 1 to floor 0. "
    "Full 180/min throughput.",
    setup_lift_down
)


# =============================================================================
# SCENARIO 28: Multilevel routing
# =============================================================================
def setup_multilevel_routing(sim):
    """Complex routing across multiple floors."""
    # Floor 0: west to lift
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.LIFT_UP, 5, 5, 0, Rotation.EAST)
    # Floor 1: continue east, then lift down
    for x in range(6, 10):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.LIFT_DOWN, 10, 5, 0, Rotation.EAST)
    # Floor 0: final stretch to output
    for x in range(11, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(14, 5, 0)

SCENARIO_28 = lambda: create_scenario(
    "Multilevel Routing",
    "EXPECTED: Items go floor0 -> lift up -> floor1 -> lift down -> floor0. "
    "Full 180/min throughput maintained.",
    setup_multilevel_routing
)


# =============================================================================
# SCENARIO 29: Multilevel processing
# =============================================================================
def setup_multilevel_processing(sim):
    """Two floors feeding into stacker."""
    # Floor 0 input
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Floor 1 input via lift
    for x in range(2):
        sim.place_building(BuildingType.BELT_FORWARD, x, 8, 0, Rotation.EAST)
    sim.place_building(BuildingType.LIFT_UP, 2, 8, 0, Rotation.EAST)
    for x in range(3, 4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 8, 1, Rotation.EAST)
    # Route floor 1 north to y=5
    for y in range(7, 4, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 3, y, 1, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 1, Rotation.EAST)
    # Stacker
    sim.place_building(BuildingType.STACKER, 4, 5, 0, Rotation.EAST)
    # Output
    for x in range(5, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "Cu------", 30.0)
    sim.set_input(-1, 8, 0, "--Ru----", 30.0)
    sim.set_output(14, 5, 0)

SCENARIO_29 = lambda: create_scenario(
    "Multilevel Processing",
    "EXPECTED: Floor 0 and lifted floor 1 inputs feed stacker. "
    "Output at 30/min.",
    setup_multilevel_processing
)


# =============================================================================
# SCENARIO 30: Splitter Left Priority
# =============================================================================
def setup_splitter_left(sim):
    """Splitter with left priority."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.SPLITTER_LEFT, 5, 6, 0, Rotation.EAST)
    # North output (left priority)
    for y in range(5, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.NORTH)
    # South output
    for y in range(7, 14):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.SOUTH)
    sim.set_input(-1, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, -1, 0)   # North (left)
    sim.set_output(5, 14, 0)   # South (right)

SCENARIO_30 = lambda: create_scenario(
    "Splitter Left Priority",
    "EXPECTED: Splitter prioritizes left (north) output. "
    "Both outputs get 90/min if both connected.",
    setup_splitter_left
)


# =============================================================================
# SCENARIO 31: Splitter Right Priority
# =============================================================================
def setup_splitter_right(sim):
    """Splitter with right priority."""
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.SPLITTER_RIGHT, 5, 6, 0, Rotation.EAST)
    # North output
    for y in range(5, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.NORTH)
    # South output (right priority)
    for y in range(7, 14):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.SOUTH)
    sim.set_input(-1, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, -1, 0)   # North
    sim.set_output(5, 14, 0)   # South (right)

SCENARIO_31 = lambda: create_scenario(
    "Splitter Right Priority",
    "EXPECTED: Splitter prioritizes right (south) output. "
    "Both outputs get 90/min if both connected.",
    setup_splitter_right
)


# =============================================================================
# SCENARIO 32: Cutter facing WEST
# =============================================================================
def setup_cutter_west(sim):
    """Cutter rotated to face WEST."""
    # Input from east at y=6 (cutter input after rotation)
    for x in range(13, 8, -1):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.WEST)
    # Cutter at (8, 5) facing WEST - occupies (8,5) and (8,6)
    sim.place_building(BuildingType.CUTTER, 8, 5, 0, Rotation.WEST)
    # Outputs go west
    for x in range(7, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.WEST)
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.WEST)
    sim.set_input(14, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(-1, 5, 0)
    sim.set_output(-1, 6, 0)

SCENARIO_32 = lambda: create_scenario(
    "Cutter facing WEST",
    "EXPECTED: Cutter works correctly when facing west. "
    "Each output gets 45/min.",
    setup_cutter_west
)


# =============================================================================
# SCENARIO 33: Cutter facing NORTH
# =============================================================================
def setup_cutter_north(sim):
    """Cutter rotated to face NORTH."""
    # Input from south at x=9 (cutter input after rotation)
    for y in range(13, 8, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 9, y, 0, Rotation.NORTH)
    # Cutter at (8, 8) facing NORTH - occupies (8,8) and (9,8)
    sim.place_building(BuildingType.CUTTER, 8, 8, 0, Rotation.NORTH)
    # Outputs go north
    for y in range(7, -1, -1):
        sim.place_building(BuildingType.BELT_FORWARD, 8, y, 0, Rotation.NORTH)
        sim.place_building(BuildingType.BELT_FORWARD, 9, y, 0, Rotation.NORTH)
    sim.set_input(9, 14, 0, "CuCuCuCu", 180.0)
    sim.set_output(8, -1, 0)
    sim.set_output(9, -1, 0)

SCENARIO_33 = lambda: create_scenario(
    "Cutter facing NORTH",
    "EXPECTED: Cutter works correctly when facing north. "
    "Each output gets 45/min.",
    setup_cutter_north
)


# =============================================================================
# SCENARIO 34: Complex factory - cutter -> stacker
# =============================================================================
def setup_cutter_stacker_combo(sim):
    """Cut shape in half, stack halves together."""
    # Input
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Cutter
    sim.place_building(BuildingType.CUTTER, 4, 5, 0, Rotation.EAST)
    # Left half continues east, lifts to floor 1
    for x in range(5, 7):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.LIFT_UP, 7, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 8, 5, 1, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 8, 6, 1, Rotation.EAST)
    # Right half goes to y=6, needs belt all the way to stacker input
    for x in range(5, 9):  # 5, 6, 7, 8 - connects to stacker IN[0] at (9,6,0) from W
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    # Stacker combines them - IN[0] from floor 0, IN[1] from floor 1
    sim.place_building(BuildingType.STACKER, 9, 6, 0, Rotation.EAST)
    # Output
    for x in range(10, 14):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuRuGrBl", 45.0)
    sim.set_output(14, 6, 0)

SCENARIO_34 = lambda: create_scenario(
    "Cutter -> Stacker Combo",
    "EXPECTED: Shape cut in half, halves route to different floors, "
    "then stack back together.",
    setup_cutter_stacker_combo
)


# =============================================================================
# SCENARIO 35: Three-way split
# =============================================================================
def setup_three_way_split(sim):
    """Chain splitters for 3-way split."""
    # Input from west at y=6
    for x in range(4):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    # First splitter at (4, 6) splits N/S
    sim.place_building(BuildingType.SPLITTER, 4, 6, 0, Rotation.EAST)
    # North branch (50%) goes to second splitter
    for y in range(5, 1, -1):  # y=5,4,3,2 -> connects to splitter at (4, 1)
        sim.place_building(BuildingType.BELT_FORWARD, 4, y, 0, Rotation.NORTH)
    # Second splitter at (4, 1) - will split N/S again
    # But valid output north is at (5, -1), so we need to route via x=5
    sim.place_building(BuildingType.BELT_FORWARD, 4, 1, 0, Rotation.EAST)
    sim.place_building(BuildingType.SPLITTER, 5, 1, 0, Rotation.EAST)
    # North branch from second splitter to output at (5, -1)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 0, 0, Rotation.NORTH)
    # South branch from second splitter to output at (5, 14) via long route
    for y in range(2, 14):  # y=2,3,...,13 going south
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.SOUTH)
    # South branch from first splitter (50%) goes to output at (6, 14)
    for y in range(7, 14):  # y=7,8,...,13 going south
        sim.place_building(BuildingType.BELT_FORWARD, 4, y, 0, Rotation.SOUTH)
    # Route first splitter south output to (6, 14) via turn east
    sim.place_building(BuildingType.BELT_FORWARD, 4, 13, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 13, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 13, 0, Rotation.SOUTH)
    sim.set_input(-1, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, -1, 0)   # ~45/min (25% of 180) - north from second splitter
    sim.set_output(5, 14, 0)   # ~45/min (25% of 180) - south from second splitter
    sim.set_output(6, 14, 0)   # ~90/min (50% of 180) - south from first splitter

SCENARIO_35 = lambda: create_scenario(
    "Three-way Split",
    "EXPECTED: 180 in -> first split 90/90 -> second split 45/45 + 90. "
    "Three outputs: ~45, ~45, ~90.",
    setup_three_way_split
)


# =============================================================================
# SCENARIO 36: 2x2 Foundation - Simple Belt Chain
# =============================================================================
def setup_2x2_belt_chain(sim):
    """Simple belt chain across 2x2 foundation (34x34 grid)."""
    # Belt chain from west to east at y=5
    for x in range(34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(34, 5, 0)

SCENARIO_36 = lambda: create_scenario(
    "2x2 Foundation - Belt Chain",
    "EXPECTED: Simple belt chain works on larger 2x2 foundation. "
    "Full 180/min throughput.",
    setup_2x2_belt_chain,
    foundation="2x2"
)


# =============================================================================
# SCENARIO 37: 2x2 Foundation - Cutter with Routing
# =============================================================================
def setup_2x2_cutter(sim):
    """Cutter on 2x2 foundation with both outputs going east."""
    # Input belt from west at y=5
    for x in range(10):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Cutter at (10, 5) - outputs at (10,5) and (10,6) going east
    sim.place_building(BuildingType.CUTTER, 10, 5, 0, Rotation.EAST)
    # Left half continues east at y=5
    for x in range(11, 34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Right half continues east at y=6
    for x in range(11, 34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(34, 5, 0)
    sim.set_output(34, 6, 0)

SCENARIO_37 = lambda: create_scenario(
    "2x2 Foundation - Cutter",
    "EXPECTED: Cutter works on 2x2 foundation. "
    "Each output gets 45/min.",
    setup_2x2_cutter,
    foundation="2x2"
)


# =============================================================================
# SCENARIO 38: 3x3 Foundation - Long Belt Chain
# =============================================================================
def setup_3x3_belt_chain(sim):
    """Long belt chain across 3x3 foundation (54x54 grid)."""
    # Belt chain from west to east at y=5
    for x in range(54):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(54, 5, 0)

SCENARIO_38 = lambda: create_scenario(
    "3x3 Foundation - Belt Chain",
    "EXPECTED: Long belt chain works on 3x3 foundation. "
    "Full 180/min throughput across 54 cells.",
    setup_3x3_belt_chain,
    foundation="3x3"
)


# =============================================================================
# SCENARIO 39: 2x1 Foundation - Wide Belt Chain
# =============================================================================
def setup_2x1_belt_chain(sim):
    """Belt chain on wide 2x1 foundation (34x14 grid)."""
    # Belt chain from west to east at y=5
    for x in range(34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(34, 5, 0)

SCENARIO_39 = lambda: create_scenario(
    "2x1 Foundation - Wide",
    "EXPECTED: Belt chain works on wide 2x1 foundation (34x14). "
    "Full 180/min throughput.",
    setup_2x1_belt_chain,
    foundation="2x1"
)


# =============================================================================
# SCENARIO 40: 1x2 Foundation - Tall Belt Chain
# =============================================================================
def setup_1x2_belt_chain(sim):
    """Belt chain on tall 1x2 foundation (14x34 grid)."""
    # Belt chain from north to south at x=5
    for y in range(34):
        sim.place_building(BuildingType.BELT_FORWARD, 5, y, 0, Rotation.SOUTH)
    sim.set_input(5, -1, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, 34, 0)

SCENARIO_40 = lambda: create_scenario(
    "1x2 Foundation - Tall",
    "EXPECTED: Vertical belt chain works on tall 1x2 foundation (14x34). "
    "Full 180/min throughput.",
    setup_1x2_belt_chain,
    foundation="1x2"
)


# =============================================================================
# SCENARIO 41: 2x2 Foundation - Cross-Foundation Routing
# =============================================================================
def setup_2x2_cross_routing(sim):
    """Route from one quadrant to another on 2x2 foundation."""
    # Input from west side, first unit (y=5)
    for x in range(17):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Turn south at x=17
    for y in range(5, 26):
        sim.place_building(BuildingType.BELT_FORWARD, 17, y, 0, Rotation.SOUTH)
    # Turn east again at y=25
    for x in range(18, 34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 25, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(34, 25, 0)

SCENARIO_41 = lambda: create_scenario(
    "2x2 Foundation - Cross Routing",
    "EXPECTED: Route from top-left to bottom-right quadrant. "
    "Full 180/min throughput after turns.",
    setup_2x2_cross_routing,
    foundation="2x2"
)


# =============================================================================
# SCENARIO 42: 2x2 Foundation - Multi-Machine
# =============================================================================
def setup_2x2_multi_machine(sim):
    """Cutter + Rotator chain on 2x2 foundation."""
    # Input belt to cutter
    for x in range(5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Cutter at (5, 5)
    sim.place_building(BuildingType.CUTTER, 5, 5, 0, Rotation.EAST)
    # Left half to rotator, then to east edge
    for x in range(6, 15):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.ROTATOR_CW, 15, 5, 0, Rotation.EAST)
    for x in range(16, 34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Right half continues east at y=6
    for x in range(6, 34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 6, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(34, 5, 0)
    sim.set_output(34, 6, 0)

SCENARIO_42 = lambda: create_scenario(
    "2x2 Foundation - Multi-Machine",
    "EXPECTED: Cutter + Rotator chain on 2x2 foundation. "
    "Both outputs receive flow.",
    setup_2x2_multi_machine,
    foundation="2x2"
)


# =============================================================================
# SCENARIO 43: T-Shaped Foundation
# =============================================================================
def setup_t_foundation(sim):
    """Test T-shaped irregular foundation (54x34 grid)."""
    # T foundation: 3 units wide on top row, 1 unit centered below
    # Grid is 54x34 (3 units wide = 54 cells)
    # Simple belt from west to east at y=5
    for x in range(54):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(54, 5, 0)

SCENARIO_43 = lambda: create_scenario(
    "T-Shaped Foundation",
    "EXPECTED: Belt chain works on T-shaped irregular foundation.",
    setup_t_foundation,
    foundation="T"
)


# =============================================================================
# SCENARIO 44: L-Shaped Foundation
# =============================================================================
def setup_l_foundation(sim):
    """Test L-shaped irregular foundation (34x34 grid)."""
    # L foundation: 2x2 bounding box with one corner missing
    # Grid is 34x34
    # Simple belt from west to east at y=5
    for x in range(34):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(-1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(34, 5, 0)

SCENARIO_44 = lambda: create_scenario(
    "L-Shaped Foundation",
    "EXPECTED: Belt chain works on L-shaped irregular foundation.",
    setup_l_foundation,
    foundation="L"
)


# =============================================================================
# ALL SCENARIOS
# =============================================================================
ALL_SCENARIOS = [
    SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5,
    SCENARIO_6, SCENARIO_7, SCENARIO_8, SCENARIO_9, SCENARIO_10,
    SCENARIO_11, SCENARIO_12, SCENARIO_13, SCENARIO_14, SCENARIO_15,
    SCENARIO_16, SCENARIO_17, SCENARIO_18, SCENARIO_19, SCENARIO_20,
    SCENARIO_21, SCENARIO_22, SCENARIO_23, SCENARIO_24, SCENARIO_25,
    SCENARIO_26, SCENARIO_27, SCENARIO_28, SCENARIO_29, SCENARIO_30,
    SCENARIO_31, SCENARIO_32, SCENARIO_33, SCENARIO_34, SCENARIO_35,
    SCENARIO_36, SCENARIO_37, SCENARIO_38, SCENARIO_39, SCENARIO_40,
    SCENARIO_41, SCENARIO_42, SCENARIO_43, SCENARIO_44,
]


def get_scenario_count() -> int:
    """Get the total number of scenarios."""
    return len(ALL_SCENARIOS)


def get_scenario(index: int) -> dict:
    """Get a specific scenario by index (0-based)."""
    if 0 <= index < len(ALL_SCENARIOS):
        return ALL_SCENARIOS[index]()
    return None


def run_all_scenarios():
    """Run all scenarios and print results."""
    for i, scenario_func in enumerate(ALL_SCENARIOS, 1):
        scenario = scenario_func()
        print(f"\n{'='*70}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Description: {scenario['description']}")
        print()

        sim = scenario['sim']
        report = sim.simulate()
        sim.print_grid(0)
        print(report)


if __name__ == "__main__":
    run_all_scenarios()
