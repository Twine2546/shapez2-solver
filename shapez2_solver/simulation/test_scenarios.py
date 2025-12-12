"""
Test scenarios for the flow simulator.
Run with: python -m shapez2_solver.simulation.test_scenarios
"""

from shapez2_solver.simulation.flow_simulator import FlowSimulator, BuildingType, Rotation


def create_scenario(name: str, description: str, setup_func) -> dict:
    """Create a test scenario."""
    sim = FlowSimulator(14, 14, 4)
    setup_func(sim)
    return {
        'name': name,
        'description': description,
        'sim': sim,
    }


# =============================================================================
# SCENARIO 1: Simple belt chain
# =============================================================================
def setup_simple_belt(sim):
    """Simple belt chain - input to output."""
    for x in range(1, 8):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    sim.set_input(1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(8, 5, 0)

SCENARIO_1 = lambda: create_scenario(
    "Simple Belt Chain",
    "EXPECTED: 180/min flows straight through all belts to output.",
    setup_simple_belt
)


# =============================================================================
# SCENARIO 2: Belt splitting (T-junction, no conflict)
# =============================================================================
def setup_belt_split_no_conflict(sim):
    """Belt splits to EAST and NORTH (no 180° conflict)."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)  # Continue east
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.NORTH)  # Branch north
    sim.place_building(BuildingType.BELT_FORWARD, 3, 3, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 2, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0)
    sim.set_output(4, 2, 0)

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
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)  # Continue east
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.NORTH)  # North (first)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.SOUTH)  # South (conflict!)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 7, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0)
    sim.set_output(4, 3, 0)
    sim.set_output(4, 7, 0)

SCENARIO_3 = lambda: create_scenario(
    "Belt Split (180° conflict)",
    "EXPECTED: SOUTH dropped due to 180° conflict with NORTH. "
    "EAST gets 90, SOUTH gets 90, NORTH gets 0.",
    setup_belt_split_180_conflict
)


# =============================================================================
# SCENARIO 4: Belt merging (2 inputs to 1 belt)
# =============================================================================
def setup_belt_merge(sim):
    """Two belts merge into one."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 4, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 6, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)  # Merge point
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.set_input(2, 4, 0, "CuCuCuCu", 90.0)
    sim.set_input(2, 6, 0, "CuCuCuCu", 90.0)
    sim.set_output(5, 5, 0)

SCENARIO_4 = lambda: create_scenario(
    "Belt Merge (2 to 1)",
    "EXPECTED: Two 90/min inputs merge to 180/min at merge point.",
    setup_belt_merge
)


# =============================================================================
# SCENARIO 5: Single cutter with both outputs
# =============================================================================
def setup_cutter_both_outputs(sim):
    """Cutter with both outputs connected."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.CUTTER, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0, "Cu--Cu--")  # Left half
    sim.set_output(6, 6, 0, "--Cu--Cu")  # Right half

SCENARIO_5 = lambda: create_scenario(
    "Cutter (both outputs)",
    "EXPECTED: Input splits into left half (45/min) and right half (45/min). "
    "Both outputs connected, no backup.",
    setup_cutter_both_outputs
)


# =============================================================================
# SCENARIO 6: Cutter with missing output (should error)
# =============================================================================
def setup_cutter_missing_output(sim):
    """Cutter with one output missing - should cause backup error."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.CUTTER, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    # Missing belt at (4, 6) for second output!
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0)

SCENARIO_6 = lambda: create_scenario(
    "Cutter (missing output - ERROR)",
    "EXPECTED: ERROR - Right half output has no belt, will back up!",
    setup_cutter_missing_output
)


# =============================================================================
# SCENARIO 7: Cutter outputs merge back together
# =============================================================================
def setup_cutter_outputs_merge(sim):
    """Cutter outputs merge back into single belt."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.CUTTER, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.SOUTH)  # Goes down
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)   # Merge point
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 6, 0)

SCENARIO_7 = lambda: create_scenario(
    "Cutter outputs merge",
    "EXPECTED: Both cutter outputs (45 each) merge to 90/min at (4,6).",
    setup_cutter_outputs_merge
)


# =============================================================================
# SCENARIO 8: Splitter machine
# =============================================================================
def setup_splitter(sim):
    """Splitter splits flow to N and S."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.SPLITTER, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.EAST)  # North output
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.EAST)  # South output
    sim.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, 4, 0)
    sim.set_output(5, 6, 0)

SCENARIO_8 = lambda: create_scenario(
    "Splitter machine",
    "EXPECTED: Splitter outputs 90/min to NORTH and 90/min to SOUTH.",
    setup_splitter
)


# =============================================================================
# SCENARIO 9: Two cutters, all outputs merge
# =============================================================================
def setup_two_cutters_merge(sim):
    """Splitter to 2 cutters, all 4 outputs merge."""
    # Input
    sim.place_building(BuildingType.BELT_FORWARD, 1, 5, 0, Rotation.EAST)

    # Splitter
    sim.place_building(BuildingType.SPLITTER, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 6, 0, Rotation.EAST)

    # Belts to cutters
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.EAST)

    # Two cutters
    sim.place_building(BuildingType.CUTTER, 4, 4, 0, Rotation.EAST)  # Outputs at y=4,5
    sim.place_building(BuildingType.CUTTER, 4, 6, 0, Rotation.EAST)  # Outputs at y=6,7

    # Output belts merging down
    sim.place_building(BuildingType.BELT_FORWARD, 5, 4, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 7, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 7, 0, Rotation.EAST)

    sim.set_input(1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(7, 7, 0)

SCENARIO_9 = lambda: create_scenario(
    "Two cutters, merge all outputs",
    "EXPECTED: 180 in -> split 90 each -> cutters 45 each output -> "
    "merge cascade: 45->90->135->180 at output.",
    setup_two_cutters_merge
)


# =============================================================================
# SCENARIO 10: Rotator
# =============================================================================
def setup_rotator(sim):
    """Rotator CW rotates shape."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.ROTATOR_CW, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCu----", 180.0)  # Top half
    sim.set_output(6, 5, 0, "--Cu--Cu")  # Rotated to right half

SCENARIO_10 = lambda: create_scenario(
    "Rotator CW",
    "EXPECTED: Top half 'CuCu----' rotates CW to right half '--Cu--Cu'.",
    setup_rotator
)


# =============================================================================
# SCENARIO 11: Belt turns (left/right)
# =============================================================================
def setup_belt_turns(sim):
    """Belt left and right turns."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_LEFT, 3, 5, 0, Rotation.EAST)  # Turns north
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_RIGHT, 3, 3, 0, Rotation.NORTH)  # Turns east
    sim.place_building(BuildingType.BELT_FORWARD, 4, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 3, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 3, 0)

SCENARIO_11 = lambda: create_scenario(
    "Belt turns (left/right)",
    "EXPECTED: Belt turns left then right, 180/min flows through.",
    setup_belt_turns
)


# =============================================================================
# SCENARIO 12: Stacker (two inputs)
# =============================================================================
def setup_stacker(sim):
    """Stacker combines two shapes."""
    # Floor 0 input belt (bottom layer)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Floor 1 input belt (top layer)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 1, Rotation.EAST)

    # Stacker at (4, 5) - inputs from west at floor 0 and floor 1
    sim.place_building(BuildingType.STACKER, 4, 5, 0, Rotation.EAST)

    # Output on floor 0
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "Cu------", 30.0)  # Floor 0 - bottom layer
    sim.set_input(2, 5, 1, "--Cu----", 30.0)  # Floor 1 - top layer
    sim.set_output(7, 5, 0)

SCENARIO_12 = lambda: create_scenario(
    "Stacker (two inputs)",
    "EXPECTED: Stacker combines bottom 'Cu------' with top '--Cu----'. "
    "Output at 30/min (stacker speed limit).",
    setup_stacker
)


# =============================================================================
# SCENARIO 13: Machine facing different directions
# =============================================================================
def setup_rotated_cutter(sim):
    """Cutter facing SOUTH instead of EAST."""
    sim.place_building(BuildingType.BELT_FORWARD, 5, 2, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.CUTTER, 5, 3, 0, Rotation.SOUTH)  # Input from N
    sim.place_building(BuildingType.BELT_FORWARD, 5, 4, 0, Rotation.SOUTH)  # Left out
    sim.place_building(BuildingType.BELT_FORWARD, 6, 4, 0, Rotation.SOUTH)  # Right out
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)
    sim.set_input(5, 2, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0)
    sim.set_output(7, 5, 0)

SCENARIO_13 = lambda: create_scenario(
    "Rotated cutter (facing SOUTH)",
    "EXPECTED: Cutter works correctly when facing SOUTH. "
    "Input from NORTH, outputs to SOUTH.",
    setup_rotated_cutter
)


# =============================================================================
# SCENARIO 14: Merger machine (2 in -> 1 out)
# =============================================================================
def setup_merger(sim):
    """Merger combines two inputs into one output (alternating)."""
    # Input from north
    sim.place_building(BuildingType.BELT_FORWARD, 4, 3, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.SOUTH)
    # Input from south
    sim.place_building(BuildingType.BELT_FORWARD, 4, 7, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.NORTH)
    # Merger
    sim.place_building(BuildingType.MERGER, 4, 5, 0, Rotation.EAST)
    # Output
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim.set_input(4, 3, 0, "CuCuCuCu", 90.0)
    sim.set_input(4, 7, 0, "RuRuRuRu", 90.0)
    sim.set_output(7, 5, 0)

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
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.ROTATOR_CCW, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCu----", 90.0)  # Top half
    sim.set_output(6, 5, 0, "Cu--Cu--")  # Rotated CCW to left half

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
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.ROTATOR_180, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCu----", 90.0)  # Top half
    sim.set_output(6, 5, 0, "----CuCu")  # Rotated 180 to bottom half

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
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.HALF_CUTTER, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0, "Cu--Cu--")  # Left half only

SCENARIO_17 = lambda: create_scenario(
    "Half Cutter",
    "EXPECTED: Full shape 'CuCuCuCu' becomes left half only 'Cu--Cu--'. "
    "Right half is destroyed.",
    setup_half_cutter
)


# =============================================================================
# SCENARIO 18: Cutter Mirrored
# =============================================================================
def setup_cutter_mirrored(sim):
    """Cutter mirrored - input at bottom, second output goes north."""
    # Input belt to bottom cell of cutter (y+1)
    sim.place_building(BuildingType.BELT_FORWARD, 2, 6, 0, Rotation.EAST)
    # Cutter mirrored at (3, 5) - occupies (3,5) and (3,6)
    # Input at (3, 6), outputs at (3, 5) and (3, 6) going east
    sim.place_building(BuildingType.CUTTER_MIRRORED, 3, 5, 0, Rotation.EAST)
    # Output belts
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)  # Top output (second)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)  # Bottom output (first)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.EAST)

    sim.set_input(2, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 5, 0, "--Cu--Cu")  # Top = right half
    sim.set_output(6, 6, 0, "Cu--Cu--")  # Bottom = left half

SCENARIO_18 = lambda: create_scenario(
    "Cutter Mirrored",
    "EXPECTED: Input at bottom (y+1), outputs split: "
    "bottom=left half, top=right half. Mirror of regular cutter.",
    setup_cutter_mirrored
)


# =============================================================================
# SCENARIO 19: Swapper
# =============================================================================
def setup_swapper(sim):
    """Swapper swaps halves between two parallel items."""
    # Top input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.EAST)
    # Bottom input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Swapper occupies (4, 4) and (4, 5)
    sim.place_building(BuildingType.SWAPPER, 4, 4, 0, Rotation.EAST)
    # Outputs
    sim.place_building(BuildingType.BELT_FORWARD, 5, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim.set_input(2, 4, 0, "Cu--Cu--", 45.0)  # Top input - left half copper
    sim.set_input(2, 5, 0, "--Ru--Ru", 45.0)  # Bottom input - right half ruby
    sim.set_output(7, 4, 0)  # After swap: Cu left + Ru right = CuRuCuRu
    sim.set_output(7, 5, 0)  # After swap: Ru left + Cu right = RuCuRuCu

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
    """Bent stacker - inputs from west, output to south."""
    # Floor 0 input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Floor 1 input - need lift up first
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 1, Rotation.EAST)
    # Bent stacker
    sim.place_building(BuildingType.STACKER_BENT, 4, 5, 0, Rotation.EAST)
    # Output goes south (bent)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 7, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 7, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "Cu------", 45.0)  # Bottom layer
    sim.set_input(2, 5, 1, "--Ru----", 45.0)  # Top layer
    sim.set_output(6, 7, 0)

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
    """Bent stacker mirrored - inputs from west, output to north."""
    # Floor 0 input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Floor 1 input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 1, Rotation.EAST)
    # Bent stacker mirrored
    sim.place_building(BuildingType.STACKER_BENT_MIRRORED, 4, 5, 0, Rotation.EAST)
    # Output goes north (mirrored)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 3, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "Cu------", 45.0)
    sim.set_input(2, 5, 1, "--Gr----", 45.0)
    sim.set_output(6, 3, 0)

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
    # Input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Unstacker
    sim.place_building(BuildingType.UNSTACKER, 4, 5, 0, Rotation.EAST)
    # Floor 0 output (bottom layer)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)
    # Floor 1 output (top layer)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 1, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuRuCuRu", 30.0)  # Stacked shape
    sim.set_output(7, 5, 0)  # Bottom layer
    sim.set_output(7, 5, 1)  # Top layer

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
    """Painter paints shapes with color from north input."""
    # Shape input from west
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Color input from north
    sim.place_building(BuildingType.BELT_FORWARD, 4, 3, 0, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.SOUTH)
    # Painter at (4, 5) - occupies 1x2
    sim.place_building(BuildingType.PAINTER, 4, 5, 0, Rotation.EAST)
    # Output
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuCuCuCu", 45.0)  # Shape
    sim.set_input(4, 3, 0, "RuRuRuRu", 45.0)  # Color (ruby red)
    sim.set_output(7, 5, 0)

SCENARIO_23 = lambda: create_scenario(
    "Painter",
    "EXPECTED: Shape from west painted with color from north. "
    "Color input comes from NORTH edge of painter.",
    setup_painter
)


# =============================================================================
# SCENARIO 24: Painter Mirrored
# =============================================================================
def setup_painter_mirrored(sim):
    """Painter mirrored - color input from south."""
    # Shape input from west
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Color input from south
    sim.place_building(BuildingType.BELT_FORWARD, 4, 7, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.NORTH)
    # Painter mirrored
    sim.place_building(BuildingType.PAINTER_MIRRORED, 4, 5, 0, Rotation.EAST)
    # Output
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuCuCuCu", 45.0)  # Shape
    sim.set_input(4, 7, 0, "GrGrGrGr", 45.0)  # Color (green)
    sim.set_output(7, 5, 0)

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
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.TRASH, 5, 5, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    # No output - items are destroyed

SCENARIO_25 = lambda: create_scenario(
    "Trash",
    "EXPECTED: All 180/min items destroyed. No output. "
    "Trash shows 180 throughput but no backed up errors.",
    setup_trash
)


# =============================================================================
# SCENARIO 26: Lift Up (single floor)
# =============================================================================
def setup_lift_up(sim):
    """Lift moves items from floor 0 to floor 1."""
    # Floor 0 input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Lift up
    sim.place_building(BuildingType.LIFT_UP, 4, 5, 0, Rotation.EAST)
    # Floor 1 output
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 1, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(7, 5, 1)

SCENARIO_26 = lambda: create_scenario(
    "Lift Up",
    "EXPECTED: Items move from floor 0 to floor 1. "
    "Input at floor 0, output at floor 1.",
    setup_lift_up
)


# =============================================================================
# SCENARIO 27: Lift Down
# =============================================================================
def setup_lift_down(sim):
    """Lift moves items from floor 1 to floor 0."""
    # Floor 1 input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 1, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 1, Rotation.EAST)
    # Lift down
    sim.place_building(BuildingType.LIFT_DOWN, 4, 5, 0, Rotation.EAST)
    # Floor 0 output
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim.set_input(2, 5, 1, "RuRuRuRu", 180.0)
    sim.set_output(7, 5, 0)

SCENARIO_27 = lambda: create_scenario(
    "Lift Down",
    "EXPECTED: Items move from floor 1 to floor 0. "
    "Input at floor 1, output at floor 0.",
    setup_lift_down
)


# =============================================================================
# SCENARIO 28: Multilevel routing with lifts
# =============================================================================
def setup_multilevel_routing(sim):
    """Complex routing across multiple floors."""
    # Floor 0: Input line going east
    for x in range(1, 5):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)
    # Lift to floor 1
    sim.place_building(BuildingType.LIFT_UP, 5, 5, 0, Rotation.EAST)
    # Floor 1: Continue east then down
    for x in range(6, 9):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 1, Rotation.EAST)
    # Lift back to floor 0
    sim.place_building(BuildingType.LIFT_DOWN, 9, 5, 0, Rotation.EAST)
    # Floor 0: Final output
    for x in range(10, 13):
        sim.place_building(BuildingType.BELT_FORWARD, x, 5, 0, Rotation.EAST)

    sim.set_input(1, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(13, 5, 0)

SCENARIO_28 = lambda: create_scenario(
    "Multilevel Routing",
    "EXPECTED: Items go floor0 -> lift up -> floor1 -> lift down -> floor0. "
    "Full 180/min throughput maintained.",
    setup_multilevel_routing
)


# =============================================================================
# SCENARIO 29: Multilevel processing (stacker example)
# =============================================================================
def setup_multilevel_processing(sim):
    """Two floors feeding into stacker."""
    # Floor 0 input line
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Floor 1 input - lifted from a different source
    sim.place_building(BuildingType.BELT_FORWARD, 1, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.LIFT_UP, 2, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 3, 1, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 1, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 1, Rotation.EAST)
    # Stacker at (4, 5) spans floors 0 and 1
    sim.place_building(BuildingType.STACKER, 4, 5, 0, Rotation.EAST)
    # Output on floor 0
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "Cu------", 30.0)  # Floor 0 - bottom layer
    sim.set_input(1, 3, 0, "--Ru----", 30.0)  # Lifts to floor 1 - top layer
    sim.set_output(7, 5, 0)

SCENARIO_29 = lambda: create_scenario(
    "Multilevel Processing",
    "EXPECTED: Floor 0 and floor 1 inputs feed stacker. "
    "Top layer comes from lifted route.",
    setup_multilevel_processing
)


# =============================================================================
# SCENARIO 30: Splitter Left (priority)
# =============================================================================
def setup_splitter_left(sim):
    """Splitter with left priority."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.SPLITTER_LEFT, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.EAST)  # North output (left priority)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.EAST)  # South output
    sim.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, 4, 0)  # North (left)
    sim.set_output(5, 6, 0)  # South (right)

SCENARIO_30 = lambda: create_scenario(
    "Splitter Left Priority",
    "EXPECTED: Splitter prioritizes left (north) output. "
    "Both outputs should get 90/min if both connected.",
    setup_splitter_left
)


# =============================================================================
# SCENARIO 31: Splitter Right (priority)
# =============================================================================
def setup_splitter_right(sim):
    """Splitter with right priority."""
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.SPLITTER_RIGHT, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.EAST)  # North output
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.EAST)  # South output (right priority)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(5, 4, 0)  # North
    sim.set_output(5, 6, 0)  # South (right)

SCENARIO_31 = lambda: create_scenario(
    "Splitter Right Priority",
    "EXPECTED: Splitter prioritizes right (south) output. "
    "Both outputs should get 90/min if both connected.",
    setup_splitter_right
)


# =============================================================================
# SCENARIO 32: Cutter facing WEST
# =============================================================================
def setup_cutter_west(sim):
    """Cutter rotated to face WEST."""
    # Cutter at (5, 5) facing WEST occupies (5,5) and (5,6)
    # Input at (5, 6) accepts from E, outputs at (5, 5) and (5, 6) go W
    # Input belts at y=6 (not y=5!)
    sim.place_building(BuildingType.BELT_FORWARD, 7, 6, 0, Rotation.WEST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 6, 0, Rotation.WEST)
    # Cutter facing west
    sim.place_building(BuildingType.CUTTER, 5, 5, 0, Rotation.WEST)
    # Outputs to west - at (4, 5) and (4, 6)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.WEST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.WEST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.WEST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.WEST)

    sim.set_input(7, 6, 0, "CuCuCuCu", 180.0)
    sim.set_output(2, 5, 0)
    sim.set_output(2, 6, 0)

SCENARIO_32 = lambda: create_scenario(
    "Cutter facing WEST",
    "EXPECTED: Cutter works correctly when rotated 180°. "
    "Input from east, outputs to west.",
    setup_cutter_west
)


# =============================================================================
# SCENARIO 33: Cutter facing NORTH
# =============================================================================
def setup_cutter_north(sim):
    """Cutter rotated to face NORTH."""
    # Cutter at (5, 6) facing NORTH occupies (5,6) and (6,6) - horizontal
    # Input at (6, 6) accepts from S, outputs at (5, 6) and (6, 6) go N
    # Input belt at x=6, y=7 (not x=5!)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 8, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 7, 0, Rotation.NORTH)
    # Cutter facing north
    sim.place_building(BuildingType.CUTTER, 5, 6, 0, Rotation.NORTH)
    # Outputs to north - at (5, 5) and (6, 5)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 5, 0, Rotation.NORTH)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 6, 4, 0, Rotation.EAST)

    sim.set_input(6, 8, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 4, 0)
    sim.set_output(7, 4, 0)

SCENARIO_33 = lambda: create_scenario(
    "Cutter facing NORTH",
    "EXPECTED: Cutter works correctly when rotated 270°. "
    "Input from south, outputs to north.",
    setup_cutter_north
)


# =============================================================================
# SCENARIO 34: Complex factory - cutter -> stacker
# =============================================================================
def setup_cutter_stacker_combo(sim):
    """Cut shape in half, stack halves back together."""
    # Input
    sim.place_building(BuildingType.BELT_FORWARD, 1, 5, 0, Rotation.EAST)
    # Cutter
    sim.place_building(BuildingType.CUTTER, 2, 5, 0, Rotation.EAST)
    # Left half goes straight
    sim.place_building(BuildingType.BELT_FORWARD, 3, 5, 0, Rotation.EAST)
    # Right half goes down then east
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.EAST)
    # Lift left half to floor 1
    sim.place_building(BuildingType.LIFT_UP, 4, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 1, Rotation.SOUTH)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 1, Rotation.EAST)
    # Right half continues on floor 0
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.EAST)
    # Stacker combines them
    sim.place_building(BuildingType.STACKER, 6, 6, 0, Rotation.EAST)
    # Output
    sim.place_building(BuildingType.BELT_FORWARD, 7, 6, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 8, 6, 0, Rotation.EAST)

    sim.set_input(1, 5, 0, "CuRuGrBl", 45.0)  # Full shape
    sim.set_output(9, 6, 0)

SCENARIO_34 = lambda: create_scenario(
    "Cutter -> Stacker Combo",
    "EXPECTED: Shape cut in half, halves route to different floors, "
    "then stack back together.",
    setup_cutter_stacker_combo
)


# =============================================================================
# SCENARIO 35: Three-way split using splitters
# =============================================================================
def setup_three_way_split(sim):
    """Chain splitters for 3-way split."""
    # Input
    sim.place_building(BuildingType.BELT_FORWARD, 2, 5, 0, Rotation.EAST)
    # First splitter - 50/50
    sim.place_building(BuildingType.SPLITTER, 3, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 3, 4, 0, Rotation.EAST)  # North 50%
    sim.place_building(BuildingType.BELT_FORWARD, 3, 6, 0, Rotation.EAST)  # South 50%
    # Second splitter on north branch
    sim.place_building(BuildingType.SPLITTER, 4, 4, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 4, 3, 0, Rotation.EAST)  # 25%
    sim.place_building(BuildingType.BELT_FORWARD, 4, 5, 0, Rotation.EAST)  # 25%
    # Continue south branch
    sim.place_building(BuildingType.BELT_FORWARD, 4, 6, 0, Rotation.EAST)  # 50%
    # Outputs
    sim.place_building(BuildingType.BELT_FORWARD, 5, 3, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 5, 0, Rotation.EAST)
    sim.place_building(BuildingType.BELT_FORWARD, 5, 6, 0, Rotation.EAST)

    sim.set_input(2, 5, 0, "CuCuCuCu", 180.0)
    sim.set_output(6, 3, 0)  # 45/min
    sim.set_output(6, 5, 0)  # 45/min
    sim.set_output(6, 6, 0)  # 90/min

SCENARIO_35 = lambda: create_scenario(
    "Three-way Split",
    "EXPECTED: 180 in -> first split 90/90 -> second split 45/45 + 90. "
    "Three outputs: 45, 45, 90.",
    setup_three_way_split
)


# =============================================================================
# ALL SCENARIOS
# =============================================================================
ALL_SCENARIOS = [
    SCENARIO_1,
    SCENARIO_2,
    SCENARIO_3,
    SCENARIO_4,
    SCENARIO_5,
    SCENARIO_6,
    SCENARIO_7,
    SCENARIO_8,
    SCENARIO_9,
    SCENARIO_10,
    SCENARIO_11,
    SCENARIO_12,
    SCENARIO_13,
    SCENARIO_14,
    SCENARIO_15,
    SCENARIO_16,
    SCENARIO_17,
    SCENARIO_18,
    SCENARIO_19,
    SCENARIO_20,
    SCENARIO_21,
    SCENARIO_22,
    SCENARIO_23,
    SCENARIO_24,
    SCENARIO_25,
    SCENARIO_26,
    SCENARIO_27,
    SCENARIO_28,
    SCENARIO_29,
    SCENARIO_30,
    SCENARIO_31,
    SCENARIO_32,
    SCENARIO_33,
    SCENARIO_34,
    SCENARIO_35,
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
