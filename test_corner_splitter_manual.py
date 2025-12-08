#!/usr/bin/env python3
"""Generate a corner splitter blueprint with manual placement for better layout."""

import sys
sys.path.insert(0, '/config/projects/programming/games/shape2')

from shapez2_solver.blueprint.encoder import BlueprintEncoder
from shapez2_solver.blueprint.building_types import BuildingType, Rotation


def create_corner_splitter_blueprint():
    """
    Create a corner splitter blueprint with manual placement.

    Layout (flow left to right):

         [IN] → [Cutter1] ─→ [Rotate1] → [Cutter2] → [Out0]
                    │                          └→ [Out1]
                    └────→ [Rotate2] → [Cutter3] → [Out2]
                                             └→ [Out3]

    Position coordinates:
    - Input belt at X=0
    - Cutter 1 at X=2 (1x2 building)
    - Rotators at X=5
    - Cutters 2,3 at X=8
    - Output belts at X=11

    Y positions:
    - Top line (east half): Y=0
    - Bottom line (west half): Y=4
    """
    encoder = BlueprintEncoder()

    # Input belt
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 0, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 1, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })

    # First cutter at X=2, Y=2 (note: cutter is 1x2, outputs at Y and Y+1)
    encoder.entries.append({
        "T": BuildingType.CUTTER.value,
        "X": 2, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })

    # Belt from cutter output 0 (Y=2) to rotator 1
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 3, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 4, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })

    # Belt from cutter output 1 (Y=3) down to rotator 2
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 3, "Y": 3, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_RIGHT.value,
        "X": 4, "Y": 3, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 4, "Y": 4, "L": 0, "R": Rotation.SOUTH.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 4, "Y": 5, "L": 0, "R": Rotation.SOUTH.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_LEFT.value,
        "X": 4, "Y": 6, "L": 0, "R": Rotation.SOUTH.value
    })

    # Rotator 1 (CW) at X=5, Y=2
    encoder.entries.append({
        "T": BuildingType.ROTATOR_CW.value,
        "X": 5, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })

    # Rotator 2 (CW) at X=5, Y=6
    encoder.entries.append({
        "T": BuildingType.ROTATOR_CW.value,
        "X": 5, "Y": 6, "L": 0, "R": Rotation.EAST.value
    })

    # Belts from rotator 1 to cutter 2
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 6, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 7, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })

    # Belts from rotator 2 to cutter 3
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 6, "Y": 6, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 7, "Y": 6, "L": 0, "R": Rotation.EAST.value
    })

    # Cutter 2 at X=8, Y=2 (outputs at Y=2 and Y=3)
    encoder.entries.append({
        "T": BuildingType.CUTTER.value,
        "X": 8, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })

    # Cutter 3 at X=8, Y=6 (outputs at Y=6 and Y=7)
    encoder.entries.append({
        "T": BuildingType.CUTTER.value,
        "X": 8, "Y": 6, "L": 0, "R": Rotation.EAST.value
    })

    # Output belts from cutter 2
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 9, "Y": 2, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 9, "Y": 3, "L": 0, "R": Rotation.EAST.value
    })

    # Output belts from cutter 3
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 9, "Y": 6, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 9, "Y": 7, "L": 0, "R": Rotation.EAST.value
    })

    return encoder.encode()


def create_simple_corner_splitter():
    """
    Create an even simpler, more compact corner splitter.

    Layout:
        [Belt] → [Cutter] → [Rotator] → [Cutter] → [Out0]
                     │                       └──→ [Out1]
                     └──→ [Rotator] → [Cutter] → [Out2]
                                           └──→ [Out3]

    Buildings positioned in straight lines with minimal belt routing.
    """
    encoder = BlueprintEncoder()

    # Row 0 (Y=0): Input → Cutter1 → Belt → Rotator1 → Belt → Cutter2 → Outputs
    # Row 1 (Y=1): Cutter1 secondary output
    # Row 3 (Y=3): Second path: Belt → Rotator2 → Belt → Cutter3 → Outputs
    # Row 4 (Y=4): Cutter3 secondary output

    # --- Top processing line ---
    # Input at (0,0)
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 0, "Y": 0, "L": 0, "R": Rotation.EAST.value
    })

    # Cutter 1 at (1,0) - outputs at (2,0) and (2,1)
    encoder.entries.append({
        "T": BuildingType.CUTTER.value,
        "X": 1, "Y": 0, "L": 0, "R": Rotation.EAST.value
    })

    # Belt from top output (2,0) to Rotator 1
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 2, "Y": 0, "L": 0, "R": Rotation.EAST.value
    })

    # Rotator 1 at (3,0)
    encoder.entries.append({
        "T": BuildingType.ROTATOR_CW.value,
        "X": 3, "Y": 0, "L": 0, "R": Rotation.EAST.value
    })

    # Belt to Cutter 2
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 4, "Y": 0, "L": 0, "R": Rotation.EAST.value
    })

    # Cutter 2 at (5,0) - outputs at (6,0) and (6,1)
    encoder.entries.append({
        "T": BuildingType.CUTTER.value,
        "X": 5, "Y": 0, "L": 0, "R": Rotation.EAST.value
    })

    # Output belts from Cutter 2
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 6, "Y": 0, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 6, "Y": 1, "L": 0, "R": Rotation.EAST.value
    })

    # --- Bottom processing line (from Cutter 1's secondary output) ---
    # Belt from (2,1) going down then right
    encoder.entries.append({
        "T": BuildingType.BELT_RIGHT.value,
        "X": 2, "Y": 1, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 2, "Y": 2, "L": 0, "R": Rotation.SOUTH.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_LEFT.value,
        "X": 2, "Y": 3, "L": 0, "R": Rotation.SOUTH.value
    })

    # Rotator 2 at (3,3)
    encoder.entries.append({
        "T": BuildingType.ROTATOR_CW.value,
        "X": 3, "Y": 3, "L": 0, "R": Rotation.EAST.value
    })

    # Belt to Cutter 3
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 4, "Y": 3, "L": 0, "R": Rotation.EAST.value
    })

    # Cutter 3 at (5,3) - outputs at (6,3) and (6,4)
    encoder.entries.append({
        "T": BuildingType.CUTTER.value,
        "X": 5, "Y": 3, "L": 0, "R": Rotation.EAST.value
    })

    # Output belts from Cutter 3
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 6, "Y": 3, "L": 0, "R": Rotation.EAST.value
    })
    encoder.entries.append({
        "T": BuildingType.BELT_FORWARD.value,
        "X": 6, "Y": 4, "L": 0, "R": Rotation.EAST.value
    })

    return encoder.encode()


def create_3_floor_corner_splitter():
    """
    Create a 3-floor corner splitter (one per floor).

    Each floor has:
    Input → Cutter → Rotator → Cutter → 2 outputs
                └→ Rotator → Cutter → 2 outputs

    3 inputs total, 12 outputs total.
    """
    encoder = BlueprintEncoder()

    for floor in range(3):
        # --- Top processing line ---
        # Input at (0,0)
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 0, "Y": 0, "L": floor, "R": Rotation.EAST.value
        })

        # Cutter 1 at (1,0)
        encoder.entries.append({
            "T": BuildingType.CUTTER.value,
            "X": 1, "Y": 0, "L": floor, "R": Rotation.EAST.value
        })

        # Belt from top output to Rotator 1
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 2, "Y": 0, "L": floor, "R": Rotation.EAST.value
        })

        # Rotator 1 at (3,0)
        encoder.entries.append({
            "T": BuildingType.ROTATOR_CW.value,
            "X": 3, "Y": 0, "L": floor, "R": Rotation.EAST.value
        })

        # Belt to Cutter 2
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 4, "Y": 0, "L": floor, "R": Rotation.EAST.value
        })

        # Cutter 2 at (5,0)
        encoder.entries.append({
            "T": BuildingType.CUTTER.value,
            "X": 5, "Y": 0, "L": floor, "R": Rotation.EAST.value
        })

        # Output belts from Cutter 2
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 6, "Y": 0, "L": floor, "R": Rotation.EAST.value
        })
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 6, "Y": 1, "L": floor, "R": Rotation.EAST.value
        })

        # --- Bottom processing line ---
        # Belt from (2,1) going down then right
        encoder.entries.append({
            "T": BuildingType.BELT_RIGHT.value,
            "X": 2, "Y": 1, "L": floor, "R": Rotation.EAST.value
        })
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 2, "Y": 2, "L": floor, "R": Rotation.SOUTH.value
        })
        encoder.entries.append({
            "T": BuildingType.BELT_LEFT.value,
            "X": 2, "Y": 3, "L": floor, "R": Rotation.SOUTH.value
        })

        # Rotator 2 at (3,3)
        encoder.entries.append({
            "T": BuildingType.ROTATOR_CW.value,
            "X": 3, "Y": 3, "L": floor, "R": Rotation.EAST.value
        })

        # Belt to Cutter 3
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 4, "Y": 3, "L": floor, "R": Rotation.EAST.value
        })

        # Cutter 3 at (5,3)
        encoder.entries.append({
            "T": BuildingType.CUTTER.value,
            "X": 5, "Y": 3, "L": floor, "R": Rotation.EAST.value
        })

        # Output belts from Cutter 3
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 6, "Y": 3, "L": floor, "R": Rotation.EAST.value
        })
        encoder.entries.append({
            "T": BuildingType.BELT_FORWARD.value,
            "X": 6, "Y": 4, "L": floor, "R": Rotation.EAST.value
        })

    return encoder.encode()


def main():
    print("=" * 70)
    print("Corner Splitter Blueprint Generator - Manual Placement")
    print("=" * 70)

    print("\n### SIMPLE CORNER SPLITTER (1 floor) ###")
    print("Takes 1 input, produces 4 corner outputs")
    print("-" * 70)
    blueprint1 = create_simple_corner_splitter()
    print(blueprint1)

    # Decode and show entries
    decoded = BlueprintEncoder.decode(blueprint1)
    print(f"\nEntries: {len(decoded['BP']['Entries'])}")

    print("\n\n### 3-FLOOR CORNER SPLITTER ###")
    print("Takes 3 inputs (one per floor), produces 12 corner outputs")
    print("-" * 70)
    blueprint3 = create_3_floor_corner_splitter()
    print(blueprint3)

    decoded3 = BlueprintEncoder.decode(blueprint3)
    print(f"\nEntries: {len(decoded3['BP']['Entries'])}")


if __name__ == "__main__":
    main()
