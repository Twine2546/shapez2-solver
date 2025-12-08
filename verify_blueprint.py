#!/usr/bin/env python3
"""Verify blueprint structure and visualize layout."""

import sys
sys.path.insert(0, '/config/projects/programming/games/shape2')

from shapez2_solver.blueprint.encoder import BlueprintEncoder


def visualize_blueprint(blueprint_code: str, title: str = "Blueprint"):
    """Visualize a blueprint as ASCII art."""
    decoded = BlueprintEncoder.decode(blueprint_code)
    entries = decoded['BP']['Entries']

    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"Version: {decoded['V']}")
    print(f"Entries: {len(entries)}")

    # Organize by floor
    floors = {}
    for entry in entries:
        floor = entry.get('L', 0)
        if floor not in floors:
            floors[floor] = []
        floors[floor].append(entry)

    # Symbol mapping
    symbols = {
        'BeltDefaultForwardInternalVariant': '→',
        'BeltDefaultLeftInternalVariant': '↰',
        'BeltDefaultRightInternalVariant': '↱',
        'RotatorOneQuadInternalVariant': 'R',
        'RotatorOneQuadCCWInternalVariant': 'r',
        'CutterDefaultInternalVariant': 'C',
        'CutterHalfInternalVariant': 'H',
        'HalvesSwapperDefaultInternalVariant': 'X',
        'StackerStraightInternalVariant': 'S',
    }

    # Rotation arrows for belts
    belt_arrows = {
        ('BeltDefaultForwardInternalVariant', 0): '→',  # East
        ('BeltDefaultForwardInternalVariant', 1): '↓',  # South
        ('BeltDefaultForwardInternalVariant', 2): '←',  # West
        ('BeltDefaultForwardInternalVariant', 3): '↑',  # North
        ('BeltDefaultLeftInternalVariant', 0): '↱',
        ('BeltDefaultLeftInternalVariant', 1): '↰',
        ('BeltDefaultLeftInternalVariant', 2): '↲',
        ('BeltDefaultLeftInternalVariant', 3): '↳',
        ('BeltDefaultRightInternalVariant', 0): '↳',
        ('BeltDefaultRightInternalVariant', 1): '↱',
        ('BeltDefaultRightInternalVariant', 2): '↰',
        ('BeltDefaultRightInternalVariant', 3): '↲',
    }

    for floor in sorted(floors.keys()):
        floor_entries = floors[floor]
        print(f"\nFloor {floor}:")

        # Find bounds
        min_x = min(e['X'] for e in floor_entries)
        max_x = max(e['X'] for e in floor_entries)
        min_y = min(e['Y'] for e in floor_entries)
        max_y = max(e['Y'] for e in floor_entries)

        # Add margin
        min_x -= 1
        max_x += 2
        min_y -= 1
        max_y += 2

        # Create grid
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        grid = [['·' for _ in range(width)] for _ in range(height)]

        # Place entries
        for entry in floor_entries:
            x = entry['X'] - min_x
            y = entry['Y'] - min_y
            t = entry['T']
            r = entry.get('R', 0)

            # Get symbol
            symbol = belt_arrows.get((t, r), symbols.get(t, '?'))

            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = symbol

        # Print grid with coordinates
        print(f"  X: {min_x} to {max_x}, Y: {min_y} to {max_y}")
        print("  " + "".join(str(i % 10) for i in range(min_x, max_x + 1)))
        for row_idx, row in enumerate(grid):
            print(f"{min_y + row_idx:2d} {''.join(row)}")

    # Print entry details
    print(f"\nAll entries:")
    for i, entry in enumerate(entries):
        t = entry['T'].replace('InternalVariant', '')
        print(f"  [{i:2d}] {t:30s} at ({entry['X']:2d}, {entry['Y']:2d}, L{entry.get('L', 0)}) R={entry.get('R', 0)}")


def main():
    # Simple corner splitter
    simple_bp = "SHAPEZ2-3-H4sIAP9INWkC/62UMWvDMBCF/8uRUUNlJRk0pmkhEEhrgkkpHo5aSQXiXJQTIRj/98hphwZsN6Aub5He9yTecQ0UoKXMMgGLF9ANTPj8ZUDDIlhXWTqAgNVHTd3REhlBv1NwTtxK2Qp4IvbWHOOFBrYdwDhemj0Gx8+1P6GvVsTGE7oCvUXiSN6BfhDwdtX1VfOorfgmPAaOhh9Gr1kOm++Mz4YJec3Itd+QeQ3Y71bJ+dOU78+S4+f/RJB/EHJ7+OSRAmRyhdkvguwhrM1+7AGqz373BKjkCVApE6CS+0snTG8IZVwnltCfC+OPtlsf3Y5p2wvzWQwObwQAAA==$"

    # 3-floor corner splitter
    three_floor_bp = "SHAPEZ2-3-H4sIAP9INWkC/63XwUrDQBAG4HcZPO6hM6kecqxVKBTUIEGRHha7rQvLpmwnSAl5d5OqUCGthb+XuSTzT8IuH0xDJeXMIoYmj5Q3dKW7jaOcJrUPSx/XZGj2XsX+0dSqpfwt1iGYv2XRGrqLmrzbdi809NwHuKBTt7J10Psqfdq0nEV1KdpQ2uRt1C75hfKRodd9ne9r0dXWfCfc1to1/GQMNvPx5jPHy/GEolKrVXqI7qm2w90ZPH+M/P41PP7mQgn8T0Lh1x964gAYPkI5SOCBhLlbnfqAbKj97BuQwTcgQ25ABp8fnjAGEg4NYMQAhg1gyACGDWDEAIYNYNgABg1g2ADGDGDIAIYNYMQAhg1g2ACGDRDEAIENEMgAgQ0QxACBDRDYAAENENgAwQwQyACBDRDEAIENENiA34RFt1b4aNOudGnr+zWi3zXa9gtLXPg/dwwAAA==$"

    visualize_blueprint(simple_bp, "Simple Corner Splitter (1 floor)")
    visualize_blueprint(three_floor_bp, "3-Floor Corner Splitter")


if __name__ == "__main__":
    main()
