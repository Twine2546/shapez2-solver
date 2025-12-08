"""Blueprint code encoder for Shapez 2."""

import base64
import gzip
import json
from typing import Dict, List, Any

from .placer import PlacedBuilding
from .router import BeltSegment, Route
from .building_types import BuildingType, Rotation


class BlueprintEncoder:
    """Encodes placements and routes into Shapez 2 blueprint format."""

    BLUEPRINT_VERSION = 1122  # Current game version (1105-1220 supported)
    FORMAT_VERSION = 3  # Blueprint format version (3 is current)

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add_building(self, placement: PlacedBuilding) -> None:
        """Add a building to the blueprint.

        For multi-floor buildings (like stackers), we only add one entry
        at the base position - the game handles the vertical extent.
        """
        from .building_types import BUILDING_SPECS, BuildingSpec

        spec = BUILDING_SPECS.get(placement.building_type, BuildingSpec(1, 1, 1, 1, 1, 1, 30, 90))

        entry = {
            "T": placement.building_type.value,
            "X": placement.x,
            "Y": placement.y,
            "L": placement.layer,  # Base layer - multi-floor buildings extend upward
            "R": placement.rotation.value,
        }
        self.entries.append(entry)

    def add_belt(self, segment: BeltSegment) -> None:
        """Add a belt segment to the blueprint."""
        entry = {
            "T": segment.belt_type.value,
            "X": segment.x,
            "Y": segment.y,
            "L": segment.layer,
            "R": segment.rotation.value,
        }
        self.entries.append(entry)

    def add_placements(self, placements: Dict[str, PlacedBuilding]) -> None:
        """Add all placements to the blueprint."""
        for node_id, placement in placements.items():
            # Skip input/output markers (they're just for routing)
            if node_id.startswith("in_") or node_id.startswith("out_"):
                continue
            self.add_building(placement)

    def add_routes(self, routes: List[Route]) -> None:
        """Add all belt segments from routes."""
        for route in routes:
            for segment in route.segments:
                self.add_belt(segment)

    def encode(self) -> str:
        """
        Encode the blueprint to a Shapez 2 blueprint code.

        Returns:
            Blueprint code string (SHAPEZ2-1-xxxxx$)
        """
        # Build the blueprint JSON structure
        # Icon.Data contains 4 shape codes (null = no icon)
        blueprint = {
            "V": self.BLUEPRINT_VERSION,
            "BP": {
                "$type": "Building",
                "Icon": {"Data": [None, None, None, None]},
                "Entries": self.entries,
                "BinaryVersion": self.BLUEPRINT_VERSION
            }
        }

        # Convert to JSON
        json_str = json.dumps(blueprint, separators=(',', ':'))

        # Compress with gzip
        compressed = gzip.compress(json_str.encode('utf-8'))

        # Encode as base64
        encoded = base64.b64encode(compressed).decode('ascii')

        # Format as blueprint code
        return f"SHAPEZ2-{self.FORMAT_VERSION}-{encoded}$"

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()

    @staticmethod
    def decode(blueprint_code: str) -> Dict[str, Any]:
        """
        Decode a Shapez 2 blueprint code.

        Args:
            blueprint_code: The blueprint code string

        Returns:
            Decoded blueprint JSON object
        """
        # Strip prefix and suffix
        code = blueprint_code.strip()
        if code.endswith('$'):
            code = code[:-1]

        parts = code.split('-', 2)
        if len(parts) < 3 or parts[0] != "SHAPEZ2":
            raise ValueError("Invalid blueprint code format")

        data = parts[2]

        # Pad base64 if needed
        while len(data) % 4 != 0:
            data += '='

        # Decode base64
        binary = base64.b64decode(data)

        # Decompress gzip
        json_data = gzip.decompress(binary)

        # Parse JSON
        return json.loads(json_data)
