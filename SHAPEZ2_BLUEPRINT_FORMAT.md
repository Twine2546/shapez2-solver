# Shapez 2 Blueprint Format Reference

## Blueprint String Format

```
SHAPEZ2-{format_version}-{base64_gzip_json}$
```

- **Format Version**: Currently `3`
- **Data**: Base64-encoded gzip-compressed JSON
- **Terminator**: `$`

## JSON Structure

```json
{
  "V": 1122,
  "BP": {
    "$type": "Building",
    "Icon": {"Data": [null, null, null, null]},
    "Entries": [...],
    "BinaryVersion": 1122
  }
}
```

### Fields
- **V**: Game version (1105-1220 supported as of Dec 2025)
- **$type**: "Building" or "Island"
- **BinaryVersion**: Same as V
- **Icon.Data**: Array of 4 icon identifiers (can be null, "icon:Name", or "shape:CODE")

## Entry Structure

```json
{
  "T": "BuildingTypeInternalVariant",
  "X": 0,
  "Y": 0,
  "L": 0,
  "R": 0
}
```

- **T**: Building type (required)
- **X, Y**: Position (optional, default 0)
- **L**: Layer/floor (optional, default 0)
- **R**: Rotation (0=East, 1=South, 2=West, 3=North)

## Correct Building Type Names

| Building | Internal Type Name |
|----------|-------------------|
| Belt Forward | `BeltDefaultForwardInternalVariant` |
| Belt Left | `BeltDefaultLeftInternalVariant` |
| Belt Right | `BeltDefaultRightInternalVariant` |
| Rotator CW | `RotatorOneQuadInternalVariant` |
| Rotator CCW | `RotatorOneQuadCCWInternalVariant` |
| Cutter | `CutterDefaultInternalVariant` |
| Cutter Mirrored | `CutterDefaultInternalVariantMirrored` |
| Half Cutter | `CutterHalfInternalVariant` |
| Swapper | `HalvesSwapperDefaultInternalVariant` |
| Stacker | `StackerStraightInternalVariant` |
| Belt Port Sender | `BeltPortSenderInternalVariant` |
| Belt Port Receiver | `BeltPortReceiverInternalVariant` |
| Splitter 1-to-3 | `Splitter1To3InternalVariant` |
| Splitter T-Shape | `SplitterTShapeInternalVariant` |

## Python Encoder

```python
import base64
import gzip
import json

def encode_blueprint(entries, game_version=1122):
    blueprint = {
        "V": game_version,
        "BP": {
            "$type": "Building",
            "Icon": {"Data": [None, None, None, None]},
            "Entries": entries,
            "BinaryVersion": game_version
        }
    }
    json_str = json.dumps(blueprint, separators=(',', ':'))
    compressed = gzip.compress(json_str.encode('utf-8'))
    encoded = base64.b64encode(compressed).decode('ascii')
    return f"SHAPEZ2-3-{encoded}$"
```

## Example: Single Cutter

```python
entries = [{"T": "CutterDefaultInternalVariant", "X": 0, "Y": 0, "R": 0}]
print(encode_blueprint(entries))
```

## Multi-Floor Buildings

Use the `L` field to specify floor (0, 1, 2, etc.). Connect floors with lifts or stackers.

---
*Last updated: 2025-12-07*
