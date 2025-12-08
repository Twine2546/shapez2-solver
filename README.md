# Shapez 2 Solver

An evolutionary algorithm-based solver for Shapez 2 shape puzzles. Given an input shape and desired output shape, the solver evolves a solution using basic shape operations.

## Features

- **CP-SAT Solver with Throughput Optimization**: Google OR-Tools constraint solver finds optimal solutions with maximum throughput
  - Automatic machine selection (splitters vs cutters for 4x throughput improvement)
  - Multi-tree routing for independent input streams
  - Blueprint code export for direct import into Shapez 2
  - Guaranteed routing or automatic foundation size scaling
- **Shape Code Parser**: Full support for Shapez 2 shape codes
- **All Operations**: Cutter, half destroyer, swapper, rotator, stacker, unstacker, painter, crystal generator, pin pusher
- **Foundation Evolution**: Evolve building layouts on foundations with multiple input/output ports
- **All Foundation Sizes**: Support for 1x1 through 3x3, plus irregular shapes (T, L, S, Cross)
- **Multiple Algorithms**: CP-SAT (optimal), Evolutionary, Simulated Annealing, Hybrid, Two-Phase
- **Graphical Interface**: Integrated GUI with visual layout and blueprint export
- **Layout Viewer**: Visual display of foundation layouts with pan, zoom, and floor navigation

## Installation

```bash
# Clone the repository
git clone https://github.com/Twine2546/shapez2-solver.git
cd shapez2-solver

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- pygame-ce (for GUI)
- pygame_gui (for GUI)

## Usage

### CP-SAT GUI (Recommended for Foundation Solving)

```bash
python3 run_cpsat_gui.py
```

The CP-SAT GUI provides:
- **Optimal Solutions**: Uses Google OR-Tools constraint programming for guaranteed optimal/near-optimal solutions
- **Maximum Throughput**: Automatically optimizes for highest items/min with fully upgraded equipment
  - Uses splitters (180 items/min) for pure splitting = 4x faster than cutters!
  - Uses cutters (45 items/min) only when shape transformation needed
  - Creates independent trees for multiple inputs
- **Blueprint Export**: Copy solution directly to clipboard or console for import into Shapez 2
- **Visual Layout**: View the complete solution with buildings and routing
- **Example**: 12 inputs → 48 outputs creates 36 cutters for 540 items/min total throughput

### Full GUI Mode

```bash
python3 main.py gui
# or just
python3 main.py
```

The full GUI has multiple modes and algorithms:
- **Shape Transform**: Evolve a sequence of operations to transform one shape into another
- **Foundation Evolution**: Choose from CP-SAT, Evolutionary, Simulated Annealing, Hybrid, or Two-Phase algorithms

### Foundation Evolution CLI

For foundation-specific evolution with multiple ports:

```bash
# Simple corner splitter on 2x2 foundation
python3 evolve_foundation.py --foundation 2x2 \
    --input "W,0,0,CuCuCuCu" \
    --output "E,0,0,Cu------" \
    --output "E,0,1,--Cu----" \
    --output "E,1,0,----Cu--" \
    --output "E,1,1,------Cu"

# Interactive mode
python3 evolve_foundation.py --interactive

# List available foundations
python3 evolve_foundation.py --list-foundations

# Show shape code format
python3 evolve_foundation.py --list-shapes
```

### Shape Transform CLI

```bash
# Solve a shape transformation
python3 main.py solve -i CuCuCuCu -t CrCrCrCr -p 50 -g 100

# Parse and display a shape code
python3 main.py parse CrRgSbWy:RuRuRuRu

# List available foundation types
python3 main.py foundations
```

## Foundation Port Format

Ports are specified as: `SIDE,POSITION,FLOOR,SHAPE_CODE`

- **SIDE**: N (North), E (East), S (South), W (West)
- **POSITION**: Port index (0-3 for first 1x1 unit, 4-7 for second, etc.)
- **FLOOR**: 0-3
- **SHAPE_CODE**: Shape code for the port

### Examples
```
W,0,0,CuCuCuCu    # West side, position 0, floor 0, full circle
E,1,2,Cu------    # East side, position 1, floor 2, NE corner only
```

## Shape Code Format

Shapes are encoded as strings with layers separated by `:` (bottom to top).

Each layer has 4 parts (quadrants): NE, NW, SW, SE (starting top-right, going counter-clockwise).

Each part is 2 characters: shape type + color.

### Shape Types
- `C` - Circle
- `R` - Square
- `S` - Star
- `W` - Diamond
- `P` - Pin (no color)
- `c` - Crystal
- `-` - Empty

### Colors
- `u` - Uncolored
- `r` - Red
- `g` - Green
- `b` - Blue
- `c` - Cyan
- `m` - Magenta
- `y` - Yellow
- `w` - White

### Shape Code Examples
```
CuCuCuCu          # Full circle (all 4 quadrants)
Cu------          # NE corner only
--Cu----          # NW corner only
----Cu--          # SW corner only
------Cu          # SE corner only
CrCgCbCy          # Circle with 4 different colors
CuCuCuCu:RrRrRrRr # Two layers: circles bottom, red squares top
```

## Foundation Sizes

| Name  | Units | Grid Size | Ports/Side |
|-------|-------|-----------|------------|
| 1x1   | 1x1   | 14x14     | 4          |
| 2x1   | 2x1   | 34x14     | N=8, E=4   |
| 2x2   | 2x2   | 34x34     | 8          |
| 3x3   | 3x3   | 54x54     | 12         |
| T     | 3x2   | 54x34     | N=12, E=8  |
| L     | 2x2   | 34x34     | 8          |
| Cross | 3x3   | 54x54     | 12         |

Each 1x1 unit = 14x14 internal grid tiles. Additional units add 20 tiles per axis.

## GUI Controls

### Main Window
- **Mode Dropdown**: Switch between Shape Transform and Foundation Evolution
- **Start Evolution**: Begin the evolutionary search
- **View Layout**: Open layout viewer (Foundation mode, after evolution)

### Layout Viewer
- **Arrow Keys / WASD**: Pan view
- **+/- / Mouse Scroll**: Zoom in/out
- **PgUp / PgDown**: Change floor
- **Tab**: Cycle through solutions
- **Esc**: Close viewer
- **Click + Drag**: Pan view

## Project Structure

```
shapez2-solver/
├── main.py                    # Main entry point
├── evolve_foundation.py       # Foundation evolution CLI
├── requirements.txt           # Dependencies
├── shapez2_solver/
│   ├── shapes/               # Shape representation
│   ├── operations/           # Shape operations
│   ├── foundations/          # Platform definitions
│   ├── simulator/            # Design simulation
│   ├── evolution/            # Evolutionary algorithms
│   │   ├── algorithm.py      # Shape transform evolution
│   │   ├── foundation_config.py    # Foundation specs
│   │   └── foundation_evolution.py # Foundation evolution
│   ├── visualization/        # Graphics
│   │   ├── pygame_layout_viewer.py # Layout viewer
│   │   └── ...
│   ├── blueprint/            # Blueprint export
│   └── ui/                   # GUI application
└── tests/                    # Unit tests
```

## License

MIT
