# Shapez 2 Solver

An evolutionary algorithm-based solver for Shapez 2 shape puzzles. Given an input shape and desired output shape, the solver evolves a solution using basic shape operations.

## Features

- **Shape Code Parser**: Full support for Shapez 2 shape codes
- **All Operations**: Cutter, half destroyer, swapper, rotator, stacker, unstacker, painter, crystal generator, pin pusher
- **Foundation Support**: All foundation sizes from 1x1 to 3x3
- **Evolutionary Algorithm**: Population-based search with configurable parameters
- **Graphical Visualization**: Real-time view of evolution progress (requires pygame)
- **Decoupled Architecture**: Simulator independent of search algorithm

## Installation

```bash
# Clone the repository
git clone https://github.com/Twine2546/shapez2-solver.git
cd shapez2-solver

# Install dependencies (optional, for GUI)
pip install -r requirements.txt
```

## Usage

### GUI Mode (requires pygame)

```bash
python3 main.py gui
```

### CLI Mode

```bash
# Solve a shape transformation
python3 main.py solve -i CuCuCuCu -t CrCrCrCr -p 50 -g 100

# Parse and display a shape code
python3 main.py parse CrRgSbWy:RuRuRuRu

# List available foundation types
python3 main.py foundations
```

### CLI Options

```
-i, --input       Input shape code (e.g., CuCuCuCu)
-t, --target      Target shape code (e.g., CrCrCrCr)
-f, --foundation  Foundation type (default: 1x1)
-p, --population  Population size (default: 50)
-g, --generations Number of generations (default: 100)
-m, --mutation-rate Mutation rate (default: 0.3)
```

## Shape Code Format

Shapes are encoded as strings with layers separated by `:` (bottom to top).

Each layer has 4 parts (quadrants) starting from top-right, going clockwise.

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

### Examples
- `CuCuCuCu` - Four uncolored circles
- `CrRgSbWy` - Circle(red), Square(green), Star(blue), Diamond(yellow)
- `CuCuCuCu:RrRrRrRr` - Two layers: uncolored circles on bottom, red squares on top

## Project Structure

```
shapez2-solver/
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
├── shapez2_solver/
│   ├── shapes/               # Shape representation
│   ├── operations/           # Shape operations
│   ├── foundations/          # Platform definitions
│   ├── simulator/            # Design simulation
│   ├── evolution/            # Evolutionary algorithm
│   ├── visualization/        # Graphics (pygame)
│   └── ui/                   # GUI application
└── tests/                    # Unit tests
```

## License

MIT
