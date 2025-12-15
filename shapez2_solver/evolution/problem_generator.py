"""
Programmatic problem generator for training ML models.

Generates diverse problems covering:
- All foundation types (rectangular and irregular)
- Multi-floor layouts (floors 0, 1, 2)
- Grouped outputs (same shape per 4 outputs on a floor)
- Various machine types (cutters, stackers, painters, rotators)
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from .foundation_config import FOUNDATION_SPECS, FoundationSpec, Side


class ProblemType(Enum):
    """Types of problems to generate."""
    CORNER_SPLITTER = auto()      # 1 input -> 4 corner outputs (uses cutter)
    HALF_SPLITTER = auto()        # 1 input -> 2 half outputs (uses cutter)
    COLOR_PAINTER = auto()        # shape + color -> painted shape
    SHAPE_ROTATOR = auto()        # rotate input shape
    SHAPE_STACKER = auto()        # stack two shapes
    MULTI_MACHINE = auto()        # combination of operations
    PASSTHROUGH = auto()          # simple routing (no machines)


class ShapeColor(Enum):
    """Shape colors."""
    UNCOLORED = "u"
    RED = "r"
    GREEN = "g"
    BLUE = "b"
    CYAN = "c"
    MAGENTA = "m"
    YELLOW = "y"
    WHITE = "w"


class ShapeType(Enum):
    """Shape types."""
    CIRCLE = "C"
    SQUARE = "R"
    STAR = "S"
    DIAMOND = "W"


@dataclass
class IOPort:
    """Input or output port specification."""
    x: int
    y: int
    floor: int
    shape: Optional[str] = None  # For inputs
    expected_shape: Optional[str] = None  # For outputs
    throughput: float = 45.0  # For inputs


@dataclass
class ProblemSpec:
    """Specification for a generated problem."""
    name: str
    foundation: str
    inputs: List[IOPort]
    outputs: List[IOPort]
    problem_type: ProblemType
    num_floors_used: int
    machines_needed: List[str]


class ProblemGenerator:
    """Generates diverse training problems."""

    # Foundation weights for selection (favor variety)
    FOUNDATION_WEIGHTS = {
        # Small foundations
        "1x1": 1.0,
        "2x1": 1.5,
        "1x2": 1.5,
        "3x1": 1.5,
        "1x3": 1.5,
        "4x1": 1.0,
        "1x4": 1.0,
        # Medium foundations
        "2x2": 2.0,
        "3x2": 2.0,
        "2x3": 2.0,
        "4x2": 1.5,
        "2x4": 1.5,
        # Large foundation
        "3x3": 1.5,
        # Irregular foundations (important for variety)
        "T": 2.5,
        "L": 2.5,
        "L4": 2.0,
        "S4": 2.0,
        "Cross": 2.0,
    }

    # Problem type weights
    PROBLEM_WEIGHTS = {
        ProblemType.CORNER_SPLITTER: 2.5,
        ProblemType.HALF_SPLITTER: 2.0,
        ProblemType.COLOR_PAINTER: 2.0,
        ProblemType.SHAPE_ROTATOR: 1.5,
        ProblemType.SHAPE_STACKER: 2.0,
        ProblemType.MULTI_MACHINE: 3.0,
        ProblemType.PASSTHROUGH: 1.0,
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
        self.problem_counter = 0

    def _weighted_choice(self, weights: Dict) -> str:
        """Make a weighted random choice."""
        items = list(weights.keys())
        probs = [weights[k] for k in items]
        total = sum(probs)
        probs = [p / total for p in probs]
        return random.choices(items, weights=probs, k=1)[0]

    def _random_shape(self) -> str:
        """Generate a random full shape code (4 quadrants)."""
        shape = random.choice(list(ShapeType))
        color = random.choice(list(ShapeColor))
        quadrant = f"{shape.value}{color.value}"
        return quadrant * 4  # e.g., "CuCuCuCu"

    def _random_partial_shape(self, num_quadrants: int = 2) -> str:
        """Generate a shape with only some quadrants filled."""
        shape = random.choice(list(ShapeType))
        color = random.choice(list(ShapeColor))
        quadrant = f"{shape.value}{color.value}"
        empty = "--"

        parts = [empty] * 4
        positions = random.sample(range(4), num_quadrants)
        for pos in positions:
            parts[pos] = quadrant
        return "".join(parts)

    def _get_corner_shapes(self, base_shape: str) -> List[str]:
        """Get 4 corner shapes from a full shape."""
        # Parse base shape
        quadrant = base_shape[:2]  # e.g., "Cu" from "CuCuCuCu"
        empty = "--"

        return [
            f"{quadrant}{empty}{empty}{empty}",  # NE corner (quadrant 0)
            f"{empty}{quadrant}{empty}{empty}",  # NW corner (quadrant 1)
            f"{empty}{empty}{quadrant}{empty}",  # SW corner (quadrant 2)
            f"{empty}{empty}{empty}{quadrant}",  # SE corner (quadrant 3)
        ]

    def _get_half_shapes(self, base_shape: str) -> Tuple[str, str]:
        """Get left and right halves from a full shape."""
        quadrant = base_shape[:2]
        empty = "--"

        left_half = f"{quadrant}{empty}{quadrant}{empty}"   # NE and SW (left in game)
        right_half = f"{empty}{quadrant}{empty}{quadrant}"  # NW and SE (right in game)
        return left_half, right_half

    def _get_foundation_io_positions(
        self,
        spec: FoundationSpec,
        side: Side,
        floor: int,
        num_ports: int = 4
    ) -> List[Tuple[int, int]]:
        """Get valid I/O positions for a foundation side."""
        positions = []
        ports_on_side = spec.ports_per_side[side]

        # Select ports spread across the side
        if num_ports <= ports_on_side:
            step = max(1, ports_on_side // num_ports)
            port_indices = list(range(0, ports_on_side, step))[:num_ports]
        else:
            port_indices = list(range(ports_on_side))

        for port_idx in port_indices:
            grid_x, grid_y = spec.get_port_grid_position(side, port_idx)

            # Convert to external position (outside foundation)
            if side == Side.NORTH:
                positions.append((grid_x, -1))
            elif side == Side.SOUTH:
                positions.append((grid_x, spec.grid_height))
            elif side == Side.WEST:
                positions.append((-1, grid_y))
            elif side == Side.EAST:
                positions.append((spec.grid_width, grid_y))

        return positions

    def _generate_corner_splitter(
        self,
        foundation_name: str,
        num_floors: int
    ) -> ProblemSpec:
        """Generate a corner splitter problem."""
        spec = FOUNDATION_SPECS[foundation_name]

        # Random input shape
        input_shape = self._random_shape()
        corner_shapes = self._get_corner_shapes(input_shape)

        inputs = []
        outputs = []

        # Place inputs on west side, outputs on other sides
        input_side = random.choice([Side.WEST, Side.NORTH])
        output_sides = [s for s in Side if s != input_side]

        for floor in range(num_floors):
            # One input per floor
            input_positions = self._get_foundation_io_positions(spec, input_side, floor, 1)
            if input_positions:
                x, y = input_positions[0]
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=input_shape))

            # 4 outputs per floor (grouped - all same corner shape)
            corner_idx = floor % 4
            output_shape = corner_shapes[corner_idx]

            # Spread outputs across available sides
            outputs_placed = 0
            for out_side in output_sides:
                if outputs_placed >= 4:
                    break
                out_positions = self._get_foundation_io_positions(
                    spec, out_side, floor,
                    min(2, 4 - outputs_placed)
                )
                for x, y in out_positions:
                    if outputs_placed < 4:
                        outputs.append(IOPort(x=x, y=y, floor=floor, expected_shape=output_shape))
                        outputs_placed += 1

        return ProblemSpec(
            name=f"Corner Splitter {foundation_name} {num_floors}F",
            foundation=foundation_name,
            inputs=inputs,
            outputs=outputs,
            problem_type=ProblemType.CORNER_SPLITTER,
            num_floors_used=num_floors,
            machines_needed=["CUTTER", "HALF_CUTTER"]
        )

    def _generate_half_splitter(
        self,
        foundation_name: str,
        num_floors: int
    ) -> ProblemSpec:
        """Generate a half splitter problem."""
        spec = FOUNDATION_SPECS[foundation_name]

        input_shape = self._random_shape()
        left_half, right_half = self._get_half_shapes(input_shape)

        inputs = []
        outputs = []

        input_side = random.choice([Side.WEST, Side.NORTH])

        for floor in range(num_floors):
            # Input on one side
            input_positions = self._get_foundation_io_positions(spec, input_side, floor, 1)
            if input_positions:
                x, y = input_positions[0]
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=input_shape))

            # 2 outputs per floor - grouped by half type
            # Even floors: left half, Odd floors: right half
            output_shape = left_half if floor % 2 == 0 else right_half

            output_side = input_side.opposite
            out_positions = self._get_foundation_io_positions(spec, output_side, floor, 2)
            for x, y in out_positions[:2]:
                outputs.append(IOPort(x=x, y=y, floor=floor, expected_shape=output_shape))

        return ProblemSpec(
            name=f"Half Splitter {foundation_name} {num_floors}F",
            foundation=foundation_name,
            inputs=inputs,
            outputs=outputs,
            problem_type=ProblemType.HALF_SPLITTER,
            num_floors_used=num_floors,
            machines_needed=["CUTTER"]
        )

    def _generate_painter(
        self,
        foundation_name: str,
        num_floors: int
    ) -> ProblemSpec:
        """Generate a painting problem."""
        spec = FOUNDATION_SPECS[foundation_name]

        # Uncolored input shape
        shape_type = random.choice(list(ShapeType))
        input_shape = f"{shape_type.value}u" * 4

        # Target color
        target_color = random.choice([c for c in ShapeColor if c != ShapeColor.UNCOLORED])
        output_shape = f"{shape_type.value}{target_color.value}" * 4
        color_shape = f"--{target_color.value.upper()}{target_color.value}----"  # Color crystal

        inputs = []
        outputs = []

        for floor in range(num_floors):
            # Shape input on west
            shape_positions = self._get_foundation_io_positions(spec, Side.WEST, floor, 1)
            if shape_positions:
                x, y = shape_positions[0]
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=input_shape))

            # Color input on north or south
            color_side = random.choice([Side.NORTH, Side.SOUTH])
            color_positions = self._get_foundation_io_positions(spec, color_side, floor, 1)
            if color_positions:
                x, y = color_positions[0]
                # Use a full color shape for painting
                full_color = f"C{target_color.value}" * 4
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=full_color))

            # Output on east - all same painted shape
            out_positions = self._get_foundation_io_positions(spec, Side.EAST, floor, 4)
            for x, y in out_positions[:4]:
                outputs.append(IOPort(x=x, y=y, floor=floor, expected_shape=output_shape))

        return ProblemSpec(
            name=f"Painter {foundation_name} {num_floors}F",
            foundation=foundation_name,
            inputs=inputs,
            outputs=outputs,
            problem_type=ProblemType.COLOR_PAINTER,
            num_floors_used=num_floors,
            machines_needed=["PAINTER", "PAINTER_MIRRORED"]
        )

    def _generate_rotator(
        self,
        foundation_name: str,
        num_floors: int
    ) -> ProblemSpec:
        """Generate a rotation problem."""
        spec = FOUNDATION_SPECS[foundation_name]

        # Asymmetric input shape
        input_shape = self._random_partial_shape(2)

        inputs = []
        outputs = []

        for floor in range(num_floors):
            # Input on west
            input_positions = self._get_foundation_io_positions(spec, Side.WEST, floor, 1)
            if input_positions:
                x, y = input_positions[0]
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=input_shape))

            # Rotated output on east
            # Rotation depends on floor
            rotation_steps = (floor + 1) % 4
            output_shape = self._rotate_shape(input_shape, rotation_steps)

            out_positions = self._get_foundation_io_positions(spec, Side.EAST, floor, 4)
            for x, y in out_positions[:4]:
                outputs.append(IOPort(x=x, y=y, floor=floor, expected_shape=output_shape))

        return ProblemSpec(
            name=f"Rotator {foundation_name} {num_floors}F",
            foundation=foundation_name,
            inputs=inputs,
            outputs=outputs,
            problem_type=ProblemType.SHAPE_ROTATOR,
            num_floors_used=num_floors,
            machines_needed=["ROTATOR_CW", "ROTATOR_CCW", "ROTATOR_180"]
        )

    def _rotate_shape(self, shape: str, steps: int) -> str:
        """Rotate a shape clockwise by steps."""
        if len(shape) != 8:
            return shape

        # Parse into quadrants
        quadrants = [shape[i:i+2] for i in range(0, 8, 2)]

        # Rotate
        steps = steps % 4
        rotated = quadrants[-steps:] + quadrants[:-steps]

        return "".join(rotated)

    def _generate_stacker(
        self,
        foundation_name: str,
        num_floors: int
    ) -> ProblemSpec:
        """Generate a stacking problem."""
        spec = FOUNDATION_SPECS[foundation_name]

        # Two input shapes to stack
        bottom_shape = self._random_shape()
        top_shape = self._random_shape()

        # Stacked shape (simplified - just combines)
        output_shape = f"{bottom_shape}:{top_shape}"

        inputs = []
        outputs = []

        for floor in range(min(num_floors, 2)):  # Stackers use 2 floors
            # Bottom shape on floor 0
            if floor == 0:
                input_positions = self._get_foundation_io_positions(spec, Side.WEST, 0, 1)
                if input_positions:
                    x, y = input_positions[0]
                    inputs.append(IOPort(x=x, y=y, floor=0, shape=bottom_shape))

            # Top shape on floor 1
            if floor == 1 or num_floors == 1:
                input_positions = self._get_foundation_io_positions(spec, Side.WEST, min(1, spec.num_floors - 1), 1)
                if input_positions:
                    x, y = input_positions[0]
                    inputs.append(IOPort(x=x, y=y, floor=min(1, spec.num_floors - 1), shape=top_shape))

        # Outputs on floor 0
        out_positions = self._get_foundation_io_positions(spec, Side.EAST, 0, 4)
        for x, y in out_positions[:4]:
            outputs.append(IOPort(x=x, y=y, floor=0, expected_shape=output_shape))

        return ProblemSpec(
            name=f"Stacker {foundation_name} {num_floors}F",
            foundation=foundation_name,
            inputs=inputs,
            outputs=outputs,
            problem_type=ProblemType.SHAPE_STACKER,
            num_floors_used=min(num_floors, 2),
            machines_needed=["STACKER", "STACKER_BENT"]
        )

    def _generate_multi_machine(
        self,
        foundation_name: str,
        num_floors: int
    ) -> ProblemSpec:
        """Generate a problem requiring multiple machine types."""
        spec = FOUNDATION_SPECS[foundation_name]

        # Cut then paint
        shape_type = random.choice(list(ShapeType))
        input_shape = f"{shape_type.value}u" * 4

        target_color = random.choice([c for c in ShapeColor if c != ShapeColor.UNCOLORED])

        # Cut into halves, then paint
        left_painted = f"{shape_type.value}{target_color.value}--{shape_type.value}{target_color.value}--"
        right_painted = f"--{shape_type.value}{target_color.value}--{shape_type.value}{target_color.value}"

        inputs = []
        outputs = []

        for floor in range(num_floors):
            # Shape input
            input_positions = self._get_foundation_io_positions(spec, Side.WEST, floor, 1)
            if input_positions:
                x, y = input_positions[0]
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=input_shape))

            # Color input
            color_positions = self._get_foundation_io_positions(spec, Side.NORTH, floor, 1)
            if color_positions:
                x, y = color_positions[0]
                full_color = f"C{target_color.value}" * 4
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=full_color))

            # Outputs - even floors get left half painted, odd get right
            output_shape = left_painted if floor % 2 == 0 else right_painted

            out_positions = self._get_foundation_io_positions(spec, Side.EAST, floor, 4)
            for x, y in out_positions[:4]:
                outputs.append(IOPort(x=x, y=y, floor=floor, expected_shape=output_shape))

        return ProblemSpec(
            name=f"Multi-Machine {foundation_name} {num_floors}F",
            foundation=foundation_name,
            inputs=inputs,
            outputs=outputs,
            problem_type=ProblemType.MULTI_MACHINE,
            num_floors_used=num_floors,
            machines_needed=["CUTTER", "PAINTER", "PAINTER_MIRRORED"]
        )

    def _generate_passthrough(
        self,
        foundation_name: str,
        num_floors: int
    ) -> ProblemSpec:
        """Generate a simple passthrough routing problem."""
        spec = FOUNDATION_SPECS[foundation_name]

        input_shape = self._random_shape()

        inputs = []
        outputs = []

        # Choose opposing sides
        side_pairs = [(Side.WEST, Side.EAST), (Side.NORTH, Side.SOUTH)]
        input_side, output_side = random.choice(side_pairs)

        for floor in range(num_floors):
            # Input
            input_positions = self._get_foundation_io_positions(spec, input_side, floor, 1)
            if input_positions:
                x, y = input_positions[0]
                inputs.append(IOPort(x=x, y=y, floor=floor, shape=input_shape))

            # Output - same shape (passthrough)
            out_positions = self._get_foundation_io_positions(spec, output_side, floor, 4)
            for x, y in out_positions[:4]:
                outputs.append(IOPort(x=x, y=y, floor=floor, expected_shape=input_shape))

        return ProblemSpec(
            name=f"Passthrough {foundation_name} {num_floors}F",
            foundation=foundation_name,
            inputs=inputs,
            outputs=outputs,
            problem_type=ProblemType.PASSTHROUGH,
            num_floors_used=num_floors,
            machines_needed=[]
        )

    def generate_problem(
        self,
        foundation_name: Optional[str] = None,
        problem_type: Optional[ProblemType] = None,
        num_floors: Optional[int] = None
    ) -> ProblemSpec:
        """Generate a single problem with optional constraints."""
        # Select foundation
        if foundation_name is None:
            foundation_name = self._weighted_choice(self.FOUNDATION_WEIGHTS)

        # Select problem type
        if problem_type is None:
            problem_type = self._weighted_choice(self.PROBLEM_WEIGHTS)

        # Select number of floors (bias toward multi-floor)
        if num_floors is None:
            # Weighted toward using all 3 floors
            num_floors = random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5], k=1)[0]

        # Generate based on problem type
        generators = {
            ProblemType.CORNER_SPLITTER: self._generate_corner_splitter,
            ProblemType.HALF_SPLITTER: self._generate_half_splitter,
            ProblemType.COLOR_PAINTER: self._generate_painter,
            ProblemType.SHAPE_ROTATOR: self._generate_rotator,
            ProblemType.SHAPE_STACKER: self._generate_stacker,
            ProblemType.MULTI_MACHINE: self._generate_multi_machine,
            ProblemType.PASSTHROUGH: self._generate_passthrough,
        }

        return generators[problem_type](foundation_name, num_floors)

    def problem_to_json(self, problem: ProblemSpec) -> dict:
        """Convert problem spec to JSON format compatible with solver."""
        self.problem_counter += 1

        return {
            "name": f"{problem.name} #{self.problem_counter}",
            "created": datetime.now().isoformat(),
            "foundation": problem.foundation,
            "buildings": [],  # Empty - solver will place buildings
            "inputs": [
                {
                    "x": inp.x,
                    "y": inp.y,
                    "floor": inp.floor,
                    "shape": inp.shape,
                    "throughput": inp.throughput
                }
                for inp in problem.inputs
            ],
            "outputs": [
                {
                    "x": out.x,
                    "y": out.y,
                    "floor": out.floor,
                    "expected_shape": out.expected_shape
                }
                for out in problem.outputs
            ],
            "_metadata": {
                "problem_type": problem.problem_type.name,
                "num_floors_used": problem.num_floors_used,
                "machines_needed": problem.machines_needed
            }
        }

    def generate_training_set(
        self,
        num_problems: int,
        output_dir: Optional[Path] = None,
        ensure_coverage: bool = True
    ) -> List[dict]:
        """Generate a diverse training set.

        Args:
            num_problems: Number of problems to generate
            output_dir: Optional directory to save JSON files
            ensure_coverage: If True, ensure all foundations and problem types are covered

        Returns:
            List of problem JSON dicts
        """
        problems = []

        if ensure_coverage:
            # First pass: ensure at least one of each foundation and problem type
            all_foundations = list(FOUNDATION_SPECS.keys())
            all_problem_types = list(ProblemType)

            # Cover all foundations
            for foundation in all_foundations:
                problem_type = random.choice(all_problem_types)
                num_floors = random.choice([2, 3])  # Prefer multi-floor
                spec = self.generate_problem(foundation, problem_type, num_floors)
                problems.append(self.problem_to_json(spec))

            # Cover all problem types
            for ptype in all_problem_types:
                foundation = random.choice(all_foundations)
                num_floors = random.choice([2, 3])
                spec = self.generate_problem(foundation, ptype, num_floors)
                problems.append(self.problem_to_json(spec))

        # Fill remaining with random problems
        while len(problems) < num_problems:
            spec = self.generate_problem()
            problems.append(self.problem_to_json(spec))

        # Shuffle to mix coverage problems with random ones
        random.shuffle(problems)

        # Save to files if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, problem in enumerate(problems):
                filepath = output_dir / f"training_{i:04d}.json"
                with open(filepath, 'w') as f:
                    json.dump(problem, f, indent=2)

        return problems

    def generate_difficulty_progression(
        self,
        num_per_level: int = 10
    ) -> Dict[str, List[dict]]:
        """Generate problems organized by difficulty.

        Returns:
            Dict with keys 'easy', 'medium', 'hard' containing problem lists
        """
        # Easy: small foundations, single floor, passthrough or half splitter
        easy_foundations = ["1x1", "2x1", "1x2", "2x2"]
        easy_types = [ProblemType.PASSTHROUGH, ProblemType.HALF_SPLITTER]

        # Medium: medium foundations, 2 floors, various types
        medium_foundations = ["3x2", "2x3", "3x3", "L", "T"]
        medium_types = [
            ProblemType.CORNER_SPLITTER,
            ProblemType.COLOR_PAINTER,
            ProblemType.SHAPE_ROTATOR
        ]

        # Hard: large/irregular foundations, 3 floors, multi-machine
        hard_foundations = ["4x2", "2x4", "Cross", "L4", "S4"]
        hard_types = [
            ProblemType.MULTI_MACHINE,
            ProblemType.SHAPE_STACKER,
            ProblemType.CORNER_SPLITTER
        ]

        result = {"easy": [], "medium": [], "hard": []}

        for _ in range(num_per_level):
            # Easy
            spec = self.generate_problem(
                random.choice(easy_foundations),
                random.choice(easy_types),
                1
            )
            result["easy"].append(self.problem_to_json(spec))

            # Medium
            spec = self.generate_problem(
                random.choice(medium_foundations),
                random.choice(medium_types),
                2
            )
            result["medium"].append(self.problem_to_json(spec))

            # Hard
            spec = self.generate_problem(
                random.choice(hard_foundations),
                random.choice(hard_types),
                3
            )
            result["hard"].append(self.problem_to_json(spec))

        return result


def main():
    """Generate sample training problems."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training problems")
    parser.add_argument("--num", type=int, default=100, help="Number of problems")
    parser.add_argument("--output", type=str, default="training_problems", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    generator = ProblemGenerator(seed=args.seed)

    print(f"Generating {args.num} training problems...")
    problems = generator.generate_training_set(
        num_problems=args.num,
        output_dir=Path(args.output),
        ensure_coverage=True
    )

    # Print statistics
    foundations = {}
    problem_types = {}
    floors = {}

    for p in problems:
        f = p["foundation"]
        foundations[f] = foundations.get(f, 0) + 1

        pt = p["_metadata"]["problem_type"]
        problem_types[pt] = problem_types.get(pt, 0) + 1

        nf = p["_metadata"]["num_floors_used"]
        floors[nf] = floors.get(nf, 0) + 1

    print(f"\nGenerated {len(problems)} problems")
    print(f"\nFoundation distribution:")
    for f, count in sorted(foundations.items()):
        print(f"  {f}: {count}")

    print(f"\nProblem type distribution:")
    for pt, count in sorted(problem_types.items()):
        print(f"  {pt}: {count}")

    print(f"\nFloor distribution:")
    for f, count in sorted(floors.items()):
        print(f"  {f} floors: {count}")


if __name__ == "__main__":
    main()
