"""
Two-Phase Evolution - Unified interface for system + layout search.

Phase 1: System Search - Find which machines and connections solve the problem
Phase 2: Layout Search - Find optimal physical placement for that system

This approach separates the "what to build" from "where to build it",
making the search much more effective.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..shapes.shape import Shape
from ..blueprint.building_types import BuildingType, Rotation
from .system_search import (
    SystemSearch, SystemSearchSA, SystemSearchHybrid,
    SystemDesign, SystemSimulator,
    evaluate_system, find_system_for_transformation,
    MachineNode, InputPort, OutputPort
)
from .layout_search import (
    LayoutSearch, LayoutSearchSA, LayoutSearchHybrid,
    LayoutCandidate, PlacedMachine,
    layout_system_design
)
from .foundation_evolution import PlacedBuilding, Candidate


@dataclass
class TwoPhaseResult:
    """Result of two-phase evolution."""
    # Phase 1 results
    system_design: Optional[SystemDesign]
    system_fitness: float

    # Phase 2 results
    layout: Optional[LayoutCandidate]
    layout_fitness: float

    # Combined results
    success: bool
    total_fitness: float

    # For compatibility with existing viewer
    buildings: List[PlacedBuilding] = field(default_factory=list)

    def to_candidate(self) -> Optional[Candidate]:
        """Convert to Candidate format for compatibility."""
        if not self.buildings:
            return None
        return Candidate(
            buildings=self.buildings,
            fitness=self.total_fitness
        )


class TwoPhaseEvolution:
    """
    Two-phase evolution solver.

    Phase 1: Search for a system design (machines + logical connections)
    Phase 2: Search for optimal layout (positions + routing)
    """

    def __init__(
        self,
        input_specs: List[Tuple[str, int, int, str]],
        output_specs: List[Tuple[str, int, int, str]],
        grid_width: int,
        grid_height: int,
        num_floors: int = 4,
        system_population: int = 30,
        layout_population: int = 50,
        max_machines: int = 10,
    ):
        """
        Initialize two-phase evolution.

        Args:
            input_specs: List of (side, position, floor, shape_code) for inputs
            output_specs: List of (side, position, floor, shape_code) for outputs
            grid_width: Foundation grid width
            grid_height: Foundation grid height
            num_floors: Number of floors available
            system_population: Population size for system search
            layout_population: Population size for layout search
            max_machines: Maximum machines allowed
        """
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_floors = num_floors
        self.system_population = system_population
        self.layout_population = layout_population
        self.max_machines = max_machines

        self.system_search: Optional[SystemSearch] = None
        self.layout_search: Optional[LayoutSearch] = None
        self.result: Optional[TwoPhaseResult] = None

        # For viewer compatibility
        self.top_solutions: List[Candidate] = []

    def run(
        self,
        system_generations: int = 100,
        layout_generations: int = 100,
        algorithm: str = 'evolution',
        verbose: bool = False
    ) -> TwoPhaseResult:
        """
        Run the two-phase evolution.

        Args:
            system_generations: Max generations for system search
            layout_generations: Max generations for layout search
            algorithm: 'evolution', 'sa', or 'hybrid'
            verbose: Print progress

        Returns:
            TwoPhaseResult with the best solution found
        """
        if verbose:
            print("=" * 50)
            print(f"PHASE 1: System Search ({algorithm})")
            print("=" * 50)
            print(f"Finding system to transform {len(self.input_specs)} inputs -> {len(self.output_specs)} outputs")

        # Phase 1: Find system design
        if algorithm == 'sa':
            self.system_search = SystemSearchSA(
                input_specs=self.input_specs,
                output_specs=self.output_specs,
                max_machines=self.max_machines,
            )
        elif algorithm == 'hybrid':
            self.system_search = SystemSearchHybrid(
                input_specs=self.input_specs,
                output_specs=self.output_specs,
                population_size=self.system_population,
                max_machines=self.max_machines,
            )
        else:
            self.system_search = SystemSearch(
                input_specs=self.input_specs,
                output_specs=self.output_specs,
                population_size=self.system_population,
                max_machines=self.max_machines,
            )

        system = self.system_search.run(
            max_generations=system_generations,
            target_fitness=1.0,
            verbose=verbose
        )

        system_fitness = self.system_search.best_fitness

        if verbose:
            print(f"\nPhase 1 complete: fitness={system_fitness:.4f}")
            if system:
                print(f"  Machines: {len(system.machines)}")
                for m in system.machines:
                    print(f"    - {m.building_type.name}")

        if system is None or system_fitness < 0.5:
            # System search failed
            self.result = TwoPhaseResult(
                system_design=system,
                system_fitness=system_fitness,
                layout=None,
                layout_fitness=0.0,
                success=False,
                total_fitness=system_fitness * 50  # Scale to 0-100
            )
            return self.result

        if verbose:
            print("\n" + "=" * 50)
            print(f"PHASE 2: Layout Search ({algorithm})")
            print("=" * 50)
            print(f"Placing {len(system.machines)} machines on {self.grid_width}x{self.grid_height} grid")

        # Phase 2: Find optimal layout
        if algorithm == 'sa':
            self.layout_search = LayoutSearchSA(
                system_design=system,
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                num_floors=self.num_floors,
            )
        elif algorithm == 'hybrid':
            self.layout_search = LayoutSearchHybrid(
                system_design=system,
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                num_floors=self.num_floors,
                population_size=self.layout_population,
            )
        else:
            self.layout_search = LayoutSearch(
                system_design=system,
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                num_floors=self.num_floors,
                population_size=self.layout_population,
            )

        layout = self.layout_search.run(
            max_generations=layout_generations,
            verbose=verbose
        )

        layout_fitness = self.layout_search.best_fitness if layout else 0.0

        if verbose:
            print(f"\nPhase 2 complete: fitness={layout_fitness:.2f}")
            if layout:
                print(f"  Machines placed: {len(layout.machines)}")
                print(f"  Belts: {len(layout.belts)}")
                print(f"  Routing success: {layout.routing_success}")

        # Create result
        success = (system_fitness >= 0.99 and layout is not None and layout.routing_success)
        total_fitness = (system_fitness * 50) + (layout_fitness / 2) if layout else system_fitness * 50

        self.result = TwoPhaseResult(
            system_design=system,
            system_fitness=system_fitness,
            layout=layout,
            layout_fitness=layout_fitness,
            success=success,
            total_fitness=total_fitness
        )

        # Convert to buildings for viewer compatibility
        if layout:
            self._convert_to_buildings()

        return self.result

    def _convert_to_buildings(self):
        """Convert layout to PlacedBuilding list for viewer compatibility."""
        if not self.result or not self.result.layout:
            return

        layout = self.result.layout
        buildings = []
        building_id = 0

        # Add machines
        for m in layout.machines:
            buildings.append(PlacedBuilding(
                building_id=building_id,
                building_type=m.building_type,
                x=m.x,
                y=m.y,
                floor=m.floor,
                rotation=m.rotation
            ))
            building_id += 1

        # Add belts
        for x, y, floor, belt_type, rotation in layout.belts:
            buildings.append(PlacedBuilding(
                building_id=building_id,
                building_type=belt_type,
                x=x,
                y=y,
                floor=floor,
                rotation=rotation
            ))
            building_id += 1

        self.result.buildings = buildings

        # Create candidate for top_solutions
        candidate = self.result.to_candidate()
        if candidate:
            self.top_solutions = [candidate]


def create_two_phase_evolution(
    foundation_type: str,
    input_specs: List[Tuple[str, int, int, str]],
    output_specs: List[Tuple[str, int, int, str]],
    **kwargs
) -> TwoPhaseEvolution:
    """
    Create a two-phase evolution instance for a foundation type.

    Args:
        foundation_type: Foundation type string (e.g., '1x1', '2x2', '3x3')
        input_specs: List of (side, position, floor, shape_code)
        output_specs: List of (side, position, floor, shape_code)
        **kwargs: Additional parameters passed to TwoPhaseEvolution

    Returns:
        Configured TwoPhaseEvolution instance
    """
    from .foundation_config import FOUNDATION_SPECS

    spec = FOUNDATION_SPECS.get(foundation_type)
    if spec is None:
        raise ValueError(f"Unknown foundation type: {foundation_type}")

    return TwoPhaseEvolution(
        input_specs=input_specs,
        output_specs=output_specs,
        grid_width=spec.grid_width,
        grid_height=spec.grid_height,
        num_floors=spec.num_floors,
        **kwargs
    )


def solve_transformation(
    foundation_type: str,
    input_specs: List[Tuple[str, int, int, str]],
    output_specs: List[Tuple[str, int, int, str]],
    system_generations: int = 100,
    layout_generations: int = 100,
    algorithm: str = 'evolution',
    verbose: bool = False
) -> TwoPhaseResult:
    """
    Solve a shape transformation problem using two-phase evolution.

    Args:
        foundation_type: Foundation type string
        input_specs: Input port specifications
        output_specs: Output port specifications
        system_generations: Generations for system search
        layout_generations: Generations for layout search
        algorithm: 'evolution', 'sa', or 'hybrid'
        verbose: Print progress

    Returns:
        TwoPhaseResult with the solution
    """
    evo = create_two_phase_evolution(
        foundation_type=foundation_type,
        input_specs=input_specs,
        output_specs=output_specs,
    )
    return evo.run(
        system_generations=system_generations,
        layout_generations=layout_generations,
        algorithm=algorithm,
        verbose=verbose
    )
