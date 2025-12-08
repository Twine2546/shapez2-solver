"""Evolution module for genetic algorithm-based solution search."""

from .algorithm import EvolutionaryAlgorithm, EvolutionConfig
from .fitness import FitnessFunction, ShapeMatchFitness
from .operators import MutationOperator, CrossoverOperator
from .candidate import Candidate

# Two-phase evolution imports
from .system_search import (
    SystemSearch, SystemDesign, MachineNode,
    find_system_for_transformation, evaluate_system
)
from .layout_search import (
    LayoutSearch, LayoutCandidate, PlacedMachine,
    layout_system_design
)
from .two_phase_evolution import (
    TwoPhaseEvolution, TwoPhaseResult,
    create_two_phase_evolution, solve_transformation
)

# CP-SAT solver imports
from .cpsat_solver import (
    CPSATFullSolver, CPSATLayoutSolver, CPSATSystemSolver,
    CPSATSolution, solve_with_cpsat
)

__all__ = [
    "EvolutionaryAlgorithm",
    "EvolutionConfig",
    "FitnessFunction",
    "ShapeMatchFitness",
    "MutationOperator",
    "CrossoverOperator",
    "Candidate",
    # Two-phase evolution
    "SystemSearch",
    "SystemDesign",
    "MachineNode",
    "find_system_for_transformation",
    "evaluate_system",
    "LayoutSearch",
    "LayoutCandidate",
    "PlacedMachine",
    "layout_system_design",
    "TwoPhaseEvolution",
    "TwoPhaseResult",
    "create_two_phase_evolution",
    "solve_transformation",
    # CP-SAT solver
    "CPSATFullSolver",
    "CPSATLayoutSolver",
    "CPSATSystemSolver",
    "CPSATSolution",
    "solve_with_cpsat",
]
