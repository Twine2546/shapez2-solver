"""Solver module - CP-SAT solver and ML-enhanced routing.

This module provides:
- CP-SAT constraint programming solver for optimal machine placement
- A* router with ML enhancements for belt connections
- ML models for placement and routing optimization
- Training infrastructure for ML models
"""

# Core types
from .core_types import (
    PlacedBuilding, Candidate, CellType, GridCell, LogicalConnection,
    OPERATION_BUILDINGS, BELT_TYPES, ROUTING_BUILDINGS,
    EVOLVABLE_BUILDINGS, CONVEYOR_TYPES, DUAL_INPUT_BUILDINGS,
)

# Foundation configuration
from .foundation_config import (
    FoundationConfig, FoundationSpec, Side, PortType, PortConfig,
    FOUNDATION_SPECS,
)

# CP-SAT solver
from .cpsat_solver import (
    CPSATFullSolver, CPSATLayoutSolver, CPSATSystemSolver,
    CPSATSolution, solve_with_cpsat,
)

# A* Router
from .router import BeltRouter, Connection, RouteResult

# Database utilities
from .databases import TrainingSampleDB

# Evaluation
from .evaluation import PlacementInfo, DefaultSolutionEvaluator

__all__ = [
    # Core types
    "PlacedBuilding",
    "Candidate",
    "CellType",
    "GridCell",
    "LogicalConnection",
    "OPERATION_BUILDINGS",
    "BELT_TYPES",
    "ROUTING_BUILDINGS",
    "EVOLVABLE_BUILDINGS",
    "CONVEYOR_TYPES",
    "DUAL_INPUT_BUILDINGS",
    # Foundation
    "FoundationConfig",
    "FoundationSpec",
    "Side",
    "PortType",
    "PortConfig",
    "FOUNDATION_SPECS",
    # CP-SAT solver
    "CPSATFullSolver",
    "CPSATLayoutSolver",
    "CPSATSystemSolver",
    "CPSATSolution",
    "solve_with_cpsat",
    # Router
    "BeltRouter",
    "Connection",
    "RouteResult",
    # Database
    "TrainingSampleDB",
    # Evaluation
    "PlacementInfo",
    "DefaultSolutionEvaluator",
]
