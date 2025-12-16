"""Solver module for Shapez 2.

This module provides constraint-programming and pathfinding based solvers for
finding optimal machine layouts on foundations.

Main components:
- cpsat_solver: CP-SAT constraint programming solver (Google OR-Tools)
- router: A* pathfinding for belt routing
- core: Core data structures (Candidate, PlacedBuilding, etc.)
- foundation_config: Foundation specifications and configurations
"""

from .core import (
    Candidate, PlacedBuilding, LogicalConnection,
    CellType, GridCell, SolverResult, FoundationEvolution,
    OPERATION_BUILDINGS, BELT_TYPES, ROUTING_BUILDINGS,
    EVOLVABLE_BUILDINGS, CONVEYOR_TYPES, DUAL_INPUT_BUILDINGS,
)

from .foundation_config import (
    FoundationConfig, FoundationSpec, Side, PortType, PortConfig,
    FOUNDATION_SPECS,
)

from .cpsat_solver import (
    CPSATFullSolver, CPSATLayoutSolver, CPSATSystemSolver,
    CPSATSolution, solve_with_cpsat,
)

from .router import (
    BeltRouter, Connection, RouteResult, GridNode,
    route_candidate_connections,
)

__all__ = [
    # Core data structures
    "Candidate",
    "PlacedBuilding",
    "LogicalConnection",
    "CellType",
    "GridCell",
    "SolverResult",
    "FoundationEvolution",
    # Building lists
    "OPERATION_BUILDINGS",
    "BELT_TYPES",
    "ROUTING_BUILDINGS",
    "EVOLVABLE_BUILDINGS",
    "CONVEYOR_TYPES",
    "DUAL_INPUT_BUILDINGS",
    # Foundation config
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
    "GridNode",
    "route_candidate_connections",
]
