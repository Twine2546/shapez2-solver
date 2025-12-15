"""
Evaluation functions for CP-SAT solver and A* routing.

This module provides abstract base classes that allow custom evaluation
functions to be plugged in, including ML-based evaluations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
from ..blueprint.building_types import BuildingType, Rotation


@dataclass
class PlacementInfo:
    """Information about a machine placement for evaluation."""
    building_type: BuildingType
    x: int
    y: int
    floor: int
    rotation: Rotation


@dataclass
class RoutingInfo:
    """Information about routing for evaluation."""
    belts: List[Tuple[int, int, int, BuildingType, Rotation]]
    success: bool
    total_length: int


@dataclass
class SolutionInfo:
    """Complete solution information for evaluation."""
    machines: List[PlacementInfo]
    routing: RoutingInfo
    grid_width: int
    grid_height: int
    num_floors: int
    num_inputs: int
    num_outputs: int
    throughput_per_output: float


class SolutionEvaluator(ABC):
    """
    Abstract base class for evaluating CP-SAT solutions.

    Implement this to create custom evaluation functions, including
    ML-based evaluators that can learn from successful solutions.
    """

    @abstractmethod
    def evaluate(self, solution: SolutionInfo) -> float:
        """
        Evaluate a solution and return a fitness score.

        Args:
            solution: Complete solution information

        Returns:
            Fitness score (higher is better)
        """
        pass

    def on_solution_found(self, solution: SolutionInfo, fitness: float) -> None:
        """
        Called when a successful solution is found.

        Override to collect training data for ML models.

        Args:
            solution: The successful solution
            fitness: The fitness score
        """
        pass


class PlacementEvaluator(ABC):
    """
    Abstract base class for evaluating machine placements before routing.

    This allows early rejection of placements that are unlikely to route
    successfully, saving computation time.
    """

    @abstractmethod
    def evaluate(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
    ) -> Tuple[float, bool]:
        """
        Evaluate a placement and predict routing success.

        Args:
            machines: List of machine placements
            grid_width: Width of the grid
            grid_height: Height of the grid
            num_floors: Number of floors
            input_positions: Input port positions
            output_positions: Output port positions

        Returns:
            Tuple of (score 0-1, should_reject)
        """
        pass


class RoutingHeuristic(ABC):
    """
    Abstract base class for A* routing heuristics.

    Implement this to create custom heuristics, including ML-based
    ones that can learn optimal routing patterns.
    """

    @abstractmethod
    def __call__(
        self,
        current: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate heuristic value for A* search.

        Args:
            current: Current position (x, y, floor)
            goal: Goal position (x, y, floor)
            context: Optional context (occupied cells, grid bounds, etc.)

        Returns:
            Estimated cost to reach goal (must be admissible - never overestimate)
        """
        pass

    def on_path_found(
        self,
        path: List[Tuple[int, int, int, str]],
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
    ) -> None:
        """
        Called when a path is successfully found.

        Override to collect training data for ML models.

        Args:
            path: The found path
            start: Start position
            goal: Goal position
        """
        pass


class MoveCostFunction(ABC):
    """
    Abstract base class for calculating move costs in A* routing.

    This allows custom cost functions that can penalize certain
    moves or encourage specific routing patterns.
    """

    @abstractmethod
    def __call__(
        self,
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
        move_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the cost of moving from current to neighbor.

        Args:
            current: Current position
            neighbor: Neighbor position
            move_type: Type of move ('horizontal', 'lift_up', 'lift_down', 'belt_port')
            context: Optional context

        Returns:
            Cost of the move (must be positive)
        """
        pass


# =============================================================================
# Default Implementations
# =============================================================================

class DefaultSolutionEvaluator(SolutionEvaluator):
    """
    Default solution evaluator using hardcoded scoring.

    Scoring:
    - Throughput: 0-50 points based on throughput ratio
    - Routing success: 50 points if successful
    - Compactness: Up to 20 bonus points for fewer belts
    """

    def __init__(
        self,
        throughput_weight: float = 50.0,
        routing_weight: float = 50.0,
        compactness_weight: float = 20.0,
        belt_penalty: float = 0.5,
    ):
        self.throughput_weight = throughput_weight
        self.routing_weight = routing_weight
        self.compactness_weight = compactness_weight
        self.belt_penalty = belt_penalty

    def evaluate(self, solution: SolutionInfo) -> float:
        fitness = 0.0

        # Throughput score
        if solution.machines:
            num_outputs = solution.num_outputs
            theoretical_max = 180.0 / max(1, num_outputs)
            throughput_ratio = solution.throughput_per_output / theoretical_max if theoretical_max > 0 else 0
            fitness += min(self.throughput_weight, throughput_ratio * self.throughput_weight)

        # Routing success
        if solution.routing.success:
            fitness += self.routing_weight

            # Compactness bonus
            belt_count = len(solution.routing.belts)
            fitness += max(0, self.compactness_weight - belt_count * self.belt_penalty)

        return fitness


class DefaultPlacementEvaluator(PlacementEvaluator):
    """Default placement evaluator that always accepts placements."""

    def __init__(self, reject_threshold: float = 0.0):
        self.reject_threshold = reject_threshold

    def evaluate(
        self,
        machines: List[PlacementInfo],
        grid_width: int,
        grid_height: int,
        num_floors: int,
        input_positions: List[Tuple[int, int, int]],
        output_positions: List[Tuple[int, int, int]],
    ) -> Tuple[float, bool]:
        # Default: accept all placements with neutral score
        return 0.5, False


class DefaultRoutingHeuristic(RoutingHeuristic):
    """
    Default A* heuristic using Manhattan distance.

    Floor changes are weighted 2x to account for lift costs.
    """

    def __init__(self, floor_weight: float = 2.0):
        self.floor_weight = floor_weight

    def __call__(
        self,
        current: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        return (
            abs(current[0] - goal[0]) +
            abs(current[1] - goal[1]) +
            abs(current[2] - goal[2]) * self.floor_weight
        )


class DefaultMoveCostFunction(MoveCostFunction):
    """
    Default move cost function.

    Costs:
    - Horizontal move: 1.0
    - Lift up/down: 1.5
    - Belt port teleport: Variable based on distance
    """

    def __init__(
        self,
        horizontal_cost: float = 1.0,
        lift_cost: float = 1.5,
        belt_port_base_cost: float = 2.0,
        belt_port_distance_factor: float = 0.1,
    ):
        self.horizontal_cost = horizontal_cost
        self.lift_cost = lift_cost
        self.belt_port_base_cost = belt_port_base_cost
        self.belt_port_distance_factor = belt_port_distance_factor

    def __call__(
        self,
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
        move_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if move_type in ('lift_up', 'lift_down'):
            return self.lift_cost
        elif move_type == 'belt_port':
            distance = (
                abs(neighbor[0] - current[0]) +
                abs(neighbor[1] - current[1])
            )
            return self.belt_port_base_cost + distance * self.belt_port_distance_factor
        else:
            return self.horizontal_cost


# =============================================================================
# Factory Functions
# =============================================================================

def create_solution_evaluator(
    evaluator_type: str = "default",
    **kwargs,
) -> SolutionEvaluator:
    """
    Factory function to create solution evaluators.

    Args:
        evaluator_type: Type of evaluator ("default" or custom)
        **kwargs: Arguments for the evaluator

    Returns:
        SolutionEvaluator instance
    """
    if evaluator_type == "default":
        return DefaultSolutionEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


def create_routing_heuristic(
    heuristic_type: str = "default",
    **kwargs,
) -> RoutingHeuristic:
    """
    Factory function to create routing heuristics.

    Args:
        heuristic_type: Type of heuristic ("default" or custom)
        **kwargs: Arguments for the heuristic

    Returns:
        RoutingHeuristic instance
    """
    if heuristic_type == "default":
        return DefaultRoutingHeuristic(**kwargs)
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")


def create_move_cost_function(
    cost_type: str = "default",
    **kwargs,
) -> MoveCostFunction:
    """
    Factory function to create move cost functions.

    Args:
        cost_type: Type of cost function ("default" or custom)
        **kwargs: Arguments for the cost function

    Returns:
        MoveCostFunction instance
    """
    if cost_type == "default":
        return DefaultMoveCostFunction(**kwargs)
    else:
        raise ValueError(f"Unknown cost function type: {cost_type}")
