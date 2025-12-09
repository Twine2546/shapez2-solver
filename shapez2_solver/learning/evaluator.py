"""
High-level solution evaluator.

Provides easy-to-use interface for evaluating routing solutions
before or after solving.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

from .features import (
    SolutionFeatures,
    extract_solution_features,
    calculate_crossing_potential,
)
from .models import QualityPredictor, DifficultyPredictor


@dataclass
class EvaluationResult:
    """Result of solution evaluation."""

    # Predictions
    success_probability: float = 0.0
    predicted_throughput: float = 0.0

    # Quality scores (0-1, higher = better)
    overall_score: float = 0.0
    routability_score: float = 0.0
    efficiency_score: float = 0.0
    congestion_score: float = 0.0

    # Warnings/suggestions
    warnings: List[str] = None
    suggestions: List[str] = None

    # Detailed features (for debugging)
    features: Optional[SolutionFeatures] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Overall Score: {self.overall_score:.2f}/1.00",
            f"Success Probability: {self.success_probability:.1%}",
            f"Predicted Throughput: {self.predicted_throughput:.1f} items/min",
            f"",
            f"Sub-scores:",
            f"  Routability: {self.routability_score:.2f}",
            f"  Efficiency:  {self.efficiency_score:.2f}",
            f"  Congestion:  {self.congestion_score:.2f}",
        ]

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for s in self.suggestions:
                lines.append(f"  - {s}")

        return "\n".join(lines)


class SolutionEvaluator:
    """
    Evaluates routing solutions for quality.

    Can be used:
    1. Before solving: evaluate placement to predict routability
    2. After solving: evaluate complete solution quality
    """

    def __init__(
        self,
        quality_model_path: Optional[str] = None,
        difficulty_model_path: Optional[str] = None,
    ):
        """
        Initialize evaluator.

        Args:
            quality_model_path: Path to trained quality model
            difficulty_model_path: Path to trained difficulty model
        """
        self.quality_predictor = QualityPredictor(quality_model_path)
        self.difficulty_predictor = DifficultyPredictor(difficulty_model_path)

    def evaluate_placement(
        self,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        machines: List[Tuple],
        input_positions: List[Tuple],
        output_positions: List[Tuple],
        connections: List[Tuple],
        occupied: Optional[Set[Tuple]] = None,
        foundation_type: str = "",
    ) -> EvaluationResult:
        """
        Evaluate a placement BEFORE routing.

        Useful for filtering/ranking candidate placements.
        """
        result = EvaluationResult()

        # Extract features (no belts/paths yet)
        features = extract_solution_features(
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machines=machines,
            belts=[],
            input_positions=input_positions,
            output_positions=output_positions,
            connections=connections,
            paths=None,
            occupied=occupied,
            foundation_type=foundation_type,
        )
        result.features = features

        # Get ML predictions
        success_prob, throughput = self.quality_predictor.predict(features)
        result.success_probability = success_prob
        result.predicted_throughput = throughput

        # Calculate sub-scores
        result.routability_score = self._calculate_routability_score(features, connections)
        result.efficiency_score = self._calculate_efficiency_score(features)
        result.congestion_score = self._calculate_congestion_score(features)

        # Overall score (weighted average)
        result.overall_score = (
            0.5 * result.routability_score +
            0.3 * result.efficiency_score +
            0.2 * result.congestion_score
        )

        # Generate warnings
        self._add_warnings(result, features)

        # Generate suggestions
        self._add_suggestions(result, features)

        return result

    def evaluate_solution(
        self,
        grid_width: int,
        grid_height: int,
        num_floors: int,
        machines: List[Tuple],
        belts: List[Tuple],
        input_positions: List[Tuple],
        output_positions: List[Tuple],
        connections: List[Tuple],
        paths: List[List[Tuple]],
        occupied: Optional[Set[Tuple]] = None,
        routing_success: bool = False,
        actual_throughput: float = 0.0,
        solve_time: float = 0.0,
        foundation_type: str = "",
    ) -> EvaluationResult:
        """
        Evaluate a complete solution AFTER routing.
        """
        result = EvaluationResult()

        # Extract full features
        features = extract_solution_features(
            grid_width=grid_width,
            grid_height=grid_height,
            num_floors=num_floors,
            machines=machines,
            belts=belts,
            input_positions=input_positions,
            output_positions=output_positions,
            connections=connections,
            paths=paths,
            occupied=occupied,
            routing_success=routing_success,
            throughput=actual_throughput,
            solve_time=solve_time,
            foundation_type=foundation_type,
        )
        result.features = features

        # Use actual values if available
        result.success_probability = 1.0 if routing_success else 0.0
        result.predicted_throughput = actual_throughput

        # Calculate sub-scores
        result.routability_score = features.routing_success_rate
        result.efficiency_score = self._calculate_efficiency_score(features)
        result.congestion_score = self._calculate_congestion_score(features)

        # Adjust efficiency based on actual path lengths
        if features.avg_path_stretch > 0:
            # Penalize long detours
            stretch_penalty = max(0, 1.0 - (features.avg_path_stretch - 1.0) * 0.2)
            result.efficiency_score *= stretch_penalty

        # Overall score
        result.overall_score = (
            0.5 * result.routability_score +
            0.3 * result.efficiency_score +
            0.2 * result.congestion_score
        )

        # Warnings/suggestions
        self._add_warnings(result, features)
        self._add_suggestions(result, features)

        return result

    def rank_placements(
        self,
        placements: List[Dict],
        grid_width: int,
        grid_height: int,
        num_floors: int,
    ) -> List[Tuple[int, EvaluationResult]]:
        """
        Rank multiple candidate placements by predicted quality.

        Args:
            placements: List of placement dicts with keys:
                - machines: List of machine tuples
                - input_positions: List of input positions
                - output_positions: List of output positions
                - connections: List of connections

        Returns:
            List of (index, evaluation_result) sorted by score (best first)
        """
        results = []

        for i, placement in enumerate(placements):
            eval_result = self.evaluate_placement(
                grid_width=grid_width,
                grid_height=grid_height,
                num_floors=num_floors,
                machines=placement.get('machines', []),
                input_positions=placement.get('input_positions', []),
                output_positions=placement.get('output_positions', []),
                connections=placement.get('connections', []),
                occupied=placement.get('occupied'),
                foundation_type=placement.get('foundation_type', ''),
            )
            results.append((i, eval_result))

        # Sort by overall score (descending)
        results.sort(key=lambda x: x[1].overall_score, reverse=True)

        return results

    def _calculate_routability_score(
        self,
        features: SolutionFeatures,
        connections: List[Tuple],
    ) -> float:
        """Calculate routability score (0-1)."""
        score = 1.0

        # Penalize high crossing potential
        crossing = calculate_crossing_potential(connections)
        score -= crossing * 0.3

        # Penalize low input-output separation (hard to route around machines)
        if features.input_output_separation < 0.3:
            score -= 0.2

        # Penalize too many connections relative to space
        connection_density = features.num_connections / max(1, features.total_cells / 100)
        if connection_density > 0.1:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _calculate_efficiency_score(self, features: SolutionFeatures) -> float:
        """Calculate space efficiency score (0-1)."""
        score = 1.0

        # Penalize very spread out machines (inefficient use of space)
        if features.machine_spread > 0.6:
            score -= 0.2

        # Reward balanced floor utilization
        score += features.floor_balance * 0.1

        # Penalize off-center layouts (usually harder to route)
        center_offset = abs(features.center_of_mass_x - 0.5) + abs(features.center_of_mass_y - 0.5)
        score -= center_offset * 0.1

        return max(0.0, min(1.0, score))

    def _calculate_congestion_score(self, features: SolutionFeatures) -> float:
        """Calculate congestion score (0-1, higher = less congested)."""
        score = 1.0

        # Penalize high local density
        score -= features.max_local_density_3x3 * 0.5

        # Penalize high variance (uneven distribution)
        score -= features.congestion_variance * 0.2

        return max(0.0, min(1.0, score))

    def _add_warnings(self, result: EvaluationResult, features: SolutionFeatures):
        """Add warnings based on features."""
        if features.max_local_density_3x3 > 0.7:
            result.warnings.append(
                f"High congestion detected ({features.max_local_density_3x3:.0%} in worst region)"
            )

        if features.crossing_potential > 0.5:
            result.warnings.append(
                f"Many potential path crossings ({features.crossing_potential:.0%})"
            )

        if features.input_output_separation < 0.2:
            result.warnings.append(
                "Inputs and outputs are very close together"
            )

        if features.num_connections > 8:
            result.warnings.append(
                f"High number of connections ({features.num_connections}) may cause routing conflicts"
            )

    def _add_suggestions(self, result: EvaluationResult, features: SolutionFeatures):
        """Add improvement suggestions."""
        if features.floor_balance < 0.3:
            result.suggestions.append(
                "Consider using multiple floors to reduce congestion"
            )

        if features.machine_clustering > 0.6:
            result.suggestions.append(
                "Machines are spread out - consider clustering for shorter paths"
            )

        if features.max_local_density_3x3 > 0.5 and features.floor_utilization[1] < 0.1:
            result.suggestions.append(
                "Floor 1 is underutilized - route some paths there"
            )


def quick_evaluate(
    machines: List[Tuple],
    connections: List[Tuple],
    grid_width: int,
    grid_height: int,
    num_floors: int = 4,
) -> float:
    """
    Quick evaluation without full feature extraction.

    Returns score 0-1 (higher = better predicted routability).
    """
    # Simple heuristics
    score = 1.0

    # Connection density
    total_cells = grid_width * grid_height * num_floors
    connection_density = len(connections) / max(1, total_cells / 50)
    score -= min(0.3, connection_density * 0.3)

    # Machine density
    machine_cells = len(machines) * 4  # Estimate
    machine_density = machine_cells / total_cells
    if machine_density > 0.3:
        score -= 0.2

    # Crossing potential
    crossing = calculate_crossing_potential(connections)
    score -= crossing * 0.3

    # Average connection length
    total_manhattan = 0
    for src, dst in connections:
        total_manhattan += abs(src[0] - dst[0]) + abs(src[1] - dst[1]) + abs(src[2] - dst[2])

    avg_manhattan = total_manhattan / max(1, len(connections))
    max_manhattan = grid_width + grid_height

    # Long connections are harder
    if avg_manhattan > max_manhattan * 0.5:
        score -= 0.15

    return max(0.0, min(1.0, score))


def compare_solutions(
    solutions: List[Dict],
    evaluator: Optional[SolutionEvaluator] = None,
) -> List[Tuple[int, EvaluationResult]]:
    """
    Compare multiple complete solutions.

    Args:
        solutions: List of solution dicts with all required fields
        evaluator: Optional pre-initialized evaluator

    Returns:
        Sorted list of (index, result) by quality
    """
    if evaluator is None:
        evaluator = SolutionEvaluator()

    results = []

    for i, sol in enumerate(solutions):
        result = evaluator.evaluate_solution(
            grid_width=sol['grid_width'],
            grid_height=sol['grid_height'],
            num_floors=sol.get('num_floors', 4),
            machines=sol.get('machines', []),
            belts=sol.get('belts', []),
            input_positions=sol.get('input_positions', []),
            output_positions=sol.get('output_positions', []),
            connections=sol.get('connections', []),
            paths=sol.get('paths', []),
            routing_success=sol.get('routing_success', False),
            actual_throughput=sol.get('throughput', 0),
        )
        results.append((i, result))

    results.sort(key=lambda x: x[1].overall_score, reverse=True)
    return results
