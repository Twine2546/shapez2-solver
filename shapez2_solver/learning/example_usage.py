#!/usr/bin/env python3
"""
Example usage of the learning module.

Demonstrates:
1. Logging routing attempts
2. Training models from logged data
3. Evaluating solutions
4. Using learned difficulty for routing order
"""

import sys
from pathlib import Path

# Add parent to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shapez2_solver.learning import (
    # Features
    SolutionFeatures,
    extract_solution_features,
    extract_connection_features,
    # Logging
    RoutingLogger,
    DataStore,
    # Models
    QualityPredictor,
    DifficultyPredictor,
    train_quality_model,
    train_difficulty_model,
    # Evaluation
    SolutionEvaluator,
    quick_evaluate,
)
from shapez2_solver.learning.data_logger import create_sample_data


def example_feature_extraction():
    """Example: Extract features from a solution."""
    print("\n" + "="*60)
    print("Example 1: Feature Extraction")
    print("="*60)

    # Example solution data
    machines = [
        ("SPLITTER", 5, 5, 0, "NORTH"),
        ("SPLITTER", 10, 5, 0, "NORTH"),
    ]

    belts = [
        (3, 5, 0, "BELT", "EAST"),
        (4, 5, 0, "BELT", "EAST"),
        (6, 5, 0, "BELT", "EAST"),
        (7, 5, 0, "BELT", "EAST"),
    ]

    input_positions = [(0, 5, 0)]
    output_positions = [(14, 3, 0), (14, 7, 0)]

    connections = [
        ((0, 5, 0), (5, 5, 0)),   # input to splitter 1
        ((5, 5, 0), (14, 3, 0)),  # splitter 1 to output 1
        ((5, 5, 0), (14, 7, 0)),  # splitter 1 to output 2
    ]

    # Extract features
    features = extract_solution_features(
        grid_width=14,
        grid_height=14,
        num_floors=3,
        machines=machines,
        belts=belts,
        input_positions=input_positions,
        output_positions=output_positions,
        connections=connections,
        paths=[[(0,5,0), (1,5,0), (2,5,0)], [(6,5,0), (7,5,0)], [(6,5,0), (6,6,0)]],
        routing_success=True,
        throughput=90.0,
        foundation_type="1x1",
    )

    print(f"\nExtracted features for 1x1 foundation:")
    print(f"  Machines: {features.num_machines}")
    print(f"  Machine density: {features.machine_density:.3f}")
    print(f"  Belts: {features.num_belts}")
    print(f"  Belt density: {features.belt_density:.3f}")
    print(f"  Connections: {features.num_connections}")
    print(f"  Avg path length: {features.avg_path_length:.1f}")
    print(f"  Max local density (3x3): {features.max_local_density_3x3:.3f}")
    print(f"  Input-output separation: {features.input_output_separation:.3f}")
    print(f"  Crossing potential: {features.crossing_potential:.3f}")

    # Get feature vector for ML
    vector = features.to_feature_vector()
    print(f"\nFeature vector length: {len(vector)}")
    print(f"Feature names: {SolutionFeatures.feature_names()[:5]}...")


def example_logging():
    """Example: Log routing attempts for training data."""
    print("\n" + "="*60)
    print("Example 2: Logging Routing Attempts")
    print("="*60)

    # Create logger with test database
    logger = RoutingLogger("test_routing_data.db")

    # Simulate starting a routing attempt
    logger.start_attempt(
        foundation_type="2x2",
        grid_width=34,
        grid_height=34,
        num_floors=3,
        routing_mode="hybrid",
        input_positions=[(0, 10, 0), (0, 20, 0)],
        output_positions=[(33, 10, 0), (33, 20, 0)],
        connections=[
            ((0, 10, 0), (33, 10, 0)),
            ((0, 20, 0), (33, 20, 0)),
        ],
    )

    # Simulate logging machines
    machines = [
        ("SPLITTER", 15, 15, 0, "EAST"),
    ]
    logger.log_machines(machines)

    # Simulate logging result
    belts = [(x, 10, 0, "BELT", "EAST") for x in range(1, 33)]
    paths = [
        [(x, 10, 0) for x in range(34)],
        [(x, 20, 0) for x in range(34)],
    ]

    logger.log_result(
        machines=machines,
        belts=belts,
        paths=paths,
        input_positions=[(0, 10, 0), (0, 20, 0)],
        output_positions=[(33, 10, 0), (33, 20, 0)],
        connections=[((0, 10, 0), (33, 10, 0)), ((0, 20, 0), (33, 20, 0))],
        occupied=set(),
        routing_success=True,
        throughput=180.0,
    )

    # Save to database
    logger.save()

    print("\nLogged 1 routing attempt")
    print(f"Database stats: {logger.get_stats()}")


def example_training():
    """Example: Train models from logged data."""
    print("\n" + "="*60)
    print("Example 3: Training Models")
    print("="*60)

    # Create sample data for training
    print("\nCreating 200 sample routing attempts...")
    create_sample_data("training_data.db", num_samples=200)

    # Train quality predictor
    print("\nTraining quality model...")
    predictor, metrics = train_quality_model(
        db_path="training_data.db",
        model_path="test_quality_model.pkl",
    )

    print(f"\nModel trained!")
    print(f"  Accuracy: {metrics.accuracy:.1%}")
    print(f"  Cross-validation: {metrics.cross_val_mean:.1%} (+/- {metrics.cross_val_std:.1%})")

    # Train difficulty predictor
    print("\nTraining difficulty model...")
    diff_predictor, diff_metrics = train_difficulty_model(
        db_path="training_data.db",
        model_path="test_difficulty_model.pkl",
    )


def example_evaluation():
    """Example: Evaluate solutions."""
    print("\n" + "="*60)
    print("Example 4: Solution Evaluation")
    print("="*60)

    # Create evaluator (uses heuristics if no model trained)
    evaluator = SolutionEvaluator()

    # Evaluate a placement (before routing)
    machines = [
        ("SPLITTER", 17, 17, 0, "EAST"),
        ("SPLITTER", 17, 25, 0, "EAST"),
    ]

    connections = [
        ((0, 17, 0), (17, 17, 0)),
        ((17, 17, 0), (33, 10, 0)),
        ((17, 17, 0), (33, 17, 0)),
        ((0, 25, 0), (17, 25, 0)),
        ((17, 25, 0), (33, 25, 0)),
        ((17, 25, 0), (33, 30, 0)),
    ]

    result = evaluator.evaluate_placement(
        grid_width=34,
        grid_height=34,
        num_floors=3,
        machines=machines,
        input_positions=[(0, 17, 0), (0, 25, 0)],
        output_positions=[(33, 10, 0), (33, 17, 0), (33, 25, 0), (33, 30, 0)],
        connections=connections,
        foundation_type="2x2",
    )

    print("\nPlacement evaluation:")
    print(result.summary())

    # Quick evaluation (no evaluator needed)
    print("\n\nQuick evaluation score:", quick_evaluate(
        machines=machines,
        connections=connections,
        grid_width=34,
        grid_height=34,
    ))


def example_difficulty_ranking():
    """Example: Rank connections by difficulty."""
    print("\n" + "="*60)
    print("Example 5: Connection Difficulty Ranking")
    print("="*60)

    predictor = DifficultyPredictor()

    connections = [
        ((0, 5, 0), (30, 5, 0)),    # Long horizontal
        ((15, 0, 0), (15, 30, 0)),  # Long vertical (crosses center)
        ((0, 0, 0), (5, 5, 0)),     # Short diagonal
        ((0, 15, 0), (30, 15, 1)),  # Long + floor change
    ]

    occupied = {(15, 15, 0), (16, 15, 0), (15, 16, 0)}  # Some obstacles

    ranking = predictor.rank_connections(
        connections=connections,
        grid_width=34,
        grid_height=34,
        occupied=occupied,
    )

    print("\nConnections ranked by difficulty (easiest first):")
    for rank, idx in enumerate(ranking):
        src, dst = connections[idx]
        features = extract_connection_features(
            connection=(src, dst),
            grid_width=34,
            grid_height=34,
            occupied=occupied,
        )
        difficulty = predictor.predict(features)
        print(f"  {rank+1}. Connection {idx}: {src} -> {dst}")
        print(f"     Difficulty: {difficulty:.2f}, Manhattan: {features.manhattan_distance}")


def example_compare_foundations():
    """Example: Compare solutions across different foundations."""
    print("\n" + "="*60)
    print("Example 6: Comparing Across Foundations")
    print("="*60)

    evaluator = SolutionEvaluator()

    # Same logical problem on different foundations
    foundations = [
        ("1x1", 14, 14),
        ("2x1", 34, 14),
        ("2x2", 34, 34),
        ("3x3", 54, 54),
    ]

    print("\nEvaluating 1-to-4 split on different foundations:")
    print("-" * 50)

    for name, width, height in foundations:
        # Adapt positions to foundation size
        cx, cy = width // 2, height // 2

        machines = [("SPLITTER", cx, cy, 0, "EAST")]

        connections = [
            ((0, cy, 0), (cx, cy, 0)),
            ((cx, cy, 0), (width-1, cy-3, 0)),
            ((cx, cy, 0), (width-1, cy-1, 0)),
            ((cx, cy, 0), (width-1, cy+1, 0)),
            ((cx, cy, 0), (width-1, cy+3, 0)),
        ]

        result = evaluator.evaluate_placement(
            grid_width=width,
            grid_height=height,
            num_floors=3,
            machines=machines,
            input_positions=[(0, cy, 0)],
            output_positions=[
                (width-1, cy-3, 0),
                (width-1, cy-1, 0),
                (width-1, cy+1, 0),
                (width-1, cy+3, 0),
            ],
            connections=connections,
            foundation_type=name,
        )

        print(f"\n{name} ({width}x{height}):")
        print(f"  Overall Score: {result.overall_score:.2f}")
        print(f"  Success Prob:  {result.success_probability:.1%}")
        print(f"  Congestion:    {result.congestion_score:.2f}")


def main():
    """Run all examples."""
    print("="*60)
    print("Learning Module Examples")
    print("="*60)

    example_feature_extraction()
    example_logging()
    example_evaluation()
    example_difficulty_ranking()
    example_compare_foundations()

    # Training example requires sklearn
    try:
        import sklearn
        example_training()
    except ImportError:
        print("\n[Skipping training example - sklearn not installed]")

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)

    # Cleanup test files
    import os
    for f in ["test_routing_data.db", "training_data.db",
              "test_quality_model.pkl", "test_difficulty_model.pkl"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up: {f}")


if __name__ == "__main__":
    main()
