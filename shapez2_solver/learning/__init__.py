"""
Learning module for Shapez 2 routing quality prediction.

This module provides tools to:
1. Extract features from layouts and routing solutions
2. Log routing attempts for training data collection
3. Train models to predict solution quality
4. Evaluate solutions before running expensive solvers

Usage:
    from shapez2_solver.learning import (
        SolutionFeatures,
        RoutingLogger,
        QualityPredictor,
        extract_features,
    )
"""

from .features import (
    SolutionFeatures,
    ConnectionFeatures,
    LocalRegionFeatures,
    extract_solution_features,
    extract_connection_features,
)

from .data_logger import (
    RoutingLogger,
    RoutingAttempt,
    DataStore,
)

from .models import (
    QualityPredictor,
    DifficultyPredictor,
    train_quality_model,
    train_difficulty_model,
)

from .evaluator import (
    SolutionEvaluator,
    quick_evaluate,
)

from .blueprint_downloader import (
    VortexAPI,
    BlueprintStore,
    DownloadedBlueprint,
    download_blueprints,
)

from .blueprint_analyzer import (
    analyze_blueprint,
    BlueprintAnalysis,
    extract_routing_problem,
    is_suitable_for_ml,
)

from .ml_models import (
    SolvabilityClassifier,
    DirectionPredictor,
    DirectionPredictor3D,
    MLGuidedRouter,
    extract_features,
    ProblemFeatures,
    DIRECTIONS,
)

from .continuous_learning import (
    ContinuousLearner,
)

__all__ = [
    # Features
    'SolutionFeatures',
    'ConnectionFeatures',
    'LocalRegionFeatures',
    'extract_solution_features',
    'extract_connection_features',
    # Logging
    'RoutingLogger',
    'RoutingAttempt',
    'DataStore',
    # Models
    'QualityPredictor',
    'DifficultyPredictor',
    'train_quality_model',
    'train_difficulty_model',
    # Evaluation
    'SolutionEvaluator',
    'quick_evaluate',
    # Blueprint downloading
    'VortexAPI',
    'BlueprintStore',
    'DownloadedBlueprint',
    'download_blueprints',
    # Blueprint analysis
    'analyze_blueprint',
    'BlueprintAnalysis',
    'extract_routing_problem',
    'is_suitable_for_ml',
    # ML Models
    'SolvabilityClassifier',
    'DirectionPredictor',
    'DirectionPredictor3D',
    'MLGuidedRouter',
    'extract_features',
    'ProblemFeatures',
    'DIRECTIONS',
    # Continuous Learning
    'ContinuousLearner',
]
