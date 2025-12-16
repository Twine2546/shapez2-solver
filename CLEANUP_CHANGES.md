# Code Cleanup Changes

This document describes the restructuring done to simplify the codebase and focus on the core functionality.

## Summary

The codebase has been restructured to focus on:
- **CP-SAT solver** for optimal machine placement
- **A* routing** with ML enhancements for belt connections
- **ML training** via `model_comparison.py`
- **Simulation** for solution validation

## Kept Functionality

### Core Solver (shapez2_solver/evolution/)
- `cpsat_solver.py` - CP-SAT constraint programming solver
- `router.py` - A* pathfinding router with ML-enhanced variants
- `foundation_config.py` - Foundation specifications and configurations
- `core_types.py` - Core data types (PlacedBuilding, Candidate, etc.)
- `databases.py` - Database utilities (TrainingSampleDB)
- `evaluation.py` - Solution evaluation
- `ml_evaluators.py` - ML evaluators for placement
- `advanced_ml_models.py` - CNN, GNN, Transformer models
- `ml_routing.py` - ML-enhanced routing
- `model_comparison.py` - ML training and comparison
- `problem_generator.py` - Problem generation for training
- `training_runner.py` - Training infrastructure

### Simulation (shapez2_solver/simulation/)
- `grid_simulator.py` - Grid-based simulation
- `flow_simulator.py` - Flow simulation
- `validate_solution.py` - Solution validation
- All simulation test files

### Other Kept Modules
- `shapez2_solver/shapes/` - Shape parsing and encoding
- `shapez2_solver/blueprint/` - Blueprint encoding and building types
- `shapez2_solver/operations/` - Shape operations
- `shapez2_solver/foundations/` - Foundation layouts
- `shapez2_solver/simulator/` - Design simulator
- `shapez2_solver/learning/` - Learning utilities
- `shapez2_solver/ui/cpsat_app.py` - CP-SAT GUI application
- `shapez2_solver/visualization/shape_renderer.py` - Shape rendering
- `shapez2_solver/visualization/solution_display.py` - Solution display

## Archived Files

The following files have been moved to the `archive/` folder for reference:

### Evolution Algorithms (archive/evolution/)
- `algorithm.py` - EvolutionaryAlgorithm
- `algorithms.py` - SimulatedAnnealing, HybridAlgorithm
- `candidate.py` - Evolution candidate
- `fitness.py` - Fitness functions
- `foundation_evolution.py` - FoundationEvolution class
- `grid_evolution.py` - Grid evolution
- `layout_search.py` - Layout search
- `operators.py` - Mutation/crossover operators
- `system_search.py` - System search
- `two_phase_evolution.py` - Two-phase evolution

### Visualization (archive/visualization/)
- `evolution_visualizer.py` - Evolution progress visualization
- `layout_viewer.py` - Tkinter layout viewer
- `pygame_layout_viewer.py` - Pygame layout viewer
- `view_training_samples.py` - Training sample viewer

### UI (archive/ui/)
- `app.py` - Old evolution-based UI

### Root Scripts (archive/)
- `main.py` - Old CLI with evolution support
- `evolve_foundation.py` - Evolution script

## New File Structure

```
shapez2_solver/evolution/
├── __init__.py           # Updated exports
├── core_types.py         # NEW: Core data types extracted
├── databases.py          # NEW: Database utilities extracted
├── cpsat_solver.py       # CP-SAT solver
├── router.py             # A* router
├── foundation_config.py  # Foundation specs
├── evaluation.py         # Evaluation
├── ml_evaluators.py      # ML evaluators
├── advanced_ml_models.py # Neural network models
├── ml_routing.py         # ML-enhanced routing
├── model_comparison.py   # ML training (simplified)
├── problem_generator.py  # Problem generation
└── training_runner.py    # Training infrastructure
```

## Usage

### ML Training
```bash
python -m shapez2_solver.evolution.model_comparison --ab-test --problems 20 --test-problems 10 --epochs 10
```

### Simulation Tests
```bash
python test_simulation.py
```

### CP-SAT Solver
```python
from shapez2_solver.evolution import CPSATFullSolver, FOUNDATION_SPECS

solver = CPSATFullSolver(
    foundation_type="2x2",
    input_specs=[("N", 0, 0, "CuCuCuCu")],
    output_specs=[("S", 0, 0, "CuCu----")],
)
solution = solver.solve()
```

## Algorithms Kept

1. **CP-SAT** - Constraint programming for machine placement
2. **CP-SAT with ML** - CP-SAT with ML placement evaluation
3. **A*** - A* pathfinding for belt routing
4. **A* with ML** - A* with learned heuristics and move costs

## Algorithms Removed

- Evolutionary algorithms (genetic algorithm, simulated annealing)
- Two-phase evolution
- Grid evolution
- Hybrid algorithms

All removed algorithms are preserved in the `archive/` folder for reference.
