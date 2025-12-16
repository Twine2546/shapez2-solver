# Changelog

## [Cleanup] - 2024-12-16

Major cleanup and restructuring of the codebase to focus on the CP-SAT solver.

### Summary of Changes

The solver has been streamlined to focus on **CP-SAT constraint programming** as the
primary solving approach, with **A* pathfinding** for belt routing. Evolutionary
algorithms and other search methods have been archived.

### Kept Functionality

1. **CP-SAT Solver** (`shapez2_solver/evolution/cpsat_solver.py`)
   - Constraint programming solver using Google OR-Tools
   - Port-aware placement heuristics
   - Flow direction optimization
   - Automatic machine selection based on input/output requirements

2. **A* Belt Router** (`shapez2_solver/evolution/router.py`)
   - Pathfinding-based belt routing
   - Multi-floor support
   - Belt port (teleporter) support
   - Collision avoidance

3. **ML Training** (`shapez2_solver/evolution/model_comparison.py`)
   - A/B testing framework for heuristic comparison
   - Problem generation for training
   - Heuristic evolution for placement optimization
   - Usage: `python -m shapez2_solver.evolution.model_comparison --ab-test --problems 20 --test-problems 10 --epochs 10`

4. **Core Data Structures** (`shapez2_solver/evolution/core.py`)
   - `PlacedBuilding` - Building placement on grid
   - `Candidate` - Solution with buildings and fitness
   - `SolverResult` / `FoundationEvolution` - Interface for visualization

5. **Foundation Configuration** (`shapez2_solver/evolution/foundation_config.py`)
   - All foundation specifications (1x1 to 3x3, T, L, S, Cross shapes)
   - Grid dimensions and port configurations

6. **Visualization**
   - `pygame_layout_viewer.py` - Interactive layout viewer
   - `layout_viewer.py` - Tkinter-based viewer
   - Shape rendering and solution display

7. **Simulator** (`shapez2_solver/simulator/`)
   - Design representation and simulation
   - Shape operations and transformations

8. **Blueprint Export** (`shapez2_solver/blueprint/`)
   - Export solutions to Shapez 2 blueprint format

### Archived Code

The following files have been moved to `shapez2_solver/archive/`:

- `algorithm.py` - Evolutionary algorithm implementation
- `algorithms.py` - Simulated annealing and hybrid algorithms
- `operators.py` - Genetic operators (mutation, crossover)
- `fitness.py` - Fitness functions
- `grid_evolution.py` - Legacy grid-based evolution
- `system_search.py` - Two-phase system search
- `layout_search.py` - Two-phase layout search
- `two_phase_evolution.py` - Two-phase evolution coordinator
- `foundation_evolution.py` - Foundation-aware evolution class
- `candidate.py` - Old Candidate class (design-based)
- `evolution_visualizer.py` - Evolution progress visualization
- `old_app.py` - Old multi-algorithm UI

### Updated Entry Points

**Main CLI** (`main.py`):
```bash
# Run GUI
python main.py gui

# Solve with CP-SAT
python main.py solve -f 2x2 -i "W,0,0,CuCuCuCu" -o "E,0,0,Cu------" -o "E,1,0,--Cu----"

# Parse shape code
python main.py parse CuCuCuCu

# List foundations
python main.py foundations
```

**CP-SAT GUI** (`run_cpsat_gui.py`):
```bash
python run_cpsat_gui.py
```

**ML Training**:
```bash
python -m shapez2_solver.evolution.model_comparison --ab-test --problems 20 --test-problems 10 --epochs 10
```

### Algorithm Focus

The solver now supports four main approaches:

1. **CP-SAT** - Pure constraint programming (default)
2. **CP-SAT + ML** - CP-SAT with learned placement heuristics
3. **A*** - Pure pathfinding for routing
4. **A* + ML** - A* with learned cost functions

### File Structure After Cleanup

```
shapez2_solver/
├── archive/           # Archived old code
├── blueprint/         # Blueprint export
├── evolution/         # Solver module
│   ├── __init__.py
│   ├── core.py        # Core data structures
│   ├── cpsat_solver.py
│   ├── foundation_config.py
│   ├── model_comparison.py  # ML training
│   └── router.py
├── foundations/       # Foundation definitions
├── operations/        # Shape operations
├── shapes/           # Shape representation
├── simulation/       # Grid simulation
├── simulator/        # Design simulation
├── ui/              # User interfaces
│   ├── __init__.py
│   └── cpsat_app.py
└── visualization/    # Visualization tools
```

### Breaking Changes

- `EvolutionaryAlgorithm`, `EvolutionConfig` no longer available from main module
- Old `SolverApp` GUI replaced with `CPSATSolverApp`
- `main.py solve` command now uses CP-SAT instead of evolutionary algorithm
