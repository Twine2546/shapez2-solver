"""Archive module - contains old/unused code.

This module contains deprecated code that was used in earlier versions
of the solver. These files are kept for reference but are not used
by the current implementation.

Archived files:
- algorithm.py: Old evolutionary algorithm implementation
- algorithms.py: Simulated annealing and hybrid algorithms
- operators.py: Genetic operators (mutation, crossover)
- fitness.py: Fitness functions
- grid_evolution.py: Legacy grid-based evolution
- system_search.py: Two-phase system search
- layout_search.py: Two-phase layout search
- two_phase_evolution.py: Two-phase evolution coordinator
- foundation_evolution.py: Foundation-aware evolution
- candidate.py: Old Candidate class (design-based)

The active solver uses CP-SAT (constraint programming) instead of
evolutionary algorithms. See shapez2_solver.evolution.cpsat_solver.
"""
