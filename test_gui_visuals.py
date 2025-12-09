#!/usr/bin/env python3
"""Quick test of GUI with foundation and I/O visuals."""

from shapez2_solver.ui.cpsat_app import main

if __name__ == "__main__":
    print("Launching CP-SAT GUI with visual previews...")
    print("Features:")
    print("  - Foundation shape preview (updates when selecting different foundations)")
    print("  - I/O port visualization (green = inputs, red = outputs)")
    print("  - Auto-scaling with irregular foundations (T, L, L4, S4, Cross)")
    print()
    main()
