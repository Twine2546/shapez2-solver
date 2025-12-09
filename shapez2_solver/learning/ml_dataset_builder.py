#!/usr/bin/env python3
"""
ML Dataset Builder for Shapez 2 routing.

This script:
1. Downloads community blueprints from Shapez Vortex (positive examples)
2. Parses blueprints to extract routing problems
3. Runs CP-SAT solver to attempt re-solving the routing
4. Stores results:
   - Positive examples: Community blueprints (human solutions)
   - Negative examples: Solver failures (bad routing)

Usage:
    python ml_dataset_builder.py --download 500
    python ml_dataset_builder.py --solve --limit 100
    python ml_dataset_builder.py --stats
    python ml_dataset_builder.py --export training_data.json
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent to path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shapez2_solver.learning.blueprint_downloader import (
    VortexAPI,
    BlueprintStore,
    DownloadedBlueprint,
    download_blueprints,
)
from shapez2_solver.learning.blueprint_analyzer import (
    analyze_blueprint,
    BlueprintAnalysis,
    extract_routing_problem,
    is_suitable_for_ml,
)

# Try to import solver components
try:
    from shapez2_solver.evolution.cpsat_solver import GlobalBeltRouter
    from shapez2_solver.blueprint.building_types import BuildingType, Rotation
    HAS_SOLVER = True
except ImportError as e:
    print(f"Warning: Solver not available: {e}")
    HAS_SOLVER = False


@dataclass
class SolverTestResult:
    """Result of testing the solver on a blueprint."""
    blueprint_id: str
    blueprint_title: str
    solver_mode: str
    success: bool
    solve_time: float
    num_belts_original: int
    num_belts_solver: int
    error_message: str
    analysis: Dict[str, Any]
    routing_problem: Optional[Dict[str, Any]]


def test_solver_on_blueprint(
    blueprint: DownloadedBlueprint,
    solver_mode: str = 'global',
    time_limit: float = 30.0,
    verbose: bool = False,
) -> SolverTestResult:
    """
    Test the CP-SAT solver on a blueprint.

    We extract the routing problem from the blueprint and try to
    solve it from scratch using our solver.

    Args:
        blueprint: The downloaded blueprint
        solver_mode: Routing mode ('global', 'astar', 'hybrid')
        time_limit: Solver time limit in seconds
        verbose: Print debug info

    Returns:
        SolverTestResult with solve outcome
    """
    start_time = time.time()

    # Analyze the blueprint
    analysis = analyze_blueprint(blueprint.blueprint_code)

    # Check if suitable
    suitable, reason = is_suitable_for_ml(analysis)
    if not suitable:
        return SolverTestResult(
            blueprint_id=blueprint.id,
            blueprint_title=blueprint.title,
            solver_mode=solver_mode,
            success=False,
            solve_time=time.time() - start_time,
            num_belts_original=analysis.belt_count,
            num_belts_solver=0,
            error_message=f"Not suitable: {reason}",
            analysis=analysis.to_dict(),
            routing_problem=None,
        )

    # Extract routing problem
    routing_problem = extract_routing_problem(analysis)
    if not routing_problem:
        return SolverTestResult(
            blueprint_id=blueprint.id,
            blueprint_title=blueprint.title,
            solver_mode=solver_mode,
            success=False,
            solve_time=time.time() - start_time,
            num_belts_original=analysis.belt_count,
            num_belts_solver=0,
            error_message="Could not extract routing problem",
            analysis=analysis.to_dict(),
            routing_problem=None,
        )

    if not HAS_SOLVER:
        return SolverTestResult(
            blueprint_id=blueprint.id,
            blueprint_title=blueprint.title,
            solver_mode=solver_mode,
            success=False,
            solve_time=time.time() - start_time,
            num_belts_original=analysis.belt_count,
            num_belts_solver=0,
            error_message="Solver not available",
            analysis=analysis.to_dict(),
            routing_problem=routing_problem,
        )

    # Build occupied set from machines
    occupied = set()
    for machine in routing_problem['machines']:
        pos = tuple(machine['position'])
        occupied.add(pos)

    # Build connections from inputs to outputs (simplified)
    # In a real scenario, we'd need to trace the actual connections
    inputs = routing_problem.get('inputs', [])
    outputs = routing_problem.get('outputs', [])

    if not inputs or not outputs:
        return SolverTestResult(
            blueprint_id=blueprint.id,
            blueprint_title=blueprint.title,
            solver_mode=solver_mode,
            success=False,
            solve_time=time.time() - start_time,
            num_belts_original=analysis.belt_count,
            num_belts_solver=0,
            error_message="No inputs or outputs found",
            analysis=analysis.to_dict(),
            routing_problem=routing_problem,
        )

    # Create simple connections (each input to each output - simplified)
    # For real ML training, we'd need to trace actual connections
    connections = []
    for i, inp in enumerate(inputs[:4]):  # Limit to avoid explosion
        if i < len(outputs):
            connections.append((tuple(inp), tuple(outputs[i])))

    if not connections:
        return SolverTestResult(
            blueprint_id=blueprint.id,
            blueprint_title=blueprint.title,
            solver_mode=solver_mode,
            success=False,
            solve_time=time.time() - start_time,
            num_belts_original=analysis.belt_count,
            num_belts_solver=0,
            error_message="No connections to route",
            analysis=analysis.to_dict(),
            routing_problem=routing_problem,
        )

    try:
        # Run the solver
        router = GlobalBeltRouter(
            grid_width=routing_problem['grid_width'],
            grid_height=routing_problem['grid_height'],
            num_floors=routing_problem['num_floors'],
            occupied=occupied,
            time_limit=time_limit,
        )

        belts, success = router.route_all(connections, verbose=verbose)
        solve_time = time.time() - start_time

        return SolverTestResult(
            blueprint_id=blueprint.id,
            blueprint_title=blueprint.title,
            solver_mode=solver_mode,
            success=success,
            solve_time=solve_time,
            num_belts_original=analysis.belt_count,
            num_belts_solver=len(belts),
            error_message="" if success else "Routing failed",
            analysis=analysis.to_dict(),
            routing_problem=routing_problem,
        )

    except Exception as e:
        return SolverTestResult(
            blueprint_id=blueprint.id,
            blueprint_title=blueprint.title,
            solver_mode=solver_mode,
            success=False,
            solve_time=time.time() - start_time,
            num_belts_original=analysis.belt_count,
            num_belts_solver=0,
            error_message=f"Solver error: {e}",
            analysis=analysis.to_dict(),
            routing_problem=routing_problem,
        )


def build_dataset(
    db_path: str = "blueprints.db",
    max_blueprints: int = 100,
    solver_mode: str = 'global',
    time_limit: float = 30.0,
    min_buildings: int = 5,
    max_buildings: int = 100,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """
    Build ML dataset by solving blueprints.

    Args:
        db_path: Path to blueprint database
        max_blueprints: Max blueprints to process
        solver_mode: Routing mode to use
        time_limit: Solver time limit per blueprint
        min_buildings: Minimum building count filter
        max_buildings: Maximum building count filter
        verbose: Print debug info

    Returns:
        Tuple of (processed, successes, failures)
    """
    store = BlueprintStore(db_path)

    # Get blueprints not yet solved
    blueprints = store.get_blueprints(
        min_buildings=min_buildings,
        max_buildings=max_buildings,
        limit=max_blueprints,
        exclude_solved=True,
        solver_mode=solver_mode,
    )

    if not blueprints:
        print("No blueprints to process (may already be solved or no data)")
        return 0, 0, 0

    print(f"\nProcessing {len(blueprints)} blueprints...")
    print(f"  Solver mode: {solver_mode}")
    print(f"  Time limit: {time_limit}s per blueprint")
    print()

    processed = 0
    successes = 0
    failures = 0

    for i, bp in enumerate(blueprints):
        print(f"[{i+1}/{len(blueprints)}] {bp.title[:40]}...", end=" ", flush=True)

        result = test_solver_on_blueprint(
            bp,
            solver_mode=solver_mode,
            time_limit=time_limit,
            verbose=verbose,
        )

        # Store result
        store.save_solver_result(
            blueprint_id=bp.id,
            solver_mode=solver_mode,
            success=result.success,
            solve_time=result.solve_time,
            num_belts=result.num_belts_solver,
            throughput=0.0,  # Would need simulation to calculate
            error_message=result.error_message,
            features=result.analysis,
        )

        processed += 1
        if result.success:
            successes += 1
            print(f"OK ({result.solve_time:.1f}s, {result.num_belts_solver} belts)")
        else:
            failures += 1
            print(f"FAIL: {result.error_message[:50]}")

    print(f"\nProcessed: {processed}")
    print(f"Successes: {successes} ({100*successes/max(1,processed):.1f}%)")
    print(f"Failures: {failures} ({100*failures/max(1,processed):.1f}%)")

    return processed, successes, failures


def export_training_data(
    db_path: str = "blueprints.db",
    output_path: str = "training_data.json",
    include_positive: bool = True,
    include_negative: bool = True,
) -> int:
    """
    Export training data to JSON file.

    Args:
        db_path: Path to blueprint database
        output_path: Output JSON file path
        include_positive: Include successful solves
        include_negative: Include failed solves

    Returns:
        Number of examples exported
    """
    store = BlueprintStore(db_path)

    data = []

    if include_positive:
        positive = store.get_training_data(positive_only=True)
        for item in positive:
            item['label'] = 'positive'
            item['source'] = 'community'  # Human solution
            data.append(item)

    if include_negative:
        negative = store.get_training_data(negative_only=True)
        for item in negative:
            item['label'] = 'negative'
            item['source'] = 'solver_failure'  # Bad routing example
            data.append(item)

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Exported {len(data)} examples to {output_path}")
    print(f"  Positive: {sum(1 for d in data if d['label'] == 'positive')}")
    print(f"  Negative: {sum(1 for d in data if d['label'] == 'negative')}")

    return len(data)


def print_stats(db_path: str = "blueprints.db"):
    """Print database statistics."""
    store = BlueprintStore(db_path)
    stats = store.get_stats()

    print("\n" + "="*60)
    print("Blueprint Database Statistics")
    print("="*60)

    print(f"\nTotal blueprints: {stats['total_blueprints']}")

    print("\nBy type:")
    for bp_type, count in stats.get('by_type', {}).items():
        print(f"  {bp_type}: {count}")

    print("\nBuilding count distribution:")
    for range_name, count in stats.get('building_count_ranges', {}).items():
        print(f"  {range_name}: {count}")

    print("\nSolver results:")
    for mode, results in stats.get('solver_results', {}).items():
        total = results['total']
        successes = results['successes']
        rate = 100 * successes / total if total > 0 else 0
        avg_time = results['avg_time']
        print(f"  {mode}:")
        print(f"    Total: {total}")
        print(f"    Successes: {successes} ({rate:.1f}%)")
        print(f"    Avg solve time: {avg_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Build ML dataset for Shapez 2 routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 500 blueprints from Shapez Vortex
    python ml_dataset_builder.py --download 500

    # Run solver on 100 blueprints
    python ml_dataset_builder.py --solve --limit 100

    # Show statistics
    python ml_dataset_builder.py --stats

    # Export training data
    python ml_dataset_builder.py --export training_data.json
        """
    )

    parser.add_argument("--db", type=str, default="blueprints.db",
                       help="Database file path")

    # Download options
    parser.add_argument("--download", type=int, metavar="COUNT",
                       help="Download COUNT blueprints from Shapez Vortex")
    parser.add_argument("--tags", type=str, nargs="*",
                       help="Tags to filter downloads")

    # Solve options
    parser.add_argument("--solve", action="store_true",
                       help="Run solver on blueprints")
    parser.add_argument("--limit", type=int, default=100,
                       help="Max blueprints to process")
    parser.add_argument("--mode", type=str, default="global",
                       choices=["global", "astar", "hybrid"],
                       help="Solver routing mode")
    parser.add_argument("--time-limit", type=float, default=30.0,
                       help="Solver time limit per blueprint (seconds)")
    parser.add_argument("--min-buildings", type=int, default=5,
                       help="Minimum building count filter")
    parser.add_argument("--max-buildings", type=int, default=100,
                       help="Maximum building count filter")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    # Export options
    parser.add_argument("--export", type=str, metavar="FILE",
                       help="Export training data to JSON file")
    parser.add_argument("--positive-only", action="store_true",
                       help="Export only positive examples")
    parser.add_argument("--negative-only", action="store_true",
                       help="Export only negative examples")

    # Stats
    parser.add_argument("--stats", action="store_true",
                       help="Show database statistics")

    args = parser.parse_args()

    # Handle actions
    if args.download:
        print(f"Downloading {args.download} blueprints...")
        download_blueprints(
            max_count=args.download,
            db_path=args.db,
            tags=args.tags,
        )

    if args.solve:
        build_dataset(
            db_path=args.db,
            max_blueprints=args.limit,
            solver_mode=args.mode,
            time_limit=args.time_limit,
            min_buildings=args.min_buildings,
            max_buildings=args.max_buildings,
            verbose=args.verbose,
        )

    if args.export:
        export_training_data(
            db_path=args.db,
            output_path=args.export,
            include_positive=not args.negative_only,
            include_negative=not args.positive_only,
        )

    if args.stats:
        print_stats(args.db)

    # Default: show help if no action
    if not any([args.download, args.solve, args.export, args.stats]):
        parser.print_help()


if __name__ == "__main__":
    main()
