#!/usr/bin/env python3
"""
Benchmark comparison: ML-enhanced vs baseline routing.

Generates synthetic problems and compares:
1. Baseline A* routing (random connection order)
2. ML-enhanced routing (smart connection ordering + placement filtering)

Measures:
- Success rate
- Average solve time
- Total nodes explored
- Routing efficiency (belts per connection)
"""

import argparse
import json
import time
import random
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .synthetic_generator import (
    SyntheticProblemGenerator,
    SyntheticProblem,
    _solve_with_ml_data,
)
from .features import extract_connection_features
from ..evolution.router import BeltRouter, Connection
from ..blueprint.building_types import Rotation


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    problem_id: str
    difficulty: str
    mode: str  # 'baseline' or 'ml_enhanced'

    # Outcome
    success: bool
    connections_routed: int
    connections_total: int
    routing_progress: float

    # Performance
    solve_time: float
    total_nodes_explored: int
    num_belts: int

    # Efficiency
    belts_per_connection: float
    nodes_per_connection: float


def solve_baseline(problem: SyntheticProblem) -> BenchmarkResult:
    """
    Solve using baseline A* routing with default connection order.
    """
    start_time = time.time()

    router = BeltRouter(
        problem.grid_width,
        problem.grid_height,
        problem.num_floors,
    )
    router.set_occupied(problem.get_occupied())

    all_belts = []
    failed_indices = []
    total_nodes = 0

    # Route in original order (no ML optimization)
    for i, (src, dst) in enumerate(problem.connections):
        conn = Connection(
            from_pos=src,
            to_pos=dst,
            from_direction=Rotation.EAST,
            to_direction=Rotation.EAST,
        )

        result = router.route_connection_with_stats(conn)
        total_nodes += result.nodes_explored

        if result.success:
            all_belts.extend(result.belts)
        else:
            failed_indices.append(i)

    solve_time = time.time() - start_time
    num_connections = len(problem.connections)
    connections_routed = num_connections - len(failed_indices)

    return BenchmarkResult(
        problem_id=problem.problem_id,
        difficulty=problem.difficulty,
        mode='baseline',
        success=len(failed_indices) == 0,
        connections_routed=connections_routed,
        connections_total=num_connections,
        routing_progress=connections_routed / max(1, num_connections),
        solve_time=solve_time,
        total_nodes_explored=total_nodes,
        num_belts=len(all_belts),
        belts_per_connection=len(all_belts) / max(1, connections_routed),
        nodes_per_connection=total_nodes / max(1, num_connections),
    )


def solve_with_reroute(problem: SyntheticProblem, use_smart_ports: bool = False) -> BenchmarkResult:
    """
    Solve using the reroute loop - when routing fails, identify blockers and retry.
    """
    start_time = time.time()

    router = BeltRouter(
        problem.grid_width,
        problem.grid_height,
        problem.num_floors,
    )
    router.set_occupied(problem.get_occupied())

    # Build connections list
    connections = []
    for src, dst in problem.connections:
        conn = Connection(
            from_pos=src,
            to_pos=dst,
            from_direction=Rotation.EAST,
            to_direction=Rotation.EAST,
        )
        connections.append(conn)

    # Use the reroute loop
    if use_smart_ports:
        results, stats = router.route_all_with_retry(connections, max_retries=3, use_smart_ports=True)
    else:
        results, stats = router.route_with_reroute_loop(connections, max_retries=3)

    solve_time = time.time() - start_time

    # Count successes and belts
    all_belts = []
    failed_indices = []
    total_nodes = 0

    for i, result in enumerate(results):
        total_nodes += getattr(result, 'nodes_explored', 0)
        if result.success:
            all_belts.extend(result.belts)
        else:
            failed_indices.append(i)

    num_connections = len(problem.connections)
    connections_routed = num_connections - len(failed_indices)

    mode = 'reroute_smart' if use_smart_ports else 'reroute_loop'

    return BenchmarkResult(
        problem_id=problem.problem_id,
        difficulty=problem.difficulty,
        mode=mode,
        success=len(failed_indices) == 0,
        connections_routed=connections_routed,
        connections_total=num_connections,
        routing_progress=connections_routed / max(1, num_connections),
        solve_time=solve_time,
        total_nodes_explored=total_nodes,
        num_belts=len(all_belts),
        belts_per_connection=len(all_belts) / max(1, connections_routed),
        nodes_per_connection=total_nodes / max(1, num_connections),
    )


def solve_ml_enhanced(
    problem: SyntheticProblem,
    connection_predictor: Optional[Any] = None,
    placement_predictor: Optional[Any] = None,
) -> BenchmarkResult:
    """
    Solve using ML-enhanced routing:
    1. Order connections by predicted difficulty (easy first)
    2. Use connection quality predictions for early termination
    """
    start_time = time.time()

    router = BeltRouter(
        problem.grid_width,
        problem.grid_height,
        problem.num_floors,
    )
    initial_occupied = problem.get_occupied()
    router.set_occupied(initial_occupied)

    # Build connection data with features
    connection_data = []
    current_occupied = initial_occupied.copy()

    for i, (src, dst) in enumerate(problem.connections):
        conn_features = extract_connection_features(
            connection=(src, dst),
            grid_width=problem.grid_width,
            grid_height=problem.grid_height,
            occupied=current_occupied,
            connection_index=i,
            total_connections=len(problem.connections),
            connections_routed=0,
        )

        connection_data.append({
            'index': i,
            'src': src,
            'dst': dst,
            'features': {
                'manhattan_distance': conn_features.manhattan_distance,
                'normalized_distance': conn_features.normalized_distance,
                'src_local_density': conn_features.src_local_density,
                'dst_local_density': conn_features.dst_local_density,
                'path_corridor_density': conn_features.path_corridor_density,
                'crosses_center': conn_features.crosses_center,
                'floor_change': conn_features.floor_change,
                'direction_complexity': conn_features.direction_complexity,
            },
        })

    # Order connections by ML prediction (or heuristic fallback)
    if connection_predictor and connection_predictor.is_trained:
        ordered = connection_predictor.order_connections(connection_data)
        routing_order = [idx for idx, _ in ordered]
    else:
        # Fallback: order by manhattan distance (shorter first)
        sorted_data = sorted(connection_data,
                           key=lambda x: x['features']['manhattan_distance'])
        routing_order = [d['index'] for d in sorted_data]

    # Route in optimized order
    all_belts = []
    failed_indices = []
    total_nodes = 0

    for i in routing_order:
        src, dst = problem.connections[i]
        conn = Connection(
            from_pos=src,
            to_pos=dst,
            from_direction=Rotation.EAST,
            to_direction=Rotation.EAST,
        )

        result = router.route_connection_with_stats(conn)
        total_nodes += result.nodes_explored

        if result.success:
            all_belts.extend(result.belts)
        else:
            failed_indices.append(i)

    solve_time = time.time() - start_time
    num_connections = len(problem.connections)
    connections_routed = num_connections - len(failed_indices)

    return BenchmarkResult(
        problem_id=problem.problem_id,
        difficulty=problem.difficulty,
        mode='ml_enhanced',
        success=len(failed_indices) == 0,
        connections_routed=connections_routed,
        connections_total=num_connections,
        routing_progress=connections_routed / max(1, num_connections),
        solve_time=solve_time,
        total_nodes_explored=total_nodes,
        num_belts=len(all_belts),
        belts_per_connection=len(all_belts) / max(1, connections_routed),
        nodes_per_connection=total_nodes / max(1, num_connections),
    )


def run_benchmark(
    num_problems: int = 100,
    difficulties: List[str] = None,
    model_dir: str = "models",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the benchmark comparison.

    Args:
        num_problems: Number of problems to generate
        difficulties: List of difficulties to test
        model_dir: Directory containing trained models
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dict with benchmark results and statistics
    """
    if difficulties is None:
        difficulties = ['easy', 'medium', 'hard']

    # Try to load ML models
    connection_predictor = None
    placement_predictor = None

    model_path = Path(model_dir)

    try:
        from .ml_models import ConnectionQualityPredictor
        conn_model_path = model_path / "connection_predictor.pkl"
        if conn_model_path.exists():
            connection_predictor = ConnectionQualityPredictor()
            connection_predictor.load(str(conn_model_path))
            if verbose:
                print(f"Loaded connection predictor from {conn_model_path}")
    except Exception as e:
        if verbose:
            print(f"Could not load connection predictor: {e}")

    try:
        from .ml_models import PlacementPredictor
        place_model_path = model_path / "placement_predictor.pkl"
        if place_model_path.exists():
            placement_predictor = PlacementPredictor()
            placement_predictor.load(str(place_model_path))
            if verbose:
                print(f"Loaded placement predictor from {place_model_path}")
    except Exception as e:
        if verbose:
            print(f"Could not load placement predictor: {e}")

    # Generate problems
    generator = SyntheticProblemGenerator(seed=seed)

    problems_per_difficulty = num_problems // len(difficulties)
    all_problems = []

    for diff in difficulties:
        for _ in range(problems_per_difficulty):
            problem = generator.generate_problem(difficulty=diff)
            all_problems.append(problem)

    if verbose:
        print(f"\nGenerated {len(all_problems)} problems")
        print(f"Difficulties: {difficulties}")
        print("\n" + "="*60)
        print("Running benchmark...")
        print("="*60)

    # Run benchmark
    baseline_results = []
    ml_results = []
    reroute_results = []
    reroute_smart_results = []

    for i, problem in enumerate(all_problems):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(all_problems)}")

        # Run baseline
        baseline = solve_baseline(problem)
        baseline_results.append(baseline)

        # Run ML-enhanced
        ml = solve_ml_enhanced(problem, connection_predictor, placement_predictor)
        ml_results.append(ml)

        # Run reroute loop
        reroute = solve_with_reroute(problem, use_smart_ports=False)
        reroute_results.append(reroute)

        # Run reroute with smart ports
        reroute_smart = solve_with_reroute(problem, use_smart_ports=True)
        reroute_smart_results.append(reroute_smart)

        # Memory cleanup
        gc.collect()

    # Calculate statistics
    def calc_stats(results: List[BenchmarkResult]) -> Dict[str, float]:
        if not results:
            return {}

        successes = [r for r in results if r.success]

        return {
            'count': len(results),
            'success_rate': len(successes) / len(results),
            'avg_routing_progress': sum(r.routing_progress for r in results) / len(results),
            'avg_solve_time': sum(r.solve_time for r in results) / len(results),
            'avg_nodes_explored': sum(r.total_nodes_explored for r in results) / len(results),
            'avg_belts': sum(r.num_belts for r in results) / len(results),
            'avg_nodes_per_connection': sum(r.nodes_per_connection for r in results) / len(results),
        }

    # Overall stats
    baseline_stats = calc_stats(baseline_results)
    ml_stats = calc_stats(ml_results)
    reroute_stats = calc_stats(reroute_results)
    reroute_smart_stats = calc_stats(reroute_smart_results)

    # Per-difficulty stats
    difficulty_stats = {}
    for diff in difficulties:
        base_diff = [r for r in baseline_results if r.difficulty == diff]
        ml_diff = [r for r in ml_results if r.difficulty == diff]
        reroute_diff = [r for r in reroute_results if r.difficulty == diff]
        reroute_smart_diff = [r for r in reroute_smart_results if r.difficulty == diff]

        difficulty_stats[diff] = {
            'baseline': calc_stats(base_diff),
            'ml_enhanced': calc_stats(ml_diff),
            'reroute_loop': calc_stats(reroute_diff),
            'reroute_smart': calc_stats(reroute_smart_diff),
        }

    # Calculate improvements
    improvements = {}
    if baseline_stats and ml_stats:
        base_sr = baseline_stats['success_rate']
        ml_sr = ml_stats['success_rate']
        improvements['success_rate_delta'] = ml_sr - base_sr
        improvements['success_rate_pct_improvement'] = (
            (ml_sr - base_sr) / max(0.001, base_sr) * 100
        )

        base_time = baseline_stats['avg_solve_time']
        ml_time = ml_stats['avg_solve_time']
        improvements['time_reduction_pct'] = (
            (base_time - ml_time) / max(0.001, base_time) * 100
        )

        base_nodes = baseline_stats['avg_nodes_explored']
        ml_nodes = ml_stats['avg_nodes_explored']
        improvements['nodes_reduction_pct'] = (
            (base_nodes - ml_nodes) / max(1, base_nodes) * 100
        )

    # Calculate reroute improvements vs baseline
    reroute_improvements = {}
    if baseline_stats and reroute_stats:
        base_sr = baseline_stats['success_rate']
        reroute_sr = reroute_stats['success_rate']
        reroute_improvements['success_rate_delta'] = reroute_sr - base_sr
        reroute_improvements['success_rate_pct_improvement'] = (
            (reroute_sr - base_sr) / max(0.001, base_sr) * 100
        )

    reroute_smart_improvements = {}
    if baseline_stats and reroute_smart_stats:
        base_sr = baseline_stats['success_rate']
        smart_sr = reroute_smart_stats['success_rate']
        reroute_smart_improvements['success_rate_delta'] = smart_sr - base_sr
        reroute_smart_improvements['success_rate_pct_improvement'] = (
            (smart_sr - base_sr) / max(0.001, base_sr) * 100
        )

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_problems': num_problems,
            'difficulties': difficulties,
            'seed': seed,
            'has_connection_predictor': connection_predictor is not None and connection_predictor.is_trained,
            'has_placement_predictor': placement_predictor is not None and placement_predictor.is_trained,
        },
        'baseline': baseline_stats,
        'ml_enhanced': ml_stats,
        'reroute_loop': reroute_stats,
        'reroute_smart': reroute_smart_stats,
        'improvements': improvements,
        'reroute_improvements': reroute_improvements,
        'reroute_smart_improvements': reroute_smart_improvements,
        'by_difficulty': difficulty_stats,
    }

    return results


def print_results(results: Dict[str, Any]):
    """Pretty print benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    config = results['config']
    print(f"\nConfiguration:")
    print(f"  Problems: {config['num_problems']}")
    print(f"  Difficulties: {', '.join(config['difficulties'])}")
    print(f"  ML Connection Predictor: {'Yes' if config['has_connection_predictor'] else 'No (using heuristic)'}")
    print(f"  ML Placement Predictor: {'Yes' if config['has_placement_predictor'] else 'No'}")

    print("\n" + "-"*60)
    print("OVERALL COMPARISON")
    print("-"*60)

    baseline = results['baseline']
    ml = results['ml_enhanced']
    reroute = results.get('reroute_loop', {})
    reroute_smart = results.get('reroute_smart', {})

    print(f"\n{'Metric':<22} {'Baseline':>12} {'ML-Order':>12} {'Reroute':>12} {'Reroute+':>12}")
    print("-"*70)

    if reroute and reroute_smart:
        print(f"{'Success Rate':<22} {baseline['success_rate']:>11.1%} {ml['success_rate']:>11.1%} {reroute['success_rate']:>11.1%} {reroute_smart['success_rate']:>11.1%}")
        print(f"{'Routing Progress':<22} {baseline['avg_routing_progress']:>11.1%} {ml['avg_routing_progress']:>11.1%} {reroute['avg_routing_progress']:>11.1%} {reroute_smart['avg_routing_progress']:>11.1%}")
        print(f"{'Solve Time (s)':<22} {baseline['avg_solve_time']:>12.4f} {ml['avg_solve_time']:>12.4f} {reroute['avg_solve_time']:>12.4f} {reroute_smart['avg_solve_time']:>12.4f}")
        print(f"{'Nodes Explored':<22} {baseline['avg_nodes_explored']:>12.0f} {ml['avg_nodes_explored']:>12.0f} {reroute['avg_nodes_explored']:>12.0f} {reroute_smart['avg_nodes_explored']:>12.0f}")
    else:
        print(f"{'Success Rate':<22} {baseline['success_rate']:>11.1%} {ml['success_rate']:>11.1%}")
        print(f"{'Routing Progress':<22} {baseline['avg_routing_progress']:>11.1%} {ml['avg_routing_progress']:>11.1%}")

    print("\n" + "-"*60)
    print("IMPROVEMENTS vs BASELINE")
    print("-"*60)

    imp = results['improvements']
    print(f"\n  ML-Order:     {imp['success_rate_delta']:+.1%} ({imp['success_rate_pct_improvement']:+.1f}%)")

    if 'reroute_improvements' in results and results['reroute_improvements']:
        rimp = results['reroute_improvements']
        print(f"  Reroute:      {rimp['success_rate_delta']:+.1%} ({rimp['success_rate_pct_improvement']:+.1f}%)")

    if 'reroute_smart_improvements' in results and results['reroute_smart_improvements']:
        simp = results['reroute_smart_improvements']
        print(f"  Reroute+:     {simp['success_rate_delta']:+.1%} ({simp['success_rate_pct_improvement']:+.1f}%)")

    print("\n" + "-"*60)
    print("BY DIFFICULTY")
    print("-"*60)

    for diff, stats in results['by_difficulty'].items():
        base = stats['baseline']
        ml = stats['ml_enhanced']
        reroute = stats.get('reroute_loop', {})
        reroute_smart = stats.get('reroute_smart', {})

        print(f"\n{diff.upper()}:")
        if reroute and reroute_smart:
            print(f"  Success: {base['success_rate']:.1%} -> ML:{ml['success_rate']:.1%} -> Reroute:{reroute['success_rate']:.1%} -> Reroute+:{reroute_smart['success_rate']:.1%}")
        else:
            print(f"  Success Rate: {base['success_rate']:.1%} -> {ml['success_rate']:.1%}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ML-enhanced vs baseline routing")
    parser.add_argument("--num-problems", type=int, default=100,
                       help="Number of problems to test")
    parser.add_argument("--difficulties", nargs="+", default=['easy', 'medium', 'hard'],
                       help="Difficulty levels to test")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")

    args = parser.parse_args()

    results = run_benchmark(
        num_problems=args.num_problems,
        difficulties=args.difficulties,
        model_dir=args.model_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print_results(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
