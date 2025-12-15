"""
View training samples from the database in the pygame viewer.

Loads training samples from the SQLite database and displays them
in the pygame layout viewer for inspection.

Supports two database formats:
1. training_samples.db - New format from model_comparison.py with checkpointing
2. synthetic_training.db - Older format with synthetic problems and solver results

Usage:
    python -m shapez2_solver.visualization.view_training_samples [--db PATH] [--filter routed|failed|all]
"""

import argparse
import json
import sqlite3
from typing import List, Optional, Dict, Any

from ..evolution.model_comparison import TrainingSampleDB
from ..evolution.foundation_config import FoundationConfig, FOUNDATION_SPECS
from ..evolution.foundation_evolution import (
    FoundationEvolution, Candidate, PlacedBuilding
)
from ..evolution.evaluation import PlacementInfo
from ..blueprint.building_types import BuildingType, Rotation


# Mapping from string building types to BuildingType enum
BUILDING_TYPE_MAP = {
    'ROTATOR_CW': BuildingType.ROTATOR_CW,
    'ROTATOR_CCW': BuildingType.ROTATOR_CCW,
    'ROTATOR_180': BuildingType.ROTATOR_180,
    'CUTTER': BuildingType.CUTTER,
    'CUTTER_MIRRORED': BuildingType.CUTTER_MIRRORED,
    'HALF_CUTTER': BuildingType.HALF_CUTTER,
    'SWAPPER': BuildingType.SWAPPER,
    'STACKER': BuildingType.STACKER,
    'STACKER_BENT': BuildingType.STACKER_BENT,
    'UNSTACKER': BuildingType.UNSTACKER,
    'PIN_PUSHER': BuildingType.PIN_PUSHER,
    'TRASH': BuildingType.TRASH,
    'PAINTER': BuildingType.PAINTER,
    'BELT_FORWARD': BuildingType.BELT_FORWARD,
    'BELT_LEFT': BuildingType.BELT_LEFT,
    'BELT_RIGHT': BuildingType.BELT_RIGHT,
    'SPLITTER': BuildingType.SPLITTER,
    'MERGER': BuildingType.MERGER,
    'LIFT_UP': BuildingType.LIFT_UP,
    'LIFT_DOWN': BuildingType.LIFT_DOWN,
}

ROTATION_MAP = {
    'EAST': Rotation.EAST,
    'SOUTH': Rotation.SOUTH,
    'WEST': Rotation.WEST,
    'NORTH': Rotation.NORTH,
}


class TrainingSampleViewer:
    """Viewer for training samples from the database."""

    def __init__(self, db_path: str = "training_samples.db"):
        self.db_path = db_path
        self.samples: List[dict] = []
        self.current_index = 0
        self._db_format: str = "unknown"

    def _detect_db_format(self) -> str:
        """Detect the database format."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        if 'synthetic_problems' in tables and 'solver_results' in tables:
            return "synthetic"
        elif 'training_samples' in tables:
            # Check if it's the new format with machines column
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(training_samples)")
            cols = {row[1] for row in cursor.fetchall()}
            conn.close()
            if 'machines' in cols:
                return "checkpointed"
            return "ml_features"
        return "unknown"

    def _load_synthetic_samples(self, filter_mode: str, limit: Optional[int]) -> List[dict]:
        """Load samples from synthetic_training.db format."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Join problems with results to get both machine placements and routing success
        query = '''
            SELECT p.problem_id, p.foundation_type, p.grid_width, p.grid_height,
                   p.num_floors, p.problem_json, r.success, r.num_belts
            FROM synthetic_problems p
            JOIN solver_results r ON p.problem_id = r.problem_id
        '''

        if filter_mode == "routed":
            query += " WHERE r.success = 1"
        elif filter_mode == "failed":
            query += " WHERE r.success = 0"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        samples = []
        for row in rows:
            problem_id, foundation, grid_w, grid_h, num_floors, problem_json, success, num_belts = row
            problem = json.loads(problem_json)

            # Convert machines from problem JSON
            machines = []
            for m in problem.get('machines', []):
                bt = BUILDING_TYPE_MAP.get(m.get('type', ''), BuildingType.ROTATOR_CW)
                rot = ROTATION_MAP.get(m.get('rotation', 'EAST'), Rotation.EAST)
                machines.append(PlacementInfo(
                    building_type=bt,
                    x=m.get('x', 0),
                    y=m.get('y', 0),
                    floor=m.get('floor', 0),
                    rotation=rot,
                ))

            samples.append({
                'problem_name': problem_id,
                'foundation': foundation,
                'grid_width': grid_w,
                'grid_height': grid_h,
                'num_floors': num_floors,
                'machines': machines,
                'routing_success': bool(success),
                'num_belts': num_belts,
                'input_positions': [],  # Not stored in this format
                'output_positions': [],
            })

        return samples

    def load_samples(
        self,
        filter_mode: str = "all",
        limit: Optional[int] = None
    ) -> int:
        """Load samples from database.

        Args:
            filter_mode: "all", "routed", or "failed"
            limit: Max samples to load

        Returns:
            Number of samples loaded
        """
        self._db_format = self._detect_db_format()

        if self._db_format == "synthetic":
            self.samples = self._load_synthetic_samples(filter_mode, limit)
        elif self._db_format == "checkpointed":
            # Use TrainingSampleDB for new format
            db = TrainingSampleDB(self.db_path)
            train = db.get_samples(is_test=False)
            test = db.get_samples(is_test=True)
            all_samples = train + test

            # Filter based on mode
            if filter_mode == "routed":
                all_samples = [s for s in all_samples if s['routing_success']]
            elif filter_mode == "failed":
                all_samples = [s for s in all_samples if not s['routing_success']]

            # Apply limit
            if limit:
                all_samples = all_samples[:limit]

            self.samples = all_samples
        else:
            print(f"Unknown database format: {self._db_format}")
            self.samples = []

        return len(self.samples)

    def _convert_to_candidate(self, sample: dict) -> Candidate:
        """Convert a training sample to a Candidate for the viewer."""
        candidate = Candidate()
        candidate.routing_success = sample['routing_success']
        candidate.fitness = 1.0 if sample['routing_success'] else 0.0

        # Convert PlacementInfo machines to PlacedBuilding
        for i, machine in enumerate(sample['machines']):
            if isinstance(machine, PlacementInfo):
                building = PlacedBuilding(
                    building_id=i,
                    building_type=machine.building_type,
                    x=machine.x,
                    y=machine.y,
                    floor=machine.floor,
                    rotation=machine.rotation,
                )
            else:
                # Handle dict format (older samples)
                building = PlacedBuilding(
                    building_id=i,
                    building_type=machine.get('building_type', BuildingType.BELT_FORWARD),
                    x=machine.get('x', 0),
                    y=machine.get('y', 0),
                    floor=machine.get('floor', 0),
                    rotation=machine.get('rotation', Rotation.EAST),
                )
            candidate.buildings.append(building)

        return candidate

    def _create_evolution_for_sample(self, sample: dict) -> FoundationEvolution:
        """Create a FoundationEvolution object for a sample."""
        # Get foundation spec
        foundation_name = sample.get('foundation', '2x2')
        if foundation_name not in FOUNDATION_SPECS:
            foundation_name = '2x2'  # Default fallback

        # Create minimal config (we don't have full port info but that's ok for viewing)
        config = FoundationConfig(foundation_name)

        # Create evolution object
        evolution = FoundationEvolution(
            config=config,
            population_size=1,
            max_buildings=100,
            num_top_solutions=1,
        )

        # Convert sample to candidate
        candidate = self._convert_to_candidate(sample)
        evolution.top_solutions = [candidate]

        return evolution

    def show(self, start_index: int = 0):
        """Show the pygame viewer with loaded samples."""
        try:
            from .pygame_layout_viewer import PygameLayoutViewer
            import pygame
        except ImportError:
            print("Error: pygame is required. Install with: pip install pygame-ce")
            return

        if not self.samples:
            print("No samples loaded. Call load_samples() first.")
            return

        self.current_index = start_index

        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((1280, 800))
        pygame.display.set_caption("Training Sample Viewer")
        font = pygame.font.SysFont("monospace", 14)
        clock = pygame.time.Clock()

        running = True
        while running and self.current_index < len(self.samples):
            sample = self.samples[self.current_index]
            evolution = self._create_evolution_for_sample(sample)

            # Create viewer for this sample
            viewer = PygameLayoutViewer(evolution, 1280, 800)
            viewer.screen = screen
            viewer.font = font
            viewer.small_font = pygame.font.SysFont("monospace", 10)
            viewer.tiny_font = pygame.font.SysFont("monospace", 9)
            viewer.running = True
            viewer._update_blueprint()

            # Custom rendering loop to add sample navigation
            while viewer.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        viewer.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            viewer.running = False
                        elif event.key == pygame.K_n or event.key == pygame.K_RIGHT:
                            # Next sample
                            if self.current_index < len(self.samples) - 1:
                                self.current_index += 1
                                viewer.running = False
                        elif event.key == pygame.K_p or event.key == pygame.K_LEFT:
                            # Previous sample
                            if self.current_index > 0:
                                self.current_index -= 1
                                viewer.running = False
                        elif event.key == pygame.K_HOME:
                            # First sample
                            self.current_index = 0
                            viewer.running = False
                        elif event.key == pygame.K_END:
                            # Last sample
                            self.current_index = len(self.samples) - 1
                            viewer.running = False
                        else:
                            viewer._handle_event(event)
                    else:
                        viewer._handle_event(event)

                # Render
                viewer._render()

                # Draw sample navigation overlay
                self._draw_navigation(screen, font, sample)

                pygame.display.flip()
                clock.tick(30)

        pygame.quit()

    def _draw_navigation(self, screen, font, sample):
        """Draw navigation overlay."""
        import pygame

        # Top bar with sample info
        pygame.draw.rect(screen, (40, 40, 50), (0, 0, 1280, 50))

        # Sample counter
        text = f"Sample {self.current_index + 1}/{len(self.samples)}"
        surf = font.render(text, True, (255, 255, 255))
        screen.blit(surf, (120, 10))

        # Foundation info
        foundation = sample.get('foundation', 'unknown')
        text = f"Foundation: {foundation}"
        surf = font.render(text, True, (200, 200, 200))
        screen.blit(surf, (350, 10))

        # Routing status
        if sample['routing_success']:
            text = "ROUTED"
            color = (100, 255, 100)
        else:
            text = "FAILED"
            color = (255, 100, 100)
        surf = font.render(text, True, color)
        screen.blit(surf, (550, 10))

        # Machine count
        text = f"Machines: {len(sample['machines'])}"
        surf = font.render(text, True, (200, 200, 200))
        screen.blit(surf, (700, 10))

        # Navigation help
        help_text = "← Prev | → Next | Home/End | Esc=Quit"
        surf = font.render(help_text, True, (150, 150, 150))
        screen.blit(surf, (350, 30))


def main():
    import os

    parser = argparse.ArgumentParser(
        description="View training samples in pygame viewer"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to training samples database (auto-detects if not specified)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["all", "routed", "failed"],
        default="all",
        help="Filter samples by routing success"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to load"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting sample index"
    )
    args = parser.parse_args()

    # Auto-detect database if not specified
    db_path = args.db
    if db_path is None:
        # Try common database files in order of preference
        candidates = [
            "training_samples.db",
            "synthetic_training.db",
            "comparison_training.db",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                db_path = candidate
                break

        if db_path is None:
            print("No training database found. Generate training data first with:")
            print("  python -m shapez2_solver.evolution.model_comparison --problems 100")
            print("\nOr specify a database path with --db")
            return

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    viewer = TrainingSampleViewer(db_path)
    count = viewer.load_samples(filter_mode=args.filter, limit=args.limit)

    print(f"Loaded {count} samples from {db_path}")
    print(f"Database format: {viewer._db_format}")
    if count == 0:
        print("No samples found matching filter.")
        return

    # Count routed vs failed
    routed = sum(1 for s in viewer.samples if s['routing_success'])
    failed = count - routed
    print(f"Routing: {routed} success, {failed} failed")

    print(f"Filter: {args.filter}")
    print("\nControls:")
    print("  ← / P: Previous sample")
    print("  → / N: Next sample")
    print("  Home:  First sample")
    print("  End:   Last sample")
    print("  Tab:   Cycle solutions (if multiple)")
    print("  PgUp/PgDn: Change floor")
    print("  +/-/Scroll: Zoom")
    print("  WASD/Arrows: Pan")
    print("  C:     Copy blueprint")
    print("  Esc:   Quit")
    print()

    viewer.show(start_index=args.start)


if __name__ == "__main__":
    main()
