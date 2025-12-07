"""Evolution progress visualization."""

from typing import List, Optional, Dict, Tuple, Callable
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..evolution.candidate import Candidate
from ..evolution.algorithm import EvolutionaryAlgorithm
from ..shapes.shape import Shape
from .shape_renderer import ShapeRenderer


class EvolutionVisualizer:
    """Visualizes the evolution process in real-time."""

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        fps: int = 30
    ):
        """
        Initialize the visualizer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            fps: Target frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.shape_renderer = ShapeRenderer(cell_size=30)

        self._screen = None
        self._clock = None
        self._font = None
        self._running = False

        # State
        self.current_generation = 0
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.fitness_history: List[Tuple[float, float]] = []
        self.top_candidates: List[Candidate] = []
        self.input_shape: Optional[Shape] = None
        self.target_shape: Optional[Shape] = None

    def initialize(self) -> bool:
        """
        Initialize pygame and create the window.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not PYGAME_AVAILABLE:
            print("Warning: pygame not available, visualization disabled")
            return False

        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Shapez 2 Solver - Evolution Visualizer")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._large_font = pygame.font.SysFont("monospace", 20)
        self._running = True
        return True

    def close(self) -> None:
        """Close the visualizer."""
        self._running = False
        # Don't call pygame.quit() - let the caller handle that
        # Just close the display
        if PYGAME_AVAILABLE and self._screen:
            pygame.display.quit()
            self._screen = None

    def update(
        self,
        generation: int,
        population: List[Candidate],
        input_shape: Optional[Shape] = None,
        target_shape: Optional[Shape] = None
    ) -> bool:
        """
        Update the visualization with new data.

        Args:
            generation: Current generation number
            population: Current population
            input_shape: The input shape being transformed
            target_shape: The target shape to achieve

        Returns:
            True if should continue, False if user closed window
        """
        if not self._running or not PYGAME_AVAILABLE:
            return True

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return False

        # Update state
        self.current_generation = generation
        self.input_shape = input_shape
        self.target_shape = target_shape

        if population:
            sorted_pop = sorted(population, reverse=True)
            self.top_candidates = sorted_pop[:5]
            self.best_fitness = sorted_pop[0].fitness
            self.avg_fitness = sum(c.fitness for c in population) / len(population)
            self.fitness_history.append((self.best_fitness, self.avg_fitness))

        # Render
        self._render()
        self._clock.tick(self.fps)

        return True

    def _render(self) -> None:
        """Render the current state."""
        if not PYGAME_AVAILABLE or not self._screen:
            return

        self._screen.fill((30, 30, 40))

        # Draw header
        self._draw_header()

        # Draw fitness graph
        self._draw_fitness_graph()

        # Draw shapes section
        self._draw_shapes_section()

        # Draw top candidates
        self._draw_top_candidates()

        pygame.display.flip()

    def _draw_header(self) -> None:
        """Draw the header with generation and fitness info."""
        if not self._large_font:
            return

        header_text = f"Generation: {self.current_generation}  |  Best: {self.best_fitness:.4f}  |  Avg: {self.avg_fitness:.4f}"
        text_surface = self._large_font.render(header_text, True, (255, 255, 255))
        self._screen.blit(text_surface, (20, 10))

        # Draw a separator line
        pygame.draw.line(self._screen, (100, 100, 100), (0, 40), (self.width, 40), 2)

    def _draw_fitness_graph(self) -> None:
        """Draw the fitness history graph."""
        graph_rect = pygame.Rect(20, 50, 400, 150)
        pygame.draw.rect(self._screen, (50, 50, 60), graph_rect)
        pygame.draw.rect(self._screen, (100, 100, 100), graph_rect, 2)

        # Draw axes labels
        if self._font:
            label = self._font.render("Fitness History", True, (200, 200, 200))
            self._screen.blit(label, (graph_rect.x + 5, graph_rect.y + 5))

        if not self.fitness_history:
            return

        # Draw graph
        max_points = 200
        history = self.fitness_history[-max_points:]

        if len(history) < 2:
            return

        # Scale to graph area
        graph_left = graph_rect.x + 10
        graph_right = graph_rect.right - 10
        graph_top = graph_rect.y + 25
        graph_bottom = graph_rect.bottom - 10
        graph_height = graph_bottom - graph_top

        x_step = (graph_right - graph_left) / (len(history) - 1)

        # Draw best fitness line (green)
        best_points = []
        for i, (best, avg) in enumerate(history):
            x = graph_left + i * x_step
            y = graph_bottom - best * graph_height
            best_points.append((x, y))

        if len(best_points) >= 2:
            pygame.draw.lines(self._screen, (50, 255, 50), False, best_points, 2)

        # Draw average fitness line (yellow)
        avg_points = []
        for i, (best, avg) in enumerate(history):
            x = graph_left + i * x_step
            y = graph_bottom - avg * graph_height
            avg_points.append((x, y))

        if len(avg_points) >= 2:
            pygame.draw.lines(self._screen, (255, 255, 50), False, avg_points, 2)

    def _draw_shapes_section(self) -> None:
        """Draw the input and target shapes."""
        section_y = 220

        if self._font:
            label = self._font.render("Input Shape:", True, (200, 200, 200))
            self._screen.blit(label, (20, section_y))

            label = self._font.render("Target Shape:", True, (200, 200, 200))
            self._screen.blit(label, (150, section_y))

        if self.input_shape:
            surface = self.shape_renderer.render_to_surface(self.input_shape)
            if surface:
                self._screen.blit(surface, (20, section_y + 20))

        if self.target_shape:
            surface = self.shape_renderer.render_to_surface(self.target_shape)
            if surface:
                self._screen.blit(surface, (150, section_y + 20))

    def _draw_top_candidates(self) -> None:
        """Draw the top candidate solutions."""
        section_x = 450
        section_y = 50

        if self._font:
            label = self._font.render("Top Candidates:", True, (200, 200, 200))
            self._screen.blit(label, (section_x, section_y))

        y_offset = section_y + 25
        for i, candidate in enumerate(self.top_candidates):
            if self._font:
                text = f"#{i+1}: Fitness={candidate.fitness:.4f}, Ops={len(candidate.design.operations)}"
                text_surface = self._font.render(text, True, (180, 180, 180))
                self._screen.blit(text_surface, (section_x, y_offset))
            y_offset += 20


def run_with_visualization(
    algorithm: EvolutionaryAlgorithm,
    input_shape: Optional[Shape] = None,
    target_shape: Optional[Shape] = None
) -> Optional[Candidate]:
    """
    Run the evolutionary algorithm with real-time visualization.

    Args:
        algorithm: The evolutionary algorithm to run
        input_shape: Optional input shape for display
        target_shape: Optional target shape for display

    Returns:
        The best candidate found
    """
    visualizer = EvolutionVisualizer()

    if not visualizer.initialize():
        # Fall back to non-visual mode
        print("Visualization not available, running in console mode...")
        return algorithm.run(lambda gen, fit: print(f"Gen {gen}: Best fitness = {fit:.4f}") or True)

    def on_generation(generation: int, population: List[Candidate]) -> None:
        visualizer.update(generation, population, input_shape, target_shape)

    algorithm.on_generation = on_generation

    try:
        result = algorithm.run(
            callback=lambda gen, fit: visualizer._running
        )
    finally:
        # Keep window open briefly to show final result
        if PYGAME_AVAILABLE:
            time.sleep(1)
        visualizer.close()

    return result
