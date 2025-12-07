"""Main application UI for the Shapez 2 Solver."""

from typing import Dict, List, Optional, Type
import sys

try:
    import pygame
    import pygame_gui
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..shapes.shape import Shape, Color
from ..shapes.parser import ShapeCodeParser
from ..operations.base import Operation
from ..operations.cutter import CutOperation, HalfDestroyerOperation, SwapperOperation
from ..operations.rotator import RotateOperation
from ..operations.stacker import StackOperation, UnstackOperation
from ..operations.painter import PaintOperation
from ..foundations.foundation import Foundation, FoundationType
from ..foundations.layouts import get_foundation, list_available_foundations
from ..evolution.algorithm import EvolutionaryAlgorithm, EvolutionConfig
from ..visualization.evolution_visualizer import EvolutionVisualizer
from ..visualization.shape_renderer import ShapeRenderer


# Available operations with display names
OPERATION_OPTIONS: Dict[str, Type[Operation]] = {
    "Cutter": CutOperation,
    "Half Destroyer": HalfDestroyerOperation,
    "Swapper": SwapperOperation,
    "Rotator": RotateOperation,
    "Stacker": StackOperation,
    "Unstacker": UnstackOperation,
    "Painter": PaintOperation,
}


class SolverApp:
    """Main application for the Shapez 2 Solver."""

    def __init__(self, width: int = 1200, height: int = 800):
        """
        Initialize the application.

        Args:
            width: Window width
            height: Window height
        """
        self.width = width
        self.height = height
        self.running = False

        # UI state
        self.selected_foundation: FoundationType = FoundationType.SIZE_1X1
        self.selected_operations: List[Type[Operation]] = list(OPERATION_OPTIONS.values())
        self.population_size: int = 50
        self.generations: int = 100
        self.input_shape_code: str = "CuCuCuCu"
        self.target_shape_code: str = "CrCrCrCr"

        # Pygame elements
        self._screen = None
        self._manager = None
        self._clock = None
        self._font = None

        # UI elements
        self._ui_elements: Dict = {}

        # Shape renderer
        self.shape_renderer = ShapeRenderer(cell_size=35)

        # Evolution state
        self.evolution_running = False
        self.current_algorithm: Optional[EvolutionaryAlgorithm] = None

    def run(self) -> None:
        """Run the application main loop."""
        if not PYGAME_AVAILABLE:
            print("Error: pygame and pygame_gui are required for the UI")
            print("Install with: pip install pygame pygame_gui")
            self._run_console_mode()
            return

        self._initialize()

        while self.running:
            time_delta = self._clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    self._handle_button(event)

                if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                    self._handle_text_entry(event)

                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    self._handle_dropdown(event)

                self._manager.process_events(event)

            self._manager.update(time_delta)
            self._render()

        pygame.quit()

    def _initialize(self) -> None:
        """Initialize pygame and create UI elements."""
        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Shapez 2 Solver")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)

        self._manager = pygame_gui.UIManager((self.width, self.height))

        self._create_ui()
        self.running = True

    def _create_ui(self) -> None:
        """Create the UI elements."""
        # Left panel - Configuration
        panel_width = 300
        y = 20

        # Title
        self._ui_elements['title'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            text="Shapez 2 Solver",
            manager=self._manager
        )
        y += 50

        # Foundation selection
        self._ui_elements['foundation_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Foundation Type:",
            manager=self._manager
        )
        y += 30

        foundation_options = [f.value for f in list_available_foundations()]
        self._ui_elements['foundation_dropdown'] = pygame_gui.elements.UIDropDownMenu(
            options_list=foundation_options,
            starting_option=self.selected_foundation.value,
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        y += 50

        # Population size
        self._ui_elements['pop_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Population Size:",
            manager=self._manager
        )
        y += 30

        self._ui_elements['pop_input'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        self._ui_elements['pop_input'].set_text(str(self.population_size))
        y += 50

        # Generations
        self._ui_elements['gen_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Generations:",
            manager=self._manager
        )
        y += 30

        self._ui_elements['gen_input'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        self._ui_elements['gen_input'].set_text(str(self.generations))
        y += 50

        # Input shape
        self._ui_elements['input_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Input Shape Code:",
            manager=self._manager
        )
        y += 30

        self._ui_elements['input_shape'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        self._ui_elements['input_shape'].set_text(self.input_shape_code)
        y += 50

        # Target shape
        self._ui_elements['target_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Target Shape Code:",
            manager=self._manager
        )
        y += 30

        self._ui_elements['target_shape'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        self._ui_elements['target_shape'].set_text(self.target_shape_code)
        y += 50

        # Start button
        self._ui_elements['start_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 40)),
            text="Start Evolution",
            manager=self._manager
        )
        y += 60

        # Stop button
        self._ui_elements['stop_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 40)),
            text="Stop",
            manager=self._manager
        )

    def _handle_button(self, event) -> None:
        """Handle button press events."""
        if event.ui_element == self._ui_elements.get('start_button'):
            self._start_evolution()
        elif event.ui_element == self._ui_elements.get('stop_button'):
            self._stop_evolution()

    def _handle_text_entry(self, event) -> None:
        """Handle text entry completion."""
        if event.ui_element == self._ui_elements.get('pop_input'):
            try:
                self.population_size = int(event.text)
            except ValueError:
                pass
        elif event.ui_element == self._ui_elements.get('gen_input'):
            try:
                self.generations = int(event.text)
            except ValueError:
                pass
        elif event.ui_element == self._ui_elements.get('input_shape'):
            self.input_shape_code = event.text
        elif event.ui_element == self._ui_elements.get('target_shape'):
            self.target_shape_code = event.text

    def _handle_dropdown(self, event) -> None:
        """Handle dropdown selection changes."""
        if event.ui_element == self._ui_elements.get('foundation_dropdown'):
            for ft in FoundationType:
                if ft.value == event.text:
                    self.selected_foundation = ft
                    break

    def _start_evolution(self) -> None:
        """Start the evolutionary algorithm."""
        try:
            input_shape = ShapeCodeParser.parse(self.input_shape_code)
            target_shape = ShapeCodeParser.parse(self.target_shape_code)
        except ValueError as e:
            print(f"Error parsing shape codes: {e}")
            return

        foundation = get_foundation(self.selected_foundation)

        config = EvolutionConfig(
            population_size=self.population_size,
            generations=self.generations,
            allowed_operations=self.selected_operations,
        )

        algorithm = EvolutionaryAlgorithm(
            foundation=foundation,
            input_shapes={"input": input_shape},
            expected_outputs={"output": target_shape},
            config=config,
        )

        self.evolution_running = True
        self.current_algorithm = algorithm

        # Run with visualization
        from ..visualization.evolution_visualizer import run_with_visualization
        result = run_with_visualization(algorithm, input_shape, target_shape)

        self.evolution_running = False

        if result:
            print(f"Best solution found: Fitness={result.fitness:.4f}")
            print(f"Operations: {len(result.design.operations)}")

    def _stop_evolution(self) -> None:
        """Stop the current evolution run."""
        self.evolution_running = False

    def _render(self) -> None:
        """Render the application."""
        self._screen.fill((40, 40, 50))

        # Draw right panel with shape previews
        self._draw_shape_preview()

        self._manager.draw_ui(self._screen)
        pygame.display.flip()

    def _draw_shape_preview(self) -> None:
        """Draw shape previews in the right panel."""
        preview_x = 350
        preview_y = 50

        if self._font:
            label = self._font.render("Input Shape Preview:", True, (200, 200, 200))
            self._screen.blit(label, (preview_x, preview_y))

        try:
            input_shape = ShapeCodeParser.parse(self.input_shape_code)
            surface = self.shape_renderer.render_to_surface(input_shape)
            if surface:
                self._screen.blit(surface, (preview_x, preview_y + 25))
        except (ValueError, Exception):
            pass

        preview_y += 150

        if self._font:
            label = self._font.render("Target Shape Preview:", True, (200, 200, 200))
            self._screen.blit(label, (preview_x, preview_y))

        try:
            target_shape = ShapeCodeParser.parse(self.target_shape_code)
            surface = self.shape_renderer.render_to_surface(target_shape)
            if surface:
                self._screen.blit(surface, (preview_x, preview_y + 25))
        except (ValueError, Exception):
            pass

    def _run_console_mode(self) -> None:
        """Run the solver in console mode (no pygame)."""
        print("=" * 50)
        print("Shapez 2 Solver - Console Mode")
        print("=" * 50)

        print(f"\nInput shape: {self.input_shape_code}")
        print(f"Target shape: {self.target_shape_code}")
        print(f"Population: {self.population_size}")
        print(f"Generations: {self.generations}")

        try:
            input_shape = ShapeCodeParser.parse(self.input_shape_code)
            target_shape = ShapeCodeParser.parse(self.target_shape_code)
        except ValueError as e:
            print(f"Error parsing shape codes: {e}")
            return

        foundation = get_foundation(self.selected_foundation)

        config = EvolutionConfig(
            population_size=self.population_size,
            generations=self.generations,
            allowed_operations=self.selected_operations,
            parallel_evaluation=True,
        )

        algorithm = EvolutionaryAlgorithm(
            foundation=foundation,
            input_shapes={"input": input_shape},
            expected_outputs={"output": target_shape},
            config=config,
        )

        def progress_callback(gen: int, fitness: float) -> bool:
            print(f"Generation {gen}: Best fitness = {fitness:.4f}")
            return True

        print("\nStarting evolution...")
        result = algorithm.run(callback=progress_callback)

        if result:
            print(f"\n{'=' * 50}")
            print("RESULT")
            print(f"{'=' * 50}")
            print(f"Best fitness: {result.fitness:.4f}")
            print(f"Operations: {len(result.design.operations)}")
            print(f"Connections: {len(result.design.connections)}")
        else:
            print("\nNo solution found.")


def main():
    """Entry point for the application."""
    app = SolverApp()
    app.run()


if __name__ == "__main__":
    main()
