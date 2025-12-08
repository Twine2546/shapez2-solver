"""Main application UI for the Shapez 2 Solver."""

from typing import Dict, List, Optional, Type
from enum import Enum
import sys

PYGAME_AVAILABLE = False
_PYGAME_ERROR = ""

try:
    import pygame
    import pygame_gui
    PYGAME_AVAILABLE = True
except ImportError as e:
    _PYGAME_ERROR = str(e)
except Exception as e:
    _PYGAME_ERROR = str(e)

from ..shapes.shape import Shape, Color
from ..shapes.parser import ShapeCodeParser
from ..shapes.encoder import ShapeCodeEncoder
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

# Foundation evolution imports
from ..evolution.foundation_config import FOUNDATION_SPECS, Side
from ..evolution.foundation_evolution import create_evolution_from_spec


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


class AppMode(Enum):
    SHAPE_TRANSFORM = "shape"
    FOUNDATION = "foundation"


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

        # App mode
        self.mode = AppMode.SHAPE_TRANSFORM

        # UI state - Shape Transform mode
        self.selected_foundation: FoundationType = FoundationType.SIZE_1X1
        self.selected_operations: List[Type[Operation]] = list(OPERATION_OPTIONS.values())
        self.population_size: int = 50
        self.generations: int = 100
        self.input_shape_code: str = "CrRgSbWy"
        self.target_shape_code: str = "WyCrRgSb"  # 90 degree rotation
        self.enable_painting: bool = False

        # UI state - Foundation mode
        self.foundation_type: str = "2x2"
        self.foundation_inputs: List[str] = ["W,0,0,CuCuCuCu"]
        self.foundation_outputs: List[str] = ["E,0,0,Cu------", "E,0,1,--Cu----", "E,1,0,----Cu--", "E,1,1,------Cu"]
        self.foundation_population: int = 50
        self.foundation_generations: int = 100
        self.foundation_max_buildings: int = 20
        self.foundation_algorithm: str = "Evolutionary"  # "Evolutionary", "Simulated Annealing", "Hybrid"

        # Evolution results for foundation mode
        self.foundation_evolution = None
        self.foundation_solutions = None

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
            print("Error: pygame_gui is required for the GUI")
            if _PYGAME_ERROR:
                print(f"Import error: {_PYGAME_ERROR}")
            print("\nTo fix, try:")
            print("  pip uninstall pygame pygame-ce pygame_gui")
            print("  pip install pygame-ce pygame_gui")
            print("\nFalling back to console mode...")
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

    def _clear_ui(self) -> None:
        """Clear all UI elements."""
        for element in self._ui_elements.values():
            element.kill()
        self._ui_elements.clear()

    def _create_ui(self) -> None:
        """Create the UI elements based on current mode."""
        self._clear_ui()

        # Left panel - Configuration
        panel_width = 350
        y = 20

        # Title
        self._ui_elements['title'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            text="Shapez 2 Solver",
            manager=self._manager
        )
        y += 40

        # Mode selector
        self._ui_elements['mode_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Mode:",
            manager=self._manager
        )
        y += 25

        mode_options = ["Shape Transform", "Foundation Evolution"]
        current_mode = "Shape Transform" if self.mode == AppMode.SHAPE_TRANSFORM else "Foundation Evolution"
        self._ui_elements['mode_dropdown'] = pygame_gui.elements.UIDropDownMenu(
            options_list=mode_options,
            starting_option=current_mode,
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        y += 50

        # Create mode-specific UI
        if self.mode == AppMode.SHAPE_TRANSFORM:
            self._create_shape_transform_ui(y, panel_width)
        else:
            self._create_foundation_ui(y, panel_width)

    def _create_shape_transform_ui(self, y: int, panel_width: int) -> None:
        """Create UI elements for shape transform mode."""
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

        # Painting checkbox (using button as toggle)
        self._ui_elements['painting_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            text="Painting: OFF",
            manager=self._manager
        )
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

    def _create_foundation_ui(self, y: int, panel_width: int) -> None:
        """Create UI elements for foundation evolution mode."""
        # Foundation type selection
        self._ui_elements['fnd_type_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Foundation Type:",
            manager=self._manager
        )
        y += 25

        foundation_options = list(FOUNDATION_SPECS.keys())
        self._ui_elements['fnd_type_dropdown'] = pygame_gui.elements.UIDropDownMenu(
            options_list=foundation_options,
            starting_option=self.foundation_type,
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        y += 40

        # Foundation info
        spec = FOUNDATION_SPECS.get(self.foundation_type)
        if spec:
            info_text = f"Grid: {spec.grid_width}x{spec.grid_height} | Ports: N={spec.ports_per_side[Side.NORTH]}"
            self._ui_elements['fnd_info'] = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((20, y), (panel_width - 40, 20)),
                text=info_text,
                manager=self._manager
            )
            y += 30

        # Inputs
        self._ui_elements['inputs_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Inputs (SIDE,POS,FLOOR,SHAPE):",
            manager=self._manager
        )
        y += 25

        self._ui_elements['inputs_text'] = pygame_gui.elements.UITextEntryBox(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 60)),
            manager=self._manager
        )
        self._ui_elements['inputs_text'].set_text("\n".join(self.foundation_inputs))
        y += 70

        # Outputs
        self._ui_elements['outputs_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Outputs (SIDE,POS,FLOOR,SHAPE):",
            manager=self._manager
        )
        y += 25

        self._ui_elements['outputs_text'] = pygame_gui.elements.UITextEntryBox(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 100)),
            manager=self._manager
        )
        self._ui_elements['outputs_text'].set_text("\n".join(self.foundation_outputs))
        y += 110

        # Evolution parameters
        self._ui_elements['fnd_params_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Evolution Parameters:",
            manager=self._manager
        )
        y += 30

        # Population
        self._ui_elements['fnd_pop_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (100, 25)),
            text="Population:",
            manager=self._manager
        )
        self._ui_elements['fnd_pop_input'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((130, y), (80, 25)),
            manager=self._manager
        )
        self._ui_elements['fnd_pop_input'].set_text(str(self.foundation_population))
        y += 35

        # Generations
        self._ui_elements['fnd_gen_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (100, 25)),
            text="Generations:",
            manager=self._manager
        )
        self._ui_elements['fnd_gen_input'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((130, y), (80, 25)),
            manager=self._manager
        )
        self._ui_elements['fnd_gen_input'].set_text(str(self.foundation_generations))
        y += 35

        # Max buildings
        self._ui_elements['fnd_bld_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (100, 25)),
            text="Max Buildings:",
            manager=self._manager
        )
        self._ui_elements['fnd_bld_input'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((130, y), (80, 25)),
            manager=self._manager
        )
        self._ui_elements['fnd_bld_input'].set_text(str(self.foundation_max_buildings))
        y += 40

        # Algorithm selector
        self._ui_elements['algorithm_label'] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 25)),
            text="Algorithm:",
            manager=self._manager
        )
        y += 25

        algorithm_options = ["Evolutionary", "Simulated Annealing", "Hybrid", "Two-Phase", "Two-Phase SA", "Two-Phase Hybrid"]
        self._ui_elements['algorithm_dropdown'] = pygame_gui.elements.UIDropDownMenu(
            options_list=algorithm_options,
            starting_option=self.foundation_algorithm,
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 30)),
            manager=self._manager
        )
        y += 45

        # Start button
        self._ui_elements['start_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 40)),
            text="Start Evolution",
            manager=self._manager
        )
        y += 50

        # View Layout button
        self._ui_elements['view_layout_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, y), (panel_width - 40, 40)),
            text="View Layout",
            manager=self._manager
        )

    def _handle_button(self, event) -> None:
        """Handle button press events."""
        if event.ui_element == self._ui_elements.get('start_button'):
            if self.mode == AppMode.SHAPE_TRANSFORM:
                self._start_evolution()
            else:
                self._start_foundation_evolution()
        elif event.ui_element == self._ui_elements.get('stop_button'):
            self._stop_evolution()
        elif event.ui_element == self._ui_elements.get('painting_button'):
            self.enable_painting = not self.enable_painting
            self._ui_elements['painting_button'].set_text(
                f"Painting: {'ON' if self.enable_painting else 'OFF'}"
            )
        elif event.ui_element == self._ui_elements.get('view_layout_button'):
            self._view_foundation_layout()

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
        # Foundation mode text entries
        elif event.ui_element == self._ui_elements.get('fnd_pop_input'):
            try:
                self.foundation_population = int(event.text)
            except ValueError:
                pass
        elif event.ui_element == self._ui_elements.get('fnd_gen_input'):
            try:
                self.foundation_generations = int(event.text)
            except ValueError:
                pass
        elif event.ui_element == self._ui_elements.get('fnd_bld_input'):
            try:
                self.foundation_max_buildings = int(event.text)
            except ValueError:
                pass

    def _handle_dropdown(self, event) -> None:
        """Handle dropdown selection changes."""
        if event.ui_element == self._ui_elements.get('foundation_dropdown'):
            for ft in FoundationType:
                if ft.value == event.text:
                    self.selected_foundation = ft
                    break
        elif event.ui_element == self._ui_elements.get('mode_dropdown'):
            if event.text == "Shape Transform":
                self.mode = AppMode.SHAPE_TRANSFORM
            else:
                self.mode = AppMode.FOUNDATION
            self._create_ui()
        elif event.ui_element == self._ui_elements.get('fnd_type_dropdown'):
            self.foundation_type = event.text
            self._create_ui()  # Rebuild to update info
        elif event.ui_element == self._ui_elements.get('algorithm_dropdown'):
            self.foundation_algorithm = event.text

    def _start_evolution(self) -> None:
        """Start the evolutionary algorithm for shape transformation."""
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
            enable_painting=self.enable_painting,
        )

        algorithm = EvolutionaryAlgorithm(
            foundation=foundation,
            input_shapes={"input": input_shape},
            expected_outputs={"output": target_shape},
            config=config,
        )

        self.evolution_running = True
        self.current_algorithm = algorithm

        # Run with visualization (this will close pygame when done)
        from ..visualization.evolution_visualizer import run_with_visualization
        result = run_with_visualization(algorithm, input_shape, target_shape)

        self.evolution_running = False

        if result:
            print(f"Best solution found: Fitness={result.fitness:.4f}")
            print(f"Operations: {len(result.design.operations)}")

        # Reinitialize the main UI after visualization closes
        self._initialize()

    def _start_foundation_evolution(self) -> None:
        """Start the foundation evolution with the selected algorithm."""
        # Parse inputs and outputs from text boxes
        inputs_text = self._ui_elements.get('inputs_text')
        outputs_text = self._ui_elements.get('outputs_text')

        if inputs_text:
            self.foundation_inputs = [line.strip() for line in inputs_text.get_text().split('\n') if line.strip()]
        if outputs_text:
            self.foundation_outputs = [line.strip() for line in outputs_text.get_text().split('\n') if line.strip()]

        if not self.foundation_inputs:
            print("Error: At least one input is required")
            return
        if not self.foundation_outputs:
            print("Error: At least one output is required")
            return

        # Parse port specs
        def parse_port_spec(spec: str):
            parts = spec.split(',', 3)
            if len(parts) != 4:
                raise ValueError(f"Invalid port spec: {spec}")
            return (parts[0].strip().upper(), int(parts[1].strip()),
                    int(parts[2].strip()), parts[3].strip())

        try:
            inputs = [parse_port_spec(s) for s in self.foundation_inputs]
            outputs = [parse_port_spec(s) for s in self.foundation_outputs]
        except ValueError as e:
            print(f"Error parsing port specs: {e}")
            return

        # Create evolution
        print(f"\nStarting foundation evolution on {self.foundation_type}...")
        print(f"Algorithm: {self.foundation_algorithm}")
        print(f"Inputs: {len(inputs)}, Outputs: {len(outputs)}")
        print(f"Population: {self.foundation_population}, Generations: {self.foundation_generations}")

        try:
            evolution = create_evolution_from_spec(
                self.foundation_type,
                inputs,
                outputs,
                population_size=self.foundation_population,
                max_buildings=self.foundation_max_buildings
            )

            evolution.print_goal()

            # Run evolution with the selected algorithm
            if self.foundation_algorithm == "Simulated Annealing":
                self.foundation_solutions = self._run_simulated_annealing(evolution)
            elif self.foundation_algorithm == "Hybrid":
                self.foundation_solutions = self._run_hybrid_algorithm(evolution)
            elif self.foundation_algorithm.startswith("Two-Phase"):
                self.foundation_solutions = self._run_two_phase(inputs, outputs)
            else:  # Default: Evolutionary
                self.foundation_solutions = evolution.run(self.foundation_generations, verbose=True)

            # Ensure top_solutions is populated for the viewer
            if self.foundation_solutions:
                evolution.top_solutions = self.foundation_solutions[:5]

            self.foundation_evolution = evolution

            print("\n" + "=" * 60)
            print("EVOLUTION COMPLETE")
            print("=" * 60)

            if self.foundation_solutions:
                for i, sol in enumerate(self.foundation_solutions[:3]):
                    print(f"Solution {i+1}: Fitness={sol.fitness:.2f}, Buildings={len(sol.buildings)}")

                print("\nClick 'View Layout' to see the solutions.")

        except Exception as e:
            import traceback
            print(f"Error during evolution: {e}")
            traceback.print_exc()

    def _run_simulated_annealing(self, evolution):
        """Run simulated annealing algorithm."""
        from ..evolution.algorithms import SimulatedAnnealing, AlgorithmType
        from ..evolution.foundation_evolution import EVOLVABLE_BUILDINGS

        # Create SA algorithm with evolution's evaluate function
        sa = SimulatedAnnealing(
            config=evolution.config,
            evaluate_fn=evolution.evaluate_fitness,
            valid_buildings=evolution._get_valid_buildings(),
            max_buildings=evolution.max_buildings,
            initial_temp=100.0,
            cooling_rate=0.995,
            min_temp=0.1,
        )

        # Run SA for generations * population iterations
        iterations = self.foundation_generations * 10
        print(f"\nRunning Simulated Annealing for {iterations} iterations...")
        best = sa.run(iterations, verbose=True)

        # Return as list for compatibility
        return [best] if best else []

    def _run_hybrid_algorithm(self, evolution):
        """Run hybrid algorithm (SA + EA)."""
        from ..evolution.algorithms import HybridAlgorithm

        # Create hybrid algorithm
        hybrid = HybridAlgorithm(
            config=evolution.config,
            evaluate_fn=evolution.evaluate_fitness,
            valid_buildings=evolution._get_valid_buildings(),
            max_buildings=evolution.max_buildings,
            sa_iterations=self.foundation_generations * 5,
            ea_iterations=self.foundation_generations * 5,
        )

        print(f"\nRunning Hybrid Algorithm...")
        best = hybrid.run(self.foundation_generations, verbose=True)

        return [best] if best else []

    def _run_two_phase(self, inputs, outputs):
        """Run two-phase evolution (system search + layout search)."""
        from ..evolution.two_phase_evolution import create_two_phase_evolution

        # Determine which sub-algorithm to use
        if self.foundation_algorithm == "Two-Phase SA":
            algorithm = 'sa'
        elif self.foundation_algorithm == "Two-Phase Hybrid":
            algorithm = 'hybrid'
        else:
            algorithm = 'evolution'

        print(f"\nRunning Two-Phase Evolution ({algorithm})...")
        print("Phase 1: System Search - Finding optimal machine topology")
        print("Phase 2: Layout Search - Placing machines and routing belts")

        # Create two-phase evolution
        two_phase = create_two_phase_evolution(
            foundation_type=self.foundation_type,
            input_specs=inputs,
            output_specs=outputs,
        )

        # Run with half generations for each phase
        system_gens = self.foundation_generations // 2
        layout_gens = self.foundation_generations // 2

        result = two_phase.run(
            system_generations=system_gens,
            layout_generations=layout_gens,
            algorithm=algorithm,
            verbose=True
        )

        # Store the two-phase evolution for the viewer
        self.foundation_evolution = two_phase

        # Return the candidate for compatibility
        candidate = result.to_candidate()
        return [candidate] if candidate else []

    def _view_foundation_layout(self) -> None:
        """Open the layout viewer for foundation solutions."""
        if not self.foundation_evolution or not self.foundation_solutions:
            print("No foundation solutions available. Run evolution first.")
            return

        # Use the pygame layout viewer
        from ..visualization.pygame_layout_viewer import show_layout_pygame
        show_layout_pygame(self.foundation_evolution)

    def _stop_evolution(self) -> None:
        """Stop the current evolution run."""
        self.evolution_running = False

    def _render(self) -> None:
        """Render the application."""
        self._screen.fill((40, 40, 50))

        # Draw right panel based on mode
        if self.mode == AppMode.SHAPE_TRANSFORM:
            self._draw_shape_preview()
        else:
            self._draw_foundation_preview()

        self._manager.draw_ui(self._screen)
        pygame.display.flip()

    def _draw_shape_preview(self) -> None:
        """Draw shape previews in the right panel."""
        preview_x = 400
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

    def _draw_foundation_preview(self) -> None:
        """Draw foundation info in the right panel."""
        preview_x = 400
        preview_y = 50

        spec = FOUNDATION_SPECS.get(self.foundation_type)
        if not spec:
            return

        if self._font:
            # Title
            label = self._font.render(f"Foundation: {self.foundation_type}", True, (200, 200, 200))
            self._screen.blit(label, (preview_x, preview_y))
            preview_y += 30

            # Dimensions
            label = self._font.render(f"Units: {spec.units_x}x{spec.units_y}", True, (150, 150, 150))
            self._screen.blit(label, (preview_x, preview_y))
            preview_y += 25

            label = self._font.render(f"Grid: {spec.grid_width}x{spec.grid_height} tiles", True, (150, 150, 150))
            self._screen.blit(label, (preview_x, preview_y))
            preview_y += 25

            label = self._font.render(f"Floors: {spec.num_floors}", True, (150, 150, 150))
            self._screen.blit(label, (preview_x, preview_y))
            preview_y += 35

            # Ports per side
            label = self._font.render("Ports per side:", True, (200, 200, 200))
            self._screen.blit(label, (preview_x, preview_y))
            preview_y += 25

            for side in [Side.NORTH, Side.EAST, Side.SOUTH, Side.WEST]:
                count = spec.ports_per_side[side]
                label = self._font.render(f"  {side.value}: {count} ports", True, (150, 150, 150))
                self._screen.blit(label, (preview_x, preview_y))
                preview_y += 20

            preview_y += 20

            # Shape code help
            label = self._font.render("Shape code format:", True, (200, 200, 200))
            self._screen.blit(label, (preview_x, preview_y))
            preview_y += 25

            help_lines = [
                "CuCuCuCu = Full circle",
                "Cu------ = NE corner only",
                "--Cu---- = NW corner only",
                "----Cu-- = SW corner only",
                "------Cu = SE corner only",
            ]
            for line in help_lines:
                label = self._font.render(line, True, (120, 120, 130))
                self._screen.blit(label, (preview_x, preview_y))
                preview_y += 20

            # Solution info if available
            if self.foundation_solutions:
                preview_y += 20
                label = self._font.render(f"Solutions found: {len(self.foundation_solutions)}", True, (100, 200, 100))
                self._screen.blit(label, (preview_x, preview_y))
                preview_y += 25

                best = self.foundation_solutions[0]
                label = self._font.render(f"Best fitness: {best.fitness:.2f}", True, (100, 200, 100))
                self._screen.blit(label, (preview_x, preview_y))

    def _check_color_change(self, input_shape: Shape, target_shape: Shape) -> bool:
        """Check if transformation requires color changes (not currently supported)."""
        input_colors = set()
        target_colors = set()

        for layer in input_shape.layers:
            for part in layer.parts:
                if not part.is_empty():
                    input_colors.add(part.color)

        for layer in target_shape.layers:
            for part in layer.parts:
                if not part.is_empty():
                    target_colors.add(part.color)

        # Check if target has colors not in input
        new_colors = target_colors - input_colors
        return len(new_colors) > 0

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

        # Check if painting is needed
        needs_painting = self._check_color_change(input_shape, target_shape)
        if needs_painting:
            print("\n*** PAINTING MODE ENABLED ***")
            print("The target shape contains colors not present in the input.")
            print("Enabling painting mode with all available colors.")
            print("*****************************\n")
            self.enable_painting = True

        foundation = get_foundation(self.selected_foundation)

        config = EvolutionConfig(
            population_size=self.population_size,
            generations=self.generations,
            allowed_operations=self.selected_operations,
            parallel_evaluation=True,
            enable_painting=self.enable_painting,
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

            # Display the solution pipeline
            from ..visualization.solution_display import display_solution
            print("\n")
            display_solution(result.design, {"in_0": input_shape})
        else:
            print("\nNo solution found.")


def main():
    """Entry point for the application."""
    app = SolverApp()
    app.run()


if __name__ == "__main__":
    main()
