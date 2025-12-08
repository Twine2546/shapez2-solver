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
from ..simulator.design import Design, OperationNode, InputNode, OutputNode
from .shape_renderer import ShapeRenderer


# Building colors for visualization
BUILDING_COLORS = {
    "RotateOperation": (100, 200, 100),      # Green - rotator
    "CutOperation": (200, 100, 100),          # Red - cutter
    "HalfDestroyerOperation": (180, 80, 80),  # Dark red - half destroyer
    "StackOperation": (100, 100, 200),        # Blue - stacker
    "UnstackOperation": (80, 80, 180),        # Dark blue - unstacker
    "SwapperOperation": (200, 200, 100),      # Yellow - swapper
    "PaintOperation": (200, 100, 200),        # Purple - painter
    "Input": (50, 150, 50),                   # Dark green - input
    "Output": (150, 50, 50),                  # Dark red - output
}

BUILDING_SYMBOLS = {
    "RotateOperation": "R",
    "CutOperation": "C",
    "HalfDestroyerOperation": "H",
    "StackOperation": "S",
    "UnstackOperation": "U",
    "SwapperOperation": "X",
    "PaintOperation": "P",
    "Input": "I",
    "Output": "O",
}


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
        self.best_blueprint_code: Optional[str] = None

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

            # Generate blueprint code for best solution if fitness is good
            if self.best_fitness >= 0.95 and sorted_pop[0].design.operations:
                try:
                    from ..blueprint import export_blueprint
                    self.best_blueprint_code = export_blueprint(sorted_pop[0].design)
                except Exception:
                    self.best_blueprint_code = None

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

        # Draw legend
        self._draw_legend()

        # Draw blueprint code if available
        self._draw_blueprint_code()

        pygame.display.flip()

    def _draw_blueprint_code(self) -> None:
        """Draw the blueprint code for the best solution."""
        if not self._font or not self.best_blueprint_code:
            return

        code_y = self.height - 50

        # Draw background
        bg_rect = pygame.Rect(0, code_y - 5, self.width, 55)
        pygame.draw.rect(self._screen, (30, 50, 30), bg_rect)
        pygame.draw.line(self._screen, (80, 150, 80), (0, code_y - 5), (self.width, code_y - 5), 2)

        # Label
        label = self._font.render("Blueprint Code (copy to game):", True, (100, 255, 100))
        self._screen.blit(label, (10, code_y))

        # Code (truncated if too long)
        code = self.best_blueprint_code
        max_len = 120
        if len(code) > max_len:
            code = code[:max_len-3] + "..."

        code_surface = self._font.render(code, True, (200, 255, 200))
        self._screen.blit(code_surface, (10, code_y + 18))

        # Hint
        hint = self._font.render("(Full code printed to console when evolution ends)", True, (150, 150, 150))
        self._screen.blit(hint, (10, code_y + 36))

    def _draw_legend(self) -> None:
        """Draw a legend for the building symbols."""
        if not self._font:
            return

        legend_x = 20
        legend_y = self.height - 120

        # Title
        title = self._font.render("Legend:", True, (200, 200, 200))
        self._screen.blit(title, (legend_x, legend_y))

        # Legend items
        items = [
            ("I", BUILDING_COLORS["Input"], "Input"),
            ("R", BUILDING_COLORS["RotateOperation"], "Rotate"),
            ("C", BUILDING_COLORS["CutOperation"], "Cut"),
            ("S", BUILDING_COLORS["StackOperation"], "Stack"),
            ("U", BUILDING_COLORS["UnstackOperation"], "Unstack"),
            ("X", BUILDING_COLORS["SwapperOperation"], "Swap"),
            ("O", BUILDING_COLORS["Output"], "Output"),
        ]

        x = legend_x
        y = legend_y + 20

        for symbol, color, name in items:
            # Draw colored box
            box_rect = pygame.Rect(x, y, 14, 14)
            pygame.draw.rect(self._screen, color, box_rect)
            pygame.draw.rect(self._screen, (200, 200, 200), box_rect, 1)

            # Draw symbol
            sym = self._font.render(symbol, True, (255, 255, 255))
            sym_rect = sym.get_rect(center=box_rect.center)
            self._screen.blit(sym, sym_rect)

            # Draw name
            name_surface = self._font.render(name, True, (180, 180, 180))
            self._screen.blit(name_surface, (x + 18, y))

            x += 80
            if x > 350:
                x = legend_x
                y += 20

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
        """Draw the top candidate solutions with mini layouts."""
        section_x = 450
        section_y = 50

        if self._font:
            label = self._font.render("Top Candidates:", True, (200, 200, 200))
            self._screen.blit(label, (section_x, section_y))

        # Layout parameters
        layout_width = 160
        layout_height = 100
        layouts_per_row = 3
        y_offset = section_y + 25

        for i, candidate in enumerate(self.top_candidates):
            row = i // layouts_per_row
            col = i % layouts_per_row

            x = section_x + col * (layout_width + 20)
            y = y_offset + row * (layout_height + 40)

            # Draw mini layout for this candidate
            self._draw_mini_layout(candidate, x, y, layout_width, layout_height)

    def _draw_mini_layout(self, candidate: Candidate, x: int, y: int, width: int, height: int) -> None:
        """Draw a mini factory layout for a candidate."""
        if not PYGAME_AVAILABLE or not self._screen:
            return

        design = candidate.design

        # Draw background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self._screen, (40, 45, 55), bg_rect)
        pygame.draw.rect(self._screen, (80, 80, 100), bg_rect, 1)

        # Draw header with fitness
        if self._font:
            ops_list = [op.operation.__class__.__name__.replace("Operation", "")
                       for op in design.operations]
            ops_str = ", ".join(ops_list[:3])
            if len(ops_list) > 3:
                ops_str += "..."

            header = f"F:{candidate.fitness:.2f} | {len(design.operations)} ops"
            text = self._font.render(header, True, (200, 200, 200))
            self._screen.blit(text, (x + 5, y + 2))

        # Calculate grid dimensions
        grid_top = y + 20
        grid_height = height - 25
        cell_size = 18

        # Get execution order and positions
        nodes_info = self._get_nodes_layout(design)

        if not nodes_info:
            return

        # Calculate grid bounds
        min_col = min(n['col'] for n in nodes_info)
        max_col = max(n['col'] for n in nodes_info)
        min_row = min(n['row'] for n in nodes_info)
        max_row = max(n['row'] for n in nodes_info)

        cols = max(1, max_col - min_col + 1)
        rows = max(1, max_row - min_row + 1)

        # Scale to fit
        scale_x = min(cell_size, (width - 10) // cols)
        scale_y = min(cell_size, grid_height // rows)
        scale = min(scale_x, scale_y, cell_size)

        # Center the layout
        total_width = cols * scale
        total_height = rows * scale
        offset_x = x + (width - total_width) // 2
        offset_y = grid_top + (grid_height - total_height) // 2

        # Draw connections first (behind nodes)
        for conn in design.connections:
            src_info = next((n for n in nodes_info if n['id'] == conn.source_id), None)
            tgt_info = next((n for n in nodes_info if n['id'] == conn.target_id), None)

            if src_info and tgt_info:
                src_x = offset_x + (src_info['col'] - min_col) * scale + scale // 2
                src_y = offset_y + (src_info['row'] - min_row) * scale + scale // 2
                tgt_x = offset_x + (tgt_info['col'] - min_col) * scale + scale // 2
                tgt_y = offset_y + (tgt_info['row'] - min_row) * scale + scale // 2

                # Draw connection line
                pygame.draw.line(self._screen, (100, 100, 120), (src_x, src_y), (tgt_x, tgt_y), 1)

        # Draw nodes
        for node_info in nodes_info:
            node_x = offset_x + (node_info['col'] - min_col) * scale
            node_y = offset_y + (node_info['row'] - min_row) * scale

            color = node_info['color']
            symbol = node_info['symbol']

            # Draw node rectangle
            node_rect = pygame.Rect(node_x + 1, node_y + 1, scale - 2, scale - 2)
            pygame.draw.rect(self._screen, color, node_rect)
            pygame.draw.rect(self._screen, (200, 200, 200), node_rect, 1)

            # Draw symbol
            if self._font and scale >= 12:
                sym_surface = self._font.render(symbol, True, (255, 255, 255))
                sym_rect = sym_surface.get_rect(center=node_rect.center)
                self._screen.blit(sym_surface, sym_rect)

    def _get_nodes_layout(self, design: Design) -> List[Dict]:
        """Get node positions for layout visualization."""
        nodes_info = []
        col = 0

        # Inputs on the left
        for i, inp in enumerate(design.inputs):
            nodes_info.append({
                'id': inp.node_id,
                'col': 0,
                'row': i,
                'color': BUILDING_COLORS.get("Input", (100, 100, 100)),
                'symbol': 'I',
            })

        # Get operations in execution order
        from collections import deque
        in_degree: Dict[str, int] = {}
        adjacency: Dict[str, List[str]] = {}

        for node in design.operations:
            in_degree[node.node_id] = 0
            adjacency[node.node_id] = []

        for conn in design.connections:
            if conn.source_id in adjacency and conn.target_id in adjacency:
                in_degree[conn.target_id] += 1
                adjacency[conn.source_id].append(conn.target_id)

        queue = deque([n for n, d in in_degree.items() if d == 0])
        col = 1
        row_in_col = {}

        while queue:
            level_size = len(queue)
            row = 0
            for _ in range(level_size):
                node_id = queue.popleft()
                node = design.get_node(node_id)

                if isinstance(node, OperationNode):
                    op_name = node.operation.__class__.__name__
                    color = BUILDING_COLORS.get(op_name, (150, 150, 150))
                    symbol = BUILDING_SYMBOLS.get(op_name, "?")

                    nodes_info.append({
                        'id': node_id,
                        'col': col,
                        'row': row,
                        'color': color,
                        'symbol': symbol,
                    })
                    row += 1

                for neighbor in adjacency.get(node_id, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if level_size > 0:
                col += 1

        # Outputs on the right
        for i, out in enumerate(design.outputs):
            nodes_info.append({
                'id': out.node_id,
                'col': col,
                'row': i,
                'color': BUILDING_COLORS.get("Output", (100, 100, 100)),
                'symbol': 'O',
            })

        return nodes_info


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

    # Print blueprint code if we found a good solution
    if result and result.fitness >= 0.95 and result.design.operations:
        try:
            from ..blueprint import export_blueprint
            blueprint_code = export_blueprint(result.design)
            print("\n" + "=" * 60)
            print("BLUEPRINT CODE")
            print("=" * 60)
            print("Copy this code and paste it in the game to import:")
            print()
            print(blueprint_code)
            print()
            print("=" * 60)
        except Exception as e:
            print(f"Could not generate blueprint: {e}")

    return result
