"""Simplified GUI for CP-SAT Solver - displays solutions and blueprint code."""

import sys
import json

PYGAME_AVAILABLE = False
_PYGAME_ERROR = ""

try:
    import pygame
    import pygame_gui
    PYGAME_AVAILABLE = True
except ImportError as e:
    _PYGAME_ERROR = str(e)

from ..evolution.cpsat_solver import solve_with_cpsat


class CPSATSolverApp:
    """Simplified CP-SAT solver GUI with blueprint export."""

    def __init__(self, width=1400, height=900):
        self.width = width
        self.height = height
        self.running = False

        # Solution state
        self.solution = None
        self.blueprint_code = ""

        # UI state
        self.foundation_type = "2x2"
        self.inputs_text = "W,0,0,CuCuCuCu"
        self.outputs_text = "E,0,0,Cu------\nE,1,0,--Cu----\nE,2,0,----Cu--\nE,3,0,------Cu"
        self.max_machines = 20
        self.time_limit = 60

        # Pygame elements
        self._screen = None
        self._manager = None
        self._clock = None
        self._ui_elements = {}

        # Visual areas (set during UI creation)
        self._foundation_visual_rect = None
        self._io_visual_rect = None

    def run(self):
        """Run the application."""
        if not PYGAME_AVAILABLE:
            print("Error: pygame_gui is required")
            if _PYGAME_ERROR:
                print(f"Import error: {_PYGAME_ERROR}")
            print("\nTo fix: pip install pygame-ce pygame_gui")
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

    def _initialize(self):
        """Initialize pygame and UI."""
        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Shapez 2 CP-SAT Solver")
        self._manager = pygame_gui.UIManager((self.width, self.height))
        self._clock = pygame.time.Clock()

        self._create_ui()
        self.running = True

    def _create_ui(self):
        """Create UI elements."""
        left_panel_width = 450
        margin = 10
        y_pos = 10

        # Title
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 40),
            text="CP-SAT Solver - Maximum Throughput",
            manager=self._manager
        )
        y_pos += 50

        # Foundation dropdown
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(margin, y_pos, 120, 25),
            text="Foundation:",
            manager=self._manager
        )
        self._ui_elements['foundation_dropdown'] = pygame_gui.elements.UIDropDownMenu(
            options_list=["1x1", "2x1", "1x2", "2x2", "3x2", "2x3", "3x3", "T", "L", "L4", "S4", "Cross"],
            starting_option=self.foundation_type,
            relative_rect=pygame.Rect(margin + 120, y_pos, 150, 30),
            manager=self._manager
        )
        y_pos += 40

        # Foundation visual preview
        self._foundation_visual_rect = pygame.Rect(margin, y_pos, left_panel_width, 120)
        y_pos += 130

        # Timeout input
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(margin, y_pos, 120, 25),
            text="Timeout (sec):",
            manager=self._manager
        )
        self._ui_elements['timeout_input'] = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(margin + 120, y_pos, 80, 30),
            manager=self._manager
        )
        self._ui_elements['timeout_input'].set_text(str(self.time_limit))
        y_pos += 40

        # Inputs text
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 25),
            text="Inputs (Side,Pos,Floor,Shape):",
            manager=self._manager
        )
        y_pos += 30
        # Create empty to avoid pygame_gui multi-line initialization bug
        self._ui_elements['inputs_text'] = pygame_gui.elements.UITextEntryBox(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 100),
            manager=self._manager,
            initial_text=""
        )
        # Set text after creation to avoid layout corruption
        if self.inputs_text:
            self._ui_elements['inputs_text'].set_text(self.inputs_text)
        y_pos += 110

        # Outputs text
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 25),
            text="Outputs (Side,Pos,Floor,Shape):",
            manager=self._manager
        )
        y_pos += 30
        # Create empty to avoid pygame_gui multi-line initialization bug
        self._ui_elements['outputs_text'] = pygame_gui.elements.UITextEntryBox(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 150),
            manager=self._manager,
            initial_text=""
        )
        # Set text after creation to avoid layout corruption
        if self.outputs_text:
            self._ui_elements['outputs_text'].set_text(self.outputs_text)
        y_pos += 160

        # I/O port visual preview
        self._io_visual_rect = pygame.Rect(margin, y_pos, left_panel_width, 100)
        y_pos += 110

        # Solve button
        self._ui_elements['solve_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(margin, y_pos, 200, 40),
            text="Solve with CP-SAT",
            manager=self._manager
        )
        y_pos += 50

        # View button
        self._ui_elements['view_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(margin, y_pos, 200, 40),
            text="View Layout",
            manager=self._manager
        )
        y_pos += 50

        # Copy blueprint button
        self._ui_elements['copy_button'] = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(margin, y_pos, 200, 40),
            text="Copy Blueprint Code",
            manager=self._manager
        )
        y_pos += 50

        # Shape code legend
        legend_text = """<font face='monospace' size=4><b>Shape Code Legend:</b>

<b>Shape Types:</b> C=Circle R=Square S=Star W=Diamond c=Crystal P=Pin -=Empty
<b>Colors:</b> u=Uncolored r=Red g=Green b=Blue c=Cyan m=Magenta y=Yellow w=White

<b>Port Format:</b> Side,Position,Floor,ShapeCode
  • Side: N/E/S/W (North/East/South/West)
  • Position: 0-3 (for 1x1), 0-7 (for 2x1/2x2), 0-11 (for 3x3)
  • Floor: 0-3
  • ShapeCode: 8 chars (4 quadrants × 2 chars)

<b>Foundation Examples:</b>

<b>1x1 Foundation (14×14 grid):</b>
Input:  W,0,0,CuCuCuCu
Output: E,0,0,Cu------
        E,1,0,--Cu----
Ports: 4 per side (positions 0-3)

<b>2x2 Foundation (34×34 grid) - Corner Splitter:</b>
Input:  W,0,0,RuRuRuRu
Output: E,0,0,Ru------
        E,1,0,--Ru----
        E,2,0,----Ru--
        E,3,0,------Ru
Ports: 8 per side (positions 0-7)

<b>3x3 Foundation (54×54 grid) - Multi-output:</b>
Input:  W,0,0,CuCuCuCu
Output: E,0,0,CuCuCuCu
        E,1,0,CuCuCuCu
        E,2,0,CuCuCuCu
        E,3,0,CuCuCuCu
Ports: 12 per side (positions 0-11)

<b>Multi-Floor Example (3 floors):</b>
Input:  W,0,0,SuSuSuSu
        W,0,1,SuSuSuSu
        W,0,2,SuSuSuSu
Output: E,0,0,Su------
        E,1,1,--Su----
        E,2,2,----Su--

<b>Tips:</b>
• Larger foundations = more space for complex routing
• Use multiple floors for parallel processing
• Positions are numbered along each side
</font>"""

        pygame_gui.elements.UITextBox(
            html_text=legend_text,
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, self.height - y_pos - margin),
            manager=self._manager
        )

        # Status/Results panel (right side)
        results_x = left_panel_width + margin * 2
        results_width = self.width - results_x - margin
        results_height = self.height - margin * 2

        self._ui_elements['results_text'] = pygame_gui.elements.UITextBox(
            html_text="<font face='monospace' size=3>No solution yet. Configure inputs/outputs and click 'Solve'.</font>",
            relative_rect=pygame.Rect(results_x, margin, results_width, results_height),
            manager=self._manager
        )

    def _handle_button(self, event):
        """Handle button clicks."""
        if event.ui_element == self._ui_elements.get('solve_button'):
            self._solve()
        elif event.ui_element == self._ui_elements.get('view_button'):
            self._view_layout()
        elif event.ui_element == self._ui_elements.get('copy_button'):
            self._copy_blueprint()

    def _handle_text_entry(self, event):
        """Handle text entry changes."""
        if event.ui_element == self._ui_elements.get('inputs_text'):
            self.inputs_text = event.text
        elif event.ui_element == self._ui_elements.get('outputs_text'):
            self.outputs_text = event.text
        elif event.ui_element == self._ui_elements.get('timeout_input'):
            try:
                self.time_limit = int(event.text)
            except ValueError:
                pass  # Ignore invalid input, keep previous value

    def _handle_dropdown(self, event):
        """Handle dropdown changes."""
        if event.ui_element == self._ui_elements.get('foundation_dropdown'):
            self.foundation_type = event.text

    def _solve(self):
        """Solve using CP-SAT."""
        # Read current text from UI elements (in case events didn't fire)
        if 'inputs_text' in self._ui_elements:
            self.inputs_text = self._ui_elements['inputs_text'].get_text()
        if 'outputs_text' in self._ui_elements:
            self.outputs_text = self._ui_elements['outputs_text'].get_text()
        if 'timeout_input' in self._ui_elements:
            try:
                self.time_limit = int(self._ui_elements['timeout_input'].get_text())
            except ValueError:
                self.time_limit = 60  # Default if invalid input

        # Parse inputs
        inputs = []
        for line in self.inputs_text.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) == 4:
                    inputs.append((parts[0].strip(), int(parts[1]), int(parts[2]), parts[3].strip()))

        # Parse outputs
        outputs = []
        for line in self.outputs_text.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) == 4:
                    outputs.append((parts[0].strip(), int(parts[1]), int(parts[2]), parts[3].strip()))

        if not inputs or not outputs:
            self._show_results("ERROR: Need at least one input and one output")
            return

        # Foundation size progression (smallest to largest by total ports)
        # Rectangular foundations by increasing size, then irregulars
        foundation_progression = [
            "1x1",   # 16 ports
            "2x1", "1x2",  # 24 ports
            "L",     # 32 ports (irregular)
            "2x2", "3x1",  # 32 ports
            "T", "L4", "S4",  # 40 ports (irregular)
            "3x2", "2x3",  # 40 ports
            "Cross", # 48 ports (irregular)
            "3x3"    # 48 ports
        ]

        # Find starting index based on current foundation
        try:
            start_idx = foundation_progression.index(self.foundation_type)
        except ValueError:
            start_idx = 0

        # Try each foundation size until success or run out of sizes
        print("\n" + "="*70)
        print("SOLVING WITH CP-SAT (Maximum Throughput Optimization)")
        print("="*70)
        print(f"Starting foundation: {self.foundation_type}")
        print(f"Inputs: {len(inputs)}, Outputs: {len(outputs)}")
        print(f"Auto-scaling enabled: Will try larger foundations if routing fails")
        print("="*70)

        self.solution = None
        tried_foundations = []

        try:
            for foundation_idx in range(start_idx, len(foundation_progression)):
                current_foundation = foundation_progression[foundation_idx]
                tried_foundations.append(current_foundation)

                print(f"\n{'='*70}")
                print(f"TRYING FOUNDATION: {current_foundation}")
                print(f"{'='*70}")

                solution = solve_with_cpsat(
                    foundation_type=current_foundation,
                    input_specs=inputs,
                    output_specs=outputs,
                    max_machines=self.max_machines,
                    time_limit=self.time_limit,
                    verbose=True
                )

                if solution and hasattr(solution, 'routing_success') and solution.routing_success:
                    # Success! Use this foundation
                    self.solution = solution
                    self.foundation_type = current_foundation

                    # Update the dropdown to show the successful foundation
                    if 'foundation_dropdown' in self._ui_elements:
                        self._ui_elements['foundation_dropdown'].selected_option = current_foundation

                    print(f"\n{'='*70}")
                    print(f"✓ SUCCESS with {current_foundation} foundation!")
                    print(f"Tried {len(tried_foundations)} foundation(s): {', '.join(tried_foundations)}")
                    print(f"{'='*70}")
                    break
                else:
                    print(f"\n⚠ {current_foundation} failed - trying next size...")

            if not self.solution or not self.solution.routing_success:
                print(f"\n{'='*70}")
                print(f"✗ All foundations exhausted without success")
                print(f"Tried: {', '.join(tried_foundations)}")
                print(f"{'='*70}")

            if self.solution:
                # Display results
                machines = len([b for b in self.solution.buildings
                               if any(m in str(b.building_type)
                                     for m in ['CUTTER', 'SPLITTER', 'ROTATOR', 'STACKER'])])
                belts = len(self.solution.buildings) - machines

                # Check if routing was successful
                if hasattr(self.solution, 'routing_success') and self.solution.routing_success:
                    # Export to blueprint (only for successful routing)
                    try:
                        self.blueprint_code = self._export_candidate_to_blueprint(self.solution)
                    except Exception as e:
                        self.blueprint_code = f"(Blueprint export failed: {e})"

                    result_text = f"""<font face='monospace' size=3><b>✓ SOLUTION FOUND!</b>

<b>Foundation:</b> {self.foundation_type}
{f"<b>Auto-scaled:</b> Tried {len(tried_foundations)} foundation(s): {', '.join(tried_foundations)}" if len(tried_foundations) > 1 else ""}

Fitness: {self.solution.fitness:.1f}
Machines: {machines}
Belts: {belts}
Total Buildings: {len(self.solution.buildings)}

<b>Blueprint Code:</b>
{self.blueprint_code[:500]}{'...' if len(self.blueprint_code) > 500 else ''}

<b>Actions:</b>
• Click 'View Layout' to see the solution visualization
• Click 'Copy Blueprint Code' to copy to clipboard
• Import the code in Shapez 2 to use the solution!

<b>Throughput Optimization:</b>
• Automatically uses splitters (180 items/min) for pure splitting
• Uses cutters (45 items/min) only when shape transformation needed
• Multiple inputs create independent processing trees
• Maximizes throughput for fully upgraded equipment
</font>"""
                    print(f"\n✓ Solution found on {self.foundation_type}! Click 'View Layout' to visualize.")
                else:
                    # Routing failed on all foundations
                    grid_sizes = {
                        "1x1": "14×14 (196 tiles)",
                        "2x1": "34×14 (476 tiles)",
                        "1x2": "14×34 (476 tiles)",
                        "2x2": "34×34 (1,156 tiles)",
                        "3x2": "54×34 (1,836 tiles)",
                        "2x3": "34×54 (1,836 tiles)",
                        "3x3": "54×54 (2,916 tiles)"
                    }

                    result_text = f"""<font face='monospace' size=3><b>⚠ ALL FOUNDATIONS EXHAUSTED</b>

The solver tried {len(tried_foundations)} foundation size(s) but couldn't complete routing.

<b>Foundations tried:</b> {', '.join(tried_foundations)}
<b>Machines needed:</b> {machines}
<b>Best attempt:</b> {belts} belts placed (incomplete routing)

<b>Problem:</b> The task is too complex even for the largest foundation (3x3).

<b>Solutions:</b>
1. <b>Increase timeout:</b> Try 120-300 seconds for more solver iterations

2. <b>Reduce complexity:</b>
   • Fewer outputs per input
   • Simpler shape transformations
   • Use fewer input streams

3. <b>Use multiple foundations:</b>
   • Split the task across 2-3 separate foundations
   • Process in stages (e.g., split first, then transform)

4. <b>Manual design:</b>
   • This task may require custom hand-crafted layout
</font>"""
                    print(f"\n⚠ All foundations tried ({', '.join(tried_foundations)}) - none succeeded.")

                self._show_results(result_text)
            else:
                self._show_results("ERROR: No solution found. Try a larger foundation or fewer machines.")
                print("\n✗ No solution found.")

        except Exception as e:
            import traceback
            error_msg = f"<font face='monospace' size=3 color='#FF0000'><b>ERROR:</b>\n{str(e)}</font>"
            self._show_results(error_msg)
            print(f"\n✗ Error: {e}")
            traceback.print_exc()

    def _view_layout(self):
        """View the solution layout."""
        if not self.solution:
            print("No solution to view")
            return

        try:
            from ..visualization.pygame_layout_viewer import show_layout_pygame
            from ..evolution.foundation_config import FOUNDATION_SPECS, FoundationConfig

            # Create a proper FoundationConfig for the viewer
            spec = FOUNDATION_SPECS.get(self.foundation_type)
            if spec is None:
                print(f"Error: Unknown foundation type '{self.foundation_type}'")
                return

            config = FoundationConfig(spec)

            # Create a mock solver object with top_solutions and config for the viewer
            class MockSolver:
                def __init__(self, solution, foundation_config):
                    self.top_solutions = [solution]
                    self.config = foundation_config

            mock = MockSolver(self.solution, config)
            show_layout_pygame(mock)

            # After returning from layout viewer, reinitialize the UI
            # This is needed because the display mode was changed
            print("Reinitializing UI after layout viewer...")

            # Save current values before recreating UI
            saved_inputs = self.inputs_text
            saved_outputs = self.outputs_text
            saved_foundation = self.foundation_type
            saved_timeout = self.time_limit

            # Properly dispose of old UI elements to avoid corruption
            for element in self._ui_elements.values():
                if element is not None:
                    element.kill()
            self._ui_elements.clear()

            # Clear the old manager
            if self._manager is not None:
                self._manager.clear_and_reset()

            # Temporarily clear text values so new elements start fresh
            self.inputs_text = ""
            self.outputs_text = ""
            self.foundation_type = saved_foundation
            self.time_limit = saved_timeout

            # Clear visual rects (they'll be recreated)
            self._foundation_visual_rect = None
            self._io_visual_rect = None

            # Recreate UI manager and elements with empty text
            self._manager = pygame_gui.UIManager((self.width, self.height))
            self._create_ui()

            # Now set the text AFTER elements are created to avoid internal state issues
            if 'inputs_text' in self._ui_elements and self._ui_elements['inputs_text'] is not None:
                self._ui_elements['inputs_text'].set_text(saved_inputs)
                self.inputs_text = saved_inputs

            if 'outputs_text' in self._ui_elements and self._ui_elements['outputs_text'] is not None:
                self._ui_elements['outputs_text'].set_text(saved_outputs)
                self.outputs_text = saved_outputs

            if 'timeout_input' in self._ui_elements and self._ui_elements['timeout_input'] is not None:
                self._ui_elements['timeout_input'].set_text(str(saved_timeout))
                self.time_limit = saved_timeout

        except Exception as e:
            print(f"Error viewing layout: {e}")
            import traceback
            traceback.print_exc()

    def _copy_blueprint(self):
        """Copy blueprint code to clipboard."""
        if not self.blueprint_code:
            print("No blueprint code to copy")
            return

        try:
            # Try to copy to clipboard
            import pyperclip
            pyperclip.copy(self.blueprint_code)
            print("✓ Blueprint code copied to clipboard!")
            self._show_results(self._ui_elements['results_text'].html_text.replace(
                "Copy Blueprint Code",
                "<b>✓ COPIED TO CLIPBOARD!</b>"
            ))
        except ImportError:
            # Fallback: print to console
            print("\n" + "="*70)
            print("BLUEPRINT CODE (copy from console):")
            print("="*70)
            print(self.blueprint_code)
            print("="*70)
            print("\nNote: Install 'pyperclip' for automatic clipboard copying:")
            print("  pip install pyperclip")

    def _export_candidate_to_blueprint(self, candidate):
        """Export a Candidate to blueprint JSON format."""
        import json
        from ..blueprint.building_types import BuildingType

        # First pass: identify belt port pairs
        # Belt ports need channel IDs to link senders with receivers
        senders = []
        receivers = []

        for building in candidate.buildings:
            bt = building.building_type
            if bt == BuildingType.BELT_PORT_SENDER:
                senders.append(building)
            elif bt == BuildingType.BELT_PORT_RECEIVER:
                receivers.append(building)

        # Match senders with receivers based on proximity and floor
        # Senders and receivers should be paired in the order they were created
        belt_port_channels = {}  # building -> channel_id
        for i, (sender, receiver) in enumerate(zip(senders, receivers)):
            channel_id = i  # Channel IDs start at 0
            belt_port_channels[id(sender)] = channel_id
            belt_port_channels[id(receiver)] = channel_id

        # Second pass: create building entries with channel IDs for belt ports
        buildings_data = []
        for building in candidate.buildings:
            # Get building properties
            bt = building.building_type

            # Convert BuildingType enum to internal name
            if hasattr(bt, 'value'):
                type_name = bt.value
            else:
                type_name = str(bt)

            # Get rotation value
            if hasattr(building.rotation, 'value'):
                rotation_val = building.rotation.value
            else:
                rotation_val = 0

            building_dict = {
                "type": type_name,
                "x": building.x,
                "y": building.y,
                "z": building.floor,
                "r": rotation_val
            }

            # Add channel ID for belt ports
            if bt in [BuildingType.BELT_PORT_SENDER, BuildingType.BELT_PORT_RECEIVER]:
                channel_id = belt_port_channels.get(id(building))
                if channel_id is not None:
                    building_dict["ch"] = channel_id  # Channel ID field

            buildings_data.append(building_dict)

        # Create blueprint structure
        blueprint = {
            "buildings": buildings_data,
            "version": 1
        }

        # Convert to JSON string
        return json.dumps(blueprint, indent=2)

    def _show_results(self, text):
        """Update results panel."""
        self._ui_elements['results_text'].html_text = text
        self._ui_elements['results_text'].rebuild()

    def _render(self):
        """Render the UI."""
        self._screen.fill((30, 30, 30))
        self._manager.draw_ui(self._screen)

        # Draw custom visuals on top
        self._draw_foundation_visual()
        self._draw_io_visual()

        pygame.display.flip()

    def _draw_foundation_visual(self):
        """Draw a visual representation of the selected foundation."""
        if not self._foundation_visual_rect:
            return

        from ..evolution.foundation_config import FOUNDATION_SPECS

        # Get foundation spec
        spec = FOUNDATION_SPECS.get(self.foundation_type)
        if not spec:
            return

        # Draw background
        rect = self._foundation_visual_rect
        pygame.draw.rect(self._screen, (50, 50, 50), rect)
        pygame.draw.rect(self._screen, (100, 100, 100), rect, 2)

        # Calculate cell size based on foundation dimensions
        max_dim = max(spec.units_x, spec.units_y)
        cell_size = min((rect.width - 40) // max_dim, (rect.height - 40) // max_dim)

        # Center the foundation drawing
        start_x = rect.x + (rect.width - spec.units_x * cell_size) // 2
        start_y = rect.y + (rect.height - spec.units_y * cell_size) // 2

        # Draw foundation cells
        if spec.present_cells:
            # Irregular foundation
            for x, y in spec.present_cells:
                cell_rect = pygame.Rect(
                    start_x + x * cell_size,
                    start_y + y * cell_size,
                    cell_size - 2,
                    cell_size - 2
                )
                pygame.draw.rect(self._screen, (80, 150, 220), cell_rect)
                pygame.draw.rect(self._screen, (120, 180, 255), cell_rect, 2)
        else:
            # Regular rectangular foundation
            for x in range(spec.units_x):
                for y in range(spec.units_y):
                    cell_rect = pygame.Rect(
                        start_x + x * cell_size,
                        start_y + y * cell_size,
                        cell_size - 2,
                        cell_size - 2
                    )
                    pygame.draw.rect(self._screen, (80, 150, 220), cell_rect)
                    pygame.draw.rect(self._screen, (120, 180, 255), cell_rect, 2)

        # Draw label
        if hasattr(pygame, 'font'):
            font = pygame.font.SysFont('monospace', 14)
            label = font.render(f"{self.foundation_type} Foundation Preview", True, (200, 200, 200))
            self._screen.blit(label, (rect.x + 5, rect.y + 5))

    def _draw_io_visual(self):
        """Draw a visual representation of inputs and outputs."""
        if not self._io_visual_rect:
            return

        from ..evolution.foundation_config import FOUNDATION_SPECS

        # Get foundation spec
        spec = FOUNDATION_SPECS.get(self.foundation_type)
        if not spec:
            return

        # Draw background
        rect = self._io_visual_rect
        pygame.draw.rect(self._screen, (50, 50, 50), rect)
        pygame.draw.rect(self._screen, (100, 100, 100), rect, 2)

        # Parse inputs and outputs
        inputs = []
        for line in self.inputs_text.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3:
                    inputs.append((parts[0].strip(), int(parts[1]), int(parts[2])))

        outputs = []
        for line in self.outputs_text.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3:
                    outputs.append((parts[0].strip(), int(parts[1]), int(parts[2])))

        # Calculate dimensions
        max_dim = max(spec.units_x, spec.units_y)
        box_size = min((rect.width - 60) // max_dim, (rect.height - 40) // max_dim)

        # Center the drawing
        start_x = rect.x + (rect.width - spec.units_x * box_size) // 2
        start_y = rect.y + 25 + (rect.height - 25 - spec.units_y * box_size) // 2

        # Draw foundation outline
        foundation_width = spec.units_x * box_size
        foundation_height = spec.units_y * box_size
        pygame.draw.rect(self._screen, (80, 80, 80),
                        pygame.Rect(start_x, start_y, foundation_width, foundation_height), 2)

        # Draw input ports (green)
        for side, pos, floor in inputs:
            self._draw_port_marker(side, pos, spec, start_x, start_y, box_size, (100, 255, 100))

        # Draw output ports (red)
        for side, pos, floor in outputs:
            self._draw_port_marker(side, pos, spec, start_x, start_y, box_size, (255, 100, 100))

        # Draw label and legend
        if hasattr(pygame, 'font'):
            font = pygame.font.SysFont('monospace', 14)
            label = font.render(f"I/O Ports: ", True, (200, 200, 200))
            self._screen.blit(label, (rect.x + 5, rect.y + 5))

            # Legend
            input_label = font.render(f"Inputs: {len(inputs)}", True, (100, 255, 100))
            output_label = font.render(f"Outputs: {len(outputs)}", True, (255, 100, 100))
            self._screen.blit(input_label, (rect.x + 100, rect.y + 5))
            self._screen.blit(output_label, (rect.x + 250, rect.y + 5))

    def _draw_port_marker(self, side, pos, spec, start_x, start_y, box_size, color):
        """Draw a port marker on the foundation edge."""
        from ..evolution.foundation_config import Side

        # Get ports per side
        ports_per_side = spec.ports_per_side

        # Calculate port position based on side
        marker_size = 6

        if side.upper() == 'N':
            total_ports = ports_per_side[Side.NORTH]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x + (pos / total_ports) * spec.units_x * box_size
                port_y = start_y - marker_size
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)

        elif side.upper() == 'S':
            total_ports = ports_per_side[Side.SOUTH]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x + (pos / total_ports) * spec.units_x * box_size
                port_y = start_y + spec.units_y * box_size + marker_size
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)

        elif side.upper() == 'E':
            total_ports = ports_per_side[Side.EAST]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x + spec.units_x * box_size + marker_size
                port_y = start_y + (pos / total_ports) * spec.units_y * box_size
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)

        elif side.upper() == 'W':
            total_ports = ports_per_side[Side.WEST]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x - marker_size
                port_y = start_y + (pos / total_ports) * spec.units_y * box_size
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)


def main():
    """Run the CP-SAT solver GUI."""
    app = CPSATSolverApp()
    app.run()


if __name__ == "__main__":
    main()
