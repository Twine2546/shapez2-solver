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
            options_list=["1x1", "2x1", "1x2", "2x2", "3x2", "2x3", "3x3"],
            starting_option=self.foundation_type,
            relative_rect=pygame.Rect(margin + 120, y_pos, 150, 30),
            manager=self._manager
        )
        y_pos += 40

        # Inputs text
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 25),
            text="Inputs (Side,Pos,Floor,Shape):",
            manager=self._manager
        )
        y_pos += 30
        self._ui_elements['inputs_text'] = pygame_gui.elements.UITextEntryBox(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 100),
            manager=self._manager,
            initial_text=self.inputs_text
        )
        y_pos += 110

        # Outputs text
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 25),
            text="Outputs (Side,Pos,Floor,Shape):",
            manager=self._manager
        )
        y_pos += 30
        self._ui_elements['outputs_text'] = pygame_gui.elements.UITextEntryBox(
            relative_rect=pygame.Rect(margin, y_pos, left_panel_width, 150),
            manager=self._manager,
            initial_text=self.outputs_text
        )
        y_pos += 160

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

        # Solve
        print("\n" + "="*70)
        print("SOLVING WITH CP-SAT (Maximum Throughput Optimization)")
        print("="*70)
        print(f"Foundation: {self.foundation_type}")
        print(f"Inputs: {len(inputs)}, Outputs: {len(outputs)}")

        try:
            self.solution = solve_with_cpsat(
                foundation_type=self.foundation_type,
                input_specs=inputs,
                output_specs=outputs,
                max_machines=self.max_machines,
                time_limit=self.time_limit,
                verbose=True
            )

            if self.solution:
                # Export to blueprint
                try:
                    self.blueprint_code = self._export_candidate_to_blueprint(self.solution)
                except Exception as e:
                    self.blueprint_code = f"(Blueprint export failed: {e})"

                # Display results
                machines = len([b for b in self.solution.buildings
                               if any(m in str(b.building_type)
                                     for m in ['CUTTER', 'SPLITTER', 'ROTATOR', 'STACKER'])])
                belts = len(self.solution.buildings) - machines

                result_text = f"""<font face='monospace' size=3><b>SOLUTION FOUND!</b>

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

                self._show_results(result_text)

                print("\n✓ Solution found! Click 'View Layout' to visualize.")
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
        pygame.display.flip()


def main():
    """Run the CP-SAT solver GUI."""
    app = CPSATSolverApp()
    app.run()


if __name__ == "__main__":
    main()
