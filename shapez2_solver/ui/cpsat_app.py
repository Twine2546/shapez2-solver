"""Simplified GUI for CP-SAT Solver - displays solutions and blueprint code."""

import sys
import json
import time

# Ensure console output isn't buffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

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

        # Auto-scaling state
        self._solving_state = None  # For multi-step foundation trying
        self._continue_dialog = None  # For continue/next dialog

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

                if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                    self._handle_dialog_confirmed(event)

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    # Check if it's our custom continue button
                    if hasattr(event.ui_element, 'text') and event.ui_element.text == "Continue This Foundation":
                        self._handle_continue_button()

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

        # Combined Foundation + I/O visual preview (larger)
        self._foundation_visual_rect = pygame.Rect(margin, y_pos, left_panel_width, 280)
        y_pos += 290

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

        # No separate I/O visual - it's combined with foundation preview now
        self._io_visual_rect = None  # Will use foundation_visual_rect for both

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

    def _handle_dialog_confirmed(self, event):
        """Handle confirmation dialog response."""
        if self._solving_state and event.ui_element == self._solving_state.get('dialog'):
            # User confirmed - try the next foundation
            self._solving_state['dialog'] = None
            self._continue_solving()

    def _handle_continue_button(self):
        """Handle continue button press."""
        if self._continue_dialog:
            self._continue_dialog.kill()
            self._continue_dialog = None
        # Continue on same foundation
        self._continue_current_foundation()

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

        # Initialize solving state for interactive auto-scaling
        self._solving_state = {
            'inputs': inputs,
            'outputs': outputs,
            'progression': foundation_progression,
            'current_idx': start_idx,
            'tried_foundations': [],
            'dialog': None,
            'nogood_placements': [],  # Persist across continuations
            'total_iterations': 0,     # Track total iterations on this foundation
            'start_time': time.time()  # Track total time on this foundation
        }

        print("\n" + "="*70)
        print("SOLVING WITH CP-SAT (Maximum Throughput Optimization)")
        print("="*70)
        print(f"Starting foundation: {self.foundation_type}")
        print(f"Inputs: {len(inputs)}, Outputs: {len(outputs)}")
        print(f"Timeout per foundation: {self.time_limit} seconds")
        print("="*70)
        sys.stdout.flush()

        self.solution = None

        # Start solving with the first foundation
        self._try_current_foundation()

    def _try_current_foundation(self, is_continuation=False):
        """Try solving with the current foundation in the progression.

        Args:
            is_continuation: If True, continue on same foundation with existing nogoods
        """
        if not self._solving_state:
            return

        state = self._solving_state
        current_idx = state['current_idx']
        progression = state['progression']

        if current_idx >= len(progression):
            # Exhausted all foundations
            self._finish_solving(all_failed=True)
            return

        current_foundation = progression[current_idx]

        # Only add to tried list if not a continuation
        if not is_continuation:
            if current_foundation not in state['tried_foundations']:
                state['tried_foundations'].append(current_foundation)
            # Reset nogood placements and iteration count for new foundation
            state['nogood_placements'] = []
            state['total_iterations'] = 0
            state['start_time'] = time.time()

        continuation_msg = " (CONTINUING)" if is_continuation else ""
        total_time = time.time() - state['start_time']

        print(f"\n{'='*70}")
        print(f"TRYING FOUNDATION: {current_foundation} ({current_idx + 1}/{len(progression)}){continuation_msg}")
        if is_continuation:
            print(f"Previous iterations: {state['total_iterations']}, Total time: {total_time:.1f}s")
            print(f"Nogood placements from previous attempts: {len(state['nogood_placements'])}")
        print(f"Timeout: {self.time_limit} seconds (per attempt)")
        print(f"{'='*70}")
        sys.stdout.flush()

        try:
            # Continue searching with fresh solver instance
            # Pass previous failed placements to exclude them
            solution, updated_nogoods = solve_with_cpsat(
                foundation_type=current_foundation,
                input_specs=state['inputs'],
                output_specs=state['outputs'],
                max_machines=self.max_machines,
                time_limit=self.time_limit,
                verbose=True,
                nogood_placements=state['nogood_placements']
            )

            # Update nogoods for next continuation attempt
            state['nogood_placements'] = updated_nogoods

            # Track iterations (approximation - solver restarts each time)
            state['total_iterations'] += 1

            if solution and hasattr(solution, 'routing_success') and solution.routing_success:
                # Success!
                self.solution = solution
                self.foundation_type = current_foundation

                # Update the dropdown
                if 'foundation_dropdown' in self._ui_elements:
                    self._ui_elements['foundation_dropdown'].selected_option = current_foundation

                print(f"\n{'='*70}")
                print(f"✓ SUCCESS with {current_foundation} foundation!")
                print(f"Tried {len(state['tried_foundations'])} foundation(s): {', '.join(state['tried_foundations'])}")
                print(f"{'='*70}")

                self._finish_solving(all_failed=False)
            else:
                # Failed - ask user if they want to try next
                print(f"\n⚠ {current_foundation} failed to find valid routing")
                self._prompt_next_foundation()

        except Exception as e:
            print(f"\n✗ Error with {current_foundation}: {e}")
            import traceback
            traceback.print_exc()
            self._prompt_next_foundation()

    def _prompt_next_foundation(self):
        """Prompt user to continue on same foundation or try next."""
        if not self._solving_state:
            return

        state = self._solving_state
        current_idx = state['current_idx']
        progression = state['progression']
        current_foundation = progression[current_idx]

        # Check if there are more foundations to try
        has_next = current_idx + 1 < len(progression)
        next_foundation = progression[current_idx + 1] if has_next else None

        total_time = time.time() - state['start_time']

        # Create custom dialog with Continue and Try Next buttons
        dialog_rect = pygame.Rect((self.width // 2 - 300, self.height // 2 - 150), (600, 300))

        if has_next:
            desc = f"""Foundation '{current_foundation}' could not complete routing.

Total time on this foundation: {total_time:.1f}s
Iterations attempted: {state['total_iterations']}
Nogood constraints: {len(state['nogood_placements'])}

Options:
• Continue: Keep searching this foundation (adds {self.time_limit}s more)
• Try Next: Move to '{next_foundation}'

Tried: {', '.join(state['tried_foundations'])}
Remaining: {len(progression) - current_idx - 1} foundation(s)"""
        else:
            desc = f"""Foundation '{current_foundation}' could not complete routing.

Total time: {total_time:.1f}s
Iterations: {state['total_iterations']}

This is the last foundation. You can:
• Continue: Keep searching (adds {self.time_limit}s more)
• Cancel: Stop and show best attempt"""

        # Create window manually to add custom button
        self._continue_dialog = pygame_gui.elements.UIWindow(
            rect=dialog_rect,
            manager=self._manager,
            window_display_title="Foundation Search Failed"
        )

        # Add message
        msg_rect = pygame.Rect(10, 10, 580, 200)
        pygame_gui.elements.UITextBox(
            html_text=f"<font face='monospace' size=3>{desc}</font>",
            relative_rect=msg_rect,
            manager=self._manager,
            container=self._continue_dialog
        )

        # Add Continue button
        continue_btn_rect = pygame.Rect(10, 220, 280, 40)
        continue_btn = pygame_gui.elements.UIButton(
            relative_rect=continue_btn_rect,
            text="Continue This Foundation",
            manager=self._manager,
            container=self._continue_dialog
        )

        # Add Try Next button (or Cancel if last foundation)
        if has_next:
            next_btn_rect = pygame.Rect(300, 220, 280, 40)
            dialog = pygame_gui.windows.UIConfirmationDialog(
                rect=pygame.Rect((self.width // 2 - 250, self.height // 2 - 100), (500, 200)),
                manager=self._manager,
                window_title="Try Next Foundation?",
                action_long_desc=f"Move to next foundation: '{next_foundation}'?",
                action_short_name="Try Next",
                blocking=False
            )
            state['dialog'] = dialog

    def _continue_solving(self):
        """Continue solving with the next foundation."""
        if not self._solving_state:
            return

        self._solving_state['current_idx'] += 1
        self._try_current_foundation(is_continuation=False)

    def _continue_current_foundation(self):
        """Continue solving on the same foundation with existing constraints."""
        if not self._solving_state:
            return

        print(f"\n{'='*70}")
        print(f"CONTINUING SEARCH ON CURRENT FOUNDATION")
        print(f"{'='*70}")
        sys.stdout.flush()

        # Continue on same foundation, keeping nogood constraints
        self._try_current_foundation(is_continuation=True)

    def _finish_solving(self, all_failed=False):
        """Finish the solving process and display results."""
        if not self._solving_state:
            return

        state = self._solving_state
        tried_foundations = state['tried_foundations']

        if all_failed:
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
                result_text = f"""<font face='monospace' size=3><b>⚠ ALL FOUNDATIONS EXHAUSTED</b>

The solver tried {len(tried_foundations)} foundation size(s) but couldn't complete routing.

<b>Foundations tried:</b> {', '.join(tried_foundations)}
<b>Machines needed:</b> {machines}
<b>Best attempt:</b> {belts} belts placed (incomplete routing)

<b>Problem:</b> The task is too complex for the foundations tried.

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
            self._show_results("ERROR: No solution found. Try a larger foundation or increase timeout.")
            print("\n✗ No solution found.")

        # Clear solving state
        self._solving_state = None

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

        # Draw custom visuals first (before UI elements)
        self._draw_combined_visual()

        # Draw UI on top so dropdowns and dialogs appear above visualization
        self._manager.draw_ui(self._screen)

        pygame.display.flip()

    def _draw_combined_visual(self):
        """Draw combined foundation + I/O port visualization with grid."""
        if not self._foundation_visual_rect:
            return

        from ..evolution.foundation_config import FOUNDATION_SPECS, Side

        # Get foundation spec
        spec = FOUNDATION_SPECS.get(self.foundation_type)
        if not spec:
            return

        rect = self._foundation_visual_rect

        # Draw background
        pygame.draw.rect(self._screen, (40, 40, 40), rect)
        pygame.draw.rect(self._screen, (100, 100, 100), rect, 2)

        # Title
        if hasattr(pygame, 'font'):
            font = pygame.font.SysFont('monospace', 14)
            bold_font = pygame.font.SysFont('monospace', 14, bold=True)

            title = bold_font.render(f"{self.foundation_type} Foundation", True, (220, 220, 220))
            self._screen.blit(title, (rect.x + 5, rect.y + 5))

        # Calculate grid size - use most of the space
        grid_margin = 40
        available_width = rect.width - grid_margin * 2
        available_height = rect.height - grid_margin - 25  # Space for title

        # Each unit is divided into a 14x14 internal grid
        max_dim = max(spec.units_x, spec.units_y)
        cell_size = min(available_width // max_dim, available_height // max_dim)

        # Center the grid
        start_x = rect.x + (rect.width - spec.units_x * cell_size) // 2
        start_y = rect.y + 30 + (rect.height - 30 - spec.units_y * cell_size) // 2

        # Draw foundation units
        if spec.present_cells:
            # Irregular foundation
            cells_set = set(spec.present_cells)
            for x, y in spec.present_cells:
                cell_rect = pygame.Rect(
                    start_x + x * cell_size,
                    start_y + y * cell_size,
                    cell_size,
                    cell_size
                )
                # Fill
                pygame.draw.rect(self._screen, (70, 130, 200), cell_rect)
                # Border
                pygame.draw.rect(self._screen, (100, 160, 240), cell_rect, 2)

                # Draw internal grid (14x14)
                self._draw_internal_grid(cell_rect, 14, (90, 140, 210))
        else:
            # Regular rectangular foundation
            for x in range(spec.units_x):
                for y in range(spec.units_y):
                    cell_rect = pygame.Rect(
                        start_x + x * cell_size,
                        start_y + y * cell_size,
                        cell_size,
                        cell_size
                    )
                    # Fill
                    pygame.draw.rect(self._screen, (70, 130, 200), cell_rect)
                    # Border
                    pygame.draw.rect(self._screen, (100, 160, 240), cell_rect, 2)

                    # Draw internal grid (14x14)
                    self._draw_internal_grid(cell_rect, 14, (90, 140, 210))

        # Now draw I/O ports on top
        self._draw_io_ports_on_foundation(spec, start_x, start_y, cell_size)

    def _draw_internal_grid(self, cell_rect, divisions, color):
        """Draw internal grid lines within a cell."""
        cell_size = cell_rect.width
        grid_spacing = cell_size / divisions

        # Only draw grid if cells are large enough
        if cell_size < 50:
            return

        for i in range(1, divisions):
            # Vertical lines
            x = cell_rect.x + int(i * grid_spacing)
            pygame.draw.line(self._screen, color,
                           (x, cell_rect.y),
                           (x, cell_rect.y + cell_rect.height), 1)
            # Horizontal lines
            y = cell_rect.y + int(i * grid_spacing)
            pygame.draw.line(self._screen, color,
                           (cell_rect.x, y),
                           (cell_rect.x + cell_rect.width, y), 1)

    def _draw_io_ports_on_foundation(self, spec, start_x, start_y, cell_size):
        """Draw I/O port markers on the foundation."""
        from ..evolution.foundation_config import Side

        # Read current text from UI elements
        current_inputs_text = self.inputs_text
        current_outputs_text = self.outputs_text

        if 'inputs_text' in self._ui_elements and self._ui_elements['inputs_text']:
            try:
                current_inputs_text = self._ui_elements['inputs_text'].get_text()
            except:
                pass

        if 'outputs_text' in self._ui_elements and self._ui_elements['outputs_text']:
            try:
                current_outputs_text = self._ui_elements['outputs_text'].get_text()
            except:
                pass

        # Parse inputs
        inputs = []
        for line in current_inputs_text.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        inputs.append((parts[0].strip(), int(parts[1]), int(parts[2])))
                    except (ValueError, IndexError):
                        pass

        # Parse outputs
        outputs = []
        for line in current_outputs_text.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        outputs.append((parts[0].strip(), int(parts[1]), int(parts[2])))
                    except (ValueError, IndexError):
                        pass

        # Draw input ports (green)
        for side, pos, floor in inputs:
            self._draw_port_marker_enhanced(side, pos, spec, start_x, start_y, cell_size, (100, 255, 100))

        # Draw output ports (red)
        for side, pos, floor in outputs:
            self._draw_port_marker_enhanced(side, pos, spec, start_x, start_y, cell_size, (255, 100, 100))

        # Draw legend
        if hasattr(pygame, 'font'):
            font = pygame.font.SysFont('monospace', 12)
            legend_y = start_y + spec.units_y * cell_size + 10

            # Input count
            pygame.draw.circle(self._screen, (100, 255, 100),
                             (start_x + 10, legend_y), 5)
            label = font.render(f"Inputs: {len(inputs)}", True, (100, 255, 100))
            self._screen.blit(label, (start_x + 20, legend_y - 8))

            # Output count
            pygame.draw.circle(self._screen, (255, 100, 100),
                             (start_x + 150, legend_y), 5)
            label = font.render(f"Outputs: {len(outputs)}", True, (255, 100, 100))
            self._screen.blit(label, (start_x + 160, legend_y - 8))

    def _draw_port_marker_enhanced(self, side, pos, spec, start_x, start_y, cell_size, color):
        """Draw a port marker on the foundation edge (enhanced version)."""
        from ..evolution.foundation_config import Side

        ports_per_side = spec.ports_per_side
        marker_size = 8  # Larger markers

        if side.upper() == 'N':
            total_ports = ports_per_side[Side.NORTH]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x + ((pos + 0.5) / total_ports) * spec.units_x * cell_size
                port_y = start_y - marker_size - 2
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)
                pygame.draw.circle(self._screen, (255, 255, 255), (int(port_x), int(port_y)), marker_size, 2)

        elif side.upper() == 'S':
            total_ports = ports_per_side[Side.SOUTH]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x + ((pos + 0.5) / total_ports) * spec.units_x * cell_size
                port_y = start_y + spec.units_y * cell_size + marker_size + 2
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)
                pygame.draw.circle(self._screen, (255, 255, 255), (int(port_x), int(port_y)), marker_size, 2)

        elif side.upper() == 'E':
            total_ports = ports_per_side[Side.EAST]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x + spec.units_x * cell_size + marker_size + 2
                port_y = start_y + ((pos + 0.5) / total_ports) * spec.units_y * cell_size
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)
                pygame.draw.circle(self._screen, (255, 255, 255), (int(port_x), int(port_y)), marker_size, 2)

        elif side.upper() == 'W':
            total_ports = ports_per_side[Side.WEST]
            if total_ports > 0 and pos < total_ports:
                port_x = start_x - marker_size - 2
                port_y = start_y + ((pos + 0.5) / total_ports) * spec.units_y * cell_size
                pygame.draw.circle(self._screen, color, (int(port_x), int(port_y)), marker_size)
                pygame.draw.circle(self._screen, (255, 255, 255), (int(port_x), int(port_y)), marker_size, 2)


def main():
    """Run the CP-SAT solver GUI."""
    app = CPSATSolverApp()
    app.run()


if __name__ == "__main__":
    main()
