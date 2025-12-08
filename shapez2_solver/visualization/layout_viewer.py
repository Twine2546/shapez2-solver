"""
GUI Layout Viewer for Shapez 2 Evolution Solutions.

Displays foundation layouts with buildings, belts, and ports using Tkinter.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..evolution.foundation_config import FoundationConfig, FoundationSpec, Side, PortType
from ..evolution.foundation_evolution import Candidate, PlacedBuilding, FoundationEvolution
from ..blueprint.building_types import BuildingType, Rotation, BUILDING_SPECS


# Color schemes for buildings
BUILDING_COLORS = {
    # Operations
    BuildingType.ROTATOR_CW: "#4CAF50",      # Green
    BuildingType.ROTATOR_CCW: "#8BC34A",     # Light green
    BuildingType.ROTATOR_180: "#CDDC39",     # Lime
    BuildingType.CUTTER: "#2196F3",          # Blue
    BuildingType.CUTTER_MIRRORED: "#03A9F4", # Light blue
    BuildingType.HALF_CUTTER: "#00BCD4",     # Cyan
    BuildingType.SWAPPER: "#9C27B0",         # Purple
    BuildingType.STACKER: "#FF9800",         # Orange
    BuildingType.UNSTACKER: "#FF5722",       # Deep orange
    BuildingType.PIN_PUSHER: "#795548",      # Brown
    BuildingType.TRASH: "#607D8B",           # Blue grey

    # Belts
    BuildingType.BELT_FORWARD: "#FFEB3B",    # Yellow
    BuildingType.BELT_LEFT: "#FFC107",       # Amber
    BuildingType.BELT_RIGHT: "#FFD54F",      # Light amber
    BuildingType.LIFT_UP: "#E91E63",         # Pink
    BuildingType.LIFT_DOWN: "#F06292",       # Light pink

    # Other
    BuildingType.SPLITTER: "#009688",        # Teal
    BuildingType.MERGER: "#00796B",          # Dark teal
}

BUILDING_SYMBOLS = {
    BuildingType.ROTATOR_CW: "R↻",
    BuildingType.ROTATOR_CCW: "R↺",
    BuildingType.ROTATOR_180: "R⟳",
    BuildingType.CUTTER: "CUT",
    BuildingType.CUTTER_MIRRORED: "CUTm",
    BuildingType.HALF_CUTTER: "½CUT",
    BuildingType.SWAPPER: "SWAP",
    BuildingType.STACKER: "STK",
    BuildingType.UNSTACKER: "USTK",
    BuildingType.PIN_PUSHER: "PIN",
    BuildingType.TRASH: "TRS",
    BuildingType.BELT_FORWARD: "→",
    BuildingType.BELT_LEFT: "↰",
    BuildingType.BELT_RIGHT: "↱",
    BuildingType.LIFT_UP: "↑",
    BuildingType.LIFT_DOWN: "↓",
    BuildingType.SPLITTER: "SPL",
    BuildingType.MERGER: "MRG",
}

ROTATION_ARROWS = {
    Rotation.EAST: "→",
    Rotation.SOUTH: "↓",
    Rotation.WEST: "←",
    Rotation.NORTH: "↑",
}


class LayoutViewer:
    """Tkinter-based GUI for viewing foundation layouts."""

    def __init__(self, evolution: FoundationEvolution):
        self.evolution = evolution
        self.config = evolution.config
        self.current_floor = 0
        self.current_solution_idx = 0
        self.cell_size = 20  # Pixels per grid cell

        # Calculate canvas dimensions
        self.canvas_width = self.config.spec.grid_width * self.cell_size + 100
        self.canvas_height = self.config.spec.grid_height * self.cell_size + 100

        self.root = None
        self.canvas = None
        self.info_label = None
        self.floor_var = None
        self.solution_var = None

    def show(self):
        """Display the GUI window."""
        self.root = tk.Tk()
        self.root.title("Shapez 2 Layout Viewer")

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Floor selector
        ttk.Label(control_frame, text="Floor:").grid(row=0, column=0, padx=(0, 5))
        self.floor_var = tk.StringVar(value="0")
        floor_combo = ttk.Combobox(control_frame, textvariable=self.floor_var,
                                   values=[str(i) for i in range(self.config.spec.num_floors)],
                                   width=5, state="readonly")
        floor_combo.grid(row=0, column=1, padx=(0, 20))
        floor_combo.bind("<<ComboboxSelected>>", self._on_floor_change)

        # Solution selector
        ttk.Label(control_frame, text="Solution:").grid(row=0, column=2, padx=(0, 5))
        self.solution_var = tk.StringVar(value="1")
        num_solutions = len(self.evolution.top_solutions) if self.evolution.top_solutions else 1
        solution_combo = ttk.Combobox(control_frame, textvariable=self.solution_var,
                                      values=[str(i+1) for i in range(num_solutions)],
                                      width=5, state="readonly")
        solution_combo.grid(row=0, column=3, padx=(0, 20))
        solution_combo.bind("<<ComboboxSelected>>", self._on_solution_change)

        # Zoom controls
        ttk.Label(control_frame, text="Zoom:").grid(row=0, column=4, padx=(0, 5))
        ttk.Button(control_frame, text="-", width=3,
                   command=lambda: self._zoom(-2)).grid(row=0, column=5)
        ttk.Button(control_frame, text="+", width=3,
                   command=lambda: self._zoom(2)).grid(row=0, column=6)

        # Canvas for drawing
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add scrollbars
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(canvas_frame,
                                width=min(800, self.canvas_width),
                                height=min(600, self.canvas_height),
                                bg="#2d2d2d",
                                xscrollcommand=h_scroll.set,
                                yscrollcommand=v_scroll.set)

        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)

        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure scroll region
        self.canvas.config(scrollregion=(0, 0, self.canvas_width, self.canvas_height))

        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Solution Info", padding="5")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.info_label = ttk.Label(info_frame, text="", justify=tk.LEFT)
        self.info_label.grid(row=0, column=0, sticky=(tk.W,))

        # Legend
        legend_frame = ttk.LabelFrame(main_frame, text="Legend", padding="5")
        legend_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self._create_legend(legend_frame)

        # Mouse bindings for tooltips
        self.canvas.bind("<Motion>", self._on_mouse_move)

        # Initial draw
        self._draw_layout()

        self.root.mainloop()

    def _create_legend(self, parent):
        """Create the color legend."""
        legend_items = [
            ("Input Port", "#00FF00", "I"),
            ("Output Port", "#FF0000", "O"),
            ("Foundation", "#555555", ""),
            ("Rotator", "#4CAF50", "R"),
            ("Cutter", "#2196F3", "C"),
            ("Belt", "#FFEB3B", "→"),
            ("Lift", "#E91E63", "↑"),
        ]

        for i, (name, color, symbol) in enumerate(legend_items):
            frame = ttk.Frame(parent)
            frame.grid(row=0, column=i, padx=5)

            canvas = tk.Canvas(frame, width=20, height=20, bg=color, highlightthickness=1)
            canvas.grid(row=0, column=0)
            if symbol:
                canvas.create_text(10, 10, text=symbol, fill="black", font=("Arial", 8))

            ttk.Label(frame, text=name, font=("Arial", 8)).grid(row=1, column=0)

    def _on_floor_change(self, event):
        """Handle floor selection change."""
        self.current_floor = int(self.floor_var.get())
        self._draw_layout()

    def _on_solution_change(self, event):
        """Handle solution selection change."""
        self.current_solution_idx = int(self.solution_var.get()) - 1
        self._draw_layout()

    def _zoom(self, delta):
        """Zoom in or out."""
        self.cell_size = max(5, min(50, self.cell_size + delta))
        self.canvas_width = self.config.spec.grid_width * self.cell_size + 100
        self.canvas_height = self.config.spec.grid_height * self.cell_size + 100
        self.canvas.config(scrollregion=(0, 0, self.canvas_width, self.canvas_height))
        self._draw_layout()

    def _on_mouse_move(self, event):
        """Show tooltip on mouse hover."""
        # Convert canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Calculate grid position
        margin = 50
        grid_x = int((x - margin) / self.cell_size)
        grid_y = int((y - margin) / self.cell_size)

        # Find building at this position
        if self.evolution.top_solutions and self.current_solution_idx < len(self.evolution.top_solutions):
            candidate = self.evolution.top_solutions[self.current_solution_idx]
            for building in candidate.buildings:
                if (building.x == grid_x and building.y == grid_y and
                    building.floor == self.current_floor):
                    self.root.title(f"Shapez 2 Layout Viewer - {building.building_type.name} at ({grid_x}, {grid_y})")
                    return

        self.root.title(f"Shapez 2 Layout Viewer - ({grid_x}, {grid_y})")

    def _draw_layout(self):
        """Draw the current layout on the canvas."""
        self.canvas.delete("all")

        margin = 50
        grid_w = self.config.spec.grid_width
        grid_h = self.config.spec.grid_height
        cell = self.cell_size

        # Draw grid background
        for x in range(grid_w):
            for y in range(grid_h):
                x1 = margin + x * cell
                y1 = margin + y * cell
                x2 = x1 + cell
                y2 = y1 + cell

                # Foundation area
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                           fill="#3d3d3d", outline="#4d4d4d")

        # Draw grid lines
        for x in range(grid_w + 1):
            x1 = margin + x * cell
            self.canvas.create_line(x1, margin, x1, margin + grid_h * cell,
                                   fill="#4d4d4d", dash=(2, 2))
        for y in range(grid_h + 1):
            y1 = margin + y * cell
            self.canvas.create_line(margin, y1, margin + grid_w * cell, y1,
                                   fill="#4d4d4d", dash=(2, 2))

        # Draw 1x1 unit boundaries (every 14 cells for first unit, then every 20)
        for ux in range(self.config.spec.units_x + 1):
            if ux == 0:
                x = 0
            else:
                x = 14 + (ux - 1) * 20
            if x <= grid_w:
                x1 = margin + x * cell
                self.canvas.create_line(x1, margin, x1, margin + grid_h * cell,
                                       fill="#888888", width=2)

        for uy in range(self.config.spec.units_y + 1):
            if uy == 0:
                y = 0
            else:
                y = 14 + (uy - 1) * 20
            if y <= grid_h:
                y1 = margin + y * cell
                self.canvas.create_line(margin, y1, margin + grid_w * cell, y1,
                                       fill="#888888", width=2)

        # Draw ports
        self._draw_ports(margin, cell)

        # Draw buildings from current solution
        if self.evolution.top_solutions and self.current_solution_idx < len(self.evolution.top_solutions):
            candidate = self.evolution.top_solutions[self.current_solution_idx]
            self._draw_buildings(candidate, margin, cell)
            self._update_info(candidate)
        else:
            self.info_label.config(text="No solutions available")

        # Draw axis labels
        for x in range(0, grid_w, 5):
            self.canvas.create_text(margin + x * cell + cell/2, margin - 10,
                                   text=str(x), fill="white", font=("Arial", 8))
        for y in range(0, grid_h, 5):
            self.canvas.create_text(margin - 15, margin + y * cell + cell/2,
                                   text=str(y), fill="white", font=("Arial", 8))

    def _draw_ports(self, margin, cell):
        """Draw input/output ports."""
        floor = self.current_floor

        for side, pos, f, shape_code in self.config.get_all_inputs():
            if f != floor:
                continue
            gx, gy = self.config.spec.get_port_grid_position(side, pos)

            # Adjust position to be outside the grid
            if side == Side.NORTH:
                gy = -1
            elif side == Side.SOUTH:
                gy = self.config.spec.grid_height
            elif side == Side.WEST:
                gx = -1
            elif side == Side.EAST:
                gx = self.config.spec.grid_width

            x1 = margin + gx * cell
            y1 = margin + gy * cell

            self.canvas.create_rectangle(x1, y1, x1 + cell, y1 + cell,
                                        fill="#00AA00", outline="#00FF00", width=2)
            self.canvas.create_text(x1 + cell/2, y1 + cell/2, text="I",
                                   fill="white", font=("Arial", max(8, cell//3), "bold"))

        for side, pos, f, shape_code in self.config.get_all_outputs():
            if f != floor:
                continue
            gx, gy = self.config.spec.get_port_grid_position(side, pos)

            # Adjust position to be outside the grid
            if side == Side.NORTH:
                gy = -1
            elif side == Side.SOUTH:
                gy = self.config.spec.grid_height
            elif side == Side.WEST:
                gx = -1
            elif side == Side.EAST:
                gx = self.config.spec.grid_width

            x1 = margin + gx * cell
            y1 = margin + gy * cell

            self.canvas.create_rectangle(x1, y1, x1 + cell, y1 + cell,
                                        fill="#AA0000", outline="#FF0000", width=2)
            self.canvas.create_text(x1 + cell/2, y1 + cell/2, text="O",
                                   fill="white", font=("Arial", max(8, cell//3), "bold"))

    def _draw_buildings(self, candidate: Candidate, margin, cell):
        """Draw buildings from a candidate solution."""
        for building in candidate.buildings:
            if building.floor != self.current_floor:
                continue

            x = building.x
            y = building.y

            # Get building spec for size
            spec = BUILDING_SPECS.get(building.building_type)
            w = spec.width if spec else 1
            h = spec.height if spec else 1

            x1 = margin + x * cell
            y1 = margin + y * cell
            x2 = x1 + w * cell
            y2 = y1 + h * cell

            # Get color
            color = BUILDING_COLORS.get(building.building_type, "#888888")

            # Draw building rectangle
            self.canvas.create_rectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1,
                                        fill=color, outline="white", width=1)

            # Draw symbol/text
            symbol = BUILDING_SYMBOLS.get(building.building_type, "?")
            rotation_arrow = ROTATION_ARROWS.get(building.rotation, "")

            text = symbol
            if building.building_type in [BuildingType.BELT_FORWARD,
                                          BuildingType.BELT_LEFT,
                                          BuildingType.BELT_RIGHT]:
                # For belts, show direction
                text = rotation_arrow

            font_size = max(6, min(cell // 2, 12))
            self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                   text=text, fill="black",
                                   font=("Arial", font_size, "bold"))

            # Draw rotation indicator for non-belt buildings
            if building.building_type not in [BuildingType.BELT_FORWARD,
                                               BuildingType.BELT_LEFT,
                                               BuildingType.BELT_RIGHT]:
                self.canvas.create_text(x2 - 5, y1 + 5, text=rotation_arrow,
                                       fill="black", font=("Arial", 6))

    def _update_info(self, candidate: Candidate):
        """Update the info panel."""
        info_lines = [
            f"Fitness: {candidate.fitness:.2f}",
            f"Buildings: {len(candidate.buildings)}",
            f"Floor {self.current_floor} buildings: {sum(1 for b in candidate.buildings if b.floor == self.current_floor)}",
        ]

        # Count building types
        type_counts = {}
        for b in candidate.buildings:
            name = b.building_type.name
            type_counts[name] = type_counts.get(name, 0) + 1

        if type_counts:
            info_lines.append("Building types: " + ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items())))

        # Show outputs
        if candidate.output_shapes:
            info_lines.append("")
            info_lines.append("Outputs:")
            for key, shape in candidate.output_shapes.items():
                expected = self.evolution.expected_outputs.get(key)
                exp_code = expected.to_code() if expected else "?"
                act_code = shape.to_code() if shape else "None"
                match = "✓" if exp_code == act_code else "✗"
                info_lines.append(f"  {key}: {act_code} {match}")

        self.info_label.config(text="\n".join(info_lines))


def show_layout(evolution: FoundationEvolution):
    """Convenience function to show the layout viewer."""
    viewer = LayoutViewer(evolution)
    viewer.show()
