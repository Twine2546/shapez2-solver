"""Visualization module for rendering shapes and solution layouts."""

from .shape_renderer import ShapeRenderer
from .solution_display import SolutionDisplay, display_solution, display_solution_ascii, display_solution_compact

# Layout viewer requires tkinter, which may not be available in all environments
try:
    from .layout_viewer import LayoutViewer, show_layout
    _HAS_TKINTER = True
except ImportError:
    LayoutViewer = None
    _HAS_TKINTER = False

    def show_layout(evolution):
        """Show layout viewer - requires tkinter."""
        print("Error: GUI viewer requires tkinter which is not installed.")
        print("Install tkinter with: apt-get install python3-tk (Debian/Ubuntu)")
        print("or: yum install python3-tkinter (RHEL/CentOS)")

# Pygame-based layout viewer
try:
    from .pygame_layout_viewer import PygameLayoutViewer, show_layout_pygame
    _HAS_PYGAME = True
except ImportError:
    PygameLayoutViewer = None
    _HAS_PYGAME = False

    def show_layout_pygame(evolution):
        """Show pygame layout viewer - requires pygame."""
        print("Error: Pygame layout viewer requires pygame which is not installed.")
        print("Install pygame with: pip install pygame-ce")

__all__ = [
    "ShapeRenderer",
    "SolutionDisplay",
    "display_solution",
    "display_solution_ascii",
    "display_solution_compact",
    "LayoutViewer",
    "show_layout",
    "PygameLayoutViewer",
    "show_layout_pygame",
]
