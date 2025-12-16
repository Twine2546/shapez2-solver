"""Visualization module for rendering shapes and solutions."""

from .shape_renderer import ShapeRenderer
from .solution_display import SolutionDisplay, display_solution, display_solution_ascii, display_solution_compact


__all__ = [
    "ShapeRenderer",
    "SolutionDisplay",
    "display_solution",
    "display_solution_ascii",
    "display_solution_compact",
]
