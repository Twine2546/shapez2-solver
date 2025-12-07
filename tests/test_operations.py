"""Tests for the operations module."""

import pytest
from shapez2_solver.shapes.shape import Shape, Color
from shapez2_solver.shapes.parser import ShapeCodeParser
from shapez2_solver.operations.cutter import CutOperation, HalfDestroyerOperation, SwapperOperation
from shapez2_solver.operations.rotator import RotateOperation
from shapez2_solver.operations.stacker import StackOperation, UnstackOperation
from shapez2_solver.operations.painter import PaintOperation


class TestHalfDestroyer:
    """Tests for HalfDestroyerOperation."""

    def test_destroy_west_half(self):
        shape = Shape.from_code("CuCuCuCu")
        op = HalfDestroyerOperation()
        result = op.execute(shape)

        assert result[0] is not None
        # East half should remain (parts 0 and 3)
        assert result[0].layers[0].parts[0].shape_type.value == "C"
        assert result[0].layers[0].parts[3].shape_type.value == "C"
        assert result[0].layers[0].parts[1].is_empty()
        assert result[0].layers[0].parts[2].is_empty()

    def test_destroy_empty_input(self):
        op = HalfDestroyerOperation()
        result = op.execute(None)
        assert result[0] is None


class TestCutter:
    """Tests for CutOperation."""

    def test_cut_shape(self):
        shape = Shape.from_code("CuCuCuCu")
        op = CutOperation()
        east, west = op.execute(shape)

        assert east is not None
        assert west is not None

    def test_cut_empty_input(self):
        op = CutOperation()
        east, west = op.execute(None)
        assert east is None
        assert west is None


class TestSwapper:
    """Tests for SwapperOperation."""

    def test_swap_halves(self):
        shape1 = Shape.from_code("CrCrCrCr")
        shape2 = Shape.from_code("RgRgRgRg")
        op = SwapperOperation()

        result1, result2 = op.execute(shape1, shape2)

        assert result1 is not None
        assert result2 is not None

    def test_swap_null_input(self):
        shape1 = Shape.from_code("CrCrCrCr")
        op = SwapperOperation()

        result1, result2 = op.execute(shape1, None)
        assert result1 is None
        assert result2 is None


class TestRotator:
    """Tests for RotateOperation."""

    def test_rotate_90(self):
        shape = Shape.from_code("CrRgSbWy")
        op = RotateOperation(steps=1)
        result = op.execute(shape)

        assert result[0] is not None
        # After 90° rotation, parts shift
        assert result[0].to_code() == "WyCrRgSb"

    def test_rotate_180(self):
        shape = Shape.from_code("CrRgSbWy")
        op = RotateOperation(steps=2)
        result = op.execute(shape)

        assert result[0] is not None
        assert result[0].to_code() == "SbWyCrRg"

    def test_rotate_empty(self):
        op = RotateOperation()
        result = op.execute(None)
        assert result[0] is None


class TestStacker:
    """Tests for StackOperation."""

    def test_stack_shapes(self):
        bottom = Shape.from_code("CuCuCuCu")
        top = Shape.from_code("RrRrRrRr")
        op = StackOperation()

        result = op.execute(bottom, top)

        assert result[0] is not None
        assert result[0].num_layers == 2

    def test_stack_with_null(self):
        shape = Shape.from_code("CuCuCuCu")
        op = StackOperation()

        result = op.execute(shape, None)
        assert result[0] is None

    def test_stack_empty_bottom(self):
        top = Shape.from_code("RrRrRrRr")
        op = StackOperation()

        result = op.execute(Shape.empty(), top)
        assert result[0] is not None


class TestUnstacker:
    """Tests for UnstackOperation."""

    def test_unstack_multilayer(self):
        shape = Shape.from_code("CuCuCuCu:RrRrRrRr")
        op = UnstackOperation()

        top, remaining = op.execute(shape)

        assert top is not None
        assert remaining is not None
        assert top.num_layers == 1
        assert remaining.num_layers == 1

    def test_unstack_single_layer(self):
        shape = Shape.from_code("CuCuCuCu")
        op = UnstackOperation()

        top, remaining = op.execute(shape)

        assert top is not None
        assert remaining is None

    def test_unstack_empty(self):
        op = UnstackOperation()
        top, remaining = op.execute(None)
        assert top is None
        assert remaining is None


class TestPainter:
    """Tests for PaintOperation."""

    def test_paint_top_layer(self):
        shape = Shape.from_code("CuCuCuCu")
        op = PaintOperation()

        result = op.execute(shape, Color.RED)

        assert result[0] is not None
        # All parts of top layer should be red
        for part in result[0].layers[-1].parts:
            if not part.is_empty():
                assert part.color == Color.RED

    def test_paint_null_shape(self):
        op = PaintOperation()
        result = op.execute(None, Color.RED)
        assert result[0] is None

    def test_paint_null_color(self):
        shape = Shape.from_code("CuCuCuCu")
        op = PaintOperation()
        result = op.execute(shape, None)
        assert result[0] is None
