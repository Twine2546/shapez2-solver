"""Tests for the shapes module."""

import pytest
from shapez2_solver.shapes.shape import Shape, ShapeLayer, ShapePart, ShapeType, Color
from shapez2_solver.shapes.parser import ShapeCodeParser
from shapez2_solver.shapes.encoder import ShapeCodeEncoder
from shapez2_solver.shapes.validator import ShapeValidator


class TestShapePart:
    """Tests for ShapePart."""

    def test_create_empty(self):
        part = ShapePart.empty()
        assert part.is_empty()
        assert part.shape_type == ShapeType.EMPTY
        assert part.color == Color.NONE

    def test_from_code(self):
        part = ShapePart.from_code("Cr")
        assert part.shape_type == ShapeType.CIRCLE
        assert part.color == Color.RED

    def test_to_code(self):
        part = ShapePart(ShapeType.SQUARE, Color.GREEN)
        assert part.to_code() == "Rg"

    def test_non_colorable_shapes(self):
        # Pin should not have a color
        part = ShapePart(ShapeType.PIN, Color.RED)
        assert part.color == Color.NONE


class TestShapeLayer:
    """Tests for ShapeLayer."""

    def test_create_empty(self):
        layer = ShapeLayer.empty()
        assert layer.is_empty()
        assert layer.num_parts == 4

    def test_from_code(self):
        layer = ShapeLayer.from_code("CrRgSbWy")
        assert layer.num_parts == 4
        assert layer.parts[0].shape_type == ShapeType.CIRCLE
        assert layer.parts[1].shape_type == ShapeType.SQUARE
        assert layer.parts[2].shape_type == ShapeType.STAR
        assert layer.parts[3].shape_type == ShapeType.DIAMOND

    def test_rotate(self):
        layer = ShapeLayer.from_code("CrRgSbWy")
        rotated = layer.rotate(1)
        # After rotating 1 step clockwise, last becomes first
        assert rotated.parts[0].shape_type == ShapeType.DIAMOND
        assert rotated.parts[1].shape_type == ShapeType.CIRCLE


class TestShape:
    """Tests for Shape."""

    def test_empty_shape(self):
        shape = Shape.empty()
        assert shape.is_empty()
        assert shape.num_layers == 0

    def test_from_code_single_layer(self):
        shape = Shape.from_code("CrCrCrCr")
        assert shape.num_layers == 1
        assert not shape.is_empty()

    def test_from_code_multiple_layers(self):
        shape = Shape.from_code("CrCrCrCr:RgRgRgRg")
        assert shape.num_layers == 2

    def test_to_code(self):
        shape = Shape.from_code("CuCuCuCu:RrRrRrRr")
        assert shape.to_code() == "CuCuCuCu:RrRrRrRr"

    def test_rotate(self):
        shape = Shape.from_code("CrRgSbWy")
        rotated = shape.rotate(1)
        assert rotated.to_code() == "WyCrRgSb"


class TestShapeCodeParser:
    """Tests for ShapeCodeParser."""

    def test_parse_simple(self):
        shape = ShapeCodeParser.parse("CuCuCuCu")
        assert shape.num_layers == 1
        assert all(p.color == Color.UNCOLORED for p in shape.layers[0].parts)

    def test_parse_empty(self):
        shape = ShapeCodeParser.parse("--")
        assert shape.is_empty()

    def test_parse_with_pins(self):
        shape = ShapeCodeParser.parse("CrP-CrP-")
        assert shape.layers[0].parts[1].is_pin()

    def test_validate_valid_code(self):
        valid, error = ShapeCodeParser.validate("CrCrCrCr")
        assert valid
        assert error is None

    def test_validate_invalid_code(self):
        valid, error = ShapeCodeParser.validate("XxXxXxXx")
        assert not valid
        assert error is not None


class TestShapeCodeEncoder:
    """Tests for ShapeCodeEncoder."""

    def test_encode_simple(self):
        shape = Shape.from_code("CuCuCuCu")
        encoded = ShapeCodeEncoder.encode(shape)
        assert encoded == "CuCuCuCu"

    def test_encode_empty(self):
        shape = Shape.empty()
        encoded = ShapeCodeEncoder.encode(shape)
        assert encoded == "--"

    def test_visual_grid(self):
        shape = Shape.from_code("CrRgSbWy")
        grid = ShapeCodeEncoder.to_visual_grid(shape)
        assert len(grid) == 1  # One layer
        assert len(grid[0]) == 2  # 2 rows
        assert len(grid[0][0]) == 2  # 2 columns


class TestShapeValidator:
    """Tests for ShapeValidator."""

    def test_apply_gravity_stable(self):
        # A full layer should remain unchanged
        shape = Shape.from_code("CuCuCuCu")
        result = ShapeValidator.apply_gravity(shape)
        assert result.to_code() == shape.to_code()

    def test_validate_valid_shape(self):
        shape = Shape.from_code("CuCuCuCu")
        valid, issues = ShapeValidator.validate_shape(shape)
        assert valid
        assert len(issues) == 0

    def test_is_valid(self):
        shape = Shape.from_code("CuCuCuCu:RrRrRrRr")
        assert ShapeValidator.is_valid(shape)
