"""Tests for the simulator module."""

import pytest
from shapez2_solver.shapes.shape import Shape
from shapez2_solver.shapes.parser import ShapeCodeParser
from shapez2_solver.foundations.foundation import Foundation, Port, PortDirection, FoundationType
from shapez2_solver.operations.rotator import RotateOperation
from shapez2_solver.operations.stacker import StackOperation
from shapez2_solver.simulator.design import Design, OperationNode
from shapez2_solver.simulator.simulator import Simulator


class TestDesign:
    """Tests for Design."""

    def test_create_empty_design(self):
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)

        assert len(design.operations) == 0
        assert len(design.inputs) == 0
        assert len(design.outputs) == 0
        assert len(design.connections) == 0

    def test_add_operation(self):
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)

        op_id = design.add_operation(RotateOperation())

        assert len(design.operations) == 1
        assert design.get_node(op_id) is not None

    def test_add_input_output(self):
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)

        in_port = Port(PortDirection.WEST, floor=0, position=0, is_input=True)
        out_port = Port(PortDirection.EAST, floor=0, position=0, is_input=False)

        in_id = design.add_input(in_port)
        out_id = design.add_output(out_port)

        assert len(design.inputs) == 1
        assert len(design.outputs) == 1

    def test_connect_nodes(self):
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)

        in_port = Port(PortDirection.WEST, floor=0, position=0, is_input=True)
        in_id = design.add_input(in_port)

        op_id = design.add_operation(RotateOperation())

        design.connect(in_id, 0, op_id, 0)

        assert len(design.connections) == 1
        incoming = design.get_incoming_connections(op_id)
        assert len(incoming) == 1

    def test_copy(self):
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)
        design.add_operation(RotateOperation())

        copy = design.copy()

        assert len(copy.operations) == len(design.operations)
        assert copy is not design


class TestSimulator:
    """Tests for Simulator."""

    def test_simple_passthrough(self):
        """Test a design that just passes input to output."""
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)

        in_port = Port(PortDirection.WEST, floor=0, position=0, is_input=True)
        out_port = Port(PortDirection.EAST, floor=0, position=0, is_input=False)

        in_id = design.add_input(in_port)
        out_id = design.add_output(out_port)

        design.connect(in_id, 0, out_id, 0)

        simulator = Simulator(design)
        input_shape = ShapeCodeParser.parse("CuCuCuCu")

        result = simulator.execute({in_id: input_shape})

        assert result.success
        assert result.get_output(out_id) is not None
        assert result.get_output(out_id).to_code() == "CuCuCuCu"

    def test_single_rotation(self):
        """Test a design with a single rotation operation."""
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)

        in_port = Port(PortDirection.WEST, floor=0, position=0, is_input=True)
        out_port = Port(PortDirection.EAST, floor=0, position=0, is_input=False)

        in_id = design.add_input(in_port)
        out_id = design.add_output(out_port)
        op_id = design.add_operation(RotateOperation(steps=1))

        design.connect(in_id, 0, op_id, 0)
        design.connect(op_id, 0, out_id, 0)

        simulator = Simulator(design)
        input_shape = ShapeCodeParser.parse("CrRgSbWy")

        result = simulator.execute({in_id: input_shape})

        assert result.success
        output = result.get_output(out_id)
        assert output is not None
        # After 90° rotation
        assert output.to_code() == "WyCrRgSb"

    def test_stacking_operation(self):
        """Test a design with stacking."""
        foundation = Foundation(FoundationType.SIZE_1X1, width=1, height=1)
        design = Design(foundation)

        in1_port = Port(PortDirection.WEST, floor=0, position=0, is_input=True)
        in2_port = Port(PortDirection.WEST, floor=1, position=0, is_input=True)
        out_port = Port(PortDirection.EAST, floor=0, position=0, is_input=False)

        in1_id = design.add_input(in1_port)
        in2_id = design.add_input(in2_port)
        out_id = design.add_output(out_port)
        op_id = design.add_operation(StackOperation())

        design.connect(in1_id, 0, op_id, 0)  # Bottom
        design.connect(in2_id, 0, op_id, 1)  # Top
        design.connect(op_id, 0, out_id, 0)

        simulator = Simulator(design)
        bottom_shape = ShapeCodeParser.parse("CuCuCuCu")
        top_shape = ShapeCodeParser.parse("RrRrRrRr")

        result = simulator.execute({
            in1_id: bottom_shape,
            in2_id: top_shape
        })

        assert result.success
        output = result.get_output(out_id)
        assert output is not None
        assert output.num_layers == 2
