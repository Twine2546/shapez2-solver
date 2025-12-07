"""Shape validation utilities implementing Shapez 2 physics rules."""

from typing import List, Set, Tuple, Optional
from .shape import Shape, ShapeLayer, ShapePart, ShapeType


class ShapeValidator:
    """Validator for Shapez 2 shape physics rules."""

    # Adjacency map for 4-quadrant shapes (index -> adjacent indices)
    # Quadrants: 0=NE, 1=NW, 2=SW, 3=SE (clockwise from top-right)
    ADJACENCY_4 = {
        0: [1, 3],  # NE adjacent to NW and SE
        1: [0, 2],  # NW adjacent to NE and SW
        2: [1, 3],  # SW adjacent to NW and SE
        3: [0, 2],  # SE adjacent to NE and SW
    }

    @classmethod
    def apply_gravity(cls, shape: Shape) -> Shape:
        """
        Apply gravity rules to a shape.

        In Shapez 2, floating pieces fall down. This method:
        1. For each layer, finds connected components
        2. Removes pieces that aren't connected to the layer below
        3. Empty layers are removed

        Args:
            shape: The shape to apply gravity to

        Returns:
            A new shape with gravity applied
        """
        if shape.is_empty():
            return Shape.empty()

        new_layers = []

        for layer_idx, layer in enumerate(shape.layers):
            if layer_idx == 0:
                # Bottom layer - just keep non-empty parts
                new_layer = cls._remove_floating_in_layer(layer)
                if not new_layer.is_empty():
                    new_layers.append(new_layer)
            else:
                # Upper layers - need support from below
                support_layer = new_layers[-1] if new_layers else None
                new_layer = cls._apply_layer_gravity(layer, support_layer)
                if not new_layer.is_empty():
                    new_layers.append(new_layer)

        return Shape(new_layers)

    @classmethod
    def _remove_floating_in_layer(cls, layer: ShapeLayer) -> ShapeLayer:
        """Remove floating parts within a layer (parts not connected to anything)."""
        # Find all connected components
        components = cls._find_connected_components(layer)

        # If there are no parts or only one component, nothing is floating
        if len(components) <= 1:
            return layer.copy()

        # Keep the largest component (or all if they're grounded differently)
        # In Shapez 2, isolated single parts without connection fall
        new_parts = []
        for i, part in enumerate(layer.parts):
            if part.is_empty() or part.is_pin():
                new_parts.append(part.copy())
            else:
                # Check if this part is in a connected component of size > 1
                # or is connected to something
                component = cls._get_component_for_part(i, components)
                if len(component) > 1 or cls._has_support_in_layer(i, layer):
                    new_parts.append(part.copy())
                else:
                    new_parts.append(ShapePart.empty())

        return ShapeLayer(new_parts)

    @classmethod
    def _apply_layer_gravity(
        cls, layer: ShapeLayer, support_layer: Optional[ShapeLayer]
    ) -> ShapeLayer:
        """Apply gravity to a layer based on support from below."""
        if support_layer is None:
            return ShapeLayer.empty(layer.num_parts)

        # Find which parts have support from the layer below
        supported = set()
        for i, part in enumerate(layer.parts):
            if not part.is_empty() and not part.is_pin():
                # A part is supported if the same position below has a non-empty part
                support_part = support_layer.get_part(i)
                if not support_part.is_empty():
                    supported.add(i)

        # Expand supported set to include connected parts
        supported = cls._expand_connected(layer, supported)

        # Create new layer with only supported parts
        new_parts = []
        for i, part in enumerate(layer.parts):
            if i in supported or part.is_empty():
                new_parts.append(part.copy())
            else:
                new_parts.append(ShapePart.empty())

        return ShapeLayer(new_parts)

    @classmethod
    def _find_connected_components(cls, layer: ShapeLayer) -> List[Set[int]]:
        """Find connected components in a layer."""
        visited = set()
        components = []

        for i in range(layer.num_parts):
            if i not in visited and not layer.parts[i].is_empty():
                component = cls._bfs_component(layer, i, visited)
                if component:
                    components.append(component)

        return components

    @classmethod
    def _bfs_component(
        cls, layer: ShapeLayer, start: int, visited: Set[int]
    ) -> Set[int]:
        """BFS to find a connected component."""
        component = set()
        queue = [start]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            part = layer.get_part(current)
            if part.is_empty() or part.is_pin():
                continue

            visited.add(current)
            component.add(current)

            # Add adjacent parts
            adjacency = cls.ADJACENCY_4 if layer.num_parts == 4 else range(layer.num_parts)
            if layer.num_parts == 4:
                for adj in cls.ADJACENCY_4.get(current, []):
                    if adj not in visited:
                        adj_part = layer.get_part(adj)
                        if not adj_part.is_empty() and not adj_part.is_pin():
                            queue.append(adj)

        return component

    @classmethod
    def _get_component_for_part(
        cls, part_idx: int, components: List[Set[int]]
    ) -> Set[int]:
        """Get the component containing a specific part."""
        for component in components:
            if part_idx in component:
                return component
        return set()

    @classmethod
    def _has_support_in_layer(cls, part_idx: int, layer: ShapeLayer) -> bool:
        """Check if a part has any adjacent non-empty, non-pin parts."""
        if layer.num_parts == 4:
            for adj in cls.ADJACENCY_4.get(part_idx, []):
                adj_part = layer.get_part(adj)
                if not adj_part.is_empty() and not adj_part.is_pin():
                    return True
        return False

    @classmethod
    def _expand_connected(cls, layer: ShapeLayer, initial: Set[int]) -> Set[int]:
        """Expand a set of indices to include all connected parts."""
        expanded = set(initial)
        changed = True

        while changed:
            changed = False
            for i in list(expanded):
                if layer.num_parts == 4:
                    for adj in cls.ADJACENCY_4.get(i, []):
                        if adj not in expanded:
                            adj_part = layer.get_part(adj)
                            if not adj_part.is_empty() and not adj_part.is_pin():
                                expanded.add(adj)
                                changed = True

        return expanded

    @classmethod
    def validate_shape(cls, shape: Shape) -> Tuple[bool, List[str]]:
        """
        Validate a shape against all Shapez 2 rules.

        Returns:
            A tuple of (is_valid, list_of_issues)
        """
        issues = []

        if shape.is_empty():
            return True, []

        # Check layer count (1-5 in insane, 1-4 otherwise)
        if shape.num_layers > 5:
            issues.append(f"Too many layers: {shape.num_layers} (max 5)")

        # Check each layer
        for i, layer in enumerate(shape.layers):
            # Check part count
            if layer.num_parts not in (4, 6):
                issues.append(f"Layer {i}: Invalid part count {layer.num_parts}")

            # Check for floating parts
            gravity_applied = cls.apply_gravity(shape)
            if gravity_applied.to_code() != shape.to_code():
                issues.append("Shape has floating parts that would fall")

        return len(issues) == 0, issues

    @classmethod
    def is_valid(cls, shape: Shape) -> bool:
        """Quick check if a shape is valid."""
        valid, _ = cls.validate_shape(shape)
        return valid
