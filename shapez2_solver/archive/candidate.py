"""Candidate solution representation."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..simulator.design import Design


@dataclass
class Candidate:
    """A candidate solution in the evolutionary algorithm."""
    design: Design
    fitness: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Candidate") -> bool:
        """Compare by fitness (higher is better)."""
        return self.fitness < other.fitness

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Candidate):
            return False
        return self.fitness == other.fitness

    def __hash__(self) -> int:
        return id(self)

    def copy(self) -> "Candidate":
        """Create a copy of this candidate."""
        return Candidate(
            design=self.design.copy(),
            fitness=self.fitness,
            generation=self.generation,
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        return f"Candidate(fitness={self.fitness:.4f}, gen={self.generation})"
