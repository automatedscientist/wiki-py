"""Pydantic schemas for WikiKG dataset extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Triplet(BaseModel):
    """Basic knowledge unit: (subject, relation, object)."""

    subject: str
    relation: str
    object: str


class EntityProfile(BaseModel):
    """All knowledge about a single entity."""

    name: str
    outgoing: list[Triplet] = Field(default_factory=list)
    incoming: list[Triplet] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)

    @property
    def total_relations(self) -> int:
        """Total number of relations (incoming + outgoing)."""
        return len(self.outgoing) + len(self.incoming)


class GraphPath(BaseModel):
    """A traversal path through the graph."""

    entities: list[str]
    relations: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of edges in the path."""
        return len(self.relations)

    def to_text(self) -> str:
        """Convert path to human-readable text format.

        Example: "Einstein -[FIELD_OF_WORK]-> Physics -[STUDIED_BY]<- Feynman"
        """
        if not self.entities:
            return ""

        parts = [self.entities[0]]
        for i, (relation, direction, entity) in enumerate(
            zip(self.relations, self.directions, self.entities[1:])
        ):
            arrow = "->" if direction == "forward" else "<-"
            parts.append(f" -[{relation}]{arrow} {entity}")

        return "".join(parts)


class Neighborhood(BaseModel):
    """Entity with its 1-hop neighbors."""

    center: str
    neighbors: list[Triplet] = Field(default_factory=list)

    @property
    def degree(self) -> int:
        """Number of neighbors."""
        return len(self.neighbors)


class Subgraph(BaseModel):
    """Connected subgraph around an entity."""

    center: str
    entities: list[str] = Field(default_factory=list)
    triplets: list[Triplet] = Field(default_factory=list)
    depth: int = 1

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_edges(self) -> int:
        return len(self.triplets)
