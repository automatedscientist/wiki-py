"""Core entity classes for WikiKG knowledge graph."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


class Entity:
    """Base class for all knowledge graph entities.

    Entities are registered in a global registry for lazy symbol resolution.
    When an entity is referenced before being defined, it is auto-created.
    """
    _registry: dict[str, Entity] = {}

    def __init__(self, name: str, props: dict[str, Any] | None = None):
        self.name = name
        self.props = props if props is not None else {}
        Entity._registry[name] = self

    def __repr__(self) -> str:
        return f"Entity({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)

    @classmethod
    def get_or_create(cls, name: str) -> Entity:
        """Get existing entity or create new one with empty props (lazy auto-create)."""
        if name not in cls._registry:
            return cls(name, {})
        return cls._registry[name]

    @classmethod
    def get(cls, name: str) -> Entity | None:
        """Get entity by name, returns None if not found."""
        return cls._registry.get(name)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered entities (useful for testing)."""
        cls._registry.clear()

    @classmethod
    def all_entities(cls) -> list[Entity]:
        """Return all registered entities."""
        return list(cls._registry.values())


@dataclass
class Citation:
    """Citation with source text pattern, article reference, and metadata.

    Attributes:
        regex: Text pattern or snippet from source
        source: Article name or identifier
        meta: Additional metadata (e.g., {"Snippet": "..."})
    """
    regex: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Citation({self.regex!r}, {self.source!r})"


@dataclass
class Event:
    """Timeline event with properties and citation.

    Attributes:
        name: Event identifier
        props: Event properties (e.g., {"Date": "1990-01-01", "Label": "..."})
        citation: Source citation for the event
    """
    name: str
    props: dict[str, Any]
    citation: Citation

    def __repr__(self) -> str:
        return f"Event({self.name!r})"

    @property
    def date(self) -> str | None:
        """Get event date if present."""
        return self.props.get("Date")

    @property
    def label(self) -> str | None:
        """Get event label if present."""
        return self.props.get("Label")
