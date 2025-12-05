"""Experta Fact classes for WikiKG knowledge graph.

Maps WikiKG structures to experta Facts for rule-based reasoning.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from experta import Fact, Field

if TYPE_CHECKING:
    from experta import KnowledgeEngine
    from .storage import Storage


class EntityFact(Fact):
    """Entity representation for experta.

    Attributes:
        name: Entity name/identifier
    """
    name = Field(str, mandatory=True)


class PropertyFact(Fact):
    """Property with optional citation for experta.

    Attributes:
        entity: Entity name this property belongs to
        key: Property key/name
        value: Property value as string
    """
    entity = Field(str, mandatory=True)
    key = Field(str, mandatory=True)
    value = Field(str, mandatory=True)


class RelationFact(Fact):
    """Relation between entities for experta.

    Attributes:
        rel_type: Type of relation (e.g., "IS_A", "TYPE_OF")
        subject: Subject entity name
        obj: Object entity name
    """
    rel_type = Field(str, mandatory=True)
    subject = Field(str, mandatory=True)
    obj = Field(str, mandatory=True)


class EventFact(Fact):
    """Timeline event for experta.

    Attributes:
        name: Event name/identifier
        date: Event date (ISO format or descriptive)
        label: Event label/description
    """
    name = Field(str, mandatory=True)
    date = Field(str, mandatory=False)
    label = Field(str, mandatory=False)


class TimelineLinkFact(Fact):
    """Link between entity and timeline event.

    Attributes:
        entity: Entity name
        event: Event name
    """
    entity = Field(str, mandatory=True)
    event = Field(str, mandatory=True)


def load_to_engine(engine: KnowledgeEngine, storage: Storage) -> int:
    """Load all facts from SQLite storage into experta engine.

    Args:
        engine: Experta KnowledgeEngine instance (must call reset() first)
        storage: WikiKG Storage instance

    Returns:
        Number of facts declared
    """
    count = 0

    # Load entities
    for entity in storage.all_entities():
        engine.declare(EntityFact(name=entity.name))
        count += 1

        # Load properties for each entity
        for prop in storage.get_properties(entity.name):
            engine.declare(PropertyFact(
                entity=entity.name,
                key=prop["key"],
                value=prop["value"]
            ))
            count += 1

    # Load relations
    for rel in storage.get_relations():
        engine.declare(RelationFact(
            rel_type=rel["type"],
            subject=rel["subject"],
            obj=rel["object"]
        ))
        count += 1

    return count


def load_from_db_path(engine: KnowledgeEngine, db_path: str | Path) -> int:
    """Load facts from SQLite database file into experta engine.

    Args:
        engine: Experta KnowledgeEngine instance (must call reset() first)
        db_path: Path to SQLite database file

    Returns:
        Number of facts declared
    """
    from .storage import Storage
    storage = Storage(db_path)
    try:
        return load_to_engine(engine, storage)
    finally:
        storage.close()
