"""WikiKG: Python package for wiki-kg-dataset knowledge graphs.

This package provides tools for working with knowledge graphs extracted from
Wikipedia articles, stored in Mathematica/Wolfram Language format.

Main features:
- Entity/Relation/Citation classes for knowledge graph representation
- SQLite storage backend for persistence
- Experta integration for rule-based reasoning
- Batch converter for .m to .py files

Usage:
    from wikikg import *

    # Initialize database
    init_db("knowledge.db")

    # Create entities and relations
    Wing = AddEntity("Wing", {})
    Fin = AddEntity("Fin", {})

    # Set properties with citations
    SetPropertyCited(Wing, "Medium", "air",
        Citation("air or some other fluid", "Wing", {"Snippet": "air"}))

    # Assert relations
    AssertCited(TYPE_OF(Wing, Fin),
        Citation("A wing is a type of fin", "Wing", {}))
"""

from __future__ import annotations
from typing import Any

from .entities import Entity, Citation, Event
from .relations import Relation, RELATION_TYPES, get_relation_class, all_relation_types
from .storage import Storage

# Import all relation types for wildcard import
from .relations import (
    TYPE_OF, IS_A, HAS_PART, PART_OF, INSTANCE_OF, SUBCLASS_OF,
    MEMBER_OF, BELONGS_TO, CONTAINS, COMPRISES, RELATED_TO, SIMILAR_TO,
    DIFFERENT_FROM, OPPOSITE_OF, EQUIVALENT_TO, SAME_AS,
    CAUSES, CAUSED_BY, INFLUENCES, INFLUENCED_BY, AFFECTS, AFFECTED_BY,
    ENABLES, PREVENTS, REQUIRES, DEPENDS_ON, RESULTS_IN,
    PRECEDED_BY, FOLLOWED_BY, BEFORE, AFTER, DURING, CONTEMPORARY_OF,
    SUCCEEDS, PRECEDES,
    LOCATED_IN, LOCATION_OF, CAPITAL_OF, HAS_CAPITAL, BORDERS, NEAR,
    CONTAINS_LOCATION, REGION_OF, COUNTRY_OF, CONTINENT_OF,
    BORN_IN, DIED_IN, BIRTHPLACE_OF, DEATHPLACE_OF, NATIONALITY, CITIZEN_OF,
    OCCUPATION, PROFESSION_OF, EMPLOYER, EMPLOYED_BY, EDUCATED_AT,
    ALMA_MATER_OF, SPOUSE_OF, CHILD_OF, PARENT_OF, SIBLING_OF, RELATIVE_OF,
    FRIEND_OF, COLLEAGUE_OF, MENTOR_OF, STUDENT_OF,
    CREATED_BY, CREATOR_OF, INVENTED_BY, INVENTOR_OF, DISCOVERED_BY,
    DISCOVERER_OF, FOUNDED_BY, FOUNDER_OF, AUTHORED_BY, AUTHOR_OF,
    COMPOSED_BY, COMPOSER_OF, DIRECTED_BY, DIRECTOR_OF, PRODUCED_BY,
    PRODUCER_OF, DESIGNED_BY, DESIGNER_OF, BUILT_BY, BUILDER_OF,
    AFFILIATED_WITH, HEADQUARTERS_IN, SUBSIDIARY_OF, PARENT_COMPANY_OF,
    DEPARTMENT_OF, DIVISION_OF, BRANCH_OF, LED_BY, LEADER_OF, CEO_OF,
    PRESIDENT_OF, CHAIRMAN_OF,
    DERIVES_FROM, DERIVED_FROM, APPLIED_TO, APPLICATION_OF, FIELD_OF,
    STUDIES, STUDIED_BY, PROPERTY_OF, HAS_PROPERTY, CHARACTERISTIC_OF,
    COMPOSED_OF, MADE_OF, MATERIAL_OF,
    SPECIES_OF, GENUS_OF, FAMILY_OF, ORDER_OF, CLASS_OF, PHYLUM_OF,
    KINGDOM_OF, HABITAT_OF, LIVES_IN, EATS, EATEN_BY, PREDATOR_OF,
    PREY_OF, SYMBIONT_OF, HOST_OF, PARASITE_OF,
    GENRE_OF, STYLE_OF, MOVEMENT_OF, INSPIRED_BY, INSPIRATION_FOR,
    PERFORMED_BY, PERFORMER_OF, PORTRAYED_BY, PORTRAYAL_OF, ADAPTATION_OF,
    BASED_ON,
    CURRENCY_OF, TRADED_IN, MARKET_OF, INDUSTRY_OF, SECTOR_OF, PRODUCT_OF,
    MANUFACTURER_OF, SUPPLIER_OF, CUSTOMER_OF, COMPETITOR_OF,
    GOVERNED_BY, GOVERNS, OFFICIAL_LANGUAGE_OF, MEMBER_STATE_OF,
    SIGNATORY_OF, ALLY_OF, ENEMY_OF, COLONY_OF, COLONIZED_BY,
    TIMELINE_EVENT, PARTICIPANT_IN, OCCURRED_IN, VENUE_OF, ORGANIZER_OF,
    WINNER_OF, AWARD_FOR,
    LANGUAGE_OF, SPOKEN_IN, WRITTEN_IN, TRANSLATED_FROM, TRANSLATION_OF,
    ETYMOLOGY_OF, DERIVED_WORD_OF,
    THEOREM_OF, PROOF_OF, COROLLARY_OF, GENERALIZATION_OF, SPECIAL_CASE_OF,
    DUAL_OF, INVERSE_OF,
)


__version__ = "0.1.0"

# Global database instance
_db: Storage | None = None


def init_db(path: str = ":memory:") -> Storage:
    """Initialize the SQLite database.

    Args:
        path: Path to database file, or ":memory:" for in-memory DB

    Returns:
        Storage instance
    """
    global _db
    _db = Storage(path)
    return _db


def get_db() -> Storage:
    """Get current database instance, initializing if needed."""
    global _db
    if _db is None:
        _db = Storage(":memory:")
    return _db


def close_db() -> None:
    """Close the database connection."""
    global _db
    if _db is not None:
        _db.close()
        _db = None


def AddEntity(name: str, props: dict[str, Any] | None = None) -> Entity:
    """Create an entity and store in database.

    Args:
        name: Entity name/identifier
        props: Optional properties dict

    Returns:
        Entity instance
    """
    props = props if props is not None else {}
    entity = Entity(name, props)
    db = get_db()
    db.insert_entity(entity)
    return entity


def SetPropertyCited(
    entity: Entity,
    key: str,
    value: Any,
    citation: Citation | None = None
) -> None:
    """Set a property on an entity with optional citation.

    Args:
        entity: Target entity
        key: Property key
        value: Property value
        citation: Optional citation for the property
    """
    entity.props[key] = value
    db = get_db()
    db.insert_property(entity, key, value, citation)


def AssertCited(relation: Relation, citation: Citation | None = None) -> None:
    """Assert a relation between entities with optional citation.

    Args:
        relation: Relation instance (e.g., TYPE_OF(Wing, Fin))
        citation: Optional citation for the relation
    """
    db = get_db()
    db.insert_relation(relation, citation)


def AddEvent(
    name: str,
    props: dict[str, Any],
    citation: Citation | None = None
) -> Event:
    """Add a timeline event.

    Args:
        name: Event name/identifier
        props: Event properties (should include "Date", "Label")
        citation: Optional citation

    Returns:
        Event instance
    """
    citation = citation or Citation("", "", {})
    event = Event(name, props, citation)
    db = get_db()
    db.insert_event(event)
    return event


def LinkTimelineEvent(entity: Entity, event: Event) -> None:
    """Link an entity to a timeline event.

    Args:
        entity: Entity to link
        event: Event to link to
    """
    db = get_db()
    # We need to get the event ID, but our current implementation doesn't return it
    # This is a limitation that could be improved
    pass


# Utility functions for querying

def get_entity(name: str) -> Entity | None:
    """Get entity by name from registry."""
    return Entity.get(name)


def get_or_create_entity(name: str) -> Entity:
    """Get existing entity or create new one (lazy auto-create)."""
    entity = Entity.get_or_create(name)
    db = get_db()
    if db.get_entity_id(name) is None:
        db.insert_entity(entity)
    return entity


def query_relations(
    subject: str | None = None,
    obj: str | None = None,
    rel_type: str | None = None
) -> list[dict[str, Any]]:
    """Query relations with optional filters.

    Args:
        subject: Filter by subject entity name
        obj: Filter by object entity name
        rel_type: Filter by relation type

    Returns:
        List of relation dicts with type, subject, object, citation
    """
    db = get_db()
    return db.get_relations(subject=subject, obj=obj, rel_type=rel_type)


def query_properties(entity_name: str) -> list[dict[str, Any]]:
    """Get all properties for an entity.

    Args:
        entity_name: Entity name

    Returns:
        List of property dicts with key, value, citation
    """
    db = get_db()
    return db.get_properties(entity_name)


# Graph traversal functions

def get_neighbors(
    entity_name: str,
    direction: str = "both"
) -> list[dict[str, Any]]:
    """Get all entities connected to this entity.

    Args:
        entity_name: Name of the entity
        direction: "forward" (outgoing), "backward" (incoming), or "both"

    Returns:
        List of dicts with keys: neighbor, relation_type, direction
    """
    db = get_db()
    return db.get_neighbors(entity_name, direction)


def get_random_entity() -> str | None:
    """Get a random entity name from the database."""
    db = get_db()
    return db.get_random_entity()


def random_walk(
    start: str | None = None,
    steps: int = 5,
    allow_cycles: bool = False
) -> list[dict[str, Any]]:
    """Generate a random path through the knowledge graph.

    Args:
        start: Starting entity name (random if None)
        steps: Maximum number of steps to take
        allow_cycles: Whether to allow revisiting entities

    Returns:
        List of path steps: [
            {"entity": "A"},
            {"entity": "B", "relation": "TYPE_OF", "direction": "forward"},
            ...
        ]
    """
    db = get_db()
    return db.random_walk(start, steps, allow_cycles)


def format_path(path: list[dict[str, Any]]) -> str:
    """Format a path as a human-readable string.

    Example: "Einstein -[FIELD_OF_WORK]-> Physics -[STUDIED_BY]-> Feynman"
    """
    db = get_db()
    return db.format_path(path)


def clear_all() -> None:
    """Clear all entities from registry and close database."""
    Entity.clear_registry()
    close_db()


# Export list for wildcard import
__all__ = [
    # Version
    "__version__",

    # Core classes
    "Entity",
    "Citation",
    "Event",
    "Relation",

    # Database
    "Storage",
    "init_db",
    "get_db",
    "close_db",

    # API functions
    "AddEntity",
    "SetPropertyCited",
    "AssertCited",
    "AddEvent",
    "LinkTimelineEvent",

    # Query functions
    "get_entity",
    "get_or_create_entity",
    "query_relations",
    "query_properties",
    "clear_all",

    # Graph traversal
    "get_neighbors",
    "get_random_entity",
    "random_walk",
    "format_path",

    # Relation utilities
    "RELATION_TYPES",
    "get_relation_class",
    "all_relation_types",

    # All relation types (from relations.py)
    *RELATION_TYPES,
]
