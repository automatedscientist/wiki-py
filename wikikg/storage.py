"""SQLite storage backend for WikiKG knowledge graph."""

from __future__ import annotations
import json
import random
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .entities import Entity, Citation, Event
from .relations import Relation


class WikiKGJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Entity objects and sets."""

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, 'name'):  # Entity-like objects
            return obj.name
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def json_dumps(obj: Any) -> str:
    """JSON dump with WikiKG encoder."""
    return json.dumps(obj, cls=WikiKGJSONEncoder)


SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    props JSON
);

CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regex TEXT,
    source TEXT,
    snippet TEXT
);

CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    key TEXT NOT NULL,
    value TEXT,
    citation_id INTEGER REFERENCES citations(id)
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    object_id INTEGER NOT NULL REFERENCES entities(id),
    citation_id INTEGER REFERENCES citations(id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT,
    label TEXT,
    props JSON,
    citation_id INTEGER REFERENCES citations(id)
);

CREATE TABLE IF NOT EXISTS timeline_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    event_id INTEGER NOT NULL REFERENCES events(id)
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_properties_entity ON properties(entity_id);
CREATE INDEX IF NOT EXISTS idx_properties_key ON properties(key);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(type);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
"""


class Storage:
    """SQLite storage for WikiKG knowledge graph."""

    def __init__(self, db_path: str | Path = ":memory:"):
        """Initialize storage with database path.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory DB
        """
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(SCHEMA)
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # Citation operations

    def insert_citation(self, citation: Citation) -> int:
        """Insert citation and return its ID."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO citations (regex, source, snippet) VALUES (?, ?, ?)",
                (citation.regex, citation.source, citation.meta.get("Snippet", ""))
            )
            return cursor.lastrowid  # type: ignore

    def get_citation(self, citation_id: int) -> Citation | None:
        """Get citation by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT regex, source, snippet FROM citations WHERE id = ?",
            (citation_id,)
        ).fetchone()
        if row:
            meta = {"Snippet": row["snippet"]} if row["snippet"] else {}
            return Citation(row["regex"], row["source"], meta)
        return None

    # Entity operations

    def insert_entity(self, entity: Entity) -> int:
        """Insert entity and return its ID."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT OR REPLACE INTO entities (name, props) VALUES (?, ?)",
                (entity.name, json_dumps(entity.props))
            )
            return cursor.lastrowid  # type: ignore

    def get_entity_id(self, name: str) -> int | None:
        """Get entity ID by name."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM entities WHERE name = ?",
            (name,)
        ).fetchone()
        return row["id"] if row else None

    def get_or_create_entity_id(self, entity: Entity | str) -> int:
        """Get entity ID, creating if needed.

        Args:
            entity: Entity object or entity name string
        """
        if isinstance(entity, str):
            name = entity
            entity_id = self.get_entity_id(name)
            if entity_id is None:
                entity_id = self.insert_entity(Entity(name, {}))
        else:
            entity_id = self.get_entity_id(entity.name)
            if entity_id is None:
                entity_id = self.insert_entity(entity)
        return entity_id

    def get_entity(self, name: str) -> Entity | None:
        """Get entity by name."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT name, props FROM entities WHERE name = ?",
            (name,)
        ).fetchone()
        if row:
            props = json.loads(row["props"]) if row["props"] else {}
            return Entity(row["name"], props)
        return None

    def all_entities(self) -> list[Entity]:
        """Get all entities."""
        conn = self._get_conn()
        rows = conn.execute("SELECT name, props FROM entities").fetchall()
        return [
            Entity(row["name"], json.loads(row["props"]) if row["props"] else {})
            for row in rows
        ]

    # Property operations

    def insert_property(
        self,
        entity: Entity | str,
        key: str,
        value: Any,
        citation: Citation | None = None
    ) -> int:
        """Insert property for entity.

        Args:
            entity: Entity object or entity name string
            key: Property key
            value: Property value
            citation: Optional citation
        """
        entity_id = self.get_or_create_entity_id(entity)
        citation_id = self.insert_citation(citation) if citation else None

        # Convert value to string, handling Entity objects
        if hasattr(value, 'name'):
            value = value.name
        value_str = str(value) if value is not None else ""

        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO properties (entity_id, key, value, citation_id) VALUES (?, ?, ?, ?)",
                (entity_id, key, value_str, citation_id)
            )
            return cursor.lastrowid  # type: ignore

    def get_properties(self, entity_name: str) -> list[dict[str, Any]]:
        """Get all properties for an entity."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT p.key, p.value, c.regex, c.source, c.snippet
            FROM properties p
            JOIN entities e ON p.entity_id = e.id
            LEFT JOIN citations c ON p.citation_id = c.id
            WHERE e.name = ?
            """,
            (entity_name,)
        ).fetchall()

        return [
            {
                "key": row["key"],
                "value": row["value"],
                "citation": Citation(
                    row["regex"],
                    row["source"],
                    {"Snippet": row["snippet"]} if row["snippet"] else {}
                ) if row["regex"] else None
            }
            for row in rows
        ]

    # Relation operations

    def insert_relation(
        self,
        relation: Relation,
        citation: Citation | None = None
    ) -> int:
        """Insert relation between entities.

        Handles both Entity objects and strings for subject/object.
        """
        # Handle subject - could be Entity, string, or have a .name attribute
        subject = relation.subject
        if hasattr(subject, 'name'):
            subject = subject.name
        subject_id = self.get_or_create_entity_id(str(subject))

        # Handle object - could be Entity, string, or have a .name attribute
        obj = relation.obj
        if hasattr(obj, 'name'):
            obj = obj.name
        object_id = self.get_or_create_entity_id(str(obj))

        citation_id = self.insert_citation(citation) if citation else None

        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO relations (type, subject_id, object_id, citation_id) VALUES (?, ?, ?, ?)",
                (relation.relation_type, subject_id, object_id, citation_id)
            )
            return cursor.lastrowid  # type: ignore

    def get_relations(
        self,
        subject: str | None = None,
        obj: str | None = None,
        rel_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Query relations with optional filters."""
        conn = self._get_conn()
        query = """
            SELECT r.type, s.name as subject, o.name as object,
                   c.regex, c.source, c.snippet
            FROM relations r
            JOIN entities s ON r.subject_id = s.id
            JOIN entities o ON r.object_id = o.id
            LEFT JOIN citations c ON r.citation_id = c.id
            WHERE 1=1
        """
        params: list[Any] = []

        if subject:
            query += " AND s.name = ?"
            params.append(subject)
        if obj:
            query += " AND o.name = ?"
            params.append(obj)
        if rel_type:
            query += " AND r.type = ?"
            params.append(rel_type)

        rows = conn.execute(query, params).fetchall()

        return [
            {
                "type": row["type"],
                "subject": row["subject"],
                "object": row["object"],
                "citation": Citation(
                    row["regex"],
                    row["source"],
                    {"Snippet": row["snippet"]} if row["snippet"] else {}
                ) if row["regex"] else None
            }
            for row in rows
        ]

    # Event operations

    def _to_str(self, val: Any) -> str | None:
        """Convert value to string, handling Entity objects."""
        if val is None:
            return None
        if hasattr(val, 'name'):
            return val.name
        return str(val)

    def _serialize_props(self, props: dict) -> str:
        """Serialize props dict, converting Entity objects to names."""
        clean_props = {}
        for k, v in props.items():
            if hasattr(v, 'name'):
                clean_props[k] = v.name
            elif isinstance(v, set):
                clean_props[k] = list(v)
            elif isinstance(v, (list, tuple)):
                clean_props[k] = [x.name if hasattr(x, 'name') else x for x in v]
            else:
                clean_props[k] = v
        return json_dumps(clean_props)

    def insert_event(self, event: Event) -> int:
        """Insert event."""
        citation_id = self.insert_citation(event.citation) if event.citation else None

        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO events (name, date, label, props, citation_id) VALUES (?, ?, ?, ?, ?)",
                (
                    self._to_str(event.name),
                    self._to_str(event.date),
                    self._to_str(event.label),
                    self._serialize_props(event.props),
                    citation_id
                )
            )
            return cursor.lastrowid  # type: ignore

    def link_timeline_event(self, entity: Entity, event_id: int) -> None:
        """Link an entity to a timeline event."""
        entity_id = self.get_or_create_entity_id(entity)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO timeline_links (entity_id, event_id) VALUES (?, ?)",
                (entity_id, event_id)
            )

    def get_timeline(self, entity_name: str) -> list[dict[str, Any]]:
        """Get timeline events for an entity, sorted by date."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT e.name, e.date, e.label, e.props
            FROM events e
            JOIN timeline_links tl ON e.id = tl.event_id
            JOIN entities ent ON tl.entity_id = ent.id
            WHERE ent.name = ?
            ORDER BY e.date
            """,
            (entity_name,)
        ).fetchall()

        return [
            {
                "name": row["name"],
                "date": row["date"],
                "label": row["label"],
                "props": json.loads(row["props"]) if row["props"] else {}
            }
            for row in rows
        ]

    # Query helpers

    def count_entities(self) -> int:
        """Count total entities."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as count FROM entities").fetchone()
        return row["count"]

    def count_relations(self) -> int:
        """Count total relations."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as count FROM relations").fetchone()
        return row["count"]

    def relation_type_counts(self) -> dict[str, int]:
        """Get counts by relation type."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT type, COUNT(*) as count FROM relations GROUP BY type"
        ).fetchall()
        return {row["type"]: row["count"] for row in rows}

    # Graph traversal operations

    def get_neighbors(
        self,
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
        conn = self._get_conn()
        neighbors = []

        if direction in ("forward", "both"):
            # Entity is subject -> find objects
            rows = conn.execute(
                """
                SELECT o.name as neighbor, r.type as relation_type
                FROM relations r
                JOIN entities s ON r.subject_id = s.id
                JOIN entities o ON r.object_id = o.id
                WHERE s.name = ?
                """,
                (entity_name,)
            ).fetchall()
            for row in rows:
                neighbors.append({
                    "neighbor": row["neighbor"],
                    "relation_type": row["relation_type"],
                    "direction": "forward"
                })

        if direction in ("backward", "both"):
            # Entity is object -> find subjects
            rows = conn.execute(
                """
                SELECT s.name as neighbor, r.type as relation_type
                FROM relations r
                JOIN entities s ON r.subject_id = s.id
                JOIN entities o ON r.object_id = o.id
                WHERE o.name = ?
                """,
                (entity_name,)
            ).fetchall()
            for row in rows:
                neighbors.append({
                    "neighbor": row["neighbor"],
                    "relation_type": row["relation_type"],
                    "direction": "backward"
                })

        return neighbors

    def get_random_entity(self) -> str | None:
        """Get a random entity name from the database."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT name FROM entities ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return row["name"] if row else None

    def random_walk(
        self,
        start: str | None = None,
        steps: int = 5,
        allow_cycles: bool = False
    ) -> list[dict[str, Any]]:
        """Generate a random path starting from an entity.

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
        if start is None:
            start = self.get_random_entity()
            if start is None:
                return []

        path = [{"entity": start}]
        visited = {start}
        current = start

        for _ in range(steps):
            neighbors = self.get_neighbors(current, direction="both")

            if not allow_cycles:
                neighbors = [n for n in neighbors if n["neighbor"] not in visited]

            if not neighbors:
                break

            chosen = random.choice(neighbors)
            current = chosen["neighbor"]
            visited.add(current)

            path.append({
                "entity": current,
                "relation": chosen["relation_type"],
                "direction": chosen["direction"]
            })

        return path

    def format_path(self, path: list[dict[str, Any]]) -> str:
        """Format a path as a human-readable string.

        Example: "Einstein -[FIELD_OF_WORK]-> Physics -[STUDIED_BY]-> Feynman"
        """
        if not path:
            return ""

        parts = [path[0]["entity"]]
        for step in path[1:]:
            arrow = "->" if step["direction"] == "forward" else "<-"
            parts.append(f" -[{step['relation']}]{arrow} {step['entity']}")

        return "".join(parts)
