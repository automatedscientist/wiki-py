"""Dataset generation utilities for WikiKG."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from .schemas import EntityProfile, GraphPath, Neighborhood, Subgraph, Triplet
from .storage import Storage


class DatasetGenerator:
    """Generate structured datasets from WikiKG for LLM training."""

    def __init__(self, storage: Storage):
        """Initialize with a storage backend.

        Args:
            storage: WikiKG Storage instance
        """
        self.storage = storage
        self._relation_filter: set[str] | None = None
        self._min_degree: int = 0

    def filter_by_relation_types(self, types: list[str]) -> "DatasetGenerator":
        """Filter to only include specific relation types.

        Args:
            types: List of relation types to include

        Returns:
            Self for chaining
        """
        self._relation_filter = set(types)
        return self

    def filter_connected_only(self, min_degree: int = 1) -> "DatasetGenerator":
        """Filter to only include entities with minimum connections.

        Args:
            min_degree: Minimum number of relations required

        Returns:
            Self for chaining
        """
        self._min_degree = min_degree
        return self

    def clear_filters(self) -> "DatasetGenerator":
        """Clear all filters."""
        self._relation_filter = None
        self._min_degree = 0
        return self

    def extract_triplets(
        self,
        limit: int | None = None,
        show_progress: bool = True
    ) -> Iterator[Triplet]:
        """Extract all triplets from the graph.

        Args:
            limit: Maximum number of triplets to yield
            show_progress: Show progress bar

        Yields:
            Triplet objects
        """
        conn = self.storage._get_conn()

        query = """
            SELECT s.name as subject, r.type as relation, o.name as object
            FROM relations r
            JOIN entities s ON r.subject_id = s.id
            JOIN entities o ON r.object_id = o.id
        """

        if self._relation_filter:
            placeholders = ",".join("?" * len(self._relation_filter))
            query += f" WHERE r.type IN ({placeholders})"
            params = list(self._relation_filter)
        else:
            params = []

        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query, params)

        count = 0
        total = limit or self.storage.count_relations()
        pbar = tqdm(total=total, desc="Extracting triplets", disable=not show_progress)

        for row in cursor:
            yield Triplet(
                subject=row["subject"],
                relation=row["relation"],
                object=row["object"]
            )
            count += 1
            pbar.update(1)
            if limit and count >= limit:
                break

        pbar.close()

    def extract_entity_profiles(
        self,
        min_relations: int = 1,
        limit: int | None = None,
        show_progress: bool = True
    ) -> Iterator[EntityProfile]:
        """Extract entity profiles with all their relations.

        Args:
            min_relations: Minimum total relations required
            limit: Maximum number of profiles to yield
            show_progress: Show progress bar

        Yields:
            EntityProfile objects
        """
        conn = self.storage._get_conn()

        # Get entities with their relation counts using subquery
        query = """
            SELECT name, out_count, in_count FROM (
                SELECT e.name,
                       (SELECT COUNT(*) FROM relations r WHERE r.subject_id = e.id) as out_count,
                       (SELECT COUNT(*) FROM relations r WHERE r.object_id = e.id) as in_count
                FROM entities e
            )
            WHERE (out_count + in_count) >= ?
        """
        params = [min_relations]

        if limit:
            query += f" LIMIT {limit}"

        entities = conn.execute(query, params).fetchall()

        pbar = tqdm(entities, desc="Extracting profiles", disable=not show_progress)

        for row in pbar:
            name = row["name"]

            # Get outgoing relations
            outgoing = []
            out_query = """
                SELECT s.name as subject, r.type as relation, o.name as object
                FROM relations r
                JOIN entities s ON r.subject_id = s.id
                JOIN entities o ON r.object_id = o.id
                WHERE s.name = ?
            """
            for rel in conn.execute(out_query, (name,)):
                if self._relation_filter and rel["relation"] not in self._relation_filter:
                    continue
                outgoing.append(Triplet(
                    subject=rel["subject"],
                    relation=rel["relation"],
                    object=rel["object"]
                ))

            # Get incoming relations
            incoming = []
            in_query = """
                SELECT s.name as subject, r.type as relation, o.name as object
                FROM relations r
                JOIN entities s ON r.subject_id = s.id
                JOIN entities o ON r.object_id = o.id
                WHERE o.name = ?
            """
            for rel in conn.execute(in_query, (name,)):
                if self._relation_filter and rel["relation"] not in self._relation_filter:
                    continue
                incoming.append(Triplet(
                    subject=rel["subject"],
                    relation=rel["relation"],
                    object=rel["object"]
                ))

            # Get properties
            props = {}
            for prop in self.storage.get_properties(name):
                props[prop["key"]] = prop["value"]

            profile = EntityProfile(
                name=name,
                outgoing=outgoing,
                incoming=incoming,
                properties=props
            )

            if profile.total_relations >= min_relations:
                yield profile

    def extract_neighborhoods(
        self,
        limit: int | None = None,
        show_progress: bool = True
    ) -> Iterator[Neighborhood]:
        """Extract 1-hop neighborhoods for all entities.

        Args:
            limit: Maximum number of neighborhoods to yield
            show_progress: Show progress bar

        Yields:
            Neighborhood objects
        """
        conn = self.storage._get_conn()

        query = "SELECT name FROM entities"
        if limit:
            query += f" LIMIT {limit}"

        entities = conn.execute(query).fetchall()
        pbar = tqdm(entities, desc="Extracting neighborhoods", disable=not show_progress)

        for row in pbar:
            name = row["name"]
            neighbors = []

            # Get all relations where this entity is involved
            for neighbor_info in self.storage.get_neighbors(name, direction="both"):
                if self._relation_filter and neighbor_info["relation_type"] not in self._relation_filter:
                    continue

                if neighbor_info["direction"] == "forward":
                    triplet = Triplet(
                        subject=name,
                        relation=neighbor_info["relation_type"],
                        object=neighbor_info["neighbor"]
                    )
                else:
                    triplet = Triplet(
                        subject=neighbor_info["neighbor"],
                        relation=neighbor_info["relation_type"],
                        object=name
                    )
                neighbors.append(triplet)

            if len(neighbors) >= self._min_degree:
                yield Neighborhood(center=name, neighbors=neighbors)

    def extract_paths(
        self,
        num_paths: int = 10000,
        min_length: int = 3,
        max_length: int = 6,
        allow_cycles: bool = False,
        show_progress: bool = True
    ) -> Iterator[GraphPath]:
        """Extract random walk paths through the graph.

        Args:
            num_paths: Number of paths to generate
            min_length: Minimum path length (entities)
            max_length: Maximum path length (entities)
            allow_cycles: Whether to allow revisiting entities
            show_progress: Show progress bar

        Yields:
            GraphPath objects
        """
        generated = 0
        attempts = 0
        max_attempts = num_paths * 10

        pbar = tqdm(total=num_paths, desc="Generating paths", disable=not show_progress)

        while generated < num_paths and attempts < max_attempts:
            attempts += 1

            path_data = self.storage.random_walk(
                start=None,
                steps=max_length - 1,
                allow_cycles=allow_cycles
            )

            if len(path_data) < min_length:
                continue

            # Convert to GraphPath
            entities = [step["entity"] for step in path_data]
            relations = [step["relation"] for step in path_data[1:]]
            directions = [step["direction"] for step in path_data[1:]]

            # Apply relation filter
            if self._relation_filter:
                if not all(r in self._relation_filter for r in relations):
                    continue

            yield GraphPath(
                entities=entities,
                relations=relations,
                directions=directions
            )

            generated += 1
            pbar.update(1)

        pbar.close()

    def extract_subgraph(
        self,
        center: str,
        depth: int = 2
    ) -> Subgraph:
        """Extract a connected subgraph around a center entity.

        Args:
            center: Center entity name
            depth: Number of hops from center

        Returns:
            Subgraph object
        """
        visited_entities: set[str] = set()
        triplets: list[Triplet] = []

        # BFS to find all entities within depth
        queue: deque[tuple[str, int]] = deque([(center, 0)])
        visited_entities.add(center)

        while queue:
            entity, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            neighbors = self.storage.get_neighbors(entity, direction="both")

            for neighbor_info in neighbors:
                neighbor = neighbor_info["neighbor"]
                relation = neighbor_info["relation_type"]
                direction = neighbor_info["direction"]

                # Apply relation filter
                if self._relation_filter and relation not in self._relation_filter:
                    continue

                # Create triplet
                if direction == "forward":
                    triplet = Triplet(subject=entity, relation=relation, object=neighbor)
                else:
                    triplet = Triplet(subject=neighbor, relation=relation, object=entity)

                triplets.append(triplet)

                if neighbor not in visited_entities:
                    visited_entities.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

        return Subgraph(
            center=center,
            entities=list(visited_entities),
            triplets=triplets,
            depth=depth
        )

    def to_jsonl(
        self,
        path: Path,
        data_type: str,
        **kwargs
    ) -> int:
        """Export dataset to JSONL file.

        Args:
            path: Output file path
            data_type: One of "triplets", "profiles", "neighborhoods", "paths"
            **kwargs: Arguments passed to the extraction method

        Returns:
            Number of records written
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        extractors = {
            "triplets": self.extract_triplets,
            "profiles": self.extract_entity_profiles,
            "neighborhoods": self.extract_neighborhoods,
            "paths": self.extract_paths,
        }

        if data_type not in extractors:
            raise ValueError(f"Unknown data_type: {data_type}. Use one of {list(extractors.keys())}")

        count = 0
        with open(path, "w") as f:
            for item in extractors[data_type](**kwargs):
                f.write(item.model_dump_json() + "\n")
                count += 1

        return count
