"""Tool-calling dataset generation for WikiKG.

Builds replayable, multihop tool-call trajectories from GraphPath records.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .entities import Citation, Entity
from .schemas import GraphPath
from .storage import Storage

def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_relations",
                "description": "Query relations with optional filters. Returns list of {type, subject, object, citation}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": ["string", "null"]},
                        "obj": {"type": ["string", "null"]},
                        "rel_type": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_neighbors",
                "description": 'Get neighbors of an entity. direction is "forward", "backward", or "both".',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string"},
                        "direction": {"type": "string", "enum": ["forward", "backward", "both"]},
                    },
                    "required": ["entity_name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_properties",
                "description": "Query properties for an entity. Returns list of {key, value, citation}.",
                "parameters": {
                    "type": "object",
                    "properties": {"entity_name": {"type": "string"}},
                    "required": ["entity_name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_entity",
                "description": "Get an entity by name. Returns {name, props} or null.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_random_entity",
                "description": "Get a random entity name, or null if none exist.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Citation):
        return {"regex": obj.regex, "source": obj.source, "meta": obj.meta}
    if isinstance(obj, Entity):
        return {"name": obj.name, "props": obj.props}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _stable_sort_dicts(items: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(item.get(k) for k in keys)

    return sorted(items, key=sort_key)


def verbalize_relation(rel_type: str) -> str:
    return rel_type.replace("_", " ").lower()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_head_sha(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None
    text = head.read_text().strip()
    if text.startswith("ref: "):
        ref_path = repo_root / ".git" / text.removeprefix("ref: ").strip()
        if ref_path.exists():
            return ref_path.read_text().strip()
        return None
    return text or None


def load_paths_jsonl(path: Path) -> list[GraphPath]:
    paths: list[GraphPath] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            if isinstance(raw, dict) and "entities" in raw:
                paths.append(GraphPath.model_validate(raw))
                continue

            if isinstance(raw, dict) and "path" in raw:
                steps = raw["path"]
                entities = [s["entity"] for s in steps]
                relations = [s["relation"] for s in steps[1:]]
                directions = [s["direction"] for s in steps[1:]]
                paths.append(GraphPath(entities=entities, relations=relations, directions=directions))
                continue

            raise ValueError(f"Unrecognized path record format: {raw.keys() if isinstance(raw, dict) else type(raw)}")
    return paths


@dataclass(frozen=True)
class ToolCallStyle:
    name: Literal["query_relations", "get_neighbors"]


@dataclass(frozen=True)
class GenerationStats:
    considered: int
    emitted: int
    skipped_duplicates: int


class ToolCallingDatasetGenerator:
    def __init__(
        self,
        storage: Storage,
        *,
        max_tool_results: int = 50,
        seed: int = 0,
    ) -> None:
        self.storage = storage
        self.max_tool_results = max_tool_results
        self._seed = seed

    def build_example(
        self,
        graph_path: GraphPath,
        *,
        style: ToolCallStyle,
        drop_if_answer_is_direct_neighbor: bool = True,
        question_family: Literal["endpoint"] = "endpoint",
        example_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if graph_path.length < 1:
            return None

        start = graph_path.entities[0]
        answer = graph_path.entities[-1]

        if drop_if_answer_is_direct_neighbor:
            neighbors = self.storage.get_neighbors(start, direction="both")
            if any(n["neighbor"] == answer for n in neighbors):
                return None

        hop_text = ", then ".join(verbalize_relation(r) for r in graph_path.relations)
        if question_family == "endpoint":
            question = (
                f"Starting from {start}, follow this sequence of relationships: "
                f"{hop_text}. What is the final entity?"
            )
        else:
            return None

        tools = tool_schema()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        hop_targets = graph_path.entities[1:]
        tool_calls_metadata: list[dict[str, Any]] = []

        current = start
        for hop_index, (rel_type, direction, expected_next) in enumerate(
            zip(graph_path.relations, graph_path.directions, hop_targets),
            start=1,
        ):
            call_id = f"call_{hop_index}"

            if style.name == "query_relations":
                if direction == "forward":
                    args = {"subject": current, "obj": None, "rel_type": rel_type}
                    results = self.storage.get_relations(subject=current, rel_type=rel_type)
                    required = {"type": rel_type, "subject": current, "object": expected_next}
                else:
                    args = {"subject": None, "obj": current, "rel_type": rel_type}
                    results = self.storage.get_relations(obj=current, rel_type=rel_type)
                    required = {"type": rel_type, "subject": expected_next, "object": current}

                results = _stable_sort_dicts(results, ["type", "subject", "object"])
                trimmed = results[: self.max_tool_results]
                if not any(
                    r.get("type") == required["type"]
                    and r.get("subject") == required["subject"]
                    and r.get("object") == required["object"]
                    for r in trimmed
                ):
                    trimmed = trimmed[: max(0, self.max_tool_results - 1)]
                    full_required = next(
                        (
                            r
                            for r in results
                            if r.get("type") == required["type"]
                            and r.get("subject") == required["subject"]
                            and r.get("object") == required["object"]
                        ),
                        None,
                    )
                    if full_required is None:
                        return None
                    trimmed.append(full_required)

                assistant_call = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "query_relations",
                                "arguments": _stable_json_dumps(args),
                            },
                        }
                    ],
                }
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "query_relations",
                    "content": _stable_json_dumps(trimmed),
                }

                tool_calls_metadata.append(
                    {
                        "tool": "query_relations",
                        "arguments": args,
                        "expected_next": expected_next,
                        "direction": direction,
                        "relation_type": rel_type,
                    }
                )

            elif style.name == "get_neighbors":
                args = {"entity_name": current, "direction": "both"}
                results = self.storage.get_neighbors(current, direction="both")
                results = _stable_sort_dicts(results, ["neighbor", "relation_type", "direction"])
                trimmed = results[: self.max_tool_results]

                required = {
                    "neighbor": expected_next,
                    "relation_type": rel_type,
                    "direction": direction,
                }
                if direction == "backward":
                    required = {
                        "neighbor": expected_next,
                        "relation_type": rel_type,
                        "direction": "backward",
                    }

                if not any(
                    r.get("neighbor") == required["neighbor"]
                    and r.get("relation_type") == required["relation_type"]
                    and r.get("direction") == required["direction"]
                    for r in trimmed
                ):
                    trimmed = trimmed[: max(0, self.max_tool_results - 1)]
                    full_required = next(
                        (
                            r
                            for r in results
                            if r.get("neighbor") == required["neighbor"]
                            and r.get("relation_type") == required["relation_type"]
                            and r.get("direction") == required["direction"]
                        ),
                        None,
                    )
                    if full_required is None:
                        return None
                    trimmed.append(full_required)

                assistant_call = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "get_neighbors",
                                "arguments": _stable_json_dumps(args),
                            },
                        }
                    ],
                }
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "get_neighbors",
                    "content": _stable_json_dumps(trimmed),
                }

                tool_calls_metadata.append(
                    {
                        "tool": "get_neighbors",
                        "arguments": args,
                        "expected_next": expected_next,
                        "direction": direction,
                        "relation_type": rel_type,
                    }
                )
            else:
                raise ValueError(f"Unknown style: {style.name}")

            messages.append(assistant_call)
            messages.append(tool_msg)
            current = expected_next

        messages.append({"role": "assistant", "content": f"The final entity is {answer}."})

        tool_calls_in_chain = len(tool_calls_metadata)
        merged_provenance = dict(provenance or {})
        merged_provenance["tool_calls_in_chain"] = tool_calls_in_chain

        return {
            "id": example_id,
            "tools": tools,
            "messages": messages,
            "metadata": {
                "style": style.name,
                "question_family": question_family,
                "path_entities": graph_path.entities,
                "path_relations": graph_path.relations,
                "path_directions": graph_path.directions,
                "num_hops": graph_path.length,
                "num_tool_calls": tool_calls_in_chain,
                "hop_targets": hop_targets,
                "tool_calls": tool_calls_metadata,
                "provenance": merged_provenance,
            },
        }


def generate_from_paths(
    storage: Storage,
    paths: Iterable[GraphPath],
    *,
    num_examples: int,
    style: ToolCallStyle,
    max_tool_results: int = 50,
    seed: int = 0,
    drop_if_answer_is_direct_neighbor: bool = True,
    min_hops: int = 2,
    max_hops: int = 6,
    provenance: dict[str, Any] | None = None,
    dedup: bool = True,
    return_stats: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], GenerationStats]:
    generator = ToolCallingDatasetGenerator(storage, max_tool_results=max_tool_results, seed=seed)
    candidates = [p for p in paths if min_hops <= p.length <= max_hops]

    rng = random.Random(seed)
    rng.shuffle(candidates)

    examples: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    skipped_duplicates = 0

    for path_index, path in enumerate(candidates):
        if len(examples) >= num_examples:
            break

        if dedup:
            signature = (
                style.name,
                tuple(path.entities),
                tuple(path.relations),
                tuple(path.directions),
            )
            if signature in seen:
                skipped_duplicates += 1
                continue
            seen.add(signature)

        ex = generator.build_example(
            path,
            style=style,
            drop_if_answer_is_direct_neighbor=drop_if_answer_is_direct_neighbor,
            example_id=f"{style.name}-{path_index}",
            provenance=provenance,
        )
        if ex is not None:
            examples.append(ex)

    stats = GenerationStats(
        considered=len(candidates),
        emitted=len(examples),
        skipped_duplicates=skipped_duplicates,
    )
    return (examples, stats) if return_stats else examples


def write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def default_provenance(db_path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "wikikg_db_sha256": sha256_file(db_path) if db_path.exists() else None,
        "generator_git_sha": git_head_sha(repo_root),
    }
