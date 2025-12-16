from __future__ import annotations

from wikikg import Entity, IS_A, PART_OF
from wikikg.schemas import GraphPath
from wikikg.storage import Storage
from wikikg.tool_dataset import ToolCallStyle, ToolCallingDatasetGenerator, generate_from_paths
from wikikg.tool_dataset_verify import verify_example


def _seed_storage() -> Storage:
    Entity.clear_registry()
    storage = Storage(":memory:")
    a = Entity("A", {})
    b = Entity("B", {})
    c = Entity("C", {})
    storage.insert_entity(a)
    storage.insert_entity(b)
    storage.insert_entity(c)
    storage.insert_relation(IS_A(a, b), None)
    storage.insert_relation(PART_OF(b, c), None)
    return storage


def test_tool_dataset_query_relations_replayable():
    storage = _seed_storage()
    try:
        path = GraphPath(
            entities=["A", "B", "C"],
            relations=["IS_A", "PART_OF"],
            directions=["forward", "forward"],
        )
        gen = ToolCallingDatasetGenerator(storage, max_tool_results=10, seed=0)
        example = gen.build_example(path, style=ToolCallStyle("query_relations"))
        assert example is not None
        ok, reason = verify_example(storage, example)
        assert ok, reason
    finally:
        storage.close()


def test_tool_dataset_get_neighbors_replayable():
    storage = _seed_storage()
    try:
        path = GraphPath(
            entities=["A", "B", "C"],
            relations=["IS_A", "PART_OF"],
            directions=["forward", "forward"],
        )
        gen = ToolCallingDatasetGenerator(storage, max_tool_results=10, seed=0)
        example = gen.build_example(path, style=ToolCallStyle("get_neighbors"))
        assert example is not None
        ok, reason = verify_example(storage, example)
        assert ok, reason
    finally:
        storage.close()


def test_tool_dataset_backward_hop_replayable():
    Entity.clear_registry()
    storage = Storage(":memory:")
    try:
        a = Entity("A", {})
        b = Entity("B", {})
        storage.insert_entity(a)
        storage.insert_entity(b)
        # A -[IS_A]-> B. Path from B to A is a backward hop.
        storage.insert_relation(IS_A(a, b), None)

        path = GraphPath(entities=["B", "A"], relations=["IS_A"], directions=["backward"])
        gen = ToolCallingDatasetGenerator(storage, max_tool_results=10, seed=0)
        example = gen.build_example(
            path,
            style=ToolCallStyle("query_relations"),
            drop_if_answer_is_direct_neighbor=False,
        )
        assert example is not None
        ok, reason = verify_example(storage, example)
        assert ok, reason
    finally:
        storage.close()


def test_generate_from_paths_dedupes_identical_paths():
    storage = _seed_storage()
    try:
        path = GraphPath(
            entities=["A", "B", "C"],
            relations=["IS_A", "PART_OF"],
            directions=["forward", "forward"],
        )
        examples, stats = generate_from_paths(
            storage,
            [path, path],
            num_examples=10,
            style=ToolCallStyle("query_relations"),
            min_hops=2,
            max_hops=6,
            drop_if_answer_is_direct_neighbor=False,
            return_stats=True,
        )
        assert len(examples) == 1
        assert stats.skipped_duplicates == 1
    finally:
        storage.close()
