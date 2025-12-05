"""Tests for wikikg package."""

import pytest
import tempfile
from pathlib import Path

from wikikg import (
    Entity, Citation, Event, Relation,
    init_db, close_db, clear_all,
    AddEntity, SetPropertyCited, AssertCited, AddEvent,
    query_relations, query_properties,
    TYPE_OF, IS_A, PART_OF, HAS_PART, CAUSES, INFLUENCES,
    get_entity, get_or_create_entity,
)
from wikikg.storage import Storage
from wikikg.converter import (
    convert_mathematica_to_python,
    convert_brackets,
    convert_associations,
    convert_rules,
    convert_comments,
    convert_add_entity,
)
from wikikg.relations import RELATION_TYPES, get_relation_class


class TestEntity:
    """Tests for Entity class."""

    def setup_method(self):
        Entity.clear_registry()

    def test_entity_creation(self):
        entity = Entity("TestEntity", {"key": "value"})
        assert entity.name == "TestEntity"
        assert entity.props == {"key": "value"}

    def test_entity_registry(self):
        Entity("Entity1", {})
        Entity("Entity2", {})
        assert "Entity1" in Entity._registry
        assert "Entity2" in Entity._registry

    def test_entity_get_or_create_new(self):
        entity = Entity.get_or_create("NewEntity")
        assert entity.name == "NewEntity"
        assert entity.props == {}

    def test_entity_get_or_create_existing(self):
        original = Entity("ExistingEntity", {"foo": "bar"})
        retrieved = Entity.get_or_create("ExistingEntity")
        assert retrieved is original

    def test_entity_equality(self):
        e1 = Entity("SameEntity", {})
        e2 = Entity.get("SameEntity")
        assert e1 == e2

    def test_entity_hash(self):
        e1 = Entity("HashEntity", {})
        entity_set = {e1}
        assert e1 in entity_set


class TestCitation:
    """Tests for Citation class."""

    def test_citation_creation(self):
        citation = Citation("some text", "ArticleName", {"Snippet": "snippet"})
        assert citation.regex == "some text"
        assert citation.source == "ArticleName"
        assert citation.meta == {"Snippet": "snippet"}

    def test_citation_default_meta(self):
        citation = Citation("text", "source")
        assert citation.meta == {}


class TestEvent:
    """Tests for Event class."""

    def test_event_creation(self):
        citation = Citation("event text", "source")
        event = Event("TestEvent", {"Date": "2000-01-01", "Label": "Test"}, citation)
        assert event.name == "TestEvent"
        assert event.date == "2000-01-01"
        assert event.label == "Test"


class TestRelations:
    """Tests for relation types."""

    def setup_method(self):
        Entity.clear_registry()

    def test_relation_creation(self):
        e1 = Entity("Subject", {})
        e2 = Entity("Object", {})
        rel = TYPE_OF(e1, e2)
        assert rel.subject == e1
        assert rel.obj == e2
        assert rel.relation_type == "TYPE_OF"

    def test_all_relation_types_exist(self):
        for rel_type in RELATION_TYPES:
            cls = get_relation_class(rel_type)
            assert cls is not None
            assert cls.relation_type == rel_type

    def test_relation_count(self):
        # Should have at least 80 relation types
        assert len(RELATION_TYPES) >= 80


class TestStorage:
    """Tests for SQLite storage."""

    def setup_method(self):
        Entity.clear_registry()
        self.storage = Storage(":memory:")

    def teardown_method(self):
        self.storage.close()

    def test_insert_entity(self):
        entity = Entity("TestEntity", {"prop": "value"})
        entity_id = self.storage.insert_entity(entity)
        assert entity_id is not None
        assert entity_id > 0

    def test_get_entity(self):
        original = Entity("RetrieveEntity", {"key": "val"})
        self.storage.insert_entity(original)
        retrieved = self.storage.get_entity("RetrieveEntity")
        assert retrieved is not None
        assert retrieved.name == "RetrieveEntity"

    def test_insert_citation(self):
        citation = Citation("regex", "source", {"Snippet": "snip"})
        citation_id = self.storage.insert_citation(citation)
        assert citation_id is not None

    def test_insert_property(self):
        entity = Entity("PropEntity", {})
        citation = Citation("cite", "src")
        self.storage.insert_entity(entity)
        prop_id = self.storage.insert_property(entity, "key", "value", citation)
        assert prop_id is not None

    def test_get_properties(self):
        entity = Entity("PropsEntity", {})
        self.storage.insert_entity(entity)
        self.storage.insert_property(entity, "key1", "val1", None)
        self.storage.insert_property(entity, "key2", "val2", None)

        props = self.storage.get_properties("PropsEntity")
        assert len(props) == 2
        keys = {p["key"] for p in props}
        assert keys == {"key1", "key2"}

    def test_insert_relation(self):
        e1 = Entity("Subject", {})
        e2 = Entity("Object", {})
        rel = TYPE_OF(e1, e2)
        citation = Citation("cite", "src")

        self.storage.insert_entity(e1)
        self.storage.insert_entity(e2)
        rel_id = self.storage.insert_relation(rel, citation)
        assert rel_id is not None

    def test_query_relations(self):
        e1 = Entity("A", {})
        e2 = Entity("B", {})
        e3 = Entity("C", {})

        self.storage.insert_entity(e1)
        self.storage.insert_entity(e2)
        self.storage.insert_entity(e3)

        self.storage.insert_relation(IS_A(e1, e2), None)
        self.storage.insert_relation(IS_A(e2, e3), None)
        self.storage.insert_relation(PART_OF(e1, e3), None)

        # Query all
        all_rels = self.storage.get_relations()
        assert len(all_rels) == 3

        # Query by type
        is_a_rels = self.storage.get_relations(rel_type="IS_A")
        assert len(is_a_rels) == 2

        # Query by subject
        a_rels = self.storage.get_relations(subject="A")
        assert len(a_rels) == 2


class TestConverter:
    """Tests for Mathematica to Python converter."""

    def test_convert_brackets(self):
        assert convert_brackets("Func[a, b]") == "Func(a, b)"
        assert convert_brackets('Func["test[string]"]') == 'Func("test[string]")'

    def test_convert_associations(self):
        assert convert_associations("<|a->b|>") == "{a->b}"
        assert convert_associations("<||>") == "{}"

    def test_convert_rules(self):
        assert convert_rules('"key"->value') == '"key": value'
        assert convert_rules('"text->arrow"') == '"text->arrow"'  # Inside string

    def test_convert_comments(self):
        assert convert_comments("(* comment *)") == "# comment"
        assert convert_comments("code (* inline *)") == "code # inline"

    def test_convert_add_entity(self):
        result = convert_add_entity('AddEntity("Wing",')
        # Should not change quoted names
        assert result == 'AddEntity("Wing",'

        result = convert_add_entity("Wing = AddEntity(\"Wing\",")
        assert "Wing" in result

    def test_full_conversion(self):
        m_code = '''
BeginPackage["WikiKG`"]
Begin["WikiKG`Private`"]

(* Article: Wing *)
AddEntity[Wing, <||>];
SetPropertyCited[Wing, "Medium", "air", Citation["air", "Wing", <|"Snippet"->"air"|>]];

End[]
EndPackage[]
'''
        py_code = convert_mathematica_to_python(m_code)

        # Should have import header
        assert "from wikikg import *" in py_code

        # Should not have package declarations
        assert "BeginPackage" not in py_code
        assert "EndPackage" not in py_code

        # Should have converted syntax
        assert "AddEntity(" in py_code
        assert "Citation(" in py_code
        assert "{" in py_code  # Converted from <||>


class TestAPIFunctions:
    """Tests for high-level API functions."""

    def setup_method(self):
        clear_all()
        init_db(":memory:")

    def teardown_method(self):
        close_db()

    def test_add_entity(self):
        wing = AddEntity("Wing", {"type": "appendage"})
        assert wing.name == "Wing"
        assert wing.props == {"type": "appendage"}

    def test_set_property_cited(self):
        entity = AddEntity("TestEntity", {})
        citation = Citation("source text", "Article")
        SetPropertyCited(entity, "key", "value", citation)

        assert entity.props["key"] == "value"

        props = query_properties("TestEntity")
        assert len(props) == 1
        assert props[0]["key"] == "key"
        assert props[0]["value"] == "value"

    def test_assert_cited(self):
        wing = AddEntity("Wing", {})
        fin = AddEntity("Fin", {})

        citation = Citation("type of fin", "Wing")
        AssertCited(TYPE_OF(wing, fin), citation)

        relations = query_relations(subject="Wing")
        assert len(relations) == 1
        assert relations[0]["type"] == "TYPE_OF"
        assert relations[0]["object"] == "Fin"

    def test_add_event(self):
        citation = Citation("event source", "Article")
        event = AddEvent("HistoricEvent", {
            "Date": "1969-07-20",
            "Label": "Moon Landing"
        }, citation)

        assert event.name == "HistoricEvent"
        assert event.date == "1969-07-20"

    def test_get_or_create_entity(self):
        # First call creates
        e1 = get_or_create_entity("NewEntity")
        assert e1.name == "NewEntity"

        # Second call returns existing
        e2 = get_or_create_entity("NewEntity")
        assert e1 is e2


class TestExpertaIntegration:
    """Tests for experta integration."""

    def test_facts_import(self):
        from wikikg.experta_facts import EntityFact, RelationFact, PropertyFact
        assert EntityFact is not None
        assert RelationFact is not None
        assert PropertyFact is not None

    def test_rules_import(self):
        from wikikg.experta_rules import WikiKGEngine, create_engine
        engine = create_engine()
        assert engine is not None

    def test_engine_transitive_inference(self):
        from wikikg.experta_rules import create_engine
        from wikikg.experta_facts import RelationFact

        engine = create_engine()
        engine.reset()
        engine.declare(RelationFact(rel_type="IS_A", subject="A", obj="B"))
        engine.declare(RelationFact(rel_type="IS_A", subject="B", obj="C"))
        engine.run()

        # After running, should have inferred IS_A(A, C)
        facts = list(engine.facts.values())
        is_a_facts = [f for f in facts
                      if isinstance(f, RelationFact) and f["rel_type"] == "IS_A"]

        # Should have original 2 + inferred 1 = 3
        assert len(is_a_facts) == 3
        subjects = {f["subject"] for f in is_a_facts}
        assert subjects == {"A", "B"}


class TestFileOperations:
    """Tests for file-based operations."""

    def test_convert_file(self):
        from wikikg.converter import convert_file

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.m"
            input_path.write_text('''
BeginPackage["Test`"]
AddEntity[Test, <||>];
End[]
''')
            output_path = convert_file(input_path)
            output_content = Path(output_path).read_text()

            assert "from wikikg import *" in output_content
            assert "AddEntity(" in output_content

    def test_storage_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Write data
            storage1 = Storage(db_path)
            entity = Entity("PersistentEntity", {"key": "value"})
            storage1.insert_entity(entity)
            storage1.close()

            Entity.clear_registry()

            # Read data
            storage2 = Storage(db_path)
            retrieved = storage2.get_entity("PersistentEntity")
            assert retrieved is not None
            assert retrieved.name == "PersistentEntity"
            storage2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
