"""Download and convert wiki-kg-dataset to experta-compatible format.

This script:
1. Downloads the dataset from HuggingFace
2. Extracts the polaris_alpha column (Mathematica code)
3. Converts each entry to Python using the wikikg converter
4. Loads into SQLite database for experta integration
"""

import sys
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm

# Add parent to path for wikikg imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wikikg import init_db, close_db, clear_all
from wikikg.converter import convert_mathematica_to_python
from wikikg.storage import Storage


def download_dataset(num_samples: int | None = None):
    """Download dataset from HuggingFace.

    Args:
        num_samples: Optional limit on number of samples to process

    Returns:
        Dataset iterator
    """
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset("AutomatedScientist/wiki-kg-dataset", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Loaded {len(dataset)} samples")
    return dataset


def convert_entry(article_name: str, m_code: str) -> str:
    """Convert a single Mathematica entry to Python.

    Args:
        article_name: Name of the Wikipedia article
        m_code: Mathematica code from polaris_alpha column

    Returns:
        Converted Python code
    """
    py_code = convert_mathematica_to_python(m_code)
    # Add article name as comment
    header = f"# Knowledge graph for: {article_name}\n"
    return header + py_code


def save_converted_files(
    dataset,
    output_dir: Path,
    max_files: int | None = None
) -> int:
    """Save converted Python files to disk.

    Args:
        dataset: HuggingFace dataset
        output_dir: Directory to save .py files
        max_files: Optional limit on number of files

    Returns:
        Number of files saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, sample in enumerate(tqdm(dataset, desc="Converting")):
        if max_files and count >= max_files:
            break

        article_name = sample["article_name"]
        m_code = sample.get("polaris_alpha", "")

        if not m_code or not m_code.strip():
            continue

        # Create safe filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in article_name)
        safe_name = safe_name[:100]  # Limit length
        filename = f"{i:05d}_{safe_name}.py"

        try:
            py_code = convert_entry(article_name, m_code)
            (output_dir / filename).write_text(py_code, encoding="utf-8")
            count += 1
        except Exception as e:
            print(f"Error converting {article_name}: {e}")
            continue

    return count


def load_to_database(
    dataset,
    db_path: Path,
    max_entries: int | None = None,
    log_file: Path | None = None,
    success_log_file: Path | None = None
) -> tuple[int, int, int, int]:
    """Load converted data directly into SQLite database.

    Args:
        dataset: HuggingFace dataset
        db_path: Path to SQLite database
        max_entries: Optional limit on entries to process
        log_file: Optional file to log errors
        success_log_file: Optional file to log successful imports

    Returns:
        Tuple of (entities_count, relations_count, success_count, errors_count)
    """
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    storage = Storage(db_path)

    total_entities = 0
    total_relations = 0
    success = 0
    errors = 0
    error_log = []
    success_log = []

    for i, sample in enumerate(tqdm(dataset, desc="Loading to DB")):
        if max_entries and i >= max_entries:
            break

        article_name = sample["article_name"]
        m_code = sample.get("polaris_alpha", "")

        if not m_code or not m_code.strip():
            continue

        try:
            py_code = convert_entry(article_name, m_code)
            # Remove import line - we use namespace functions instead
            py_code = py_code.replace('from wikikg import *', '')

            # Execute the converted code in a controlled namespace
            namespace = create_execution_namespace(storage)
            exec(py_code, namespace)

            total_entities += namespace.entity_count
            total_relations += namespace.relation_count
            success += 1
            success_log.append(f"[{i}] {article_name}: {namespace.entity_count} entities, {namespace.relation_count} relations")

        except Exception as e:
            errors += 1
            error_msg = f"[{i}] {article_name}: {type(e).__name__}: {e}"
            error_log.append(error_msg)
            continue

    storage.close()

    # Write error log if requested
    if log_file and error_log:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"Total errors: {errors}\n\n")
            for err in error_log:
                f.write(err + "\n")
        print(f"Error log written to: {log_file}")

    # Write success log if requested
    if success_log_file and success_log:
        success_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(success_log_file, "w") as f:
            f.write(f"Total successful imports: {success}\n\n")
            for entry in success_log:
                f.write(entry + "\n")
        print(f"Success log written to: {success_log_file}")

    return total_entities, total_relations, success, errors


class DynamicRelation:
    """Dynamic relation class that auto-creates missing relation types."""
    def __init__(self, rel_type: str, subject, obj):
        self.relation_type = rel_type
        self.subject = subject
        self.obj = obj


class DynamicNamespace(dict):
    """Namespace that auto-creates missing entities and relations."""

    def __init__(self, storage: Storage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage = storage
        self.entity_count = 0
        self.relation_count = 0

    def __missing__(self, key):
        # If it looks like a relation type (ALL_CAPS, with or without underscores)
        # Relations are typically: RELATION_TYPE, RELATION, etc.
        # Entities are typically: CamelCase or Capitalized
        import re
        is_relation = (
            key.isupper() and  # All uppercase
            re.match(r'^[A-Z][A-Z0-9_]*$', key) and  # Starts with letter, only caps/digits/underscores
            len(key) > 1  # At least 2 chars
        )

        if is_relation:
            def make_relation(subject, obj):
                return DynamicRelation(key, subject, obj)
            self[key] = make_relation
            return make_relation

        # Otherwise, treat as entity reference - auto-create
        from wikikg.entities import Entity
        entity = Entity.get_or_create(key)
        self.storage.insert_entity(entity)
        self.entity_count += 1
        self[key] = entity
        return entity


def create_execution_namespace(storage: Storage) -> dict:
    """Create a namespace for executing converted Python code.

    This provides all the wikikg functions that the converted code expects.
    Uses DynamicNamespace to auto-create missing entities and relations.
    """
    from wikikg.entities import Entity, Citation, Event
    from wikikg.relations import Relation, RELATION_TYPES, get_relation_class

    namespace = DynamicNamespace(storage)

    def AddEntity(name: str, props: dict = None) -> Entity:
        props = props if props is not None else {}
        entity = Entity.get_or_create(name)
        entity.props.update(props)
        storage.insert_entity(entity)
        namespace.entity_count += 1
        namespace[name] = entity  # Register in namespace
        return entity

    def SetPropertyCited(entity, key, value, citation=None):
        if hasattr(entity, 'name'):
            entity.props[key] = value
            storage.insert_property(entity, key, value, citation)

    def AssertCited(relation, citation=None):
        if isinstance(relation, Relation):
            storage.insert_relation(relation, citation)
            namespace.relation_count += 1
        elif isinstance(relation, DynamicRelation):
            # Handle dynamic relations
            from wikikg.relations import Relation as BaseRelation

            class TempRelation(BaseRelation):
                relation_type = relation.relation_type

            temp_rel = TempRelation(relation.subject, relation.obj)
            storage.insert_relation(temp_rel, citation)
            namespace.relation_count += 1

    def AddEvent(name, props, citation=None):
        citation = citation or Citation("", "", {})
        event = Event(name, props, citation)
        storage.insert_event(event)
        return event

    # Add base classes and functions
    namespace.update({
        "Entity": Entity,
        "Citation": Citation,
        "Event": Event,
        "Relation": Relation,
        "AddEntity": AddEntity,
        "SetPropertyCited": SetPropertyCited,
        "AssertCited": AssertCited,
        "AddEvent": AddEvent,
    })

    # Add all known relation classes
    for rel_name in RELATION_TYPES:
        rel_class = get_relation_class(rel_name)
        if rel_class:
            namespace[rel_name] = rel_class

    return namespace


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert wiki-kg-dataset to experta-compatible format"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/converted"),
        help="Output directory for .py files"
    )
    parser.add_argument(
        "--db-path", "-d",
        type=Path,
        default=Path("data/wikikg.db"),
        help="SQLite database path"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--save-files",
        action="store_true",
        help="Save converted .py files to output directory"
    )
    parser.add_argument(
        "--load-db",
        action="store_true",
        help="Load data into SQLite database"
    )

    args = parser.parse_args()

    # Default to both if neither specified
    if not args.save_files and not args.load_db:
        args.load_db = True

    # Download dataset
    dataset = download_dataset(args.max_samples)

    # Save files if requested
    if args.save_files:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        count = save_converted_files(dataset, args.output_dir, args.max_samples)
        print(f"Saved {count} converted files to {args.output_dir}")

    # Load to database if requested
    if args.load_db:
        args.db_path.parent.mkdir(parents=True, exist_ok=True)
        from wikikg.entities import Entity
        Entity.clear_registry()  # Clear any existing entities

        log_file = args.db_path.parent / "errors.log"
        success_log_file = args.db_path.parent / "success.log"
        entities, relations, success, errors = load_to_database(
            dataset, args.db_path, args.max_samples, log_file, success_log_file
        )
        print(f"\nDatabase: {args.db_path}")
        print(f"  Success: {success}")
        print(f"  Entities: {entities}")
        print(f"  Relations: {relations}")
        print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
