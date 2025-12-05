#!/usr/bin/env python3
"""Export WikiKG data in structured formats for LLM training.

Usage:
    uv run python scripts/export_dataset.py triplets -o data/triplets.jsonl
    uv run python scripts/export_dataset.py profiles --min-relations 5 -o data/profiles.jsonl
    uv run python scripts/export_dataset.py paths --num-paths 10000 -o data/paths.jsonl
    uv run python scripts/export_dataset.py neighborhoods -o data/neighborhoods.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wikikg.datasets import DatasetGenerator
from wikikg.storage import Storage


def main():
    parser = argparse.ArgumentParser(
        description="Export WikiKG data for LLM training"
    )
    parser.add_argument(
        "data_type",
        choices=["triplets", "profiles", "neighborhoods", "paths"],
        help="Type of data to export"
    )
    parser.add_argument(
        "--db", "-d",
        type=Path,
        default=Path("data/wikikg.db"),
        help="Path to WikiKG database"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Maximum number of records to export"
    )
    parser.add_argument(
        "--min-relations",
        type=int,
        default=1,
        help="Minimum relations for entity profiles"
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=10000,
        help="Number of paths to generate"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum path length"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=6,
        help="Maximum path length"
    )
    parser.add_argument(
        "--relation-types",
        nargs="+",
        help="Filter to specific relation types"
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=0,
        help="Minimum degree for entities"
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading database: {args.db}")
    storage = Storage(args.db)

    print(f"Entities: {storage.count_entities():,}")
    print(f"Relations: {storage.count_relations():,}")

    gen = DatasetGenerator(storage)

    # Apply filters
    if args.relation_types:
        print(f"Filtering to relation types: {args.relation_types}")
        gen.filter_by_relation_types(args.relation_types)

    if args.min_degree > 0:
        print(f"Filtering to entities with >= {args.min_degree} relations")
        gen.filter_connected_only(args.min_degree)

    # Build kwargs for extraction
    kwargs = {}

    if args.data_type == "triplets":
        kwargs["limit"] = args.limit

    elif args.data_type == "profiles":
        kwargs["min_relations"] = args.min_relations
        kwargs["limit"] = args.limit

    elif args.data_type == "neighborhoods":
        kwargs["limit"] = args.limit

    elif args.data_type == "paths":
        kwargs["num_paths"] = args.num_paths
        kwargs["min_length"] = args.min_length
        kwargs["max_length"] = args.max_length

    # Export
    print(f"Exporting {args.data_type} to {args.output}...")
    count = gen.to_jsonl(args.output, args.data_type, **kwargs)
    print(f"Exported {count:,} records to {args.output}")

    storage.close()


if __name__ == "__main__":
    main()
