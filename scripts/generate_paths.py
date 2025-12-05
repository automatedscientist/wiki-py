#!/usr/bin/env python3
"""Generate random walk paths from WikiKG for LLM training.

This script generates random paths through the knowledge graph,
outputting them in either text or JSONL format for training data.

The database only contains successfully imported data (from valid .py files),
so all generated paths are from verified knowledge.

Usage:
    uv run python scripts/generate_paths.py --db data/wikikg.db --num-paths 10000
    uv run python scripts/generate_paths.py --db data/wikikg.db --format jsonl -o data/paths.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wikikg.storage import Storage


def generate_paths(
    storage: Storage,
    num_paths: int,
    min_length: int,
    max_length: int,
    allow_cycles: bool = False
) -> list[list[dict]]:
    """Generate multiple random paths.

    Args:
        storage: Database storage instance
        num_paths: Number of paths to generate
        min_length: Minimum path length (retries if shorter)
        max_length: Maximum path length
        allow_cycles: Whether to allow cycles in paths

    Returns:
        List of paths (each path is a list of step dicts)
    """
    paths = []
    attempts = 0
    max_attempts = num_paths * 10  # Prevent infinite loops

    while len(paths) < num_paths and attempts < max_attempts:
        attempts += 1
        path = storage.random_walk(start=None, steps=max_length, allow_cycles=allow_cycles)

        # Only keep paths meeting minimum length
        if len(path) >= min_length:
            paths.append(path)

        if attempts % 1000 == 0:
            print(f"Generated {len(paths)}/{num_paths} paths ({attempts} attempts)")

    return paths


def format_text(path: list[dict]) -> str:
    """Format path as text string.

    Example: "Einstein -[FIELD_OF_WORK]-> Physics -[STUDIED_BY]-> Feynman"
    """
    if not path:
        return ""

    parts = [path[0]["entity"]]
    for step in path[1:]:
        arrow = "->" if step["direction"] == "forward" else "<-"
        parts.append(f" -[{step['relation']}]{arrow} {step['entity']}")

    return "".join(parts)


def format_jsonl(path: list[dict]) -> str:
    """Format path as JSON line."""
    return json.dumps({
        "path": path,
        "length": len(path),
        "text": format_text(path)
    })


def main():
    parser = argparse.ArgumentParser(
        description="Generate random walk paths from WikiKG"
    )
    parser.add_argument(
        "--db", "-d",
        type=Path,
        default=Path("data/wikikg.db"),
        help="Path to WikiKG database"
    )
    parser.add_argument(
        "--num-paths", "-n",
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
        default=10,
        help="Maximum path length"
    )
    parser.add_argument(
        "--allow-cycles",
        action="store_true",
        help="Allow revisiting entities in paths"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "jsonl"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (stdout if not specified)"
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading database: {args.db}", file=sys.stderr)
    storage = Storage(args.db)

    entity_count = storage.count_entities()
    relation_count = storage.count_relations()
    print(f"Database has {entity_count:,} entities and {relation_count:,} relations", file=sys.stderr)

    print(f"Generating {args.num_paths:,} paths (length {args.min_length}-{args.max_length})...", file=sys.stderr)
    paths = generate_paths(
        storage,
        args.num_paths,
        args.min_length,
        args.max_length,
        args.allow_cycles
    )

    print(f"Generated {len(paths):,} paths", file=sys.stderr)

    # Format and output
    formatter = format_jsonl if args.format == "jsonl" else format_text

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for path in paths:
                f.write(formatter(path) + "\n")
        print(f"Wrote {len(paths):,} paths to {args.output}", file=sys.stderr)
    else:
        for path in paths:
            print(formatter(path))

    storage.close()


if __name__ == "__main__":
    main()
