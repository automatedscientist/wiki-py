#!/usr/bin/env python3
"""Generate a replayable tool-calling dataset from WikiKG paths.

Usage:
  uv run python scripts/generate_tool_dataset.py --db data/wikikg.db --paths data/paths.jsonl -o data/tool_calls.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wikikg.storage import Storage
from wikikg.tool_dataset import (
    ToolCallStyle,
    default_provenance,
    generate_from_paths,
    load_paths_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tool-calling dataset from paths.jsonl")
    parser.add_argument("--db", type=Path, default=Path("data/wikikg.db"), help="Path to WikiKG database")
    parser.add_argument("--paths", type=Path, required=True, help="Input paths JSONL (GraphPath or scripts/generate_paths.py format)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output dataset JSONL")
    parser.add_argument("--num-examples", type=int, default=1000, help="Number of examples to write")
    parser.add_argument("--min-hops", type=int, default=2, help="Minimum hops per example")
    parser.add_argument("--max-hops", type=int, default=6, help="Maximum hops per example")
    parser.add_argument("--style", choices=["query_relations", "get_neighbors"], default="query_relations", help="Tool-call style")
    parser.add_argument("--max-tool-results", type=int, default=50, help="Max tool results per call (required edge is always included)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling/sampling")
    parser.add_argument("--keep-single-hop-answerable", action="store_true", help="Do not drop examples where answer is a direct neighbor of the start")

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: database not found: {args.db}", file=sys.stderr)
        raise SystemExit(2)
    if not args.paths.exists():
        print(f"Error: paths file not found: {args.paths}", file=sys.stderr)
        raise SystemExit(2)

    storage = Storage(args.db)
    try:
        paths = load_paths_jsonl(args.paths)
        provenance = default_provenance(args.db, repo_root=Path(__file__).parent.parent)
        examples, stats = generate_from_paths(
            storage,
            paths,
            num_examples=args.num_examples,
            style=ToolCallStyle(args.style),
            max_tool_results=args.max_tool_results,
            seed=args.seed,
            drop_if_answer_is_direct_neighbor=not args.keep_single_hop_answerable,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            provenance=provenance,
            dedup=True,
            return_stats=True,
        )
        count = write_jsonl(examples, args.output)
        print(
            f"Wrote {count} examples to {args.output} "
            f"(considered={stats.considered}, skipped_duplicates={stats.skipped_duplicates})"
        )
    finally:
        storage.close()


if __name__ == "__main__":
    main()
