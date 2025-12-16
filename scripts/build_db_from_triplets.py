#!/usr/bin/env python3
"""Build a WikiKG SQLite DB from a triplets.jsonl file.

This is a lightweight way to bootstrap a small database for dataset generation.

Usage:
  uv run python scripts/build_db_from_triplets.py --triplets assets/triplets.jsonl --db data/wikikg.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wikikg.entities import Entity
from wikikg.relations import get_relation_class
from wikikg.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WikiKG SQLite DB from triplets JSONL")
    parser.add_argument("--triplets", type=Path, required=True, help="Input triplets.jsonl")
    parser.add_argument("--db", type=Path, required=True, help="Output SQLite DB path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max triplets to ingest")
    args = parser.parse_args()

    if not args.triplets.exists():
        print(f"Error: triplets file not found: {args.triplets}", file=sys.stderr)
        raise SystemExit(2)

    args.db.parent.mkdir(parents=True, exist_ok=True)

    Entity.clear_registry()
    storage = Storage(str(args.db))
    try:
        inserted = 0
        skipped_unknown_rel = 0
        with open(args.triplets, "r", encoding="utf-8") as f:
            for line in f:
                if args.limit is not None and inserted >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rel_type = row["relation"]
                rel_cls = get_relation_class(rel_type)
                if rel_cls is None:
                    skipped_unknown_rel += 1
                    continue

                subj = Entity.get_or_create(row["subject"])
                obj = Entity.get_or_create(row["object"])
                storage.insert_relation(rel_cls(subj, obj), None)
                inserted += 1

        print(f"Built DB at {args.db} with {storage.count_entities():,} entities and {storage.count_relations():,} relations")
        if skipped_unknown_rel:
            print(f"Skipped {skipped_unknown_rel:,} triplets with unknown relation types", file=sys.stderr)
    finally:
        storage.close()


if __name__ == "__main__":
    main()

