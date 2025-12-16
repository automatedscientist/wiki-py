#!/usr/bin/env python3
"""Replay and verify a tool-calling dataset against a WikiKG database.

Usage:
  uv run python scripts/verify_tool_dataset.py --db data/wikikg.db --dataset data/tool_calls.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wikikg.storage import Storage
from wikikg.tool_dataset_verify import load_jsonl, verify_example


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify tool-calling dataset by replaying tool calls")
    parser.add_argument("--db", type=Path, required=True, help="Path to WikiKG database")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples verified")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: database not found: {args.db}", file=sys.stderr)
        raise SystemExit(2)
    if not args.dataset.exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        raise SystemExit(2)

    examples = load_jsonl(args.dataset)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    storage = Storage(args.db)
    try:
        failures: list[str] = []
        for i, ex in enumerate(examples):
            ok, reason = verify_example(storage, ex)
            if not ok:
                failures.append(f"example[{i}]: {reason}")

        if failures:
            for msg in failures[:20]:
                print(msg, file=sys.stderr)
            print(f"FAILED: {len(failures)}/{len(examples)} examples invalid", file=sys.stderr)
            raise SystemExit(1)

        print(f"OK: {len(examples)}/{len(examples)} examples verified")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
