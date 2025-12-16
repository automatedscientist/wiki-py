"""Replay verification for WikiKG tool-calling datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .storage import Storage


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_tool_arguments(arguments: str) -> dict[str, Any]:
    value = json.loads(arguments) if arguments else {}
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value


def verify_example(storage: Storage, example: dict[str, Any]) -> tuple[bool, str]:
    metadata = example.get("metadata") or {}
    tool_calls = metadata.get("tool_calls") or []
    hop_targets = metadata.get("hop_targets") or []

    if not tool_calls:
        return False, "missing metadata.tool_calls"
    if len(tool_calls) != len(hop_targets):
        return False, "metadata.tool_calls length mismatch"

    messages = example.get("messages") or []
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assistant_calls = [m for m in messages if m.get("role") == "assistant" and "tool_calls" in m]

    if len(tool_messages) != len(tool_calls) or len(assistant_calls) != len(tool_calls):
        return False, "messages do not contain expected tool-call sequence"

    for hop_index, (call_meta, assistant_msg, tool_msg) in enumerate(
        zip(tool_calls, assistant_calls, tool_messages),
        start=1,
    ):
        expected_next = call_meta.get("expected_next")
        relation_type = call_meta.get("relation_type")
        direction = call_meta.get("direction")
        tool = call_meta.get("tool")

        tool_call = assistant_msg["tool_calls"][0]
        tool_name = tool_call["function"]["name"]
        args = parse_tool_arguments(tool_call["function"]["arguments"])

        if tool_name != tool:
            return False, f"tool name mismatch at hop {hop_index}"

        try:
            stored_output = json.loads(tool_msg.get("content") or "null")
        except json.JSONDecodeError:
            return False, f"stored tool output is not JSON at hop {hop_index}"

        if tool == "query_relations":
            subject = args.get("subject") or None
            obj = args.get("obj") or None
            rel_type = args.get("rel_type") or None

            results = storage.get_relations(subject=subject, obj=obj, rel_type=rel_type)

            if direction == "forward":
                required = {"type": relation_type, "subject": subject, "object": expected_next}
            else:
                required = {"type": relation_type, "subject": expected_next, "object": obj}

            if not any(
                r.get("type") == required["type"]
                and r.get("subject") == required["subject"]
                and r.get("object") == required["object"]
                for r in results
            ):
                return False, f"required edge not in query_relations results at hop {hop_index}"

            if isinstance(stored_output, list) and not any(
                r.get("type") == required["type"]
                and r.get("subject") == required["subject"]
                and r.get("object") == required["object"]
                for r in stored_output
            ):
                return False, f"required edge not in stored tool output at hop {hop_index}"

        elif tool == "get_neighbors":
            entity_name = args["entity_name"]
            results = storage.get_neighbors(entity_name, direction=args.get("direction", "both"))
            if not any(
                r.get("neighbor") == expected_next
                and r.get("relation_type") == relation_type
                and r.get("direction") == direction
                for r in results
            ):
                return False, f"required neighbor not in get_neighbors results at hop {hop_index}"

            if isinstance(stored_output, list) and not any(
                r.get("neighbor") == expected_next
                and r.get("relation_type") == relation_type
                and r.get("direction") == direction
                for r in stored_output
            ):
                return False, f"required neighbor not in stored tool output at hop {hop_index}"
        else:
            return False, f"unsupported tool: {tool}"

    return True, "ok"

