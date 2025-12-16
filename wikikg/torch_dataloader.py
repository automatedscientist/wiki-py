"""PyTorch Dataset/DataLoader helpers for WikiKG tool-calling JSONL files.

The JSONL format is produced by `scripts/generate_tool_dataset.py` and contains:
  - tools: tool schema list
  - messages: chat-style messages with tool_calls/tool outputs
  - metadata: path + provenance

This module avoids importing torch at import-time so the core package remains usable
without the optional PyTorch dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence


JsonDict = dict[str, Any]
RenderMode = Literal["transcript", "final"]


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for wikikg.torch_dataloader. Install torch and try again."
        ) from e
    return torch


@dataclass(frozen=True)
class ToolCallsRenderConfig:
    include_tools: bool = True
    include_tool_outputs: bool = True
    include_metadata: bool = False


def render_tool_calls_example(
    example: JsonDict,
    *,
    config: ToolCallsRenderConfig | None = None,
) -> str:
    """Render one JSONL record to a single training string."""
    cfg = config or ToolCallsRenderConfig()

    parts: list[str] = []

    if cfg.include_tools and example.get("tools") is not None:
        parts.append("### Tools")
        parts.append(json.dumps(example["tools"], ensure_ascii=False, sort_keys=True))

    parts.append("### Conversation")
    for msg in example.get("messages", []):
        role = msg.get("role", "unknown")

        if role == "assistant" and "tool_calls" in msg:
            parts.append("ASSISTANT: <tool_calls>")
            parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False, sort_keys=True))
            continue

        if role == "tool":
            parts.append(f"TOOL[{msg.get('name', '')}]:")
            if cfg.include_tool_outputs:
                parts.append(msg.get("content", ""))
            else:
                parts.append("<omitted>")
            continue

        content = msg.get("content", "")
        parts.append(f"{role.upper()}: {content}")

    if cfg.include_metadata and example.get("metadata") is not None:
        parts.append("### Metadata")
        parts.append(json.dumps(example["metadata"], ensure_ascii=False, sort_keys=True))

    return "\n".join(parts).strip() + "\n"


class _JsonlOffsetIndex:
    """Byte-offset index for newline-delimited JSON (fast random access)."""

    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = []
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)


class ToolCallsJsonlDataset:
    """Map-style dataset (random-access) for tool-calling JSONL."""

    def __init__(
        self,
        filename: str | Path,
        *,
        render_mode: RenderMode = "transcript",
        render_config: ToolCallsRenderConfig | None = None,
    ) -> None:
        torch = _require_torch()
        self._torch = torch
        self.path = Path(filename)
        self.index = _JsonlOffsetIndex(self.path)
        self.render_mode = render_mode
        self.render_config = render_config or ToolCallsRenderConfig()

    def __len__(self) -> int:
        return len(self.index)

    def _read_json_at(self, offset: int) -> JsonDict:
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line)

    def __getitem__(self, idx: int) -> JsonDict:
        example = self._read_json_at(self.index.offsets[idx])
        if self.render_mode == "transcript":
            example["_text"] = render_tool_calls_example(example, config=self.render_config)
        elif self.render_mode == "final":
            messages = example.get("messages") or []
            user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
            final = next(
                (m.get("content") for m in reversed(messages) if m.get("role") == "assistant" and "content" in m),
                "",
            )
            example["_input_text"] = (user or "").strip() + "\n"
            example["_label_text"] = (final or "").strip() + "\n"
        else:
            raise ValueError(f"Unknown render_mode: {self.render_mode}")
        return example


class ToolCallsJsonlIterableDataset:
    """IterableDataset for streaming tool-calling JSONL (no random access, no shuffle)."""

    def __init__(
        self,
        filename: str | Path,
        *,
        render_mode: RenderMode = "transcript",
        render_config: ToolCallsRenderConfig | None = None,
    ) -> None:
        torch = _require_torch()
        self._torch = torch
        self.path = Path(filename)
        self.render_mode = render_mode
        self.render_config = render_config or ToolCallsRenderConfig()

    def __iter__(self) -> Iterable[JsonDict]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                if self.render_mode == "transcript":
                    example["_text"] = render_tool_calls_example(example, config=self.render_config)
                elif self.render_mode == "final":
                    messages = example.get("messages") or []
                    user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
                    final = next(
                        (m.get("content") for m in reversed(messages) if m.get("role") == "assistant" and "content" in m),
                        "",
                    )
                    example["_input_text"] = (user or "").strip() + "\n"
                    example["_label_text"] = (final or "").strip() + "\n"
                else:
                    raise ValueError(f"Unknown render_mode: {self.render_mode}")
                yield example


class ToolCallsCollator:
    """Collate records into batched tensors (optional tokenization)."""

    def __init__(
        self,
        *,
        tokenizer: Any | None = None,
        max_length: int | None = None,
        return_raw: bool = False,
        mode: RenderMode = "transcript",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_raw = return_raw
        self.mode = mode

    def __call__(self, batch: Sequence[JsonDict]) -> JsonDict:
        if self.mode == "transcript":
            texts = [ex["_text"] for ex in batch]
            payload: JsonDict = {"text": texts, "metadata": [ex.get("metadata") for ex in batch]}
        else:
            inputs = [ex["_input_text"] for ex in batch]
            labels = [ex["_label_text"] for ex in batch]
            payload = {"input_text": inputs, "label_text": labels, "metadata": [ex.get("metadata") for ex in batch]}

        if self.tokenizer is not None:
            kwargs: JsonDict = {"padding": True, "truncation": True, "return_tensors": "pt"}
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length

            if self.mode == "transcript":
                payload["tokens"] = self.tokenizer(payload["text"], **kwargs)
            else:
                payload["input_tokens"] = self.tokenizer(payload["input_text"], **kwargs)
                payload["label_tokens"] = self.tokenizer(payload["label_text"], **kwargs)

        if self.return_raw:
            payload["raw"] = list(batch)

        return payload


def build_tool_calls_dataloader(
    filename: str | Path,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    mode: RenderMode = "transcript",
    include_tools: bool = True,
    include_tool_outputs: bool = True,
    include_metadata: bool = False,
    stream: bool = False,
) -> Any:
    """Create a PyTorch DataLoader from a tool_calls JSONL filename."""
    torch = _require_torch()
    from torch.utils.data import DataLoader  # type: ignore

    render_cfg = ToolCallsRenderConfig(
        include_tools=include_tools,
        include_tool_outputs=include_tool_outputs,
        include_metadata=include_metadata,
    )

    if stream:
        if shuffle:
            raise ValueError("shuffle=True is not supported for stream=True iterable datasets")
        dataset = ToolCallsJsonlIterableDataset(filename, render_mode=mode, render_config=render_cfg)
    else:
        dataset = ToolCallsJsonlDataset(filename, render_mode=mode, render_config=render_cfg)

    collate_fn = ToolCallsCollator(tokenizer=tokenizer, max_length=max_length, mode=mode)

    loader_kwargs: JsonDict = {
        "batch_size": batch_size,
        "shuffle": shuffle if not stream else False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if persistent_workers is not None:
        loader_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(dataset, **loader_kwargs)


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render a few DataLoader batches from a WikiKG tool_calls JSONL file")
    parser.add_argument("filename", help="Path to tool_calls_*.jsonl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--mode", choices=["transcript", "final"], default="transcript")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling (recommended for streaming)")
    parser.add_argument("--stream", action="store_true", help="Use streaming IterableDataset (no random access)")
    parser.add_argument("--max-length", type=int, default=None, help="Optional tokenizer truncation length")
    parser.add_argument("--no-tools", action="store_true", help="Omit tool schema from rendered text")
    parser.add_argument("--no-tool-outputs", action="store_true", help="Omit tool outputs from rendered text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    dl = build_tool_calls_dataloader(
        args.filename,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        mode=args.mode,  # type: ignore[arg-type]
        stream=args.stream,
        include_tools=not args.no_tools,
        include_tool_outputs=not args.no_tool_outputs,
        include_metadata=False,
        max_length=args.max_length,
        tokenizer=None,
    )

    for i, batch in enumerate(dl):
        if i >= args.num_batches:
            break
        if args.mode == "transcript":
            print(f"\n=== batch {i} ===")
            print(batch["text"][0])
        else:
            print(f"\n=== batch {i} ===")
            print("INPUT:")
            print(batch["input_text"][0])
            print("LABEL:")
            print(batch["label_text"][0])

    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(_main())
    except ImportError as e:
        print(str(e))
        raise SystemExit(2)
