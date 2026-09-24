"""Model chat-template validation for reproducible record runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _token_ids(tokenizer, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _rendered_text(tokenizer, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    return str(rendered)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_template_probe(
    *,
    tokenizer,
    model_name: str,
    model_revision: str | None,
    active_kwargs: dict[str, Any],
    output_path: str,
    require_thinking_effect: bool,
) -> dict[str, Any]:
    """Render active/on/off templates and optionally require a real thinking switch."""
    messages = [{"role": "user", "content": "Template validation probe."}]
    enabled_kwargs = {**active_kwargs, "enable_thinking": True}
    disabled_kwargs = {**active_kwargs, "enable_thinking": False}
    active_ids = _token_ids(tokenizer, messages, active_kwargs)
    enabled_ids = _token_ids(tokenizer, messages, enabled_kwargs)
    disabled_ids = _token_ids(tokenizer, messages, disabled_kwargs)
    active_text = _rendered_text(tokenizer, messages, active_kwargs)
    enabled_text = _rendered_text(tokenizer, messages, enabled_kwargs)
    disabled_text = _rendered_text(tokenizer, messages, disabled_kwargs)
    thinking_switch_changes_tokens = enabled_ids != disabled_ids
    if require_thinking_effect and not thinking_switch_changes_tokens:
        raise RuntimeError(
            f"{model_name} did not change its rendered prompt when enable_thinking changed."
        )

    payload = {
        "artifact_schema_version": 1,
        "model": str(model_name),
        "model_revision": model_revision,
        "messages": messages,
        "active_template_kwargs": active_kwargs,
        "require_thinking_template_effect": bool(require_thinking_effect),
        "thinking_switch_changes_tokens": bool(thinking_switch_changes_tokens),
        "active": {
            "token_count": len(active_ids),
            "token_ids": active_ids,
            "rendered_text": active_text,
            "rendered_text_sha256": _sha256_text(active_text),
        },
        "thinking_enabled": {
            "token_count": len(enabled_ids),
            "token_ids": enabled_ids,
            "rendered_text": enabled_text,
            "rendered_text_sha256": _sha256_text(enabled_text),
        },
        "thinking_disabled": {
            "token_count": len(disabled_ids),
            "token_ids": disabled_ids,
            "rendered_text": disabled_text,
            "rendered_text_sha256": _sha256_text(disabled_text),
        },
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
        print(
            "[template-probe] "
            f"thinking_switch_changes_tokens={thinking_switch_changes_tokens} path={path}",
            flush=True,
        )
    return payload
