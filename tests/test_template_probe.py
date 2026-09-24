from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.template_probe import write_template_probe


class _ThinkingTokenizer:
    def apply_chat_template(self, messages, *, tokenize, enable_thinking=False, **kwargs):
        text = f"thinking={enable_thinking}|{messages[0]['content']}"
        if tokenize:
            return [1, int(enable_thinking), 2]
        return text


class _IgnoringTokenizer:
    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        if tokenize:
            return [1, 2]
        return messages[0]["content"]


class _BatchEncodingTokenizer(_ThinkingTokenizer):
    def apply_chat_template(self, messages, *, tokenize, enable_thinking=False, **kwargs):
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        return {"input_ids": rendered, "attention_mask": [1, 1, 1]} if tokenize else rendered


class TemplateProbeTest(unittest.TestCase):
    def test_records_effective_thinking_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            payload = write_template_probe(
                tokenizer=_ThinkingTokenizer(),
                model_name="example/model",
                model_revision="abc123",
                active_kwargs={"enable_thinking": True},
                output_path=str(path),
                require_thinking_effect=True,
            )

            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(payload["thinking_switch_changes_tokens"])
        self.assertEqual(saved["active_template_kwargs"], {"enable_thinking": True})

    def test_required_switch_fails_when_template_ignores_it(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not change"):
            write_template_probe(
                tokenizer=_IgnoringTokenizer(),
                model_name="example/model",
                model_revision=None,
                active_kwargs={"enable_thinking": True},
                output_path="",
                require_thinking_effect=True,
            )

    def test_accepts_batch_encoding_mapping(self) -> None:
        payload = write_template_probe(
            tokenizer=_BatchEncodingTokenizer(),
            model_name="example/model",
            model_revision=None,
            active_kwargs={"enable_thinking": True},
            output_path="",
            require_thinking_effect=True,
        )

        self.assertEqual(payload["active"]["token_ids"], [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
