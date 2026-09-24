from __future__ import annotations

import unittest

import torch

from utils.llm_generator import HFTextGenerator


class _TokenizerStub:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.last_template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_template_kwargs = kwargs
        text = messages[0]["content"]
        width = len(text) + 2
        return {
            "input_ids": torch.arange(width, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, width), dtype=torch.long),
        }


class LLMGeneratorBatchInputsTest(unittest.TestCase):
    def test_completion_extraction_keeps_first_eos_and_trims_distinct_pad(self) -> None:
        generator = HFTextGenerator(
            model=None,
            tokenizer=_TokenizerStub(),
            device=torch.device("cpu"),
            max_new_tokens=10,
            input_mode="chat",
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

        with_eos = generator._completion_ids_from_generated_row(
            torch.tensor([8, 2, 0, 0]),
            eos_id=2,
            pad_id=0,
        )
        without_eos = generator._completion_ids_from_generated_row(
            torch.tensor([8, 9, 0, 0]),
            eos_id=2,
            pad_id=0,
        )

        self.assertEqual(with_eos, [8, 2])
        self.assertEqual(without_eos, [8, 9])

    def test_batch_prompt_encoding_stays_on_cpu_until_final_transfer(self) -> None:
        generator = HFTextGenerator(
            model=None,
            tokenizer=_TokenizerStub(),
            device=torch.device("cpu"),
            max_new_tokens=10,
            input_mode="chat",
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
        )

        inputs, counts = generator.prompts_to_model_inputs(["a", "abcd"])

        self.assertEqual(counts, [3, 6])
        self.assertEqual(tuple(inputs["input_ids"].shape), (2, 6))
        self.assertEqual(tuple(inputs["attention_mask"].shape), (2, 6))
        self.assertEqual(inputs["input_ids"].device.type, "cpu")
        self.assertEqual(inputs["attention_mask"].device.type, "cpu")
        self.assertEqual(inputs["attention_mask"][0].tolist(), [0, 0, 0, 1, 1, 1])

    def test_cache_implementation_is_forwarded_to_generation(self) -> None:
        generator = HFTextGenerator(
            model=None,
            tokenizer=_TokenizerStub(),
            device=torch.device("cpu"),
            max_new_tokens=10,
            input_mode="chat",
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            cache_implementation="static",
        )

        self.assertEqual(generator.generation_kwargs()["cache_implementation"], "static")

    def test_cuda_defaults_to_one_shot_generation(self) -> None:
        generator = HFTextGenerator(
            model=None,
            tokenizer=_TokenizerStub(),
            device=torch.device("cuda"),
            max_new_tokens=10,
            input_mode="chat",
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
        )

        self.assertIsNone(generator.generation_chunk_size)
        self.assertNotIn("cache_implementation", generator.generation_kwargs())

    def test_greedy_generation_omits_sampling_only_arguments(self) -> None:
        generator = HFTextGenerator(
            model=None,
            tokenizer=_TokenizerStub(),
            device=torch.device("cpu"),
            max_new_tokens=10,
            input_mode="chat",
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
        )

        kwargs = generator.generation_kwargs()

        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)

    def test_chat_template_kwargs_are_forwarded(self) -> None:
        tokenizer = _TokenizerStub()
        generator = HFTextGenerator(
            model=None,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            max_new_tokens=10,
            input_mode="chat",
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            chat_template_kwargs={"enable_thinking": True},
        )

        generator.prompt_to_model_inputs("reason about this")

        self.assertIsNotNone(tokenizer.last_template_kwargs)
        self.assertTrue(tokenizer.last_template_kwargs["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
