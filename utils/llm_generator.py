"""Text generation wrapper used by ThinkTrap objectives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    prompt_text: str
    input_token_count: int
    completion_length: int
    completion_text: str
    completion_token_ids: list[int]
    elapsed_sec: float
    tokens_per_second: float
    hit_cap: bool
    model_input_token_ids: list[int] = field(default_factory=list)


class _OutputTokenCountStreamer:
    def __init__(self, prompt_token_count: int, callback: Callable[[int], None]) -> None:
        self.prompt_tokens_remaining = int(prompt_token_count)
        self.output_tokens = 0
        self.callback = callback

    def put(self, value) -> None:
        token_count = int(torch.as_tensor(value).numel())
        if self.prompt_tokens_remaining > 0:
            skipped = min(token_count, self.prompt_tokens_remaining)
            self.prompt_tokens_remaining -= skipped
            token_count -= skipped
        if token_count <= 0:
            return
        self.output_tokens += token_count
        self.callback(int(self.output_tokens))

    def end(self) -> None:
        self.callback(int(self.output_tokens))


class _BatchOutputTokenCountStreamer:
    def __init__(
        self,
        *,
        batch_size: int,
        callbacks: list[Callable[[int], None]],
        eos_id,
    ) -> None:
        if int(batch_size) != len(callbacks):
            raise ValueError("batch_size must match callback count.")
        self.batch_size = int(batch_size)
        self.callbacks = callbacks
        self.output_tokens = [0 for _ in range(self.batch_size)]
        self.finished = [False for _ in range(self.batch_size)]
        self.seen_prompt = False
        if eos_id is None:
            self.eos_ids: set[int] = set()
        elif isinstance(eos_id, (list, tuple, set)):
            self.eos_ids = {int(token_id) for token_id in eos_id}
        else:
            self.eos_ids = {int(eos_id)}

    def _generated_rows(self, value) -> list[list[int]]:
        array = torch.as_tensor(value).detach().to("cpu")
        if not self.seen_prompt and array.ndim == 2 and int(array.shape[0]) == self.batch_size:
            self.seen_prompt = True
            return []
        self.seen_prompt = True
        if array.ndim == 0:
            return [[int(array.item())]]
        if array.ndim == 1:
            values = [int(token_id) for token_id in array.tolist()]
            if len(values) == self.batch_size:
                return [[token_id] for token_id in values]
            return [values]
        if array.ndim == 2 and int(array.shape[0]) == self.batch_size:
            return [[int(token_id) for token_id in row.tolist()] for row in array]
        flat = [int(token_id) for token_id in array.reshape(-1).tolist()]
        if len(flat) == self.batch_size:
            return [[token_id] for token_id in flat]
        return [flat]

    def put(self, value) -> None:
        rows = self._generated_rows(value)
        if not rows:
            return
        for row_index, row_tokens in enumerate(rows[: self.batch_size]):
            if self.finished[row_index]:
                continue
            for token_id in row_tokens:
                if self.finished[row_index]:
                    break
                self.output_tokens[row_index] += 1
                self.callbacks[row_index](int(self.output_tokens[row_index]))
                if token_id in self.eos_ids:
                    self.finished[row_index] = True

    def end(self) -> None:
        for row_index, callback in enumerate(self.callbacks):
            callback(int(self.output_tokens[row_index]))


@dataclass
class HFTextGenerator:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    max_new_tokens: int
    input_mode: str
    do_sample: bool
    temperature: float
    top_p: float
    cache_implementation: Optional[str] = None
    generation_chunk_size: Optional[int] = None
    chat_template_kwargs: Optional[dict[str, object]] = None

    def _prompt_to_cpu_inputs(self, prompt_text: str) -> dict[str, torch.Tensor]:
        if self.input_mode == "chat" and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            template_kwargs = dict(self.chat_template_kwargs or {})
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    **template_kwargs,
                )
            except TypeError:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    **template_kwargs,
                )
            if isinstance(encoded, torch.Tensor):
                input_ids = encoded
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids.detach().to("cpu"),
                "attention_mask": attention_mask.detach().to("cpu"),
            }

        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.detach().to("cpu"),
            "attention_mask": attention_mask.detach().to("cpu"),
        }

    def _release_device_cache(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    def _pad_token_rows(
        self,
        rows: list[torch.Tensor],
        *,
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(row.numel()) for row in rows)
        input_ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for row_index, row in enumerate(rows):
            width = int(row.numel())
            input_ids[row_index, max_len - width:] = row
            attention_mask[row_index, max_len - width:] = 1
        return input_ids, attention_mask

    def _completion_ids_from_generated_row(
        self,
        completion_ids: torch.Tensor,
        *,
        eos_id,
        pad_id: Optional[int],
    ) -> list[int]:
        token_ids = [int(token_id) for token_id in completion_ids.detach().to("cpu").tolist()]
        eos_ids: set[int] = set()
        if eos_id is not None:
            if isinstance(eos_id, (list, tuple, set)):
                eos_ids = {int(token_id) for token_id in eos_id}
            else:
                eos_ids = {int(eos_id)}
        for index, token_id in enumerate(token_ids):
            if token_id in eos_ids:
                return token_ids[: index + 1]
        if pad_id is not None and int(pad_id) not in eos_ids:
            while token_ids and token_ids[-1] == int(pad_id):
                token_ids.pop()
        return token_ids

    def prompt_to_model_inputs(self, prompt_text: str) -> dict[str, torch.Tensor]:
        encoded = self._prompt_to_cpu_inputs(prompt_text)
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def prompts_to_model_inputs(self, prompt_texts: list[str]) -> tuple[dict[str, torch.Tensor], list[int]]:
        if not prompt_texts:
            raise ValueError("prompt_texts must not be empty.")
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        if pad_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

        encoded_rows = [
            self._prompt_to_cpu_inputs(prompt_text)["input_ids"][0]
            for prompt_text in prompt_texts
        ]
        input_token_counts = [int(row.numel()) for row in encoded_rows]
        input_ids, attention_mask = self._pad_token_rows(encoded_rows, pad_id=int(pad_id))
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }, input_token_counts

    @torch.inference_mode()
    def _generate_batch_chunked(
        self,
        prompt_texts: list[str],
        output_token_callbacks: Optional[list[Callable[[int], None]]],
        do_sample_override: Optional[bool],
    ) -> list[GenerationResult]:
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        if pad_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")
        eos_ids = (
            set()
            if eos_id is None
            else {int(token_id) for token_id in eos_id}
            if isinstance(eos_id, (list, tuple, set))
            else {int(eos_id)}
        )
        prompt_rows = [
            self._prompt_to_cpu_inputs(prompt_text)["input_ids"][0]
            for prompt_text in prompt_texts
        ]
        input_token_counts = [int(row.numel()) for row in prompt_rows]
        completion_rows: list[list[int]] = [[] for _ in prompt_texts]
        active_rows = {index: row for index, row in enumerate(prompt_rows)}
        chunk_size = max(1, int(self.generation_chunk_size or self.max_new_tokens))

        start = time.perf_counter()
        while active_rows:
            active_indices = list(active_rows)
            rows = [active_rows[index] for index in active_indices]
            remaining = int(self.max_new_tokens) - len(completion_rows[active_indices[0]])
            stage_new_tokens = min(chunk_size, remaining)
            input_ids, attention_mask = self._pad_token_rows(rows, pad_id=int(pad_id))
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            prompt_width = int(input_ids.shape[1])
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(stage_new_tokens),
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **self.generation_kwargs(do_sample_override=do_sample_override),
            )

            next_active_rows: dict[int, torch.Tensor] = {}
            for local_index, original_index in enumerate(active_indices):
                new_tokens = self._completion_ids_from_generated_row(
                    generated[local_index, prompt_width:],
                    eos_id=eos_id,
                    pad_id=pad_id,
                )
                completion_rows[original_index].extend(new_tokens)
                if output_token_callbacks is not None:
                    output_token_callbacks[original_index](len(completion_rows[original_index]))
                reached_eos = any(token_id in eos_ids for token_id in new_tokens)
                reached_cap = len(completion_rows[original_index]) >= int(self.max_new_tokens)
                if not reached_eos and not reached_cap and len(new_tokens) >= stage_new_tokens:
                    continuation = torch.tensor(new_tokens, dtype=torch.long)
                    next_active_rows[original_index] = torch.cat(
                        (active_rows[original_index], continuation),
                        dim=0,
                    )

            del generated, input_ids, attention_mask
            active_rows = next_active_rows
            self._release_device_cache()

        elapsed = max(time.perf_counter() - start, 1e-9)
        results: list[GenerationResult] = []
        for row_index, prompt_text in enumerate(prompt_texts):
            completion_token_ids = completion_rows[row_index]
            completion_len = len(completion_token_ids)
            results.append(
                GenerationResult(
                    prompt_text=prompt_text,
                    input_token_count=input_token_counts[row_index],
                    completion_length=completion_len,
                    completion_text=self.tokenizer.decode(completion_token_ids, skip_special_tokens=False),
                    completion_token_ids=completion_token_ids,
                    elapsed_sec=float(elapsed),
                    tokens_per_second=float(completion_len) / float(elapsed),
                    hit_cap=bool(completion_len >= int(self.max_new_tokens)),
                    model_input_token_ids=[int(token_id) for token_id in prompt_rows[row_index].tolist()],
                )
            )
        return results

    def generation_kwargs(self, do_sample_override: Optional[bool] = None) -> dict:
        do_sample = self.do_sample if do_sample_override is None else bool(do_sample_override)
        kwargs = {
            "do_sample": bool(do_sample),
            "num_beams": 1,
            "use_cache": True,
        }
        if do_sample:
            kwargs["temperature"] = float(self.temperature)
            kwargs["top_p"] = float(self.top_p)
        if self.cache_implementation is not None:
            kwargs["cache_implementation"] = str(self.cache_implementation)
        return kwargs

    @torch.inference_mode()
    def generate(
        self,
        prompt_text: str,
        output_token_callback: Optional[Callable[[int], None]] = None,
        do_sample_override: Optional[bool] = None,
    ) -> GenerationResult:
        model_inputs = self.prompt_to_model_inputs(prompt_text)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        streamer = None
        if output_token_callback is not None:
            streamer = _OutputTokenCountStreamer(
                prompt_token_count=int(input_ids.numel()),
                callback=output_token_callback,
            )

        start = time.perf_counter()
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(self.max_new_tokens),
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            streamer=streamer,
            **self.generation_kwargs(do_sample_override=do_sample_override),
        )
        elapsed = max(time.perf_counter() - start, 1e-9)
        completion_ids = generated[0, int(input_ids.shape[1]):].detach().to("cpu")
        completion_token_ids = self._completion_ids_from_generated_row(
            completion_ids,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        completion_len = int(len(completion_token_ids))
        if output_token_callback is not None:
            output_token_callback(completion_len)
        result = GenerationResult(
            prompt_text=prompt_text,
            input_token_count=int(input_ids.numel()),
            completion_length=completion_len,
            completion_text=self.tokenizer.decode(completion_token_ids, skip_special_tokens=False),
            completion_token_ids=completion_token_ids,
            elapsed_sec=float(elapsed),
            tokens_per_second=float(completion_len) / float(elapsed),
            hit_cap=bool(completion_len >= int(self.max_new_tokens)),
            model_input_token_ids=[int(token_id) for token_id in input_ids[0].detach().to("cpu").tolist()],
        )
        del generated, completion_ids, input_ids, attention_mask, model_inputs
        self._release_device_cache()
        return result

    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_texts: list[str],
        output_token_callbacks: Optional[list[Callable[[int], None]]] = None,
        do_sample_override: Optional[bool] = None,
    ) -> list[GenerationResult]:
        if not prompt_texts:
            return []
        if self.generation_chunk_size is not None:
            return self._generate_batch_chunked(
                prompt_texts,
                output_token_callbacks=output_token_callbacks,
                do_sample_override=do_sample_override,
            )
        if len(prompt_texts) == 1:
            callback = None
            if output_token_callbacks:
                callback = output_token_callbacks[0]
            return [
                self.generate(
                    prompt_texts[0],
                    output_token_callback=callback,
                    do_sample_override=do_sample_override,
                )
            ]

        model_inputs, input_token_counts = self.prompts_to_model_inputs(prompt_texts)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        streamer = None
        if output_token_callbacks is not None:
            streamer = _BatchOutputTokenCountStreamer(
                batch_size=len(prompt_texts),
                callbacks=output_token_callbacks,
                eos_id=eos_id,
            )

        start = time.perf_counter()
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(self.max_new_tokens),
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            streamer=streamer,
            **self.generation_kwargs(do_sample_override=do_sample_override),
        )
        elapsed = max(time.perf_counter() - start, 1e-9)
        prompt_width = int(input_ids.shape[1])
        results: list[GenerationResult] = []
        for row_index, prompt_text in enumerate(prompt_texts):
            model_input_token_ids = [
                int(token_id)
                for token_id in input_ids[row_index][attention_mask[row_index].bool()]
                .detach()
                .to("cpu")
                .tolist()
            ]
            completion_token_ids = self._completion_ids_from_generated_row(
                generated[row_index, prompt_width:],
                eos_id=eos_id,
                pad_id=pad_id,
            )
            completion_len = int(len(completion_token_ids))
            results.append(
                GenerationResult(
                    prompt_text=prompt_text,
                    input_token_count=int(input_token_counts[row_index]),
                    completion_length=completion_len,
                    completion_text=self.tokenizer.decode(completion_token_ids, skip_special_tokens=False),
                    completion_token_ids=completion_token_ids,
                    elapsed_sec=float(elapsed),
                    tokens_per_second=float(completion_len) / float(elapsed),
                    hit_cap=bool(completion_len >= int(self.max_new_tokens)),
                    model_input_token_ids=model_input_token_ids,
                )
            )
        del generated, input_ids, attention_mask, model_inputs
        self._release_device_cache()
        return results
