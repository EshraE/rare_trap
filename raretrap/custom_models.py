"""Compatibility checks for opt-in HF checkpoints; paper loaders are unchanged."""
from __future__ import annotations


def preflight(model: dict, max_new_tokens: int) -> dict:
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(model["model"], revision=model["revision"], trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model["model"], revision=model["revision"], trust_remote_code=False)
    text_config = getattr(config, "text_config", config)
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError("Custom models must be decoder-only causal language models.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Custom model tokenizer must supply a chat template; no template is invented.")
    if tokenizer.pad_token is None and tokenizer.eos_token is None:
        raise ValueError("Custom tokenizer requires an EOS or pad token.")
    context = getattr(text_config, "max_position_embeddings", None)
    if not isinstance(context, int) or context <= 0:
        raise ValueError("Custom model must declare a positive max_position_embeddings context limit.")
    kwargs = {}
    if model["thinking"] != "auto":
        kwargs["enable_thinking"] = model["thinking"] == "enabled"
    probe = tokenizer.apply_chat_template([{"role": "user", "content": "Hello."}],
                                         tokenize=True, add_generation_prompt=True, **kwargs)
    if len(probe) + max_new_tokens > context:
        raise ValueError("Requested generation cap plus chat input exceeds the custom model context. "
                         "Use a compatible checkpoint or a smaller --threshold for estimation.")
    return dict(context_limit=context, requested_revision=model["revision"],
                resolved_config_revision=getattr(config, "_commit_hash", None),
                probe_input_tokens=len(probe), trust_remote_code=False)


class ContextCheckedGenerator:
    """Check each actual prompt; delegate generation without changing its settings."""
    def __init__(self, generator, context_limit: int):
        self._generator = generator
        self._context_limit = context_limit

    def __getattr__(self, name):
        return getattr(self._generator, name)

    def _check(self, text):
        ids = self._generator._prompt_to_cpu_inputs(text)["input_ids"]
        if int(ids.numel()) + self._generator.max_new_tokens > self._context_limit:
            raise ValueError("Custom model context budget exceeded; input is not truncated and cap is not reduced.")

    def generate(self, prompt_text, **kwargs):
        self._check(prompt_text)
        return self._generator.generate(prompt_text, **kwargs)

    def generate_batch(self, prompt_texts, **kwargs):
        for text in prompt_texts:
            self._check(text)
        return self._generator.generate_batch(prompt_texts, **kwargs)
