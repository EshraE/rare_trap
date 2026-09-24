"""Real HF generation on a random CPU model, not a paper-result reproduction."""
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from raretrap.projection import build_paper_projections
from utils.llm_generator import HFTextGenerator
from utils.repetition_metrics import repetition_metrics


@pytest.mark.parametrize("arm", ["geometry_aware", "embedding_agnostic"])
def test_projection_chat_generation_and_scores_without_downloads(monkeypatch, arm):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}
    vocabulary.update({f"t{i}": i for i in range(3, 32)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[PAD]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]",
        chat_template="{{ bos_token }} {% for message in messages %}"
                      "{{ message['content'] }} {% endfor %}"
                      "{% if add_generation_prompt %}{{ bos_token }}{% endif %}",
    )
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
                         pad_token_id=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = LlamaForCausalLM(config).eval()

    projections = build_paper_projections(
        tokenizer=tokenizer, embedding_weight=model.get_input_embeddings().weight.detach(),
        S=4, D=3, epsilon=1e-5, lookup_device=torch.device("cpu"),
    )
    z = np.asarray([0.2, -0.3, 0.5], dtype=np.float32)
    prompt = projections[arm].projection.decode_z(z)
    assert len(prompt.token_ids) == 4
    assert not set(prompt.token_ids) & set(tokenizer.all_special_ids)
    assert prompt.text == tokenizer.decode(prompt.token_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    generator = HFTextGenerator(model=model, tokenizer=tokenizer, device=torch.device("cpu"),
                                max_new_tokens=8, input_mode="chat", do_sample=False,
                                temperature=1.0, top_p=1.0)
    progress = []
    result = generator.generate(prompt.text, output_token_callback=progress.append)
    repeated = generator.generate(prompt.text)
    expected_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.text}], tokenize=True, add_generation_prompt=True,
        return_dict=False,
    )
    assert result.model_input_token_ids == expected_input
    assert result.input_token_count == 6  # Four surrogate slots plus two chat markers.
    assert result.completion_token_ids == repeated.completion_token_ids
    assert result.completion_length == len(result.completion_token_ids)
    assert 0 < result.completion_length <= 8
    assert result.hit_cap == (result.completion_length == 8)
    assert progress[-1] == result.completion_length
    assert result.completion_text == tokenizer.decode(result.completion_token_ids,
                                                       skip_special_tokens=False)
    scores = repetition_metrics(result.completion_token_ids,
                                special_token_ids=tokenizer.all_special_ids)
    filtered = [i for i in result.completion_token_ids if i not in tokenizer.all_special_ids]
    diversity = []
    for n in (2, 3, 4):
        grams = [tuple(filtered[i:i + n]) for i in range(len(filtered) - n + 1)]
        diversity.append(len(set(grams)) / len(grams) if grams else 1.0)
    assert scores.repetition_score == pytest.approx(1 - np.prod(diversity))
