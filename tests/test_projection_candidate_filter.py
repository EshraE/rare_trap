import torch
import pytest

from utils.config import PROJECTION_SEED
from utils.projection_space import (
    _gaussian_block,
    build_projection_space,
    select_candidate_token_ids,
)


class DummyTokenizer:
    all_special_ids = [0]

    def __init__(self):
        self._decoded = {
            0: "<special>",
            1: "",
            2: "   ",
            3: "\x00",
            4: "\ufffd",
            5: "hello",
            6: " world",
        }

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(self._decoded[int(token_id)] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return self._decoded[int(token_id)]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class BoundedDummyTokenizer(DummyTokenizer):
    def __len__(self):
        return 5


def test_candidate_filter_excludes_invisible_decoded_tokens():
    ids = select_candidate_token_ids(
        tokenizer=DummyTokenizer(),
        vocab_size=7,
        count=0,
        rng=torch.Generator().manual_seed(0),
        mode="visible_nonspecial",
    )

    assert ids == [5, 6]


def test_legacy_nonspecial_mode_only_excludes_declared_special_ids():
    ids = select_candidate_token_ids(
        tokenizer=DummyTokenizer(),
        vocab_size=7,
        count=0,
        rng=torch.Generator().manual_seed(0),
        mode="nonspecial",
    )

    assert ids == [1, 2, 3, 4, 5, 6]


def test_candidate_ids_do_not_exceed_tokenizer_vocabulary():
    ids = select_candidate_token_ids(
        tokenizer=BoundedDummyTokenizer(),
        vocab_size=7,
        count=0,
        rng=torch.Generator().manual_seed(0),
        mode="nonspecial",
    )

    assert ids == [1, 2, 3, 4]


def test_candidate_filter_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown candidate-token mode"):
        select_candidate_token_ids(
            tokenizer=DummyTokenizer(),
            vocab_size=7,
            count=0,
            rng=torch.Generator().manual_seed(0),
            mode="typo",
        )


def test_projection_space_uses_internal_fixed_seed() -> None:
    embedding_weight = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    kwargs = {
        "tokenizer": DummyTokenizer(),
        "embedding_weight": embedding_weight,
        "input_prompt_len": 2,
        "ss_sample_space_dim": 3,
        "noise_k": 0,
        "candidate_mode": "visible_nonspecial",
        "projection_mode": "thinktrap-exact",
    }

    torch.manual_seed(1)
    first = build_projection_space(**kwargs)
    torch.manual_seed(999)
    second = build_projection_space(**kwargs)
    expected_rng = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    expected_matrix = _gaussian_block(8, 3, expected_rng)

    assert torch.equal(first.matrix, second.matrix)
    assert torch.equal(first.matrix, expected_matrix)
