from __future__ import annotations

from types import SimpleNamespace

import torch

from utils import model_assets


class _FakeTokenizer:
    def __init__(self, revision: str) -> None:
        self.pad_token = None
        self.eos_token = "<eos>"
        self.init_kwargs = {"_commit_hash": revision}


class _FakeModel:
    def __init__(self, revision: str) -> None:
        self.config = SimpleNamespace(_commit_hash=revision)
        self.embedding = torch.nn.Embedding(8, 4)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def get_input_embeddings(self):
        return self.embedding


def test_loader_forwards_and_records_pinned_revisions(monkeypatch) -> None:
    tokenizer_calls = []
    model_calls = []

    def tokenizer_from_pretrained(name, **kwargs):
        tokenizer_calls.append((name, kwargs))
        return _FakeTokenizer(str(kwargs["revision"]))

    def model_from_pretrained(name, **kwargs):
        model_calls.append((name, kwargs))
        return _FakeModel(str(kwargs["revision"]))

    monkeypatch.setattr(model_assets.AutoTokenizer, "from_pretrained", tokenizer_from_pretrained)
    monkeypatch.setattr(model_assets.AutoModelForCausalLM, "from_pretrained", model_from_pretrained)
    monkeypatch.setattr(model_assets, "pick_device", lambda: torch.device("cpu"))

    assets = model_assets.load_model_assets(
        tested_llm_model_name="target",
        surrogate_token_model_name="surrogate",
        tested_model_revision="target-sha",
        surrogate_model_revision="surrogate-sha",
    )

    assert tokenizer_calls[0][1]["revision"] == "target-sha"
    assert tokenizer_calls[1][1]["revision"] == "surrogate-sha"
    assert model_calls[0][1]["revision"] == "target-sha"
    assert model_calls[1][1]["revision"] == "surrogate-sha"
    assert assets.tested_model_revision == "target-sha"
    assert assets.surrogate_model_revision == "surrogate-sha"
