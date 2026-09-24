import json
from types import SimpleNamespace

import pytest
import torch

from raretrap.cli import main
from raretrap.custom_models import ContextCheckedGenerator, preflight


COMMIT = "a" * 40


@pytest.mark.parametrize("command", ["estimate", "projection"])
def test_custom_checkpoint_dry_run(command, tmp_path, capsys):
    main([command, "--hf-model", "example-org/chat-model", "--revision", COMMIT,
          "--dtype", "bfloat16", "--thinking", "disabled", "--output", str(tmp_path / "run"), "--dry-run"])
    spec = json.loads(capsys.readouterr().out)
    if command == "estimate":
        assert spec["custom_model"]
        assert not spec["recorded_map_matches_release"]
        assert spec["arguments"]["tested_llm_model"] == "example-org/chat-model"
        assert spec["arguments"]["surrogate_token_model"] == "example-org/chat-model"
        assert spec["arguments"]["tested_model_revision"] == COMMIT
        assert spec["arguments"]["projection_mode"] == "geometry_aware"
    else:
        assert spec["model"]["custom"]
        assert spec["model"]["dtype"] == "bfloat16"
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("extra", [[], ["--revision", "main"], ["--revision", "123"],
                                  ["--revision", "z" * 40]])
def test_unpinned_custom_checkpoint_is_rejected(extra, tmp_path):
    with pytest.raises(SystemExit):
        main(["estimate", "--hf-model", "example-org/chat-model", *extra,
              "--output", str(tmp_path / "run"), "--dry-run"])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("flag,value", [("--dtype", "float32"), ("--thinking", "disabled"), ("--revision", COMMIT)])
def test_paper_aliases_cannot_be_overridden(flag, value, tmp_path):
    with pytest.raises(SystemExit):
        main(["projection", "--model", "deepseek8b", flag, value,
              "--output", str(tmp_path / "run"), "--dry-run"])


def test_preflight_pins_and_disables_remote_code(monkeypatch):
    import transformers
    calls = []
    config = SimpleNamespace(max_position_embeddings=32768, is_encoder_decoder=False, _commit_hash=COMMIT)
    tokenizer = SimpleNamespace(chat_template="test", pad_token=None, eos_token="end",
                                apply_chat_template=lambda *a, **k: [1, 2])
    def loader(asset):
        def load(name, **kwargs):
            calls.append((name, kwargs))
            return asset
        return load
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", loader(config))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", loader(tokenizer))
    model = dict(model="example-org/chat-model", revision=COMMIT, thinking="auto")
    assert preflight(model, 20003)["context_limit"] == 32768
    assert all(kwargs == dict(revision=COMMIT, trust_remote_code=False) for _, kwargs in calls)
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        preflight(model, 20003)
    tokenizer.chat_template = "test"
    config.max_position_embeddings = 1024
    with pytest.raises(ValueError, match="context"):
        preflight(model, 20003)


def test_context_guard_delegates_without_generation_changes():
    calls = []
    generator = SimpleNamespace(max_new_tokens=10,
        _prompt_to_cpu_inputs=lambda s: {"input_ids": torch.zeros(len(s), dtype=torch.long)},
        generate=lambda text, **kwargs: calls.append((text, kwargs)) or "result",
        generate_batch=lambda texts, **kwargs: calls.append((texts, kwargs)) or ["result"])
    checked = ContextCheckedGenerator(generator, 15)
    assert checked.generate("hi", output_token_callback=None) == "result"
    assert calls == [("hi", {"output_token_callback": None})]
    assert checked.generate_batch(["hi"]) == ["result"]
    with pytest.raises(ValueError, match="not truncated"):
        checked.generate("too long")
    assert len(calls) == 2
