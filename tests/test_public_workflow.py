"""CPU integration test: real estimator/reporters, synthetic generation only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from raretrap.analysis import records, report
from raretrap.cli import main
from utils.llm_generator import GenerationResult
from utils.projection_space import ProjectedPrompt
from utils.thinktrap_objective import ThinkTrapObjective


class Projection:
    d_latent = 2
    prompt_length = 2
    candidate_count = 20
    embed_dim = 4
    mode = "factorized-embedding-whitened"

    def __call__(self, latent):
        return ProjectedPrompt(token_ids=[1, 2], text=f"{float(latent[0]):.8f}")


class Generator:
    tokenizer = SimpleNamespace(all_special_ids=[0])
    device = None
    max_new_tokens = 103
    do_sample = False

    def _prompt_to_cpu_inputs(self, text):
        return {"input_ids": torch.tensor([0, 1, 2])}

    def generate(self, text, output_token_callback=None):
        length = max(4, min(40, int(12 + 4 * float(text))))
        return GenerationResult(prompt_text=text, input_token_count=3, completion_length=length,
            completion_text="synthetic " * length, completion_token_ids=[7] * length,
            model_input_token_ids=[0, 1, 2], elapsed_sec=1, tokens_per_second=length, hit_cap=False)


@pytest.mark.parametrize("custom", [False, True])
def test_cli_preserves_full_output_contract(monkeypatch, tmp_path, custom):
    import raretrap.cli
    import utils.objective_factory

    def factory(**kwargs):
        Path(kwargs["template_probe_output"]).write_text("{}")
        return ThinkTrapObjective(generator=Generator(), projection_space=Projection(),
            threshold=kwargs["threshold"], raw_samples_path=kwargs["raw_samples_path"],
            max_saved_records=100, max_saved_output_text_records=100)

    monkeypatch.setattr(raretrap.cli, "_require_cuda", lambda: None)
    monkeypatch.setattr(utils.objective_factory, "build_obj", factory)
    extra = []
    if custom:
        import raretrap.custom_models
        monkeypatch.setattr(raretrap.custom_models, "preflight", lambda *a: {"context_limit": 1000})
        extra = ["--hf-model", "example-org/chat-model", "--revision", "a" * 40]
    output = tmp_path / "run"
    main(["estimate", "--output", str(output), "--threshold", "100", "--samples-per-level", "20",
          "--latent-dim", "2", "--prompt-length", "2", "--max-levels", "2", *extra])
    expected = {"configuration.json", "cases.json", "all_evaluations.json", "raw_samples.jsonl",
                "sampler_trace", "template_probe.json", "run.log", "DONE"}
    assert expected <= {p.name for p in output.iterdir()}
    raw = list(records(output / "raw_samples.jsonl"))
    assert raw[0]["special_token_ids"] == [0]
    for row in raw[1:]:
        assert len(row["latent_vector_float32"]) == 2
        assert row["input_token_ids"] == [1, 2]
        assert row["generations"][0]["model_input_token_ids"] == [0, 1, 2]
        assert row["generations"][0]["completion_token_ids"]
        assert "repetition_score" in row["aggregate"]
    metadata = json.loads((output / "cases.json").read_text())["metadata"]
    assert len(raw) - 1 == metadata["estimator_result"]["total_evaluations"]
    dataset = list(records(output / "dataset/samples.jsonl"))
    assert len(dataset) == len(raw) - 1
    assert dataset[0]["completion_token_ids"] == raw[1]["generations"][0]["completion_token_ids"]
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["completed_evaluations"] == len(dataset)
    slots = list(records(output / "sampler_trace/population_slots.jsonl"))
    assert len(slots) == 20 * (metadata["estimator_result"]["completed_levels"] + 1)
    assert all(s["performance_score"] is not None for s in slots)
    report([output], tmp_path / "report", cases=True)
    assert (tmp_path / "report/populations.csv").exists()
