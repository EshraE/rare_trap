import numpy as np
import torch

from scripts.run_projection_ablation import _build_projection_arms
from utils.config import PROJECTION_SEED
from utils.projection_space import build_projection_space


class Tokenizer:
    all_special_ids = []

    def __len__(self):
        return 30

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


def _map(weights):
    return build_projection_space(tokenizer=Tokenizer(), embedding_weight=weights,
        input_prompt_len=4, ss_sample_space_dim=3, noise_k=0,
        candidate_mode="visible_nonspecial", projection_mode="factorized-embedding-whitened",
        covariance_eps=1e-5, lookup_device="cpu")


def test_projection_fixed_independently_of_global_sampling_seeds():
    assert PROJECTION_SEED == 1010
    weights = torch.randn(30, 8, generator=torch.Generator().manual_seed(9))
    torch.manual_seed(17)
    first = _map(weights)
    torch.manual_seed(999)
    np.random.seed(1234)
    second = _map(weights)
    assert torch.equal(first.shared_matrix, second.shared_matrix)
    assert torch.equal(first.slot_transforms, second.slot_transforms)


def test_ablation_matches_unscaled_production_map_and_raw_euclidean_lookup():
    weights = torch.randn(30, 8, generator=torch.Generator().manual_seed(9))
    arms = _build_projection_arms(requested_arms=["current_no_norm", "thinktrap_exact"],
        tokenizer=Tokenizer(), embedding_weight=weights, prompt_length=4, d_latent=3,
        seed=PROJECTION_SEED, covariance_eps=1e-5, lookup_device=torch.device("cpu"))
    production = _map(weights)
    aware = arms["current_no_norm"].projection
    assert torch.equal(aware.shared_matrix, production.shared_matrix)
    assert torch.equal(aware.slot_transforms, production.slot_transforms)
    z = np.random.default_rng(20260806).standard_normal(3).astype(np.float32)
    for name, arm in arms.items():
        p = arm.projection
        if name == "current_no_norm":
            vectors = torch.stack([p.bias[s] + p.shared_matrix @ p.slot_transforms[s] @ torch.from_numpy(z) for s in range(4)])
        else:
            assert torch.count_nonzero(p.bias) == 0
            vectors = (p.matrix @ torch.from_numpy(z)).reshape(4, 8)
        expected = torch.argmin(torch.cdist(vectors, p.embed_table), dim=1)
        assert p.decode_z(z).token_ids == p.token_ids[expected].tolist()
