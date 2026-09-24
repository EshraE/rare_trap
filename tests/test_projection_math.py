from __future__ import annotations

import numpy as np
import torch

from utils.projection_space import ProjectionSpace, _embedding_covariance_sqrt


class _Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return ",".join(str(int(token_id)) for token_id in token_ids)


def test_covariance_square_root_reconstructs_sample_covariance() -> None:
    embeddings = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
        dtype=torch.float32,
    )

    mean, covariance_sqrt = _embedding_covariance_sqrt(embeddings, eps=1e-8)
    centered = embeddings - embeddings.mean(dim=0)
    covariance = centered.T.matmul(centered) / float(embeddings.shape[0] - 1)

    torch.testing.assert_close(mean, embeddings.mean(dim=0))
    torch.testing.assert_close(
        covariance_sqrt.matmul(covariance_sqrt),
        covariance,
        rtol=1e-5,
        atol=1e-6,
    )


def test_factorized_projection_and_exact_euclidean_decoder() -> None:
    table = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    projection = ProjectionSpace(
        tokenizer=_Tokenizer(),
        token_ids=torch.tensor([10, 11, 12]),
        embed_table=table,
        embed_norms=torch.sum(table * table, dim=1),
        prompt_length=2,
        d_latent=2,
        matrix=None,
        bias=torch.zeros((2, 2), dtype=torch.float32),
        mode="test-factorized",
        lookup_device=torch.device("cpu"),
        shared_matrix=torch.eye(2),
        slot_transforms=torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            dtype=torch.float32,
        ),
    )

    slot_embeddings = projection.project(np.asarray([1.8, 0.1], dtype=np.float32))
    selected = projection.slot_embeddings_to_token_ids_batch(slot_embeddings)

    torch.testing.assert_close(
        slot_embeddings,
        torch.tensor([[[1.8, 0.1], [0.1, 1.8]]]),
    )
    assert selected == [[11, 12]]
