"""The paper's two latent-to-embedding constructions.

For z in R^D and s = 1,...,S:
    geometry_aware:    e_s(z) = mu + C_epsilon^(1/2) Q R_s z
    embedding_agnostic: e_s(z) = A_s z

Q has orthonormal columns; each R_s is orthogonal. The entries of A_s
are Gaussian with variance 1/D. Both use raw Euclidean nearest-token lookup.
No additional normalization is applied to the geometry-aware construction.
"""
from scripts.run_projection_ablation import _build_projection_arms
from raretrap.terminology import IMPLEMENTATION_ARMS, PAPER_PROJECTION_ARMS
from utils.config import PROJECTION_SEED


def build_paper_projections(*, tokenizer, embedding_weight, S, D, epsilon, lookup_device):
    """Construct the frozen maps without changing production RNG or operations.

    S is the number of surrogate-token slots, D the latent dimension, and epsilon
    the covariance eigenvalue floor. ProjectionSpace stores B=C_epsilon^(1/2)Q
    as shared_matrix, {R_s} as slot_transforms, and the repeated mu as bias.
    The Gaussian baseline stores the vertically stacked {A_s} as matrix.
    These fields describe the tensor representation of the paper's factors.
    """
    projections = _build_projection_arms(
        requested_arms=[IMPLEMENTATION_ARMS[name] for name in PAPER_PROJECTION_ARMS],
        tokenizer=tokenizer, embedding_weight=embedding_weight,
        prompt_length=S, d_latent=D, seed=PROJECTION_SEED,
        covariance_eps=epsilon, lookup_device=lookup_device,
    )
    return {name: projections[IMPLEMENTATION_ARMS[name]] for name in PAPER_PROJECTION_ARMS}
