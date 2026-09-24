"""Projection terminology used in the paper and newly generated artifacts."""

GEOMETRY_AWARE = "geometry_aware"
EMBEDDING_AGNOSTIC = "embedding_agnostic"
PAPER_PROJECTION_ARMS = (GEOMETRY_AWARE, EMBEDDING_AGNOSTIC)
PROJECTION_LABELS = {
    GEOMETRY_AWARE: "Geometry-aware",
    EMBEDDING_AGNOSTIC: "Embedding-agnostic",
}
PROJECTION_EQUATIONS = {
    GEOMETRY_AWARE: "e_s(z) = mu + C_epsilon^(1/2) Q R_s z",
    EMBEDDING_AGNOSTIC: "e_s(z) = A_s z; (A_s)_ij ~ N(0, 1/D)",
}

# Private identifiers in the protected numerical implementation. These are not
# accepted by the public artifact readers or exposed as experiment choices.
IMPLEMENTATION_ARMS = {
    GEOMETRY_AWARE: "current_no_norm",
    EMBEDDING_AGNOSTIC: "thinktrap_exact",
}
IMPLEMENTATION_ESTIMATOR_MODE = "factorized-embedding-whitened"


def projection_name(name: str) -> str:
    """Accept only the paper's two projection identifiers."""
    if name in PAPER_PROJECTION_ARMS:
        return name
    raise ValueError("Projection artifacts must use geometry_aware or embedding_agnostic")


def projection_label(name: str) -> str:
    return PROJECTION_LABELS[projection_name(name)]
