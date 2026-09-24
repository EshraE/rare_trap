"""Latent-to-token projection spaces for ThinkTrap experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union
import unicodedata

import numpy as np
import torch
from transformers import AutoTokenizer

from utils.config import PROJECTION_SEED


def _token_has_visible_decoded_text(tokenizer: AutoTokenizer, token_id: int) -> bool:
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not text or not text.strip():
        return False
    for char in text:
        if char == "\ufffd":
            return False
        if unicodedata.category(char).startswith("C"):
            return False
    return True


def select_candidate_token_ids(
    tokenizer: AutoTokenizer,
    vocab_size: int,
    count: int,
    rng: torch.Generator,
    mode: str,
) -> List[int]:
    if mode not in {"visible_nonspecial", "nonspecial", "ascii"}:
        raise ValueError(f"Unknown candidate-token mode: {mode}")
    special = {
        int(token_id)
        for token_id in getattr(tokenizer, "all_special_ids", [])
        if 0 <= int(token_id) < int(vocab_size)
    }
    try:
        tokenizer_size = int(len(tokenizer))
    except (TypeError, AttributeError):
        tokenizer_size = int(vocab_size)
    decodable_size = min(int(vocab_size), tokenizer_size)
    pool: List[int] = []
    for token_id in range(decodable_size):
        if token_id in special:
            continue
        if mode in {"visible_nonspecial", "ascii"} and not _token_has_visible_decoded_text(
            tokenizer,
            token_id,
        ):
            continue
        if mode == "ascii":
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token is None:
                continue
            token_text = tokenizer.convert_tokens_to_string([token])
            if not token_text or not all(ord(ch) < 128 for ch in token_text):
                continue
        pool.append(token_id)
    if not pool:
        raise RuntimeError("Candidate token pool empty.")
    if count <= 0 or count >= len(pool):
        return pool
    perm = torch.randperm(len(pool), generator=rng).tolist()
    return [pool[index] for index in perm[:count]]


def decode_token_ids(tokenizer: AutoTokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _orthogonal_block(embed_dim: int, d_latent: int, rng: torch.Generator) -> torch.Tensor:
    if d_latent <= embed_dim:
        raw = torch.randn(embed_dim, d_latent, generator=rng, dtype=torch.float32)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        return q.contiguous()
    raw = torch.randn(d_latent, embed_dim, generator=rng, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.contiguous()


def _gaussian_block(rows: int, cols: int, rng: torch.Generator) -> torch.Tensor:
    scale = float(max(1, cols)) ** -0.5
    return (torch.randn(rows, cols, generator=rng, dtype=torch.float32) * scale).contiguous()


def _embedding_covariance_sqrt(
    embeddings: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    table = embeddings.to("cpu", torch.float32)
    mean = table.mean(dim=0)
    centered = table - mean.unsqueeze(0)
    cov = centered.T.matmul(centered) / float(max(int(table.shape[0]) - 1, 1))
    cov = 0.5 * (cov + cov.T)
    if not bool(torch.isfinite(cov).all()):
        raise RuntimeError("Embedding covariance contains non-finite values.")
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov)
    except RuntimeError:
        # Some CPU LAPACK builds fail on large embedding covariances with an
        # internal illegal-argument assert. Retry using NumPy's LAPACK path.
        cov_np = cov.numpy().astype(np.float64, copy=False)
        try:
            eigvals_np, eigvecs_np = np.linalg.eigh(cov_np)
        except Exception as np_exc:
            raise RuntimeError(
                "Failed to eigendecompose embedding covariance with both "
                "torch.linalg.eigh and numpy.linalg.eigh."
            ) from np_exc
        eigvals = torch.from_numpy(eigvals_np.astype(np.float32, copy=False))
        eigvecs = torch.from_numpy(eigvecs_np.astype(np.float32, copy=False))
    eigvals = eigvals.clamp_min(float(eps))
    scaled_eigvecs = eigvecs * torch.sqrt(eigvals).unsqueeze(0)
    cov_sqrt = scaled_eigvecs.matmul(eigvecs.T)
    return mean, cov_sqrt


def _build_projection_components(
    *,
    mode: str,
    slots: int,
    embed_dim: int,
    d_latent: int,
    rng: torch.Generator,
    surrogate_embed_table: torch.Tensor,
    covariance_eps: float,
) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mode not in {"embedding-whitened", "factorized-embedding-whitened", "thinktrap-exact"}:
        raise ValueError(f"Unknown projection mode: {mode}")

    if mode == "thinktrap-exact":
        projection_matrix = _gaussian_block(slots * embed_dim, d_latent, rng)
        projection_bias = torch.zeros(slots, embed_dim, dtype=torch.float32)
        return projection_matrix, projection_bias, None, None

    mean, cov_sqrt = _embedding_covariance_sqrt(
        embeddings=surrogate_embed_table,
        eps=float(covariance_eps),
    )
    projection_bias = mean.unsqueeze(0).repeat(slots, 1).contiguous()
    if mode == "factorized-embedding-whitened":
        q_shared = _orthogonal_block(embed_dim, d_latent, rng)
        shared_matrix = cov_sqrt.matmul(q_shared).contiguous()
        slot_transforms = torch.stack(
            [_orthogonal_block(d_latent, d_latent, rng) for _ in range(slots)],
            dim=0,
        ).contiguous()
        return None, projection_bias, shared_matrix, slot_transforms

    blocks = []
    for _ in range(slots):
        q = _orthogonal_block(embed_dim, d_latent, rng)
        blocks.append(cov_sqrt.matmul(q))
    projection_matrix = torch.cat(blocks, dim=0)
    return projection_matrix, projection_bias, None, None


@dataclass(frozen=True)
class ProjectedPrompt:
    token_ids: List[int]
    text: str


@dataclass(frozen=True)
class ProjectionSpace:
    tokenizer: AutoTokenizer
    token_ids: torch.Tensor
    embed_table: torch.Tensor
    embed_norms: torch.Tensor
    prompt_length: int
    d_latent: int
    matrix: Optional[torch.Tensor]
    bias: torch.Tensor
    mode: str
    lookup_device: torch.device
    nearest_chunk_size: int = 512
    shared_matrix: Optional[torch.Tensor] = None
    slot_transforms: Optional[torch.Tensor] = None

    @property
    def embed_dim(self) -> int:
        return int(self.embed_table.shape[1])

    @property
    def candidate_count(self) -> int:
        return int(self.token_ids.numel())

    @torch.inference_mode()
    def project(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(z, np.ndarray):
            z_tensor = torch.from_numpy(z.astype(np.float32))
        else:
            z_tensor = z.to(torch.float32)
        if z_tensor.ndim == 1:
            z_tensor = z_tensor.unsqueeze(0)
        device = torch.device(self.lookup_device)
        z_tensor = z_tensor.to(device=device, dtype=torch.float32)
        if self.shared_matrix is not None and self.slot_transforms is not None:
            shared_matrix = self.shared_matrix.to(device=device, dtype=torch.float32)
            slot_transforms = self.slot_transforms.to(device=device, dtype=torch.float32)
            per_slot_latents = torch.einsum("nd,sid->nsi", z_tensor, slot_transforms)
            projected = torch.einsum("nsi,ei->nse", per_slot_latents, shared_matrix)
        else:
            if self.matrix is None:
                raise RuntimeError("Projection matrix is missing for non-factorized projection mode.")
            projected = torch.matmul(z_tensor, self.matrix.T.to(device=device, dtype=torch.float32))
            projected = projected.view(int(z_tensor.shape[0]), int(self.prompt_length), -1)
        bias = self.bias.to(device=device, dtype=torch.float32).unsqueeze(0)
        return projected + bias

    @torch.inference_mode()
    def slot_embeddings_to_token_ids(self, slot_embeddings: torch.Tensor) -> List[int]:
        return self.slot_embeddings_to_token_ids_batch(slot_embeddings.unsqueeze(0))[0]

    @torch.inference_mode()
    def slot_embeddings_to_token_ids_batch(self, slot_embeddings: torch.Tensor) -> List[List[int]]:
        if slot_embeddings.ndim != 3:
            raise ValueError(f"Expected slot embeddings with shape [N, S, E], got {tuple(slot_embeddings.shape)}.")
        batch_size = int(slot_embeddings.shape[0])
        slots = int(slot_embeddings.shape[1])
        flat = slot_embeddings.reshape(batch_size * slots, int(slot_embeddings.shape[2]))
        device = torch.device(self.lookup_device)
        table = self.embed_table.to(device=device, dtype=torch.float32)
        table_norms = self.embed_norms.to(device=device, dtype=torch.float32)
        token_ids = self.token_ids.to(device=device)
        chunk_size = max(1, int(self.nearest_chunk_size))
        selected: List[int] = []
        table_t = table.T
        for start in range(0, int(flat.shape[0]), chunk_size):
            chunk = flat[start: start + chunk_size].to(device=device, dtype=torch.float32)
            chunk_norms = torch.sum(chunk * chunk, dim=-1, keepdim=True)
            scores = 2.0 * torch.matmul(chunk, table_t) - table_norms.unsqueeze(0) - chunk_norms
            best = torch.argmax(scores, dim=-1)
            selected.extend(int(token_id) for token_id in token_ids.index_select(0, best).detach().to("cpu").tolist())
        return [selected[index: index + slots] for index in range(0, len(selected), slots)]

    @torch.inference_mode()
    def projected_to_token_ids(self, projected: torch.Tensor) -> List[int]:
        return self.slot_embeddings_to_token_ids(projected[0])

    @torch.inference_mode()
    def z_to_token_ids(self, z: Union[np.ndarray, torch.Tensor]) -> List[int]:
        return self.projected_to_token_ids(self.project(z))

    def token_ids_to_text(self, token_ids: List[int]) -> str:
        return decode_token_ids(self.tokenizer, token_ids)

    @torch.inference_mode()
    def decode_z(self, z: Union[np.ndarray, torch.Tensor]) -> ProjectedPrompt:
        token_ids = self.z_to_token_ids(z)
        return ProjectedPrompt(token_ids=token_ids, text=self.token_ids_to_text(token_ids))

    @torch.inference_mode()
    def decode_batch(self, z: Union[np.ndarray, torch.Tensor]) -> List[ProjectedPrompt]:
        projected = self.project(z)
        token_id_rows = self.slot_embeddings_to_token_ids_batch(projected)
        prompts: List[ProjectedPrompt] = []
        for token_ids in token_id_rows:
            prompts.append(ProjectedPrompt(token_ids=token_ids, text=self.token_ids_to_text(token_ids)))
        return prompts

    @torch.inference_mode()
    def z_to_text(self, z: Union[np.ndarray, torch.Tensor]) -> str:
        return self.decode_z(z).text

    @torch.inference_mode()
    def __call__(self, z: Union[np.ndarray, torch.Tensor]) -> ProjectedPrompt:
        return self.decode_z(z)


def build_projection_space(
    *,
    tokenizer: AutoTokenizer,
    embedding_weight: torch.Tensor,
    input_prompt_len: int,
    ss_sample_space_dim: int,
    noise_k: int,
    candidate_mode: str,
    projection_mode: str = "embedding-whitened",
    covariance_eps: float = 1e-5,
    lookup_device: Union[str, torch.device] = "cpu",
) -> ProjectionSpace:
    # This private generator makes the latent-to-prompt map invariant to the
    # independently configurable Monte Carlo/MCMC seed.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(PROJECTION_SEED)
    embed_weight = embedding_weight.detach().to("cpu", torch.float32)
    candidate_ids = select_candidate_token_ids(
        tokenizer=tokenizer,
        vocab_size=int(embed_weight.shape[0]),
        count=int(noise_k),
        rng=rng,
        mode=str(candidate_mode),
    )
    token_ids = torch.tensor(candidate_ids, dtype=torch.long)
    embed_table = embed_weight.index_select(0, token_ids)
    embed_norms = torch.sum(embed_table * embed_table, dim=-1)
    lookup_device = torch.device(lookup_device)
    if lookup_device.type == "cuda" and not torch.cuda.is_available():
        lookup_device = torch.device("cpu")
    if lookup_device.type == "mps":
        lookup_device = torch.device("cpu")
    lookup_token_ids = token_ids.to(lookup_device)
    lookup_embed_table = embed_table.to(lookup_device)
    lookup_embed_norms = embed_norms.to(lookup_device)

    slots = max(1, int(input_prompt_len))
    latent_dim = int(ss_sample_space_dim)
    projection_matrix, projection_bias, shared_matrix, slot_transforms = _build_projection_components(
        mode=str(projection_mode),
        slots=slots,
        embed_dim=int(embed_weight.shape[1]),
        d_latent=latent_dim,
        rng=rng,
        surrogate_embed_table=embed_table,
        covariance_eps=float(covariance_eps),
    )
    if projection_matrix is not None:
        projection_matrix = projection_matrix.to(lookup_device)
    projection_bias = projection_bias.to(lookup_device)
    if shared_matrix is not None:
        shared_matrix = shared_matrix.to(lookup_device)
    if slot_transforms is not None:
        slot_transforms = slot_transforms.to(lookup_device)
    return ProjectionSpace(
        tokenizer=tokenizer,
        token_ids=lookup_token_ids,
        embed_table=lookup_embed_table,
        embed_norms=lookup_embed_norms,
        prompt_length=slots,
        d_latent=latent_dim,
        matrix=projection_matrix,
        bias=projection_bias,
        mode=str(projection_mode),
        lookup_device=lookup_device,
        shared_matrix=shared_matrix,
        slot_transforms=slot_transforms,
    )
