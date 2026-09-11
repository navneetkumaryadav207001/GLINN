"""Pure injection and normalization utilities for NLA-Lite.

Implements the vector-to-token injection hook and L2-normalization logic
following the Anthropic Natural Language Autoencoders methodology.
"""

from __future__ import annotations

import math
import re
import torch
import numpy as np

EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"
EXPLANATION_RE = re.compile(
    f"{re.escape(EXPLANATION_OPEN)}\\s*(.*?)\\s*{re.escape(EXPLANATION_CLOSE)}",
    re.DOTALL,
)


def wrap_explanation(text: str) -> str:
    """Wrap explanation payload in canonical tags."""
    return f"{EXPLANATION_OPEN}\n{text.strip()}\n{EXPLANATION_CLOSE}"


def extract_explanation(response: str) -> str | None:
    """Extract text between explanation tags; returns None on miss."""
    m = EXPLANATION_RE.search(response)
    return m.group(1).strip() if m else None


def normalize_activation(v: torch.Tensor, target_scale: float | None) -> torch.Tensor:
    """Scale vectors to target_scale L2-norm, or pass-through if None.

    Symmetrically used for:
      - Verbalizer injection: target_scale = injection_scale
      - Reconstructor MSE loss: target_scale = mse_scale (sqrt(d_model))
    """
    if target_scale is None:
        return v
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale_factor = (target_scale / norm).to(v.dtype)
    return v * scale_factor


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    """Overwrite embedding rows at injection marker positions with activation vectors.

    Args:
        input_ids: [B, S] token IDs tensor.
        embeddings: [B, S, d] tensor of token embeddings.
        vectors: [B, d] or [N, d] activation vectors to inject.
        inj_id: Token ID of injection character (e.g. '㊗').
        left_id: Token ID of left neighbor (e.g. '>').
        right_id: Token ID of right neighbor (e.g. '</').

    Returns:
        embeddings tensor with injection positions overwritten.
    """
    assert input_ids.ndim == 2, f"Expected input_ids [B, S], got {input_ids.shape}"
    assert embeddings.ndim == 3, f"Expected embeddings [B, S, d], got {embeddings.shape}"
    B, S = input_ids.shape
    d = embeddings.shape[-1]
    assert vectors.ndim == 2 and vectors.shape[-1] == d, (
        f"vectors must be [N, {d}], got {vectors.shape}"
    )

    out = embeddings.clone()
    vec_device = out.device
    vec_dtype = out.dtype
    vectors = vectors.to(device=vec_device, dtype=vec_dtype)

    matches = (input_ids == inj_id).nonzero()  # [M, 2] -> (b, p)
    vec_idx = 0
    for b, p in matches.tolist():
        if p == 0 or p == S - 1:
            continue
        # Verify canonical neighbors around the marker
        if input_ids[b, p - 1] == left_id and input_ids[b, p + 1] == right_id:
            if vec_idx < vectors.shape[0]:
                out[b, p] = vectors[vec_idx]
            vec_idx += 1

    expected = vectors.shape[0]
    if vec_idx != expected:
        raise RuntimeError(
            f"Found {vec_idx} valid injection sites matching neighbors, "
            f"but expected {expected} vectors. Check prompt template or tokenizer."
        )

    return out


def compute_predict_mean_baselines(
    vectors: torch.Tensor, mse_scale: float | None
) -> tuple[float, float]:
    """Compute predict-the-mean baseline MSEs for Fraction of Variance Explained (FVE).

    Returns:
        (meannorm_baseline, raw_variance_baseline):
        - meannorm_baseline: MSE(v_norm, normalize(mean(v_norm)))
        - raw_variance_baseline: MSE(v_norm, mean(v_norm)) [Canonical FVE denominator]
    """
    v_norm = normalize_activation(vectors.float(), mse_scale)
    mu = v_norm.mean(dim=0, keepdim=True)
    mu_normed = normalize_activation(mu, mse_scale)
    mse_meannorm = ((v_norm - mu_normed) ** 2).mean().item()
    mse_rawvar = ((v_norm - mu) ** 2).mean().item()
    return mse_meannorm, mse_rawvar
