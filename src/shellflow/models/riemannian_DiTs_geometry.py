"""
Riemannian geometry primitives for flow matching on the product manifold
R (log energy-scale) × S² (direction) × R (log mass, massive types only).

Implements:
  - Taylor-branch stable exp/log maps on S²
  - Geodesic interpolant and conditional velocity on S² (RFM)
  - Tangent-space projection on S²
  - Per-type normalisation constants shared by encode/decode

All 4-vectors use (E, px, py, pz) ordering with Minkowski metric (-,+,+,+);
collator order is (px, py, pz, E) — see the reordering utilities at the end.
"""

import math

import torch

# =============================================================================
# Particle type classification and normalisation constants
# =============================================================================

MASSLESS_TYPES: set[int] = {11, 13, 22}  # electron, muon, photon
MASSIVE_TYPES: set[int] = {15, 1, 10}  # tau, jet, large-R jet

E_REF: dict[int, float] = {
    11: 50.0,  # electron
    13: 60.0,  # muon
    22: 70.0,  # photon
    15: 50.0,  # tau
    1: 150.0,  # jet
    10: 500.0,  # large-R jet
}

MASS_FLOOR: dict[int, float] = {
    15: 0.05,  # tau — ATLAS visible-reco mass tail extends to ~0.11 GeV (below m_π)
    1: 0.5,  # jet
    10: 5.0,  # large-R jet
}

# Log-mass normalisation: (mu, sigma) computed from training data
LOGM_STATS: dict[int, tuple[float, float]] = {
    15: (math.log(1.5), 0.4),  # tau
    1: (math.log(20.0), 1.0),  # jet
    10: (math.log(100.0), 0.8),  # large-R jet
}

# Chi (boost rapidity) normalisation: (mu, sigma) of arccosh(E/m) per massive type
# Fallback defaults — prefer data-driven values from dataloader_config YAML
CHI_STATS: dict[int, tuple[float, float]] = {
    15: (6.71, 0.88),  # tau
    1: (3.23, 0.66),  # jet
    10: (2.50, 0.59),  # large-R jet
}

# Minimum energy floors (GeV) for log computation
E_MIN: dict[int, float] = {
    11: 1.0,
    13: 3.0,
    22: 1.0,
}


# =============================================================================
# Sphere S² operations — Taylor-branch stable
# =============================================================================

_TAYLOR_THRESHOLD = 1e-3  # well above float32 precision floor (1.2e-7)


def log_map_sphere_stable(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = _TAYLOR_THRESHOLD,
) -> torch.Tensor:
    """Log map on S². x, y: (..., 3) unit vectors.

    Taylor: θ/sin(θ) → 1 + (1-cos θ)/3 as θ→0.
    """
    cos_t = (x * y).sum(-1).clamp(-1.0 + eps, 1.0 - eps)
    one_m_cos = 1.0 - cos_t  # ≈ θ²/2 for small θ

    use_taylor = one_m_cos < threshold
    # Safe-guard exact branch: replace near-1.0 cos_t with 0.0 so
    # acos/sin never see near-zero arguments. torch.where computes gradients
    # through BOTH branches — without this, NaN gradient leaks.
    safe_cos_t = torch.where(use_taylor, torch.zeros_like(cos_t), cos_t)
    theta_safe = torch.acos(safe_cos_t)
    coeff_exact = theta_safe / torch.sin(theta_safe).clamp(min=eps)
    coeff_taylor = 1.0 + one_m_cos / 3.0

    coeff = torch.where(use_taylor, coeff_taylor, coeff_exact)
    return coeff.unsqueeze(-1) * (y - cos_t.unsqueeze(-1) * x)


def exp_map_sphere_stable(
    x: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = _TAYLOR_THRESHOLD,
) -> torch.Tensor:
    """Exp map on S². x: (..., 3) on S², v: (..., 3) tangent.

    Taylor: sin(||v||)/||v|| → 1 - ||v||²/6 as ||v||→0.
    Returns re-normalized unit vector.
    """
    v_norm = torch.sqrt((v**2).sum(-1, keepdim=True) + eps)
    use_taylor = v_norm < threshold
    # Safe-guard exact branch: replace near-zero v_norm with 1.0 so
    # sin(v_norm)/v_norm never divides by ~0. torch.where computes gradients
    # through BOTH branches — without this, NaN gradient leaks.
    safe_v_norm = torch.where(use_taylor, torch.ones_like(v_norm), v_norm)
    sinc_exact = torch.sin(safe_v_norm) / safe_v_norm
    sinc_taylor = 1.0 - v_norm**2 / 6.0
    sinc = torch.where(use_taylor, sinc_taylor, sinc_exact)
    result = torch.cos(v_norm) * x + sinc * v
    # Re-normalize to unit sphere (FP drift safety)
    return result / result.norm(dim=-1, keepdim=True).clamp(min=eps)


def sample_uniform_sphere(ref: torch.Tensor) -> torch.Tensor:
    """Sample uniform point on S². Shape matches ref: (..., 3)."""
    v = torch.randn_like(ref)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# =============================================================================
# Coordinate reordering utilities
# =============================================================================


def collator_to_minkowski(kin: torch.Tensor) -> torch.Tensor:
    """Convert collator output (px, py, pz, E) to Minkowski order (E, px, py, pz)."""
    return torch.stack([kin[..., 3], kin[..., 0], kin[..., 1], kin[..., 2]], dim=-1)


def minkowski_to_collator(p: torch.Tensor) -> torch.Tensor:
    """Convert Minkowski order (E, px, py, pz) to collator output (px, py, pz, E)."""
    return torch.stack([p[..., 1], p[..., 2], p[..., 3], p[..., 0]], dim=-1)


# =============================================================================
# Riemannian Flow Matching primitives
# =============================================================================


def geodesic_interpolant_S2(
    n0: torch.Tensor,
    n1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Geodesic interpolant ψ_t = exp_{n0}(t · log_{n0}(n1)) on S².

    Args:
        n0: (..., 3) prior unit direction
        n1: (..., 3) data unit direction
        t: (...,) interpolation time in [0, 1]
    Returns:
        psi_t: (..., 3) unit direction on S²
    """
    v = log_map_sphere_stable(n0, n1)
    return exp_map_sphere_stable(n0, t.unsqueeze(-1) * v)


def conditional_velocity_S2(
    psi_t_n: torch.Tensor,
    n1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Conditional velocity u_t = log_{ψ_t}(n1) / (1 - t) on S².

    Args:
        psi_t_n: (..., 3) current interpolant direction on S²
        n1: (..., 3) data target direction on S²
        t: (...,) current time (must be < 1)
    Returns:
        u_t: (..., 3) tangent vector at psi_t_n on S²
    """
    log_v = log_map_sphere_stable(psi_t_n, n1)
    return log_v / (1.0 - t + eps).unsqueeze(-1)


def tangent_project_S2(
    v_raw: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """Project v_raw to tangent space T_n S²: v_tan = v_raw - (v_raw · n) n.

    Guarantees n · v_tan = 0 exactly.

    Args:
        v_raw: (..., 3) raw 3-vector
        n: (..., 3) unit direction on S²
    Returns:
        v_tan: (..., 3) tangent vector at n
    """
    return v_raw - (v_raw * n).sum(-1, keepdim=True) * n
