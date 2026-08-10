"""Unit tests for the free Euclidean chart (E1 ablation).

The free chart (`model.chart: "free"`) is the Euclidean-ablation chart used by
the V1/V3 ablation variants: (logE_norm, nx, ny, nz, log|p|_norm) for ALL
particle types, with E and |p| as independent coordinates (no on-shell tie) and
R³ linear direction geometry (no S² geodesics). These tests pin down:

  1. the chart guard (free requires R3),
  2. encode → decode roundtrip of E and |p| independent of mass,
  3. off-shell representability (the on-shell tie is really gone),
  4. that the on-shell chart remains the untouched default,
  5. velocity postprocessing keeps s4 live for massless types,
  6. the free-chart decode helpers (values + velocity chain rule),
  7. the R³ interpolant is exactly linear on all 5 dims with live s4.
"""

import pytest
import torch
import torch.nn.functional as F

from shellflow.config.training_config import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from shellflow.models.riemannian_DiTs_model import (
    RiemannianDiTsModel,
    create_riemannian_dits,
)
from shellflow.training.riemannian_DiTs_lightning import (
    RiemannianDiTsModule,
    _decode_particle_4mom,
    _decode_particle_4mom_with_velocity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MUON, ELECTRON, PHOTON, JET = 13, 11, 22, 1


def _make_model(chart="free", direction_geometry="R3", **kw):
    return RiemannianDiTsModel(
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_extra_features=13,
        met_embed_dim=8,
        num_met_tokens=2,
        chart=chart,
        direction_geometry=direction_geometry,
        **kw,
    )


def _make_module(chart="free", direction_geometry="R3"):
    model_config = {
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "num_met_tokens": 2,
        "num_extra_features": 13,
        "huber_delta": 1.0,
        "sigma_prior": 1.0,
        "chart": chart,
        "direction_geometry": direction_geometry,
    }
    return RiemannianDiTsModule(
        model_config=model_config,
        optimizer_config=OptimizerConfig(lr=3e-4, weight_decay=0.01),
        scheduler_config=SchedulerConfig(name="cosine", warmup_pct=0.05, min_lr=1e-6),
        training_config=TrainingConfig(epochs=1, grad_clip=1.0, seed=42),
        total_steps=100,
    )


def _onshell_kin(p3, m):
    """Collator-order (px, py, pz, E) four-vector with mass m."""
    E = torch.sqrt((p3 * p3).sum(-1) + m * m)
    return torch.cat([p3, E.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------------------
# 1. Guard
# ---------------------------------------------------------------------------


def test_chart_guard_free_requires_r3():
    with pytest.raises(ValueError, match="requires direction_geometry='R3'"):
        _make_model(chart="free", direction_geometry="S2")
    with pytest.raises(ValueError, match="chart must be"):
        _make_model(chart="euclidean", direction_geometry="R3")
    # Valid combos construct fine
    _make_model(chart="free", direction_geometry="R3")
    _make_model(chart="onshell", direction_geometry="S2")


def test_factory_reads_chart_key():
    cfg = {
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 2,
        "chart": "free",
        "direction_geometry": "R3",
    }
    model = create_riemannian_dits(cfg)
    assert model.chart == "free"
    # Default is the paper chart
    cfg.pop("chart")
    cfg["direction_geometry"] = "S2"
    assert create_riemannian_dits(cfg).chart == "onshell"


# ---------------------------------------------------------------------------
# 2./3. Encode/decode: roundtrip and off-shell representability
# ---------------------------------------------------------------------------


def test_free_chart_roundtrip_recovers_E_and_p_independently():
    torch.manual_seed(0)
    model = _make_model()
    B, N = 4, 5
    p_type = torch.tensor([MUON, ELECTRON, PHOTON, JET, MUON]).expand(B, N)
    p_mask = torch.ones(B, N)

    # Deliberately OFF-shell inputs: E has an extra mass-like offset that the
    # chart must carry through |p| and E independently.
    p3 = F.normalize(torch.randn(B, N, 3), dim=-1) * (20.0 + 80.0 * torch.rand(B, N, 1))
    m_fake = 5.0 + 10.0 * torch.rand(B, N)  # unphysical for e/mu/gamma
    kin = _onshell_kin(p3, m_fake)

    manifold = model.encode_to_manifold(kin, p_type, p_mask)
    kin_back = model.decode_from_manifold(manifold, p_type, p_mask)

    # E and p⃗ both survive the roundtrip — mass plays no role in the chart.
    assert torch.allclose(kin_back, kin, rtol=1e-4, atol=1e-3), (
        (kin_back - kin).abs().max()
    )


def test_free_chart_decodes_offshell_states():
    model = _make_model()
    B, N = 1, 1
    p_type = torch.full((B, N), PHOTON, dtype=torch.long)
    p_mask = torch.ones(B, N)
    e_ref = model._e_ref_lut[model.map_type_to_idx(p_type)]  # (1,1)

    # Hand-crafted state: E = 50 GeV but |p| = 30 GeV — a 40 GeV "photon mass".
    E_tgt, p_tgt = 50.0, 30.0
    state = torch.zeros(B, N, 5)
    state[..., 0] = torch.log(E_tgt / e_ref)
    state[..., 1:4] = F.normalize(torch.tensor([[[1.0, 2.0, 3.0]]]), dim=-1)
    state[..., 4] = torch.log(p_tgt / e_ref)

    kin = model.decode_from_manifold(state, p_type, p_mask)
    E = kin[..., 3]
    p_mag = kin[..., :3].norm(dim=-1)
    assert torch.allclose(E, torch.tensor(E_tgt), rtol=1e-5)
    assert torch.allclose(p_mag, torch.tensor(p_tgt), rtol=1e-5)
    m2 = E * E - p_mag * p_mag
    assert (m2 > 100.0).all(), "free chart must allow off-shell decode"


def test_onshell_chart_still_enforces_shell():
    """Control: the default chart forces |p| = sqrt(E² − m_rest²) regardless
    of what s4 says — photon lightlike, muon exactly on the muon shell."""
    model = _make_model(chart="onshell", direction_geometry="S2")
    B, N = 1, 2
    p_type = torch.tensor([[PHOTON, MUON]])
    p_mask = torch.ones(B, N)
    state = torch.zeros(B, N, 5)
    state[..., 0] = 0.3
    state[..., 1:4] = F.normalize(torch.randn(B, N, 3), dim=-1)
    state[..., 4] = 1.7  # garbage in s4 — must be ignored for massless types

    kin = model.decode_from_manifold(state, p_type, p_mask)
    E = kin[..., 3]
    p_mag = kin[..., :3].norm(dim=-1)
    m2 = E * E - p_mag * p_mag
    m_mu = 0.105658
    # atol reflects fp32 cancellation in E² − |p|² at E ~ 80 GeV (ulp ~ 5e-4
    # on E² ~ 6.5e3), not decode error — the decode itself is exact.
    assert torch.allclose(m2[0, 0], torch.tensor(0.0), atol=5e-3)  # photon
    assert torch.allclose(m2[0, 1], torch.tensor(m_mu**2), atol=5e-3)  # muon


# ---------------------------------------------------------------------------
# 5. Velocity postprocessing keeps s4 live in the free chart
# ---------------------------------------------------------------------------


def test_postprocess_velocity_keeps_s4_for_massless():
    torch.manual_seed(1)
    B, N = 2, 3
    p_type = torch.tensor([[MUON, ELECTRON, PHOTON]]).expand(B, N)
    p_mask = torch.ones(B, N)
    raw = torch.randn(B, N, 5)
    z_t = torch.randn(B, N, 5)

    model_free = _make_model()
    out_free = model_free._postprocess_velocity(
        raw.clone(), z_t, model_free.map_type_to_idx(p_type), p_mask
    )
    assert torch.allclose(out_free[..., 4], raw[..., 4]), (
        "free chart must not zero the s4 velocity of massless types"
    )
    # R3 mode: direction dims pass through untouched too
    assert torch.allclose(out_free[..., 1:4], raw[..., 1:4])

    model_on = _make_model(chart="onshell", direction_geometry="S2")
    out_on = model_on._postprocess_velocity(
        raw.clone(), z_t, model_on.map_type_to_idx(p_type), p_mask
    )
    assert torch.all(out_on[..., 4] == 0.0), (
        "on-shell chart zeroes s4 velocity for massless types (control)"
    )


# ---------------------------------------------------------------------------
# 6. Free-chart decode helpers: values and velocity chain rule
# ---------------------------------------------------------------------------


def _helper_luts(B, N, e_ref_val=50.0):
    e_ref = torch.full((B, N), e_ref_val)
    ones = torch.ones(B, N)
    zeros = torch.zeros(B, N)
    return {
        "e_ref": e_ref,
        "is_massless": torch.ones(B, N, dtype=torch.bool),
        "chi_mu": zeros,
        "chi_sigma": ones,
        "logm_mu": zeros,
        "logm_sigma": ones,
        "m_rest": zeros,
    }


def test_decode_particle_4mom_free_values():
    torch.manual_seed(2)
    B, N = 2, 4
    x = torch.randn(B, N, 5)
    luts = _helper_luts(B, N)
    E, p3 = _decode_particle_4mom(x, free_chart=True, **luts)
    assert torch.allclose(E, luts["e_ref"] * torch.exp(x[..., 0]), rtol=1e-6)
    assert torch.allclose(
        p3.norm(dim=-1), luts["e_ref"] * torch.exp(x[..., 4]), rtol=1e-5
    )
    n_hat = F.normalize(x[..., 1:4], dim=-1)
    assert torch.allclose(F.normalize(p3, dim=-1), n_hat, atol=1e-5)


def test_decode_particle_4mom_with_velocity_free_chain_rule():
    """Finite-difference check: decoding x + eps·v must move (E, p⃗) by
    eps·(dE/dt, dp⃗/dt) to first order."""
    torch.manual_seed(3)
    B, N, eps = 2, 4, 1e-4
    x = torch.randn(B, N, 5)
    x[..., 1:4] = F.normalize(x[..., 1:4], dim=-1)  # start near the sphere
    v = torch.randn(B, N, 5)
    luts = _helper_luts(B, N)

    E0, p30, dE, dp3 = _decode_particle_4mom_with_velocity(
        x, v, free_chart=True, **luts
    )
    E1, p31, _, _ = _decode_particle_4mom_with_velocity(
        x + eps * v, v, free_chart=True, **luts
    )
    assert torch.allclose((E1 - E0) / eps, dE, rtol=1e-2, atol=1e-2)
    assert torch.allclose((p31 - p30) / eps, dp3, rtol=1e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# 7. R³ interpolant: exactly linear, s4 live for massless
# ---------------------------------------------------------------------------


def test_rfm_interpolant_linear_and_s4_live_free_chart():
    torch.manual_seed(4)
    module = _make_module(chart="free", direction_geometry="R3")
    B, N = 3, 4
    p_type = torch.tensor([[MUON, ELECTRON, PHOTON, JET]]).expand(B, N)
    p_mask = torch.ones(B, N)
    ti = module._precompute_type_info(p_type, p_mask)

    x_data = torch.randn(B, N, 5)
    x_data[..., 1:4] = F.normalize(x_data[..., 1:4], dim=-1)
    t = torch.rand(B)

    psi_t, u_t, x_0 = module._rfm_interpolant(x_data, t, ti)

    t_exp = t.view(B, 1, 1)
    assert torch.allclose(psi_t, (1 - t_exp) * x_0 + t_exp * x_data, atol=1e-5), (
        "free-chart R³ interpolant must be a straight line on ALL 5 dims"
    )
    assert torch.allclose(u_t, x_data - x_0, atol=1e-5), (
        "free-chart target velocity must be the constant OT difference"
    )
    # s4 must be a live coordinate for massless types: the prior draw is
    # Gaussian, so exactly-zero values have measure zero.
    massless = ti["is_massless"]
    assert (x_0[..., 4][massless] != 0.0).all(), (
        "massless s4 prior must not be zeroed in the free chart"
    )


def test_rfm_interpolant_onshell_still_zeroes_massless_s4():
    torch.manual_seed(5)
    module = _make_module(chart="onshell", direction_geometry="S2")
    B, N = 2, 3
    p_type = torch.tensor([[MUON, ELECTRON, PHOTON]]).expand(B, N)
    p_mask = torch.ones(B, N)
    ti = module._precompute_type_info(p_type, p_mask)

    x_data = torch.randn(B, N, 5)
    x_data[..., 1:4] = F.normalize(x_data[..., 1:4], dim=-1)
    x_data[..., 4] = 0.0  # encode zeroes massless s4
    t = torch.rand(B)

    psi_t, u_t, x_0 = module._rfm_interpolant(x_data, t, ti)
    assert torch.all(psi_t[..., 4] == 0.0)
    assert torch.all(u_t[..., 4] == 0.0)
    assert torch.all(x_0[..., 4] == 0.0)
