"""Unit tests for the category-conditioned distribution-matching weight w*.

w*(m_truth) = clamp((p̂_truth(m_truth) / p̂_gen(m_truth))^α, w_min, w_max)
evaluated per tuple against THAT tuple's category KDE (type-composition +
charge-sign). See tuple_categories.py and test_tuple_categories.py for the
category-key tests; this file validates the lookup + the w* mechanics + the
end-to-end aux-loss integration using a single catch-all category (so the
behavior reduces to the per-category w* of one channel).

Validates:
1. `_density_aware_log_p(..., cat=)` linear interp lookup (truth and gen sides).
2. Identity at init: when p̂_gen == p̂_truth, w*(any m) = 1.
3. Sign correctness: w* > 1 in deficit regions; w* < 1 in excess regions.
4. Uniform log-shift: shifting log p̂_gen by Δ produces w* = exp(−α·Δ).
5. Clamp behavior at extremes.
6. End-to-end gradient flow with w* enabled (category-conditioned).

Run with: uv run pytest tests/unit/test_density_aware_multiplier.py -v
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from shellflow.data.tuple_categories import tuple_keys


def _build_lm(k_max: int = 4):
    """Build a tiny lightning module with distribution_matching disabled by
    default (no inline KDE build); tests inject buffers directly via
    _inject_dm_buffers when needed."""
    from shellflow.config.training_config import (
        OptimizerConfig,
        SchedulerConfig,
        TrainingConfig,
    )
    from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

    N = 6
    manifold_stats = {
        "11": {
            "e_ref": 30.0,
            "chi_mu": 0.0,
            "chi_sigma": 1.0,
            "logm_mu": 0.0,
            "logm_sigma": 1.0,
            "is_massless": True,
        },
        "13": {
            "e_ref": 30.0,
            "chi_mu": 0.0,
            "chi_sigma": 1.0,
            "logm_mu": 0.0,
            "logm_sigma": 1.0,
            "is_massless": True,
        },
    }
    aux_head_cfg = {
        "enabled": True,
        "lambda_pos": 1.0,
        "lambda_vel": 1.0,
        "pos_t_gate": 0.3,
        "k_max": k_max,
        "k_weights": {str(k): 1.0 for k in range(2, k_max + 1)},
        "hierarchical": {"enabled": True},
        "distribution_matching": {"enabled": False},
    }
    model_config = {
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "num_met_tokens": 2,
        "num_extra_features": 13,
        "max_particles": N,
        "manifold_stats": manifold_stats,
        "aux_head": aux_head_cfg,
        "huber_delta": 1.0,
        "sigma_prior": 1.0,
        "direction_geometry": "S2",
    }
    lm = RiemannianDiTsModule(
        model_config=model_config,
        optimizer_config=OptimizerConfig(lr=3e-4, weight_decay=0.01),
        scheduler_config=SchedulerConfig(name="cosine", warmup_pct=0.05, min_lr=1e-6),
        training_config=TrainingConfig(epochs=1, grad_clip=1.0, seed=42),
        total_steps=100,
    )
    return lm, N


# Test particles are all electrons with charge index 2 (e+), so every K-tuple
# maps to a single category key per K. The injected buffers register that one
# category as c0.
_TEST_TYPE = 11
_TEST_CHARGE_IDX = 2


def _single_key(K: int) -> int:
    t = torch.full((1, K), _TEST_TYPE)
    c = torch.full((1, K), _TEST_CHARGE_IDX)
    return int(tuple_keys(t, c, K)[0])


def _inject_dm_buffers(
    lm,
    K_max: int = 4,
    truth_peak_log_m: float = 0.0,
    gen_log_shift: float = 0.0,
    truth_log_p_grid: torch.Tensor | None = None,
    gen_log_p_grid: torch.Tensor | None = None,
):
    """Register a SINGLE-category (c0) set of distribution_matching buffers.

    All test tuples (all-electron, charge +) map to one category key per K, so
    a single category reproduces the old uniform-w* behavior per channel.

    `truth_peak_log_m`: centre of the Gaussian bump in the truth log_p grid.
    `gen_log_shift`: uniform additive shift on the gen grid vs truth.
    Pass explicit `truth_log_p_grid` / `gen_log_p_grid` to override the bumps.
    """
    log_m_grid = torch.linspace(-3.0, 7.5, 4096)
    if truth_log_p_grid is None:
        truth_log_p_grid = 5.0 * torch.exp(
            -((log_m_grid - truth_peak_log_m) ** 2) / (2 * 2.0**2)
        )
    if gen_log_p_grid is None:
        gen_log_p_grid = truth_log_p_grid + gen_log_shift
    log_p_floor = torch.tensor(-10.0)
    lm._dm_scheme = "type_charge"
    lm._dm_n_cat = {}
    for K in range(2, K_max + 1):
        lm.register_buffer(f"_da_log_m_grid_k{K}", log_m_grid.clone())
        lm.register_buffer(
            f"_da_keys_k{K}", torch.tensor([_single_key(K)], dtype=torch.int64)
        )
        lm.register_buffer(f"_da_log_p_grid_k{K}_c0", truth_log_p_grid.clone())
        lm.register_buffer(f"_da_log_p_gen_k{K}_c0", gen_log_p_grid.clone())
        lm.register_buffer(f"_da_log_p_floor_k{K}_c0", log_p_floor.clone())
        lm._dm_n_cat[K] = 1
    lm.use_aux_head_distribution_matching = True


def _make_inputs(N, B=2, seed=0):
    torch.manual_seed(seed)
    v_theta_aux = torch.randn(B, N, 5) * 0.1
    u_t = torch.randn(B, N, 5) * 0.1
    psi_t = torch.zeros(B, N, 5)
    psi_t[..., 1:4] = F.normalize(torch.randn(B, N, 3), dim=-1)
    x_data = torch.randn(B, N, 5)
    x_data[..., 1:4] = F.normalize(x_data[..., 1:4], dim=-1)
    x_data[..., 4] = 0.0
    t = torch.full((B,), 0.5)
    p_mask = torch.zeros(B, N)
    p_mask[:, :4] = 1.0
    ti = {
        "is_massless": torch.ones(B, N, dtype=torch.bool),
        "e_ref": torch.full((B, N), 30.0),
        "chi_mu": torch.zeros(B, N),
        "chi_sigma": torch.ones(B, N),
        "logm_mu": torch.zeros(B, N),
        "logm_sigma": torch.ones(B, N),
        "m_rest": torch.zeros(B, N),
        "active": p_mask > 0.5,
    }
    # All electrons (type 11), charge index 2 (+) → single category per K.
    p_type = torch.full((B, N), _TEST_TYPE, dtype=torch.long)
    p_charge = torch.full((B, N), _TEST_CHARGE_IDX, dtype=torch.long)
    return v_theta_aux, u_t, psi_t, x_data, t, ti, p_mask, p_type, p_charge


def _aux_loss(lm, inputs):
    """Call _compute_aux_head_loss threading p_type/p_charge for DM."""
    v, u, p, x, t, ti, m, p_type, p_charge = inputs
    return lm._compute_aux_head_loss(
        v, u, p, x, t, ti, m, p_type=p_type, p_charge=p_charge
    )


# --- Lookup correctness ----------------------------------------------------


class TestDensityAwareLogP:
    def test_returns_grid_value_at_grid_point_truth_side(self):
        lm, _N = _build_lm()
        _inject_dm_buffers(lm)
        grid = lm._da_log_m_grid_k2
        log_p = lm._da_log_p_grid_k2_c0
        sample_log_m = grid[1:6]
        result = lm._density_aware_log_p(sample_log_m, K=2, side="truth", cat=0)
        assert torch.allclose(result, log_p[1:6], atol=1e-5)

    def test_returns_grid_value_at_grid_point_gen_side(self):
        lm, _N = _build_lm()
        _inject_dm_buffers(lm, gen_log_shift=1.5)
        grid = lm._da_log_m_grid_k2
        log_p_gen = lm._da_log_p_gen_k2_c0
        sample_log_m = grid[1:6]
        result = lm._density_aware_log_p(sample_log_m, K=2, side="gen", cat=0)
        assert torch.allclose(result, log_p_gen[1:6], atol=1e-5)

    def test_default_side_is_truth(self):
        lm, _N = _build_lm()
        _inject_dm_buffers(lm, gen_log_shift=2.0)
        grid = lm._da_log_m_grid_k2
        sample_log_m = grid[100:110]
        result_default = lm._density_aware_log_p(sample_log_m, K=2, cat=0)
        result_truth = lm._density_aware_log_p(sample_log_m, K=2, side="truth", cat=0)
        assert torch.allclose(result_default, result_truth)

    def test_linear_interpolation_midpoints(self):
        lm, _N = _build_lm()
        _inject_dm_buffers(lm)
        grid = lm._da_log_m_grid_k2
        log_p = lm._da_log_p_grid_k2_c0
        mid_log_m = 0.5 * (grid[:-1] + grid[1:])
        expected = 0.5 * (log_p[:-1] + log_p[1:])
        result = lm._density_aware_log_p(mid_log_m[10:20], K=2, cat=0)
        assert torch.allclose(result, expected[10:20], atol=1e-4)

    def test_floor_clamp_applied(self):
        lm, _N = _build_lm()
        log_m_grid = torch.linspace(-3.0, 7.5, 4096)
        log_p_grid = torch.full_like(log_m_grid, -50.0)
        _inject_dm_buffers(
            lm, truth_log_p_grid=log_p_grid, gen_log_p_grid=log_p_grid.clone()
        )
        result = lm._density_aware_log_p(torch.tensor([0.0, 1.0, 2.0]), K=2, cat=0)
        assert (result == -10.0).all()


# --- w* helper correctness -------------------------------------------------


def _w_star(lm, log_m, K, alpha, eta):
    """Categorized w* with all tuples in the single category c0."""
    cat_ids = torch.zeros_like(log_m, dtype=torch.long)
    return lm._distribution_matching_weight_categorized(
        log_m, cat_ids, K, alpha=alpha, eta=eta
    )


def _expected_w(d, alpha, eta):
    """w* = (p̂_truth / ((1-η)·p̂_gen + η·p̂_truth))^α, with d = log(p̂_gen/p̂_truth)."""
    return ((1.0 - eta) * math.exp(d) + eta) ** (-alpha)


class TestDistributionMatchingWeight:
    """Tests of `_distribution_matching_weight_categorized` (single category).

    w* = (p̂_truth / ((1-η)·p̂_gen + η·p̂_truth))^α — truth-prior-regularized,
    bounded above by (1/η)^α, → 0 in excess, no hard clamp.
    """

    def test_identity_at_init(self):
        lm, _N = _build_lm()
        _inject_dm_buffers(lm, gen_log_shift=0.0)  # gen identical to truth
        log_m = torch.linspace(-2.0, 5.0, 20)
        w = _w_star(lm, log_m, K=2, alpha=1.0, eta=0.2)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-6)

    def test_uniform_log_shift_gen_above_truth(self):
        lm, _N = _build_lm()
        delta = 1.5  # gen above truth (excess) → w* < 1
        _inject_dm_buffers(lm, gen_log_shift=delta)
        log_m = torch.linspace(-1.5, 4.5, 15)
        eta = 0.2
        for alpha in (0.5, 1.0, 2.0):
            w = _w_star(lm, log_m, K=2, alpha=alpha, eta=eta)
            expected = torch.full_like(w, _expected_w(delta, alpha, eta))
            assert torch.all(w < 1.0)
            assert torch.allclose(w, expected, atol=1e-5)

    def test_uniform_log_shift_gen_below_truth(self):
        lm, _N = _build_lm()
        delta = 1.0  # gen below truth (deficit) → w* > 1
        _inject_dm_buffers(lm, gen_log_shift=-delta)
        log_m = torch.linspace(-1.5, 4.5, 15)
        eta = 0.2
        w = _w_star(lm, log_m, K=2, alpha=1.0, eta=eta)
        expected = torch.full_like(w, _expected_w(-delta, 1.0, eta))
        assert torch.all(w > 1.0)
        assert torch.allclose(w, expected, atol=1e-5)

    def test_sign_correctness_pointwise(self):
        lm, _N = _build_lm()
        log_m_grid = torch.linspace(-3.0, 7.5, 4096)
        truth_log_p = 5.0 * torch.exp(-((log_m_grid - 0.0) ** 2) / 8.0)
        gen_log_p = 5.0 * torch.exp(-((log_m_grid - 2.0) ** 2) / 8.0)
        _inject_dm_buffers(lm, truth_log_p_grid=truth_log_p, gen_log_p_grid=gen_log_p)
        # Where truth is denser than gen (around log_m = 0): w* > 1
        w_truth_peak = _w_star(lm, torch.tensor([0.0]), K=2, alpha=1.0, eta=0.2)
        assert w_truth_peak.item() > 1.0
        # Where gen is denser than truth (around log_m = 2): w* < 1
        w_gen_peak = _w_star(lm, torch.tensor([2.0]), K=2, alpha=1.0, eta=0.2)
        assert w_gen_peak.item() < 1.0

    def test_eta_bounds(self):
        """Upper side saturates smoothly at (1/η)^α; excess side → 0 (no floor)."""
        lm, _N = _build_lm()
        eta = 0.2
        # Huge deficit (gen → 0): w* → ceiling (1/η)^α = 5.0, smoothly (no clamp).
        _inject_dm_buffers(lm, gen_log_shift=-20.0)
        w = _w_star(lm, torch.tensor([0.0]), K=2, alpha=1.0, eta=eta)
        assert abs(w.item() - (1.0 / eta)) < 1e-3

        # Huge excess (gen → ∞): w* → 0 — there is no lower clamp.
        _inject_dm_buffers(lm, gen_log_shift=20.0)
        w = _w_star(lm, torch.tensor([0.0]), K=2, alpha=1.0, eta=eta)
        assert w.item() < 1e-3

    def test_fallback_category_w_is_one(self):
        """Tuples with cat_id == -1 (unknown key) get exactly w*=1."""
        lm, _N = _build_lm()
        _inject_dm_buffers(lm, gen_log_shift=-2.0)  # would give w* > 1 if matched
        log_m = torch.zeros(1, 3)
        cat_ids = torch.tensor([[0, -1, 0]])  # middle tuple is fallback
        w = lm._distribution_matching_weight_categorized(
            log_m, cat_ids, K=2, alpha=1.0, eta=0.2
        )
        assert w[0, 1].item() == 1.0  # fallback
        assert w[0, 0].item() > 1.0 and w[0, 2].item() > 1.0  # matched


# --- Aux head loss integration --------------------------------------------


class TestAuxLossWithDistributionMatching:
    def test_disabled_is_bit_identical(self):
        """distribution_matching=false leaves the loss unchanged (regression guard)."""
        lm, N = _build_lm()
        inp = _make_inputs(N)
        loss1, _ = _aux_loss(lm, inp)

        lm2, _ = _build_lm()
        inp2 = _make_inputs(N, seed=0)
        loss2, _ = _aux_loss(lm2, inp2)
        assert torch.allclose(loss1, loss2, rtol=0, atol=0)

    def test_enabled_identity_at_init_matches_disabled(self):
        lm_off, N = _build_lm()
        lm_on, _ = _build_lm()
        _inject_dm_buffers(lm_on, gen_log_shift=0.0)  # identity init

        inp = _make_inputs(N, seed=1)
        loss_off, _ = _aux_loss(lm_off, inp)
        loss_on, _ = _aux_loss(lm_on, inp)
        assert torch.allclose(loss_off, loss_on, atol=1e-6)

    def test_uniform_excess_reduces_loss(self):
        lm_off, N = _build_lm()
        lm_on, _ = _build_lm()
        _inject_dm_buffers(lm_on, gen_log_shift=1.0)  # gen above truth → w* < 1

        inp = _make_inputs(N, seed=2)
        loss_off, _ = _aux_loss(lm_off, inp)
        loss_on, _ = _aux_loss(lm_on, inp)
        assert loss_on.item() < loss_off.item()

    def test_uniform_deficit_increases_loss(self):
        lm_off, N = _build_lm()
        lm_on, _ = _build_lm()
        _inject_dm_buffers(lm_on, gen_log_shift=-1.0)  # gen below truth → w* > 1

        inp = _make_inputs(N, seed=3)
        loss_off, _ = _aux_loss(lm_off, inp)
        loss_on, _ = _aux_loss(lm_on, inp)
        assert loss_on.item() > loss_off.item()

    def test_gradient_flows_through_weight(self):
        lm, N = _build_lm()
        _inject_dm_buffers(lm, gen_log_shift=-0.5)
        inp = list(_make_inputs(N, seed=4))
        inp[0] = inp[0].detach().clone().requires_grad_(True)
        loss, _ = _aux_loss(lm, tuple(inp))
        loss.backward()
        assert inp[0].grad is not None
        assert torch.isfinite(inp[0].grad).all()
        assert (inp[0].grad.abs() > 0).any()
