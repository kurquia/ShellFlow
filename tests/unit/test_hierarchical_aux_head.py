"""Unit tests for hierarchical (analytical) K-body aux head + quality-weighted loss.

The hierarchical mode reorganizes the K-body sum computation around pair
sums: pair (E_ij, p_ij, dE/dt, dp/dt) is computed once and reused for K=3
(pair + single) and K=4 (canonical pair + pair). It is mathematically
identical to the flat per-K-tuple sum because 4-momentum sums are
permutation-invariant in the K particles. NO new learned parameters.

Validates:
1. QualityScorer module (the only new module): outputs in [0, 1], MET → q=1,
   clean leptons → high q, bad leptons → low q, inactive → q=1.
2. Buffer correctness: triplet/quad pair-index buffers correctly map K-tuples
   to upper-triangle pair indices for the canonical decomposition.
3. Analytical hierarchy equivalence: with hierarchical=true, the loss must
   be NUMERICALLY IDENTICAL to hierarchical=false (since the only difference
   is the order/grouping of summation, which is associative for floats up to
   small rounding).
4. Quality weighting: enables the per-particle quality metric, applies
   multiplicatively to the K-body mask, and validates required args.
5. End-to-end: forward + backward through aux head with hierarchical and/or
   quality enabled produces finite loss with gradient flowing to backbone.

Run with: uv run pytest tests/unit/test_hierarchical_aux_head.py -v
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shellflow.models.riemannian_DiTs_model import (
    QualityScorer,
    RiemannianDiTsModel,
)

# --- QualityScorer tests --------------------------------------------------


def _strong_lep_extras(B=2, N=6):
    """Leptons with low iso (clean), low d0sig (prompt)."""
    extras = torch.zeros(B, N, 13)
    extras[..., 5] = 0.5  # ptvarcone30 = 0.5 GeV → iso/pT ≈ 0.01 (well below 0.15)
    extras[..., 6] = 0.5  # topoetcone20
    extras[..., 7] = 0.5  # |d0sig| = 0.5 (well below 3.0)
    return extras


class TestQualityScorer:
    def test_outputs_in_unit_range(self):
        scorer = QualityScorer({})
        B, N = 2, 6
        p_type = torch.tensor([[11, 13, 22, 1, 15, 10], [11, 11, 13, 13, 22, 1]])
        extras = torch.randn(B, N, 13)
        p3_true = torch.randn(B, N, 3) * 50
        mask = torch.ones(B, N, dtype=torch.bool)
        q = scorer(p_type, extras, p3_true, mask)
        assert q.shape == (B, N)
        assert (q >= 0).all() and (q <= 1).all()

    def test_met_gets_q_one(self):
        scorer = QualityScorer({})
        p_type = torch.zeros(1, 4, dtype=torch.long)  # all MET
        extras = torch.randn(1, 4, 13)
        p3_true = torch.randn(1, 4, 3)
        mask = torch.ones(1, 4, dtype=torch.bool)
        q = scorer(p_type, extras, p3_true, mask)
        assert torch.allclose(q, torch.ones_like(q))

    def test_inactive_particle_gets_q_one(self):
        scorer = QualityScorer({})
        p_type = torch.tensor([[11, 11, 11, 11]])
        extras = torch.zeros(1, 4, 13)
        extras[..., 5] = 100  # bad iso so q would normally be small
        extras[..., 7] = 100  # bad IP
        p3_true = torch.zeros(1, 4, 3)
        p3_true[..., 0] = 50
        mask = torch.tensor([[True, True, False, False]])
        q = scorer(p_type, extras, p3_true, mask)
        assert q[0, 0] < 0.5 and q[0, 1] < 0.5
        assert q[0, 2] == 1.0 and q[0, 3] == 1.0

    def test_clean_lepton_higher_q_than_dirty_lepton(self):
        scorer = QualityScorer({})
        p_type = torch.full((1, 4), 11, dtype=torch.long)
        p3_true = torch.zeros(1, 4, 3)
        p3_true[..., 0] = 50
        clean_extras = _strong_lep_extras(1, 4)
        dirty_extras = torch.zeros(1, 4, 13)
        dirty_extras[..., 5] = 50.0
        dirty_extras[..., 6] = 50.0
        dirty_extras[..., 7] = 10.0
        mask = torch.ones(1, 4, dtype=torch.bool)
        q_clean = scorer(p_type, clean_extras, p3_true, mask)
        q_dirty = scorer(p_type, dirty_extras, p3_true, mask)
        assert (q_clean > 0.5).all()
        assert (q_dirty < 0.1).all()
        assert (q_clean / q_dirty.clamp(min=1e-6) > 5).all()

    def test_jet_high_jvt_high_q(self):
        scorer = QualityScorer({})
        p_type = torch.full((1, 3), 1, dtype=torch.long)
        extras = torch.zeros(1, 3, 13)
        extras[..., 3] = 1.0
        p3_true = torch.zeros(1, 3, 3)
        p3_true[..., 0] = 50
        mask = torch.ones(1, 3, dtype=torch.bool)
        q = scorer(p_type, extras, p3_true, mask)
        assert (q > 0.9).all()

    def test_jet_low_jvt_low_q(self):
        scorer = QualityScorer({})
        p_type = torch.full((1, 3), 1, dtype=torch.long)
        extras = torch.zeros(1, 3, 13)
        extras[..., 3] = 0.0
        p3_true = torch.zeros(1, 3, 3)
        p3_true[..., 0] = 50
        mask = torch.ones(1, 3, dtype=torch.bool)
        q = scorer(p_type, extras, p3_true, mask)
        assert (q < 0.5).all()

    def test_no_grad(self):
        scorer = QualityScorer({})
        p_type = torch.tensor([[11, 11]])
        extras = torch.randn(1, 2, 13, requires_grad=True)
        p3_true = torch.randn(1, 2, 3, requires_grad=True) * 50
        mask = torch.ones(1, 2, dtype=torch.bool)
        q = scorer(p_type, extras, p3_true, mask)
        assert not q.requires_grad


# --- Model construction gates ---------------------------------------------


def _build_model(
    qual_enabled: bool = False,
    aux_enabled: bool = True,
    d_model: int = 32,
) -> RiemannianDiTsModel:
    torch.manual_seed(42)
    return RiemannianDiTsModel(
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_met_tokens=2,
        num_extra_features=13,
        aux_head_config={
            "enabled": aux_enabled,
            "hierarchical": {"enabled": False},  # purely analytical, no model modules
        },
        quality_weighting_config={"enabled": qual_enabled},
    )


class TestModelConstructionGates:
    def test_quality_disabled_no_scorer(self):
        m = _build_model(qual_enabled=False)
        assert m.quality_scorer is None

    def test_quality_enabled_creates_scorer(self):
        m = _build_model(qual_enabled=True)
        assert isinstance(m.quality_scorer, QualityScorer)

    def test_no_learned_hierarchical_modules_ever(self):
        """Hierarchical is purely analytical; model has no learned hierarchical modules."""
        m = _build_model(qual_enabled=True)
        assert not hasattr(m, "pair_rep_head")
        assert not hasattr(m, "triplet_rep_head")
        assert not hasattr(m, "quad_rep_head")
        assert not hasattr(m, "mass_proj")


# --- Lightning integration helpers ----------------------------------------


def _build_lm(
    hier_enabled: bool = False,
    qual_enabled: bool = False,
    k_max: int = 4,
):
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
        "aux_head": {
            "enabled": True,
            "lambda_pos": 1.0,
            "lambda_vel": 1.0,
            "pos_t_gate": 0.3,
            "k_max": k_max,
            "k_weights": {str(k): 1.0 for k in range(2, k_max + 1)},
            "hierarchical": {"enabled": hier_enabled},
        },
        "huber_delta": 1.0,
        "sigma_prior": 1.0,
        "direction_geometry": "S2",
    }
    return (
        RiemannianDiTsModule(
            model_config=model_config,
            optimizer_config=OptimizerConfig(lr=3e-4, weight_decay=0.01),
            scheduler_config=SchedulerConfig(
                name="cosine", warmup_pct=0.05, min_lr=1e-6
            ),
            training_config=TrainingConfig(epochs=1, grad_clip=1.0, seed=42),
            total_steps=100,
            quality_weighting_config={"enabled": qual_enabled},
        ),
        N,
    )


def _make_inputs(N, B=2, d_model=16, seed=0):
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
    p_type = torch.tensor([[11, 13, 11, 13, 0, 0]] * B)
    p_extra = torch.randn(B, N, 13)
    return v_theta_aux, u_t, psi_t, x_data, t, ti, p_mask, p_type, p_extra


# --- Buffer correctness ---------------------------------------------------


class TestPairIndexBuffers:
    def test_triplet_pair_idx_correctness(self):
        lm, N = _build_lm(hier_enabled=True, k_max=3)
        combs_3 = lm._comb_3
        for t_idx, (i, j, k) in enumerate(combs_3.tolist()):
            expected_pair_idx = i * N - i * (i + 1) // 2 + j - i - 1
            assert lm._triplet_pair_idx[t_idx].item() == expected_pair_idx
            assert lm._triplet_third_idx[t_idx].item() == k

    def test_quad_canonical_partition_count(self):
        """Each quad has ONE canonical partition (i,j)(k,l) for sorted tuple.
        Total = C(N, 4)."""
        lm, _N = _build_lm(hier_enabled=True, k_max=4)
        combs_4 = lm._comb_4
        n_quads = combs_4.shape[0]
        assert lm._quad_pair1_idx.shape[0] == n_quads
        assert lm._quad_pair2_idx.shape[0] == n_quads

    def test_quad_canonical_partition_correctness(self):
        lm, N = _build_lm(hier_enabled=True, k_max=4)
        combs_4 = lm._comb_4
        for q_idx, q in enumerate(combs_4):
            i, j, k, ell = q.tolist()
            expected_p1 = i * N - i * (i + 1) // 2 + j - i - 1  # (i, j)
            expected_p2 = k * N - k * (k + 1) // 2 + ell - k - 1  # (k, l)
            assert lm._quad_pair1_idx[q_idx].item() == expected_p1
            assert lm._quad_pair2_idx[q_idx].item() == expected_p2

    def test_no_hierarchical_buffers_when_disabled(self):
        lm, _ = _build_lm(hier_enabled=False, k_max=4)
        assert not hasattr(lm, "_triplet_pair_idx")
        assert not hasattr(lm, "_quad_pair1_idx")


# --- The KEY equivalence test --------------------------------------------


class TestHierarchicalEquivalence:
    """Hierarchical is purely a computational reorganization — must produce
    NUMERICALLY EQUIVALENT loss to flat (modulo small float-order rounding).
    """

    def test_hierarchical_matches_flat_loss(self):
        lm_flat, N = _build_lm(hier_enabled=False, k_max=4)
        lm_hier, _ = _build_lm(hier_enabled=True, k_max=4)
        v, u, psi, x, t, ti, mask, _p_type, _p_extra = _make_inputs(N, seed=42)

        result_flat = lm_flat._compute_aux_head_loss(v, u, psi, x, t, ti, mask)
        result_hier = lm_hier._compute_aux_head_loss(v, u, psi, x, t, ti, mask)
        assert result_flat is not None and result_hier is not None
        loss_flat, _ = result_flat
        loss_hier, _ = result_hier
        # Sum is associative for floats up to small rounding (~1e-6 atol per op).
        # Use a generous tolerance to accommodate a few accumulated roundings.
        torch.testing.assert_close(loss_flat, loss_hier, rtol=1e-5, atol=1e-5)

    def test_hierarchical_matches_flat_with_quality(self):
        lm_flat, N = _build_lm(hier_enabled=False, qual_enabled=True, k_max=4)
        lm_hier, _ = _build_lm(hier_enabled=True, qual_enabled=True, k_max=4)
        v, u, psi, x, t, ti, mask, p_type, p_extra = _make_inputs(N, seed=42)

        result_flat = lm_flat._compute_aux_head_loss(
            v, u, psi, x, t, ti, mask, p_type=p_type, p_extra=p_extra
        )
        result_hier = lm_hier._compute_aux_head_loss(
            v, u, psi, x, t, ti, mask, p_type=p_type, p_extra=p_extra
        )
        loss_flat, _ = result_flat
        loss_hier, _ = result_hier
        torch.testing.assert_close(loss_flat, loss_hier, rtol=1e-5, atol=1e-5)


# --- End-to-end loss + quality + gradient ---------------------------------


class TestEndToEnd:
    def test_hierarchical_returns_finite_with_grad(self):
        lm, N = _build_lm(hier_enabled=True)
        lm.train()
        # Use input that requires grad through v_theta_aux
        v, u, psi, x, t, ti, mask, _p_type, _p_extra = _make_inputs(N)
        v_grad = v.clone().requires_grad_(True)
        result = lm._compute_aux_head_loss(v_grad, u, psi, x, t, ti, mask)
        assert result is not None
        loss, metrics = result
        assert torch.isfinite(loss)
        assert "loss/aux_head_pos" in metrics
        loss.backward()
        assert v_grad.grad is not None and torch.isfinite(v_grad.grad).all()

    def test_quality_weighting_returns_finite_with_metric(self):
        lm, N = _build_lm(hier_enabled=False, qual_enabled=True)
        v, u, psi, x, t, ti, mask, p_type, p_extra = _make_inputs(N)
        result = lm._compute_aux_head_loss(
            v, u, psi, x, t, ti, mask, p_type=p_type, p_extra=p_extra
        )
        assert result is not None
        loss, _metrics = result
        assert torch.isfinite(loss)

    def test_combined_hierarchical_and_quality(self):
        lm, N = _build_lm(hier_enabled=True, qual_enabled=True)
        v, u, psi, x, t, ti, mask, p_type, p_extra = _make_inputs(N)
        result = lm._compute_aux_head_loss(
            v, u, psi, x, t, ti, mask, p_type=p_type, p_extra=p_extra
        )
        assert result is not None
        loss, metrics = result
        assert torch.isfinite(loss)
        # K=2 position loss should fire (largest mask weight). K≥3 may be
        # skipped with very small B and quality-reduced per-tuple weight —
        # that's correct behavior, not a bug.
        assert "loss/aux_head_pos" in metrics

    def test_quality_requires_p_type_and_p_extra(self):
        lm, N = _build_lm(qual_enabled=True)
        v, u, psi, x, t, ti, mask, _, _ = _make_inputs(N)
        try:
            lm._compute_aux_head_loss(
                v, u, psi, x, t, ti, mask, p_type=None, p_extra=None
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "quality_weighting" in str(e) or "p_type" in str(e)


# --- CFM particleness ------------------------------------------------------


def _build_lm_with_qw(cfm_q_floor: float, qw_enabled: bool = True):
    """Build a lightning module with consolidated quality_weighting set to a
    specific cfm_q_floor (and master enable flag)."""
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
        "huber_delta": 1.0,
        "sigma_prior": 1.0,
        "direction_geometry": "S2",
    }
    return (
        RiemannianDiTsModule(
            model_config=model_config,
            optimizer_config=OptimizerConfig(lr=3e-4, weight_decay=0.01),
            scheduler_config=SchedulerConfig(
                name="cosine", warmup_pct=0.05, min_lr=1e-6
            ),
            training_config=TrainingConfig(epochs=1, grad_clip=1.0, seed=42),
            total_steps=100,
            quality_weighting_config={
                "enabled": qw_enabled,
                "cfm_q_floor": cfm_q_floor,
            },
        ),
        N,
    )


def _make_cfm_inputs(N, B=2, seed=0):
    torch.manual_seed(seed)
    v_theta = torch.randn(B, N, 5) * 0.1
    u_t = torch.randn(B, N, 5) * 0.1
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
    p_type = torch.tensor([[11, 13, 11, 13, 0, 0]] * B)
    p_extra = torch.zeros(B, N, 13)
    # Dirty leptons: large iso & IP → q_i ≈ 0
    p_extra[..., 5] = 50.0
    p_extra[..., 6] = 50.0
    p_extra[..., 7] = 10.0
    kin = torch.zeros(B, N, 4)
    kin[..., 0] = 30.0  # px
    kin[..., 3] = 30.0  # E
    return v_theta, u_t, ti, p_type, p_extra, kin


class TestCFMParticleness:
    def test_cfm_floor_one_is_identity(self):
        """cfm_q_floor=1.0 → q_eff = 1 + 0·q = 1 → loss unchanged vs vanilla."""
        lm, N = _build_lm_with_qw(cfm_q_floor=1.0, qw_enabled=True)
        v, u, ti, p_type, p_extra, kin = _make_cfm_inputs(N)
        loss_unweighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=None, p_extra=None
        )
        loss_weighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=kin, p_extra=p_extra
        )
        assert torch.allclose(loss_unweighted, loss_weighted, atol=1e-6)

    def test_cfm_floor_zero_zeros_dirty_leptons(self):
        """cfm_q_floor=0.0 with dirty leptons → q_i ≈ 0 → per-particle loss ≈ 0."""
        lm, N = _build_lm_with_qw(cfm_q_floor=0.0, qw_enabled=True)
        v, u, ti, p_type, p_extra, kin = _make_cfm_inputs(N)
        loss_unweighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=None, p_extra=None
        )
        loss_weighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=kin, p_extra=p_extra
        )
        # Dirty leptons get q ≈ 0 → q_eff = 0 + 1·0 = 0 → loss collapses toward 0.
        assert loss_weighted.item() < loss_unweighted.item() * 0.1

    def test_cfm_floor_half_partial_softening(self):
        """cfm_q_floor=0.5 → q_eff ∈ [0.5, 1]; dirty leptons → q_eff ≈ 0.5 →
        weighted loss is half of unweighted (within tolerance)."""
        lm, N = _build_lm_with_qw(cfm_q_floor=0.5, qw_enabled=True)
        v, u, ti, p_type, p_extra, kin = _make_cfm_inputs(N)
        loss_unweighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=None, p_extra=None
        )
        loss_weighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=kin, p_extra=p_extra
        )
        ratio = (loss_weighted / loss_unweighted.clamp(min=1e-8)).item()
        assert 0.45 < ratio < 0.6, f"expected ratio ≈ 0.5, got {ratio}"

    def test_cfm_qw_disabled_no_weight(self):
        """quality_weighting.enabled=False → kin/p_extra ignored → vanilla CFM."""
        lm, N = _build_lm_with_qw(cfm_q_floor=0.0, qw_enabled=False)
        v, u, ti, p_type, p_extra, kin = _make_cfm_inputs(N)
        loss_unweighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=None, p_extra=None
        )
        loss_weighted, _ = lm._compute_cfm_loss(
            v, u, ti, p_type, kin=kin, p_extra=p_extra
        )
        assert torch.allclose(loss_unweighted, loss_weighted, atol=1e-6)
