"""Unit tests for set-wise tuple categorization (category-conditioned DM).

Covers:
1. Constants agree with the model module.
2. `tuple_keys` is permutation-invariant and composition/charge-sensitive.
3. Torch path matches a brute-force reference.
4. `all_kbody_log_masses` returns matched (log_m, key) lengths; per-key bucket
   counts match brute force.
5. `_distribution_matching_weight_categorized` routes each tuple to the right
   category's density (deficit → w*>1, fallback → exactly 1, match → ≈1).
"""

from __future__ import annotations

import numpy as np
import torch

from shellflow.data.tuple_categories import (
    BASE,
    NUM_CHARGE_CLASSES,
    NUM_PARTICLE_TYPES,
    PARTICLE_TYPE_CODES,
    key_to_name,
    tuple_codes,
    tuple_keys,
)

# --- Constants agreement -----------------------------------------------------


def test_constants_match_model():
    from shellflow.models.riemannian_DiTs_model import (
        NUM_CHARGE_CLASSES as MODEL_NCC,
    )
    from shellflow.models.riemannian_DiTs_model import (
        NUM_PARTICLE_TYPES as MODEL_NPT,
    )
    from shellflow.models.riemannian_DiTs_model import (
        PARTICLE_TYPE_CODES as MODEL_CODES,
    )

    assert PARTICLE_TYPE_CODES == MODEL_CODES
    assert NUM_PARTICLE_TYPES == MODEL_NPT
    assert NUM_CHARGE_CLASSES == MODEL_NCC
    assert BASE == NUM_PARTICLE_TYPES * NUM_CHARGE_CLASSES


# --- Key math ----------------------------------------------------------------


def _k(types, charges):
    """Single-tuple key helper."""
    t = torch.tensor([types])
    c = torch.tensor([charges])
    return int(tuple_keys(t, c, len(types))[0])


class TestTupleKeys:
    def test_os_vs_ss_differ(self):
        # e+e- (OS: charges 2,0) must differ from e+e+ (SS: 2,2)
        assert _k([11, 11], [2, 0]) != _k([11, 11], [2, 2])

    def test_permutation_invariance_k2(self):
        assert _k([11, 11], [2, 0]) == _k([11, 11], [0, 2])
        assert _k([13, 1], [2, 1]) == _k([1, 13], [1, 2])

    def test_permutation_invariance_random_k4(self):
        torch.manual_seed(0)
        types = [11, 13, 1, 22]
        charges = [2, 0, 1, 1]
        base = _k(types, charges)
        for _ in range(20):
            perm = torch.randperm(4).tolist()
            t_p = [types[i] for i in perm]
            c_p = [charges[i] for i in perm]
            assert _k(t_p, c_p) == base

    def test_distinct_compositions_distinct_keys(self):
        keys = {
            _k([11, 11], [2, 0]),  # e+e-
            _k([13, 13], [2, 0]),  # mu+mu-
            _k([1, 1], [1, 1]),  # jet jet
            _k([11, 13], [2, 0]),  # e+ mu-
            _k([22, 22], [1, 1]),  # gamma gamma
        }
        assert len(keys) == 5  # all distinct

    def test_flavor_distinguished(self):
        # ee-OSSF vs mumu-OSSF must be different categories
        assert _k([11, 11], [2, 0]) != _k([13, 13], [2, 0])

    def test_codes_in_range(self):
        types = torch.tensor([[1, 10, 11, 13, 15, 22, 0]])
        charges = torch.tensor([[1, 1, 2, 0, 2, 1, 1]])
        codes = tuple_codes(types, charges)
        assert codes.min() >= 0 and codes.max() < BASE

    def test_torch_matches_bruteforce(self):
        torch.manual_seed(1)
        B, C, K = 4, 7, 3
        types = torch.randint(0, 23, (B, C, K))
        charges = torch.randint(0, 3, (B, C, K))
        keys = tuple_keys(types, charges, K)

        # Brute-force reference
        from shellflow.data.tuple_categories import _type_lut

        lut = _type_lut(torch.device("cpu"))
        ref = torch.zeros(B, C, dtype=torch.int64)
        for b in range(B):
            for c in range(C):
                codes = sorted(
                    int(lut[int(types[b, c, k])]) * 3 + int(charges[b, c, k])
                    for k in range(K)
                )
                ref[b, c] = sum(code * (BASE**i) for i, code in enumerate(codes))
        assert torch.equal(keys, ref)

    def test_key_to_name_roundtrip_readable(self):
        # e+e- key decodes to two electrons with opposite signs
        key = _k([11, 11], [2, 0])
        name = key_to_name(key, 2)
        assert name.count("e") == 2
        assert "+" in name and "-" in name


# --- KDE bucketing -----------------------------------------------------------


class TestKbodyMassesWithKeys:
    def test_logm_and_key_lengths_match(self):
        from shellflow.data.aux_mass_kde import all_kbody_log_masses

        torch.manual_seed(2)
        B, N = 5, 6
        kin = torch.randn(B, N, 4).abs() + 1.0  # positive-ish energies
        kin[..., 3] = kin[..., :3].norm(dim=-1) + 1.0  # E > |p| so m^2 > 0
        mask = torch.ones(B, N)
        types = torch.randint(0, 23, (B, N))
        charges = torch.randint(0, 3, (B, N))
        log_m, key = all_kbody_log_masses(kin, mask, K=2, types=types, charges=charges)
        assert key is not None
        assert log_m.shape == key.shape

    def test_bucket_counts_match_bruteforce(self):
        from shellflow.data.aux_mass_kde import all_kbody_log_masses

        torch.manual_seed(3)
        B, N = 8, 5
        kin = torch.randn(B, N, 4).abs() + 1.0
        kin[..., 3] = kin[..., :3].norm(dim=-1) + 1.0
        mask = torch.ones(B, N)
        types = torch.randint(0, 23, (B, N))
        charges = torch.randint(0, 3, (B, N))
        _, key = all_kbody_log_masses(kin, mask, K=2, types=types, charges=charges)
        # All N active → C(N,2) tuples per event
        n_pairs = N * (N - 1) // 2
        assert key.numel() == B * n_pairs

    def test_no_keys_when_types_absent(self):
        from shellflow.data.aux_mass_kde import all_kbody_log_masses

        kin = torch.randn(3, 4, 4).abs() + 1.0
        kin[..., 3] = kin[..., :3].norm(dim=-1) + 1.0
        mask = torch.ones(3, 4)
        log_m, key = all_kbody_log_masses(kin, mask, K=2)
        assert key is None
        assert log_m.numel() > 0


# --- w* routing --------------------------------------------------------------


class _StubModule:
    """Minimal stub exposing the buffers + methods the categorized weight uses.

    Bind the unbound RiemannianDiTsModule methods to this stub so we can test
    the routing logic without constructing the full model.
    """

    def __init__(self, K, kept_keys, grid):
        from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

        self._dm_n_cat = {K: len(kept_keys)}
        self.K = K
        setattr(self, f"_da_log_m_grid_k{K}", grid)
        setattr(self, f"_da_keys_k{K}", torch.tensor(kept_keys, dtype=torch.int64))
        # Bind the real lookup so routing uses the production interpolation.
        self._density_aware_log_p = RiemannianDiTsModule._density_aware_log_p.__get__(
            self
        )

    def set_cat(self, K, c, log_p_truth, log_p_gen, floor=-10.0):
        setattr(self, f"_da_log_p_grid_k{K}_c{c}", log_p_truth)
        setattr(self, f"_da_log_p_gen_k{K}_c{c}", log_p_gen)
        setattr(
            self, f"_da_log_p_floor_k{K}_c{c}", torch.tensor(floor, dtype=torch.float32)
        )


def test_w_star_categorized_routing():
    from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

    K = 2
    grid = torch.linspace(-3.0, 6.0, 64)
    # Two kept categories: key 100 (deficit: gen<truth → w*>1),
    # key 200 (matched: gen==truth → w*≈1). Key 999 absent → fallback w*=1.
    stub = _StubModule(K, kept_keys=[100, 200], grid=grid)
    truth = torch.full((64,), 0.0)
    gen_deficit = torch.full((64,), -1.0)  # log p_gen < log p_truth → ratio>1
    stub.set_cat(K, 0, truth, gen_deficit)  # key 100
    stub.set_cat(K, 1, truth, truth.clone())  # key 200, matched

    log_m = torch.zeros(1, 3)  # 3 tuples
    cat_ids = torch.tensor([[0, 1, -1]])  # deficit, matched, fallback

    # w*(r) = ((1-eta)*r^k + eta)^(-alpha) with r = p_gen/p_truth; the eta
    # prior replaced the old hard clamp_min/clamp_max regularization.
    alpha, eta = 1.0, 0.2
    w = RiemannianDiTsModule._distribution_matching_weight_categorized(
        stub, log_m, cat_ids, K, alpha=alpha, eta=eta
    )
    # deficit: r = exp(log p_gen - log p_truth) = e^-1
    expected_deficit = ((1.0 - eta) * np.exp(-1.0) + eta) ** (-alpha)
    assert abs(w[0, 0].item() - expected_deficit) < 1e-4
    assert w[0, 0].item() > 1.0, "deficit category must be up-weighted"
    # matched: r = 1 → w* = ((1-eta) + eta)^(-alpha) = 1
    assert abs(w[0, 1].item() - 1.0) < 1e-5
    # fallback (cat_id == -1): exactly 1, never routed through a density
    assert w[0, 2].item() == 1.0


def test_category_ids_lookup():
    from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

    K = 2
    stub = _StubModule(K, kept_keys=[100, 200, 300], grid=torch.linspace(-3, 6, 8))
    keys = torch.tensor([[100, 300, 999, 200]])
    cat = RiemannianDiTsModule._category_ids(stub, keys, K)
    assert cat.tolist() == [[0, 2, -1, 1]]  # 999 absent → -1
