"""Test per-event vs per-particle loss normalisation.

Builds a dummy flat batch with 4 events having [5, 19, 2, 7] particles.
Verifies that:
  - per-event loss: each event contributes equally (mean within event, then across events)
  - per-particle loss: each particle contributes equally (global mean per type)
  - the two give different results when multiplicities differ
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Type codes (same as in riemannian_DiTs_lightning.py)
# ---------------------------------------------------------------------------
TYPE_CODES = {
    11: "electron",
    13: "muon",
    22: "photon",
    15: "tau",
    1: "jet",
    10: "largeRjet",
}
TC_TENSOR = torch.tensor(list(TYPE_CODES.keys()), dtype=torch.long)
N_TYPES = len(TC_TENSOR)

_HUBER_DELTA = 1.0

# Massless type codes (photon = 22)
_MASSLESS_CODES = {22}


# ---------------------------------------------------------------------------
# Copied loss functions (standalone, no self/class dependency)
# ---------------------------------------------------------------------------


def _per_particle_huber(v_theta, u_t, active, is_massless):
    """Shared Huber computation → per-particle scalar loss."""
    dv_s0 = F.huber_loss(
        v_theta[..., 0], u_t[..., 0], reduction="none", delta=_HUBER_DELTA
    )
    dv_s2 = F.huber_loss(
        v_theta[..., 1:4], u_t[..., 1:4], reduction="none", delta=_HUBER_DELTA
    ).sum(dim=-1)
    dv_s4 = F.huber_loss(
        v_theta[..., 4], u_t[..., 4], reduction="none", delta=_HUBER_DELTA
    )
    dv_s4 = torch.where(is_massless, torch.zeros_like(dv_s4), dv_s4)
    per_particle = torch.where(active, dv_s0 + dv_s2 + dv_s4, torch.zeros_like(dv_s0))
    per_particle = torch.nan_to_num(
        per_particle, nan=0.0, posinf=100.0, neginf=0.0
    ).clamp(max=100.0)
    return per_particle


def compute_loss_per_particle(v_theta, u_t, p_type, active, is_massless):
    """Per-particle normalisation (old _compute_cfm_loss_core for flat tensors)."""
    per_particle = _per_particle_huber(v_theta, u_t, active, is_massless)

    type_oh = ((p_type.unsqueeze(-1) == TC_TENSOR) & active.unsqueeze(-1)).float()
    type_counts = type_oh.sum(dim=0)
    type_sums = (per_particle.unsqueeze(-1) * type_oh).sum(dim=0)
    type_means = type_sums / type_counts.clamp(min=1)
    present = (type_counts > 0.5).float()
    loss = (type_means * present).sum() / present.sum().clamp(min=1)
    return loss, type_means, type_counts


def compute_loss_per_event(v_theta, u_t, p_type, batch_idx, B, active, is_massless):
    """Per-event normalisation (new _compute_cfm_loss_flat)."""
    per_particle = _per_particle_huber(v_theta, u_t, active, is_massless)

    type_match = ((p_type.unsqueeze(-1) == TC_TENSOR) & active.unsqueeze(-1)).float()

    ev_type_sum = torch.zeros(B, N_TYPES, dtype=per_particle.dtype)
    ev_type_cnt = torch.zeros(B, N_TYPES, dtype=per_particle.dtype)
    expand_idx = batch_idx.unsqueeze(-1).expand_as(type_match)
    ev_type_sum.scatter_add_(0, expand_idx, per_particle.unsqueeze(-1) * type_match)
    ev_type_cnt.scatter_add_(0, expand_idx, type_match)

    ev_type_mean = ev_type_sum / ev_type_cnt.clamp(min=1)
    ev_has_type = (ev_type_cnt > 0.5).float()
    type_means = (ev_type_mean * ev_has_type).sum(dim=0) / ev_has_type.sum(dim=0).clamp(
        min=1
    )
    type_counts = ev_type_cnt.sum(dim=0)

    present = (type_counts > 0.5).float()
    loss = (type_means * present).sum() / present.sum().clamp(min=1)
    return loss, type_means, type_counts


# ---------------------------------------------------------------------------
# Build dummy batch: 4 events with [5, 19, 2, 7] particles
# ---------------------------------------------------------------------------


def build_dummy_batch():
    """Create a flat batch with known per-particle losses.

    Event layout (particles per event):
      event 0: 5 particles  — 2 electrons, 3 jets
      event 1: 19 particles — 3 electrons, 1 muon, 15 jets
      event 2: 2 particles  — 2 muons
      event 3: 7 particles  — 1 electron, 2 muons, 4 jets

    Total: 33 particles, all active, none massless (no photons).
    """
    counts = [5, 19, 2, 7]
    T = sum(counts)  # 33
    B = len(counts)

    # Type assignments per event
    types_per_event = [
        [11, 11, 1, 1, 1],  # event 0
        [11, 11, 11, 13] + [1] * 15,  # event 1
        [13, 13],  # event 2
        [11, 13, 13, 1, 1, 1, 1],  # event 3
    ]
    p_type = torch.tensor([t for ev in types_per_event for t in ev], dtype=torch.long)
    batch_idx = torch.tensor(
        [i for i, c in enumerate(counts) for _ in range(c)], dtype=torch.long
    )
    active = torch.ones(T, dtype=torch.bool)
    is_massless = torch.zeros(T, dtype=torch.bool)

    # Use fixed "predictions" and "targets" so per-particle losses are deterministic
    torch.manual_seed(42)
    v_theta = torch.randn(T, 5)
    u_t = torch.randn(T, 5)

    return (
        v_theta,
        u_t,
        p_type,
        batch_idx,
        B,
        active,
        is_massless,
        counts,
        types_per_event,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_per_event_equals_manual():
    """Verify per-event loss matches a manual Python loop over events."""
    (
        v_theta,
        u_t,
        p_type,
        batch_idx,
        B,
        active,
        is_massless,
        counts,
        _types_per_event,
    ) = build_dummy_batch()

    loss_pe, type_means_pe, _ = compute_loss_per_event(
        v_theta, u_t, p_type, batch_idx, B, active, is_massless
    )

    # Manual: compute per-particle losses, then average within each event per type
    pp = _per_particle_huber(v_theta, u_t, active, is_massless)

    # Accumulate per-type: for each type, collect per-event means, then average
    type_code_list = TC_TENSOR.tolist()
    manual_type_means = {}
    for ti, tc in enumerate(type_code_list):
        event_means = []
        offset = 0
        for ev in range(B):
            n = counts[ev]
            ev_pp = pp[offset : offset + n]
            ev_types = p_type[offset : offset + n]
            mask = ev_types == tc
            if mask.any():
                event_means.append(ev_pp[mask].mean().item())
            offset += n
        if event_means:
            manual_type_means[tc] = sum(event_means) / len(event_means)

    # Check type means match
    for ti, tc in enumerate(type_code_list):
        if tc in manual_type_means:
            assert abs(type_means_pe[ti].item() - manual_type_means[tc]) < 1e-5, (
                f"Type {tc}: got {type_means_pe[ti].item():.6f}, expected {manual_type_means[tc]:.6f}"
            )

    # Check final loss = mean of present type means
    manual_loss = sum(manual_type_means.values()) / len(manual_type_means)
    assert abs(loss_pe.item() - manual_loss) < 1e-5, (
        f"Loss: got {loss_pe.item():.6f}, expected {manual_loss:.6f}"
    )

    print(f"  per-event loss = {loss_pe.item():.6f} (matches manual)")


def test_per_particle_equals_manual():
    """Verify per-particle loss matches a simple global-mean-per-type computation."""
    (
        v_theta,
        u_t,
        p_type,
        _batch_idx,
        _B,
        active,
        is_massless,
        _counts,
        _types_per_event,
    ) = build_dummy_batch()

    loss_pp, type_means_pp, _ = compute_loss_per_particle(
        v_theta, u_t, p_type, active, is_massless
    )

    pp = _per_particle_huber(v_theta, u_t, active, is_massless)

    type_code_list = TC_TENSOR.tolist()
    manual_type_means = {}
    for ti, tc in enumerate(type_code_list):
        mask = p_type == tc
        if mask.any():
            manual_type_means[tc] = pp[mask].mean().item()

    for ti, tc in enumerate(type_code_list):
        if tc in manual_type_means:
            assert abs(type_means_pp[ti].item() - manual_type_means[tc]) < 1e-5, (
                f"Type {tc}: got {type_means_pp[ti].item():.6f}, expected {manual_type_means[tc]:.6f}"
            )

    manual_loss = sum(manual_type_means.values()) / len(manual_type_means)
    assert abs(loss_pp.item() - manual_loss) < 1e-5, (
        f"Loss: got {loss_pp.item():.6f}, expected {manual_loss:.6f}"
    )

    print(f"  per-particle loss = {loss_pp.item():.6f} (matches manual)")


def test_losses_differ():
    """Per-event and per-particle losses must differ with unequal multiplicities."""
    (
        v_theta,
        u_t,
        p_type,
        batch_idx,
        B,
        active,
        is_massless,
        _counts,
        _types_per_event,
    ) = build_dummy_batch()

    loss_pe, _type_means_pe, _ = compute_loss_per_event(
        v_theta, u_t, p_type, batch_idx, B, active, is_massless
    )
    loss_pp, _type_means_pp, _ = compute_loss_per_particle(
        v_theta, u_t, p_type, active, is_massless
    )

    assert abs(loss_pe.item() - loss_pp.item()) > 1e-4, (
        f"Losses should differ but got per-event={loss_pe.item():.6f}, per-particle={loss_pp.item():.6f}"
    )

    print(f"  per-event  = {loss_pe.item():.6f}")
    print(f"  per-particle = {loss_pp.item():.6f}")
    print(f"  difference = {abs(loss_pe.item() - loss_pp.item()):.6f}")


def test_equal_multiplicity_same_loss():
    """When all events have the same multiplicity and same type, both losses should agree."""
    T = 12  # 4 events × 3 particles each
    B = 4
    p_type = torch.tensor([1, 1, 1] * 4, dtype=torch.long)  # all jets
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long)
    active = torch.ones(T, dtype=torch.bool)
    is_massless = torch.zeros(T, dtype=torch.bool)

    torch.manual_seed(99)
    v_theta = torch.randn(T, 5)
    u_t = torch.randn(T, 5)

    loss_pe, _, _ = compute_loss_per_event(
        v_theta, u_t, p_type, batch_idx, B, active, is_massless
    )
    loss_pp, _, _ = compute_loss_per_particle(v_theta, u_t, p_type, active, is_massless)

    assert abs(loss_pe.item() - loss_pp.item()) < 1e-6, (
        f"Should match for equal multiplicity: per-event={loss_pe.item():.6f}, per-particle={loss_pp.item():.6f}"
    )

    print(
        f"  equal-mult: per-event={loss_pe.item():.6f}, per-particle={loss_pp.item():.6f} (match ✓)"
    )


if __name__ == "__main__":
    print("test_per_event_equals_manual:")
    test_per_event_equals_manual()
    print("test_per_particle_equals_manual:")
    test_per_particle_equals_manual()
    print("test_losses_differ:")
    test_losses_differ()
    print("test_equal_multiplicity_same_loss:")
    test_equal_multiplicity_same_loss()
    print("\nAll tests passed.")
