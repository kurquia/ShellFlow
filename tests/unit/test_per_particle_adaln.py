"""Unit tests for per-particle adaLN-Zero conditioning.

Validates:
1. Block-identity-at-init: with flag=True, each transformer block acts as the
   identity function at init (because adaln_linear is zero-init, regardless of
   cond_proj being standard-init). This is the actual load-bearing adaLN-Zero
   property, NOT extras-invariance — cond_proj deliberately uses standard init
   so extras-aware gradient reaches adaln_linear from step 0.
2. Parameter count delta matches expected additions (cond_proj weights/biases and
   widened identity_film[0] columns).
3. All four forward variants (padded, geometric, flat, njt) run end-to-end with
   the flag enabled.

Run with: uv run pytest tests/unit/test_per_particle_adaln.py -v
"""

from __future__ import annotations

import pytest
import torch

from shellflow.models.riemannian_DiTs_model import RiemannianDiTsModel

# --- Fixtures --------------------------------------------------------------


def _build_model(per_particle_adaln: bool, d_model: int = 64) -> RiemannianDiTsModel:
    torch.manual_seed(42)
    return RiemannianDiTsModel(
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_met_tokens=2,
        num_extra_features=13,
        per_particle_adaln=per_particle_adaln,
    )


def _make_padded_batch(B: int = 2, N: int = 6, seed: int = 0):
    """Create a padded (B, N, *) batch of synthetic inputs."""
    torch.manual_seed(seed)
    # z_t: (B, N, 5) — keep S² dim normalized
    z_t = torch.randn(B, N, 5)
    z_t[..., 1:4] = torch.nn.functional.normalize(z_t[..., 1:4], dim=-1)

    # Realistic-ish type codes: lepton=11/13 and a few others.
    # First N//2 particles are "active" (non-zero type), rest are padding (type=0).
    p_type = torch.zeros(B, N, dtype=torch.long)
    for b in range(B):
        p_type[b, 0] = 11  # electron-
        p_type[b, 1] = 11  # electron+
        p_type[b, 2] = 13  # muon-
        p_type[b, 3] = 13  # muon+
        # positions 4..N are padding (type=0)

    p_charge = torch.zeros(B, N, dtype=torch.long)
    p_charge[:, 0] = 0  # -1
    p_charge[:, 1] = 2  # +1
    p_charge[:, 2] = 0  # -1
    p_charge[:, 3] = 2  # +1

    p_extra = torch.randn(B, N, 13)
    # Make padding cleanly zero
    mask = torch.zeros(B, N)
    mask[:, :4] = 1.0
    p_extra = p_extra * mask.unsqueeze(-1)

    t = torch.rand(B) * 0.98 + 0.01

    met = torch.randn(B, 2)

    return {
        "z_t": z_t,
        "particle_type": p_type,
        "particle_charge": p_charge,
        "particle_extra": p_extra,
        "t": t,
        "particle_mask": mask,
        "met_vector": met,
    }


# --- Tests -----------------------------------------------------------------


class TestPerParticleAdalnBlockIdentity:
    def test_block_is_identity_at_init(self):
        """With flag=True at init, each transformer block should act as the identity.

        This is the load-bearing adaLN-Zero property: alpha_{1,2} = 0 (and for
        cross-attention, alpha_c = 0) → block residual contribution is zero →
        block_output = block_input. cond_proj being standard-init means cond
        itself is non-zero and extras-dependent, but the zero-init of adaln_linear
        ensures adaln_out = 0 → gamma/beta/alpha = 0 → no effective modulation.

        We verify by taking the embedded input h, running it through all blocks
        manually, and asserting the output equals h.
        """
        model = _build_model(per_particle_adaln=True, d_model=32)
        model.eval()

        batch = _make_padded_batch()

        with torch.no_grad():
            active = batch["particle_mask"] > 0.5
            h, _type_idx, cond = model._embed_input(
                batch["z_t"],
                batch["particle_type"],
                batch["particle_charge"],
                batch["particle_extra"],
                active,
            )
            time_emb = model.time_embed(batch["t"] * 1000.0)

            # MET context (matches forward's flow for num_met_tokens > 0 path)
            B_met = batch["met_vector"].shape[0]
            met_context = model.met_kv_proj(batch["met_vector"]).view(
                B_met, model.num_met_tokens, model.d_model
            )

            h_pre = h.clone()
            for block in model.blocks:
                h = block(
                    h,
                    time_emb,
                    attention_mask=batch["particle_mask"],
                    met_context=met_context,
                    cond=cond,
                )

        # Each block should be identity → h unchanged after all blocks
        max_diff = (h - h_pre).abs().max().item()
        assert max_diff < 1e-5, (
            f"Block output drifted from input at init: max_diff={max_diff} "
            f"(expected <1e-5). adaLN-Zero property violated."
        )

    def test_cond_is_nonzero_and_extras_dependent_at_init(self):
        """At init, cond_proj is standard-init, so cond depends on extras.

        This is the intended behavior — cond being extras-dependent at init means
        SiLU(time_emb + cond) carries extras signal into the gradient to adaln_linear
        from step 0, avoiding the cascaded-zero-init warmup pathology.
        """
        model = _build_model(per_particle_adaln=True, d_model=32)
        model.eval()

        batch = _make_padded_batch()

        with torch.no_grad():
            active = batch["particle_mask"] > 0.5
            _, _, cond_a = model._embed_input(
                batch["z_t"],
                batch["particle_type"],
                batch["particle_charge"],
                batch["particle_extra"],
                active,
            )
            # Different extras → different cond
            extras_b = torch.randn_like(batch["particle_extra"]) * 3.0
            _, _, cond_b = model._embed_input(
                batch["z_t"],
                batch["particle_type"],
                batch["particle_charge"],
                extras_b,
                active,
            )

        # cond should not be trivially zero
        assert cond_a.abs().max().item() > 1e-4, (
            "cond appears zero at init — cond_proj may be accidentally zero-init"
        )
        # cond should depend on extras (active positions only)
        diff = (cond_a - cond_b)[active]
        assert diff.abs().max().item() > 1e-4, (
            "cond does not vary with extras — path is broken"
        )


class TestPerParticleAdalnParameterCount:
    def test_expected_param_delta(self):
        """flag=True should add cond_proj params and widen identity_film[0] columns.

        Expected additions (vs flag=False):
          + cond_proj: identity_dim_new * d_model + d_model
          + identity_film[0] columns: num_extra_features * d_model
          - extra_proj removed: -(num_extra_features * d_model + d_model)
        Net: cond_proj + identity_film[0] widening - extra_proj removal
        """
        d_model = 32
        num_extra = 13
        charge_embed_dim = 8
        # S2: particle type is NOT in the FiLM identity anymore (it enters
        # structurally via the per-type input projection), so identity = charge
        # (+ extras when flag=True).
        identity_dim_old = charge_embed_dim  # 8
        identity_dim_new = identity_dim_old + num_extra  # 21

        model_f = _build_model(per_particle_adaln=False, d_model=d_model)
        model_t = _build_model(per_particle_adaln=True, d_model=d_model)

        pc_f = sum(p.numel() for p in model_f.parameters())
        pc_t = sum(p.numel() for p in model_t.parameters())

        # Deltas
        cond_proj_params = identity_dim_new * d_model + d_model  # weight + bias
        identity_widen = num_extra * d_model  # extra columns of identity_film[0].weight
        extra_proj_removed = num_extra * d_model + d_model

        expected_delta = cond_proj_params + identity_widen - extra_proj_removed
        actual_delta = pc_t - pc_f

        assert actual_delta == expected_delta, (
            f"Parameter delta mismatch: expected {expected_delta}, got {actual_delta}"
        )


class TestPerParticleAdalnForwardVariants:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_all_four_variants_run(self):
        """Smoke test: all four forward variants should run end-to-end with flag=True."""
        device = torch.device("cuda")
        model = _build_model(per_particle_adaln=True, d_model=64).to(device)
        model.eval()

        # Padded batch
        batch = _make_padded_batch(B=2, N=6)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            out_padded = model.forward(**batch)
        assert out_padded.shape == (2, 6, 5)
        assert torch.isfinite(out_padded).all()

        # Geometric / flat / njt all take flat inputs with a "batch" index.
        # Build flat views from the padded batch (active particles only).
        active = batch["particle_mask"] > 0.5
        idx = active.nonzero(as_tuple=False)  # (T, 2) — (event, pos)
        batch_idx = idx[:, 0].long()
        z_t_flat = batch["z_t"][active]
        p_type_flat = batch["particle_type"][active]
        p_charge_flat = batch["particle_charge"][active]
        p_extra_flat = batch["particle_extra"][active]
        ptr = torch.zeros(3, dtype=torch.long, device=device)
        for b in range(2):
            ptr[b + 1] = ptr[b] + active[b].sum()

        with torch.no_grad():
            out_geo = model.forward_geometric(
                z_t_flat,
                p_type_flat,
                p_charge_flat,
                p_extra_flat,
                batch["t"],
                batch_idx,
                ptr,
                met_vector=batch["met_vector"],
            )
        assert out_geo.shape[0] == z_t_flat.shape[0]
        assert torch.isfinite(out_geo).all()

        # forward_flat expects T_total == T_real (only real particles passed)
        with torch.no_grad():
            out_flat = model.forward_flat(
                z_t_flat,
                p_type_flat,
                p_charge_flat,
                p_extra_flat,
                batch["t"],
                batch_idx,
                ptr,
                met_vector=batch["met_vector"],
                T_real=z_t_flat.shape[0],
            )
        assert out_flat.shape == out_geo.shape
        assert torch.isfinite(out_flat).all()

        # forward_njt — uses NestedTensor ops
        with torch.no_grad():
            out_njt = model.forward_njt(
                z_t_flat,
                p_type_flat,
                p_charge_flat,
                p_extra_flat,
                batch["t"],
                batch_idx,
                ptr,
                met_vector=batch["met_vector"],
            )
        assert out_njt.shape == out_geo.shape
        assert torch.isfinite(out_njt).all()
