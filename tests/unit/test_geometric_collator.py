"""Tests for ParticleVectorCollatorGeometric and geometric attention fallback.

Verifies that the geometric (sparse) collator produces the same per-particle
kinematics, types, charges, and extras as the padded collator, and that the
geometric attention forward path produces valid outputs.
"""

import pytest
import torch

from shellflow.data.particle_vector_collator import (
    ParticleVectorCollator,
    ParticleVectorCollatorGeometric,
    geometric_to_dense,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_event_batch():
    """Create a batch of 4 synthetic events with known particle structure.

    Each event dict has: features (L,), feature_idxs (L,), particle_idxs (L,)
    matching the format produced by ParticleDataset(normalize=False).
    """
    events = []
    # Feature IDs: 1=type, 2=pT, 3=eta, 4=phi, 5=E, 6=charge
    for n_particles in [3, 5, 4, 6]:
        feats, fids, pids = [], [], []

        # MET (particle_idx=1, type=0)
        for fid, val in [(1, 0.0), (2, 50.0), (4, 1.2)]:
            feats.append(val)
            fids.append(fid)
            pids.append(1)

        # Real particles (particle_idx=2..n_particles+1)
        for p in range(n_particles):
            pid = p + 2
            ptype = [11, 13, 1, 11, 13][p % 5]  # electron, muon, jet, ...
            charge = [-1, 1, 0, 1, -1][p % 5]
            pt = 20.0 + p * 10.0
            eta = -1.0 + p * 0.5
            phi = -3.0 + p * 1.0
            energy = pt * 1.5

            for fid, val in [
                (1, float(ptype)),
                (2, pt),
                (3, eta),
                (4, phi),
                (5, energy),
                (6, float(charge)),
            ]:
                feats.append(val)
                fids.append(fid)
                pids.append(pid)

        # Padding
        pad_len = 50 - len(feats)
        feats.extend([0.0] * pad_len)
        fids.extend([0] * pad_len)
        pids.extend([0] * pad_len)

        events.append(
            {
                "features": torch.tensor(feats, dtype=torch.float32),
                "feature_idxs": torch.tensor(fids, dtype=torch.long),
                "particle_idxs": torch.tensor(pids, dtype=torch.long),
            }
        )
    return events


@pytest.fixture
def padded_collator():
    return ParticleVectorCollator(max_particles=30)


@pytest.fixture
def geometric_collator():
    return ParticleVectorCollatorGeometric()


# ============================================================================
# Collator Tests
# ============================================================================


class TestGeometricCollator:
    def test_output_keys(self, geometric_collator, synthetic_event_batch):
        """Geometric collator returns expected keys."""
        result = geometric_collator(synthetic_event_batch)
        expected_keys = {
            "kinematics",
            "particle_type",
            "particle_charge",
            "particle_extra",
            "batch",
            "ptr",
            "met_vector",
            "max_seqlen",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self, geometric_collator, synthetic_event_batch):
        """Verify output tensor shapes are consistent."""
        result = geometric_collator(synthetic_event_batch)
        B = len(synthetic_event_batch)
        T = result["kinematics"].shape[0]

        assert result["kinematics"].shape == (T, 4)
        assert result["particle_type"].shape == (T,)
        assert result["particle_charge"].shape == (T,)
        assert result["particle_extra"].shape == (T, 13)
        assert result["batch"].shape == (T,)
        assert result["ptr"].shape == (B + 1,)
        assert result["met_vector"].shape == (B, 2)

    def test_ptr_consistency(self, geometric_collator, synthetic_event_batch):
        """ptr cumsum matches actual particle counts."""
        result = geometric_collator(synthetic_event_batch)
        T = result["kinematics"].shape[0]

        # ptr starts at 0 and ends at T_total
        assert result["ptr"][0] == 0
        assert result["ptr"][-1] == T
        # ptr is monotonically non-decreasing
        assert (result["ptr"][1:] >= result["ptr"][:-1]).all()
        # ptr is int32
        assert result["ptr"].dtype == torch.int32

    def test_batch_idx_consistency(self, geometric_collator, synthetic_event_batch):
        """batch index matches ptr boundaries."""
        result = geometric_collator(synthetic_event_batch)
        ptr = result["ptr"]
        batch_idx = result["batch"]

        for i in range(len(synthetic_event_batch)):
            start, end = int(ptr[i].item()), int(ptr[i + 1].item())
            assert (batch_idx[start:end] == i).all()

    def test_no_padding_particles(self, geometric_collator, synthetic_event_batch):
        """All particles in geometric output should be real (non-zero kinematics)."""
        result = geometric_collator(synthetic_event_batch)
        kin = result["kinematics"]
        # Every particle should have non-zero energy (last column)
        assert (kin[:, 3].abs() > 0).all(), (
            "Found zero-energy particles in geometric output"
        )

    def test_particle_counts_match(
        self, padded_collator, geometric_collator, synthetic_event_batch
    ):
        """Geometric and padded collators should produce same particle count per event."""
        padded = padded_collator(synthetic_event_batch)
        geometric = geometric_collator(synthetic_event_batch)

        # Count real particles in padded output
        padded_counts = (padded["particle_mask"] > 0.5).sum(dim=1)  # (B,)
        # Count from ptr
        geo_counts = geometric["ptr"][1:] - geometric["ptr"][:-1]

        assert torch.equal(padded_counts.int(), geo_counts)

    def test_kinematics_match(
        self, padded_collator, geometric_collator, synthetic_event_batch
    ):
        """Per-particle kinematics should match between padded and geometric collators."""
        padded = padded_collator(synthetic_event_batch)
        geometric = geometric_collator(synthetic_event_batch)
        ptr = geometric["ptr"]

        for i in range(len(synthetic_event_batch)):
            mask = padded["particle_mask"][i] > 0.5
            padded_kin = padded["kinematics"][i][mask]  # (n_i, 4)

            start, end = int(ptr[i].item()), int(ptr[i + 1].item())
            geo_kin = geometric["kinematics"][start:end]  # (n_i, 4)

            assert padded_kin.shape == geo_kin.shape, f"Event {i}: shape mismatch"
            assert torch.allclose(padded_kin, geo_kin, atol=1e-5), (
                f"Event {i}: kinematics mismatch"
            )

    def test_types_match(
        self, padded_collator, geometric_collator, synthetic_event_batch
    ):
        """Particle types should match between collators."""
        padded = padded_collator(synthetic_event_batch)
        geometric = geometric_collator(synthetic_event_batch)
        ptr = geometric["ptr"]

        for i in range(len(synthetic_event_batch)):
            mask = padded["particle_mask"][i] > 0.5
            padded_types = padded["particle_type"][i][mask]
            start, end = int(ptr[i].item()), int(ptr[i + 1].item())
            geo_types = geometric["particle_type"][start:end]
            assert torch.equal(padded_types, geo_types), f"Event {i}: type mismatch"

    def test_charges_match(
        self, padded_collator, geometric_collator, synthetic_event_batch
    ):
        """Particle charges should match between collators."""
        padded = padded_collator(synthetic_event_batch)
        geometric = geometric_collator(synthetic_event_batch)
        ptr = geometric["ptr"]

        for i in range(len(synthetic_event_batch)):
            mask = padded["particle_mask"][i] > 0.5
            padded_charges = padded["particle_charge"][i][mask]
            start, end = int(ptr[i].item()), int(ptr[i + 1].item())
            geo_charges = geometric["particle_charge"][start:end]
            assert torch.equal(padded_charges, geo_charges), (
                f"Event {i}: charge mismatch"
            )

    def test_met_vector_match(
        self, padded_collator, geometric_collator, synthetic_event_batch
    ):
        """MET vectors should match between collators."""
        padded = padded_collator(synthetic_event_batch)
        geometric = geometric_collator(synthetic_event_batch)
        assert torch.allclose(padded["met_vector"], geometric["met_vector"], atol=1e-5)

    def test_total_particles(self, geometric_collator, synthetic_event_batch):
        """Total particle count should equal sum of real particles across events."""
        result = geometric_collator(synthetic_event_batch)
        # Events have 3, 5, 4, 6 real particles
        expected = 3 + 5 + 4 + 6
        assert result["kinematics"].shape[0] == expected


# ============================================================================
# geometric_to_dense Tests
# ============================================================================


class TestGeometricToDense:
    def test_roundtrip(self, geometric_collator, synthetic_event_batch):
        """geometric → dense → extract should recover original."""
        geo = geometric_collator(synthetic_event_batch)
        dense, mask = geometric_to_dense(geo["kinematics"], geo["ptr"])

        B = geo["ptr"].shape[0] - 1
        assert dense.shape[0] == B
        assert mask.shape[0] == B

        # Verify real particles match
        for i in range(B):
            start = int(geo["ptr"][i].item())
            end = int(geo["ptr"][i + 1].item())
            n = end - start
            assert mask[i, :n].all()
            assert not mask[i, n:].any()
            assert torch.allclose(dense[i, :n], geo["kinematics"][start:end])


# ============================================================================
# Attention Forward Geometric Tests (fallback path)
# ============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen_attn requires CUDA")
class TestAttentionGeometric:
    """Tests for geometric attention paths. Requires CUDA (varlen_attn is GPU-only)."""

    def _device(self):
        return torch.device("cuda")

    def test_self_attention_geometric_output_shape(self):
        """forward_geometric produces correct output shape."""
        from shellflow.models.dit_blocks import MultiHeadAttention

        dev = self._device()
        attn = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0).to(dev)
        T_total = 18
        x = torch.randn(T_total, 32, device=dev)
        ptr = torch.tensor([0, 6, 12, 18], dtype=torch.int32, device=dev)
        out = attn.forward_geometric(x, ptr, max_seqlen=6)
        assert out.shape == (T_total, 32)

    def test_self_attention_geometric_vs_padded(self):
        """Geometric and padded paths should produce same output."""
        from shellflow.models.dit_blocks import MultiHeadAttention

        dev = self._device()
        torch.manual_seed(42)
        attn = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0).to(dev)
        attn.eval()

        # Create a small batch: 2 events, 3 and 5 particles
        x_flat = torch.randn(8, 32, device=dev)
        ptr = torch.tensor([0, 3, 8], dtype=torch.int32, device=dev)

        # Geometric forward
        out_geo = attn.forward_geometric(x_flat, ptr, max_seqlen=5)

        # Padded forward: pad to (2, 5, 32)
        x_pad = torch.zeros(2, 5, 32, device=dev)
        x_pad[0, :3] = x_flat[:3]
        x_pad[1, :5] = x_flat[3:8]
        mask = torch.zeros(2, 5, device=dev)
        mask[0, :3] = 1.0
        mask[1, :5] = 1.0

        out_pad = attn(x_pad, attention_mask=mask)

        # Extract real particles from padded output
        out_pad_flat = torch.cat([out_pad[0, :3], out_pad[1, :5]], dim=0)

        # bf16 cast in varlen path means tolerance needs to be wider
        assert torch.allclose(out_geo, out_pad_flat, atol=5e-3), (
            f"Max diff: {(out_geo - out_pad_flat).abs().max().item()}"
        )

    def test_met_cross_attention_geometric_output_shape(self):
        """MET cross-attention geometric produces correct shape."""
        from shellflow.models.dit_blocks import METCrossAttention

        dev = self._device()
        cross_attn = METCrossAttention(d_model=32, num_heads=4, dropout=0.0).to(dev)
        T_total = 11
        B = 2
        M = 4
        x = torch.randn(T_total, 32, device=dev)
        met = torch.randn(B, M, 32, device=dev)
        ptr = torch.tensor([0, 5, 11], dtype=torch.int32, device=dev)

        out = cross_attn.forward_geometric(x, met, ptr, max_seqlen_q=6)
        assert out.shape == (T_total, 32)

    def test_adaln_block_geometric_output_shape(self):
        """AdaLN block geometric produces correct shape."""
        from shellflow.models.dit_blocks import AdaLNZeroDiTBlock

        dev = self._device()
        block = AdaLNZeroDiTBlock(
            d_model=32,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            met_cross_attention=True,
        ).to(dev)
        T_total = 11
        B = 2
        x = torch.randn(T_total, 32, device=dev)
        time_emb = torch.randn(B, 32, device=dev)
        batch_idx = torch.cat(
            [
                torch.zeros(5, dtype=torch.long, device=dev),
                torch.ones(6, dtype=torch.long, device=dev),
            ]
        )
        ptr = torch.tensor([0, 5, 11], dtype=torch.int32, device=dev)
        met_context = torch.randn(B, 4, 32, device=dev)

        out = block.forward_geometric(
            x,
            time_emb,
            batch_idx,
            ptr,
            max_seqlen=6,
            met_context=met_context,
        )
        assert out.shape == (T_total, 32)
