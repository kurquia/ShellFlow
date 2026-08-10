"""Benchmark: padded vs geometric collator + forward pass throughput.

Compares wall-clock time for:
1. Collator only (CPU): batching events into tensors
2. Full forward pass (GPU): collate → encode → interpolant → model forward → loss
"""

import time

import pytest
import torch

from shellflow.data.particle_vector_collator import (
    ParticleVectorCollator,
    ParticleVectorCollatorGeometric,
)


def _make_events(n_events, seed=42):
    """Generate n_events synthetic events with realistic particle counts (4-8)."""
    torch.manual_seed(seed)
    events = []
    for ev in range(n_events):
        n_particles = 4 + (ev % 5)  # cycles through 4,5,6,7,8
        feats, fids, pids = [], [], []

        # MET
        for fid, val in [(1, 0.0), (2, 30.0 + ev * 0.1), (4, 0.5 + ev * 0.01)]:
            feats.append(val)
            fids.append(fid)
            pids.append(1)

        for p in range(n_particles):
            pid = p + 2
            ptype = [11, 13, 1, 11, 13, 1, 15, 22][p % 8]
            charge = [-1, 1, 0, 1, -1, 0, -1, 0][p % 8]
            pt = 10.0 + p * 12.0 + (ev % 7) * 2.0
            eta = -2.0 + p * 0.7
            phi = -3.0 + p * 1.1
            energy = pt * 1.4

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

        pad_len = 60 - len(feats)
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


class TestCollatorBenchmark:
    """Benchmark collator throughput (CPU only, no GPU needed)."""

    @pytest.mark.parametrize("batch_size", [64, 256, 1024])
    def test_collator_speed(self, batch_size):
        events = _make_events(batch_size)
        padded = ParticleVectorCollator(max_particles=30)
        geometric = ParticleVectorCollatorGeometric()

        # Warmup
        padded(events[:16])
        geometric(events[:16])

        n_iters = 5

        # Padded
        t0 = time.perf_counter()
        for _ in range(n_iters):
            padded(events)
        t_padded = (time.perf_counter() - t0) / n_iters

        # Geometric
        t0 = time.perf_counter()
        for _ in range(n_iters):
            geometric(events)
        t_geometric = (time.perf_counter() - t0) / n_iters

        # Report
        geo_result = geometric(events)
        T_total = geo_result["kinematics"].shape[0]
        mean_particles = T_total / batch_size

        print(
            f"\n  B={batch_size}, mean_particles={mean_particles:.1f}:"
            f"\n    Padded:    {t_padded * 1000:.1f} ms"
            f"\n    Geometric: {t_geometric * 1000:.1f} ms"
            f"\n    Ratio:     {t_padded / t_geometric:.2f}x"
            f"\n    Padded tokens:    {batch_size * 30}"
            f"\n    Geometric tokens: {T_total}"
            f"\n    Token ratio:      {batch_size * 30 / T_total:.2f}x"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestForwardBenchmark:
    """Benchmark full forward pass: collate → encode → forward → loss."""

    @pytest.mark.parametrize("batch_size", [64, 256, 1024])
    def test_forward_speed(self, batch_size):
        from shellflow.models.riemannian_DiTs_model import RiemannianDiTsModel

        device = torch.device("cuda")
        torch.manual_seed(42)

        model = RiemannianDiTsModel(
            d_model=128,
            num_heads=4,
            num_layers=6,
            mlp_ratio=4.0,
            dropout=0.0,
            num_met_tokens=4,
        ).to(device)
        model.eval()

        events = _make_events(batch_size)
        padded_col = ParticleVectorCollator(max_particles=30)
        geometric_col = ParticleVectorCollatorGeometric()

        # Pre-collate both
        batch_pad = padded_col(events)
        batch_geo = geometric_col(events)

        # Move to GPU; the geometric batch contains plain-int entries
        # (e.g. max_seqlen), which must not be .to()'d
        bp = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_pad.items()
        }
        bg = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_geo.items()
        }

        B = batch_size
        T_total = bg["kinematics"].shape[0]

        # Warmup (important for CUDA)
        for _ in range(3):
            with torch.no_grad():
                t = 0.5 * torch.ones(B, device=device)
                x_data_pad = model.encode_to_manifold(
                    bp["kinematics"], bp["particle_type"], bp["particle_mask"]
                )
                psi_pad = x_data_pad + 0.1 * torch.randn_like(x_data_pad)
                psi_pad[..., 1:4] = torch.nn.functional.normalize(
                    psi_pad[..., 1:4], dim=-1
                )
                model(
                    psi_pad,
                    bp["particle_type"],
                    bp["particle_charge"],
                    bp["particle_extra"],
                    t,
                    bp["particle_mask"],
                    met_vector=bp["met_vector"],
                )

                x_data_geo = model.encode_to_manifold(
                    bg["kinematics"], bg["particle_type"]
                )
                psi_geo = x_data_geo + 0.1 * torch.randn_like(x_data_geo)
                psi_geo[..., 1:4] = torch.nn.functional.normalize(
                    psi_geo[..., 1:4], dim=-1
                )
                model.forward_geometric(
                    psi_geo,
                    bg["particle_type"],
                    bg["particle_charge"],
                    bg["particle_extra"],
                    t,
                    bg["batch"],
                    bg["ptr"],
                    met_vector=bg["met_vector"],
                    max_seqlen=bg["max_seqlen"],
                )
        torch.cuda.synchronize()

        n_iters = 10

        # Padded forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                t = 0.5 * torch.ones(B, device=device)
                x_data_pad = model.encode_to_manifold(
                    bp["kinematics"], bp["particle_type"], bp["particle_mask"]
                )
                psi_pad = x_data_pad + 0.1 * torch.randn_like(x_data_pad)
                psi_pad[..., 1:4] = torch.nn.functional.normalize(
                    psi_pad[..., 1:4], dim=-1
                )
                v = model(
                    psi_pad,
                    bp["particle_type"],
                    bp["particle_charge"],
                    bp["particle_extra"],
                    t,
                    bp["particle_mask"],
                    met_vector=bp["met_vector"],
                )
                v.pow(2).mean()
        torch.cuda.synchronize()
        t_padded = (time.perf_counter() - t0) / n_iters

        # Geometric forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                t = 0.5 * torch.ones(B, device=device)
                x_data_geo = model.encode_to_manifold(
                    bg["kinematics"], bg["particle_type"]
                )
                psi_geo = x_data_geo + 0.1 * torch.randn_like(x_data_geo)
                psi_geo[..., 1:4] = torch.nn.functional.normalize(
                    psi_geo[..., 1:4], dim=-1
                )
                v = model.forward_geometric(
                    psi_geo,
                    bg["particle_type"],
                    bg["particle_charge"],
                    bg["particle_extra"],
                    t,
                    bg["batch"],
                    bg["ptr"],
                    met_vector=bg["met_vector"],
                    max_seqlen=bg["max_seqlen"],
                )
                v.pow(2).mean()
        torch.cuda.synchronize()
        t_geometric = (time.perf_counter() - t0) / n_iters

        speedup = t_padded / t_geometric if t_geometric > 0 else float("inf")

        print(
            f"\n  B={batch_size} (d=128, L=6, H=4):"
            f"\n    Padded:    {t_padded * 1000:.2f} ms  ({B * 30} tokens)"
            f"\n    Geometric: {t_geometric * 1000:.2f} ms  ({T_total} tokens)"
            f"\n    Speedup:   {speedup:.2f}x"
            f"\n    Token reduction: {B * 30 / T_total:.1f}x"
        )
