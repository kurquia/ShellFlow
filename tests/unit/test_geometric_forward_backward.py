"""End-to-end forward+backward test for geometric mode.

Creates a small RiemannianDiTsModel, runs two separate geometric batches
through the full pipeline (encode → interpolant → forward_geometric → loss → backward),
and verifies gradients flow correctly.

Tests both:
1. Simple MSE loss (smoke test)
2. The actual CFM Huber loss + auxiliary pair losses via _shared_step
"""

import pytest
import torch

from shellflow.data.particle_vector_collator import ParticleVectorCollatorGeometric


def _make_synthetic_events(n_particles_list, seed=0):
    """Build a list of event dicts for the geometric collator.

    Each event has MET (particle 1, type=0) + n real particles.
    Particles include OSSF electron pairs (type=11, opposite charge)
    to trigger the auxiliary pair loss.
    """
    torch.manual_seed(seed)
    events = []
    for n_particles in n_particles_list:
        feats, fids, pids = [], [], []

        # MET (particle_idx=1, type=0)
        for fid, val in [(1, 0.0), (2, 40.0), (4, 0.8)]:
            feats.append(val)
            fids.append(fid)
            pids.append(1)

        # Real particles — ensure at least one OSSF pair (e+e-)
        for p in range(n_particles):
            pid = p + 2
            if p == 0:
                ptype, charge = 11, -1  # electron-
            elif p == 1:
                ptype, charge = 11, 1  # electron+ (OSSF partner)
            elif p == 2:
                ptype, charge = 13, 1  # muon+
            else:
                ptype = [1, 13, 1, 15, 22][p % 5]
                charge = [0, -1, 0, -1, 0][p % 5]

            pt = 10.0 + p * 15.0 + torch.rand(1).item() * 5.0
            eta = -2.0 + p * 0.8 + torch.rand(1).item() * 0.2
            phi = -3.0 + p * 1.2 + torch.rand(1).item() * 0.3
            energy = pt * (1.2 + torch.rand(1).item() * 0.5)

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

        # Padding to fixed length
        pad_len = 80 - len(feats)
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestGeometricForwardBackward:
    def _run_one_step(self, model, collator, events, device):
        """Run one forward+backward step with simple MSE loss."""
        batch = collator(events)

        kin = batch["kinematics"].to(device)
        p_type = batch["particle_type"].to(device)
        p_charge = batch["particle_charge"].to(device)
        p_extra = batch["particle_extra"].to(device)
        batch_idx = batch["batch"].to(device)
        ptr = batch["ptr"].to(device)
        met_vector = batch["met_vector"].to(device)
        B = ptr.shape[0] - 1

        x_data = model.encode_to_manifold(kin, p_type)

        t = 0.01 + 0.98 * torch.rand(B, device=device)
        t_exp = t[batch_idx].unsqueeze(-1)
        x_noise = torch.randn_like(x_data)
        x_noise[..., 1:4] = torch.nn.functional.normalize(x_noise[..., 1:4], dim=-1)
        psi_t = (1 - t_exp) * x_noise + t_exp * x_data
        psi_t = torch.cat(
            [
                psi_t[..., :1],
                torch.nn.functional.normalize(psi_t[..., 1:4], dim=-1),
                psi_t[..., 4:],
            ],
            dim=-1,
        )

        target_v = x_data - x_noise

        v_theta = model.forward_geometric(
            psi_t,
            p_type,
            p_charge,
            p_extra,
            t,
            batch_idx,
            ptr,
            met_vector=met_vector,
        )

        loss = ((v_theta - target_v) ** 2).mean()
        loss.backward()

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        return loss.item(), grad_norm

    def test_forward_backward_two_batches(self):
        """Full forward+backward with MSE loss on two different geometric batches."""
        from shellflow.models.riemannian_DiTs_model import RiemannianDiTsModel

        device = torch.device("cuda")
        torch.manual_seed(42)

        model = RiemannianDiTsModel(
            d_model=64,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_met_tokens=2,
        ).to(device)

        collator = ParticleVectorCollatorGeometric()

        events_1 = _make_synthetic_events([4, 6], seed=100)
        model.zero_grad()
        loss_1, grad_1 = self._run_one_step(model, collator, events_1, device)

        assert torch.isfinite(torch.tensor(loss_1)), f"Loss 1 not finite: {loss_1}"
        assert grad_1 > 0, f"No gradients in batch 1: grad_norm={grad_1}"

        events_2 = _make_synthetic_events([3, 7], seed=200)
        model.zero_grad()
        loss_2, grad_2 = self._run_one_step(model, collator, events_2, device)

        assert torch.isfinite(torch.tensor(loss_2)), f"Loss 2 not finite: {loss_2}"
        assert grad_2 > 0, f"No gradients in batch 2: grad_norm={grad_2}"
        assert loss_1 != loss_2, "Both batches produced identical loss"

        print(
            f"\n  MSE: batch1 loss={loss_1:.4f} grad={grad_1:.4f} | batch2 loss={loss_2:.4f} grad={grad_2:.4f}"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestGeometricSharedStep:
    """Test geometric mode through the actual Lightning _shared_step.

    Uses the real CFM Huber loss, RFM geodesic interpolant, and auxiliary
    paired log-mass loss — the same pipeline used in training.
    """

    def _make_module(self, device):
        """Create a small RiemannianDiTsModule for testing."""
        from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

        model_config = {
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "num_met_tokens": 2,
        }
        optimizer_config = {"lr": 1e-3, "weight_decay": 0.0}
        scheduler_config = {"name": "constant"}
        training_config = {}

        module = RiemannianDiTsModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            training_config=training_config,
            total_steps=100,
        )
        module = module.to(device)
        module.train()
        # Fake global_step and current_epoch for aux loss gating
        module._current_fx_name = "training_step"
        return module

    def _run_shared_step(self, module, collator, events, device):
        """Run _shared_step and backward, return loss, grad norm, and metrics."""
        batch = collator(events)
        # Move batch to device
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v  # Python int (e.g. max_seqlen)

        module.zero_grad()
        # _shared_step returns (loss, metrics); any enabled auxiliary losses
        # are already folded into `loss` internally.
        loss, metrics = module._shared_step(batch_gpu)
        loss.backward()

        grad_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        return loss.item(), grad_norm, metrics

    def test_cfm_huber_loss_geometric(self):
        """CFM Huber loss through _shared_step with geometric batching."""
        device = torch.device("cuda")
        torch.manual_seed(42)

        module = self._make_module(device)
        collator = ParticleVectorCollatorGeometric()

        events_1 = _make_synthetic_events([4, 6], seed=100)
        loss_1, grad_1, metrics_1 = self._run_shared_step(
            module,
            collator,
            events_1,
            device,
        )

        assert torch.isfinite(torch.tensor(loss_1)), f"CFM loss not finite: {loss_1}"
        assert grad_1 > 0, f"No gradients: {grad_1}"
        assert "loss" in metrics_1
        assert "n_active" in metrics_1
        assert metrics_1["batch_size"] == 2

        events_2 = _make_synthetic_events([3, 7], seed=200)
        loss_2, grad_2, _metrics_2 = self._run_shared_step(
            module,
            collator,
            events_2,
            device,
        )

        assert torch.isfinite(torch.tensor(loss_2)), f"CFM loss 2 not finite: {loss_2}"
        assert grad_2 > 0

        print(f"\n  CFM Huber: batch1={loss_1:.4f} | batch2={loss_2:.4f}")
        # Print per-type losses if available
        for k, v in metrics_1.items():
            if k.startswith("loss_"):
                print(f"    {k}={v:.4f}")
