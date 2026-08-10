"""
Generation quality monitoring callback for Riemannian Flow Matching Transformer.

Generates samples via Euler ODE integration on per-particle manifolds,
converts back to (pT, eta, phi) for histograms, and computes physics metrics
via PhysicsMetricsHelper.
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Subset

from shellflow.models.riemannian_DiTs_geometry import (
    MASSIVE_TYPES,
    MASSLESS_TYPES,
    exp_map_sphere_stable,
    minkowski_to_collator,
    sample_uniform_sphere,
)
from shellflow.training.physics_metrics import PhysicsMetricsHelper
from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)


# W&B grouping bucket per physics metric name (see _record_metric callers).
# Bucket is by physical meaning: 1-particle = single-particle observable,
# 2-particle = 2-object observable (incl. dilepton+MET / lepton+MET), n-particle
# = event-wide aggregate, 4-particle = 4-lepton observable. Unmapped names log
# at the top-level generation/ namespace.
_METRIC_BUCKETS: dict[str, str] = {
    "lead_pT": "1-particle",
    "sublead_pT": "1-particle",
    "ossf_mee": "2-particle",
    "ossf_mmm": "2-particle",
    "ossf_mll": "2-particle",
    "ossf_dR": "2-particle",
    "mT": "2-particle",
    "dphi_ll_met": "2-particle",
    "HT": "n-particle",
    "higgs_m4l_noq": "4-particle",
    "higgs_m4l_q": "4-particle",
}


class RiemannianDiTsSamplePlotCallback(Callback):
    """End-of-epoch callback: RFM Euler ODE generation + comparison histograms.

    Generates events via Euler ODE integration on per-particle manifolds
    and compares with validation data distributions.
    """

    def __init__(
        self,
        val_dataset,
        collator,
        num_samples: int = 10_000,
        batch_size: int = 64,
        n_gen_steps: int = 50,
        seed: int = 42,
        every_n_epochs: int = 5,
        eta: float = 0.0,  # kept for config compat, unused
        # ── Step-based gen-KDE refresh for the DM weight w* ───────────
        # gen_kde_dataset: the TRAIN reference whose (type,charge) templates
        #   p̂_gen is generated from — keeps p̂_gen off the held-out val set (no
        #   train/val leak). Falls back to val_dataset if None (legacy).
        # gen_kde_refresh_steps: refresh p̂_gen every N optimizer steps. None =
        #   legacy epoch-based refresh (every `every_n_epochs`).
        # gen_kde_ema: EMA factor λ for the gen buffer: p̂_gen ← (1-λ)·old + λ·new.
        #   1.0 = wholesale replace (legacy).
        gen_kde_dataset=None,
        gen_kde_refresh_steps: int | None = None,
        gen_kde_ema: float = 0.3,
        # The DM gen-KDE is a SEPARATE generation from the monitoring plots
        # (different purpose: training-loss reweighting vs result-checking), so
        # it gets its own sample count / ODE-step budget. None → fall back to the
        # monitoring values (num_samples / n_gen_steps).
        gen_kde_num_samples: int | None = None,
        gen_kde_n_steps: int | None = None,
        # Dedicated reference for the MONITORING generation (kinematic/resonance
        # plots + metrics), mirroring the higgs 4lep reference. On a union
        # dataset (~⅔ single-lepton) set this to a 2to4lep dataset so the 50k
        # sample is drawn from already-multilepton events (all yield OSSF pairs,
        # baseline-comparable stats) instead of the union (~⅓ do). None → use
        # val_dataset (correct when the training data is itself 2to4lep).
        monitor_dataset=None,
        # ── Higgs end-of-training plot (optional, opt-in) ─────────────
        higgs_enabled: bool = False,
        higgs_dataloader_config: str | None = None,
        higgs_data_dir: str = "data/run2_data_4lep_reduced",
        higgs_raw_data_path: str | None = None,
        higgs_batch_size: int = 16384,
        higgs_shard_dir: str = "/tmp/higgs_shards",
        higgs_every_n_epochs: int | None = None,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.collator = collator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.n_gen_steps = n_gen_steps
        self.seed = seed
        self.every_n_epochs = every_n_epochs
        self.gen_kde_dataset = gen_kde_dataset
        self.gen_kde_refresh_steps = gen_kde_refresh_steps
        self.gen_kde_ema = gen_kde_ema
        self.gen_kde_num_samples = (
            gen_kde_num_samples if gen_kde_num_samples is not None else num_samples
        )
        self.gen_kde_n_steps = (
            gen_kde_n_steps if gen_kde_n_steps is not None else n_gen_steps
        )
        # Monitoring reference: 2to4lep cut applies ONLY when this is set (union
        # training); when None the monitoring uses val_dataset as-is (correct
        # when the training data is already 2to4lep — the cut is skipped).
        self.monitor_dataset = monitor_dataset

        # Higgs end-of-training plot config — runs once after the training
        # loop finishes, on the 4-lepton dataset, to produce the m_4ℓ
        # H→ZZ*→4ℓ plot.
        self.higgs_enabled = higgs_enabled
        self.higgs_dataloader_config = higgs_dataloader_config
        self.higgs_data_dir = higgs_data_dir
        self.higgs_raw_data_path = (
            higgs_raw_data_path
            if higgs_raw_data_path is not None
            else str(Path(higgs_data_dir) / "transformer_input_0000.npy")
        )
        self.higgs_batch_size = higgs_batch_size
        self.higgs_shard_dir = Path(higgs_shard_dir)
        # When None: Higgs runs only at end of training (legacy behavior).
        # When set: Higgs ALSO runs periodically in on_train_epoch_end at
        # `epoch % higgs_every_n_epochs == 0` AND at the last epoch
        # (`is_last`). Setting `higgs_every_n_epochs == training.epochs`
        # collapses to "last epoch only" — matches the legacy behavior.
        self.higgs_every_n_epochs = higgs_every_n_epochs

        # Historical metric scores for physics plots (persists across epochs).
        # Structure: {metric_name: {"epochs": [], "w1": [], "skl": []}}.
        # Persisted across Lightning checkpoint save/load via
        # state_dict/load_state_dict below — without that, resuming a run
        # would restart the W1/SKL history panels from scratch and lose the
        # convergence trajectory from earlier epochs.
        self._physics_metric_history: dict[str, dict] = {}

        self._helper = PhysicsMetricsHelper()

    # ------------------------------------------------------------------
    # Lightning state_dict hooks — persist W1/SKL history across resume
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Saved into Lightning checkpoint under `callbacks/<class_name>`.
        Only rank 0 actually populates this dict (history is rank-0-only),
        but Lightning saves whatever this returns regardless of rank, and
        the rank-0 file is the canonical one written to disk.
        """
        return {
            "physics_metric_history": dict(self._physics_metric_history),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores W1/SKL/BC history on resume so history panels show the
        full trajectory from epoch 0 instead of restarting at the resume
        point. Called on every rank with the rank-0 saved state."""
        history = state_dict.get("physics_metric_history", {})
        if not history:
            return
        # Backward compat: checkpoints saved before BC was added don't
        # have a "bc" key. Pad with NaN so it aligns with epochs and the
        # history panel renders a "BC line starts mid-trace" effect.
        for h in history.values():
            if "bc" not in h:
                h["bc"] = [float("nan")] * len(h.get("epochs", []))
        self._physics_metric_history = history
        n_metrics = len(history)
        n_points = sum(len(h.get("epochs", [])) for h in history.values())
        logger.info(
            f"[RiemannianDiTsPlot] restored W1/SKL/BC history "
            f"({n_metrics} metrics, {n_points} total datapoints) from checkpoint"
        )

    # ------------------------------------------------------------------
    # RFM Generation via Euler ODE integration
    # ------------------------------------------------------------------
    def _init_rfm_prior(self, p_type, p_mask, device, sigma_prior=1.0, chart="onshell"):
        """Initialize RFM generation from t=0 prior on R × S² (× R).

        Unified prior for both massless and massive:
          dim 0: s0 ~ N(0, σ)  (logE or χ)
          dims 1:4: n ~ Uniform(S²)
          dim 4: s4 ~ N(0, σ) for massive, 0 for massless
        """
        B, N = p_type.shape

        is_massless = torch.zeros(B, N, dtype=torch.bool, device=device)
        for tc in MASSLESS_TYPES:
            is_massless = is_massless | (p_type == tc)
        active = p_mask > 0.5

        # Unified prior: s0, n, s4
        s0_init = torch.randn(B, N, device=device) * sigma_prior
        n_init = sample_uniform_sphere(torch.randn(B, N, 3, device=device))
        s4_init = torch.randn(B, N, device=device) * sigma_prior

        state = torch.cat(
            [s0_init.unsqueeze(-1), n_init, s4_init.unsqueeze(-1)], dim=-1
        )

        # Zero dim 4 for massless (on-shell chart only — the free chart has
        # s4 = log|p| live for all types)
        if chart != "free":
            state[..., 4] = torch.where(
                is_massless & active, torch.zeros_like(state[..., 4]), state[..., 4]
            )
        # Zero padding
        state = state * active.unsqueeze(-1).float()

        return state

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def _generate_batch_rfm(self, model, batch, pl_module, n_gen_steps=None):
        """Generate kinematics via Euler ODE integration (RFM velocity mode).

        Integrates t: 0 → 1 using model velocity predictions. `n_gen_steps`
        overrides the default step count (used so the DM gen-KDE can integrate
        with a different fidelity than the monitoring generation).
        """
        steps = n_gen_steps if n_gen_steps is not None else self.n_gen_steps
        device = next(model.parameters()).device
        p_type = batch["particle_type"].to(device)
        p_charge = batch["particle_charge"].to(device)
        p_extra = batch["particle_extra"].to(device)
        p_mask = batch["particle_mask"].to(device)
        met_vector = batch.get("met_vector")
        if met_vector is not None:
            met_vector = met_vector.to(device)

        B, N = p_type.shape

        # Build type masks
        is_massless = torch.zeros(B, N, dtype=torch.bool, device=device)
        for tc in MASSLESS_TYPES:
            is_massless = is_massless | (p_type == tc)
        is_massive = torch.zeros(B, N, dtype=torch.bool, device=device)
        for tc in MASSIVE_TYPES:
            is_massive = is_massive | (p_type == tc)
        active = p_mask > 0.5

        # Initialize at t=0 (prior)
        state = self._init_rfm_prior(
            p_type,
            p_mask,
            device,
            sigma_prior=pl_module.sigma_prior,
            chart=getattr(pl_module, "chart", "onshell"),
        )

        # Guard init NaN
        init_nan = state.isnan().any(dim=-1, keepdim=True) | state.isinf().any(
            dim=-1, keepdim=True
        )
        if init_nan.any():
            state = torch.where(init_nan, torch.zeros_like(state), state)

        # Euler ODE: t = 0 → 1
        dt = 1.0 / steps
        use_r3 = getattr(pl_module, "direction_geometry", "S2") == "R3"

        t_min = getattr(pl_module, "t_min", 0.01)
        t_max = getattr(pl_module, "t_max", 0.99)
        for k in range(steps):
            t_k = k * dt
            # Stay within training range [t_min, t_max]
            t_k_clamped = min(max(t_k, t_min), t_max)
            t_tensor = torch.full((B,), t_k_clamped, device=device)

            # Model predicts velocity
            v_theta = model(
                state,
                p_type,
                p_charge,
                p_extra,
                t_tensor,
                p_mask,
                met_vector=met_vector,
            )

            # Unified Euler step: flat on dim 0, S²/R³ on dims 1:4, flat on dim 4
            s0_new = state[..., 0] + dt * v_theta[..., 0]
            if use_r3:
                n_new = state[..., 1:4] + dt * v_theta[..., 1:4]
            else:
                n_new = exp_map_sphere_stable(state[..., 1:4], dt * v_theta[..., 1:4])
            s4_new = state[..., 4] + dt * v_theta[..., 4]

            new_state = torch.cat(
                [s0_new.unsqueeze(-1), n_new, s4_new.unsqueeze(-1)], dim=-1
            )
            # Keep inactive particles at zero
            state = new_state * active.unsqueeze(-1).float()

        # Decode final state to (px, py, pz, E)
        state = torch.where(state.isnan(), torch.zeros_like(state), state)
        gen_kin = self._decode_rfm_output(state, p_type, p_mask, model)
        return gen_kin.cpu()

    def _decode_rfm_output(self, state, p_type, p_mask, model):
        """Decode RFM final state to collator-order kinematics (legacy chart
        + muon-on-shell).

        Massless: state = (logE_norm, nx, ny, nz, 0)
                  E = E_ref · exp(s_0);
                  |p⃗| = sqrt(E² − m_rest²);  p⃗ = |p⃗| · n̂
                  m_rest = m_μ for muon, 0 for photon/composite (lightlike).
        Massive : state = (χ_norm, nx, ny, nz, logm_norm)
                  χ = s_0·σ_χ + μ_χ;  m = exp(s_4·σ_logm + μ_logm)
                  E = m·cosh χ; |p⃗| = m·sinh χ; p⃗ = |p⃗|·n̂
        """
        type_idx = model.map_type_to_idx(p_type)
        is_massless = model._is_massless_lut[type_idx]
        is_massive = model._is_massive_lut[type_idx]
        active = p_mask > 0.5

        e_ref = model._e_ref_lut[type_idx]
        m_rest = model._m_rest_lut[type_idx]
        chi_mu = model._chi_mu_lut[type_idx]
        chi_sigma = model._chi_sigma_lut[type_idx].clamp(min=0.01)
        logm_mu = model._logm_mu_lut[type_idx]
        logm_sigma = model._logm_sigma_lut[type_idx].clamp(min=0.01)

        n_hat = F.normalize(state[..., 1:4], dim=-1, eps=1e-6)

        if getattr(model, "chart", "onshell") == "free":
            # Free Euclidean chart: E = e_ref·exp(s_0), |p| = e_ref·exp(s_4),
            # independent coordinates — no on-shell tie.
            E_free = e_ref * torch.exp(state[..., 0].clamp(-10, 10))
            p_mag_free = e_ref * torch.exp(state[..., 4].clamp(-10, 10))
            mink_free = torch.cat(
                [E_free.unsqueeze(-1), p_mag_free.unsqueeze(-1) * n_hat], dim=-1
            )
            any_active_f = ((is_massless | is_massive) & active).unsqueeze(-1)
            mink_free = torch.where(
                any_active_f, mink_free, torch.zeros_like(mink_free)
            )
            return minkowski_to_collator(mink_free)

        # Massless (with muon-onshell): E = E_ref · exp(s_0);
        # |p| = sqrt(E² − m_rest²); p⃗ = |p| · n̂. m_rest = 0 (photon/composite)
        # collapses to |p| = E, the lightlike form.
        E_ml = e_ref * torch.exp(state[..., 0].clamp(-10, 10))
        p_mag_ml = (E_ml * E_ml - m_rest * m_rest).clamp(min=0.0).sqrt()
        p3_ml = p_mag_ml.unsqueeze(-1) * n_hat

        # Massive: χ = s_0·σ + μ; m = exp(s_4·σ + μ); E = m·cosh χ; |p| = m·sinh χ
        chi = (state[..., 0] * chi_sigma + chi_mu).clamp(0, 12)
        logm_raw = (state[..., 4] * logm_sigma + logm_mu).clamp(-10, 10)
        m_phys = torch.exp(logm_raw)
        E_mv = m_phys * torch.cosh(chi)
        p_mag_mv = m_phys * torch.sinh(chi)
        p3_mv = p_mag_mv.unsqueeze(-1) * n_hat

        ml = is_massless.unsqueeze(-1)
        E = torch.where(is_massless, E_ml, E_mv)
        p_spatial = torch.where(ml, p3_ml, p3_mv)

        mink = torch.cat([E.unsqueeze(-1), p_spatial], dim=-1)

        # Zero-out inactive particles
        any_active = ((is_massless | is_massive) & active).unsqueeze(-1)
        mink = torch.where(any_active, mink, torch.zeros_like(mink))

        return minkowski_to_collator(mink)

    @torch.no_grad()
    def _generate_paired_samples(self, pl_module, dataset=None):
        """Generate paired validation + generated samples.

        `dataset` overrides the reference (default val_dataset) — used to draw a
        dedicated 2to4lep sample for the OSSF invariant-mass panel only.
        """
        model = getattr(pl_module, "ema_model", None) or pl_module.model
        was_training = model.training
        model.eval()

        gen_fn = self._generate_batch_rfm

        ds = dataset if dataset is not None else self.val_dataset
        rng = np.random.RandomState(self.seed)
        n = min(self.num_samples, len(ds))
        indices = rng.choice(len(ds), n, replace=False)

        loader = DataLoader(
            Subset(ds, indices.tolist()),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        val_kins, gen_kins = [], []
        val_types, val_charges, val_masks = [], [], []
        val_mets = []

        for batch in loader:
            val_kins.append(batch["kinematics"])
            val_types.append(batch["particle_type"])
            val_charges.append(batch["particle_charge"])
            val_masks.append(batch["particle_mask"])
            if "met_vector" in batch:
                val_mets.append(batch["met_vector"])

            gen_raw = gen_fn(model, batch, pl_module)
            gen_kins.append(gen_raw)

        met_cat = torch.cat(val_mets, dim=0) if val_mets else None

        val_data = {
            "kinematics": torch.cat(val_kins, dim=0),
            "particle_type": torch.cat(val_types, dim=0),
            "particle_charge": torch.cat(val_charges, dim=0),
            "particle_mask": torch.cat(val_masks, dim=0),
            "met_vector": met_cat,
        }
        gen_data = {
            "kinematics": torch.cat(gen_kins, dim=0),
            "particle_type": val_data["particle_type"],
            "particle_charge": val_data["particle_charge"],
            "particle_mask": val_data["particle_mask"],
            "met_vector": met_cat,
        }

        if was_training:
            model.train()
        return val_data, gen_data

    # ------------------------------------------------------------------
    # Physics metrics (delegated)
    # ------------------------------------------------------------------
    @staticmethod
    def _cartesian_to_physics(px, py, pz):
        return PhysicsMetricsHelper._cartesian_to_physics(px, py, pz)

    def _extract_lepton_kinematics(self, kin, p_type, p_mask):
        return self._helper._extract_lepton_kinematics(kin, p_type, p_mask)

    def _get_leptons_per_event(self, kin, p_type, p_charge, p_mask):
        return self._helper._get_leptons_per_event(kin, p_type, p_charge, p_mask)

    @staticmethod
    def _find_best_ossf_pair(lep_dict):
        return PhysicsMetricsHelper._find_best_ossf_pair(lep_dict)

    def _compute_sfos_invariant_mass(self, kin, p_type, p_charge, p_mask):
        return self._helper._compute_sfos_invariant_mass(kin, p_type, p_charge, p_mask)

    def _compute_delta_r_ossf(
        self, kin, p_type, p_charge, p_mask, leptons_per_event=None
    ):
        return self._helper._compute_delta_r_ossf(
            kin, p_type, p_charge, p_mask, leptons_per_event
        )

    def _compute_transverse_mass(
        self, kin, p_type, p_charge, p_mask, met_vector, leptons_per_event=None
    ):
        return self._helper._compute_transverse_mass(
            kin, p_type, p_charge, p_mask, met_vector, leptons_per_event
        )

    def _compute_ht(self, kin, p_type, p_mask):
        return self._helper._compute_ht(kin, p_type, p_mask)

    def _compute_delta_phi_dilepton_met(
        self, kin, p_type, p_charge, p_mask, met_vector, leptons_per_event=None
    ):
        return self._helper._compute_delta_phi_dilepton_met(
            kin, p_type, p_charge, p_mask, met_vector, leptons_per_event
        )

    def _compute_leading_subleading_lepton_pt(self, kin, p_type, p_mask):
        return self._helper._compute_leading_subleading_lepton_pt(kin, p_type, p_mask)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    _TYPE_NAMES_PLOT = {
        11: "Electron",
        13: "Muon",
        22: "Photon",
        15: "Tau",
        1: "Jet",
        10: "Large-R Jet",
    }

    _COLOR_TRUTH = "#1f77b4"  # muted blue
    _COLOR_GEN = "#ff7f0e"  # muted orange

    # (name, mass_GeV, color, y_ax) — y_ax is axes-relative [0,1] for label placement
    _RESONANCES = [
        (r"$\omega$", 0.782, "#e07b39", 0.82),
        (r"$\phi$", 1.020, "#d44f4f", 0.72),
        (r"$J/\psi$", 3.097, "#5a8c3e", 0.82),
        (r"$\psi(2S)$", 3.686, "#3e7a8c", 0.72),
        (r"$\Upsilon(1S)$", 9.460, "#7a5fa0", 0.82),
        (r"$\Upsilon(2S)$", 10.023, "#9c7ac8", 0.72),
        (r"$\Upsilon(3S)$", 10.355, "#b89fd6", 0.62),
        (r"$Z$", 91.188, "#c0392b", 0.82),
    ]

    @staticmethod
    def _style_ax(ax, xlabel, ylabel, title, fontsize=13):
        """Apply consistent styling to an axis."""
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.tick_params(axis="both", which="minor", labelsize=9)
        ax.grid(True, which="major", linestyle="-", alpha=0.3, linewidth=0.8)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2, linewidth=0.5)
        ax.minorticks_on()
        ax.legend(fontsize=10)

    # ------------------------------------------------------------------
    # Distributional metrics: W1 and symmetric KL
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_w1(val_arr, gen_arr):
        """Wasserstein-1 distance (sorted approximation)."""
        if len(val_arr) == 0 or len(gen_arr) == 0:
            return float("nan")
        n = min(len(val_arr), len(gen_arr))
        v = np.sort(val_arr[:n])
        g = np.sort(gen_arr[:n])
        return float(np.mean(np.abs(v - g)))

    @staticmethod
    def _compute_symmetric_kl(val_arr, gen_arr, n_bins=100):
        """Symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2."""
        if len(val_arr) == 0 or len(gen_arr) == 0:
            return float("nan")
        lo = min(val_arr.min(), gen_arr.min())
        hi = max(val_arr.max(), gen_arr.max())
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(val_arr, bins=bins, density=True)
        q, _ = np.histogram(gen_arr, bins=bins, density=True)
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        kl_pq = float(np.sum(p * np.log(p / q)))
        kl_qp = float(np.sum(q * np.log(q / p)))
        return (kl_pq + kl_qp) / 2.0

    @staticmethod
    def _compute_bray_curtis(val_arr, gen_arr, n_bins=100):
        """Bray-Curtis dissimilarity on raw count histograms.

            BC = ∑_b |c_gen(b) − c_truth(b)| / (N_gen + N_truth)

        Range [0, 1]. Reduces to Total Variation when N_gen == N_truth.
        Captures count mismatch (which W1 and SKL both normalize away).
        Like SKL, depends on n_bins → callers should match the displayed
        histogram's bin count for "what you see is what's measured."
        """
        if len(val_arr) == 0 and len(gen_arr) == 0:
            return float("nan")
        lo = min(
            val_arr.min() if len(val_arr) else 0,
            gen_arr.min() if len(gen_arr) else 0,
        )
        hi = max(
            val_arr.max() if len(val_arr) else 1,
            gen_arr.max() if len(gen_arr) else 1,
        )
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(val_arr, bins=bins)
        q, _ = np.histogram(gen_arr, bins=bins)
        denom = p.sum() + q.sum()
        if denom == 0:
            return float("nan")
        return float(np.abs(p - q).sum() / denom)

    def _record_metric(
        self, name, epoch, val_arr, gen_arr, log_scale=False, n_bins=100
    ):
        """Compute W1 + SKL + BC and append to history.

        `epoch` is 0-indexed (matches Lightning's `trainer.current_epoch`).
        It is appended verbatim to the history so the W1/SKL/BC panels'
        x-axis ticks match the plot suptitles, output directory names, the
        Higgs file suffixes, and Lightning's auto-logged W&B `epoch`
        scalar — all 0-indexed.

        `n_bins` should match the displayed histogram's bin count so the
        metric measures the same distribution the user sees on the plot.
        Affects SKL and BC; W1 is bin-free.
        """
        if log_scale and len(val_arr) > 0 and len(gen_arr) > 0:
            val_arr = np.log(val_arr[val_arr > 0] + 1e-8)
            gen_arr = np.log(gen_arr[gen_arr > 0] + 1e-8)
        w1 = self._compute_w1(val_arr, gen_arr)
        skl = self._compute_symmetric_kl(val_arr, gen_arr, n_bins=n_bins)
        bc = self._compute_bray_curtis(val_arr, gen_arr, n_bins=n_bins)
        if name not in self._physics_metric_history:
            self._physics_metric_history[name] = {
                "epochs": [],
                "w1": [],
                "skl": [],
                "bc": [],
            }
        h = self._physics_metric_history[name]
        h["epochs"].append(epoch)
        h["w1"].append(w1)
        h["skl"].append(skl)
        h["bc"].append(bc)
        return w1, skl, bc

    _COLOR_W1 = "#2ca02c"  # green   — left axis
    _COLOR_SKL = "#9467bd"  # purple  — right axis (twin 1)
    _COLOR_BC = "#ff7f0e"  # orange  — far-right axis (twin 2, offset)

    def _draw_history_panel_stacked(self, axes, metric_name):
        """Draw W1 + SKL + BC history as 3 stacked sub-panels.

        ``axes`` is a 3-element sequence of axes (top→bottom: W1, SKL, BC).
        Each sub-panel:
          - plots the metric with its color + marker
          - stars the best-epoch point
          - shows last + best in the upper-right legend (monospace)
          - integer-only x-ticks
        Empty history → all 3 axes hidden.
        """
        hist = self._physics_metric_history.get(metric_name)
        if hist is None or len(hist["epochs"]) == 0:
            for a in axes:
                a.set_visible(False)
            return

        epochs = hist["epochs"]
        w1_vals = hist["w1"]
        skl_vals = hist["skl"]
        bc_vals = hist.get("bc", [float("nan")] * len(epochs))

        specs = [
            ("W1", w1_vals, self._COLOR_W1, "o"),
            ("SKL", skl_vals, self._COLOR_SKL, "s"),
            ("BC", bc_vals, self._COLOR_BC, "^"),
        ]
        for ax, (label, vals, color, marker) in zip(axes, specs):
            all_nan = all(np.isnan(v) for v in vals)
            best_idx = int(np.nanargmin(vals)) if not all_nan else -1
            if best_idx >= 0:
                # last value can be NaN even when not-all-NaN (e.g., a
                # mid-history epoch with empty val/gen arrays). Render
                # "—" so the legend doesn't show "nan".
                last_str = f"{vals[-1]:.3f}" if not np.isnan(vals[-1]) else "—"
                legend_label = (
                    f"{label}    last: {last_str}\n"
                    f"          best: {vals[best_idx]:.3f} @ ep {epochs[best_idx]}"
                )
            else:
                legend_label = f"{label}    (no data)"
            ax.plot(
                epochs,
                vals,
                marker + "-",
                color=color,
                markersize=4,
                linewidth=1.4,
                label=legend_label,
            )
            if best_idx >= 0:
                ax.plot(
                    epochs[best_idx],
                    vals[best_idx],
                    "*",
                    color=color,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=5,
                )
            ax.set_ylabel(label, fontsize=10, color=color)
            ax.tick_params(axis="y", labelcolor=color, labelsize=9)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(True, alpha=0.20, linewidth=0.6)
            if len(epochs) > 1:
                ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(
                fontsize=8,
                loc="upper right",
                framealpha=0.85,
                prop={"family": "monospace", "size": 7.5},
            )
        # Hide x-tick labels on the top two sub-panels — only the bottom
        # one keeps the "Epoch" label.
        for a in axes[:-1]:
            a.tick_params(labelbottom=False)
        axes[-1].set_xlabel("Epoch", fontsize=10)

    def _extract_type_kinematics(self, kin, p_type, p_mask, tcode):
        """Extract (pT, eta, phi, E) for a specific particle type."""
        mask = (p_type == tcode) & (p_mask > 0.5)
        px = kin[..., 0][mask].numpy()
        py = kin[..., 1][mask].numpy()
        pz = kin[..., 2][mask].numpy()
        E = kin[..., 3][mask].numpy()
        pt, eta, phi = self._cartesian_to_physics(px, py, pz)
        return pt, eta, phi, E

    def _extract_type_mass(self, kin, p_type, p_mask, tcode):
        """Extract invariant mass for a specific particle type."""
        mask = (p_type == tcode) & (p_mask > 0.5)
        px = kin[..., 0][mask].numpy()
        py = kin[..., 1][mask].numpy()
        pz = kin[..., 2][mask].numpy()
        E = kin[..., 3][mask].numpy()
        m2 = E**2 - px**2 - py**2 - pz**2
        return np.sqrt(np.maximum(m2, 0.0))

    def _compute_sfos_invariant_mass_by_flavor(
        self, kin, p_type, p_charge, p_mask, flavor_code
    ):
        """Compute OSSF invariant mass for a specific lepton flavor."""
        B = kin.shape[0]
        masses = []
        for b in range(B):
            lep_mask = (p_type[b] == flavor_code) & (p_mask[b] > 0.5)
            indices = torch.where(lep_mask)[0]
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    ii, jj = indices[i].item(), indices[j].item()
                    ch_i = p_charge[b, ii].item() - 1
                    ch_j = p_charge[b, jj].item() - 1
                    if ch_i * ch_j >= 0:
                        continue
                    e1, e2 = kin[b, ii, 3].item(), kin[b, jj, 3].item()
                    px1, px2 = kin[b, ii, 0].item(), kin[b, jj, 0].item()
                    py1, py2 = kin[b, ii, 1].item(), kin[b, jj, 1].item()
                    pz1, pz2 = kin[b, ii, 2].item(), kin[b, jj, 2].item()
                    m2 = (
                        (e1 + e2) ** 2
                        - (px1 + px2) ** 2
                        - (py1 + py2) ** 2
                        - (pz1 + pz2) ** 2
                    )
                    masses.append(np.sqrt(max(m2, 0.0)))
        return np.array(masses) if masses else np.array([])

    # ------------------------------------------------------------------
    # Plot 1: 4-kinematics per particle type
    # ------------------------------------------------------------------
    _MASSIVE_TYPES_SET = {15, 1, 10}  # tau, jet, large-R jet

    # Fixed VIEW ranges for the pT / E / mass columns. The bin edges are
    # unchanged (still derived from per-type percentiles below) — only the
    # x-axis xlim is clamped so that leakage bins outside the typical range
    # become visible. Without this, dynamic xlims from the percentile-based
    # bin edges hide any out-of-distribution counts the model produces.
    # Per-type ranges: large-R jet has a much harder pT/E spectrum than
    # leptons / photons / small-R jets / taus, so it gets its own window.
    _XLIM_PT_DEFAULT = (5.0, 500.0)  # e, μ, γ, τ, jet
    _XLIM_PT_LRJ = (100.0, 1000.0)  # large-R jet
    _XLIM_E_DEFAULT = (5.0, 1000.0)  # e, μ, γ, τ, jet
    _XLIM_E_LRJ = (100.0, 2000.0)  # large-R jet
    # Mass: linear x (per spec), log y. Per-type windows match the natural
    # mass scale of each composite type.
    _XLIM_M_PER_TYPE = {
        15: (0.0, 0.3),  # tau (visible-decay)
        1: (0.0, 50.0),  # jet (R=0.4)
        10: (0.0, 400.0),  # large-R jet
    }

    @staticmethod
    def _emphasize_grid(ax):
        """More prominent grid for the compact 4-kinematics panels.

        `_style_ax` already adds a grid, but on this plot the per-panel area
        is small (5 cols × N rows) and the default alpha makes lines barely
        visible. Repeat with stronger alpha/linewidth to match the look of
        the larger physics-metric plot.
        """
        ax.grid(True, which="major", linestyle="-", alpha=0.45, linewidth=0.9)
        ax.grid(True, which="minor", linestyle=":", alpha=0.30, linewidth=0.6)

    def _plot_4kinematics(self, val_data, gen_data, epoch):
        """Create Ntype x 5 grid: pT, eta, phi, E, mass per particle type.

        pT and E use log-log axes. Mass column is blank for massless types.
        """
        matplotlib.use("Agg")
        type_codes = [
            tc
            for tc in self._TYPE_NAMES_PLOT
            if (
                (val_data["particle_type"] == tc) & (val_data["particle_mask"] > 0.5)
            ).any()
        ]
        n_types = len(type_codes)
        if n_types == 0:
            return None

        fig, axes = plt.subplots(n_types, 5, figsize=(28, 4.0 * n_types))
        if n_types == 1:
            axes = axes[np.newaxis, :]

        for row, tcode in enumerate(type_codes):
            name = self._TYPE_NAMES_PLOT[tcode]
            v_pt, v_eta, v_phi, v_E = self._extract_type_kinematics(
                val_data["kinematics"],
                val_data["particle_type"],
                val_data["particle_mask"],
                tcode,
            )
            g_pt, g_eta, g_phi, g_E = self._extract_type_kinematics(
                gen_data["kinematics"],
                gen_data["particle_type"],
                gen_data["particle_mask"],
                tcode,
            )

            pt_hi = np.percentile(v_pt, 99) if len(v_pt) > 0 else 200
            E_hi = np.percentile(v_E, 99) if len(v_E) > 0 else 500

            # pT (log-log) — 61 edges = 60 bins (annotated once in suptitle)
            ax = axes[row, 0]
            bins_pt = np.logspace(
                np.log10(max(v_pt.min(), 0.5) if len(v_pt) > 0 else 0.5),
                np.log10(pt_hi),
                61,
            )
            if len(v_pt) > 0:
                ax.hist(
                    v_pt,
                    bins=bins_pt,
                    label=f"Truth (n={len(v_pt):,})",
                    histtype="step",
                    color=self._COLOR_TRUTH,
                    linewidth=1.8,
                )
            if len(g_pt) > 0:
                ax.hist(
                    g_pt,
                    bins=bins_pt,
                    label=f"Gen (n={len(g_pt):,})",
                    histtype="step",
                    color=self._COLOR_GEN,
                    linewidth=1.8,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            self._style_ax(ax, r"$p_T$ [GeV]", "Events", f"{name} $p_T$")
            ax.set_xlim(*(self._XLIM_PT_LRJ if tcode == 10 else self._XLIM_PT_DEFAULT))
            self._emphasize_grid(ax)

            # eta (linear)
            ax = axes[row, 1]
            bins_eta = np.linspace(-5, 5, 61)
            if len(v_eta) > 0:
                ax.hist(
                    v_eta,
                    bins=bins_eta,
                    label=f"Truth (n={len(v_eta):,})",
                    histtype="step",
                    color=self._COLOR_TRUTH,
                    linewidth=1.8,
                )
            if len(g_eta) > 0:
                ax.hist(
                    g_eta,
                    bins=bins_eta,
                    label=f"Gen (n={len(g_eta):,})",
                    histtype="step",
                    color=self._COLOR_GEN,
                    linewidth=1.8,
                )
            self._style_ax(ax, r"$\eta$", "Events", f"{name} $\\eta$")
            self._emphasize_grid(ax)

            # phi (linear)
            ax = axes[row, 2]
            bins_phi = np.linspace(-np.pi, np.pi, 61)
            if len(v_phi) > 0:
                ax.hist(
                    v_phi,
                    bins=bins_phi,
                    label=f"Truth (n={len(v_phi):,})",
                    histtype="step",
                    color=self._COLOR_TRUTH,
                    linewidth=1.8,
                )
            if len(g_phi) > 0:
                ax.hist(
                    g_phi,
                    bins=bins_phi,
                    label=f"Gen (n={len(g_phi):,})",
                    histtype="step",
                    color=self._COLOR_GEN,
                    linewidth=1.8,
                )
            self._style_ax(ax, r"$\phi$", "Events", f"{name} $\\phi$")
            self._emphasize_grid(ax)

            # E (log-log)
            ax = axes[row, 3]
            bins_E = np.logspace(
                np.log10(max(v_E.min(), 0.5) if len(v_E) > 0 else 0.5),
                np.log10(E_hi),
                61,
            )
            if len(v_E) > 0:
                ax.hist(
                    v_E,
                    bins=bins_E,
                    label=f"Truth (n={len(v_E):,})",
                    histtype="step",
                    color=self._COLOR_TRUTH,
                    linewidth=1.8,
                )
            if len(g_E) > 0:
                ax.hist(
                    g_E,
                    bins=bins_E,
                    label=f"Gen (n={len(g_E):,})",
                    histtype="step",
                    color=self._COLOR_GEN,
                    linewidth=1.8,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            self._style_ax(ax, r"$E$ [GeV]", "Events", f"{name} $E$")
            ax.set_xlim(*(self._XLIM_E_LRJ if tcode == 10 else self._XLIM_E_DEFAULT))
            self._emphasize_grid(ax)

            # Mass (linear x, per-type window, log-y) for massive; blank for massless
            ax = axes[row, 4]
            if tcode in self._MASSIVE_TYPES_SET:
                v_mass = self._extract_type_mass(
                    val_data["kinematics"],
                    val_data["particle_type"],
                    val_data["particle_mask"],
                    tcode,
                )
                g_mass = self._extract_type_mass(
                    gen_data["kinematics"],
                    gen_data["particle_type"],
                    gen_data["particle_mask"],
                    tcode,
                )
                bins_m = None
                if len(v_mass) > 0:
                    m_lo = (
                        max(v_mass[v_mass > 0].min(), 0.1)
                        if (v_mass > 0).any()
                        else 0.1
                    )
                    m_hi = (
                        np.percentile(v_mass[v_mass > 0], 99)
                        if (v_mass > 0).any()
                        else 200
                    )
                    # Degenerate window guard: with few positive-mass entries
                    # (tiny smoke samples, or a broken/off-shell gen variant)
                    # m_hi can collapse onto m_lo — logspace then emits equal
                    # bin edges and np.histogram raises, killing the run from
                    # a monitoring plot. Widen instead.
                    if not np.isfinite(m_hi) or m_hi <= m_lo:
                        m_hi = m_lo * 10.0
                    bins_m = np.logspace(np.log10(m_lo), np.log10(m_hi), 61)
                    ax.hist(
                        v_mass[v_mass > 0],
                        bins=bins_m,
                        label=f"Truth (n={(v_mass > 0).sum():,})",
                        histtype="step",
                        color=self._COLOR_TRUTH,
                        linewidth=1.8,
                    )
                if bins_m is not None and len(g_mass) > 0 and (g_mass > 0).any():
                    ax.hist(
                        g_mass[g_mass > 0],
                        bins=bins_m,
                        label=f"Gen (n={(g_mass > 0).sum():,})",
                        histtype="step",
                        color=self._COLOR_GEN,
                        linewidth=1.8,
                    )
                # x: linear, per-type window (user-requested fixed ranges so
                # leakage bins outside the natural mass scale are visible).
                # y: log (mass distributions span many orders of magnitude).
                ax.set_xscale("linear")
                ax.set_yscale("log")
                self._style_ax(ax, r"$m$ [GeV]", "Events", f"{name} $m$")
                ax.set_xlim(*self._XLIM_M_PER_TYPE.get(tcode, (0.0, 500.0)))
                self._emphasize_grid(ax)
            else:
                ax.set_visible(False)

        # All distributions in this grid use the same 60-bin (61-edge) binning,
        # so annotate the count once in the suptitle instead of per-axis.
        # `epoch` is 0-indexed (matches Lightning's `trainer.current_epoch`).
        fig.suptitle(
            f"Epoch {epoch}: Kinematics per Particle Type "
            f"(N={self.num_samples:,}, 60 bins)",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    # ------------------------------------------------------------------
    # Plot 2: Physics metrics
    # ------------------------------------------------------------------
    # Fixed axes for OSSF mass plots — wider range to include the full
    # mass distribution + all resonance peaks at consistent scale.
    _OSSF_X_LIM = (3e-3, 6e3)  # mass [GeV]
    _OSSF_Y_LIM = (6e-1, 2e4)  # event count

    # Muon-only resonance zoom row (added at the bottom of the physics grid).
    # Per-zoom: bin edges + resonance overlays. Plot design is consistent with
    # the rest of the callback (truth blue / gen orange, log-y, resonance
    # vline with name annotation). No history panel / no W1+SKL recorded.
    # Each `bins` entry: 101 edges → 100 bins (annotated as such in the title).
    _MU_ZOOM_CONFIGS = [
        {
            "title": r"$\omega / \phi$ region",
            "bins": np.linspace(0.5, 2.0, 101),
            "resonances": [
                (r"$\omega$", 0.782, "#e07b39"),
                (r"$\phi$", 1.020, "#d44f4f"),
            ],
        },
        {
            "title": r"$J/\psi$ region",
            "bins": np.linspace(2.0, 4.5, 101),
            "resonances": [
                (r"$J/\psi$", 3.097, "#5a8c3e"),
                (r"$\psi(2S)$", 3.686, "#3e7a8c"),
            ],
        },
        {
            "title": r"$\Upsilon$ region",
            "bins": np.linspace(6.0, 14.0, 101),
            "resonances": [
                (r"$\Upsilon(1S)$", 9.460, "#7a5fa0"),
                (r"$\Upsilon(2S)$", 10.023, "#9c7ac8"),
                (r"$\Upsilon(3S)$", 10.355, "#b89fd6"),
            ],
        },
        {
            "title": r"$Z$ peak",
            "bins": np.linspace(60.0, 120.0, 101),
            "resonances": [
                (r"$m_Z$", 91.188, "#c0392b"),
            ],
        },
    ]

    def _plot_physics_histograms(
        self, val_data, gen_data, epoch, ossf_val_data=None, ossf_gen_data=None
    ):
        """Create physics metrics grid with history panels + μμ resonance zoom row.

        `ossf_val_data`/`ossf_gen_data`, when provided, are a dedicated 2to4lep
        sample used ONLY for the OSSF invariant-mass rows (mee/mmm/mll + the μμ
        resonance zoom) — so those panels get full multilepton statistics on a
        union run. All other panels use the main (full-union) val/gen sample.

        Layout: 7 rows × 12 GridSpec cols (LCM of 3 and 4 column groups).
        - Rows 0-5: existing 3-col grid (each plot spans 4 GridSpec cols).
          - Row 0: OSSF invariant mass histograms (e+e-, mu+mu-, combined),
                   FIXED axes [3e-3, 6e3] × [6e-1, 2e4]
          - Row 1: history panels for row 0
          - Row 2: DeltaR, Transverse Mass, Deltaphi histograms
          - Row 3: history panels for row 2
          - Row 4: HT, Leading pT, Subleading pT histograms
          - Row 5: history panels for row 4
        - Row 6 (NEW): muon-only resonance zoom row (4 plots, each spans
          3 GridSpec cols): ω/φ, J/ψ, Υ, Z. No history / no W1+SKL.
        """
        matplotlib.use("Agg")
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        fig = plt.figure(figsize=(28, 30))
        gs = GridSpec(
            7,
            12,
            figure=fig,
            # History rows now host 3 stacked W1/SKL/BC sub-panels via a
            # nested GridSpec, so they get height_ratios=3 (was 1).
            height_ratios=[4, 3, 4, 3, 4, 3, 4],
            hspace=0.32,
            wspace=0.55,
        )

        def _ax3(row, col):
            """Place a plot in the 3-col grid (each col spans 4 GridSpec cols)."""
            return fig.add_subplot(gs[row, col * 4 : (col + 1) * 4])

        def _hist_axes_3(row, col):
            """3 stacked sub-axes inside one history-row cell (W1/SKL/BC)."""
            outer = gs[row, col * 4 : (col + 1) * 4]
            inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer, hspace=0.10)
            return [fig.add_subplot(inner[i]) for i in range(3)]

        def _ax_zoom(col):
            """Place a plot in the new 4-col zoom row (each spans 3 cols)."""
            return fig.add_subplot(gs[6, col * 3 : (col + 1) * 3])

        kin_v = val_data["kinematics"]
        kin_g = gen_data["kinematics"]
        p_type = val_data["particle_type"]
        p_charge = val_data["particle_charge"]
        p_mask = val_data["particle_mask"]
        met = val_data.get("met_vector")

        val_leptons = self._get_leptons_per_event(kin_v, p_type, p_charge, p_mask)
        gen_leptons = self._get_leptons_per_event(kin_g, p_type, p_charge, p_mask)

        # Dedicated 2to4lep sample for the OSSF invariant-mass panels only
        # (full multilepton stats on a union run); else reuse the main sample.
        if ossf_val_data is not None and ossf_gen_data is not None:
            o_kin_v = ossf_val_data["kinematics"]
            o_kin_g = ossf_gen_data["kinematics"]
            o_type = ossf_val_data["particle_type"]
            o_charge = ossf_val_data["particle_charge"]
            o_mask = ossf_val_data["particle_mask"]
        else:
            o_kin_v, o_kin_g = kin_v, kin_g
            o_type, o_charge, o_mask = p_type, p_charge, p_mask

        def _plot_metric(
            ax,
            val_vals,
            gen_vals,
            bins,
            xlabel,
            title,
            log_y=False,
            log_x=False,
        ):
            if len(val_vals) > 0:
                ax.hist(
                    val_vals,
                    bins=bins,
                    label=f"Truth (n={len(val_vals):,})",
                    histtype="step",
                    color=self._COLOR_TRUTH,
                    linewidth=1.8,
                )
            if len(gen_vals) > 0:
                ax.hist(
                    gen_vals,
                    bins=bins,
                    label=f"Gen (n={len(gen_vals):,})",
                    histtype="step",
                    color=self._COLOR_GEN,
                    linewidth=1.8,
                )
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            # Annotate bin count in the title (matches notebook convention).
            n_bins = len(bins) - 1
            self._style_ax(ax, xlabel, "Events", f"{title}  [{n_bins} bins]")

        def _plot_ossf_mass(ax, val_m, gen_m, title):
            # Fixed wide range so the full mass distribution + all resonances
            # are visible at consistent scale across runs/epochs.
            x_lo, x_hi = self._OSSF_X_LIM
            y_lo, y_hi = self._OSSF_Y_LIM
            # 200 bins (201 edges), matching the analysis-notebook convention
            # (log-spaced broad binning over the full OSSF mass range).
            bins = np.logspace(np.log10(x_lo), np.log10(x_hi), 201)
            val_f = val_m[val_m > x_lo] if len(val_m) > 0 else val_m
            gen_f = gen_m[gen_m > x_lo] if len(gen_m) > 0 else gen_m
            if len(val_f) > 0:
                ax.hist(
                    val_f,
                    bins=bins,
                    histtype="step",
                    linewidth=1.8,
                    color=self._COLOR_TRUTH,
                    label=f"Truth (n={len(val_f):,})",
                )
            if len(gen_f) > 0:
                ax.hist(
                    gen_f,
                    bins=bins,
                    histtype="step",
                    linewidth=1.8,
                    color=self._COLOR_GEN,
                    label=f"Gen (n={len(gen_f):,})",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            for name, mass, color, y_ax in self._RESONANCES:
                ax.axvline(mass, color=color, linewidth=1.0, linestyle="--", alpha=0.8)
                ax.text(
                    mass * 1.04,
                    y_ax,
                    name,
                    color=color,
                    fontsize=7,
                    va="top",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                )
            n_bins = len(bins) - 1
            self._style_ax(
                ax, r"$m_{\ell\ell}$ [GeV]", "Events", f"{title}  [{n_bins} bins]"
            )

        def _plot_mu_zoom(ax, val_mmm_arr, gen_mmm_arr, cfg):
            """Muon-only resonance zoom — design consistent with the rest of
            the callback (truth blue / gen orange, log-y, vline + name label
            for each resonance). No history panel / no W1+SKL recorded."""
            lo, hi = cfg["bins"][0], cfg["bins"][-1]
            val_f = val_mmm_arr[(val_mmm_arr > lo) & (val_mmm_arr < hi)]
            gen_f = gen_mmm_arr[(gen_mmm_arr > lo) & (gen_mmm_arr < hi)]
            if len(val_f) > 0:
                ax.hist(
                    val_f,
                    bins=cfg["bins"],
                    histtype="step",
                    linewidth=1.8,
                    color=self._COLOR_TRUTH,
                    label=f"Truth (n={len(val_f):,})",
                )
            if len(gen_f) > 0:
                ax.hist(
                    gen_f,
                    bins=cfg["bins"],
                    histtype="step",
                    linewidth=1.8,
                    color=self._COLOR_GEN,
                    label=f"Gen (n={len(gen_f):,})",
                )
            for name, mass, color in cfg["resonances"]:
                ax.axvline(
                    mass,
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.9,
                    label=f"{name} = {mass:.3f} GeV",
                )
            ax.set_yscale("log")
            n_bins = len(cfg["bins"]) - 1
            self._style_ax(
                ax,
                r"$m_{\mu\mu}$ [GeV]",
                "Events",
                f"{cfg['title']}  [{n_bins} bins]",
            )

        # Row 0: OSSF invariant mass — uses the dedicated 2to4lep sample (o_*)
        # when provided (union runs), else the main sample. The μμ zoom row
        # below reuses val_mmm/gen_mmm, so it inherits the same cut.
        val_mee = self._compute_sfos_invariant_mass_by_flavor(
            o_kin_v, o_type, o_charge, o_mask, 11
        )
        gen_mee = self._compute_sfos_invariant_mass_by_flavor(
            o_kin_g, o_type, o_charge, o_mask, 11
        )
        _plot_ossf_mass(_ax3(0, 0), val_mee, gen_mee, r"OSSF $m(e^+e^-)$")
        self._record_metric(
            "ossf_mee", epoch, val_mee, gen_mee, log_scale=True, n_bins=200
        )

        val_mmm = self._compute_sfos_invariant_mass_by_flavor(
            o_kin_v, o_type, o_charge, o_mask, 13
        )
        gen_mmm = self._compute_sfos_invariant_mass_by_flavor(
            o_kin_g, o_type, o_charge, o_mask, 13
        )
        _plot_ossf_mass(_ax3(0, 1), val_mmm, gen_mmm, r"OSSF $m(\mu^+\mu^-)$")
        self._record_metric(
            "ossf_mmm", epoch, val_mmm, gen_mmm, log_scale=True, n_bins=200
        )

        val_mll = self._compute_sfos_invariant_mass(o_kin_v, o_type, o_charge, o_mask)
        gen_mll = self._compute_sfos_invariant_mass(o_kin_g, o_type, o_charge, o_mask)
        _plot_ossf_mass(_ax3(0, 2), val_mll, gen_mll, r"OSSF $m(\ell^+\ell^-)$")
        self._record_metric(
            "ossf_mll", epoch, val_mll, gen_mll, log_scale=True, n_bins=200
        )

        # Row 1: history panels for OSSF mass (3 stacked W1/SKL/BC each)
        for col, name in enumerate(["ossf_mee", "ossf_mmm", "ossf_mll"]):
            self._draw_history_panel_stacked(_hist_axes_3(1, col), name)

        # Row 2: DeltaR, Transverse Mass, Deltaphi
        val_dr = self._compute_delta_r_ossf(
            kin_v, p_type, p_charge, p_mask, val_leptons
        )
        gen_dr = self._compute_delta_r_ossf(
            kin_g, p_type, p_charge, p_mask, gen_leptons
        )
        _plot_metric(
            _ax3(2, 0),
            val_dr,
            gen_dr,
            np.linspace(0, 5, 61),
            r"$\Delta R_{\ell\ell}^{\mathrm{OSSF}}$",
            r"OSSF Dilepton $\Delta R$",
            log_y=True,
        )
        self._record_metric(
            "ossf_dR", epoch, val_dr, gen_dr, log_scale=False, n_bins=60
        )

        val_mt = self._compute_transverse_mass(
            kin_v, p_type, p_charge, p_mask, met, val_leptons
        )
        gen_mt = self._compute_transverse_mass(
            kin_g, p_type, p_charge, p_mask, met, gen_leptons
        )
        _plot_metric(
            _ax3(2, 1),
            val_mt,
            gen_mt,
            np.logspace(np.log10(1.0), np.log10(200.0), 61),
            r"$m_T$ [GeV]",
            "Transverse Mass",
            log_x=True,
            log_y=True,
        )
        self._record_metric("mT", epoch, val_mt, gen_mt, log_scale=True, n_bins=60)

        val_dphi = self._compute_delta_phi_dilepton_met(
            kin_v, p_type, p_charge, p_mask, met, val_leptons
        )
        gen_dphi = self._compute_delta_phi_dilepton_met(
            kin_g, p_type, p_charge, p_mask, met, gen_leptons
        )
        _plot_metric(
            _ax3(2, 2),
            val_dphi,
            gen_dphi,
            np.linspace(-np.pi, np.pi, 61),
            r"$\Delta\phi(\ell\ell, E_T^{\mathrm{miss}})$",
            r"Dilepton $\Delta\phi$",
        )
        self._record_metric(
            "dphi_ll_met", epoch, val_dphi, gen_dphi, log_scale=False, n_bins=60
        )

        # Row 3: history panels for row 2 (3 stacked W1/SKL/BC each)
        for col, name in enumerate(["ossf_dR", "mT", "dphi_ll_met"]):
            self._draw_history_panel_stacked(_hist_axes_3(3, col), name)

        # Row 4: HT, Leading pT, Subleading pT
        val_ht = self._compute_ht(kin_v, p_type, p_mask)
        gen_ht = self._compute_ht(kin_g, p_type, p_mask)
        _plot_metric(
            _ax3(4, 0),
            val_ht,
            gen_ht,
            np.logspace(np.log10(10.0), np.log10(1000.0), 61),
            r"$H_T$ [GeV]",
            r"$H_T$ (Scalar Sum $p_T$)",
            log_x=True,
            log_y=True,
        )
        self._record_metric("HT", epoch, val_ht, gen_ht, log_scale=True, n_bins=60)

        val_lead, val_sublead = self._compute_leading_subleading_lepton_pt(
            kin_v, p_type, p_mask
        )
        gen_lead, gen_sublead = self._compute_leading_subleading_lepton_pt(
            kin_g, p_type, p_mask
        )
        _plot_metric(
            _ax3(4, 1),
            val_lead,
            gen_lead,
            np.logspace(np.log10(1.0), np.log10(300.0), 61),
            r"Leading lepton $p_T$ [GeV]",
            r"Leading Lepton $p_T$",
            log_x=True,
            log_y=True,
        )
        self._record_metric(
            "lead_pT", epoch, val_lead, gen_lead, log_scale=True, n_bins=60
        )

        _plot_metric(
            _ax3(4, 2),
            val_sublead,
            gen_sublead,
            np.logspace(np.log10(1.0), np.log10(150.0), 61),
            r"Subleading lepton $p_T$ [GeV]",
            r"Subleading Lepton $p_T$",
            log_x=True,
            log_y=True,
        )
        self._record_metric(
            "sublead_pT", epoch, val_sublead, gen_sublead, log_scale=True, n_bins=60
        )

        # Row 5: history panels for row 4 (3 stacked W1/SKL/BC each)
        for col, name in enumerate(["HT", "lead_pT", "sublead_pT"]):
            self._draw_history_panel_stacked(_hist_axes_3(5, col), name)

        # Row 6 (NEW): muon-only resonance zoom row — ω/φ, J/ψ, Υ, Z.
        # Reuses val_mmm/gen_mmm computed for Row 0 (no extra OSSF pass).
        # No history panel / no W1+SKL recorded — these are visual-only zooms.
        for col, cfg in enumerate(self._MU_ZOOM_CONFIGS):
            _plot_mu_zoom(_ax_zoom(col), val_mmm, gen_mmm, cfg)

        # `epoch` is 0-indexed (matches Lightning's `trainer.current_epoch`).
        fig.suptitle(
            f"Epoch {epoch}: Physics Metrics (N={self.num_samples:,})",
            fontsize=16,
        )
        return fig

    # ------------------------------------------------------------------
    # Gen-KDE buffer update (for distribution_matching weight w*)
    # ------------------------------------------------------------------
    def _generate_shard_for_kde(self, pl_module, rank, world_size):
        """Generate this rank's contiguous shard of the gen-KDE reference
        dataset (gen_kde_dataset, falling back to val_dataset) for the
        gen-KDE update. Returns the decoded (B, N, 4) kinematics tensor in
        collator order (px, py, pz, E), the per-particle active mask, and the
        conditioning (particle_type, particle_charge) used to generate it.

        Type/charge are generation *inputs* (not generated), so they're carried
        through unchanged — needed to assign each gen tuple its category for
        the category-conditioned gen KDE.

        All ranks call this in lockstep — each rank produces a disjoint
        slice of the same deterministically-shuffled index list. With
        4 ranks and gen_kde_num_samples=50000, each rank generates ~12500
        events; combined across ranks via all_gather later, the gen KDE
        sees the full 50k.
        """
        from torch.utils.data import Subset

        model = getattr(pl_module, "ema_model", None) or pl_module.model
        was_training = model.training
        model.eval()

        # Generate p̂_gen from the TRAIN reference (no train/val leak — w* feeds
        # the training loss). Fixed seed → fixed reference set across refreshes.
        gen_ds = (
            self.gen_kde_dataset
            if self.gen_kde_dataset is not None
            else self.val_dataset
        )
        rng = np.random.RandomState(self.seed)
        n_total = min(self.gen_kde_num_samples, len(gen_ds))
        all_indices = rng.choice(len(gen_ds), n_total, replace=False)

        shard_size = (n_total + world_size - 1) // world_size
        start = rank * shard_size
        end = min(start + shard_size, n_total)
        empty = (
            torch.zeros(0, 1, 4),
            torch.zeros(0, 1),
            torch.zeros(0, 1, dtype=torch.long),
            torch.zeros(0, 1, dtype=torch.long),
        )
        if start >= n_total:
            # More ranks than samples (edge case); this rank generates nothing
            if was_training:
                model.train()
            return empty

        rank_indices = all_indices[start:end]
        loader = DataLoader(
            Subset(gen_ds, rank_indices.tolist()),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        gen_kins, masks, types, charges = [], [], [], []
        try:
            for batch in loader:
                gen_raw = self._generate_batch_rfm(
                    model, batch, pl_module, n_gen_steps=self.gen_kde_n_steps
                )
                gen_kins.append(gen_raw.cpu())
                masks.append(batch["particle_mask"].cpu())
                types.append(batch["particle_type"].cpu())
                charges.append(batch["particle_charge"].cpu())
        except Exception as e:
            # A generation failure on ONE rank (e.g. transient CUDA error) must
            # NOT hang the other 7 on the downstream all_gather. Degrade to an
            # empty shard — this rank simply contributes nothing to this refresh.
            logger.warning(
                f"[gen-KDE rank {rank}] generation failed, contributing empty "
                f"shard this refresh: {e}"
            )
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return empty
        finally:
            if was_training:
                model.train()

        if not gen_kins:
            return empty
        return (
            torch.cat(gen_kins, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(types, dim=0),
            torch.cat(charges, dim=0),
        )

    def _get_gloo_group(self):
        """Lazily create and cache a CPU (gloo) process group for the gen-KDE
        gather. All ranks enter the refresh in lockstep (synced global_step), so
        new_group() is called collectively. Keeping the gather on gloo/CPU is
        what makes it immune to CUDA-graph memory-pool aliasing under
        max-autotune (see _update_gen_kde_buffers). Created once, reused."""
        from datetime import timedelta

        import torch.distributed as dist

        if getattr(self, "_gloo_pg", None) is None:
            # gloo does NOT inherit the DDPStrategy NCCL timeout, so set a long
            # one explicitly — otherwise this collective can hit gloo's short
            # default while a peer is still generating its shard.
            self._gloo_pg = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
        return self._gloo_pg

    def _update_gen_kde_buffers(self, trainer, pl_module) -> None:
        """Rebuild per-category p̂_gen on the truth-side log m grid (DDP-safe).

        All ranks generate a disjoint shard of val_dataset, compute per-K
        (log m_K, category key) locally, then `dist.all_gather` BOTH streams
        so every rank sees the combined sample. Each rank then buckets by the
        truth-defined categories (`_da_keys_k{K}`) and builds the same per-
        category KDE deterministically, writing to its local
        `_da_log_p_gen_k{K}_c{c}` buffer — no `dist.broadcast` needed.

        Two gathers per K (log m + key) with the SAME `sizes` exchange keep
        the collective sequence symmetric across ranks (no deadlock risk).

        Buffer state is part of the Lightning module's state_dict, so it
        is also persisted in checkpoints and restored on resume.
        """
        import torch.distributed as dist

        from shellflow.data.aux_mass_kde import fit_kde_grid
        from shellflow.data.tuple_categories import tuple_keys

        rank = trainer.global_rank if trainer is not None else 0
        world_size = (
            trainer.world_size if trainer is not None and trainer.world_size else 1
        )
        ddp_initialized = dist.is_available() and dist.is_initialized()

        # Resolve effective KDE settings — prefer aux_head's distribution_matching
        # block (it owns the canonical settings when both losses are enabled).
        if getattr(pl_module, "_aux_head_needs_kde", False):
            bandwidth = pl_module.aux_head_dm_bandwidth
            grid_size = pl_module.aux_head_dm_grid_size
            grid_lo = pl_module.aux_head_dm_grid_lo
            grid_hi = pl_module.aux_head_dm_grid_hi
            max_kde_samples = pl_module.aux_head_dm_max_kde_samples
            min_samples = pl_module.aux_head_dm_category_min_samples
            K_max = pl_module.aux_head_k_max
        elif getattr(pl_module, "_aux_loss_needs_kde", False):
            bandwidth = pl_module.aux_loss_dm_bandwidth
            grid_size = pl_module.aux_loss_dm_grid_size
            grid_lo = pl_module.aux_loss_dm_grid_lo
            grid_hi = pl_module.aux_loss_dm_grid_hi
            max_kde_samples = pl_module.aux_loss_dm_max_kde_samples
            min_samples = pl_module.aux_loss_dm_category_min_samples
            K_max = pl_module.aux_loss_k_max
        else:
            return  # nothing to update

        # ── All ranks generate their shard (kin + conditioning type/charge) ──
        logger.info(
            f"[gen-KDE rank {rank}/{world_size}] generating shard of "
            f"{self.num_samples // max(world_size, 1):,} samples"
        )
        gen_kin, gen_mask, gen_type, gen_charge = self._generate_shard_for_kde(
            pl_module, rank, world_size
        )
        px = gen_kin[..., 0]
        py = gen_kin[..., 1]
        pz = gen_kin[..., 2]
        E = gen_kin[..., 3]
        active = gen_mask > 0.5

        # ── Per K: compute local (log m_K, key), all-gather both, bucket, KDE ──
        for K in range(2, K_max + 1):
            if K == 2:
                combs = torch.stack(
                    [pl_module._pair_i.cpu(), pl_module._pair_j.cpu()],
                    dim=-1,
                )  # (C_K, K)
            else:
                combs = getattr(pl_module, f"_comb_{K}").cpu()

            if gen_kin.shape[0] > 0:
                pair_mask = active[:, combs].all(dim=-1)  # (B, C_K)
                E_sum = E[:, combs].sum(dim=-1)
                px_sum = px[:, combs].sum(dim=-1)
                py_sum = py[:, combs].sum(dim=-1)
                pz_sum = pz[:, combs].sum(dim=-1)
                m2 = E_sum.pow(2) - px_sum.pow(2) - py_sum.pow(2) - pz_sum.pow(2)
                valid = pair_mask & (m2 > 1e-4)
                if valid.any():
                    key_full = tuple_keys(
                        gen_type[:, combs], gen_charge[:, combs], K
                    )  # (B, C_K)
                    log_m_local = 0.5 * torch.log(m2[valid].clamp(min=1e-4))
                    key_local = key_full[valid].to(torch.int64)
                else:
                    log_m_local = torch.zeros(0)
                    key_local = torch.zeros(0, dtype=torch.int64)
            else:
                log_m_local = torch.zeros(0)
                key_local = torch.zeros(0, dtype=torch.int64)

            # Gather the per-K samples on a CPU (gloo) collective. These buffers
            # MUST stay off the GPU: with max-autotune CUDA graphs, GPU tensors
            # freshly allocated in this callback can land in the graph's static
            # memory pool and be clobbered by float32 generation data — the int64
            # size buffer was overwritten (max_n ~ 4.6e18 -> torch.zeros overflow
            # -> rank crash -> NCCL watchdog hang). CPU/gloo allocations cannot
            # alias the CUDA-graph pool, so this is immune by construction. Uses
            # all_gather_object (variable sizes natively, no max_n to corrupt).
            # The gen samples are already on CPU and the payload is tiny; this
            # fires only on the ~13-min refresh, so there is no per-step impact.
            if ddp_initialized:
                gloo = self._get_gloo_group()
                local_obj = (
                    log_m_local.detach().cpu().numpy().astype(np.float32),
                    key_local.detach().cpu().numpy().astype(np.int64),
                )
                gathered: list = [None] * world_size
                dist.all_gather_object(gathered, local_obj, group=gloo)
                m_parts = [g[0] for g in gathered if g is not None and g[0].size]
                k_parts = [g[1] for g in gathered if g is not None and g[1].size]
                log_m_all = (
                    torch.from_numpy(np.concatenate(m_parts)).float()
                    if m_parts
                    else torch.zeros(0)
                )
                key_all = (
                    torch.from_numpy(np.concatenate(k_parts)).to(torch.int64)
                    if k_parts
                    else torch.zeros(0, dtype=torch.int64)
                )
            else:
                log_m_all = log_m_local
                key_all = key_local

            if log_m_all.numel() == 0:
                logger.warning(
                    f"[gen-KDE] no valid K={K} tuples in combined batch; "
                    "skipping update for this K"
                )
                continue

            # Bucket by the truth-defined categories and rebuild each gen KDE.
            kept_keys = getattr(pl_module, f"_da_keys_k{K}").cpu()  # (n_cat,)
            for c in range(int(kept_keys.numel())):
                samples = log_m_all[key_all == int(kept_keys[c])]
                if samples.numel() < min_samples:
                    continue  # leave gen buffer as-is (truth clone → w*≈1)
                samples_np = samples.detach().numpy()
                # A category whose generated log-masses are (near-)constant gives
                # gaussian_kde a singular covariance → LinAlgError/ValueError. All
                # ranks see the same gathered samples, so they'd all raise — but an
                # uncaught raise still tears down the collective. Skip such a
                # category (keep its current buffer → w*≈1) instead of crashing.
                if float(samples_np.std()) < 1e-6:
                    continue
                try:
                    res = fit_kde_grid(
                        samples_np,
                        bandwidth=bandwidth,
                        grid_size=grid_size,
                        grid_lo=grid_lo,
                        grid_hi=grid_hi,
                        max_kde_samples=max_kde_samples,
                    )
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning(
                        f"[gen-KDE] KDE fit failed for K={K} cat {c} "
                        f"({samples.numel()} samples): {e}; keeping current buffer"
                    )
                    continue
                if res is None:
                    continue
                buf = getattr(pl_module, f"_da_log_p_gen_k{K}_c{c}")
                # Compute the EMA blend on CPU and write the result straight into
                # the persistent GPU buffer (buf.copy_ is a direct H2D write, not
                # a fresh aliasable allocation). This keeps the whole post-
                # generation path off the CUDA-graph memory pool — see the gather
                # note above. Math is identical to the on-GPU logaddexp.
                new_logp = res["log_p_grid"].to(dtype=buf.dtype).cpu()
                lam = self.gen_kde_ema
                if lam is not None and lam < 1.0:
                    # EMA in PROBABILITY space: p ← (1-λ)·p_old + λ·p_new, then
                    # back to log via logaddexp. Buffers start as the truth clone
                    # (w*≈1), so the weight eases in over the first refreshes
                    # instead of snapping — and per-refresh jumps are smoothed.
                    blended = torch.logaddexp(
                        buf.detach().cpu() + math.log(1.0 - lam),
                        new_logp + math.log(lam),
                    )
                    buf.copy_(blended)
                else:
                    buf.copy_(new_logp)  # wholesale replace (legacy)

        # Release the generation intermediates and reclaim the CUDA reserved
        # pool. The sibling generation paths (monitoring in on_train_epoch_end,
        # higgs in _run_higgs_generation) already do this; the gen-KDE refresh
        # fires ~10x more often (every
        # ~13 min), so WITHOUT this the allocator's reserved memory ratchets up
        # every refresh until the host OOM-killer fires ~2.5h in. Cadence-only
        # cost (every ~13 min) — zero per-step training impact.
        del gen_kin, gen_mask, gen_type, gen_charge, px, py, pz, E, active
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Lightning hook
    # ------------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Step-based gen-KDE refresh for the DM weight w*.

        Fires every `gen_kde_refresh_steps` optimizer steps (when set), so p̂_gen
        tracks the model near-continuously instead of going stale for whole
        epochs. `trainer.global_step` is synced across ranks, so all ranks enter
        `_update_gen_kde_buffers` in lockstep (its all_gather is collective-safe).
        """
        if self.gen_kde_refresh_steps is None:
            return
        needs_kde = getattr(pl_module, "use_aux_head_distribution_matching", False)
        if not needs_kde:
            return
        step = trainer.global_step
        if step > 0 and step % self.gen_kde_refresh_steps == 0:
            self._update_gen_kde_buffers(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Convention: 0-indexed epoch everywhere — matches Lightning's
        # `trainer.current_epoch`, the auto-logged W&B `epoch` scalar, and
        # the checkpoint filename `epoch=NN.ckpt`. No `+1` conversions
        # downstream; the previous dual-convention setup was the source of
        # multiple off-by-one bugs.
        epoch = trainer.current_epoch
        is_last = epoch == trainer.max_epochs - 1

        # Periodic gate: fire at multiples of N (so 0, N, 2N, ...) plus the
        # final epoch as a boundary guard. The first epoch (epoch == 0)
        # collapses naturally into the modulo since 0 % N == 0. With N=5
        # and epochs=100 we get 0, 5, 10, 15, ..., 95, 99 (21 plots).
        # With N=10 we get 0, 10, 20, ..., 90, 99 (11 plots).
        def _should_fire(every_n: int) -> bool:
            return is_last or epoch % every_n == 0

        # Higgs end-of-training plot (periodic mode) — runs on ALL ranks
        # (multi-rank generation; _run_higgs_generation manages rank-0 plotting
        # internally). Must be BEFORE the rank-0 guard below.
        if (
            self.higgs_enabled
            and self.higgs_every_n_epochs is not None
            and _should_fire(self.higgs_every_n_epochs)
        ):
            self._run_higgs_generation(trainer, pl_module, epoch=epoch)

        # Distribution-matching weight w* needs an up-to-date p̂_gen KDE,
        # refreshed from generated samples. All ranks generate disjoint
        # shards of the gen-KDE reference dataset (gen_kde_dataset, else
        # val_dataset), all_gather the per-K log m_K samples, then
        # each rank builds the same KDE deterministically from the combined
        # data. No `dist.broadcast(src=0)` — that pattern previously
        # deadlocked because rank 0's serial generation outran the NCCL
        # collective timeout while non-zero ranks idled on the broadcast.
        # Multi-rank — must be BEFORE the rank-0 guard so all ranks call
        # the all_gather collectives in lockstep.
        # gen-KDE refresh: epoch-based ONLY when step-based is disabled
        # (gen_kde_refresh_steps is None). Otherwise on_train_batch_end owns it.
        needs_kde_update = self.gen_kde_refresh_steps is None and getattr(
            pl_module, "use_aux_head_distribution_matching", False
        )
        if needs_kde_update and _should_fire(self.every_n_epochs):
            self._update_gen_kde_buffers(trainer, pl_module)

        # Rank-0 guard for the standard kinematic plots: under DDP, all ranks
        # fire on_train_epoch_end. Without this guard, all ranks would (a)
        # duplicate the 50K-sample generation and (b) race on W&B uploads at
        # the same trainer.global_step, causing plots to be silently dropped.
        if trainer.global_rank != 0:
            return

        if not _should_fire(self.every_n_epochs):
            return

        logger.info(
            f"[RiemannianDiTsPlot] Generating {self.num_samples:,} samples "
            f"at epoch {epoch}..."
        )

        # The gen-KDE update (multi-rank, fired before this guard when
        # distribution_matching is on) does its own shard-based generation,
        # so the plot's 50k-sample generation here is a separate pass.
        val_data, gen_data = self._generate_paired_samples(pl_module)
        fig1 = self._plot_4kinematics(val_data, gen_data, epoch)
        # On a union run, draw a dedicated 2to4lep sample for the OSSF
        # invariant-mass panels only (full multilepton stats); other panels use
        # the full-union sample above. Skipped when monitor_dataset is None
        # (training data already 2to4lep).
        ossf_val_data = ossf_gen_data = None
        if self.monitor_dataset is not None:
            ossf_val_data, ossf_gen_data = self._generate_paired_samples(
                pl_module, dataset=self.monitor_dataset
            )
        fig2 = self._plot_physics_histograms(
            val_data,
            gen_data,
            epoch,
            ossf_val_data=ossf_val_data,
            ossf_gen_data=ossf_gen_data,
        )

        if trainer.logger is not None:
            try:
                import wandb

                # Don't write `epoch` here — Lightning's WandbLogger already
                # auto-logs `epoch` (0-indexed `trainer.current_epoch`) at
                # every step. Now that we use 0-indexed everywhere, both
                # values agree and the chart is monotonic.
                log_dict = {}
                if fig1 is not None:
                    log_dict["generation/4-kinematics"] = wandb.Image(fig1)
                log_dict["generation/physics_metrics"] = wandb.Image(fig2)
                # Log W1/BC scalars for each physics metric
                for mname, mhist in self._physics_metric_history.items():
                    if mhist["epochs"]:
                        bucket = _METRIC_BUCKETS.get(mname, "")
                        prefix = f"generation/{bucket}/" if bucket else "generation/"
                        log_dict[f"{prefix}{mname}_w1"] = mhist["w1"][-1]
                        log_dict[f"{prefix}{mname}_bc"] = mhist["bc"][-1]
                # commit=True: flush the buffer immediately so the plot is
                # written to W&B at this step rather than deferred to next
                # commit (which historically interacted badly with multi-rank
                # logging and tempfile cleanup).
                trainer.logger.experiment.log(
                    log_dict,
                    step=trainer.global_step,
                    commit=True,
                )
                logger.info(
                    f"[RiemannianDiTsPlot] Logged plots to W&B for epoch {epoch}"
                )
            except Exception as e:
                logger.warning(f"[RiemannianDiTsPlot] W&B logging failed: {e}")

        if fig1 is not None:
            plt.close(fig1)
        plt.close(fig2)
        del val_data, gen_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Higgs end-of-training m_4ℓ plot (multi-rank, opt-in)
    # ------------------------------------------------------------------
    # Each rank generates a contiguous shard of the 4-lepton dataset on its
    # own GPU, dumps its shard tensors to a per-rank .pt file, then rank 0
    # reads all shards and plots.
    # Multi-rank speedup; no torch.distributed gather complexity.
    HIGGS_MASS = 125.1
    HIGGS_Z_MASS = 91.1876

    def _higgs_compute_quality_mask(self, n_events: int, device) -> torch.Tensor:
        """Lepton quality cut (isolation + d0 significance) on the raw .npy."""
        data = np.load(self.higgs_raw_data_path)
        if data.shape[0] != n_events:
            logger.warning(
                f"[higgs] raw data has {data.shape[0]} events, "
                f"dataset has {n_events}; mask may be misaligned"
            )
        vals_t = torch.from_numpy(data[:, 0, :]).to(device)
        idx = data[:, 1, :]
        pidx_t = torch.from_numpy((idx // 1000).astype(np.int32)).to(device)
        fidx_t = torch.from_numpy((idx % 1000).astype(np.int32)).to(device)
        N, S = vals_t.shape
        is_lepton = (fidx_t == 1) & ((vals_t == 11) | (vals_t == 13))
        nlep = is_lepton.sum(dim=1)
        ev4_idx = torch.where(nlep == 4)[0]
        n4 = len(ev4_idx)
        if n4 == 0:
            return torch.zeros(N, dtype=torch.bool)
        sv, sp, sf = vals_t[ev4_idx], pidx_t[ev4_idx], fidx_t[ev4_idx]
        is_lep = (sf == 1) & ((sv == 11) | (sv == 13))
        row_i, col_i = torch.where(is_lep)
        lep_pids = sp[row_i, col_i].reshape(n4, 4)
        lep_types = sv[row_i, col_i].reshape(n4, 4).int()
        is_lep_particle = (
            (sp == lep_pids[:, 0:1])
            | (sp == lep_pids[:, 1:2])
            | (sp == lep_pids[:, 2:3])
            | (sp == lep_pids[:, 3:4])
        )
        lep_slot = torch.full((n4, S), -1, dtype=torch.int32, device=device)
        for k in range(4):
            lep_slot = torch.where(sp == lep_pids[:, k : k + 1], k, lep_slot)
        pt_4 = torch.zeros(n4, 4, device=device)
        ptvarcone_4 = torch.zeros(n4, 4, device=device)
        topoetcone_4 = torch.zeros(n4, 4, device=device)
        d0sig_4 = torch.zeros(n4, 4, device=device)
        for fid, arr in [
            (2, pt_4),
            (39, ptvarcone_4),
            (40, topoetcone_4),
            (42, d0sig_4),
        ]:
            mask = is_lep_particle & (sf == fid)
            ev_i, tok_i = torch.where(mask)
            slots = lep_slot[ev_i, tok_i]
            arr[ev_i, slots] = sv[ev_i, tok_i]
        iso_track = ptvarcone_4 / pt_4.clamp(min=0.1) < 0.15
        iso_calo = topoetcone_4 / pt_4.clamp(min=0.1) < 0.20
        d0sig_cut = torch.where(
            lep_types == 11,
            torch.abs(d0sig_4) < 5.0,
            torch.abs(d0sig_4) < 3.0,
        )
        all_pass = (iso_track & iso_calo & d0sig_cut).all(dim=1)
        result = torch.zeros(N, dtype=torch.bool, device=device)
        result[ev4_idx] = all_pass
        return result.cpu()

    def _higgs_compute_m4l(
        self,
        kin,
        p_type,
        p_charge,
        p_mask,
        quality_mask=None,
        device="cpu",
    ):
        """ZZ*+pT selection + 4-lepton invariant mass."""
        kin = kin.to(device)
        p_type = p_type.to(device)
        p_charge = p_charge.to(device)
        p_mask = p_mask.to(device)
        N, M, _ = kin.shape
        px, py, pz, E = kin[..., 0], kin[..., 1], kin[..., 2], kin[..., 3]
        pt = torch.sqrt(px**2 + py**2)
        active = p_mask > 0.5
        is_lepton = ((p_type == 11) | (p_type == 13)) & active
        nlep = is_lepton.sum(dim=1)
        has4 = nlep == 4
        lep_priority = is_lepton.float()
        _, sorted_idx = lep_priority.sort(dim=1, descending=True)
        idx4 = sorted_idx[:, :4]
        bi = torch.arange(N, device=device).unsqueeze(1).expand(-1, 4)
        px_4 = px[bi, idx4]
        py_4 = py[bi, idx4]
        pz_4 = pz[bi, idx4]
        E_4 = E[bi, idx4]
        pt_4 = pt[bi, idx4]
        type_4 = p_type[bi, idx4]
        charge_4 = p_charge[bi, idx4].float() - 1.0
        pt_sorted, _ = pt_4.sort(dim=1, descending=True)
        pt_cut = (
            (pt_sorted[:, 0] > 20) & (pt_sorted[:, 1] > 15) & (pt_sorted[:, 2] > 10)
        )
        E_tot = E_4.sum(dim=1)
        m2_4l = (
            E_tot**2
            - px_4.sum(dim=1) ** 2
            - py_4.sum(dim=1) ** 2
            - pz_4.sum(dim=1) ** 2
        )
        m4l = torch.sqrt(torch.clamp(m2_4l, min=0))
        Z = self.HIGGS_Z_MASS
        pairings = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]
        best_valid = torch.zeros(N, dtype=torch.bool, device=device)
        best_mZ1_dist = torch.full((N,), 1e9, device=device)
        for a, b, c, d in pairings:
            ossf_ab = (type_4[:, a] == type_4[:, b]) & (
                charge_4[:, a] * charge_4[:, b] < 0
            )
            ossf_cd = (type_4[:, c] == type_4[:, d]) & (
                charge_4[:, c] * charge_4[:, d] < 0
            )
            both_ossf = ossf_ab & ossf_cd
            if both_ossf.sum() == 0:
                continue

            def pair_m(i, j):
                dE = E_4[:, i] + E_4[:, j]
                dpx = px_4[:, i] + px_4[:, j]
                dpy = py_4[:, i] + py_4[:, j]
                dpz = pz_4[:, i] + pz_4[:, j]
                return torch.sqrt(torch.clamp(dE**2 - dpx**2 - dpy**2 - dpz**2, min=0))

            m_ab = pair_m(a, b)
            m_cd = pair_m(c, d)
            ab_closer = torch.abs(m_ab - Z) < torch.abs(m_cd - Z)
            m_Z1 = torch.where(ab_closer, m_ab, m_cd)
            m_Z2 = torch.where(ab_closer, m_cd, m_ab)
            valid = both_ossf & (m_Z1 > 50) & (m_Z1 < 106) & (m_Z2 > 12) & (m_Z2 < 115)
            z1_dist = torch.abs(m_Z1 - Z)
            better = valid & (z1_dist < best_mZ1_dist)
            best_mZ1_dist = torch.where(better, z1_dist, best_mZ1_dist)
            best_valid = best_valid | valid
        final = has4 & pt_cut & best_valid
        if quality_mask is not None:
            final = final & quality_mask.to(device)
        return m4l[final].cpu().numpy()

    def _higgs_plot_column(
        self,
        axes,
        m4l_truth,
        m4l_gen,
        title_suffix,
        metric_name,
        bins,
        soi_bins,
    ):
        """One column of the Higgs figure: histogram + bkg-subtracted residual
        + 3 stacked W1/SKL/BC history sub-panels.

        ``axes`` is a 5-element sequence: [hist, residual, W1, SKL, BC].
        ``bins`` is the full-range histogram binning (variable range, 2.5 GeV/bin).
        ``soi_bins`` is the 80–170 GeV bin array for the residual sideband fit.
        """
        bin_width = float(bins[1] - bins[0])
        ax = axes[0]
        ax.hist(
            m4l_truth,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=self._COLOR_TRUTH,
            label=f"Truth (n={len(m4l_truth):,})",
        )
        ax.hist(
            m4l_gen,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=self._COLOR_GEN,
            label=f"Gen (n={len(m4l_gen):,})",
        )
        ax.axvline(
            self.HIGGS_MASS,
            color="#c0392b",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax.text(
            self.HIGGS_MASS + 1.5,
            0.85,
            r"$m_H$",
            color="#c0392b",
            fontsize=10,
            va="top",
            rotation=90,
            transform=ax.get_xaxis_transform(),
        )
        ax.set_yscale("log")
        self._style_ax(
            ax,
            r"$m_{4\ell}$ [GeV]",
            f"Events / {bin_width:g} GeV",
            rf"$H \to ZZ^* \to 4\ell$: {title_suffix}",
        )

        # Residual stays in the 80–170 SOI window where the polynomial
        # sideband fit is meaningful.
        ax2 = axes[1]
        bin_centers = 0.5 * (soi_bins[:-1] + soi_bins[1:])
        for masses, color, label in [
            (m4l_truth, self._COLOR_TRUTH, "Truth"),
            (m4l_gen, self._COLOR_GEN, "Gen"),
        ]:
            counts, _ = np.histogram(masses, bins=soi_bins)
            sideband = (bin_centers < 115) | (bin_centers > 135)
            if sideband.sum() < 3 or len(masses) < 10:
                continue
            coeffs = np.polyfit(bin_centers[sideband], counts[sideband], deg=4)
            bkg = np.polyval(coeffs, bin_centers)
            residual = counts - bkg
            errors = np.sqrt(np.maximum(counts, 1))
            ax2.errorbar(
                bin_centers + (0.3 if color == self._COLOR_GEN else -0.3),
                residual,
                yerr=errors,
                fmt="o",
                markersize=3,
                color=color,
                linewidth=1.2,
                label=f"{label} $-$ bkg",
            )
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        ax2.axvline(
            self.HIGGS_MASS,
            color="#c0392b",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax2.axvspan(115, 135, color="#c0392b", alpha=0.07)
        ax2.text(
            self.HIGGS_MASS + 1,
            0.88,
            r"$m_H$",
            color="#c0392b",
            fontsize=10,
            va="top",
            transform=ax2.get_xaxis_transform(),
        )
        ax2.set_xlim(80, 170)
        self._style_ax(
            ax2,
            r"$m_{4\ell}$ [GeV]",
            "Events $-$ Bkg",
            "Background-subtracted (poly fit to sidebands, 80–170 SOI)",
        )

        # 3 stacked W1/SKL/BC history sub-panels under the residual.
        self._draw_history_panel_stacked(axes[2:5], metric_name)

    def _higgs_plot_figure(self, m4l_truth_noq, m4l_gen_noq, m4l_truth_q, m4l_gen_q):
        """5×2 Higgs figure with stacked history sub-panels per column.

        Outer 3×2 grid: histogram / residual / history-block. The
        history-block expands into a nested 3×1 (W1, SKL, BC).
        Bin width preserved at 2.5 GeV; range covers all truth data,
        capped at p99.5 to avoid an empty high-mass tail. Histogram is
        log-y so the bulk and the tail are both visible. Residual stays
        in the 80–170 GeV SOI window where the sideband fit is meaningful.
        """
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        # Bin range — full truth coverage, p99.5 cap, 2.5 GeV/bin.
        # Defensive fallback: if BOTH truth arrays are empty (rare; can
        # happen in test_mode with strict quality cuts on a small sample),
        # fall back to the legacy 80–170 SOI range so .min()/percentile
        # don't crash on an empty array.
        bin_width = 2.5
        all_truth = np.concatenate([m4l_truth_noq, m4l_truth_q])
        if all_truth.size == 0:
            lo, hi = 80.0, 170.0
        else:
            lo = max(40.0, float(np.floor(all_truth.min() / 5) * 5))
            hi = float(np.ceil(np.percentile(all_truth, 99.5) / 10) * 10)
            # Degenerate case: lo >= hi (e.g., all data clustered in one
            # bin). Pad so np.arange returns ≥ 1 bin.
            if hi <= lo:
                hi = lo + bin_width
        bins = np.arange(lo, hi + bin_width, bin_width)
        soi_bins = np.arange(80, 170 + bin_width, bin_width)
        n_bins = len(bins) - 1

        fig = plt.figure(figsize=(22, 16))
        gs_outer = GridSpec(
            3,
            2,
            figure=fig,
            height_ratios=[2.2, 1.1, 2.7],
            hspace=0.42,
            wspace=0.14,
        )

        def _column_axes(c):
            ax_hist = fig.add_subplot(gs_outer[0, c])
            ax_res = fig.add_subplot(gs_outer[1, c])
            gs_inner = GridSpecFromSubplotSpec(
                3, 1, subplot_spec=gs_outer[2, c], hspace=0.12
            )
            ax_hist_panels = [fig.add_subplot(gs_inner[i]) for i in range(3)]
            return [ax_hist, ax_res, *ax_hist_panels]

        self._higgs_plot_column(
            _column_axes(0),
            m4l_truth_noq,
            m4l_gen_noq,
            r"ZZ* + $p_T$ cuts only",
            metric_name="higgs_m4l_noq",
            bins=bins,
            soi_bins=soi_bins,
        )
        self._higgs_plot_column(
            _column_axes(1),
            m4l_truth_q,
            m4l_gen_q,
            r"ZZ* + $p_T$ + quality cuts",
            metric_name="higgs_m4l_q",
            bins=bins,
            soi_bins=soi_bins,
        )

        fig.suptitle(
            f"Higgs Metrics: m_4ℓ ({n_bins} bins, "
            f"{bin_width:g} GeV/bin, range {lo:g}–{hi:g} GeV, log-y)",
            fontsize=14,
            y=0.995,
        )
        return fig

    def _higgs_generate_shard(self, trainer, pl_module, shard_dir):
        """Each rank: build 4-lep dataset, generate its contiguous shard, dump
        to per-rank .pt file inside `shard_dir`. Returns the dataset size for
        rank 0's later quality-mask computation. `shard_dir` is per-call (e.g.,
        `higgs_shard_dir/epoch_010`) so that there's NEVER cross-call shard
        contamination — eliminates the stale-cleanup race that wiped other
        ranks' shards when rank 0 lagged."""
        from shellflow.data.particle_dataset import ParticleDataset
        from shellflow.data.particle_vector_collator import ParticleVectorCollator

        rank = trainer.global_rank
        world_size = max(trainer.world_size, 1)

        # Use the training's dataloader config + override data_dir to 4-lep.
        cfg_path = self.higgs_dataloader_config
        if cfg_path is None:
            raise ValueError(
                "higgs_dataloader_config must be set when higgs_enabled=true"
            )
        dataset = ParticleDataset(
            config_path=cfg_path,
            normalize=True,
            lazy_loading=True,
            data_dir_override=self.higgs_data_dir,
        )
        N_total = len(dataset)
        shard_size = (N_total + world_size - 1) // world_size
        start = rank * shard_size
        end = min(start + shard_size, N_total)
        if start >= N_total:
            # rank has nothing to do (more ranks than shards)
            return N_total

        from torch.utils.data import Subset

        shard = Subset(dataset, list(range(start, end)))
        collator = ParticleVectorCollator(max_particles=10)
        loader = DataLoader(
            shard,
            batch_size=self.higgs_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )

        model = getattr(pl_module, "ema_model", None) or pl_module.model
        was_training = model.training
        model.eval()

        val_kins, gen_kins, p_types, p_charges, p_masks = [], [], [], [], []
        try:
            for i, batch in enumerate(loader):
                gen = self._generate_batch_rfm(model, batch, pl_module)
                val_kins.append(batch["kinematics"].cpu())
                gen_kins.append(gen.cpu())
                p_types.append(batch["particle_type"].cpu())
                p_charges.append(batch["particle_charge"].cpu())
                p_masks.append(batch["particle_mask"].cpu())
                if (i + 1) % 5 == 0:
                    logger.info(
                        f"[higgs rank {rank}] batch {i + 1}/{len(loader)} "
                        f"({(i + 1) * self.higgs_batch_size:,} events)"
                    )
        finally:
            # Restore train mode even on exception (model is shared with the
            # rest of the post-training cleanup).
            if was_training:
                model.train()

        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"shard_{rank:02d}.pt"
        torch.save(
            {
                "val_kin": torch.cat(val_kins, dim=0),
                "gen_kin": torch.cat(gen_kins, dim=0),
                "p_type": torch.cat(p_types, dim=0),
                "p_charge": torch.cat(p_charges, dim=0),
                "p_mask": torch.cat(p_masks, dim=0),
                "shard_indices": (start, end),
            },
            shard_path,
        )
        logger.info(f"[higgs rank {rank}] dumped {end - start:,} events → {shard_path}")
        return N_total

    def on_train_end(self, trainer, pl_module) -> None:
        """End-of-training Higgs plot. Skipped when higgs_every_n_epochs is set
        (in that case, on_train_epoch_end already covered the last epoch via
        its is_last branch — avoids redundant ~10 min generation)."""
        if not self.higgs_enabled:
            return
        if self.higgs_every_n_epochs is not None:
            # Periodic generation handled by on_train_epoch_end's is_last gate
            return
        self._run_higgs_generation(trainer, pl_module, epoch=None)

    def _run_higgs_generation(
        self, trainer, pl_module, epoch: int | None = None
    ) -> None:
        """Multi-rank Higgs m_4ℓ generation + rank-0 plot. Used by both
        on_train_end (legacy mode, epoch=None) and on_train_epoch_end
        (periodic mode, epoch=current epoch number).

        Each call uses a per-invocation shard subdirectory
        (`higgs_shard_dir / "epoch_NNN"` or `.../"final"`). This avoids the
        stale-shard race that bit the prior implementation: under the global
        shared dir, a slow rank 0 could enter `_run_higgs_generation` AFTER
        ranks 1..N-1 had already dumped their shards, run "stale cleanup"
        before barrier 1, and delete those just-written shards. Per-call
        dirs are created fresh, written by each rank, read by rank 0, and
        rmtree'd at end — no possibility of cross-call contamination.
        """
        rank = trainer.global_rank
        world_size = max(trainer.world_size, 1)
        tag = f"epoch {epoch}" if epoch is not None else "end-of-training"
        logger.info(f"[higgs rank {rank}] starting Higgs plot ({tag})")

        # Per-invocation shard directory. `epoch` is 0-indexed
        # (`trainer.current_epoch`); use "final" for the legacy on_train_end
        # mode where there's no epoch in scope.
        shard_dir = self.higgs_shard_dir / (
            f"epoch_{epoch:03d}" if epoch is not None else "final"
        )

        # 1. Each rank generates its shard into the per-call dir. CRITICAL:
        # on exception we must NOT return early — that would leave other
        # ranks hanging at the post-shard barrier. Set N_total = 0 and fall
        # through; rank 0 will see no shards and skip plotting; the final
        # barrier still syncs everyone.
        N_total = 0
        try:
            N_total = self._higgs_generate_shard(trainer, pl_module, shard_dir)
        except Exception as e:
            logger.error(f"[higgs rank {rank}] generation failed: {e}")

        # 2. Synchronize all ranks before rank 0 reads the shard files
        if world_size > 1:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                logger.warning(f"[higgs] post-shard barrier failed: {e}")

        # 3. Rank-0 only: plot + W&B upload. CRITICAL: every code path below
        # must fall through to the final barrier so non-zero ranks don't sit
        # in the next training step's gradient allreduce while rank 0 is busy
        # plotting (NCCL collective timeout = 30 min by default → crash).
        try:
            if rank == 0:
                # 4. Read all shards in order from THIS call's per-epoch dir.
                shards = sorted(shard_dir.glob("shard_*.pt"))
                if len(shards) != world_size:
                    logger.error(
                        f"[higgs] expected {world_size} shard files in "
                        f"{shard_dir}, found {len(shards)} — skipping plot"
                    )
                    shards = []
                if not shards:
                    pass  # logged above; fall through to final barrier
                else:
                    all_data = [torch.load(p, weights_only=False) for p in shards]
                    all_data.sort(key=lambda d: d["shard_indices"][0])
                    val_kin = torch.cat([d["val_kin"] for d in all_data], dim=0)
                    gen_kin = torch.cat([d["gen_kin"] for d in all_data], dim=0)
                    p_type = torch.cat([d["p_type"] for d in all_data], dim=0)
                    p_charge = torch.cat([d["p_charge"] for d in all_data], dim=0)
                    p_mask = torch.cat([d["p_mask"] for d in all_data], dim=0)

                    # Defensive size check — if val_kin doesn't match N_total,
                    # something's wrong (incomplete shards, corrupted save,
                    # etc.). Use the actual concatenated size for the chunk
                    # loop, validate against quality_mask, and abort cleanly
                    # rather than crashing mid-plot with a tensor shape error.
                    N_actual = val_kin.shape[0]
                    if N_actual != N_total:
                        logger.warning(
                            f"[higgs] concatenated shard size {N_actual:,} "
                            f"differs from N_total {N_total:,} (likely "
                            f"incomplete shard set); using {N_actual:,}"
                        )

                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    logger.info(
                        f"[higgs] computing quality mask on {N_actual:,} raw events"
                    )
                    quality_mask = self._higgs_compute_quality_mask(N_actual, device)

                    if quality_mask.shape[0] != N_actual:
                        logger.error(
                            f"[higgs] quality_mask size {quality_mask.shape[0]:,} "
                            f"!= concatenated shard size {N_actual:,} — "
                            f"skipping plot"
                        )
                        # Fall through to final barrier without plotting.
                        raise RuntimeError("quality_mask size mismatch")

                    # 5. Compute m_4ℓ for each (cut, source) combination, in chunks
                    CHUNK = 100_000
                    truth_noq, gen_noq, truth_q, gen_q = [], [], [], []
                    for s in range(0, N_actual, CHUNK):
                        e = min(s + CHUNK, N_actual)
                        vk = val_kin[s:e]
                        gk = gen_kin[s:e]
                        pt_ = p_type[s:e]
                        pc = p_charge[s:e]
                        pm = p_mask[s:e]
                        qm = quality_mask[s:e]
                        truth_noq.append(
                            self._higgs_compute_m4l(vk, pt_, pc, pm, device=device)
                        )
                        gen_noq.append(
                            self._higgs_compute_m4l(gk, pt_, pc, pm, device=device)
                        )
                        truth_q.append(
                            self._higgs_compute_m4l(vk, pt_, pc, pm, qm, device)
                        )
                        gen_q.append(
                            self._higgs_compute_m4l(gk, pt_, pc, pm, qm, device)
                        )
                    m4l_truth_noq = np.concatenate(truth_noq)
                    m4l_gen_noq = np.concatenate(gen_noq)
                    m4l_truth_q = np.concatenate(truth_q)
                    m4l_gen_q = np.concatenate(gen_q)
                    logger.info(
                        f"[higgs] no-quality: truth={len(m4l_truth_noq):,}, "
                        f"gen={len(m4l_gen_noq):,}; "
                        f"quality: truth={len(m4l_truth_q):,}, "
                        f"gen={len(m4l_gen_q):,}"
                    )

                    # 6. Record W1 + SKL into history. `epoch` is 0-indexed
                    # when called from on_train_epoch_end; for the legacy
                    # on_train_end mode (epoch=None) fall back to
                    # `trainer.current_epoch`, which is also 0-indexed.
                    epoch_for_history = (
                        epoch if epoch is not None else trainer.current_epoch
                    )
                    self._record_metric(
                        "higgs_m4l_noq",
                        epoch_for_history,
                        m4l_truth_noq,
                        m4l_gen_noq,
                        log_scale=False,
                        n_bins=36,
                    )
                    self._record_metric(
                        "higgs_m4l_q",
                        epoch_for_history,
                        m4l_truth_q,
                        m4l_gen_q,
                        log_scale=False,
                        n_bins=36,
                    )

                    # 7. Plot + log
                    fig = self._higgs_plot_figure(
                        m4l_truth_noq, m4l_gen_noq, m4l_truth_q, m4l_gen_q
                    )
                    if trainer.logger is not None:
                        try:
                            import wandb

                            trainer.logger.experiment.log(
                                {"generation/higgs_metrics": wandb.Image(fig)},
                                commit=True,
                            )
                            logger.info("[higgs] logged to W&B")
                        except Exception as e:
                            logger.warning(f"[higgs] W&B logging failed: {e}")
                    plt.close(fig)

        except Exception as e:
            logger.error(f"[higgs rank {rank}] plot/log block failed: {e}")
        finally:
            # Cleanup the per-call shard directory entirely. With per-call
            # subdirs there's no cross-call race; rmtree here regardless of
            # success or failure so /tmp doesn't accumulate stale shards
            # (each shard set is ~700 MB on the d=192 4-lep config).
            if rank == 0:
                import shutil

                try:
                    if shard_dir.exists():
                        shutil.rmtree(shard_dir)
                except OSError as e:
                    logger.warning(f"[higgs] failed to cleanup {shard_dir}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 9. FINAL barrier — guarantees ALL ranks resync before any returns
            # to the training loop. Without this, ranks 1..N-1 race ahead into
            # the next training step's gradient allreduce while rank 0 is still
            # plotting, hitting the NCCL collective timeout (30 min default).
            if world_size > 1:
                try:
                    import torch.distributed as dist

                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                except Exception as e:
                    logger.warning(f"[higgs] final barrier failed: {e}")
