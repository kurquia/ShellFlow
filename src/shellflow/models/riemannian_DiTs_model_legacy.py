"""Riemannian Diffusion Transformer — PAPER ARCHITECTURE (legacy).

This is the model exactly as used for the released paper checkpoints
(arXiv:2607.16144): particle type enters through a 32-dim type embedding
concatenated into the FiLM/adaLN conditioning, and the manifold input/output
projections are shared per sector (massless/massive) rather than per type.
The successor architecture in riemannian_DiTs_model.py moved type identity
into per-type grouped projections and removed it from the conditioning, so
checkpoints are not interchangeable between the two.

Selected via model_config["architecture"] == "legacy_type_film" (the released
checkpoints carry this flag in their hyperparameters). Do not refactor: state
dict names must keep matching the released checkpoints.

Supports two prediction modes:
  - "x0": VP-scheduled x₀-prediction (original)
  - "velocity": Riemannian Flow Matching velocity prediction (RFM)

Per-particle manifolds:
  - Massless (e, mu, photon): R⁺ × S² (log-energy + unit direction on 2-sphere)
  - Massive (tau, jet, largeRjet): H³_{m̃} × R (per-sample mass hyperboloid + log-mass)

Reuses DiffusionTransformerBlock, MET cross-attention from particle_diffusion.py.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from shellflow.models.dit_blocks import (
    AdaLNZeroDiTBlock,
    RMSNorm,
    SinusoidalPositionEmbedding,
)
from shellflow.models.riemannian_DiTs_geometry import (
    CHI_STATS,
    E_MIN,
    E_REF,
    LOGM_STATS,
    MASS_FLOOR,
    MASSIVE_TYPES,
    MASSLESS_TYPES,
    collator_to_minkowski,
    minkowski_to_collator,
    tangent_project_S2,
)

# Particle type codes (same ordering as ParticleVectorDiffusion)
PARTICLE_TYPE_CODES = [1, 10, 11, 13, 15, 22]
_TYPE_TO_IDX = {t: i for i, t in enumerate(PARTICLE_TYPE_CODES)}
NUM_PARTICLE_TYPES = len(PARTICLE_TYPE_CODES) + 1  # +1 for unknown/padding
NUM_CHARGE_CLASSES = 3  # {-1, 0, +1} stored as {0, 1, 2}


# ---------------------------------------------------------------------------
# Aux-head supporting modules
#
# The hierarchical K-body aux head (see tmp/hierarchical_kbody_head_design.md)
# is purely analytical — no new learned parameters. Pair sums of per-particle
# 4-momenta and their time derivatives are computed once from the existing
# `out_proj_aux` output and reused at K=3 (pair + single) and K=4 (pair + pair).
# Implementation lives entirely in the lightning module's `_compute_aux_head_loss`.
# ---------------------------------------------------------------------------


class QualityScorer(nn.Module):
    """Per-particle detector quality score q_i ∈ [0, 1] from extras.

    Computed via fixed-threshold sigmoids on physically meaningful features
    (lepton isolation + IP significance, jet JVT, photon isolation, tau RNN,
    large-R D2). NOT learnable — q is a multiplicative weight on the aux loss
    and gradient flows through the loss VALUE only, not through q.

    Returns q = 1 for inactive particles and unknown types (e.g. MET) so the
    active-particle mask remains the only exclusion mechanism.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.iso_cut_lep = float(config.get("iso_cut_lep", 0.15))
        self.iso_alpha_lep = float(config.get("iso_alpha_lep", 10.0))
        self.ip_cut_lep = float(config.get("ip_cut_lep", 3.0))
        self.ip_alpha_lep = float(config.get("ip_alpha_lep", 1.0))
        self.iso_cut_photon = float(config.get("iso_cut_photon", 0.15))
        self.iso_alpha_photon = float(config.get("iso_alpha_photon", 10.0))
        self.jvt_cut_jet = float(config.get("jvt_cut_jet", 0.5))
        self.jvt_alpha_jet = float(config.get("jvt_alpha_jet", 5.0))
        self.rnn_cut_tau = float(config.get("rnn_cut_tau", 0.5))
        self.rnn_alpha_tau = float(config.get("rnn_alpha_tau", 5.0))
        self.d2_cut_lrj = float(config.get("d2_cut_lrj", 1.5))
        self.d2_alpha_lrj = float(config.get("d2_alpha_lrj", 5.0))

    @torch.no_grad()
    def forward(
        self,
        p_type: torch.Tensor,
        extras: torch.Tensor,
        p3_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            p_type:  (B, N) — particle type codes
            extras:  (B, N, 13) — per-particle extras from collator
                     (see particle_vector_collator.py:242 for layout)
            p3_true: (B, N, 3) — truth 3-momentum (for relative isolation pT)
            mask:    (B, N) — active-particle mask (bool)
        Returns:
            q: (B, N) — quality scores ∈ [0, 1]; q = 1 for inactive / unknown
        """
        # Truth pT (clamp to 1 GeV to avoid /0 for soft particles).
        pt_true = p3_true[..., :2].pow(2).sum(-1).clamp(min=1.0).sqrt()  # (B, N)

        is_lep = (p_type == 11) | (p_type == 13)
        is_photon = p_type == 22
        is_jet = p_type == 1
        is_tau = p_type == 15
        is_lrj = p_type == 10

        # Lepton: relative isolation = min(ptvarcone30, topoetcone20) / pT;
        # IP significance = |d0sig|.
        iso_lep = torch.minimum(extras[..., 5], extras[..., 6]) / pt_true
        ipsig_lep = extras[..., 7].abs()
        q_lep = torch.sigmoid(
            self.iso_alpha_lep * (self.iso_cut_lep - iso_lep)
        ) * torch.sigmoid(self.ip_alpha_lep * (self.ip_cut_lep - ipsig_lep))

        # Photon: relative isolation = min(topoetcone40, ptcone20) / pT.
        iso_photon = torch.minimum(extras[..., 10], extras[..., 11]) / pt_true
        q_photon = torch.sigmoid(
            self.iso_alpha_photon * (self.iso_cut_photon - iso_photon)
        )

        # Jet: JVT (higher = real hard jet, lower = pileup).
        q_jet = torch.sigmoid(self.jvt_alpha_jet * (extras[..., 3] - self.jvt_cut_jet))

        # Tau: RNNJetScore (higher = real tau).
        q_tau = torch.sigmoid(self.rnn_alpha_tau * (extras[..., 9] - self.rnn_cut_tau))

        # Large-R jet: D2 (lower = boson-like substructure).
        q_lrj = torch.sigmoid(self.d2_alpha_lrj * (self.d2_cut_lrj - extras[..., 12]))

        # Combine via type masks. Default = 1 for unknown / MET / padding.
        q = torch.ones_like(pt_true)
        q = torch.where(is_lep, q_lep, q)
        q = torch.where(is_photon, q_photon, q)
        q = torch.where(is_jet, q_jet, q)
        q = torch.where(is_tau, q_tau, q)
        q = torch.where(is_lrj, q_lrj, q)

        # Inactive particles get q = 1 — the active-mask handles tuple exclusion.
        q = torch.where(mask, q, torch.ones_like(q))

        return q


class RiemannianDiTsModel(nn.Module):
    """Riemannian Flow Matching Diffusion Transformer.

    Input per particle: 5-dim manifold representation + conditioning.
      - Massless: (logE_norm, nx, ny, nz, 0) — R × S²
      - Massive:  (χ_norm, nx, ny, nz, logm_norm) — R × S² × R  (Log H³_1)
        where χ = arccosh(E/m), n̂ = p⃗/|p⃗|

    Output (velocity on product manifold):
      - Massless: (d_logE/dt, v_n_x, v_n_y, v_n_z, 0) — velocity on R⁺ × S²
      - Massive:  (v⁰, v¹, v², v³, d_logm/dt) — tangent to H³ × R

    Time is continuous t in [0, 1], scaled by 1000 for the sinusoidal embedding.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        type_embed_dim: int = 32,
        charge_embed_dim: int = 8,
        met_embed_dim: int = 64,
        num_met_tokens: int = 4,
        num_extra_features: int = 13,
        manifold_stats: dict | None = None,
        direction_geometry: str = "S2",
        aux_head_config: dict | None = None,
        auxiliary_loss_config: dict | None = None,
        per_particle_adaln: bool = False,
        quality_weighting_config: dict | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_met_tokens = num_met_tokens
        self.num_extra_features = num_extra_features
        self.direction_geometry = direction_geometry
        self.per_particle_adaln = per_particle_adaln

        # Path 1: Two-path manifold projection (massless vs massive)
        # Separate projections because dim 0 = logE (massless) vs chi (massive),
        # dim 4 = unused (massless) vs logm (massive).
        self.manifold_proj_massless = nn.Linear(5, d_model)
        self.manifold_proj_massive = nn.Linear(5, d_model)

        # Path 2: FiLM identity conditioning — type + charge (+ extras) → (gamma, beta)
        # Multiplicative modulation: h = (1 + gamma) * h_manifold + beta
        # When per_particle_adaln=True, extras join the FiLM input so detector features
        # drive multiplicative modulation (instead of weak additive extra_proj).
        self.type_embed = nn.Embedding(NUM_PARTICLE_TYPES, type_embed_dim)
        self.charge_embed = nn.Embedding(NUM_CHARGE_CLASSES, charge_embed_dim)
        if per_particle_adaln:
            identity_dim = type_embed_dim + charge_embed_dim + num_extra_features
        else:
            identity_dim = type_embed_dim + charge_embed_dim
        self.identity_film = nn.Sequential(
            nn.Linear(identity_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        # FiLM at the input embedding is ALWAYS-ON (default Kaiming-uniform
        # init kept on the last Linear — no zero_ here). At step 0, γ, β are
        # small random vectors (PyTorch default for Linear(128, 256) gives
        # bound ≈ √(1/(3·128)) ≈ 0.05 → std ≈ 0.03), so
        # h_embed = (1+γ)⊙h_man + β ≈ h_man + small per-particle perturbation
        # already carrying type/charge/extras identity. The optimizer therefore
        # receives identity-aware gradient signal from the first step instead
        # of discovering identity through the coupled W_1 → W_2 chain bootstrap.
        #
        # Rank collapse (the original DiT justification for zero-init) is an
        # attention-mixing phenomenon; FiLM sits BEFORE any self-attention,
        # so input-layer diversity cannot cause it. Identity-at-init is still
        # enforced where it matters: the per-block adaLN-Zero residual gates
        # (adaln_mlp zero-init at particle_diffusion.py:671-672) and the
        # final-layer adaLN (model.py final_adaln zero-init below).

        # Path 3: Extra features conditioning
        # When per_particle_adaln=False: additive projection (legacy behavior).
        # When per_particle_adaln=True: extras are injected via FiLM (above) and
        # also drive per-particle adaLN-Zero modulation at every transformer block
        # via cond_proj. cond_proj uses standard (Kaiming) init — NOT zero-init —
        # so SiLU(base) = SiLU(time_emb + cond) depends on extras from step 0.
        # Block identity-at-init is still enforced by zero-init of the block's
        # adaln_linear (the true load-bearing constraint for adaLN-Zero); making
        # cond_proj zero-init too would delay extras-aware gradient to adaln_linear
        # until the next step, cascading into a long warmup (~10k steps observed).
        if per_particle_adaln:
            self.cond_proj = nn.Linear(identity_dim, d_model)
        else:
            self.extra_proj = nn.Linear(num_extra_features, d_model)

        # Continuous time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # MET conditioning via cross-attention
        if num_met_tokens > 0:
            self.met_kv_proj = nn.Sequential(
                nn.Linear(2, met_embed_dim),
                nn.SiLU(),
                nn.Linear(met_embed_dim, num_met_tokens * d_model),
            )
        else:
            self.met_proj = nn.Sequential(
                nn.Linear(2, met_embed_dim),
                nn.SiLU(),
                nn.Linear(met_embed_dim, d_model),
            )

        # Transformer blocks (adaLN-Zero: zero-initialized output gates)
        use_cross_attn = num_met_tokens > 0
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroDiTBlock(
                    d_model,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    met_cross_attention=use_cross_attn,
                )
                for _ in range(num_layers)
            ]
        )

        # Final-layer adaLN modulation (DiT pattern)
        self.out_norm = RMSNorm(d_model)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)

        # Output: 5-component x_0 prediction per particle.
        # Zero-init by default (model starts predicting zero → bounded initial
        # Minkowski loss), but ONLY when SOME other Kaiming-init head exists
        # to propagate gradient to backbone params at step 0. Without that,
        # zero-init out_proj leaves all backbone params with zero gradient
        # (the dL/dh = W.T @ dL/dy formula is zero when W=0), which DDP
        # rejects as "unused parameters" and torch.compile can elide entirely.
        # When auxiliary_loss is enabled WITHOUT aux_head, the primary head
        # IS the only path to backbone — switch to Kaiming init in that case.
        ahc = aux_head_config or {}
        self.use_aux_head = ahc.get("enabled", False)
        alc_for_init = auxiliary_loss_config or {}
        _al_enabled_for_init = bool(alc_for_init.get("enabled", False))
        self.out_proj = nn.Linear(d_model, 5)
        if self.use_aux_head:
            # aux_head provides Kaiming-init gradient path → safe to zero-init
            # out_proj (preserves bounded-initial-loss design).
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        elif _al_enabled_for_init:
            # auxiliary_loss is the only path to backbone → keep Kaiming init
            # (PyTorch default) so gradient flows to backbone from step 0.
            pass
        else:
            # Neither aux head present → preserve legacy zero-init behavior.
            # Caller is responsible for some other gradient path (e.g. CFM
            # loss alone with non-zero target).
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        # Auxiliary per-particle output head (dual out_proj):
        # separate output layer sharing backbone, trained only by K-body loss
        # to provide peak-sharpening signal without distorting primary out_proj.
        if self.use_aux_head:
            # Default (Kaiming) init — NOT zero-init. Same DDP "unused
            # parameter" rationale as above: this provides the backbone
            # gradient path when out_proj is zero-init.
            self.out_proj_aux = nn.Linear(d_model, 5)

        # Per-particle quality scorer (consolidated, single instance).
        # Fixed-threshold sigmoids on detector quality features (iso, IP, JVT,
        # RNN, D2). NOT learnable. Used by whichever K-body consumer is active
        # (aux_head OR auxiliary_loss; they're mutually exclusive) and, when
        # opted-in, by the CFM primary loss as a particleness weight.
        #
        # Reject the legacy per-consumer `quality_weighting` blocks so old
        # configs get a clear migration error rather than silently disabling
        # the scorer.
        ahc_legacy = ahc if isinstance(ahc, dict) else {}
        alc_legacy = auxiliary_loss_config or {}
        for parent_name, parent_cfg in (
            ("aux_head", ahc_legacy),
            ("auxiliary_loss", alc_legacy),
        ):
            if "quality_weighting" in parent_cfg:
                raise ValueError(
                    f"{parent_name}.quality_weighting block is removed. Migrate "
                    "to the top-level `quality_weighting:` config block."
                )

        qw_cfg = quality_weighting_config or {}
        self.use_quality_weighting = bool(qw_cfg.get("enabled", False))
        if self.use_quality_weighting:
            self.quality_scorer = QualityScorer(qw_cfg)
        else:
            self.quality_scorer = None

        # --- LUT buffers ---
        _lut_size = 100
        # Type code → embedding index
        lut = torch.full((_lut_size,), NUM_PARTICLE_TYPES - 1, dtype=torch.long)
        for code, idx in _TYPE_TO_IDX.items():
            lut[code] = idx
        self.register_buffer("_type_lut", lut)

        # Boolean LUTs indexed by type_idx
        n_types = len(PARTICLE_TYPE_CODES) + 1
        is_massless = [False] * n_types
        is_massive = [False] * n_types
        e_ref_vals = [1.0] * n_types
        mass_floor_vals = [0.0] * n_types
        logm_mu_vals = [0.0] * n_types
        logm_sigma_vals = [1.0] * n_types
        e_min_vals = [1.0] * n_types
        chi_mu_vals = [0.0] * n_types
        chi_sigma_vals = [1.0] * n_types

        for tcode in MASSLESS_TYPES:
            is_massless[_TYPE_TO_IDX[tcode]] = True
        for tcode in MASSIVE_TYPES:
            is_massive[_TYPE_TO_IDX[tcode]] = True

        # Load from manifold_stats config (data-driven), fallback to geometry.py constants
        ms = manifold_stats or {}
        cfg_e_ref = ms.get("e_ref", {})
        cfg_logm = ms.get("logm_stats", {})
        cfg_chi = ms.get("chi_stats", {})
        cfg_mfloor = ms.get("mass_floor", {})

        for tcode, eref in E_REF.items():
            if tcode in _TYPE_TO_IDX:
                e_ref_vals[_TYPE_TO_IDX[tcode]] = float(
                    cfg_e_ref.get(str(tcode), cfg_e_ref.get(tcode, eref))
                )
        for tcode, mf in MASS_FLOOR.items():
            if tcode in _TYPE_TO_IDX:
                mass_floor_vals[_TYPE_TO_IDX[tcode]] = float(
                    cfg_mfloor.get(str(tcode), cfg_mfloor.get(tcode, mf))
                )
        for tcode, (mu, sig) in LOGM_STATS.items():
            if tcode in _TYPE_TO_IDX:
                cfg_lm = cfg_logm.get(str(tcode), cfg_logm.get(tcode, {}))
                if isinstance(cfg_lm, dict):
                    logm_mu_vals[_TYPE_TO_IDX[tcode]] = float(cfg_lm.get("mu", mu))
                    logm_sigma_vals[_TYPE_TO_IDX[tcode]] = float(
                        cfg_lm.get("sigma", sig)
                    )
                else:
                    logm_mu_vals[_TYPE_TO_IDX[tcode]] = mu
                    logm_sigma_vals[_TYPE_TO_IDX[tcode]] = sig
        for tcode, (mu, sig) in CHI_STATS.items():
            if tcode in _TYPE_TO_IDX:
                cfg_ch = cfg_chi.get(str(tcode), cfg_chi.get(tcode, {}))
                if isinstance(cfg_ch, dict):
                    chi_mu_vals[_TYPE_TO_IDX[tcode]] = float(cfg_ch.get("mu", mu))
                    chi_sigma_vals[_TYPE_TO_IDX[tcode]] = float(
                        cfg_ch.get("sigma", sig)
                    )
                else:
                    chi_mu_vals[_TYPE_TO_IDX[tcode]] = mu
                    chi_sigma_vals[_TYPE_TO_IDX[tcode]] = sig
        for tcode, emin in E_MIN.items():
            if tcode in _TYPE_TO_IDX:
                e_min_vals[_TYPE_TO_IDX[tcode]] = emin

        self.register_buffer(
            "_is_massless_lut", torch.tensor(is_massless, dtype=torch.bool)
        )
        self.register_buffer(
            "_is_massive_lut", torch.tensor(is_massive, dtype=torch.bool)
        )
        self.register_buffer(
            "_e_ref_lut", torch.tensor(e_ref_vals, dtype=torch.float32)
        )
        self.register_buffer(
            "_mass_floor_lut", torch.tensor(mass_floor_vals, dtype=torch.float32)
        )
        self.register_buffer(
            "_logm_mu_lut", torch.tensor(logm_mu_vals, dtype=torch.float32)
        )
        self.register_buffer(
            "_logm_sigma_lut", torch.tensor(logm_sigma_vals, dtype=torch.float32)
        )
        self.register_buffer(
            "_e_min_lut", torch.tensor(e_min_vals, dtype=torch.float32)
        )
        self.register_buffer(
            "_chi_mu_lut", torch.tensor(chi_mu_vals, dtype=torch.float32)
        )
        self.register_buffer(
            "_chi_sigma_lut", torch.tensor(chi_sigma_vals, dtype=torch.float32)
        )

        # Per-type fundamental rest mass for "massless"-type decode. At decode
        # we use E = sqrt(m_rest² + |p|²) instead of E = |p| for types where
        # m_rest > 0. Muon-only by scope: the dimuon kinematic threshold
        # 2·m_μ ≈ 0.211 GeV is observed in the data and currently leaks at
        # decode. Electron correction is below FP32 noise at the 7 GeV pT cut;
        # photon is exactly m=0 and gets the default 0. Composite types (jet,
        # large-R jet, tau) get m_rest = 0 so we don't double-count — their
        # mass comes from the learnable s_4 = logm_norm coord. PDG value
        # hardcoded since it's a universal constant.
        PDG_REST_MASS_GEV = {
            13: 0.105658,  # muon
        }
        m_rest_vals = [0.0] * n_types
        for tcode, m in PDG_REST_MASS_GEV.items():
            if tcode in _TYPE_TO_IDX:
                m_rest_vals[_TYPE_TO_IDX[tcode]] = m
        self.register_buffer(
            "_m_rest_lut", torch.tensor(m_rest_vals, dtype=torch.float32)
        )

    def map_type_to_idx(self, particle_type: torch.Tensor) -> torch.Tensor:
        safe = particle_type.clamp(0, self._type_lut.shape[0] - 1)
        return self._type_lut[safe]

    # ------------------------------------------------------------------
    # Manifold encoding / decoding
    # ------------------------------------------------------------------

    def encode_to_manifold(
        self,
        kin: torch.Tensor,
        p_type: torch.Tensor,
        p_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert collator kinematics to manifold representation.

        Legacy chart (pre-bounded-pT):
            Massless: s_0 = log(E / E_ref[type])               — R × S²
            Massive:  s_0 = χ_norm = (χ − μ_χ)/σ_χ, χ = arccosh(E/m)
                      s_4 = (log m − μ_logm)/σ_logm             — R × S² × R

        Args:
            kin:    (..., 4) in (px, py, pz, E)  — any leading dims
            p_type: (...,) type codes
            p_mask: (...,) mask (None = all active, for geometric mode)

        Returns:
            manifold_repr: (..., 5)
              Massless: (log(E/E_ref), nx, ny, nz, 0)
              Massive:  (χ_norm, nx, ny, nz, logm_norm)
        """
        p = collator_to_minkowski(kin)  # (E, px, py, pz)

        type_idx = self.map_type_to_idx(p_type)
        is_massless = self._is_massless_lut[type_idx]
        is_massive = self._is_massive_lut[type_idx]
        active = (
            p_mask > 0.5
            if p_mask is not None
            else torch.ones_like(p_type, dtype=torch.bool)
        )

        e_ref = self._e_ref_lut[type_idx]
        mass_floor = self._mass_floor_lut[type_idx]
        logm_mu = self._logm_mu_lut[type_idx]
        logm_sigma = self._logm_sigma_lut[type_idx]

        # --- Massless path: R⁺ × S² (logE_norm, nx, ny, nz, 0) ---
        e_min = self._e_min_lut[type_idx]
        E = p[..., 0].clamp(min=e_min.clamp(min=1e-6))
        logE_norm = torch.log(E / e_ref.clamp(min=1.0))
        # Unit direction vector on S²: n = p/|p|
        p_spatial = p[..., 1:]  # (B, N, 3)
        p_mag = p_spatial.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        n_hat = p_spatial / p_mag  # (B, N, 3)

        massless_repr = torch.cat(
            [
                logE_norm.unsqueeze(-1),
                n_hat,
                torch.zeros_like(logE_norm.unsqueeze(-1)),
            ],
            dim=-1,
        )

        # --- Massive path: Log H³_1 → (χ_norm, nx, ny, nz, logm_norm) ---
        # Compute invariant mass with floor
        m_sq = (p[..., 0] ** 2 - p_spatial.pow(2).sum(dim=-1)).clamp(min=0.0)
        m = torch.sqrt(m_sq + 1e-12)
        m_floored = torch.where(is_massive, m.clamp(min=mass_floor.clamp(min=0.01)), m)
        logm_raw = torch.log(m_floored.clamp(min=1e-6))
        logm_norm = (logm_raw - logm_mu) / logm_sigma.clamp(min=0.01)

        # Chi = arccosh(E/m) — boost rapidity, normalized per-type
        chi_mu = self._chi_mu_lut[type_idx]
        chi_sigma = self._chi_sigma_lut[type_idx]
        E_massive = p[..., 0].clamp(min=m_floored + 1e-6)
        gamma = (E_massive / m_floored.clamp(min=1e-6)).clamp(min=1.0 + 1e-6)
        chi = torch.acosh(gamma)
        chi_norm = (chi - chi_mu) / chi_sigma.clamp(min=0.01)

        # n_hat already computed above — reuse for both sectors
        massive_repr = torch.cat(
            [chi_norm.unsqueeze(-1), n_hat, logm_norm.unsqueeze(-1)], dim=-1
        )

        # Compose
        manifold = torch.zeros_like(massless_repr)
        fm = (is_massless & active).unsqueeze(-1)
        mm = (is_massive & active).unsqueeze(-1)
        manifold = torch.where(fm, massless_repr, manifold)
        manifold = torch.where(mm, massive_repr, manifold)

        return manifold

    def decode_from_manifold(
        self,
        manifold_repr: torch.Tensor,
        p_type: torch.Tensor,
        p_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert manifold representation back to (px, py, pz, E).

        Inverts encode_to_manifold (legacy chart):
          - Massless: E = E_ref · exp(s_0); p⃗ = E · n̂  (lightlike)
          - Massive : χ = s_0·σ_χ + μ_χ; m = exp(s_4·σ_logm + μ_logm);
                      E = m·cosh(χ);  |p⃗| = m·sinh(χ);  p⃗ = |p⃗| · n̂

        Args:
            manifold_repr: (..., 5)
            p_type: (...,)
            p_mask: (...,) or None (all active, for geometric mode)

        Returns:
            kin: (..., 4) in (px, py, pz, E) collator order
        """
        type_idx = self.map_type_to_idx(p_type)
        is_massless = self._is_massless_lut[type_idx]
        is_massive = self._is_massive_lut[type_idx]
        active = (
            p_mask > 0.5
            if p_mask is not None
            else torch.ones_like(p_type, dtype=torch.bool)
        )
        e_ref = self._e_ref_lut[type_idx]
        m_rest = self._m_rest_lut[type_idx]

        # Massless: R⁺ × S² → Minkowski. We decode E from the manifold scale
        # E_scale = E_ref · exp(s_0), then enforce on-shell for types with a
        # fundamental rest mass (muon): |p| = sqrt(E_scale² − m_rest²) and
        # E = sqrt(m_rest² + |p|²) = E_scale exactly. For m_rest = 0 (photon
        # and composite types) this collapses to |p| = E, the lightlike form.
        logE_norm = manifold_repr[..., 0]
        n_raw = manifold_repr[..., 1:4]  # (B, N, 3)
        n_hat = F.normalize(n_raw, dim=-1, eps=1e-6)  # project to S²
        E_scale = e_ref * torch.exp(logE_norm.clamp(-10, 10))
        p_mag_ml = (E_scale * E_scale - m_rest * m_rest).clamp(min=0.0).sqrt()
        E_ml = (m_rest * m_rest + p_mag_ml * p_mag_ml).clamp(min=1e-12).sqrt()
        mink_massless = torch.cat(
            [E_ml.unsqueeze(-1), p_mag_ml.unsqueeze(-1) * n_hat], dim=-1
        )

        # Massive: Log H³_1 → (χ_norm, nx, ny, nz, logm_norm)
        chi_mu = self._chi_mu_lut[type_idx]
        chi_sigma = self._chi_sigma_lut[type_idx]
        logm_mu = self._logm_mu_lut[type_idx]
        logm_sigma = self._logm_sigma_lut[type_idx]

        chi = manifold_repr[..., 0] * chi_sigma.clamp(min=0.01) + chi_mu
        n_mv = F.normalize(manifold_repr[..., 1:4], dim=-1, eps=1e-6)
        logm_raw = manifold_repr[..., 4] * logm_sigma.clamp(min=0.01) + logm_mu
        m_phys = torch.exp(logm_raw.clamp(-10, 10))
        E_mv = m_phys * torch.cosh(chi.clamp(0, 12))
        p_mag = m_phys * torch.sinh(chi.clamp(0, 12))
        mink_massive = torch.cat(
            [E_mv.unsqueeze(-1), p_mag.unsqueeze(-1) * n_mv], dim=-1
        )

        # Select
        fm = (is_massless & active).unsqueeze(-1)
        mm = (is_massive & active).unsqueeze(-1)
        p = torch.where(fm, mink_massless, torch.zeros_like(mink_massless))
        p = torch.where(mm, mink_massive, p)

        return minkowski_to_collator(p)

    # ------------------------------------------------------------------
    # FiLM two-path input embedding (shared by all forward paths)
    # ------------------------------------------------------------------

    def _embed_input(
        self,
        z_t: torch.Tensor,
        particle_type: torch.Tensor,
        particle_charge: torch.Tensor,
        particle_extra: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """FiLM two-path input embedding — shape-agnostic.

        Works on (B, N, 5) padded or (T, 5) flat tensors.

        Returns:
            h: (..., d_model) embedded input
            type_idx: (...,) mapped type indices
            cond: (..., d_model) per-particle conditioning for adaLN-Zero, or None
                  when per_particle_adaln is False.
        """
        type_idx = self.map_type_to_idx(particle_type)
        is_massless = self._is_massless_lut[type_idx]
        is_massive = self._is_massive_lut[type_idx]
        fm = (is_massless & active).unsqueeze(-1)
        mm = (is_massive & active).unsqueeze(-1)

        # Path 1: Two-path manifold projection (select by massless/massive)
        h_ml = self.manifold_proj_massless(z_t)
        h_mv = self.manifold_proj_massive(z_t)
        h = torch.where(fm, h_ml, torch.where(mm, h_mv, torch.zeros_like(h_ml)))

        # Path 2: FiLM identity modulation — type + charge (+ extras if enabled)
        if self.per_particle_adaln:
            identity = torch.cat(
                [
                    self.type_embed(type_idx),
                    self.charge_embed(particle_charge),
                    particle_extra[..., : self.num_extra_features],
                ],
                dim=-1,
            )
        else:
            identity = torch.cat(
                [self.type_embed(type_idx), self.charge_embed(particle_charge)],
                dim=-1,
            )
        film_out = self.identity_film(identity)
        gamma, beta = film_out.chunk(2, dim=-1)
        h = (1 + gamma) * h + beta

        # Path 3: Extra features — FiLM (above) + per-particle cond OR legacy additive
        cond = None
        if self.per_particle_adaln:
            # Per-particle adaLN-Zero conditioning (zero-init → identity at init)
            cond = self.cond_proj(identity)
            cond = cond * active.unsqueeze(-1).float()
        else:
            # Legacy: additive projection of extras
            h = h + self.extra_proj(particle_extra[..., : self.num_extra_features])

        # Zero padding tokens (FiLM beta leaks into inactive slots)
        h = h * active.unsqueeze(-1).float()

        return h, type_idx, cond

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_t: torch.Tensor,
        particle_type: torch.Tensor,
        particle_charge: torch.Tensor,
        particle_extra: torch.Tensor,
        t: torch.Tensor,
        particle_mask: torch.Tensor,
        met_vector: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity from interpolated state on R × S² (× R for massive).

        Args:
            return_aux: if True, also return aux-head velocity (B, N, 5)

        Returns:
            velocity: (B, N, 5) on R × S² (× R)
            When return_aux=True, returns (velocity, raw_aux) where
              raw_aux: (B, N, 5) — aux head velocity, or None if disabled
        """
        # FiLM two-path input embedding
        active = particle_mask > 0.5
        h, type_idx, cond = self._embed_input(
            z_t, particle_type, particle_charge, particle_extra, active
        )

        # Time embedding
        time_emb = self.time_embed(t * 1000.0)

        # MET conditioning
        met_context = None
        if self.num_met_tokens > 0 and met_vector is not None:
            B_met = met_vector.shape[0]
            met_context = self.met_kv_proj(met_vector)
            met_context = met_context.view(B_met, self.num_met_tokens, self.d_model)
        elif self.num_met_tokens == 0 and met_vector is not None:
            met_emb = self.met_proj(met_vector)
            time_emb = time_emb + met_emb

        # Transformer blocks
        for block in self.blocks:
            h = block(
                h,
                time_emb,
                attention_mask=particle_mask,
                met_context=met_context,
                cond=cond,
            )

        # Final-layer adaLN modulation + output projection
        # Per-particle: base = time_emb.unsqueeze(1) + cond; final_adaln sees (B, N, D)
        # Per-event:   base = time_emb (B, D); chunks unsqueezed to (B, 1, D) for broadcast
        if cond is not None:
            final_base = time_emb.unsqueeze(1) + cond  # (B, N, D)
            gamma_final, beta_final = self.final_adaln(final_base).chunk(2, dim=-1)
        else:
            gamma_final, beta_final = (
                v.unsqueeze(1) for v in self.final_adaln(time_emb).chunk(2, dim=-1)
            )
        h_modulated = self.out_norm(h) * (1 + gamma_final) + beta_final
        raw = self.out_proj(h_modulated)

        # Tangent-space projection for velocity output
        raw = self._postprocess_velocity(raw, z_t, type_idx, particle_mask)

        # Auxiliary head (dual out_proj): separate output layer sharing backbone
        raw_aux = None
        if self.use_aux_head:
            raw_aux = self.out_proj_aux(h_modulated)
            raw_aux = self._postprocess_velocity(raw_aux, z_t, type_idx, particle_mask)

        if return_aux:
            return raw, raw_aux
        return raw

    def _postprocess_velocity(
        self,
        raw: torch.Tensor,
        z_t: torch.Tensor,
        type_idx: torch.Tensor,
        particle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Post-process raw output for velocity prediction (RFM mode).

        Unified for both massless and massive (Log H³_1) sectors:
          - Dim 0 (logE or chi velocity): pass through unconstrained (R)
          - Dims 1:4: tangent-project to T_n S²
          - Dim 4: pass through for massive, zeroed for massless
        """
        is_massless = self._is_massless_lut[type_idx]
        active = particle_mask > 0.5

        out = raw.clone()

        # S² mode: project velocity to tangent plane T_n S²
        # R3 mode: raw dims 1:4 pass through unconstrained
        if self.direction_geometry == "S2" and active.any():
            n = F.normalize(z_t[..., 1:4], dim=-1, eps=1e-6)
            v_tan_s2 = tangent_project_S2(raw[..., 1:4], n)
            a = active.unsqueeze(-1)
            out[..., 1:4] = torch.where(a, v_tan_s2, out[..., 1:4])

        # Zero dim 4 for massless only
        out[..., 4] = torch.where(
            is_massless & active, torch.zeros_like(out[..., 4]), out[..., 4]
        )

        # Zero padding
        return out * active.unsqueeze(-1).float()

    # ------------------------------------------------------------------
    # Geometric (varlen) forward — no padding
    # ------------------------------------------------------------------

    def forward_geometric(
        self,
        z_t: torch.Tensor,
        particle_type: torch.Tensor,
        particle_charge: torch.Tensor,
        particle_extra: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        met_vector: Optional[torch.Tensor] = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Predict velocity from flat geometric tensors (no padding).

        Args:
            z_t:             (T_total, 5)  manifold state at time t
            particle_type:   (T_total,)    type codes
            particle_charge: (T_total,)    shifted charge {0,1,2}
            particle_extra:  (T_total, 13) auxiliary features
            t:               (B,)          continuous time [0,1]
            batch:           (T_total,)    event index per particle
            ptr:             (B+1,)        int32 cumulative particle counts
            met_vector:      (B, 2) or None
            max_seqlen:      pre-computed max particles per event (avoids GPU sync)

        Returns:
            velocity: (T_total, 5) on R × S² (× R)
        """
        B_ev = ptr.shape[0] - 1
        if max_seqlen is None:
            max_seqlen = int((ptr[1:] - ptr[:-1]).max().item())

        cu_seqlens = ptr.to(torch.int32)

        # FiLM two-path input embedding (all flat, T_total — all particles real)
        active = torch.ones(z_t.shape[0], dtype=torch.bool, device=z_t.device)
        h, type_idx, cond = self._embed_input(
            z_t, particle_type, particle_charge, particle_extra, active
        )

        # Time embedding (per-event)
        time_emb = self.time_embed(t * 1000.0)  # (B, d_model)

        # MET conditioning
        met_context = None
        if self.num_met_tokens > 0 and met_vector is not None:
            met_context = self.met_kv_proj(met_vector)
            met_context = met_context.view(B_ev, self.num_met_tokens, self.d_model)
        elif self.num_met_tokens == 0 and met_vector is not None:
            met_emb = self.met_proj(met_vector)
            time_emb = time_emb + met_emb

        # --- Transformer blocks — fully flat, no padding ---
        for blk in self.blocks:
            h = blk.forward_geometric(
                h,
                time_emb,
                batch,
                cu_seqlens,
                max_seqlen,
                met_context=met_context,
                cond=cond,
            )

        # Final-layer adaLN modulation + output projection
        if cond is not None:
            final_base = time_emb[batch] + cond  # (T_total, D)
            gamma_final, beta_final = self.final_adaln(final_base).chunk(2, dim=-1)
            raw = self.out_proj(self.out_norm(h) * (1 + gamma_final) + beta_final)
        else:
            gamma_final, beta_final = (
                v for v in self.final_adaln(time_emb).chunk(2, dim=-1)
            )
            raw = self.out_proj(
                self.out_norm(h) * (1 + gamma_final[batch]) + beta_final[batch]
            )

        # Tangent-space projection (flat)
        self._postprocess_velocity_flat(raw, z_t, type_idx)

        return raw

    def forward_flat(
        self,
        z_t: torch.Tensor,
        particle_type: torch.Tensor,
        particle_charge: torch.Tensor,
        particle_extra: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        met_vector: Optional[torch.Tensor] = None,
        max_seqlen: int | None = None,
        T_real: int | None = None,
    ) -> torch.Tensor:
        """Fully flat forward — zero [batch] gathers, fully compilable.

        Per-event quantities (time embedding, MET) are expanded to
        per-particle ONCE at the top. All blocks then operate purely
        on (T, D) tensors with varlen_attn for event-aware attention.

        Args:
            z_t:             (T, 5)  manifold state
            particle_type:   (T,)    type codes
            particle_charge: (T,)    shifted charge
            particle_extra:  (T, 13) auxiliary features
            t:               (B,)    continuous time [0,1]
            batch:           (T,)    event index per particle
            ptr:             (B+1,)  int32 cumulative counts
            met_vector:      (B, 2) or None
            max_seqlen:      max particles per event
            T_real:          number of real particles (Python int, no GPU sync)

        Returns:
            velocity: (T, 5)
        """
        B_ev = ptr.shape[0] - 1
        if max_seqlen is None:
            max_seqlen = int((ptr[1:] - ptr[:-1]).max().item())
        cu_seqlens = ptr.to(torch.int32)
        if T_real is None:
            T_real = int(cu_seqlens[-1].item())  # fallback with GPU sync

        # FiLM two-path input embedding (flat — padding tokens have type=0)
        active = particle_type > 0
        h, type_idx, cond = self._embed_input(
            z_t, particle_type, particle_charge, particle_extra, active
        )

        # Time embedding: compute per-event, expand to per-particle ONCE
        time_emb = self.time_embed(t * 1000.0)  # (B, D)
        if self.num_met_tokens == 0 and met_vector is not None:
            time_emb = time_emb + self.met_proj(met_vector)
        time_emb_flat = time_emb[batch]  # (T, D) — the ONLY [batch] gather

        # MET context stays (B, M, D) — cross-attn handles it via cu_seqlens
        met_context = None
        if self.num_met_tokens > 0 and met_vector is not None:
            met_context = self.met_kv_proj(met_vector)
            met_context = met_context.view(B_ev, self.num_met_tokens, self.d_model)

        # Slice to real particles only for the transformer blocks.
        # Padding tokens (p_type==0) must not enter varlen_attn because
        # its backward produces NaN gradients for out-of-range positions.
        # T_real is passed as a Python int from the collator — no GPU sync.
        T_total = h.shape[0]
        h_real = h[:T_real]
        te_real = time_emb_flat[:T_real]
        cond_real = cond[:T_real] if cond is not None else None

        # Transformer blocks — only real particles
        for blk in self.blocks:
            h_real = blk.forward_flat(
                h_real,
                te_real,
                cu_seqlens,
                max_seqlen,
                met_context_flat=met_context,
                cond=cond_real,
            )

        # Final adaLN + output projection (only real particles)
        final_base = te_real + cond_real if cond_real is not None else te_real
        final_adaln_out = self.final_adaln(final_base)
        gamma_final, beta_final = final_adaln_out.chunk(2, dim=-1)
        raw_real = self.out_proj(self.out_norm(h_real) * (1 + gamma_final) + beta_final)

        # Pad back to T_total with zeros
        if T_real < T_total:
            raw = torch.zeros(T_total, 5, device=raw_real.device, dtype=raw_real.dtype)
            raw[:T_real] = raw_real
        else:
            raw = raw_real

        # Tangent-space projection (flat)
        self._postprocess_velocity_flat(raw, z_t, type_idx)
        return raw

    def _postprocess_velocity_flat(
        self,
        raw: torch.Tensor,
        z_t: torch.Tensor,
        type_idx: torch.Tensor,
    ) -> None:
        """In-place tangent projection + massless zeroing for flat tensors."""
        is_massless = self._is_massless_lut[type_idx]

        # S² mode: tangent projection. R3 mode: pass through.
        if self.direction_geometry == "S2":
            n = z_t[..., 1:4]
            n = n / (n * n).sum(dim=-1, keepdim=True).clamp(min=1e-12).sqrt()
            v_s2 = raw[..., 1:4]
            dot = (v_s2 * n).sum(dim=-1, keepdim=True)
            raw[..., 1:4] = v_s2 - dot * n

        # Zero dim 4 for massless
        raw[..., 4] = torch.where(
            is_massless, torch.zeros_like(raw[..., 4]), raw[..., 4]
        )

    # ------------------------------------------------------------------
    # NJT (NestedTensor) forward — torch.compile-friendly
    # ------------------------------------------------------------------

    def _njt_final_block(self, h_njt, time_emb):
        """Final adaLN + output projection on NJT (compilable).

        Per-event variant (cond is None). The per-particle path is inlined
        in forward_njt (see _njt_final_block_per_particle) because compiling
        it requires NJT-aware adaln_mlp input.
        """
        gamma_final, beta_final = (
            v.unsqueeze(1) for v in self.final_adaln(time_emb).chunk(2, dim=-1)
        )
        return self.out_proj(self.out_norm(h_njt) * (1 + gamma_final) + beta_final)

    def forward_njt(
        self,
        z_t: torch.Tensor,
        particle_type: torch.Tensor,
        particle_charge: torch.Tensor,
        particle_extra: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        met_vector: Optional[torch.Tensor] = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Predict velocity using NJT for AdaLN/MLP/Norm, varlen for attention.

        All pointwise ops (RMSNorm, AdaLN modulation, SwiGLU MLP, residuals)
        run on NestedTensor (jagged layout), which torch.compile can fuse
        into a few large kernels. Attention drops to flat tensors for
        varlen_attn, then wraps back.

        Args:
            z_t:             (T_total, 5)  manifold state at time t
            particle_type:   (T_total,)    type codes
            particle_charge: (T_total,)    shifted charge {0,1,2}
            particle_extra:  (T_total, 13) auxiliary features
            t:               (B,)          continuous time [0,1]
            batch:           (T_total,)    event index per particle
            ptr:             (B+1,)        int32 cumulative particle counts
            met_vector:      (B, 2) or None
            max_seqlen:      pre-computed max particles per event

        Returns:
            velocity: (T_total, 5) on R × S² (× R)
        """
        B_ev = ptr.shape[0] - 1
        if max_seqlen is None:
            max_seqlen = int((ptr[1:] - ptr[:-1]).max().item())

        # FiLM two-path input embedding (flat — all particles real in NJT mode)
        active = torch.ones(z_t.shape[0], dtype=torch.bool, device=z_t.device)
        h_flat, type_idx, cond = self._embed_input(
            z_t, particle_type, particle_charge, particle_extra, active
        )

        # Time embedding (per-event)
        time_emb = self.time_embed(t * 1000.0)  # (B, d_model)

        # MET conditioning
        met_context = None
        if self.num_met_tokens > 0 and met_vector is not None:
            met_context = self.met_kv_proj(met_vector)
            met_context = met_context.view(B_ev, self.num_met_tokens, self.d_model)
        elif self.num_met_tokens == 0 and met_vector is not None:
            met_emb = self.met_proj(met_vector)
            time_emb = time_emb + met_emb

        # --- Wrap as NJT for the transformer blocks ---
        offsets = ptr.long()
        h_njt = torch.nested.nested_tensor_from_jagged(h_flat, offsets=offsets)

        cu_seqlens = ptr.to(torch.int32)

        # Compile NJT helpers on first call (fuses norm+modulation+MLP).
        # Note: the per-particle-cond path inlines its own NJT ops (does not use
        # _njt_pre_attn / _njt_post_attn) so compilation of those is only useful
        # when cond is None.
        if not getattr(self, "_njt_compiled", False):
            compile_opts = dict(mode="max-autotune-no-cudagraphs")
            for block in self.blocks:
                block._njt_pre_attn = torch.compile(block._njt_pre_attn, **compile_opts)
                block._njt_post_attn = torch.compile(
                    block._njt_post_attn, **compile_opts
                )
                if block.met_cross_attention:
                    block._njt_pre_cross_attn = torch.compile(
                        block._njt_pre_cross_attn, **compile_opts
                    )
            self._njt_final_block = torch.compile(self._njt_final_block, **compile_opts)
            self._njt_compiled = True

        # Transformer blocks
        for block in self.blocks:
            h_njt = block.forward_njt(
                h_njt,
                time_emb,
                cu_seqlens,
                max_seqlen,
                met_context=met_context,
                cond=cond,
            )

        # --- Final-layer adaLN + output projection (on NJT) ---
        if cond is not None:
            cond_njt = torch.nested.nested_tensor_from_jagged(cond, offsets=offsets)
            final_base_njt = time_emb.unsqueeze(1) + cond_njt
            gamma_final, beta_final = self.final_adaln(final_base_njt).chunk(2, dim=-1)
            h_njt = self.out_proj(self.out_norm(h_njt) * (1 + gamma_final) + beta_final)
        else:
            h_njt = self._njt_final_block(h_njt, time_emb)

        # --- Extract flat output and postprocess ---
        raw = h_njt.values()  # (T_total, 5)

        # Tangent-space projection (uses F.normalize internally, not NJT-safe)
        is_massless = self._is_massless_lut[type_idx]

        out = raw.clone()

        # S² mode: project velocity to tangent plane. R3 mode: pass through.
        if self.direction_geometry == "S2":
            n = z_t[..., 1:4]
            n_norm = (n * n).sum(dim=-1, keepdim=True).clamp(min=1e-12).sqrt()
            n = n / n_norm  # manual normalize (F.normalize not supported on NJT)
            v_s2 = raw[..., 1:4]
            dot = (v_s2 * n).sum(dim=-1, keepdim=True)
            out[..., 1:4] = v_s2 - dot * n

        out[..., 4] = torch.where(
            is_massless, torch.zeros_like(out[..., 4]), out[..., 4]
        )

        return out


def create_riemannian_dits(config: dict) -> RiemannianDiTsModel:
    """Factory function to create a RiemannianDiTsModel from a config dict."""
    return RiemannianDiTsModel(
        d_model=config.get("d_model", 128),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.1),
        met_embed_dim=config.get("met_embed_dim", 64),
        num_met_tokens=config.get("num_met_tokens", 4),
        num_extra_features=config.get("num_extra_features", 13),
        manifold_stats=config.get("manifold_stats"),
        direction_geometry=config.get("direction_geometry", "S2"),
        aux_head_config=config.get("aux_head"),
        auxiliary_loss_config=config.get("auxiliary_loss"),
        per_particle_adaln=config.get("per_particle_adaln", False),
        quality_weighting_config=config.get("quality_weighting"),
    )
