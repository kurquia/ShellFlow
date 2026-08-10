"""
Collator that converts tokenized particle data into per-particle Cartesian vectors.

Regroups flat (feature_value, feature_idx, particle_idx) tokens produced by
ParticleDataset into per-particle 4-vectors (px, py, pz, E). The kinematic
features (pT, eta, phi, E) must arrive in physical units — either via
normalize=False, or via a config whose kinematic entries set
apply_norm: false (the *_reduced dataloader configs). The remaining tokens
become discrete conditioning features (type, charge) and continuous auxiliary
features (z0, d0, btag_quantile, jvt, largeRJet_m, ptvarcone30, topoetcone20,
d0sig, tau_nTracks, tau_RNNJetScore, photon_topoetcone40, photon_ptcone20,
largeRJet_D2).

MET is extracted as a global event-level conditioning vector (px, py)
rather than included in the particle sequence, since MET is a derived
quantity — not a real particle.

All operations are vectorized via scatter_add — no Python loops over batch
items or particles.
"""

from typing import Dict, List

import torch

from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)

# Feature IDs (from data_feature_06.yaml)
_FID_TYPE = 1
_FID_PT = 2
_FID_ETA = 3
_FID_PHI = 4
_FID_E = 5
_FID_CHARGE = 6
_FID_TAU_NTRACKS = 7
_FID_TAU_RNNJET = 8
_FID_BTAG = 10
_FID_PHOTON_TOPOETCONE40 = 11
_FID_Z0 = 33
_FID_PHOTON_PTCONE20 = 38
_FID_LEP_PTVARCONE30 = 39
_FID_LEP_TOPOETCONE20 = 40
_FID_D0 = 41
_FID_LEP_D0SIG = 42
_FID_JVT = 44
_FID_LARGERJET_M = 50
_FID_LARGERJET_D2 = 51

# Particle type codes
_TYPE_MET = 0
_TYPE_LARGERJET = 10

# sinh(5.0) ≈ 74.2, safe for float32; detector |η| < 2.5 anyway
_ETA_CLAMP = 5.0


class ParticleVectorCollator:
    """Collate tokenized events into per-particle Cartesian vectors.

    MET is separated from the particle sequence and returned as a global
    conditioning vector.  The particle sequence contains only real particles
    (electrons, muons, taus, jets, photons, large-R jets).

    Output dictionary
    -----------------
    kinematics      : (B, N, 4)  float32  (px, py, pz, E) physical units (no MET)
    kinematics_mask : (B, N, 4)  float32  1=diffuse, 0=fixed (all 4 active)
    particle_type   : (B, N)     int64    type codes (1,10,11,13,15,22)
    particle_charge : (B, N)     int64    shifted to {0,1,2} ↔ {-1,0,+1}
    particle_extra  : (B, N, 13) float32  [z0, d0, btag, jvt, lRj_m, ptvarcone30, topoetcone20, d0sig, tau_nTrk, tau_RNNJet, ph_topoet40, ph_ptcone20, lRj_D2]
    particle_mask   : (B, N)     float32  1=real particle, 0=padding
    met_vector      : (B, 2)     float32  MET (px, py) in physical units
    """

    # Feature ID → slot index for single-pass scatter (order matters for _SLOT_*)
    _FEAT_SLOTS = [
        _FID_PT,
        _FID_ETA,
        _FID_PHI,
        _FID_E,
        _FID_TYPE,
        _FID_CHARGE,
        _FID_BTAG,
        _FID_Z0,
        _FID_D0,
        _FID_JVT,
        _FID_LARGERJET_M,
        _FID_LEP_PTVARCONE30,
        _FID_LEP_TOPOETCONE20,
        _FID_LEP_D0SIG,
        _FID_TAU_NTRACKS,
        _FID_TAU_RNNJET,
        _FID_PHOTON_TOPOETCONE40,
        _FID_PHOTON_PTCONE20,
        _FID_LARGERJET_D2,
    ]
    _NUM_SLOTS = len(_FEAT_SLOTS)
    # Slot constants for readability
    _S_PT, _S_ETA, _S_PHI, _S_E = 0, 1, 2, 3
    _S_TYPE, _S_CHARGE = 4, 5
    _S_BTAG, _S_Z0, _S_D0, _S_JVT, _S_LRJ_M = 6, 7, 8, 9, 10
    _S_PVC, _S_TEC, _S_D0S, _S_TNK, _S_TRJ, _S_PTE, _S_PPC, _S_D2 = (
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    )

    def __init__(self, max_particles: int = 30):
        self.max_particles = max_particles
        # Build LUT: feature_id → slot (0-based), -1 = unknown
        max_fid = max(self._FEAT_SLOTS) + 1
        self._slot_lut = torch.full((max_fid,), -1, dtype=torch.long)
        for slot, fid in enumerate(self._FEAT_SLOTS):
            self._slot_lut[fid] = slot

    # ------------------------------------------------------------------
    # Single-pass scatter: all features in 2 scatter_add ops
    # ------------------------------------------------------------------
    def _scatter_all(
        self,
        features: torch.Tensor,
        feature_idxs: torch.Tensor,
        particle_idxs: torch.Tensor,
        valid: torch.Tensor,
        B: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter all known features into (B, N, num_slots) in one pass.

        Returns (values, has_feat) each (B, N, num_slots).
        """
        S = self._NUM_SLOTS
        lut = self._slot_lut.to(feature_idxs.device)
        # Map feature IDs to slot indices; unknown → -1
        safe_fid = feature_idxs.clamp(0, lut.shape[0] - 1)
        slots = lut[safe_fid]  # (B, L)
        known = (slots >= 0) & valid  # (B, L)

        # Build flat scatter index: batch_offset + particle * S + slot
        # Use slot 0 of particle 0 as waste bin for unknown tokens (zeroed source)
        p_idx = particle_idxs.clamp(0, N - 1)
        batch_offset = torch.arange(B, device=features.device).unsqueeze(1) * (N * S)
        flat_idx = batch_offset + p_idx * S + slots.clamp(min=0)
        flat_idx = torch.where(known, flat_idx, torch.zeros_like(flat_idx))

        src = torch.where(known, features, torch.zeros_like(features))

        # +1 offset: index 0 is waste bin
        flat_idx_shifted = flat_idx + 1
        flat_idx_shifted = torch.where(
            known, flat_idx_shifted, torch.zeros_like(flat_idx_shifted)
        )

        values_flat = torch.zeros(
            B * N * S + 1, device=features.device, dtype=features.dtype
        )
        values_flat.scatter_add_(0, flat_idx_shifted.reshape(-1), src.reshape(-1))
        values = values_flat[1:].reshape(B, N, S)

        counts_flat = torch.zeros(
            B * N * S + 1, device=features.device, dtype=features.dtype
        )
        counts_flat.scatter_add_(
            0, flat_idx_shifted.reshape(-1), known.float().reshape(-1)
        )
        has_feat = counts_flat[1:].reshape(B, N, S) > 0.5

        return values, has_feat

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        features = torch.stack([item["features"] for item in batch])  # (B, L)
        feature_idxs = torch.stack([item["feature_idxs"] for item in batch])  # (B, L)
        particle_idxs = torch.stack([item["particle_idxs"] for item in batch])  # (B, L)
        has_raw = "features_raw" in batch[0]
        if has_raw:
            features_raw = torch.stack([item["features_raw"] for item in batch])

        B, L = features.shape
        N = self.max_particles
        valid = feature_idxs != 0  # non-padding

        # --- Single-pass scatter for all features ---
        vals, has = self._scatter_all(
            features, feature_idxs, particle_idxs, valid, B, N
        )

        pt = vals[..., self._S_PT]
        eta = vals[..., self._S_ETA]
        phi = vals[..., self._S_PHI]
        energy = vals[..., self._S_E]
        ptype_raw = vals[..., self._S_TYPE]
        charge_raw = vals[..., self._S_CHARGE]

        has_pt = has[..., self._S_PT]
        has_eta = has[..., self._S_ETA]
        has_e = has[..., self._S_E]
        hc = has[..., self._S_CHARGE]

        # --- discrete features ---
        particle_type = ptype_raw.round().long()  # (B, N)
        particle_charge = (charge_raw.round().long() + 1).clamp(0, 2)
        particle_charge[~hc] = 1  # default = 0 charge

        # --- Cartesian conversion (vectorized) ---
        eta_c = eta.clamp(-_ETA_CLAMP, _ETA_CLAMP)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta_c)
        pz[~has_eta] = 0.0  # MET has no η → pz = 0

        is_met = particle_type == _TYPE_MET
        is_largeR = particle_type == _TYPE_LARGERJET
        E = torch.where(has_e, energy, pt)

        # --- Extract MET ---
        met_px = (px * is_met.float()).sum(dim=1)
        met_py = (py * is_met.float()).sum(dim=1)
        met_vector = torch.stack([met_px, met_py], dim=-1)

        real_particle = has_pt & ~is_met
        real_particle = real_particle & (pt < 1000.0)

        kinematics = torch.stack([px, py, pz, E], dim=-1)
        particle_mask = real_particle.float()
        kin_mask = particle_mask.unsqueeze(-1).expand(B, N, 4).contiguous()

        # --- Build extra features from scatter slots ---
        hm = has[..., self._S_LRJ_M]
        h_d2 = has[..., self._S_D2]
        mass_raw = vals[..., self._S_LRJ_M]
        lrj_d2 = vals[..., self._S_D2]

        particle_extra = torch.stack(
            [
                vals[..., self._S_Z0] * has[..., self._S_Z0].float(),
                vals[..., self._S_D0] * has[..., self._S_D0].float(),
                vals[..., self._S_BTAG] * has[..., self._S_BTAG].float(),
                vals[..., self._S_JVT] * has[..., self._S_JVT].float(),
                torch.where(is_largeR & hm, mass_raw, torch.zeros_like(mass_raw)),
                vals[..., self._S_PVC] * has[..., self._S_PVC].float(),
                vals[..., self._S_TEC] * has[..., self._S_TEC].float(),
                vals[..., self._S_D0S] * has[..., self._S_D0S].float(),
                vals[..., self._S_TNK] * has[..., self._S_TNK].float(),
                vals[..., self._S_TRJ] * has[..., self._S_TRJ].float(),
                vals[..., self._S_PTE] * has[..., self._S_PTE].float(),
                vals[..., self._S_PPC] * has[..., self._S_PPC].float(),
                torch.where(is_largeR & h_d2, lrj_d2, torch.zeros_like(lrj_d2)),
            ],
            dim=-1,
        )

        result = {
            "kinematics": kinematics,
            "kinematics_mask": kin_mask,
            "particle_type": particle_type,
            "particle_charge": particle_charge,
            "particle_extra": particle_extra,
            "particle_mask": particle_mask,
            "met_vector": met_vector,
        }

        # Raw extra features for quality-cut filtering
        if has_raw:
            vals_r, has_r = self._scatter_all(
                features_raw, feature_idxs, particle_idxs, valid, B, N
            )
            result["particle_extra_raw"] = torch.stack(
                [
                    vals_r[..., self._S_Z0] * has_r[..., self._S_Z0].float(),
                    vals_r[..., self._S_D0] * has_r[..., self._S_D0].float(),
                    vals_r[..., self._S_BTAG] * has_r[..., self._S_BTAG].float(),
                    vals_r[..., self._S_JVT] * has_r[..., self._S_JVT].float(),
                    torch.where(
                        is_largeR & has_r[..., self._S_LRJ_M],
                        vals_r[..., self._S_LRJ_M],
                        torch.zeros(B, N),
                    ),
                    vals_r[..., self._S_PVC] * has_r[..., self._S_PVC].float(),
                    vals_r[..., self._S_TEC] * has_r[..., self._S_TEC].float(),
                    vals_r[..., self._S_D0S] * has_r[..., self._S_D0S].float(),
                    vals_r[..., self._S_TNK] * has_r[..., self._S_TNK].float(),
                    vals_r[..., self._S_TRJ] * has_r[..., self._S_TRJ].float(),
                    vals_r[..., self._S_PTE] * has_r[..., self._S_PTE].float(),
                    vals_r[..., self._S_PPC] * has_r[..., self._S_PPC].float(),
                    torch.where(
                        is_largeR & has_r[..., self._S_D2],
                        vals_r[..., self._S_D2],
                        torch.zeros(B, N),
                    ),
                ],
                dim=-1,
            )

        return result


# ======================================================================
# Geometric (sparse) collator — no padding, flat concatenation
# ======================================================================


def geometric_to_dense(
    x: torch.Tensor,
    ptr: torch.Tensor,
    fill_value: float = 0.0,
    fixed_N: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flat geometric tensor to padded dense (B, N_max, ...) format.

    Uses vectorized advanced indexing (no Python loop over B).

    Args:
        x:    (T_total, ...) flat tensor of particle features.
        ptr:  (B+1,) cumulative particle counts (int32).
        fill_value: value for padding slots.
        fixed_N: if set, always pad to this size (enables torch.compile
                 by making output shapes static across batches).

    Returns:
        out:  (B, N_max, ...) padded tensor.
        mask: (B, N_max) bool, True = real particle.
    """
    B = ptr.shape[0] - 1
    lengths = (ptr[1:] - ptr[:-1]).long()  # (B,)
    N_max = fixed_N if fixed_N is not None else int(lengths.max().item())

    # Build (B, N_max) index map into flat T_total tensor
    offsets = torch.arange(N_max, device=x.device).unsqueeze(0)  # (1, N_max)
    mask = offsets < lengths.unsqueeze(1)  # (B, N_max) bool
    starts = ptr[:-1].long().unsqueeze(1)  # (B, 1)
    flat_idx = (starts + offsets).clamp(max=x.shape[0] - 1)  # (B, N_max)
    flat_idx = flat_idx * mask.long()  # zero out padding positions

    # Gather into dense tensor
    out = x[flat_idx.reshape(-1)].reshape(B, N_max, *x.shape[1:])
    # Zero (or fill) padding slots
    mask_expand = mask.reshape(B, N_max, *([1] * (x.dim() - 1)))
    if fill_value == 0.0:
        out = out * mask_expand
    else:
        out = torch.where(mask_expand, out, torch.full_like(out, fill_value))

    return out, mask


class ParticleVectorCollatorGeometric:
    """Collate tokenized events into flat concatenated tensors (no padding).

    Internally reuses the vectorized ``scatter_add_`` from
    :class:`ParticleVectorCollator` to build a padded ``(B, N, 19)`` grid,
    then strips padding into flat ``(T_total, ...)`` tensors.  This keeps
    the collator as fast as the padded version.

    Output dictionary
    -----------------
    kinematics      : (T_total, 4)  float32  (px, py, pz, E)
    particle_type   : (T_total,)    int64    type codes
    particle_charge : (T_total,)    int64    shifted {0,1,2} ↔ {-1,0,+1}
    particle_extra  : (T_total, 13) float32  auxiliary features
    batch           : (T_total,)    int64    event index per particle
    ptr             : (B+1,)        int32    cumulative particle counts
    met_vector      : (B, 2)        float32  MET (px, py) per event
    """

    def __init__(self, max_particles: int = 30):
        # Delegate scatter to a padded collator instance
        self._padded = ParticleVectorCollator(max_particles=max_particles)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Step 1: run the vectorized padded collator
        padded = self._padded(batch)

        kin = padded["kinematics"]  # (B, N, 4)
        p_type = padded["particle_type"]  # (B, N)
        p_charge = padded["particle_charge"]
        p_extra = padded["particle_extra"]
        p_mask = padded["particle_mask"]  # (B, N) float, 1=real
        met_vector = padded["met_vector"]

        # Step 2: count real particles per event
        real = p_mask > 0.5  # (B, N) bool
        counts = real.sum(dim=1)  # (B,)

        # Step 3: extract real particles into flat tensors
        # Boolean indexing flattens (B, N) → (T_total,) in row-major order
        kinematics = kin[real]  # (T_total, 4)
        particle_type = p_type[real]  # (T_total,)
        particle_charge = p_charge[real]
        particle_extra = p_extra[real]  # (T_total, 13)

        # Step 4: build batch index and ptr
        batch_idx = torch.repeat_interleave(
            torch.arange(counts.shape[0], dtype=torch.long), counts.long()
        )
        ptr = torch.zeros(counts.shape[0] + 1, dtype=torch.int32)
        ptr[1:] = counts.to(torch.int32).cumsum(0)

        max_seqlen = int(counts.max().item()) if counts.shape[0] > 0 else 0

        return {
            "kinematics": kinematics,
            "particle_type": particle_type,
            "particle_charge": particle_charge,
            "particle_extra": particle_extra,
            "batch": batch_idx,
            "ptr": ptr,
            "max_seqlen": max_seqlen,
            "met_vector": met_vector,
        }
