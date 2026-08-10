"""Physics-metric helpers shared by sample-plot callbacks.

Stateless conversions from generated/real event kinematics (px, py, pz, E)
to physics observables: SFOS invariant mass, Delta R, transverse mass, HT,
dilepton-MET Delta phi, lepton pT spectra. Extracted from the retired
vector-diffusion callback so the RFM callback owns its dependencies.
"""

from itertools import combinations

import numpy as np
import torch

_TYPE_ELECTRON = 11
_TYPE_MUON = 13


class PhysicsMetricsHelper:
    """Stateless helper — all methods depend only on their arguments."""

    @staticmethod
    def _cartesian_to_physics(px, py, pz):
        """Convert (px, py, pz) arrays to (pT, η, φ)."""
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        # η = arctanh(pz / |p|), clamped for safety
        ratio = np.clip(pz / np.maximum(p, 1e-8), -1 + 1e-7, 1 - 1e-7)
        eta = np.arctanh(ratio)
        phi = np.arctan2(py, px)
        return pt, eta, phi

    # ------------------------------------------------------------------
    # Extract lepton kinematics
    # ------------------------------------------------------------------
    def _extract_lepton_kinematics(self, kin, p_type, p_mask):
        """Extract (pT, η, φ) for leptons (e + μ).

        Args:
            kin: (B, N, 4) raw physical kinematics
            p_type: (B, N) particle type codes
            p_mask: (B, N) particle mask

        Returns:
            pt, eta, phi: 1D numpy arrays of lepton values
        """
        is_lepton = ((p_type == _TYPE_ELECTRON) | (p_type == _TYPE_MUON)) & (
            p_mask > 0.5
        )
        px = kin[..., 0][is_lepton].numpy()
        py = kin[..., 1][is_lepton].numpy()
        pz = kin[..., 2][is_lepton].numpy()
        pt, eta, phi = self._cartesian_to_physics(px, py, pz)
        return pt, eta, phi

    # ------------------------------------------------------------------
    # Per-event lepton extraction (shared by all pair-based metrics)
    # ------------------------------------------------------------------
    def _get_leptons_per_event(self, kin, p_type, p_charge, p_mask):
        """Extract all lepton data per event in one pass.

        Args:
            kin: (B, N, 4) raw physical kinematics (px, py, pz, E)
            p_type: (B, N) particle type codes
            p_charge: (B, N) particle charge ({0,1,2} ↔ {-1,0,+1})
            p_mask: (B, N) particle mask

        Returns:
            list[dict | None]: Per-event dict with keys
                {px, py, pz, E, pt, eta, phi, charge, flavor} as numpy arrays,
                or None if the event has no leptons.
        """
        B, N, _ = kin.shape
        is_lepton = ((p_type == _TYPE_ELECTRON) | (p_type == _TYPE_MUON)) & (
            p_mask > 0.5
        )

        # Flatten all leptons, compute physics once
        all_px = kin[..., 0][is_lepton].numpy()
        all_py = kin[..., 1][is_lepton].numpy()
        all_pz = kin[..., 2][is_lepton].numpy()
        all_E = kin[..., 3][is_lepton].numpy()
        all_pt, all_eta, all_phi = self._cartesian_to_physics(all_px, all_py, all_pz)
        all_charge = (p_charge[is_lepton].numpy() - 1).astype(
            int
        )  # {0,1,2} → {-1,0,+1}
        all_flavor = p_type[is_lepton].numpy()

        # Split per event using counts
        counts = is_lepton.sum(dim=1).tolist()
        result = []
        offset = 0
        for c in counts:
            if c == 0:
                result.append(None)
            else:
                s = slice(offset, offset + c)
                result.append(
                    {
                        "px": all_px[s],
                        "py": all_py[s],
                        "pz": all_pz[s],
                        "E": all_E[s],
                        "pt": all_pt[s],
                        "eta": all_eta[s],
                        "phi": all_phi[s],
                        "charge": all_charge[s],
                        "flavor": all_flavor[s],
                    }
                )
            offset += c
        return result

    # ------------------------------------------------------------------
    # Best OSSF pair (closest to Z mass)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_best_ossf_pair(lep_dict):
        """Find the OSSF lepton pair closest to the Z mass (91.1876 GeV).

        Uses 4-vector invariant mass: m² = (E₁+E₂)² - |p₁+p₂|²

        Args:
            lep_dict: dict with keys {px, py, pz, E, charge, flavor, ...}

        Returns:
            (idx_i, idx_j, mass) or None if no valid pair exists.
        """
        n = len(lep_dict["px"])
        charge = lep_dict["charge"]
        flavor = lep_dict["flavor"]
        Z_MASS = 91.1876
        best = None
        best_diff = np.inf

        for i in range(n):
            if charge[i] == 0:
                continue
            for j in range(i + 1, n):
                if charge[j] == 0:
                    continue
                if flavor[i] != flavor[j]:
                    continue
                if charge[i] * charge[j] >= 0:
                    continue
                # 4-vector invariant mass
                e_sum = lep_dict["E"][i] + lep_dict["E"][j]
                px_sum = lep_dict["px"][i] + lep_dict["px"][j]
                py_sum = lep_dict["py"][i] + lep_dict["py"][j]
                pz_sum = lep_dict["pz"][i] + lep_dict["pz"][j]
                m2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
                if m2 <= 0:
                    continue
                mass = np.sqrt(m2)
                diff = abs(mass - Z_MASS)
                if diff < best_diff:
                    best_diff = diff
                    best = (i, j, mass)

        return best

    # ------------------------------------------------------------------
    # SFOS invariant mass
    # ------------------------------------------------------------------
    def _compute_sfos_invariant_mass(self, kin, p_type, p_charge, p_mask):
        """Compute invariant mass of SFOS lepton pairs directly from 4-vectors.

        m² = (E₁+E₂)² - (px₁+px₂)² - (py₁+py₂)² - (pz₁+pz₂)²
        Using E from the 4th component of kinematics (px, py, pz, E).
        """
        B, N, _ = kin.shape
        all_masses = []

        for b in range(B):
            # Find leptons in this event
            lep_mask = ((p_type[b] == _TYPE_ELECTRON) | (p_type[b] == _TYPE_MUON)) & (
                p_mask[b] > 0.5
            )
            lep_indices = torch.where(lep_mask)[0]
            if len(lep_indices) < 2:
                continue

            for j, k in combinations(range(len(lep_indices)), 2):
                ij, ik = lep_indices[j].item(), lep_indices[k].item()

                # Same-flavor check
                if p_type[b, ij] != p_type[b, ik]:
                    continue

                # Opposite-sign check: charge stored as {0,1,2} ↔ {-1,0,+1}
                ch_j = p_charge[b, ij].item() - 1
                ch_k = p_charge[b, ik].item() - 1
                if ch_j * ch_k >= 0:
                    continue

                # 4-vectors (px, py, pz, E)
                px1, py1, pz1, e1 = (kin[b, ij, c].item() for c in range(4))
                px2, py2, pz2, e2 = (kin[b, ik, c].item() for c in range(4))

                m2 = (
                    (e1 + e2) ** 2
                    - (px1 + px2) ** 2
                    - (py1 + py2) ** 2
                    - (pz1 + pz2) ** 2
                )
                all_masses.append(np.sqrt(max(m2, 0.0)))

        return np.array(all_masses) if all_masses else np.array([])

    # ------------------------------------------------------------------
    # Physics metrics
    # ------------------------------------------------------------------
    def _compute_delta_r_ossf(
        self, kin, p_type, p_charge, p_mask, leptons_per_event=None
    ):
        """Compute Delta R between all OSSF lepton pairs.

        DR = sqrt(deta^2 + dphi^2), combining electrons and muons.
        """
        if leptons_per_event is None:
            leptons_per_event = self._get_leptons_per_event(
                kin, p_type, p_charge, p_mask
            )

        all_dr = []
        for lep in leptons_per_event:
            if lep is None or len(lep["pt"]) < 2:
                continue
            n = len(lep["pt"])
            charge, flavor = lep["charge"], lep["flavor"]
            eta, phi = lep["eta"], lep["phi"]

            for i in range(n):
                if charge[i] == 0:
                    continue
                for j in range(i + 1, n):
                    if charge[j] == 0:
                        continue
                    if flavor[i] != flavor[j]:
                        continue
                    if charge[i] * charge[j] >= 0:
                        continue
                    deta = eta[i] - eta[j]
                    dphi = phi[i] - phi[j]
                    dphi = ((dphi + np.pi) % (2 * np.pi)) - np.pi
                    all_dr.append(np.sqrt(deta**2 + dphi**2))

        return np.array(all_dr) if all_dr else np.array([])

    def _compute_transverse_mass(
        self, kin, p_type, p_charge, p_mask, met_vector, leptons_per_event=None
    ):
        """Compute transverse mass mT for the non-Z lepton and MET.

        mT = sqrt(2 * pT_lep * MET * (1 - cos(dphi)))
        Requires >= 3 leptons + MET.

        Args:
            met_vector: (B, 2) tensor with (MET_px, MET_py)
        """
        if met_vector is None:
            return np.array([])
        if leptons_per_event is None:
            leptons_per_event = self._get_leptons_per_event(
                kin, p_type, p_charge, p_mask
            )

        met_np = (
            met_vector.numpy() if isinstance(met_vector, torch.Tensor) else met_vector
        )
        all_mt = []

        for b, lep in enumerate(leptons_per_event):
            if lep is None or len(lep["pt"]) < 3:
                continue

            best = self._find_best_ossf_pair(lep)
            if best is None:
                continue
            idx_i, idx_j, _ = best

            remaining = [k for k in range(len(lep["pt"])) if k != idx_i and k != idx_j]
            if not remaining:
                continue

            best_k = max(remaining, key=lambda k: lep["pt"][k])
            lep_pt = lep["pt"][best_k]
            lep_phi = lep["phi"][best_k]

            met_px, met_py = met_np[b, 0], met_np[b, 1]
            met_pt = np.sqrt(met_px**2 + met_py**2)
            met_phi = np.arctan2(met_py, met_px)

            dphi = lep_phi - met_phi
            dphi = ((dphi + np.pi) % (2 * np.pi)) - np.pi
            mt2 = 2 * lep_pt * met_pt * (1 - np.cos(dphi))
            if mt2 > 0:
                all_mt.append(np.sqrt(mt2))

        return np.array(all_mt) if all_mt else np.array([])

    def _compute_ht(self, kin, p_type, p_mask):
        """Compute HT (scalar sum of pT) per event. Fully vectorized.

        Sums pT = sqrt(px^2 + py^2) for all non-MET particles.
        """
        # MET (type 0) is excluded; include all other valid particles
        valid = (p_mask > 0.5) & (p_type != 0)
        px = kin[..., 0]
        py = kin[..., 1]
        pt = torch.sqrt(px**2 + py**2)
        pt_masked = pt * valid.float()
        ht = pt_masked.sum(dim=1).numpy()
        return ht

    def _compute_delta_phi_dilepton_met(
        self, kin, p_type, p_charge, p_mask, met_vector, leptons_per_event=None
    ):
        """Compute Delta phi between the OSSF dilepton system and MET.

        Dilepton phi from Cartesian vector sum of best OSSF pair's px, py.

        Args:
            met_vector: (B, 2) tensor with (MET_px, MET_py)
        """
        if met_vector is None:
            return np.array([])
        if leptons_per_event is None:
            leptons_per_event = self._get_leptons_per_event(
                kin, p_type, p_charge, p_mask
            )

        met_np = (
            met_vector.numpy() if isinstance(met_vector, torch.Tensor) else met_vector
        )
        all_dphi = []

        for b, lep in enumerate(leptons_per_event):
            if lep is None or len(lep["pt"]) < 2:
                continue

            best = self._find_best_ossf_pair(lep)
            if best is None:
                continue
            idx_i, idx_j, _ = best

            # Dilepton system phi from Cartesian vector sum
            px_sum = lep["px"][idx_i] + lep["px"][idx_j]
            py_sum = lep["py"][idx_i] + lep["py"][idx_j]
            dilepton_phi = np.arctan2(py_sum, px_sum)

            met_phi = np.arctan2(met_np[b, 1], met_np[b, 0])
            dphi = dilepton_phi - met_phi
            dphi = ((dphi + np.pi) % (2 * np.pi)) - np.pi
            all_dphi.append(dphi)

        return np.array(all_dphi) if all_dphi else np.array([])

    def _compute_leading_subleading_lepton_pt(self, kin, p_type, p_mask):
        """Compute leading and subleading lepton pT. Fully vectorized.

        Returns:
            (leading_pt, subleading_pt): 1D numpy arrays
        """
        is_lepton = ((p_type == _TYPE_ELECTRON) | (p_type == _TYPE_MUON)) & (
            p_mask > 0.5
        )
        px = kin[..., 0]
        py = kin[..., 1]
        pt = torch.sqrt(px**2 + py**2)

        # Mask non-leptons to -inf so they sort to the end
        pt_masked = pt.clone()
        pt_masked[~is_lepton] = float("-inf")
        sorted_pt, _ = torch.sort(pt_masked, dim=1, descending=True)

        # Events with at least 2 leptons
        n_leptons = is_lepton.sum(dim=1)
        has_2 = n_leptons >= 2

        leading = sorted_pt[has_2, 0].numpy()
        subleading = sorted_pt[has_2, 1].numpy()
        return leading, subleading
