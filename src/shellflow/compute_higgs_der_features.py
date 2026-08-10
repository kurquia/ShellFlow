"""
Compute Higgs-style DER_* (derived) features for 4-lepton events.

This module maps 4-lepton topology to H→ττ topology to enable transfer learning:
- Leading 2 leptons (by pT) → "lepton" and "tau" for DER_* calculations
- Jets and MET: Use actual values from event
- Compute the same DER_* features as Higgs Challenge dataset

Feature IDs 12-27 will be populated with DER_* features to match Higgs dataset.
"""

from typing import Dict

import awkward as ak
import numpy as np


def delta_phi(phi1: ak.Array, phi2: ak.Array) -> ak.Array:
    """
    Calculate delta phi in [-pi, pi].

    Args:
        phi1, phi2: Azimuthal angles in radians

    Returns:
        Delta phi wrapped to [-pi, pi]
    """
    dphi = phi1 - phi2
    dphi = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    return dphi


def delta_r(eta1: ak.Array, phi1: ak.Array, eta2: ak.Array, phi2: ak.Array) -> ak.Array:
    """
    Calculate ΔR = sqrt(Δη² + Δφ²).

    Args:
        eta1, phi1: Pseudorapidity and azimuthal angle of first particle
        eta2, phi2: Pseudorapidity and azimuthal angle of second particle

    Returns:
        ΔR separation
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def invariant_mass(
    pt1: ak.Array,
    eta1: ak.Array,
    phi1: ak.Array,
    e1: ak.Array,
    pt2: ak.Array,
    eta2: ak.Array,
    phi2: ak.Array,
    e2: ak.Array,
) -> ak.Array:
    """
    Calculate invariant mass of two particles.

    Args:
        pt, eta, phi, e: Transverse momentum, pseudorapidity, azimuthal angle, energy

    Returns:
        Invariant mass m = sqrt((E1+E2)² - (p1+p2)²)
    """
    # Convert to Cartesian coordinates
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)

    # Sum 4-vectors
    E = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    # m² = E² - p²
    m2 = E**2 - (px**2 + py**2 + pz**2)

    # Handle negative mass² (numerical errors)
    m = np.sqrt(np.maximum(m2, 0))
    return m


def transverse_mass(
    pt_lep: ak.Array, phi_lep: ak.Array, met: ak.Array, met_phi: ak.Array
) -> ak.Array:
    """
    Calculate transverse mass between lepton and MET.

    mT = sqrt(2 * pT_lep * MET * (1 - cos(Δφ)))

    Args:
        pt_lep: Lepton transverse momentum
        phi_lep: Lepton azimuthal angle
        met: Missing transverse energy
        met_phi: MET azimuthal angle

    Returns:
        Transverse mass
    """
    dphi = delta_phi(phi_lep, met_phi)
    mt = np.sqrt(2 * pt_lep * met * (1 - np.cos(dphi)))
    return mt


def compute_higgs_der_features(big_tree: ak.Array) -> Dict[str, ak.Array]:
    """
    Compute Higgs-style DER_* features from 4-lepton events.

    Mapping strategy:
    - Leading 2 leptons (by pT) → "lepton" and "tau" for DER_* calculations
    - Use actual jets if present
    - Use actual MET

    Args:
        big_tree: Awkward array with per-event particle collections

    Returns:
        Dictionary of DER_* features as awkward arrays (one value per event)
    """
    num_events = len(big_tree["eventNumber"])

    # Extract leptons and sort by pT
    lep_pt = big_tree["lep_pt"]
    lep_eta = big_tree["lep_eta"]
    lep_phi = big_tree["lep_phi"]
    lep_e = big_tree["lep_e"]

    # Sort leptons by pT (descending)
    lep_pt_argsort = ak.argsort(lep_pt, axis=1, ascending=False)
    lep_pt_sorted = lep_pt[lep_pt_argsort]
    lep_eta_sorted = lep_eta[lep_pt_argsort]
    lep_phi_sorted = lep_phi[lep_pt_argsort]
    lep_e_sorted = lep_e[lep_pt_argsort]

    # Leading 2 leptons (map to "lepton" and "tau" roles)
    lep1_pt = ak.fill_none(ak.pad_none(lep_pt_sorted, 1, axis=1)[:, 0], 0)
    lep1_eta = ak.fill_none(ak.pad_none(lep_eta_sorted, 1, axis=1)[:, 0], 0)
    lep1_phi = ak.fill_none(ak.pad_none(lep_phi_sorted, 1, axis=1)[:, 0], 0)
    lep1_e = ak.fill_none(ak.pad_none(lep_e_sorted, 1, axis=1)[:, 0], 0)

    lep2_pt = ak.fill_none(ak.pad_none(lep_pt_sorted, 2, axis=1)[:, 1], 0)
    lep2_eta = ak.fill_none(ak.pad_none(lep_eta_sorted, 2, axis=1)[:, 1], 0)
    lep2_phi = ak.fill_none(ak.pad_none(lep_phi_sorted, 2, axis=1)[:, 1], 0)
    lep2_e = ak.fill_none(ak.pad_none(lep_e_sorted, 2, axis=1)[:, 1], 0)

    # Extract MET
    met = big_tree["met"]
    met_phi = big_tree["met_phi"]
    # MET is typically a single value per event, ensure it's properly formatted
    try:
        # Check if MET has nested structure
        _ = ak.num(met, axis=1)
        # It's already structured, extract first value per event
        met = ak.fill_none(ak.firsts(met, axis=1), 0)
        met_phi = ak.fill_none(ak.firsts(met_phi, axis=1), 0)
    except Exception:
        # MET is flat (one value per event), use directly
        pass

    # Extract jets (may not be present in all events)
    if "jet_pt" in big_tree.fields:
        jet_pt = big_tree["jet_pt"]
        jet_eta = big_tree["jet_eta"]
        jet_phi = big_tree["jet_phi"]
        jet_e = big_tree["jet_e"]
    else:
        # No jets in this dataset, create empty arrays
        jet_pt = ak.Array([[]] * num_events)
        jet_eta = ak.Array([[]] * num_events)
        jet_phi = ak.Array([[]] * num_events)
        jet_e = ak.Array([[]] * num_events)

    # Sort jets by pT (descending)
    jet_pt_argsort = ak.argsort(jet_pt, axis=1, ascending=False)
    jet_pt_sorted = jet_pt[jet_pt_argsort]
    jet_eta_sorted = jet_eta[jet_pt_argsort]
    jet_phi_sorted = jet_phi[jet_pt_argsort]
    jet_e_sorted = jet_e[jet_pt_argsort]

    # Leading 2 jets
    jet1_pt = ak.fill_none(ak.pad_none(jet_pt_sorted, 1, axis=1)[:, 0], 0)
    jet1_eta = ak.fill_none(ak.pad_none(jet_eta_sorted, 1, axis=1)[:, 0], 0)
    jet1_phi = ak.fill_none(ak.pad_none(jet_phi_sorted, 1, axis=1)[:, 0], 0)
    jet1_e = ak.fill_none(ak.pad_none(jet_e_sorted, 1, axis=1)[:, 0], 0)

    jet2_pt = ak.fill_none(ak.pad_none(jet_pt_sorted, 2, axis=1)[:, 1], 0)
    jet2_eta = ak.fill_none(ak.pad_none(jet_eta_sorted, 2, axis=1)[:, 1], 0)
    jet2_phi = ak.fill_none(ak.pad_none(jet_phi_sorted, 2, axis=1)[:, 1], 0)
    jet2_e = ak.fill_none(ak.pad_none(jet_e_sorted, 2, axis=1)[:, 1], 0)

    # Count jets per event
    n_jets = ak.num(jet_pt, axis=1)

    # ==================== Compute DER_* features ====================

    # DER_mass_MMC (feature 15): Invariant mass of leading 2 leptons
    # (Can't compute real MMC without tau decay info, use di-lepton mass instead)
    DER_mass_MMC = invariant_mass(
        lep1_pt, lep1_eta, lep1_phi, lep1_e, lep2_pt, lep2_eta, lep2_phi, lep2_e
    )

    # DER_mass_transverse_met_lep (feature 16): Transverse mass(MET, leading lepton)
    DER_mass_transverse_met_lep = transverse_mass(lep1_pt, lep1_phi, met, met_phi)

    # DER_mass_vis (feature 17): Invariant mass of leading 2 leptons
    DER_mass_vis = DER_mass_MMC  # Same as MMC for our mapping

    # DER_pt_h (feature 18): |pT_lep1 + pT_lep2 + pT_MET|
    px_h = (
        lep1_pt * np.cos(lep1_phi) + lep2_pt * np.cos(lep2_phi) + met * np.cos(met_phi)
    )
    py_h = (
        lep1_pt * np.sin(lep1_phi) + lep2_pt * np.sin(lep2_phi) + met * np.sin(met_phi)
    )
    DER_pt_h = np.sqrt(px_h**2 + py_h**2)

    # DER_deltaeta_jet_jet (feature 19): |eta_jet1 - eta_jet2| (undefined if n_jets <= 1)
    DER_deltaeta_jet_jet = np.abs(jet1_eta - jet2_eta)
    DER_deltaeta_jet_jet = ak.where(n_jets <= 1, -999.0, DER_deltaeta_jet_jet)

    # DER_mass_jet_jet (feature 20): Invariant mass(jet1, jet2) (undefined if n_jets <= 1)
    DER_mass_jet_jet = invariant_mass(
        jet1_pt, jet1_eta, jet1_phi, jet1_e, jet2_pt, jet2_eta, jet2_phi, jet2_e
    )
    DER_mass_jet_jet = ak.where(n_jets <= 1, -999.0, DER_mass_jet_jet)

    # DER_prodeta_jet_jet (feature 21): eta_jet1 * eta_jet2 (undefined if n_jets <= 1)
    DER_prodeta_jet_jet = jet1_eta * jet2_eta
    DER_prodeta_jet_jet = ak.where(n_jets <= 1, -999.0, DER_prodeta_jet_jet)

    # DER_deltar_tau_lep (feature 22): ΔR(lep1, lep2)
    DER_deltar_tau_lep = delta_r(lep1_eta, lep1_phi, lep2_eta, lep2_phi)

    # DER_pt_tot (feature 23): |pT_lep1 + pT_lep2 + pT_MET + pT_jet1 + pT_jet2|
    # NOTE: Only includes leading 2 jets (jet1, jet2), NOT all jets per PDF definition
    jet1_px = jet1_pt * np.cos(jet1_phi)
    jet1_py = jet1_pt * np.sin(jet1_phi)
    jet2_px = jet2_pt * np.cos(jet2_phi)
    jet2_py = jet2_pt * np.sin(jet2_phi)

    # px_h already contains lep1, lep2, met
    px_tot = px_h + jet1_px + jet2_px
    py_tot = py_h + jet1_py + jet2_py
    DER_pt_tot = np.sqrt(px_tot**2 + py_tot**2)

    # DER_sum_pt (feature 24): Sum of pT magnitudes
    DER_sum_pt = lep1_pt + lep2_pt + ak.sum(jet_pt_sorted, axis=1)

    # DER_pt_ratio_lep_tau (feature 25): pT_lep1 / pT_lep2
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        DER_pt_ratio_lep_tau = ak.where(lep2_pt > 0, lep1_pt / lep2_pt, 0)

    # DER_met_phi_centrality (feature 26): MET phi centrality using exact PDF formula
    # A = sin(φ_met - φ_lep) * sign(sin(φ_had - φ_lep))
    # B = sin(φ_had - φ_met) * sign(sin(φ_had - φ_lep))
    # C = (A + B) / sqrt(A² + B²)
    # Mapping: φ_lep → lep1_phi, φ_had → lep2_phi
    dphi_met_lep1 = delta_phi(met_phi, lep1_phi)
    dphi_lep2_met = delta_phi(lep2_phi, met_phi)
    dphi_lep2_lep1 = delta_phi(lep2_phi, lep1_phi)

    sign_term = np.sign(np.sin(dphi_lep2_lep1))

    A = np.sin(dphi_met_lep1) * sign_term
    B = np.sin(dphi_lep2_met) * sign_term

    # Avoid division by zero if A and B are both 0
    denom = np.sqrt(A**2 + B**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        DER_met_phi_centrality = ak.where(denom > 0, (A + B) / denom, 0)

    # DER_lep_eta_centrality (feature 27): Lepton eta centrality using exponential PDF formula
    # C = exp[-4 * (η_lep - (η_1 + η_2)/2)² / (η_1 - η_2)²]
    # Mapping: η_lep → lep1_eta, η_1 → jet1_eta, η_2 → jet2_eta
    # Set to -999.0 if n_jets <= 1
    eta_jet_diff_sq = (jet1_eta - jet2_eta) ** 2
    eta_lep_centered_sq = (lep1_eta - (jet1_eta + jet2_eta) / 2) ** 2

    # Exponent term: -4 * numerator / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = ak.where(
            eta_jet_diff_sq > 0, (-4 * eta_lep_centered_sq) / eta_jet_diff_sq, 0
        )  # Set exponent to 0 if jets overlap (avoid div by zero)

    centrality = np.exp(exponent)

    # Set to -999.0 if n_jets <= 1, as per PDF definition
    DER_lep_eta_centrality = ak.where(n_jets <= 1, -999.0, centrality)

    # ==================== Additional Higgs-specific features ====================
    # These correspond to PRI_* features in Higgs dataset but we compute from our data

    # PRI_met_sumet (feature 12): Total transverse energy (approximate as sum of all pT)
    # For 4-lepton, this is sum of lepton pT + jet pT + MET
    PRI_met_sumet = DER_sum_pt + met

    # PRI_jet_num (feature 13): Number of jets
    PRI_jet_num = ak.fill_none(n_jets, 0)

    # PRI_jet_all_pt (feature 14): Scalar sum of all jet pT
    PRI_jet_all_pt = ak.sum(jet_pt_sorted, axis=1)

    # Return dictionary of features with their feature IDs as keys
    der_features = {
        "PRI_met_sumet": PRI_met_sumet,  # feature_id: 12
        "PRI_jet_num": PRI_jet_num,  # feature_id: 13
        "PRI_jet_all_pt": PRI_jet_all_pt,  # feature_id: 14
        "DER_mass_MMC": DER_mass_MMC,  # feature_id: 15
        "DER_mass_transverse_met_lep": DER_mass_transverse_met_lep,  # feature_id: 16
        "DER_mass_vis": DER_mass_vis,  # feature_id: 17
        "DER_pt_h": DER_pt_h,  # feature_id: 18
        "DER_deltaeta_jet_jet": DER_deltaeta_jet_jet,  # feature_id: 19
        "DER_mass_jet_jet": DER_mass_jet_jet,  # feature_id: 20
        "DER_prodeta_jet_jet": DER_prodeta_jet_jet,  # feature_id: 21
        "DER_deltar_tau_lep": DER_deltar_tau_lep,  # feature_id: 22
        "DER_pt_tot": DER_pt_tot,  # feature_id: 23
        "DER_sum_pt": DER_sum_pt,  # feature_id: 24
        "DER_pt_ratio_lep_tau": DER_pt_ratio_lep_tau,  # feature_id: 25
        "DER_met_phi_centrality": DER_met_phi_centrality,  # feature_id: 26
        "DER_lep_eta_centrality": DER_lep_eta_centrality,  # feature_id: 27
    }

    return der_features


def add_der_features_to_tree(big_tree: ak.Array) -> ak.Array:
    """
    Compute DER_* features and add them to the awkward array.

    Args:
        big_tree: Input awkward array with per-event particle collections

    Returns:
        Modified awkward array with DER_* features added as new fields
    """
    der_features = compute_higgs_der_features(big_tree)

    # Add each feature as a new field to big_tree
    for feature_name, feature_values in der_features.items():
        big_tree = ak.with_field(big_tree, feature_values, feature_name)

    return big_tree
