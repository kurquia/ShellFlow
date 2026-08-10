"""Conservation law metrics for tracking during training.

This module provides functions to calculate various conservation laws
in particle physics data, which can be used as metrics during model training
to ensure physical plausibility of generated/reconstructed events.
"""

from typing import Dict, Tuple

import numpy as np

# Feature indices for the dense (n_events, max_particles, n_features)
# layout these helpers operate on (not the packed transformer_input format)
PT_IDX = 0
ETA_IDX = 1
PHI_IDX = 2
ENERGY_IDX = 3
PARTICLE_TYPE_IDX = 4
CHARGE_IDX = 5
LEP_TYPE_IDX = 6

# Particle type encodings
LEPTON_TYPE = 11
TAU_TYPE = 15
JET_TYPE = 1
PHOTON_TYPE = 22
MET_TYPE = 0

# Lepton flavor encodings
ELECTRON = 11
MUON = 13
TAU_LEPTON = 15


def get_px_py(pt: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pt and phi to px, py momentum components.

    Args:
        pt: Transverse momentum array
        phi: Azimuthal angle array

    Returns:
        Tuple of (px, py) arrays
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    return px, py


def calculate_met(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Calculate missing transverse energy (MET) magnitude.

    Args:
        sequences: Array of shape (n_events, max_particles, n_features)
        masks: Boolean mask of shape (n_events, max_particles)

    Returns:
        MET magnitude for each event, shape (n_events,)
    """
    pt = sequences[:, :, PT_IDX]
    phi = sequences[:, :, PHI_IDX]

    px, py = get_px_py(pt, phi)

    # Apply mask to exclude padded particles
    px_masked = np.where(masks, px, 0)
    py_masked = np.where(masks, py, 0)

    # Sum over particles
    total_px = px_masked.sum(axis=1)
    total_py = py_masked.sum(axis=1)

    # MET magnitude
    met = np.sqrt(total_px**2 + total_py**2)
    return met


def calculate_total_charge(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Calculate total electric charge per event.

    Args:
        sequences: Array of shape (n_events, max_particles, n_features)
        masks: Boolean mask of shape (n_events, max_particles)

    Returns:
        Total charge for each event, shape (n_events,)
    """
    charge = sequences[:, :, CHARGE_IDX]
    charge_masked = np.where(masks, charge, 0)
    total_charge = charge_masked.sum(axis=1)
    return total_charge


def calculate_total_energy(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Calculate total visible energy per event.

    Args:
        sequences: Array of shape (n_events, max_particles, n_features)
        masks: Boolean mask of shape (n_events, max_particles)

    Returns:
        Total visible energy for each event, shape (n_events,)
    """
    energy = sequences[:, :, ENERGY_IDX]
    energy_masked = np.where(masks, energy, 0)
    total_energy = energy_masked.sum(axis=1)
    return total_energy


def calculate_lepton_number(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Calculate lepton number (n_leptons - n_antileptons).

    Args:
        sequences: Array of shape (n_events, max_particles, n_features)
        masks: Boolean mask of shape (n_events, max_particles)

    Returns:
        Lepton number for each event, shape (n_events,)
    """
    # Get lepton mask
    particle_types = sequences[:, :, PARTICLE_TYPE_IDX]
    lepton_mask = (particle_types == LEPTON_TYPE) & masks.astype(bool)

    # Get charges of leptons only
    lepton_charges = sequences[:, :, CHARGE_IDX]
    lepton_charges_masked = np.where(lepton_mask, lepton_charges, 0)

    # Count leptons (negative) and antileptons (positive)
    n_leptons = (lepton_charges_masked < -0.1).sum(axis=1)
    n_antileptons = (lepton_charges_masked > 0.1).sum(axis=1)

    return n_leptons - n_antileptons


def calculate_lepton_flavor_numbers(
    sequences: np.ndarray, masks: np.ndarray
) -> Dict[str, np.ndarray]:
    """Calculate lepton flavor numbers for each flavor.

    Args:
        sequences: Array of shape (n_events, max_particles, n_features)
        masks: Boolean mask of shape (n_events, max_particles)

    Returns:
        Dictionary with keys 'electron', 'muon', 'tau' and flavor numbers
    """
    # Get lepton mask
    particle_types = sequences[:, :, PARTICLE_TYPE_IDX]
    lepton_mask = (particle_types == LEPTON_TYPE) & masks.astype(bool)

    lep_types = sequences[:, :, LEP_TYPE_IDX]
    lep_charges = sequences[:, :, CHARGE_IDX]

    def get_flavor_number(flavor_type: int) -> np.ndarray:
        flavor_mask = lepton_mask & (lep_types == flavor_type)
        charges_masked = np.where(flavor_mask, lep_charges, 0)
        n_particles = (charges_masked < -0.1).sum(axis=1)
        n_antiparticles = (charges_masked > 0.1).sum(axis=1)
        return n_particles - n_antiparticles

    return {
        "electron": get_flavor_number(ELECTRON),
        "muon": get_flavor_number(MUON),
        "tau": get_flavor_number(TAU_LEPTON),
    }


def calculate_all_conservation_metrics(
    sequences: np.ndarray, masks: np.ndarray
) -> Dict[str, np.ndarray]:
    """Calculate all conservation law metrics.

    Args:
        sequences: Array of shape (n_events, max_particles, n_features)
        masks: Boolean mask of shape (n_events, max_particles)

    Returns:
        Dictionary with all conservation metrics
    """
    metrics = {
        "met": calculate_met(sequences, masks),
        "total_charge": calculate_total_charge(sequences, masks),
        "total_energy": calculate_total_energy(sequences, masks),
        "lepton_number": calculate_lepton_number(sequences, masks),
    }

    # Add flavor numbers
    flavor_numbers = calculate_lepton_flavor_numbers(sequences, masks)
    for flavor, numbers in flavor_numbers.items():
        metrics[f"{flavor}_number"] = numbers

    return metrics


def get_conservation_summary_stats(
    metrics: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Calculate summary statistics for conservation metrics.

    Args:
        metrics: Dictionary of conservation metrics from calculate_all_conservation_metrics

    Returns:
        Dictionary with summary statistics for each metric
    """
    summary = {}

    for name, values in metrics.items():
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "mean_abs": float(np.mean(np.abs(values))),
            "fraction_nonzero": float(np.mean(np.abs(values) > 0.1)),
        }

    return summary


def check_conservation_violations(
    metrics: Dict[str, np.ndarray], tolerance: float = 0.1
) -> Dict[str, float]:
    """Check fraction of events violating conservation laws.

    Args:
        metrics: Dictionary of conservation metrics
        tolerance: Tolerance for considering a value as zero

    Returns:
        Dictionary with violation fractions for each conservation law
    """
    violations = {}

    # Charge conservation: should be exactly 0
    violations["charge"] = float(np.mean(np.abs(metrics["total_charge"]) > tolerance))

    # Lepton number conservation: should be exactly 0
    violations["lepton_number"] = float(
        np.mean(np.abs(metrics["lepton_number"]) > tolerance)
    )

    # Flavor conservation: each should be 0
    for flavor in ["electron", "muon", "tau"]:
        key = f"{flavor}_number"
        violations[key] = float(np.mean(np.abs(metrics[key]) > tolerance))

    return violations
