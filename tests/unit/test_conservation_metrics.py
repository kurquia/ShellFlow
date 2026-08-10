"""Unit tests for conservation law metrics.

Tests conservation calculations using mock events with known properties.
"""

import numpy as np
import pytest

from shellflow.training.conservation_metrics import (
    ELECTRON,
    LEPTON_TYPE,
    MUON,
    PHI_IDX,
    PT_IDX,
    calculate_all_conservation_metrics,
    calculate_lepton_flavor_numbers,
    calculate_lepton_number,
    calculate_met,
    calculate_total_charge,
    calculate_total_energy,
    check_conservation_violations,
    get_conservation_summary_stats,
)


def create_mock_lepton(pt, eta, phi, energy, charge, lep_type):
    """Create a mock lepton with specified properties.

    Returns array: [pt, eta, phi, energy, particle_type=11, charge, lep_type]
    """
    return np.array([pt, eta, phi, energy, LEPTON_TYPE, charge, lep_type])


class TestBasicConservation:
    """Test basic conservation calculations."""

    def test_perfect_charge_conservation(self):
        """Test event with perfect charge conservation: e+ e- pair."""
        # Create e+ e- pair back-to-back (phi differs by π)
        sequences = np.zeros((1, 2, 7))  # 1 event, 2 particles, 7 features
        masks = np.ones((1, 2), dtype=bool)

        # e+ at phi=0
        sequences[0, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=1, lep_type=ELECTRON
        )
        # e- at phi=π
        sequences[0, 1] = create_mock_lepton(
            pt=50, eta=0, phi=np.pi, energy=50, charge=-1, lep_type=ELECTRON
        )

        total_charge = calculate_total_charge(sequences, masks)
        assert abs(total_charge[0]) < 1e-10, "Charge should be exactly 0"

    def test_charge_violation(self):
        """Test event with charge violation: 2 electrons."""
        sequences = np.zeros((1, 2, 7))
        masks = np.ones((1, 2), dtype=bool)

        # Two electrons (both charge -1)
        sequences[0, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )
        sequences[0, 1] = create_mock_lepton(
            pt=50, eta=0, phi=np.pi, energy=50, charge=-1, lep_type=ELECTRON
        )

        total_charge = calculate_total_charge(sequences, masks)
        assert total_charge[0] == -2, "Total charge should be -2"

    def test_met_calculation_balanced(self):
        """Test MET for balanced event (back-to-back particles)."""
        sequences = np.zeros((1, 2, 7))
        masks = np.ones((1, 2), dtype=bool)

        # Particle at phi=0
        sequences[0, 0, PT_IDX] = 100
        sequences[0, 0, PHI_IDX] = 0

        # Particle at phi=π (opposite direction)
        sequences[0, 1, PT_IDX] = 100
        sequences[0, 1, PHI_IDX] = np.pi

        met = calculate_met(sequences, masks)
        assert abs(met[0]) < 0.01, f"MET should be ~0 for back-to-back, got {met[0]}"

    def test_met_calculation_imbalanced(self):
        """Test MET for imbalanced event."""
        sequences = np.zeros((1, 1, 7))
        masks = np.ones((1, 1), dtype=bool)

        # Single particle at phi=0 with pt=100
        sequences[0, 0, PT_IDX] = 100
        sequences[0, 0, PHI_IDX] = 0

        met = calculate_met(sequences, masks)
        assert abs(met[0] - 100) < 0.01, f"MET should be 100 GeV, got {met[0]}"


class TestLeptonNumber:
    """Test lepton number conservation."""

    def test_lepton_number_conserved(self):
        """Test event with L=0: 2 leptons, 2 antileptons."""
        sequences = np.zeros((1, 4, 7))
        masks = np.ones((1, 4), dtype=bool)

        # 2 electrons (charge -1)
        for i in range(2):
            sequences[0, i] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
            )

        # 2 positrons (charge +1)
        for i in range(2, 4):
            sequences[0, i] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
            )

        L = calculate_lepton_number(sequences, masks)
        assert L[0] == 0, "Lepton number should be 0"

    def test_lepton_number_violated(self):
        """Test event with L≠0: 3 leptons, 1 antilepton."""
        sequences = np.zeros((1, 4, 7))
        masks = np.ones((1, 4), dtype=bool)

        # 3 electrons (charge -1)
        for i in range(3):
            sequences[0, i] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
            )

        # 1 positron (charge +1)
        sequences[0, 3] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
        )

        L = calculate_lepton_number(sequences, masks)
        assert L[0] == 2, "Lepton number should be 2"


class TestFlavorConservation:
    """Test lepton flavor conservation."""

    def test_electron_flavor_conserved(self):
        """Test electron flavor conservation."""
        sequences = np.zeros((1, 4, 7))
        masks = np.ones((1, 4), dtype=bool)

        # 2 e-
        for i in range(2):
            sequences[0, i] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
            )

        # 2 e+
        for i in range(2, 4):
            sequences[0, i] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
            )

        flavor_numbers = calculate_lepton_flavor_numbers(sequences, masks)
        assert flavor_numbers["electron"][0] == 0, "Electron number should be 0"
        assert flavor_numbers["muon"][0] == 0, "Muon number should be 0"
        assert flavor_numbers["tau"][0] == 0, "Tau number should be 0"

    def test_mixed_flavors_conserved(self):
        """Test event with mixed flavors, all conserved."""
        sequences = np.zeros((1, 4, 7))
        masks = np.ones((1, 4), dtype=bool)

        # e-, e+
        sequences[0, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )
        sequences[0, 1] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
        )

        # μ-, μ+
        sequences[0, 2] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=MUON
        )
        sequences[0, 3] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=MUON
        )

        flavor_numbers = calculate_lepton_flavor_numbers(sequences, masks)
        assert flavor_numbers["electron"][0] == 0, "Electron number should be 0"
        assert flavor_numbers["muon"][0] == 0, "Muon number should be 0"

    def test_flavor_violated(self):
        """Test lepton flavor violation."""
        sequences = np.zeros((1, 4, 7))
        masks = np.ones((1, 4), dtype=bool)

        # 3 e-, 1 μ+
        for i in range(3):
            sequences[0, i] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
            )
        sequences[0, 3] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=MUON
        )

        flavor_numbers = calculate_lepton_flavor_numbers(sequences, masks)
        assert flavor_numbers["electron"][0] == 3, "Electron number should be 3"
        assert flavor_numbers["muon"][0] == -1, "Muon number should be -1"


class TestMasking:
    """Test that masking works correctly."""

    def test_padding_ignored(self):
        """Test that padded particles are ignored."""
        sequences = np.zeros((1, 4, 7))
        masks = np.array([[True, True, False, False]])  # Only 2 real particles

        # 2 real particles (e+, e-)
        sequences[0, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
        )
        sequences[0, 1] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )

        # 2 padded particles (should be ignored even if they have data)
        sequences[0, 2] = create_mock_lepton(
            pt=100, eta=0, phi=0, energy=100, charge=+1, lep_type=ELECTRON
        )
        sequences[0, 3] = create_mock_lepton(
            pt=100, eta=0, phi=0, energy=100, charge=+1, lep_type=ELECTRON
        )

        total_charge = calculate_total_charge(sequences, masks)
        assert total_charge[0] == 0, "Padded particles should be ignored"

        total_energy = calculate_total_energy(sequences, masks)
        assert total_energy[0] == 100, "Padded particles should be ignored"


class TestBatchProcessing:
    """Test processing multiple events."""

    def test_multiple_events(self):
        """Test batch processing of multiple events."""
        sequences = np.zeros((3, 2, 7))  # 3 events
        masks = np.ones((3, 2), dtype=bool)

        # Event 1: charge = 0
        sequences[0, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
        )
        sequences[0, 1] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )

        # Event 2: charge = -2
        sequences[1, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )
        sequences[1, 1] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )

        # Event 3: charge = +2
        sequences[2, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
        )
        sequences[2, 1] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
        )

        total_charge = calculate_total_charge(sequences, masks)
        assert total_charge[0] == 0
        assert total_charge[1] == -2
        assert total_charge[2] == +2


class TestEndToEnd:
    """Test full pipeline."""

    def test_all_metrics(self):
        """Test calculating all metrics at once."""
        sequences = np.zeros((2, 4, 7))
        masks = np.ones((2, 4), dtype=bool)

        # Event 1: H → ZZ → 4l (perfectly conserved)
        for i in range(2):
            sequences[0, i] = create_mock_lepton(
                pt=50, eta=0, phi=i * np.pi, energy=50, charge=-1, lep_type=ELECTRON
            )
            sequences[0, i + 2] = create_mock_lepton(
                pt=50, eta=0, phi=i * np.pi, energy=50, charge=+1, lep_type=ELECTRON
            )

        # Event 2: Charge violated
        for i in range(4):
            sequences[1, i] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=MUON
            )

        metrics = calculate_all_conservation_metrics(sequences, masks)

        # Check event 1
        assert abs(metrics["total_charge"][0]) < 1e-10
        assert metrics["lepton_number"][0] == 0
        assert abs(metrics["met"][0]) < 0.01

        # Check event 2
        assert metrics["total_charge"][1] == -4
        assert metrics["lepton_number"][1] == 4

    def test_summary_stats(self):
        """Test summary statistics calculation."""
        sequences = np.zeros((10, 2, 7))
        masks = np.ones((10, 2), dtype=bool)

        # Create 10 events with charge = 0
        for ev in range(10):
            sequences[ev, 0] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
            )
            sequences[ev, 1] = create_mock_lepton(
                pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
            )

        metrics = calculate_all_conservation_metrics(sequences, masks)
        summary = get_conservation_summary_stats(metrics)

        assert abs(summary["total_charge"]["mean"]) < 1e-10
        assert abs(summary["lepton_number"]["mean"]) < 1e-10
        assert summary["total_charge"]["fraction_nonzero"] == 0

    def test_violation_detection(self):
        """Test violation detection."""
        sequences = np.zeros((2, 2, 7))
        masks = np.ones((2, 2), dtype=bool)

        # Event 1: conserved
        sequences[0, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=+1, lep_type=ELECTRON
        )
        sequences[0, 1] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )

        # Event 2: violated
        sequences[1, 0] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )
        sequences[1, 1] = create_mock_lepton(
            pt=50, eta=0, phi=0, energy=50, charge=-1, lep_type=ELECTRON
        )

        metrics = calculate_all_conservation_metrics(sequences, masks)
        violations = check_conservation_violations(metrics)

        assert violations["charge"] == 0.5  # 50% violated
        assert violations["lepton_number"] == 0.5


class TestPhysicalScenarios:
    """Test realistic physics scenarios."""

    def test_h_to_zz_to_4l(self):
        """Test H → ZZ → 4l decay (should be perfectly conserved)."""
        sequences = np.zeros((1, 4, 7))
        masks = np.ones((1, 4), dtype=bool)

        # 4-lepton final state: 2e, 2μ with balanced charge
        sequences[0, 0] = create_mock_lepton(
            pt=40, eta=0.5, phi=0.3, energy=45, charge=-1, lep_type=ELECTRON
        )
        sequences[0, 1] = create_mock_lepton(
            pt=35, eta=-0.3, phi=2.1, energy=40, charge=+1, lep_type=ELECTRON
        )
        sequences[0, 2] = create_mock_lepton(
            pt=30, eta=1.2, phi=4.5, energy=35, charge=-1, lep_type=MUON
        )
        sequences[0, 3] = create_mock_lepton(
            pt=25, eta=-1.0, phi=5.8, energy=30, charge=+1, lep_type=MUON
        )

        metrics = calculate_all_conservation_metrics(sequences, masks)

        assert abs(metrics["total_charge"][0]) < 1e-10, "Charge should be conserved"
        assert metrics["lepton_number"][0] == 0, "Lepton number should be conserved"
        assert metrics["electron_number"][0] == 0, "Electron number should be conserved"
        assert metrics["muon_number"][0] == 0, "Muon number should be conserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
