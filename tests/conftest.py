"""
Pytest configuration and shared fixtures for shellflow tests.

This module provides:
- Path configuration for imports
- Common pytest markers
- Shared fixtures used across test modules
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path for all tests
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require external data files"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_data_path(fixtures_dir) -> Path:
    """Return path to sample 4-lepton data fixture."""
    path = fixtures_dir / "sample_4lep_data.npy"
    if not path.exists():
        pytest.skip(f"Sample data not found at {path}. Run create_fixtures.py first.")
    return path


@pytest.fixture
def test_config_path(fixtures_dir) -> Path:
    """Return path to test config file."""
    return fixtures_dir / "test_config.yaml"


# ============================================================================
# Device Fixtures
# ============================================================================


@pytest.fixture
def device() -> torch.device:
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Return CPU device explicitly."""
    return torch.device("cpu")


# ============================================================================
# Random Seed Fixtures
# ============================================================================


@pytest.fixture
def set_seed():
    """Set random seeds for reproducibility."""

    def _set_seed(seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    return _set_seed


@pytest.fixture(autouse=False)
def reproducible(set_seed):
    """Auto-fixture to set seeds. Use as: @pytest.mark.usefixtures('reproducible')"""
    set_seed(42)


# ============================================================================
# Data Loading Fixtures
# ============================================================================


@pytest.fixture
def sample_events(sample_data_path) -> np.ndarray:
    """Load sample events from fixture file."""
    return np.load(sample_data_path)


@pytest.fixture
def sample_event_dicts(sample_events, test_config_path) -> list:
    """
    Convert sample events to list of dicts (like Dataset output).

    Note: This mimics ParticleDataset behavior:
    - Skips first 2 positions (metadata positions)
    - Properly unpacks particle and feature indices
    - Applies normalization like the real dataset
    """
    import yaml

    # Load normalization config
    with open(test_config_path) as f:
        config = yaml.safe_load(f)

    # Build normalization lookup
    norm_stats = {item["id"]: item for item in config["normalization"]}
    max_id = max(norm_stats.keys()) if norm_stats else 72
    norm_min = torch.zeros(max_id + 1)
    norm_max = torch.ones(max_id + 1)
    norm_type = {}

    for feat_id, stats in norm_stats.items():
        norm_min[feat_id] = stats.get("min", 0.0)
        norm_max[feat_id] = stats.get("max", 1.0)
        norm_type[feat_id] = stats.get("type", "none")

    events = []
    for i in range(len(sample_events)):
        event = sample_events[i]
        # Skip first 2 positions like ParticleDataset does (data[:, 0, 2:])
        features = torch.from_numpy(event[0, 2:].copy()).float()
        packed_idx = torch.from_numpy(event[1, 2:].copy()).long()

        # Unpack indices (same as ParticleDataset)
        particle_idxs = packed_idx // 1000
        feature_idxs = packed_idx % 1000

        # Apply normalization (simplified version)
        # Handle missing values
        missing_mask = features == -999.0

        # Apply transformations based on type
        for feat_id in range(max_id + 1):
            mask = feature_idxs == feat_id
            if not mask.any():
                continue

            feat_type = norm_type.get(feat_id, "none")
            if feat_type == "log10":
                features[mask] = torch.log10(1.0 + features[mask].clamp(min=0))
            elif feat_type == "arcsinh":
                mad_scale = norm_stats.get(feat_id, {}).get("mad_scale", 1.0) or 1.0
                features[mask] = torch.asinh(features[mask] / mad_scale)

        # MinMax scaling to [-1, 1]
        mins = norm_min[feature_idxs]
        maxs = norm_max[feature_idxs]
        ranges = maxs - mins
        ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
        features = 2 * (features - mins) / ranges - 1
        features = features.clamp(-1, 1)

        # Handle missing values and padding
        features[missing_mask] = 0.0
        features[feature_idxs == 0] = 0.0

        events.append(
            {
                "features": features,
                "particle_idxs": particle_idxs,
                "feature_idxs": feature_idxs,
            }
        )

    return events
