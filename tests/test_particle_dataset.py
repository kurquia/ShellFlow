"""
Tests for ParticleDataset.

Covers both loading modes (lazy memory-mapped and in-memory), the
normalization pipeline (LUT-based min-max / arcsinh transforms), missing-value
and padding handling, and dataset statistics.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from shellflow.data.particle_dataset import ParticleDataset

REPO_ROOT = Path(__file__).parent.parent

NUM_EVENTS = 10
RAW_SEQ_LEN = 20
# The dataset strips the first 2 sequence columns (metadata slots), so
# per-event tensors have length RAW_SEQ_LEN - 2.
SEQ_LEN = RAW_SEQ_LEN - 2

# Locations of hand-placed values, in RAW column coordinates (stripped
# coordinates are RAW - 2).
MISSING_POSITIONS = [(0, 5), (1, 10)]  # value -999.0
KNOWN_EVENT, KNOWN_COL = 2, 3  # value 0.5, feature id 3 (min-max [-2.5, 2.5])
KNOWN_NORMALIZED = 0.2  # 2 * (0.5 - (-2.5)) / 5 - 1


@pytest.fixture
def synthetic_config_path(tmp_path):
    """Write a synthetic npz data file plus matching YAML config; return config path."""
    rng = np.random.default_rng(42)

    features = rng.normal(size=(NUM_EVENTS, RAW_SEQ_LEN)).astype(np.float32)
    particle_idxs = rng.integers(0, 5, size=(NUM_EVENTS, RAW_SEQ_LEN))
    feature_idxs = rng.integers(1, 6, size=(NUM_EVENTS, RAW_SEQ_LEN))

    # Padding at the end of each event (feature_idx = 0)
    feature_idxs[:, -5:] = 0
    particle_idxs[:, -5:] = 0

    # Missing-value sentinels inside the retained region (columns >= 2)
    for event, col in MISSING_POSITIONS:
        features[event, col] = -999.0

    # Deterministic value to pin down the min-max formula
    features[KNOWN_EVENT, KNOWN_COL] = 0.5
    feature_idxs[KNOWN_EVENT, KNOWN_COL] = 3

    packed = particle_idxs * 1000 + feature_idxs
    data = np.stack([features, packed.astype(np.float32)], axis=1)
    np.savez(tmp_path / "test_data.npz", data=data)

    # Schema mirrors src/shellflow/config/dataloader_config_06_4lep_reduced.yaml:
    # id/name/type/apply_norm/min/max/mad_scale per feature.
    config = {
        "data": {"data_dir": str(tmp_path), "file_pattern": "*.npz"},
        "normalization": [
            {
                "id": 0,
                "name": "padding",
                "type": "none",
                "apply_norm": False,
                "min": 0.0,
                "max": 1.0,
                "mad_scale": None,
            },
            {
                "id": 1,
                "name": "feat_a",
                "type": "none",
                "apply_norm": True,
                "min": -3.0,
                "max": 3.0,
                "mad_scale": None,
            },
            {
                "id": 2,
                "name": "feat_b",
                "type": "none",
                "apply_norm": True,
                "min": -3.0,
                "max": 3.0,
                "mad_scale": None,
            },
            {
                "id": 3,
                "name": "eta",
                "type": "none",
                "apply_norm": True,
                "min": -2.5,
                "max": 2.5,
                "mad_scale": None,
            },
            {
                "id": 4,
                "name": "phi",
                "type": "none",
                "apply_norm": True,
                "min": -3.1416,
                "max": 3.1416,
                "mad_scale": None,
            },
            {
                "id": 5,
                "name": "iso",
                "type": "arcsinh",
                "apply_norm": True,
                "min": -3.0,
                "max": 3.0,
                "mad_scale": 2.0,
            },
        ],
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    return config_file


class TestLazyLoading:
    def test_initialization(self, synthetic_config_path):
        dataset = ParticleDataset(config_path=synthetic_config_path, normalize=True)

        assert len(dataset) == NUM_EVENTS
        # Lazy mode keeps data memory-mapped: no in-memory feature tensor
        assert not hasattr(dataset, "features")

    def test_getitem_structure(self, synthetic_config_path):
        dataset = ParticleDataset(config_path=synthetic_config_path, normalize=True)
        item = dataset[0]

        assert isinstance(item, dict)
        assert set(item.keys()) == {
            "features",
            "features_raw",
            "particle_idxs",
            "feature_idxs",
        }
        for key in ("features", "features_raw", "particle_idxs", "feature_idxs"):
            assert item[key].shape == (SEQ_LEN,)
        assert item["features"].dtype == torch.float32
        assert item["features_raw"].dtype == torch.float32
        assert item["particle_idxs"].dtype == torch.long
        assert item["feature_idxs"].dtype == torch.long

    def test_getitem_without_normalization(self, synthetic_config_path):
        dataset = ParticleDataset(config_path=synthetic_config_path, normalize=False)
        item = dataset[0]

        assert "features_raw" not in item
        assert item["features"].shape == (SEQ_LEN,)

    def test_missing_values_zeroed(self, synthetic_config_path):
        dataset = ParticleDataset(config_path=synthetic_config_path, normalize=True)

        for event, col in MISSING_POSITIONS:
            item = dataset[event]
            pos = col - 2
            assert item["features"][pos] == 0.0
            # The raw copy keeps the sentinel for downstream quality cuts
            assert item["features_raw"][pos] == -999.0


class TestInMemoryLoading:
    def test_initialization(self, synthetic_config_path):
        dataset = ParticleDataset(
            config_path=synthetic_config_path, normalize=True, lazy_loading=False
        )

        assert len(dataset) == NUM_EVENTS
        assert dataset.features.shape == (NUM_EVENTS, SEQ_LEN)
        assert dataset.particle_idxs.shape == (NUM_EVENTS, SEQ_LEN)
        assert dataset.feature_idxs.shape == (NUM_EVENTS, SEQ_LEN)
        assert dataset.features_raw.shape == (NUM_EVENTS, SEQ_LEN)

    def test_no_features_raw_without_normalization(self, synthetic_config_path):
        dataset = ParticleDataset(
            config_path=synthetic_config_path, normalize=False, lazy_loading=False
        )

        assert not hasattr(dataset, "features_raw")
        assert "features_raw" not in dataset[0]

    def test_missing_values_zeroed(self, synthetic_config_path):
        dataset = ParticleDataset(
            config_path=synthetic_config_path, normalize=True, lazy_loading=False
        )

        assert not (dataset.features == -999.0).any()
        for event, col in MISSING_POSITIONS:
            assert dataset.features[event, col - 2] == 0.0
            assert dataset.features_raw[event, col - 2] == -999.0

    def test_padding_stays_zero(self, synthetic_config_path):
        dataset = ParticleDataset(
            config_path=synthetic_config_path, normalize=True, lazy_loading=False
        )

        padding_values = dataset.features[dataset.feature_idxs == 0]
        assert padding_values.numel() > 0
        assert torch.all(padding_values == 0.0)

    def test_normalization_range(self, synthetic_config_path):
        dataset = ParticleDataset(
            config_path=synthetic_config_path, normalize=True, lazy_loading=False
        )

        # Min-max scaling clamps to [-1, 1]; padding and missing map to 0
        assert torch.all(dataset.features >= -1.0)
        assert torch.all(dataset.features <= 1.0)

    def test_known_value_normalization(self, synthetic_config_path):
        dataset = ParticleDataset(
            config_path=synthetic_config_path, normalize=True, lazy_loading=False
        )

        value = dataset.features[KNOWN_EVENT, KNOWN_COL - 2].item()
        assert value == pytest.approx(KNOWN_NORMALIZED, abs=1e-6)

    def test_matches_lazy_mode(self, synthetic_config_path):
        lazy = ParticleDataset(config_path=synthetic_config_path, normalize=True)
        eager = ParticleDataset(
            config_path=synthetic_config_path, normalize=True, lazy_loading=False
        )

        for idx in range(NUM_EVENTS):
            lazy_item = lazy[idx]
            eager_item = eager[idx]
            assert torch.allclose(lazy_item["features"], eager_item["features"])
            assert torch.equal(lazy_item["particle_idxs"], eager_item["particle_idxs"])
            assert torch.equal(lazy_item["feature_idxs"], eager_item["feature_idxs"])


def test_get_statistics(synthetic_config_path):
    dataset = ParticleDataset(config_path=synthetic_config_path, normalize=True)
    stats = dataset.get_statistics()

    assert stats["num_events"] == NUM_EVENTS
    assert stats["sequence_length"] == SEQ_LEN
    # 5 padding columns per event, all inside the retained region
    assert stats["avg_non_padding_tokens"] == SEQ_LEN - 5
    assert stats["min_non_padding_tokens"] == SEQ_LEN - 5
    assert stats["max_non_padding_tokens"] == SEQ_LEN - 5


def test_file_not_found(tmp_path):
    config = {
        "data": {"data_dir": str(tmp_path / "nonexistent"), "file_pattern": "*.npz"},
        "normalization": [],
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))

    with pytest.raises(FileNotFoundError):
        ParticleDataset(config_path=config_file)


REAL_CONFIG = (
    REPO_ROOT
    / "src"
    / "shellflow"
    / "config"
    / "dataloader_config_06_4lep_reduced.yaml"
)
REAL_DATA_DIR = REPO_ROOT / "data" / "run2_data_4lep_reduced"


@pytest.mark.requires_data
@pytest.mark.skipif(
    not (REAL_CONFIG.exists() and any(REAL_DATA_DIR.glob("transformer_input*.npy"))),
    reason=f"Real 4lep data not found in {REAL_DATA_DIR}",
)
def test_real_data():
    dataset = ParticleDataset(
        config_path=REAL_CONFIG,
        normalize=True,
        data_dir_override=REAL_DATA_DIR,
    )

    assert len(dataset) > 0

    item = dataset[0]
    seq_len = item["features"].shape[0]
    assert seq_len > 0
    assert item["particle_idxs"].shape == (seq_len,)
    assert item["feature_idxs"].shape == (seq_len,)
    assert torch.all(torch.isfinite(item["features"]))

    stats = dataset.get_statistics()
    assert stats["num_events"] == len(dataset)
    assert stats["avg_non_padding_tokens"] > 0
