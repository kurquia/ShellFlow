"""
Regression checks on the converted 4lep dataset.

The ROOT -> NumPy converter writes chunked files
data/run2_data_4lep_converted/transformer_input*.npy with shape
(num_events, 2, seq_len): row 0 holds feature values, row 1 holds packed
indices (particle_idx * 1000 + feature_idx). These tests validate the layout
and act as a regression guard on the count of the locally converted ATLAS
Open Data 4lep skim.

All array access goes through mmap; value checks read a small event slice so
the full ~1.5M-event array is never materialized.
"""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[2]
CONVERTED_DIR = REPO_ROOT / "data" / "run2_data_4lep_converted"

# Regression guard: event count of the locally converted 4lep skim
EXPECTED_NUM_EVENTS = 1515848

# Events sampled for value-level checks
SAMPLE_EVENTS = 10_000

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def chunk_files():
    files = sorted(CONVERTED_DIR.glob("transformer_input*.npy"))
    if not files:
        pytest.skip(f"Converted 4lep data not found in {CONVERTED_DIR}")
    return files


@pytest.fixture(scope="module")
def sample_slice(chunk_files):
    """First SAMPLE_EVENTS events of the first chunk, materialized from mmap."""
    arr = np.load(chunk_files[0], mmap_mode="r")
    return np.asarray(arr[: min(SAMPLE_EVENTS, arr.shape[0])])


class TestDataConversion:
    def test_chunk_files_exist(self, chunk_files):
        for path in chunk_files:
            assert path.is_file()
            assert path.stat().st_size > 0

    def test_array_layout(self, chunk_files):
        for path in chunk_files:
            arr = np.load(path, mmap_mode="r")
            assert arr.ndim == 3, f"{path.name}: expected 3D array, got {arr.ndim}D"
            assert arr.shape[1] == 2, (
                f"{path.name}: expected (values, packed_idx) rows, got {arr.shape[1]}"
            )
            # At least the 2 stripped metadata columns plus real tokens
            assert arr.shape[2] > 2
            assert arr.dtype == np.float32

    def test_total_event_count(self, chunk_files):
        total = sum(np.load(p, mmap_mode="r").shape[0] for p in chunk_files)
        assert total == EXPECTED_NUM_EVENTS, (
            f"Converted 4lep skim has {total} events, expected "
            f"{EXPECTED_NUM_EVENTS} (regression guard for the local conversion)"
        )

    def test_packed_indices_decode(self, sample_slice):
        # Skip the 2 metadata columns, as ParticleDataset does
        packed = sample_slice[:, 1, 2:]

        assert np.all(packed == np.round(packed)), "Packed indices must be integral"
        assert np.all(packed >= 0)

        packed = packed.astype(np.int64)
        particle_idxs = packed // 1000
        feature_idxs = packed % 1000

        # particle_idx is a per-event sequential particle counter (small ints)
        assert particle_idxs.max() < 100
        # feature ids live in the 0..72 scheme (0 = padding)
        assert feature_idxs.max() <= 72
        assert np.any(feature_idxs == 0), "Expected padding tokens in sample"
        assert np.any(feature_idxs > 0), "Expected non-padding tokens in sample"

    def test_no_nan_or_inf_values(self, sample_slice):
        # -999.0 is the missing-value sentinel and is allowed (it is finite);
        # only NaN/Inf indicate a broken conversion.
        values = sample_slice[:, 0, 2:]
        assert np.all(np.isfinite(values)), "Found NaN/Inf in converted feature values"
