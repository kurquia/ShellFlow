"""
Script to create test fixtures by extracting sample data from the full dataset.

This script extracts a small number of events (default: 20) from the real
4-lepton converted data to create a minimal test fixture that can be used
for unit and integration tests without requiring access to the full dataset.

Usage:
    uv run python tests/fixtures/create_fixtures.py
    uv run python tests/fixtures/create_fixtures.py --n-events 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def create_sample_data(
    source_path: Path, output_path: Path, n_events: int = 20
) -> None:
    """
    Extract a sample of events from the source data.

    Args:
        source_path: Path to source .npy file
        output_path: Path to save the sample data
        n_events: Number of events to extract
    """
    print(f"Loading source data from: {source_path}")
    data = np.load(source_path, mmap_mode="r")
    print(f"  Source shape: {data.shape}")
    print(f"  Source dtype: {data.dtype}")

    # Extract first n_events
    sample = data[:n_events].copy()
    print(f"\nExtracted {n_events} events")
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample size: {sample.nbytes / 1024:.1f} KB")

    # Save sample
    np.save(output_path, sample)
    print(f"\nSaved to: {output_path}")

    # Verify
    loaded = np.load(output_path)
    assert np.array_equal(loaded, sample), "Verification failed!"
    print("  Verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Create test fixtures from real data")
    parser.add_argument(
        "--n-events",
        type=int,
        default=20,
        help="Number of events to extract (default: 20)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Source data directory containing converted .npy files "
        "(e.g. data/run2_data_4lep_converted)",
    )
    args = parser.parse_args()

    # Paths
    fixtures_dir = Path(__file__).parent
    source_dir = Path(args.source_dir)
    source_file = source_dir / "transformer_input_0000.npy"
    output_file = fixtures_dir / "sample_4lep_data.npy"

    if not source_file.exists():
        print(f"ERROR: Source file not found: {source_file}")
        print("Make sure you have access to the data directory.")
        return 1

    create_sample_data(source_file, output_file, args.n_events)

    print("\nFixtures created successfully!")
    print(f"  sample_4lep_data.npy: {args.n_events} events")

    return 0


if __name__ == "__main__":
    sys.exit(main())
