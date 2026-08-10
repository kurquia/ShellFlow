"""
Convert 4-lepton ROOT files individually by run/period.

Each ROOT file is converted to a separate NPZ file with period-specific naming:
  4lep_transformer_input_data15_periodD.npz
  4lep_transformer_input_data16_periodE.npz
  etc.

Usage:
  uv run python -m shellflow.rootfile_converter_by_run
  uv run python -m shellflow.rootfile_converter_by_run --input_dir data/4leptons_real --output_dir data/4leptons_real_converted
"""

import argparse
import gc
import glob
import logging
import os
import re

import numpy as np
import uproot

from shellflow.compute_higgs_der_features import add_der_features_to_tree
from shellflow.rootfile_to_numpy import (
    DATA_FEATURE_PATH,
    compute_global_sequence_length,
    convert_tree_to_flatsequence_fixed,
    load_feature_config,
)

# Constants for directories
ROOTFILE_DIR = "./data/4leptons_real"
SAVE_DIR = "./data/4leptons_real_converted"

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def ensure_save_dir(save_dir: str = SAVE_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logging.info(f"Created save directory: {save_dir}")


def extract_period_info(filename: str) -> tuple[str, str]:
    """
    Extract year (data15/data16) and period (periodA, periodB, etc.) from filename.

    Example: ODEO_FEB2025_v0_4lep_data16_periodK.4lep.root -> ('data16', 'periodK')
    """
    basename = os.path.basename(filename)

    # Match pattern like data15_periodD or data16_periodK
    match = re.search(r"(data\d+)_(period[A-Z])", basename)
    if match:
        return match.group(1), match.group(2)

    # Fallback: try to extract just period
    period_match = re.search(r"period([A-Z])", basename)
    if period_match:
        return "unknown", f"period{period_match.group(1)}"

    return "unknown", "unknown"


def load_root_files_with_info(rootfile_dir: str) -> list[tuple[str, str, str]]:
    """
    Load ROOT file paths with their period info.

    Returns:
        List of tuples: (file_path, year, period)
    """
    root_files = sorted(glob.glob(f"{rootfile_dir}/*.root"))
    logging.info(f"Found {len(root_files)} ROOT files in {rootfile_dir}")

    files_with_info = []
    for root_file in root_files:
        year, period = extract_period_info(root_file)
        files_with_info.append((root_file, year, period))
        logging.info(f"  {year}_{period}: {os.path.basename(root_file)}")

    return files_with_info


def convert_single_file(
    root_file: str,
    feature_config: dict,
    type_encoding: dict,
    sequence_length: int,
    batch_size: int = 100000,
) -> np.ndarray | None:
    """
    Convert a single ROOT file to transformer input numpy array.

    Returns:
        Numpy array of shape (num_events, 2, sequence_length) or None if failed.
    """
    try:
        root_file_obj = uproot.open(root_file)
        tree = root_file_obj.get("analysis;1")

        if tree is None:
            logging.warning(f"No 'analysis;1' found in {os.path.basename(root_file)}")
            root_file_obj.close()
            return None

        num_events = tree.num_entries
        logging.info(f"    Events in file: {num_events:,}")

        all_batches = []

        # Process in batches to manage memory
        for event_start in range(0, num_events, batch_size):
            event_end = min(event_start + batch_size, num_events)

            # Read batch from disk
            batch_arrays = tree.arrays(
                library="ak", entry_start=event_start, entry_stop=event_end
            )

            # Add DER features
            try:
                batch_arrays = add_der_features_to_tree(batch_arrays)
            except Exception as e:
                logging.warning(f"    Failed to compute DER features: {e}")

            # Convert to transformer input format
            batch_data = convert_tree_to_flatsequence_fixed(
                batch_arrays, feature_config, type_encoding, sequence_length
            )

            if batch_data is not None:
                all_batches.append(batch_data)

            del batch_arrays
            gc.collect()

        root_file_obj.close()

        if not all_batches:
            return None

        # Concatenate all batches
        return np.concatenate(all_batches, axis=0)

    except Exception as e:
        logging.error(f"Failed to process {root_file}: {e}")
        return None


def convert_files_by_period(
    files_with_info: list[tuple[str, str, str]],
    feature_config: dict,
    type_encoding: dict,
    sequence_length: int,
    save_dir: str,
    batch_size: int = 100000,
) -> list[dict]:
    """
    Convert each ROOT file individually and save with period-specific names.

    Returns:
        List of result dictionaries with conversion info.
    """
    results = []

    for i, (root_file, year, period) in enumerate(files_with_info):
        logging.info(f"[{i + 1}/{len(files_with_info)}] Converting {year}_{period}")

        data = convert_single_file(
            root_file,
            feature_config,
            type_encoding,
            sequence_length,
            batch_size=batch_size,
        )

        if data is None:
            logging.warning("    Skipped - no data extracted")
            continue

        # Save with period-specific name
        output_filename = f"4lep_transformer_input_{year}_{period}.npz"
        output_path = os.path.join(save_dir, output_filename)

        np.savez_compressed(output_path, data=data)

        logging.info(f"    Saved: {output_filename} ({data.shape[0]:,} events)")
        results.append(
            {
                "year": year,
                "period": period,
                "events": data.shape[0],
                "filename": output_filename,
                "shape": data.shape,
            }
        )

        del data
        gc.collect()

    return results


def print_summary(results: list[dict]):
    """Print a summary table of converted files."""
    logging.info("=" * 70)
    logging.info("CONVERSION SUMMARY")
    logging.info("=" * 70)
    logging.info(f"{'Year':<10} {'Period':<10} {'Events':>12}  Filename")
    logging.info("-" * 70)

    total_events = 0
    for r in results:
        logging.info(
            f"{r['year']:<10} {r['period']:<10} {r['events']:>12,}  {r['filename']}"
        )
        total_events += r["events"]

    logging.info("-" * 70)
    logging.info(f"{'TOTAL':<10} {'':<10} {total_events:>12,}  ({len(results)} files)")
    logging.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert 4-lepton ROOT files individually by run/period",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert real 4-lepton data (default directories)
  uv run python -m shellflow.rootfile_converter_by_run

  # Convert with custom directories
  uv run python -m shellflow.rootfile_converter_by_run --input_dir data/4leptons --output_dir data/4leptons_converted

  # Specify sequence length manually
  uv run python -m shellflow.rootfile_converter_by_run --sequence_length 159
        """,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=ROOTFILE_DIR,
        help=f"Input directory containing ROOT files (default: {ROOTFILE_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=SAVE_DIR,
        help=f"Output directory for NPZ files (default: {SAVE_DIR})",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=None,
        help="Fixed sequence length (auto-computed if not specified)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Batch size for processing (default: 100000)",
    )
    args = parser.parse_args()

    logging.info("=" * 70)
    logging.info("4-Lepton ROOT to NPZ Converter (by run/period)")
    logging.info("=" * 70)

    # Load ROOT files with period info
    files_with_info = load_root_files_with_info(args.input_dir)

    if not files_with_info:
        logging.error(f"No ROOT files found in {args.input_dir}")
        return

    # Load feature configuration
    feature_config, type_encoding = load_feature_config(DATA_FEATURE_PATH)

    # Determine sequence length
    logging.info("-" * 70)
    if args.sequence_length:
        sequence_length = args.sequence_length
        logging.info(f"Using specified sequence length: {sequence_length}")
    else:
        logging.info("Computing global sequence length across all files...")
        root_files = [f[0] for f in files_with_info]
        sequence_length = compute_global_sequence_length(
            root_files, feature_config, sample_size=1000
        )
        logging.info(f"Computed sequence length: {sequence_length}")

    # Ensure output directory exists
    ensure_save_dir(args.output_dir)

    logging.info("-" * 70)
    logging.info("Starting conversion...")
    logging.info("-" * 70)

    # Convert each file
    results = convert_files_by_period(
        files_with_info,
        feature_config,
        type_encoding,
        sequence_length,
        args.output_dir,
        batch_size=args.batch_size,
    )

    # Print summary
    print_summary(results)
    logging.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
