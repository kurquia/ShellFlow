import argparse
import gc
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import awkward as ak
import numpy as np
import psutil
import uproot
import yaml
import zarr

from shellflow import MetadataHandler, get_logger
from shellflow.compute_higgs_der_features import add_der_features_to_tree

logger = get_logger(__name__)
metadata = MetadataHandler()
metadata()  # Initialize to set shift_bits

# Constants for directories
SCRIPT_DIR = Path(__file__).parent
DATASRT_BASE_DIR = SCRIPT_DIR.parent.parent
DATA_FEATURE_PATH = SCRIPT_DIR / "config/data_feature_06.yaml"
# DATASETS = {
#     'mc_4lep': {
#         'input_dir': './data/4leptons',
#         'output_dir': './data/4leptons_converted',
#         'description': 'Monte Carlo 4-lepton events (simulation)'
#     },
#     'real_data_4lep': {
#         'input_dir': './data/4leptons_real',
#         'output_dir': './data/4leptons_real_converted',
#         'description': 'Real ATLAS 4-lepton events (data)'
#     },
#     'lhc_data': {
#         'input_dir': './data/lhc_run2_data',
#         'output_dir': './data/lhc_run2_data_converted',
#         'description': 'Real ATLAS LHC Run 2 2015 data (use --input-dir to override)'
#     },
#     '4lep_test': {
#         'input_dir': './data/4leptons_real',
#         'output_dir': './data/4leptons_test_converted',
#         'description': 'Test subset of 4-lepton events for quick runs'
#     },
# }

# Load datasets from YAML config
config_path = os.path.join(os.path.dirname(__file__), "config", "datasets.yaml")
with open(config_path, "r") as f:
    DATASETS = yaml.safe_load(f)


def log_memory_usage(context: str = "") -> None:
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    available_mb = psutil.virtual_memory().available / 1024 / 1024
    logger.debug(
        f"MEM {context}: {memory_mb:.1f} [MB used], {available_mb:.1f} [MB available]"
    )


def _check_memory_limit(max_memory_gb: float = 16) -> bool:
    """Check if we're approaching memory limit"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    if memory_gb > max_memory_gb:
        logger.warning(
            f"Memory usage {memory_gb:.1f}GB exceeds limit {max_memory_gb}GB"
        )
        return False
    return True


def load_feature_config(yaml_path: Union[str, Path]) -> Tuple[Dict, Dict]:
    """
    Load the feature configuration and type encoding from the YAML file.

    Parameters:
    - yaml_path: Path to the YAML configuration file

    Returns:
    - Tuple of (feature_config, type_encoding) dictionaries
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    feature_config = config["particles"]
    # Extract TYPE_ENCODING from the YAML file
    type_encoding = {}
    for particle, info in feature_config.items():
        if "type_encoding" in info:
            type_encoding[particle] = info["type_encoding"]

    return feature_config, type_encoding


def save_chunk(
    chunk_data: List[np.ndarray],
    output_path: Union[str, Path],
    chunk_idx: int,
    format: str = "npy",
) -> Tuple[Optional[str], int]:
    """
    Save a chunk of processed data to disk in the specified format.

    Parameters:
    - chunk_data: List of numpy arrays to concatenate and save
    - output_path: Directory where chunk will be saved
    - chunk_idx: Chunk index for sequential numbering
    - format: Save format ('npz', 'npy', or 'zarr')

    Returns:
    - Tuple of (chunk_file_path, num_events_saved)
    """
    if not chunk_data:
        return None, 0

    # Concatenate batch data into single array
    chunk_array = np.concatenate(chunk_data, axis=0)
    num_events = chunk_array.shape[0]

    # Generate base filename with sequential numbering
    chunk_filename = f"transformer_input_{chunk_idx:04d}"
    chunk_file_base = os.path.join(output_path, chunk_filename)

    # Save based on format
    if format == "npz":
        chunk_file = f"{chunk_file_base}.npz"
        np.savez_compressed(chunk_file, data=chunk_array)
    elif format == "npy":
        chunk_file = f"{chunk_file_base}.npy"
        np.save(chunk_file, chunk_array)
    elif format == "zarr":
        chunk_file = f"{chunk_file_base}.zarr"
        zarr.save(chunk_file, chunk_array)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(
        f"  Saved chunk {chunk_idx} with {num_events:,} events to {os.path.basename(chunk_file)}"
    )

    # Clean up memory
    del chunk_array
    gc.collect()

    return chunk_file, num_events


def _filter_by_particle_count(batch_arrays: ak.Array, max_particles: int) -> ak.Array:
    """
    Filter an awkward array of events, keeping only those with total
    particle count <= max_particles.

    Parameters:
    - batch_arrays: Awkward array containing the event data
    - max_particles: Maximum total particle count to keep

    Returns:
    - Filtered awkward array
    """
    count_fields = ["jet_n", "lep_n", "largeRJet_n", "photon_n", "tau_n"]
    total = sum(
        ak.to_numpy(batch_arrays[f]) for f in count_fields if f in batch_arrays.fields
    )
    mask = total <= max_particles
    return batch_arrays[mask], int(np.sum(~mask))


def process_trees_streaming(
    root_files: List[str],
    feature_config: Dict,
    type_encoding: Dict,
    output_path: Union[str, Path],
    batch_size: int,
    global_sequence_length: int,
    max_events_per_chunk: int = 1000000,
    format: str = "npy",
    max_particles: Optional[int] = None,
) -> None:
    """
    Process ROOT files in TRUE streaming fashion to avoid memory issues.
    Reads batches directly from disk using uproot's entry_start/entry_stop parameters,
    so individual ROOT files can be LARGER than available memory.
    Saves data incrementally in chunks to avoid memory buildup.

    Parameters:
    - root_files: List of ROOT file paths
    - feature_config: Feature configuration dictionary from YAML
    - type_encoding: Particle type encoding dictionary
    - output_path: Output directory for files
    - batch_size: Number of events to process at once (default: 10000)
    - fixed_sequence_length: Fixed sequence length to use
    - max_events_per_chunk: Max events to accumulate before saving chunk (default: 1M)
    - format: Save format ('npz', 'npy', or 'zarr')
    - max_particles: If set, only keep events with total particle count <= this value

    Returns:
    - None (saves chunk files to disk)

    Memory Usage:
    - Only loads batch_size events at a time from disk
    - Can handle ROOT files larger than available memory
    - Memory usage = batch_size * sequence_length * feature_dim * 4 bytes
    """

    os.makedirs(output_path, exist_ok=True)

    if max_particles is not None:
        logger.info(f"Filtering: keeping only events with <= {max_particles} particles")

    # Collect processed data in chunks
    chunk_data = []
    chunk_events = 0
    total_events_processed = 0  # Successfully processed events
    total_events_in_root_files = 0  # Total events in ROOT files
    total_events_filtered = 0  # Events removed by particle count filter
    chunk_files = []  # Track saved chunk files

    # Process files one by one with TRUE streaming (never load entire file)
    for file_idx, root_file in enumerate(root_files):
        try:
            # Open file but DON'T load data yet
            root_file_obj = uproot.open(root_file)
            tree = root_file_obj.get("analysis;1")

            if tree is None:
                logger.warning(
                    f"No 'analysis;1' found in {os.path.basename(root_file)}"
                )
                root_file_obj.close()
                continue

            # Get total events WITHOUT loading data
            try:
                num_events = tree.num_entries
                total_events_in_root_files += num_events
                logger.info("-" * 70)
                logger.info(
                    f"Processing file: {file_idx + 1}/{len(root_files)}, {num_events:,} events"
                )
                logger.info(f"File name: {root_file}")
                log_memory_usage(f"before file {file_idx + 1}")
            except Exception as e:
                logger.error(f"Failed to get event count from {root_file}: {e}")
                root_file_obj.close()
                continue

            # Process this file in event batches - reading DIRECTLY from disk
            for event_start in range(0, num_events, batch_size):
                event_end = min(event_start + batch_size, num_events)

                try:
                    # TRUE STREAMING: Read ONLY this batch from disk (not entire file)
                    batch_arrays = tree.arrays(
                        library="ak", entry_start=event_start, entry_stop=event_end
                    )

                    # Compute and add DER_* features (event-level derived features)
                    # This enables transfer learning by aligning 4-lepton with Higgs topology
                    try:
                        batch_arrays = add_der_features_to_tree(batch_arrays)
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute DER_* features for batch {event_start}-{event_end}: {e}"
                        )
                        # Continue without DER_* features if computation fails

                    # Filter by particle count if requested
                    if max_particles is not None:
                        batch_arrays, n_filtered = _filter_by_particle_count(
                            batch_arrays, max_particles
                        )
                        total_events_filtered += n_filtered
                        if len(batch_arrays) == 0:
                            del batch_arrays
                            continue

                    # Convert this batch with fixed global sequence length
                    batch_data = convert_tree_to_flatsequence_fixed(
                        batch_arrays,
                        feature_config,
                        type_encoding,
                        global_sequence_length,
                    )

                    if batch_data is None:
                        logger.warning(
                            f"Failed to convert batch {event_start}-{event_end}"
                        )
                        del batch_arrays
                        continue

                    batch_event_count = len(batch_data)

                    # Check if adding this batch would exceed chunk limit
                    if (chunk_events + batch_event_count) > max_events_per_chunk:
                        # Save current chunk before adding this batch
                        chunk_idx = len(chunk_files)
                        chunk_file, saved_events = save_chunk(
                            chunk_data, output_path, chunk_idx, format
                        )
                        if chunk_file:
                            chunk_files.append(chunk_file)
                        log_memory_usage("after chunk save")

                        # Reset chunk accumulator
                        chunk_data = []
                        chunk_events = 0

                    # Collect batch data
                    chunk_data.append(batch_data)
                    chunk_events += batch_event_count
                    total_events_processed += batch_event_count

                    # Clear memory immediately
                    del batch_arrays, batch_data

                except Exception as e:
                    logger.error(
                        f"Failed to process events {event_start}-{event_end} in {root_file}: {e}"
                    )
                    continue

            # Close the ROOT file after processing all batches
            root_file_obj.close()

        except Exception as e:
            logger.error(f"Failed to open file {root_file}: {e}")
            continue

        # Memory cleanup after each file
        gc.collect()
        log_memory_usage(f"after file {file_idx + 1}")

    # Save any remaining data as final chunk
    if chunk_data:
        chunk_idx = len(chunk_files)
        chunk_file, saved_events = save_chunk(
            chunk_data, output_path, chunk_idx, format
        )
        if chunk_file:
            chunk_files.append(chunk_file)
            logger.info(f"Saved final chunk {chunk_idx} with {saved_events:,} events")

    # Summary
    if len(chunk_files) == 0:
        logger.error("No data was processed!")
        return

    # ==================== SANITY CHECK: Verify all events are saved ====================
    logger.info("=" * 70)
    logger.info("SUMMARY CHECK: Verifying all events are properly saved...")
    logger.info("=" * 70)

    # Count events in saved files
    saved_event_count = 0
    chunk_event_counts = []

    logger.info("Chunk details:")
    for i, chunk_file in enumerate(chunk_files):
        try:
            if chunk_file.endswith(".npz"):
                chunk = np.load(chunk_file)
                data = chunk["data"]
            elif chunk_file.endswith(".npy"):
                data = np.load(chunk_file)
            elif chunk_file.endswith(".zarr"):
                data = zarr.open(chunk_file)[:]
            else:
                raise ValueError(f"Unknown file format for {chunk_file}")

            num_events = data.shape[0]
            saved_event_count += num_events
            chunk_event_counts.append(num_events)
            logger.info(
                f"  [{i}] {os.path.basename(chunk_file)} - {num_events:,} events"
            )

            # Verify data shape
            expected_shape = (num_events, 2, global_sequence_length)
            if data.shape != expected_shape:
                logger.error(
                    f"  [{i}] Shape mismatch! Expected {expected_shape}, got {data.shape}"
                )
        except Exception as e:
            logger.error(
                f"  [{i}] {os.path.basename(chunk_file)} - Could not read: {e}"
            )

    # Final verification
    logger.info("-" * 70)
    logger.info("VERIFICATION RESULTS:")
    logger.info(f"  Events in ROOT files:     {total_events_in_root_files:,}")
    if max_particles is not None:
        logger.info(
            f"  Events filtered (>{max_particles} particles): {total_events_filtered:,}"
        )
    logger.info(f"  Events processed:         {total_events_processed:,}")
    logger.info(f"  Events saved in {format.upper()}:      {saved_event_count:,}")
    total_accounted = total_events_processed + total_events_filtered
    logger.info(
        f"  Failed to process:        {total_events_in_root_files - total_accounted:,}"
    )
    logger.info(
        f"  Processing → Save diff:   {total_events_processed - saved_event_count:,}"
    )
    logger.info(f"  Total chunks saved: {len(chunk_files)}")
    logger.info(f"  Sequence length: {global_sequence_length}")

    if saved_event_count == total_events_processed:
        logger.info("  ✓ SUCCESS: All processed events are properly saved!")
    else:
        logger.error(
            f"  ✗ ERROR: {total_events_processed - saved_event_count:,} processed events are missing from NPZ files!"
        )
        logger.error("  This indicates a critical bug in the saving logic.")

    if total_events_processed < total_events_in_root_files:
        logger.warning(
            f"  ⚠ WARNING: {total_events_in_root_files - total_events_processed:,} events failed during processing"
        )
        logger.warning(
            "  Check logs above for errors in batch processing or conversion."
        )

    logger.info("-" * 70)


def get_dataset_statistics(
    root_files: List[str],
) -> Dict[str, Union[int, float, List[Dict]]]:
    """
    Gather basic statistics about the ROOT files dataset.

    Parameters:
    - root_files: List of ROOT file paths

    Returns:
    - Dictionary containing statistics (total_files, total_size_mb, total_size_gb,
      total_events, avg_events_per_file, file_details)
    """
    logger.info("Gathering dataset statistics...")

    total_size_mb = 0
    total_events = 0
    file_details = []

    for root_file in root_files:
        try:
            size_mb = os.path.getsize(root_file) / (1024 * 1024)
            total_size_mb += size_mb

            with uproot.open(root_file) as file:
                analysis1 = file.get("analysis;1")
                if analysis1 is not None:
                    num_events = len(analysis1["eventNumber"].array(library="np"))
                    total_events += num_events
                    file_details.append(
                        {
                            "name": os.path.basename(root_file),
                            "size_mb": size_mb,
                            "events": num_events,
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not read {os.path.basename(root_file)}: {e}")

    stats = {
        "total_files": len(root_files),
        "total_size_mb": total_size_mb,
        "total_size_gb": total_size_mb / 1024,
        "total_events": total_events,
        "avg_events_per_file": total_events / len(root_files) if root_files else 0,
        "file_details": file_details,
    }

    return stats


def compute_global_sequence_length(
    root_files: List[str],
    feature_config: Dict,
    sample_size: int = 100,
    max_particles: Optional[int] = None,
) -> int:
    """
    Scan ROOT files to determine the global maximum sequence length.
    Samples events from each file to find the true maximum particle multiplicities.

    Parameters:
    - root_files: List of ROOT file paths
    - feature_config: Feature configuration dictionary from YAML
    - sample_size: Number of events to sample per file (default: 100, 0 = all events)
    - max_particles: If set, only consider events with total particle count <= this value

    Returns:
    - Global maximum sequence length across all files
    """
    logger.info(
        f"Computing global sequence length by sampling {sample_size} events per file..."
    )
    if max_particles is not None:
        logger.info(
            f"  Filtering to events with <= {max_particles} particles for sequence length computation"
        )

    global_max_counts = {}
    for pt in feature_config.keys():
        global_max_counts[pt] = 0

    for file_idx, root_file in enumerate(root_files):
        try:
            with uproot.open(root_file) as file:
                analysis1 = file.get("analysis;1")
                if analysis1 is None:
                    continue

                # Get total events
                num_events = len(analysis1["eventNumber"].array(library="np"))

                # Sample events (take first N or all if smaller)
                sample_end = (
                    min(sample_size, num_events) if sample_size > 0 else num_events
                )

                # Load sample
                sample_arrays = analysis1.arrays(
                    entry_start=0, entry_stop=sample_end, library="ak"
                )

                # Filter by particle count if requested
                if max_particles is not None:
                    sample_arrays, _ = _filter_by_particle_count(
                        sample_arrays, max_particles
                    )
                    if len(sample_arrays) == 0:
                        continue

                # Get max multiplicities for this sample
                for pt, info in feature_config.items():
                    rep_field = next(
                        (
                            f
                            for f in info.get("fields", [])
                            if f in sample_arrays.fields
                        ),
                        None,
                    )
                    if rep_field is None:
                        continue

                    arr = sample_arrays[rep_field]
                    try:
                        max_count = int(ak.max(ak.num(arr)))
                    except Exception:
                        max_count = 1

                    # Update global max
                    if max_count > global_max_counts[pt]:
                        global_max_counts[pt] = max_count
                        logger.debug(f"File {file_idx + 1}: {pt} max = {max_count}")

                del sample_arrays
                gc.collect()

        except Exception as e:
            logger.warning(f"Failed to sample {root_file}: {e}")
            continue

    # Calculate sequence length from global maxima
    field_lengths = [len(info["fields"]) for info in feature_config.values()]
    max_counts = [global_max_counts[pt] for pt in feature_config.keys()]

    # Add 1 for type encoding
    total_sequence_length = sum(
        (fl + 1) * mc for fl, mc in zip(field_lengths, max_counts)
    )

    logger.info(f"Global maximum multiplicities: {global_max_counts}")

    return total_sequence_length


def get_max_particle_num_per_type(
    big_tree: ak.Array, feature_config: Dict
) -> List[Tuple[str, int]]:
    """
    Returns a list of (particle_type, observed_max_multiplicity) preserving
    the order they appear in the YAML (dict order).

    Parameters:
    - big_tree: Awkward array containing the event data
    - feature_config: Feature configuration dictionary from YAML

    Returns:
    - List of tuples (particle_type_name, max_multiplicity)
    """
    result = []
    for pt, info in feature_config.items():
        # First existing field for this particle type
        rep_field = next(
            (f for f in info.get("fields", []) if f in big_tree.fields), None
        )
        if rep_field is None:
            result.append((pt, 0))
            continue
        arr = big_tree[rep_field]
        # Try treating as jagged; if not, it's a flat (1 per event) quantity
        try:
            m = int(ak.max(ak.num(arr)))
        except Exception:
            m = 1
        result.append((pt, m))
    return result


def get_sequence_length(big_tree: ak.Array, feature_config: Dict) -> Optional[int]:
    """
    Calculate the total sequence length needed for the transformer input.

    Parameters:
    - big_tree: Awkward array containing the event data
    - feature_config: Feature configuration dictionary from YAML

    Returns:
    - Total sequence length, or None if calculation fails
    """
    field_lengths = [len(info["fields"]) for info in feature_config.values()]
    # print(feature_config.keys())
    max_pt_num_per_type = get_max_particle_num_per_type(big_tree, feature_config)
    max_counts = [m for _, m in max_pt_num_per_type]
    if len(field_lengths) != len(max_counts):
        logger.error("Length mismatch between field lengths and max multiplicities.")
        return None
    # Dot product: sum over particle types of (num_fields * observed_max_particles)
    # Here fl + 1 to account for type encoding slot
    total_feature_slots = sum(
        (fl + 1) * mc for fl, mc in zip(field_lengths, max_counts)
    )
    return total_feature_slots


def convert_tree_to_flatsequence_fixed(
    big_tree: ak.Array, feature_config: Dict, type_encoding: Dict, sequence_length: int
) -> np.ndarray:
    """
    Build array of shape (num_events, 2, sequence_length) with FIXED sequence length.
    Optimized vectorized version for much better performance.

    Parameters:
    - big_tree: Awkward array containing the event data
    - feature_config: Feature configuration dictionary from YAML
    - type_encoding: Particle type encoding dictionary
    - sequence_length: Fixed sequence length (determined from first batch)

    Returns:
    - Numpy array of shape (num_events, 2, sequence_length) where:
      * [:, 0, :] = Content A: particle features (type_encoding + fields)
      * [:, 1, :] = Content B: global_id for metadata entries, then encoded particle IDs (global_id*1000 + per_type_id)
    """
    num_events = len(big_tree["eventNumber"])
    data = np.zeros((num_events, 2, sequence_length), dtype=np.float32)

    # Keep a per-event write offset (event-dependent compaction)
    offsets = np.zeros(num_events, dtype=np.int32)
    # Per-event particle counter (Content B numbering)
    particle_id = np.zeros(num_events, dtype=np.int32)

    # Generate global_id from runNumber and eventNumber for all events
    global_ids = np.zeros(num_events, dtype=np.float32)
    if "runNumber" in big_tree.fields and "eventNumber" in big_tree.fields:
        run_numbers = ak.to_numpy(big_tree["runNumber"])
        event_numbers = ak.to_numpy(big_tree["eventNumber"])
        # Convert eventNumber from string to int if needed
        if event_numbers.dtype.kind in ("U", "S", "O"):  # Unicode, bytes, or object
            event_numbers = event_numbers.astype(np.int64)
        for i in range(num_events):
            global_ids[i] = float(
                metadata.generate_global_id(int(run_numbers[i]), int(event_numbers[i]))
            )

    for pt, info in feature_config.items():
        # Get all configured fields and track which exist in ROOT file
        all_fields = info.get("fields", [])
        existing_fields = [f for f in all_fields if f in big_tree.fields]
        missing_fields = [f for f in all_fields if f not in big_tree.fields]

        # Log warning for missing fields
        if missing_fields:
            logger.warning(
                f"Particle type '{pt}': Fields not found in ROOT file: {missing_fields}. Using default value 0."
            )

        if not existing_fields:
            continue

        # Use all fields (existing + missing) for processing
        fields = all_fields

        # Validate feature_ids if provided
        if "feature_ids" in info:
            if len(info["feature_ids"]) != len(fields):
                logger.error(
                    f"Particle type '{pt}': feature_ids length ({len(info['feature_ids'])}) "
                    f"doesn't match fields length ({len(fields)}). Using sequential IDs."
                )
                # Remove invalid feature_ids to fall back to default
                del info["feature_ids"]

        # Special handling for metadata: replace runNumber and eventNumber with global_id
        if pt == "metadata":
            # Create a single-entry array with global_id for each event
            field_arrays = [ak.singletons(global_ids)]
            segment_len = 1 + 1  # type_encoding + global_id

            # Prepare encoding
            const_val = float(type_encoding.get(pt, -1))
            enc_list = ak.full_like(field_arrays[0], const_val)

            # Build segment
            segment_stack = ak.concatenate(
                [enc_list[:, :, None], field_arrays[0][:, :, None]], axis=2
            )

            # Process metadata (always 1 entry per event)
            counts = np.ones(num_events, dtype=np.int32)
            dense_array = ak.to_numpy(segment_stack)
            flattened = dense_array.reshape(num_events, -1)

            # For metadata, use particle_id=0 with feature_ids [0, -1]
            for i in range(num_events):
                start = int(offsets[i])
                end = start + segment_len

                if end > sequence_length:
                    continue

                # Content A: type_encoding + global_id
                data[i, 0, start:end] = flattened[i, :segment_len]

                # Content B: particle 0 (metadata) with feature_ids
                # Position 0: 0*1000 + 0 = 0 (type_encoding, feature_id=0)
                # Position 1: 0*1000 + (-1) = -1 (global_id, feature_id=-1)
                data[i, 1, start:end] = [0.0, -1.0]

                offsets[i] = end

            continue  # Skip normal processing for metadata

        # Prepare jagged arrays for each field (wrap flat as singletons)
        field_arrays = []
        reference_field_for_counts = None

        for f in fields:
            # Handle missing fields by creating zero-filled arrays
            if f not in big_tree.fields:
                # Create zero array with same structure as first existing field
                if reference_field_for_counts is not None:
                    arr = ak.zeros_like(reference_field_for_counts)
                else:
                    # Defer creation until we have a reference field
                    field_arrays.append(None)
                    continue
            else:
                arr = big_tree[f]

                # Convert eventNumber from string to int (ROOT stores it as string)
                if f == "eventNumber":
                    arr = ak.values_astype(arr, np.int64)

                # Convert boolean fields to float (0.0 or 1.0)
                try:
                    arr_type = str(ak.type(arr))
                    if "bool" in arr_type:
                        arr = ak.values_astype(arr, np.float32)
                except Exception:
                    pass

                # If scalar per event, wrap to length-1 list
                try:
                    _ = ak.num(arr, axis=1)
                except Exception:
                    arr = ak.singletons(arr)

                # Use first existing field as reference for particle counts
                if reference_field_for_counts is None:
                    reference_field_for_counts = arr

            field_arrays.append(arr)

        # Fill in deferred missing fields with zeros
        for i, arr in enumerate(field_arrays):
            if arr is None and reference_field_for_counts is not None:
                field_arrays[i] = ak.zeros_like(reference_field_for_counts)

        # Determine per-event particle counts using reference field
        if reference_field_for_counts is None:
            continue
        counts = ak.to_numpy(ak.num(reference_field_for_counts, axis=1)).astype(
            np.int32
        )

        # Lepton special encoding (per particle), else constant
        if (
            pt == "lepton"
            and "type_encoding_field" in info
            and info["type_encoding_field"] in big_tree.fields
        ):
            enc_raw = big_tree[info["type_encoding_field"]]
            try:
                _ = ak.num(enc_raw, axis=1)
            except Exception:
                enc_raw = ak.singletons(enc_raw)
            enc_list = enc_raw
        else:
            const_val = float(type_encoding.get(pt, 0))
            ref = field_arrays[0]
            enc_list = ak.full_like(ref, const_val)

        # Build per-particle segments
        segment_components = [enc_list] + field_arrays
        segment_stack = ak.concatenate(
            [comp[:, :, None] for comp in segment_components], axis=2
        )
        segment_len = 1 + len(fields)

        # **OPTIMIZED: Vectorized approach using awkward operations**
        # Flatten all particle data at once
        try:
            # Get max particles across all events for this type
            max_particles = int(ak.max(counts))

            if max_particles == 0:
                continue

            # Pad to max particles for this batch
            padded = ak.pad_none(segment_stack, max_particles, axis=1, clip=True)
            filled = ak.fill_none(padded, 0.0)

            # Convert to numpy (events, max_particles, segment_len)
            dense_array = ak.to_numpy(filled)

            # Reshape to (events, max_particles * segment_len)
            flattened = dense_array.reshape(num_events, -1)

            # Get custom feature IDs or default to sequential [1, 2, 3, ...]
            # feature_ids includes type_encoding (position 0) + field IDs
            custom_feature_ids = info["feature_ids"]
            # Prepend 1 for type_encoding position, then use custom IDs
            feature_id_array = np.array([1] + custom_feature_ids, dtype=np.float32)

            # For each event, copy only the valid data based on counts
            for i in range(num_events):
                c = int(counts[i])
                if c == 0:
                    continue

                need = c * segment_len
                start = int(offsets[i])
                end = start + need

                # Check bounds
                if start >= sequence_length:
                    continue
                if end > sequence_length:
                    max_particles_fit = (sequence_length - start) // segment_len
                    if max_particles_fit <= 0:
                        continue
                    c = int(max_particles_fit)
                    need = c * segment_len
                    end = start + need

                # Copy valid data
                data[i, 0, start:end] = flattened[i, :need]

                # Global particle IDs (vectorized)
                particle_ids = np.repeat(
                    np.arange(
                        int(particle_id[i]) + 1,
                        int(particle_id[i]) + 1 + c,
                        dtype=np.float32,
                    ),
                    segment_len,
                )

                # Per-type feature IDs (using custom or sequential IDs)
                # Tiles the feature_id_array for each particle
                # Example: feature_id_array=[1,2,3,5], c=2 → [1,2,3,5, 1,2,3,5]
                feature_ids = np.tile(feature_id_array, c)
                data[i, 1, start:end] = particle_ids * 1000 + feature_ids
                particle_id[i] += c
                offsets[i] = end

        except Exception as e:
            logger.warning(f"Vectorized approach failed for {pt}, falling back: {e}")
    return data


def convert_tree_to_flatsequence(
    big_tree: ak.Array, feature_config: Dict, type_encoding: Dict
) -> Optional[np.ndarray]:
    """
    Build array of shape (num_events, 2, sequence_length) WITHOUT per‑type internal padding.
    Zeros only appear at the (right) tail of each event row.

    Parameters:
    - big_tree: Awkward array containing the event data
    - feature_config: Feature configuration dictionary from YAML
    - type_encoding: Particle type encoding dictionary

    Returns:
    - Numpy array of shape (num_events, 2, sequence_length) where:
      * [:, 0, :] = Content A: [type_id_or_lep_type, field1, field2, ...] per particle, tightly packed
      * [:, 1, :] = Content B: encoded particle IDs (global_id*1000 + per_type_id)
    - Remaining (unused) tail positions are zero (already from initialization).
    - Returns None if sequence_length calculation fails

    Packing order (YAML order):
        For each particle type, append ALL existing particles (only their real fields) for that event.
        No reservation for the type's max multiplicity; events with fewer particles just produce fewer entries.
    """
    # Upper bound (sequence_length) from max observed multiplicities (used just as total capacity)
    sequence_length = get_sequence_length(big_tree, feature_config)
    if sequence_length is None:
        logger.error("sequence_length is None.")
        return None

    return convert_tree_to_flatsequence_fixed(
        big_tree, feature_config, type_encoding, sequence_length
    )


def save_numpy_to_npz(data: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save the resulting numpy array as an .npz file.

    Parameters:
    - data: Numpy array to save
    - output_path: Output file path (without extension)

    Returns:
    - None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(output_path, data=data)
    logger.info(f"Saved numpy array to {output_path}.npz")


def main_streaming() -> None:
    """
    Memory-efficient processing for large datasets.

    Returns:
    - None (processes files and saves output to disk)
    """

    parser = argparse.ArgumentParser(
        description="Convert ROOT files to binary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Convert MC 4-lepton data (default) with streaming
            uv run python -m shellflow.rootfile_to_numpy --dataset mc_4lep --streaming
            # Convert real ATLAS 4-lepton data with streaming
            uv run python -m shellflow.rootfile_to_numpy --dataset real_data_4lep --streaming
            # Use custom config file
            uv run python -m shellflow.rootfile_to_numpy --dataset mc_4lep --config src/shellflow/config/data_feature_06.yaml

            # Convert with custom batch size
            uv run python -m shellflow.rootfile_to_numpy --dataset mc_4lep --streaming --batch_size 50000

            Available datasets:
            mc_4lep         - Monte Carlo 4-lepton events (simulation)
            real_data_4lep  - Real ATLAS 4-lepton events (data)
            lhc_data        - LHC Open Data (example dataset)
                    """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        default="mc_4lep",
        help="Dataset to convert (default: mc_4lep)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DATA_FEATURE_PATH),
        help=f"Path to data feature config YAML file (default: {DATA_FEATURE_PATH})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1.0e5,
        help="Events per batch (default: 100000)",
    )
    parser.add_argument(
        "--max_events_per_chunk",
        type=int,
        default=1.0e7,
        help="Max events to accumulate before saving chunk (default: 10M)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=None,
        help="Manually specify sequence length (skips interactive prompt)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of events to sample per file for sequence length computation (default: 10000)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "npy", "zarr"],
        default="npy",
        help="Output format for saved chunks (default: npy)",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=None,
        help="Only keep events with total particle count <= this value (default: no filter)",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=DATASRT_BASE_DIR,
        help="Base path to prepend to rootfile_dir (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for converted files (default: input_dir + _06_converted)",
    )
    args = parser.parse_args()

    # Get dataset configuration
    dataset = DATASETS[args.dataset]
    rootfile_dir = (
        str(Path(args.base_path) / dataset["output_dir"])
        if not os.path.isabs(dataset["output_dir"])
        else dataset["output_dir"]
    )
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = rootfile_dir + "_06_converted"

    logger.info("Using streaming mode for memory efficiency")
    logger.info(f"Dataset: {dataset['description']}")
    logger.info(f"Input directory: {rootfile_dir}")
    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Config file: {args.config}")

    root_files = glob.glob(f"{rootfile_dir}/*.root")
    logger.info(f"Found {len(root_files)} ROOT files")

    if not root_files:
        logger.error(f"No ROOT files found in {rootfile_dir}")
        return

    feature_config, type_encoding = load_feature_config(args.config)

    # Validate batching parameters to prevent unexpected behavior
    if args.batch_size > args.max_events_per_chunk:
        logger.error(
            f"batch_size ({args.batch_size:,}) > max_events_per_chunk ({args.max_events_per_chunk:,})"
        )
        logger.error(
            "This will result in chunks of size batch_size instead of max_events_per_chunk"
        )
        logger.error(
            "Consider setting batch_size ≤ max_events_per_chunk for proper chunking behavior"
        )
        sys.exit(1)

    # Determine sequence length
    if args.sequence_length:
        # Command-line override
        global_sequence_length = args.sequence_length
        logger.info(
            f"Using command-line specified sequence length: {global_sequence_length}"
        )
        method = "manual"
    else:
        # Auto-compute without prompt
        logger.info("Auto-computing sequence length...")
        global_sequence_length = compute_global_sequence_length(
            root_files,
            feature_config,
            sample_size=args.sample_size,
            max_particles=args.max_particles,
        )
        logger.info(f"Computed global sequence length: {global_sequence_length}")
        method = "computed"

    # Log chosen configuration
    logger.info(
        f"Starting processing with sequence_length={global_sequence_length}, method={method}"
    )
    logger.info(f"Batch size: {args.batch_size} events")
    logger.info(f"Chunk size: {args.max_events_per_chunk:,} events")
    if args.sequence_length is None:
        logger.info(f"Sequence length sample size: {args.sample_size} events per file")

    # Process with streaming
    process_trees_streaming(
        root_files,
        feature_config,
        type_encoding,
        save_dir,
        batch_size=int(args.batch_size),
        global_sequence_length=global_sequence_length,
        max_events_per_chunk=int(args.max_events_per_chunk),
        format=args.format,
        max_particles=args.max_particles,
    )

    logger.info(f"✓ Processing complete! Output saved to {save_dir}")


if __name__ == "__main__":
    # Use streaming by default for safety
    main_streaming()
