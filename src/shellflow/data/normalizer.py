"""
Normalizer - Compute normalization statistics for dataloader configuration.

This module computes min/max statistics for feature normalization by analyzing
particle physics datasets. It uses memory-efficient techniques (memmap) to handle
large datasets without loading everything into memory.

Usage (both --config and --output are required; names are relative to
src/shellflow/config/):
    python -m shellflow.data.normalizer \
        --config dataloader_config_base_06_2to4lep_reduced.yaml \
        --output dataloader_config_06_2to4lep_reduced.yaml

Normalization types:
    - none: Direct min/max values from data
    - log10: Compute min/max on log10(1 + x) transformed values
    - arcsinh: Compute MAD scale first, then min/max on arcsinh(x / mad_scale)
               where mad_scale = 1.4826 * MAD (median absolute deviation)
"""

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm

from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)

# Missing value sentinel used in the dataset
MISSING_VALUE = -999.0

# MAD scale constant (1.4826 makes MAD a consistent estimator of std for normal distribution)
MAD_SCALE_CONSTANT = 1.4826


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to YAML file with proper formatting.

    Ensures numeric values are properly formatted and None values are
    represented as 'null' in YAML.
    """

    # Custom representer for None values
    def represent_none(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:null", "null")

    # Custom representer for floats to avoid scientific notation issues
    def represent_float(dumper, data):
        if data != data:  # NaN check
            return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
        elif data == float("inf"):
            return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf")
        elif data == float("-inf"):
            return dumper.represent_scalar("tag:yaml.org,2002:float", "-.inf")
        else:
            # Use standard float representation
            return dumper.represent_scalar("tag:yaml.org,2002:float", str(data))

    # Create a custom dumper
    class CustomDumper(yaml.SafeDumper):
        pass

    CustomDumper.add_representer(type(None), represent_none)
    CustomDumper.add_representer(float, represent_float)

    with open(output_path, "w") as f:
        yaml.dump(
            config,
            f,
            Dumper=CustomDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    logger.info(f"✓ Configuration saved to: {output_path}")


def get_data_files(data_dirs: List[str], file_pattern: str) -> List[Path]:
    """
    Find all data files matching the pattern in the given directories.

    Args:
        data_dirs: List of directory paths to search
        file_pattern: Glob pattern for data files (e.g., 'transformer_input*.npy')

    Returns:
        List of Path objects for all matching files
    """
    all_files = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning(f"Directory not found: {data_dir}")
            continue

        files = sorted(data_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files in {data_dir}")
        all_files.extend(files)

    return all_files


def get_file_format(file_path: Path) -> str:
    """Determine file format from extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    elif suffix == ".npz":
        return "npz"
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_file_memmap(file_path: Path) -> np.ndarray:
    """
    Load a data file using memory mapping for efficiency.

    Args:
        file_path: Path to the data file

    Returns:
        Memory-mapped numpy array with shape (num_events, 2, seq_length)
    """
    fmt = get_file_format(file_path)

    if fmt == "npy":
        # Direct memory mapping for .npy files
        return np.load(file_path, mmap_mode="r")
    elif fmt == "npz":
        # NPZ files also support memory mapping
        npz = np.load(file_path, mmap_mode="r")
        return npz["data"]
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def extract_feature_values(
    data: np.ndarray, feature_id: int, batch_size: int = 100000
) -> np.ndarray:
    """
    Extract all valid values for a specific feature ID from a memory-mapped array.

    Uses batched processing to handle large files efficiently.

    Args:
        data: Memory-mapped data array of shape (num_events, 2, seq_length)
        feature_id: The feature ID to extract
        batch_size: Number of events to process at once

    Returns:
        1D array of all valid (non-missing) values for the feature
    """
    num_events = data.shape[0]
    all_values = []

    for start in range(0, num_events, batch_size):
        end = min(start + batch_size, num_events)

        # Load batch into memory
        batch_features = data[start:end, 0, 2:]  # Feature values
        batch_packed = data[start:end, 1, 2:]  # Packed indices

        # Extract feature IDs
        batch_feature_ids = (batch_packed % 1000).astype(np.int32)

        # Mask for this feature ID
        mask = batch_feature_ids == feature_id

        # Get values and filter out missing values
        values = batch_features[mask]
        valid_mask = values != MISSING_VALUE
        valid_values = values[valid_mask]

        if len(valid_values) > 0:
            all_values.append(valid_values)

    if all_values:
        return np.concatenate(all_values)
    else:
        return np.array([], dtype=np.float32)


def compute_mad_scale(values: np.ndarray) -> float:
    """
    Compute MAD scale for arcsinh normalization.

    MAD scale = 1.4826 * MAD, where MAD = median(|x - median(x)|)

    Args:
        values: Array of values (already filtered for valid values)

    Returns:
        MAD scale value
    """
    if len(values) == 0:
        return 1.0

    median_val = np.median(values)
    mad = np.median(np.abs(values - median_val))
    mad_scale = MAD_SCALE_CONSTANT * mad

    # Prevent division by zero
    if mad_scale == 0:
        mad_scale = 1.0

    return float(mad_scale)


def compute_feature_statistics(
    files: List[Path],
    feature_id: int,
    norm_type: str,
    mad_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute statistics for a single feature across all files.

    Args:
        files: List of data file paths
        feature_id: The feature ID to compute stats for
        norm_type: Normalization type ('none', 'log10', 'arcsinh')
        mad_scale: Pre-computed MAD scale (for arcsinh, pass None to compute)

    Returns:
        Dictionary with 'min', 'max', and optionally 'mad_scale'
    """
    all_values = []

    for file_path in files:
        data = load_file_memmap(file_path)
        values = extract_feature_values(data, feature_id)
        if len(values) > 0:
            all_values.append(values)

    if not all_values:
        logger.warning(f"No values found for feature ID {feature_id}")
        return {"min": None, "max": None, "mad_scale": None}

    # Concatenate all values
    combined_values = np.concatenate(all_values)
    logger.debug(f"Feature {feature_id}: {len(combined_values):,} total values")

    result = {"mad_scale": None}

    if norm_type == "log10":
        # Transform: log10(1 + x) - only for positive values
        positive_mask = combined_values > 0
        if positive_mask.sum() > 0:
            transformed = np.log10(1 + combined_values[positive_mask])
            result["min"] = float(transformed.min())
            result["max"] = float(transformed.max())
        else:
            # Fallback if no positive values
            result["min"] = float(np.log10(1 + np.abs(combined_values.min())))
            result["max"] = float(np.log10(1 + np.abs(combined_values.max())))

    elif norm_type == "arcsinh":
        # First compute MAD scale if not provided
        if mad_scale is None:
            mad_scale = compute_mad_scale(combined_values)
        result["mad_scale"] = mad_scale

        # Transform: arcsinh(x / mad_scale)
        transformed = np.arcsinh(combined_values / mad_scale)
        result["min"] = float(transformed.min())
        result["max"] = float(transformed.max())

    else:  # 'none'
        result["min"] = float(combined_values.min())
        result["max"] = float(combined_values.max())

    return result


def compute_arcsinh_mad_scales(
    files: List[Path], arcsinh_features: List[Dict[str, Any]], batch_size: int = 100000
) -> Dict[int, float]:
    """
    Compute MAD scales for all arcsinh features in a single pass over the data.

    This is more efficient than computing each feature separately.

    Args:
        files: List of data file paths
        arcsinh_features: List of feature configs with 'id' key
        batch_size: Events per batch

    Returns:
        Dictionary mapping feature_id -> mad_scale
    """
    feature_ids = [f["id"] for f in arcsinh_features]

    # Collect all values per feature
    feature_values = {fid: [] for fid in feature_ids}

    logger.info(f"Computing MAD scales for {len(feature_ids)} arcsinh features...")

    for file_path in tqdm(files, desc="Collecting values for MAD"):
        data = load_file_memmap(file_path)
        num_events = data.shape[0]

        for start in range(0, num_events, batch_size):
            end = min(start + batch_size, num_events)

            batch_features = data[start:end, 0, 2:]
            batch_packed = data[start:end, 1, 2:]
            batch_feature_ids = (batch_packed % 1000).astype(np.int32)

            for fid in feature_ids:
                mask = batch_feature_ids == fid
                values = batch_features[mask]
                valid_mask = values != MISSING_VALUE
                valid_values = values[valid_mask]

                if len(valid_values) > 0:
                    feature_values[fid].append(valid_values)

    # Compute MAD scale for each feature
    mad_scales = {}
    for fid in feature_ids:
        if feature_values[fid]:
            combined = np.concatenate(feature_values[fid])
            mad_scales[fid] = compute_mad_scale(combined)
            logger.debug(f"Feature {fid}: MAD scale = {mad_scales[fid]:.6f}")
        else:
            logger.warning(
                f"Feature {fid}: No values found, using default MAD scale = 1.0"
            )
            mad_scales[fid] = 1.0

    return mad_scales


def compute_all_statistics(
    files: List[Path],
    normalization_config: List[Dict[str, Any]],
    batch_size: int = 100000,
) -> List[Dict[str, Any]]:
    """
    Compute statistics for all features that need updating.

    Efficiently processes data by:
    1. First pass: Compute MAD scales for all arcsinh features
    2. Second pass: Compute min/max for all features with null values

    Args:
        files: List of data file paths
        normalization_config: List of feature normalization configs
        batch_size: Events per batch

    Returns:
        Updated normalization config list
    """
    # Identify features that need computation (null min or max)
    features_to_compute = []
    arcsinh_features = []

    for feat in normalization_config:
        needs_update = feat.get("min") is None or feat.get("max") is None
        if needs_update:
            features_to_compute.append(feat)
            if feat.get("type") == "arcsinh" and feat.get("mad_scale") is None:
                arcsinh_features.append(feat)

    if not features_to_compute:
        logger.info("No features need statistics computation")
        return normalization_config

    logger.info(f"Computing statistics for {len(features_to_compute)} features")
    logger.info(f"  - {len(arcsinh_features)} arcsinh features")
    logger.info(
        f"  - {len(features_to_compute) - len(arcsinh_features)} other features"
    )

    # Step 1: Compute MAD scales for arcsinh features
    mad_scales = {}
    if arcsinh_features:
        mad_scales = compute_arcsinh_mad_scales(files, arcsinh_features, batch_size)

    # Step 2: Compute min/max for all features
    logger.info("Computing min/max statistics...")

    # Collect all values in a single pass
    feature_ids_to_compute = [f["id"] for f in features_to_compute]
    feature_values = {fid: [] for fid in feature_ids_to_compute}

    for file_path in tqdm(files, desc="Computing statistics"):
        data = load_file_memmap(file_path)
        num_events = data.shape[0]

        for start in range(0, num_events, batch_size):
            end = min(start + batch_size, num_events)

            batch_features = data[start:end, 0, 2:]
            batch_packed = data[start:end, 1, 2:]
            batch_feature_ids = (batch_packed % 1000).astype(np.int32)

            for fid in feature_ids_to_compute:
                mask = batch_feature_ids == fid
                values = batch_features[mask]
                valid_mask = values != MISSING_VALUE
                valid_values = values[valid_mask]

                if len(valid_values) > 0:
                    feature_values[fid].append(valid_values)

    # Compute final statistics for each feature
    updated_config = []
    config_dict = {f["id"]: f for f in normalization_config}

    for fid in sorted(config_dict.keys()):
        feat = config_dict[fid].copy()

        # Only update if min/max are null
        if feat.get("min") is None or feat.get("max") is None:
            norm_type = feat.get("type", "none")

            if fid in feature_values and feature_values[fid]:
                combined_values = np.concatenate(feature_values[fid])

                if norm_type == "log10":
                    # Transform: log10(1 + x)
                    # Filter for valid transformation
                    valid_for_log = combined_values > -1
                    if valid_for_log.sum() > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            transformed = np.log10(1 + combined_values[valid_for_log])
                        feat["min"] = float(transformed.min())
                        feat["max"] = float(transformed.max())
                    else:
                        logger.warning(
                            f"Feature {fid}: No valid values for log10 transform"
                        )

                elif norm_type == "arcsinh":
                    # Use pre-computed MAD scale
                    mad_scale = mad_scales.get(fid, 1.0)
                    feat["mad_scale"] = mad_scale

                    # Transform: arcsinh(x / mad_scale)
                    transformed = np.arcsinh(combined_values / mad_scale)
                    feat["min"] = float(transformed.min())
                    feat["max"] = float(transformed.max())

                else:  # 'none'
                    feat["min"] = float(combined_values.min())
                    feat["max"] = float(combined_values.max())

                logger.info(
                    f"Feature {fid} ({feat['name']}): min={feat.get('min')}, max={feat.get('max')}"
                )
            else:
                logger.warning(
                    f"Feature {fid} ({feat['name']}): No values found in data"
                )

        updated_config.append(feat)

    return updated_config


def compute_manifold_stats(
    files: List[Path],
    batch_size: int = 100000,
) -> Dict[str, Any]:
    """
    Compute per-type manifold encoding statistics from training data.

    Computes chi (boost rapidity), log-mass, and energy reference stats for
    the Riemannian Flow Matching massive-sector encoding.  Fully vectorized.

    Args:
        files: List of data file paths
        batch_size: Events per batch

    Returns:
        Dict with chi_stats, logm_stats, e_ref, mass_floor per type code
    """
    MASSIVE_TYPES = {15, 1, 10}  # tau, jet, large-R jet
    MASSLESS_TYPES = {11, 13, 22}  # electron, muon, photon
    ALL_TYPES = MASSIVE_TYPES | MASSLESS_TYPES
    MASS_FLOOR = {
        15: 0.05,
        1: 0.5,
        10: 5.0,
    }  # must mirror MASS_FLOOR in models/riemannian_DiTs_geometry.py

    # Accumulators: per-type arrays of chi, logm, E
    type_chi: Dict[int, List[np.ndarray]] = {t: [] for t in MASSIVE_TYPES}
    type_logm: Dict[int, List[np.ndarray]] = {t: [] for t in MASSIVE_TYPES}
    type_energy: Dict[int, List[np.ndarray]] = {t: [] for t in ALL_TYPES}

    for file_path in tqdm(files, desc="Computing manifold stats"):
        data = load_file_memmap(file_path)
        num_events = data.shape[0]

        for start in range(0, num_events, batch_size):
            end = min(start + batch_size, num_events)
            features = data[start:end, 0, :]  # (B, S) — feature values
            packed = data[start:end, 1, :]  # (B, S) — packed indices

            particle_inst = (packed // 1000).astype(np.int32)
            feature_id = (packed % 1000).astype(np.int32)

            B, S = features.shape

            # For each feature ID, build (B, S) value arrays where the
            # feature matches, else NaN.  Then group by particle_inst.
            # Strategy: for each (event, token), we know particle_inst and
            # feature_id. Build a flat key = event_idx * 1000 + particle_inst
            # (max particles ~30, fits easily). Use np.unique to pivot.

            event_idx = np.arange(B)[:, None].repeat(S, axis=1)  # (B, S)

            # Flatten everything
            ev_flat = event_idx.ravel()  # (B*S,)
            pi_flat = particle_inst.ravel()  # (B*S,)
            fi_flat = feature_id.ravel()  # (B*S,)
            fv_flat = features.ravel()  # (B*S,)

            # Unique particle key = ev * 100 + pi (max particles < 100)
            pkey = ev_flat.astype(np.int64) * 100 + pi_flat.astype(np.int64)

            # Extract values for each needed feature ID
            # Build lookup: for each (pkey, feature_id) -> value
            # We need: type(1), pT(2), eta(3), E(5)

            for tcode in ALL_TYPES:
                # Find tokens where feature_id=1 (type) and value=tcode
                type_match = (fi_flat == 1) & (
                    np.round(fv_flat).astype(np.int32) == tcode
                )
                if not type_match.any():
                    continue

                # Get the particle keys for particles of this type
                pkeys_of_type = pkey[type_match]

                # Find energy (feature_id=5) for these same particle keys
                e_match = fi_flat == 5
                e_pkeys = pkey[e_match]
                e_vals = fv_flat[e_match]

                # Intersect: find energies whose pkey is in pkeys_of_type
                # Use sorted search for efficiency
                pkeys_sorted = np.sort(pkeys_of_type)
                in_type = np.searchsorted(pkeys_sorted, e_pkeys)
                in_type = np.clip(in_type, 0, len(pkeys_sorted) - 1)
                match = pkeys_sorted[in_type] == e_pkeys

                E_matched = e_vals[match]
                valid_E = E_matched[(E_matched > 0) & (E_matched != MISSING_VALUE)]

                if len(valid_E) > 0:
                    type_energy[tcode].append(valid_E)

                if tcode in MASSIVE_TYPES and len(valid_E) > 0:
                    # Also need pT and eta for mass computation
                    pt_match = fi_flat == 2
                    pt_pkeys = pkey[pt_match]
                    pt_vals = fv_flat[pt_match]

                    eta_match = fi_flat == 3
                    eta_pkeys = pkey[eta_match]
                    eta_vals = fv_flat[eta_match]

                    # Get pkeys that have ALL of: type, E, pT, eta
                    # Build set intersection via sorted merge
                    pkeys_with_E = e_pkeys[match]

                    # pT intersection
                    in_pt = np.searchsorted(pkeys_sorted, pt_pkeys)
                    in_pt = np.clip(in_pt, 0, len(pkeys_sorted) - 1)
                    pt_in_type = pkeys_sorted[in_pt] == pt_pkeys
                    pt_pkeys_valid = pt_pkeys[pt_in_type]
                    pt_vals_valid = pt_vals[pt_in_type]

                    # eta intersection
                    in_eta = np.searchsorted(pkeys_sorted, eta_pkeys)
                    in_eta = np.clip(in_eta, 0, len(pkeys_sorted) - 1)
                    eta_in_type = pkeys_sorted[in_eta] == eta_pkeys
                    eta_pkeys_valid = eta_pkeys[eta_in_type]
                    eta_vals_valid = eta_vals[eta_in_type]

                    # Now join E, pT, eta on pkey.
                    # All three should have the same pkeys (one per particle).
                    # Build dict: pkey -> (E, pT, eta)
                    E_dict = dict(zip(pkeys_with_E.tolist(), valid_E.tolist()))
                    pt_dict = dict(zip(pt_pkeys_valid.tolist(), pt_vals_valid.tolist()))
                    eta_dict = dict(
                        zip(eta_pkeys_valid.tolist(), eta_vals_valid.tolist())
                    )

                    common_keys = set(E_dict) & set(pt_dict) & set(eta_dict)
                    if not common_keys:
                        continue

                    common_list = list(common_keys)
                    E_arr = np.array([E_dict[k] for k in common_list])
                    pT_arr = np.array([pt_dict[k] for k in common_list])
                    eta_arr = np.array([eta_dict[k] for k in common_list])

                    # Filter valid
                    valid = (
                        (E_arr > 0)
                        & (pT_arr > 0)
                        & (E_arr != MISSING_VALUE)
                        & (pT_arr != MISSING_VALUE)
                        & (eta_arr != MISSING_VALUE)
                    )
                    E_arr = E_arr[valid]
                    pT_arr = pT_arr[valid]
                    eta_arr = eta_arr[valid]

                    if len(E_arr) == 0:
                        continue

                    # |p|^2 = pT^2 * cosh^2(eta)
                    p_mag_sq = pT_arr**2 * np.cosh(eta_arr) ** 2
                    m_sq = np.maximum(E_arr**2 - p_mag_sq, 0.0)
                    m = np.sqrt(m_sq)
                    m = np.maximum(m, MASS_FLOOR.get(tcode, 0.01))

                    chi = np.arccosh(np.maximum(E_arr / m, 1.0))
                    logm = np.log(m)

                    type_chi[tcode].append(chi)
                    type_logm[tcode].append(logm)

    # Aggregate statistics
    chi_stats = {}
    logm_stats = {}
    e_ref = {}
    mass_floor = dict(MASS_FLOOR)

    for tcode in MASSIVE_TYPES:
        if type_chi[tcode]:
            chi_arr = np.concatenate(type_chi[tcode])
            chi_stats[tcode] = {
                "mu": round(float(np.mean(chi_arr)), 4),
                "sigma": round(float(np.std(chi_arr)), 4),
            }
            logger.info(
                f"Type {tcode}: chi mu={chi_stats[tcode]['mu']:.4f}, "
                f"sigma={chi_stats[tcode]['sigma']:.4f} (N={len(chi_arr)})"
            )

        if type_logm[tcode]:
            logm_arr = np.concatenate(type_logm[tcode])
            logm_stats[tcode] = {
                "mu": round(float(np.mean(logm_arr)), 4),
                "sigma": round(float(np.std(logm_arr)), 4),
            }
            logger.info(
                f"Type {tcode}: logm mu={logm_stats[tcode]['mu']:.4f}, "
                f"sigma={logm_stats[tcode]['sigma']:.4f}"
            )

    for tcode in ALL_TYPES:
        if type_energy[tcode]:
            e_arr = np.concatenate(type_energy[tcode])
            e_ref[tcode] = round(float(np.median(e_arr)), 1)
            logger.info(
                f"Type {tcode}: E_ref (median)={e_ref[tcode]:.1f} GeV (N={len(e_arr)})"
            )

    return {
        "chi_stats": chi_stats,
        "logm_stats": logm_stats,
        "e_ref": e_ref,
        "mass_floor": mass_floor,
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration to ensure it's properly formatted.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    if "normalization" not in config:
        raise ValueError("Config missing 'normalization' key")

    if "data" not in config:
        raise ValueError("Config missing 'data' key")

    required_data_keys = ["normalization_data", "file_pattern"]
    for key in required_data_keys:
        if key not in config["data"]:
            raise ValueError(f"Config 'data' section missing '{key}'")

    # Validate normalization entries
    for i, norm in enumerate(config["normalization"]):
        if "id" not in norm:
            raise ValueError(f"Normalization entry {i} missing 'id'")
        if "name" not in norm:
            raise ValueError(f"Normalization entry {i} missing 'name'")
        if "type" not in norm:
            raise ValueError(f"Normalization entry {i} missing 'type'")

    return True


def remove_mean_std_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove 'mean' and 'std' fields from normalization entries.

    Args:
        config: Configuration dictionary

    Returns:
        Updated configuration without mean/std fields
    """
    for norm in config.get("normalization", []):
        norm.pop("mean", None)
        norm.pop("std", None)
    return config


def main():
    """Main function to compute normalization statistics."""
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for dataloader configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default config paths
    python -m shellflow.data.normalizer
    
    # Specify custom input/output
    python -m shellflow.data.normalizer --config dataloader_config_base.yaml --output dataloader_config.yaml
    
    # With custom batch size
    python -m shellflow.data.normalizer --batch-size 50000
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Input base config file name (relative to config dir, e.g. dataloader_config_base_06_2to4lep_reduced.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output config file name (relative to config dir, e.g. dataloader_config_06_2to4lep_reduced.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100000,
        help="Batch size for processing (default: 100000)",
    )
    parser.add_argument(
        "--manifold-stats",
        action="store_true",
        help="Compute manifold encoding stats (chi, logm, E_ref) and write to output config",
    )

    args = parser.parse_args()

    # Define paths
    base_path = Path(__file__).parent.parent  # shellflow/
    config_dir = base_path / "config"

    input_path = config_dir / args.config
    output_path = config_dir / args.output

    logger.info("=" * 70)
    logger.info("Normalization Statistics Computation")
    logger.info("=" * 70)
    logger.info(f"Input config:  {input_path}")
    logger.info(f"Output config: {output_path}")

    # Load input configuration
    logger.info(f"Loading configuration from {input_path}")
    config = load_config(input_path)

    # Remove mean/std fields if present
    config = remove_mean_std_from_config(config)

    # Validate configuration
    validate_config(config)

    # Get data directories and file pattern
    data_dirs = config["data"]["normalization_data"]
    file_pattern = config["data"]["file_pattern"]

    logger.info(f"Data directories: {len(data_dirs)}")
    for d in data_dirs:
        logger.info(f"  - {d}")
    logger.info(f"File pattern: {file_pattern}")

    # Find all data files
    files = get_data_files(data_dirs, file_pattern)

    if not files:
        raise FileNotFoundError(
            f"No data files found matching pattern '{file_pattern}'"
        )

    logger.info(f"Total files found: {len(files)}")

    # Compute feature normalization statistics
    updated_normalization = compute_all_statistics(
        files, config["normalization"], batch_size=args.batch_size
    )
    config["normalization"] = updated_normalization

    if args.manifold_stats:
        # Compute manifold encoding statistics
        logger.info("Computing manifold encoding statistics...")
        manifold_stats = compute_manifold_stats(files, batch_size=args.batch_size)
        config["manifold_stats"] = manifold_stats

        # Convert int keys to strings for YAML serialization
        for section in ["chi_stats", "logm_stats", "e_ref", "mass_floor"]:
            if section in config["manifold_stats"]:
                config["manifold_stats"][section] = {
                    str(k): v for k, v in config["manifold_stats"][section].items()
                }

    # Save output
    save_config(config, output_path)

    logger.info("=" * 70)
    logger.info("✓ Computation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
