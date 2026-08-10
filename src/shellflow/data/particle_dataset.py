"""
PyTorch Dataset for particle physics data.

This module provides a Dataset class for loading and preprocessing particle
physics events stored in the packed (num_events, 2, sequence_length) format,
with configurable per-feature normalization. It is the event source for the
Riemannian Flow Matching training pipeline (via ParticleVectorCollator).
"""

import mmap as _mmap
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)

# Access advice for the memory-mapped dataset files. Training samples events in a
# global-random order, so each access is a scattered single-event read (~2 KB).
# The kernel's default readahead then pulls ~64 KB per fault (~30x amplification),
# which saturates the NVMe bandwidth and starves the GPUs. MADV_RANDOM disables
# readahead so each fault serves only the page(s) it needs. This is a pure I/O
# hint: it changes NOTHING about which events are read, their order, or values.
# Override with HEP_MMAP_ADVISE = random | normal | none  (default: random).
_MMAP_ADVISE = os.environ.get("HEP_MMAP_ADVISE", "random").lower()
# Sort per-chunk local indices before the batched mmap read so scattered events
# are touched in monotonic file order (better page-cache / device behaviour).
# Content-neutral: results are reordered back to the requested order. Default on.
_SORT_CHUNK_READS = os.environ.get("HEP_SORT_CHUNK_READS", "1") not in ("0", "false")


def _advise_memmap(arr: np.ndarray) -> None:
    """Apply the configured madvise hint to a numpy memmap (best-effort)."""
    if _MMAP_ADVISE in ("none", ""):
        return
    handle = getattr(arr, "_mmap", None)
    if handle is None or not hasattr(handle, "madvise"):
        return
    advice = {
        "random": getattr(_mmap, "MADV_RANDOM", None),
        "normal": getattr(_mmap, "MADV_NORMAL", None),
    }.get(_MMAP_ADVISE)
    if advice is None:
        return
    try:
        handle.madvise(advice)
    except (OSError, ValueError):
        pass  # advice is advisory; never fail data loading over it


class BackgroundPrefetcher:
    """
    Background prefetcher that overlaps data loading/normalization with GPU training.

    This wrapper prefetches batches from a DataLoader in a background thread,
    allowing normalization computation to overlap with model training on GPU.

    Usage:
        train_loader = DataLoader(dataset, ...)
        prefetcher = BackgroundPrefetcher(train_loader, num_prefetch=2)
        for batch in prefetcher:
            # Training code here
            # While GPU trains on this batch, next batch is being prepared

    Benefits:
        - Zero-overhead prefetching using Python threads
        - Works with any DataLoader without modifications
        - GIL is released during I/O and numpy operations, allowing overlap
        - Configurable prefetch depth for memory/speed tradeoff

    Args:
        dataloader: PyTorch DataLoader to wrap
        num_prefetch: Number of batches to prefetch (default: 2)
    """

    def __init__(self, dataloader: DataLoader, num_prefetch: int = 2):
        self.dataloader = dataloader
        self.num_prefetch = num_prefetch
        self._queue: Queue = None
        self._thread: Thread = None
        self._stop_signal = object()  # Sentinel for end of iteration

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Iterator:
        # Create queue and start prefetch thread
        self._queue = Queue(maxsize=self.num_prefetch)
        self._thread = Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()
        return self

    def _prefetch_worker(self):
        """Background worker that loads batches into the queue."""
        try:
            for batch in self.dataloader:
                self._queue.put(batch)
        except Exception as e:
            self._queue.put(e)  # Propagate exception to main thread
        finally:
            self._queue.put(self._stop_signal)

    def __next__(self):
        if self._queue is None:
            raise StopIteration

        item = self._queue.get()

        if item is self._stop_signal:
            # Clean up
            self._thread.join(timeout=1.0)
            self._queue = None
            self._thread = None
            raise StopIteration

        if isinstance(item, Exception):
            raise item

        return item


class ParticleDataset(Dataset):
    """
    PyTorch Dataset for loading particle physics data with flexible loading modes.

    Supports two loading strategies:
    1. In-memory mode (lazy_loading=False):
       - Loads all data into RAM upfront
       - Fast random access, no I/O during training
       - High memory usage (entire dataset in RAM)
       - Best for datasets that fit in RAM and need maximum speed

    2. Lazy loading mode (lazy_loading=True):
       - Uses memory-mapped numpy arrays for zero-copy access
       - Minimal memory footprint (only accessed pages loaded by OS)
       - Scales to arbitrarily large datasets
       - Slightly slower due to disk I/O, but OS handles caching efficiently

    Data Format:
        Input data is stored as numpy arrays of shape (num_events, 2, sequence_length):
        - [:, 0, :]: Feature values (float)
        - [:, 1, :]: Packed indices (int), encoding both particle_idx and feature_idx
                     packed_idx = particle_idx * 1000 + feature_idx

    Features:
        - Dual loading modes (in-memory vs lazy/memory-mapped)
        - On-the-fly normalization (arcsinh, log10, MinMax scaling)
        - Multi-file support (automatic chunking and indexing)
        - Handles missing values (-999) gracefully
        - Multiprocessing-safe (each worker gets independent memory maps)

    Attributes:
        lazy_loading (bool): Loading mode flag
        chunk_files (list): List of data file paths
        chunk_offsets (list): Cumulative event counts for indexing
        norm_min (torch.Tensor): Normalization mins per feature type
        norm_max (torch.Tensor): Normalization maxs per feature type
        norm_type (dict): Feature normalization type ('arcsinh', 'log10', or 'none')
        mad_scale (torch.Tensor): MAD scale values for arcsinh normalization
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        lazy_loading: bool = True,
        data_dir_override: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the ParticleDataset.

        Args:
            config_path: Path to the 'dataloader_config.yaml' file. If None, uses default.
            normalize: Whether to apply feature normalization.
            lazy_loading: If True, use memory-mapped lazy loading (memory efficient).
                         If False, load all data into RAM (fast but memory intensive).
            data_dir_override: If set, use this path instead of the one in the config YAML.
        Raises:
            FileNotFoundError: If no data files found in data_dir.
        """
        super().__init__()

        # Load configuration
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent / "config" / "dataloader_config.yaml"
            )
        else:
            config_path = Path(config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.config = config
        self.normalize = normalize
        self.lazy_loading = lazy_loading

        # Get data directory (override takes priority over config)
        data_dir = (
            Path(data_dir_override)
            if data_dir_override
            else Path(config["data"]["data_dir"])
        )
        file_pattern = config["data"]["file_pattern"]
        chunk_files = sorted(data_dir.glob(file_pattern))

        if not chunk_files:
            raise FileNotFoundError(
                f"No data files matching '{file_pattern}' found in {data_dir}"
            )

        logger.info(f"Found {len(chunk_files)} chunk file(s) in {data_dir}")

        # Store chunk file paths (needed for both modes)
        self.chunk_files = [
            str(f) for f in chunk_files
        ]  # Store as strings for pickling

        # Build index mapping for all modes
        self._build_chunk_index(chunk_files)

        # Load data based on chosen strategy
        if lazy_loading:
            self._setup_lazy_loading()
        else:
            self._load_into_memory(chunk_files)

        # Load normalization configuration
        if self.normalize:
            norm_stats_list = config["normalization"]
            norm_stats_dict = {item["id"]: item for item in norm_stats_list}
            self._setup_normalization(norm_stats_dict)

            # Apply normalization for in-memory mode (lazy mode normalizes on-the-fly)
            if not lazy_loading and hasattr(self, "features"):
                # Keep raw (pre-normalization) copy, exposed downstream as
                # features_raw / particle_extra_raw for raw-value quality cuts
                self.features_raw = self.features.clone()
                self.features = self._normalize_features(
                    self.features, self.feature_idxs
                )
                logger.info(
                    "✓ Applied normalization to all features (missing values → 0.0)"
                )
                logger.info("  Kept raw feature copy for auxiliary loss filtering")

        logger.info(f"ParticleDataset initialized with {len(self)} events")

    def _build_chunk_index(self, chunk_files: List[Path]) -> None:
        """
        Build index mapping from global event index to (chunk_idx, local_idx).

        This is used by both loading modes to support multi-file datasets.
        Supports different file formats: .npz, .npy, .zarr
        """
        self.chunk_offsets = [0]
        self.chunk_shapes = []
        self.chunk_formats = []  # Track format for each chunk

        total_events = 0
        for chunk_file in chunk_files:
            # Determine format from file extension
            if chunk_file.suffix == ".npz":
                format_type = "npz"
                # Peek at shape without loading data using memory mapping
                with np.load(chunk_file, mmap_mode="r") as npz:
                    shape = npz["data"].shape
            elif chunk_file.suffix == ".npy":
                format_type = "npy"
                # Peek at shape without loading data using memory mapping
                # Note: np.load with mmap_mode returns a memmap that doesn't support 'with'
                npy = np.load(chunk_file, mmap_mode="r")
                shape = npy.shape
            elif chunk_file.suffix == ".zarr":
                format_type = "zarr"
                # For zarr, we need to open and get shape
                import zarr

                zarr_array = zarr.open(chunk_file, mode="r")
                shape = zarr_array.shape
            else:
                raise ValueError(f"Unsupported file format: {chunk_file.suffix}")

            self.chunk_formats.append(format_type)
            self.chunk_shapes.append(shape)
            total_events += shape[0]
            self.chunk_offsets.append(total_events)
            logger.info(f"  {chunk_file.name}: {shape[0]:,} events, seq_len={shape[2]}")

        self._total_events = total_events
        self._seq_len = self.chunk_shapes[0][2] - 2  # Subtract 2 for padding
        logger.info(f"Total events: {total_events:,}")

    def _setup_lazy_loading(self) -> None:
        """
        Set up lazy loading with memory-mapped numpy arrays.

        Memory Mapping Benefits:
        - Zero-copy access: OS maps file pages directly to virtual memory
        - Automatic caching: OS page cache handles hot data efficiently
        - Multiprocessing-safe: Each process gets independent memory maps
        - No explicit cache management needed
        - Memory usage bounded by OS, not by us

        The actual mmap handles are created lazily in __getitem__ or after
        unpickling in worker processes.
        """
        logger.info("Loading strategy: Lazy loading with memory-mapped arrays")
        logger.info(
            "  Memory mapping deferred until first access (multiprocessing-safe)"
        )
        logger.info(
            "  Note: Normalization will be applied on-the-fly during data access"
        )

        # Don't create mmap handles here - they can't be pickled
        # Instead, set a flag to create them lazily
        self._mmap_arrays = None  # Will be initialized on first access

    def _get_mmap_arrays(self) -> List[np.ndarray]:
        """
        Get or create memory-mapped arrays (lazy initialization).

        This is called on first data access. Creating mmaps here instead of
        in __init__ ensures each worker process creates its own independent
        memory maps after being forked/spawned.

        Supports different formats: .npz (memory-mapped), .npy (memory-mapped), .zarr (direct access)
        """
        if self._mmap_arrays is None:
            self._mmap_arrays = []
            for chunk_file, format_type in zip(self.chunk_files, self.chunk_formats):
                if format_type == "npz":
                    # mmap_mode='r' = read-only memory mapping
                    # This doesn't load data into RAM - it maps file pages to virtual memory
                    npz = np.load(chunk_file, mmap_mode="r")
                    arr = npz["data"]
                    _advise_memmap(arr)
                    self._mmap_arrays.append(arr)
                elif format_type == "npy":
                    # Direct memory mapping for .npy files
                    # Returns a numpy.memmap object for zero-copy access
                    mmap = np.load(chunk_file, mmap_mode="r")
                    _advise_memmap(mmap)
                    self._mmap_arrays.append(mmap)
                elif format_type == "zarr":
                    # Zarr doesn't support memory mapping like numpy, but we can open read-only
                    import zarr

                    zarr_array = zarr.open(chunk_file, mode="r")
                    self._mmap_arrays.append(zarr_array)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            logger.debug(f"Initialized {len(self._mmap_arrays)} data arrays")
        return self._mmap_arrays

    def _load_into_memory(self, chunk_files: List[Path]) -> None:
        """
        Load all data into RAM (high memory usage but fast access).

        Data Loading Strategy:
        - Loads all data files (.npz, .npy, .zarr) and concatenates into single torch tensors
        - Unpacks particle and feature indices
        - Stores raw (unnormalized) data in self.features
        - Normalization applied once in __init__ after this method completes
        - Provides optimal training throughput (~250k+ events/sec)
        - Fast random access via simple tensor indexing

        Memory requirements: Full dataset size in RAM (~3x data size due to torch tensors)
        """
        logger.info("Loading strategy: In-memory (fast access, high memory usage)")

        # Load and concatenate all chunks
        all_data = []
        for chunk_file, format_type in zip(chunk_files, self.chunk_formats):
            logger.info(f"  Loading {chunk_file.name} ({format_type})...")
            if format_type == "npz":
                with np.load(chunk_file) as npz:
                    chunk = npz["data"][:].copy()
            elif format_type == "npy":
                # .npy files load directly without key access
                chunk = np.load(chunk_file).copy()
            elif format_type == "zarr":
                import zarr

                chunk = zarr.open(chunk_file, mode="r")[:]
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            all_data.append(chunk)
            logger.info(f"    Loaded {chunk.shape[0]:,} events")

        data = np.concatenate(all_data, axis=0)
        logger.info(f"Total events loaded: {data.shape[0]:,}")
        logger.info(f"Total memory usage: ~{data.nbytes / 1e9:.2f} GB (numpy)")

        # Convert to torch tensors and unpack (like the old fast code)
        self.features = torch.from_numpy(data[:, 0, 2:]).float()
        packed_idxs = torch.from_numpy(data[:, 1, 2:]).long()

        self.particle_idxs = packed_idxs // 1000
        self.feature_idxs = packed_idxs % 1000

        # Free the numpy array
        del data, all_data

        logger.info(
            f"Converted to torch tensors (total ~{(self.features.numel() * 4 + self.particle_idxs.numel() * 8 + self.feature_idxs.numel() * 8) / 1e9:.2f} GB)"
        )

    # Transform type constants for LUT
    _XFORM_SKIP = 0  # apply_norm=false or padding — pass through
    _XFORM_MINMAX = 1  # MinMax only
    _XFORM_ARCSINH = 2  # arcsinh(x/mad_scale) then MinMax
    _XFORM_LOG10 = 3  # log10(1+x) then MinMax

    def _setup_normalization(self, norm_stats: Dict) -> None:
        """
        Prepare normalization LUTs indexed by feature_id for single-pass vectorized normalization.
        """
        max_id = max(norm_stats.keys()) if norm_stats else 28
        n = max_id + 1

        # Legacy attributes kept for backward compat (in-memory mode uses them)
        self.norm_min = torch.zeros(n)
        self.norm_max = torch.ones(n)
        self.norm_type = {fid: "none" for fid in range(n)}
        self.mad_scale = torch.ones(n)
        self.apply_norm = torch.ones(n, dtype=torch.bool)

        # New LUTs: transform_type, mad_scale, scale, offset (all indexed by feature_id)
        xform_type = torch.zeros(n, dtype=torch.long)  # 0=skip
        mad_scale_lut = torch.ones(n)
        norm_min_lut = torch.zeros(n)
        norm_range_lut = torch.ones(n)  # max - min (avoid div-by-zero later)

        for feat_id, stats in norm_stats.items():
            if not stats.get("apply_norm", True):
                self.apply_norm[feat_id] = False
                xform_type[feat_id] = self._XFORM_SKIP
                continue

            norm_type = stats.get("type", "none")
            self.norm_type[feat_id] = norm_type

            fmin = stats.get("min", 0.0)
            fmax = stats.get("max", 1.0)
            self.norm_min[feat_id] = fmin
            self.norm_max[feat_id] = fmax
            norm_min_lut[feat_id] = fmin
            norm_range_lut[feat_id] = max(fmax - fmin, 1e-12)

            if norm_type == "arcsinh":
                xform_type[feat_id] = self._XFORM_ARCSINH
                ms = stats.get("mad_scale")
                if ms is not None:
                    self.mad_scale[feat_id] = float(ms)
                    mad_scale_lut[feat_id] = float(ms)
                else:
                    logger.warning(
                        f"Feature {feat_id} has type 'arcsinh' but no mad_scale, using 1.0"
                    )
            elif norm_type == "log10":
                xform_type[feat_id] = self._XFORM_LOG10
            else:
                xform_type[feat_id] = self._XFORM_MINMAX

        # Store LUTs (not on GPU — workers run on CPU)
        self._xform_type_lut = xform_type
        self._mad_scale_lut = mad_scale_lut
        self._norm_min_lut = norm_min_lut
        self._norm_range_lut = norm_range_lut

        skip_count = (~self.apply_norm).sum().item()
        logger.debug(
            f"Loaded normalization for {len(norm_stats)} feature types "
            f"({skip_count} skipped via apply_norm=false)"
        )

    def __getstate__(self):
        """
        Prepare state for pickling (multiprocessing).

        Memory-mapped arrays cannot be pickled, so we exclude them.
        Each DataLoader worker will create its own memory maps after unpickling.
        """
        state = self.__dict__.copy()
        if self.lazy_loading:
            # Don't pickle mmap arrays - each worker creates its own
            state["_mmap_arrays"] = None
        return state

    def __setstate__(self, state):
        """
        Restore state after unpickling (in worker process).

        Memory maps will be lazily initialized on first data access.
        """
        self.__dict__.update(state)
        # Ensure chunk_formats exists
        if not hasattr(self, "chunk_formats"):
            # Infer formats from file extensions if not present
            self.chunk_formats = []
            for chunk_file in self.chunk_files:
                path = Path(chunk_file)
                if path.suffix == ".npz":
                    self.chunk_formats.append("npz")
                elif path.suffix == ".npy":
                    self.chunk_formats.append("npy")
                elif path.suffix == ".zarr":
                    self.chunk_formats.append("zarr")
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")

    def __len__(self) -> int:
        """Return the number of events in the dataset."""
        return self._total_events

    def _normalize_features(
        self, features: torch.Tensor, feature_idxs: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-pass LUT-based normalization.

        Uses precomputed lookup tables from _setup_normalization to avoid
        rebuilding masks (torch.isin) and list comprehensions every call.

        Args:
            features: shape (seq_len,) or (num_events, seq_len)
            feature_idxs: same shape, feature type indices

        Returns:
            Normalized features (new tensor, input unchanged)
        """
        xform = self._xform_type_lut[feature_idxs]
        mad = self._mad_scale_lut[feature_idxs]
        fmin = self._norm_min_lut[feature_idxs]
        rng = self._norm_range_lut[feature_idxs]

        missing = features == -999.0
        out = features.clone()

        # Arcsinh: arcsinh(x / mad_scale)
        m_arc = (xform == self._XFORM_ARCSINH) & ~missing
        out[m_arc] = torch.arcsinh(out[m_arc] / mad[m_arc])

        # Log10: log10(1 + x)
        m_log = (xform == self._XFORM_LOG10) & ~missing
        out[m_log] = torch.log10(1.0 + out[m_log])

        # MinMax scale to [-1, 1] for all non-skip features
        m_norm = (xform > 0) & ~missing
        out[m_norm] = (2.0 * (out[m_norm] - fmin[m_norm]) / rng[m_norm] - 1.0).clamp(
            -1.0, 1.0
        )

        # Zero out missing values and padding tokens
        out[missing | (feature_idxs == 0)] = 0.0
        return out

    def _find_chunk_for_index(self, idx: int) -> tuple:
        """
        Find which chunk contains the given global index.

        Uses binary search for efficiency with many chunks.

        Args:
            idx: Global event index

        Returns:
            Tuple of (chunk_idx, local_idx within chunk)
        """
        # Binary search for the right chunk
        import bisect

        chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
        local_idx = idx - self.chunk_offsets[chunk_idx]
        return chunk_idx, local_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single processed event.

        Behavior depends on loading mode:
        - In-memory mode: Simple tensor indexing (very fast, ~250k+ events/sec)
        - Lazy loading mode: Memory-mapped access with on-the-fly normalization

        Padding tokens (feature_idx=0) are kept as zeros.

        Args:
            idx: Event index.

        Returns:
            Dictionary containing:
                - 'features': Normalized feature values, shape (sequence_length,)
                - 'particle_idxs': Particle indices, shape (sequence_length,)
                - 'feature_idxs': Feature type indices, shape (sequence_length,)
        """
        if self.lazy_loading:
            # Lazy loading mode: Access via memory-mapped arrays
            chunk_idx, local_idx = self._find_chunk_for_index(idx)

            # Get or create memory-mapped arrays
            mmap_arrays = self._get_mmap_arrays()
            event_data = mmap_arrays[chunk_idx][local_idx]

            # Extract and convert to tensors
            # Note: We use .copy() here to create a contiguous array that's
            # independent of the mmap. This is a small copy (single event only).
            features = torch.from_numpy(event_data[0, 2:].copy()).float()
            packed_idxs = torch.from_numpy(event_data[1, 2:].copy()).long()

            particle_idxs = packed_idxs // 1000
            feature_idxs = packed_idxs % 1000

            # Apply normalization if enabled
            result = {}
            if self.normalize:
                result["features_raw"] = (
                    features  # _normalize_features clones internally
                )
                features = self._normalize_features(features, feature_idxs)
            result["features"] = features
            result["particle_idxs"] = particle_idxs
            result["feature_idxs"] = feature_idxs

            return result
        else:
            # In-memory mode: Simple tensor indexing (very fast!)
            result = {
                "features": self.features[idx],
                "particle_idxs": self.particle_idxs[idx],
                "feature_idxs": self.feature_idxs[idx],
            }
            if hasattr(self, "features_raw"):
                result["features_raw"] = self.features_raw[idx]
            return result

    def __getitems__(self, indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        """
        Optimized batch retrieval for DataLoader (PyTorch 2.0+).

        This method is called by DataLoader when fetching multiple items at once,
        enabling more efficient I/O by batching memory-mapped reads.

        For .npy files with memory mapping, this provides significant speedup
        by reducing the number of separate I/O operations.

        Optimization: Uses vectorized batch normalization instead of per-item
        normalization, significantly reducing Python overhead and leveraging
        efficient tensor operations across the entire batch.

        Args:
            indices: List of event indices to retrieve.

        Returns:
            List of dictionaries, each containing features, particle_idxs, feature_idxs.
        """
        if not self.lazy_loading:
            # In-memory mode: just use regular indexing
            return [self[idx] for idx in indices]

        # Lazy loading mode: batch by chunk for efficiency
        # Group indices by chunk to minimize I/O operations
        from collections import defaultdict

        chunk_groups = defaultdict(list)
        for original_pos, idx in enumerate(indices):
            chunk_idx, local_idx = self._find_chunk_for_index(idx)
            chunk_groups[chunk_idx].append((original_pos, local_idx))

        # Get memory-mapped arrays
        mmap_arrays = self._get_mmap_arrays()

        # Collect all data first, then batch normalize
        all_features = []
        all_particle_idxs = []
        all_feature_idxs = []
        original_positions = []

        # Process each chunk's indices together - batch I/O
        for chunk_idx, positions_and_local_indices in chunk_groups.items():
            mmap_array = mmap_arrays[chunk_idx]

            # Touch the chunk's events in monotonic file order. The output is
            # reordered back via `positions`, so this is content-neutral but
            # avoids backward seeks within the chunk for scattered batches.
            if _SORT_CHUNK_READS:
                positions_and_local_indices = sorted(
                    positions_and_local_indices, key=lambda pl: pl[1]
                )

            # Extract local indices for batch access
            local_indices = [li for _, li in positions_and_local_indices]
            positions = [pos for pos, _ in positions_and_local_indices]

            # Batch read from memory-mapped array (more efficient than individual reads)
            # This leverages OS-level read-ahead and caching
            batch_data = mmap_array[local_indices]

            # Extract features and indices for all events in this chunk
            # Shape: (batch_size, seq_len)
            chunk_features = torch.from_numpy(batch_data[:, 0, 2:].copy()).float()
            chunk_packed_idxs = torch.from_numpy(batch_data[:, 1, 2:].copy()).long()

            chunk_particle_idxs = chunk_packed_idxs // 1000
            chunk_feature_idxs = chunk_packed_idxs % 1000

            all_features.append(chunk_features)
            all_particle_idxs.append(chunk_particle_idxs)
            all_feature_idxs.append(chunk_feature_idxs)
            original_positions.extend(positions)

        # Concatenate all chunks for batch normalization
        # Shape: (total_batch_size, seq_len)
        batch_features = torch.cat(all_features, dim=0)
        batch_particle_idxs = torch.cat(all_particle_idxs, dim=0)
        batch_feature_idxs = torch.cat(all_feature_idxs, dim=0)

        # Apply vectorized batch normalization (much faster than per-item)
        batch_features_raw = None
        if self.normalize:
            batch_features_raw = batch_features  # _normalize_features clones internally
            batch_features = self._normalize_features(
                batch_features, batch_feature_idxs
            )

        # Split back into individual results in original order
        results = [None] * len(indices)
        for i, orig_pos in enumerate(original_positions):
            result = {
                "features": batch_features[i],
                "particle_idxs": batch_particle_idxs[i],
                "feature_idxs": batch_feature_idxs[i],
            }
            if batch_features_raw is not None:
                result["features_raw"] = batch_features_raw[i]
            results[orig_pos] = result

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return dataset statistics computed from a sample.

        Returns:
            Dictionary containing dataset statistics.
        """
        # Compute stats from a sample to avoid loading entire dataset
        sample_size = min(1000, len(self))
        sample_indices = np.random.choice(len(self), sample_size, replace=False)

        non_padding_counts = []
        for idx in sample_indices:
            item = self[idx]
            non_padding = (item["feature_idxs"] != 0).sum().item()
            non_padding_counts.append(non_padding)

        return {
            "num_events": len(self),
            "sequence_length": self._seq_len,
            "avg_non_padding_tokens": float(np.mean(non_padding_counts)),
            "min_non_padding_tokens": int(np.min(non_padding_counts)),
            "max_non_padding_tokens": int(np.max(non_padding_counts)),
            "note": "Statistics computed from sample of 1000 events",
        }
