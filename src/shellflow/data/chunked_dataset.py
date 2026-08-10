"""
Chunked PyTorch Dataset for large-scale particle physics data.

This module provides a Dataset class that lazily loads data chunks on demand,
enabling training on datasets larger than available memory.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)


class ChunkedParticleDataset(Dataset):
    """
    PyTorch Dataset that lazily loads chunks of particle physics data.

    Unlike ParticleDataset which loads all data into memory, this class
    only loads one chunk at a time, allowing training on datasets much
    larger than available RAM.

    Args:
        data_dir: Directory containing the '.npz' chunk files
        config_path: Path to normalization config YAML file
        normalize: Whether to apply feature normalization
        file_pattern: Glob pattern for data files (default: '*.npz')
        cache_size: Number of chunks to keep in memory (default: 1)
    """

    MISSING_VALUE = -999.0

    def __init__(
        self,
        data_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        file_pattern: str = "*.npz",
        cache_size: int = 1,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.cache_size = cache_size

        # Find all chunk files
        self.chunk_files = sorted(self.data_dir.glob(file_pattern))
        if not self.chunk_files:
            raise FileNotFoundError(
                f"No data files matching '{file_pattern}' found in {data_dir}"
            )

        logger.info(f"Found {len(self.chunk_files)} chunk files in {data_dir}")

        # Load normalization config
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent
                / "config"
                / "normalization_stats_4lepton.yaml"
            )
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if normalize:
            self._setup_normalization(self.config["normalization"])

        # Probe first file to get metadata
        probe_data = np.load(self.chunk_files[0])["data"]
        self.events_per_chunk = probe_data.shape[0]
        self.seq_len = probe_data.shape[2]
        del probe_data

        # Build index mapping: global_idx -> (chunk_idx, local_idx)
        # For efficiency, we assume all chunks have same size (except possibly last)
        self._build_index()

        # Chunk cache
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cache_order: List[int] = []  # LRU tracking

        logger.info("ChunkedParticleDataset initialized:")
        logger.info(f"  Total events: {len(self):,}")
        logger.info(f"  Sequence length: {self.seq_len}")
        logger.info(f"  Chunks: {len(self.chunk_files)}")
        logger.info(f"  Events per chunk: ~{self.events_per_chunk:,}")

    def _build_index(self):
        """Build mapping from global index to (chunk_idx, local_idx).

        Uses fast indexing: assumes all chunks have the same size as the first
        chunk (already probed), except the last one which we check separately.
        This avoids slow mmap on network storage.
        """
        self.chunk_sizes = []
        self.chunk_offsets = [0]

        n_chunks = len(self.chunk_files)

        # Fast path: assume all chunks except last have same size as first
        # Only probe last chunk if different from first
        for i in range(n_chunks - 1):
            self.chunk_sizes.append(self.events_per_chunk)
            self.chunk_offsets.append(self.chunk_offsets[-1] + self.events_per_chunk)

        # Probe last chunk for its actual size (may be smaller)
        if n_chunks > 1:
            last_data = np.load(self.chunk_files[-1])["data"]
            last_size = last_data.shape[0]
            del last_data
        else:
            last_size = self.events_per_chunk

        self.chunk_sizes.append(last_size)
        self.chunk_offsets.append(self.chunk_offsets[-1] + last_size)

        self._total_events = self.chunk_offsets[-1]
        logger.info(f"  Indexed {self._total_events:,} events across {n_chunks} chunks")

    def _setup_normalization(self, norm_stats: Dict) -> None:
        """Prepare normalization tensors."""
        max_id = max(norm_stats.keys()) if norm_stats else 53
        self.norm_min = torch.zeros(max_id + 1)
        self.norm_max = torch.ones(max_id + 1)
        self.norm_type = {fid: "none" for fid in range(max_id + 1)}
        self.log_offset = torch.ones(max_id + 1)

        for feat_id, stats in norm_stats.items():
            norm_type = stats.get("type", "none")
            self.norm_type[feat_id] = norm_type
            self.norm_min[feat_id] = stats.get("min", 0.0)
            self.norm_max[feat_id] = stats.get("max", 1.0)
            if norm_type == "log10":
                self.log_offset[feat_id] = stats.get("log_offset", 1.0)

    def _load_chunk(self, chunk_idx: int) -> Dict[str, torch.Tensor]:
        """Load and preprocess a single chunk."""
        if chunk_idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(chunk_idx)
            self._cache_order.append(chunk_idx)
            return self._cache[chunk_idx]

        # Load new chunk
        chunk_file = self.chunk_files[chunk_idx]
        logger.debug(f"Loading chunk {chunk_idx}: {chunk_file.name}")

        data = np.load(chunk_file)["data"]

        features = torch.from_numpy(data[:, 0, :]).float()
        packed_idxs = torch.from_numpy(data[:, 1, :]).long()

        particle_idxs = packed_idxs // 1000
        feature_idxs = packed_idxs % 1000

        del data  # Free memory

        if self.normalize:
            features = self._apply_normalization(features, feature_idxs)

        chunk_data = {
            "features": features,
            "particle_idxs": particle_idxs,
            "feature_idxs": feature_idxs,
        }

        # Add to cache
        self._cache[chunk_idx] = chunk_data
        self._cache_order.append(chunk_idx)

        # Evict if over capacity
        while len(self._cache) > self.cache_size:
            old_idx = self._cache_order.pop(0)
            del self._cache[old_idx]

        return chunk_data

    def _apply_normalization(
        self, features: torch.Tensor, feature_idxs: torch.Tensor
    ) -> torch.Tensor:
        """Apply normalization to features."""
        # Identify missing values
        missing_mask = features == self.MISSING_VALUE

        # Apply log10 transformation for log10 features
        log10_fids = [fid for fid, typ in self.norm_type.items() if typ == "log10"]
        if log10_fids:
            log10_mask = torch.isin(feature_idxs, torch.tensor(log10_fids))
            offsets = self.log_offset[feature_idxs]
            valid_for_log = ((features + offsets) > 0) & log10_mask & ~missing_mask
            features = features.clone()
            features[valid_for_log] = torch.log10(
                offsets[valid_for_log] + features[valid_for_log]
            )

        # Apply MinMax scaling to [-1, 1]
        non_padding_mask = feature_idxs != 0
        valid_for_norm = non_padding_mask & ~missing_mask

        mins = self.norm_min[feature_idxs]
        maxs = self.norm_max[feature_idxs]
        valid_scale = (maxs > mins) & valid_for_norm

        features = torch.where(
            valid_scale,
            2 * (features - mins) / (maxs - mins) - 1,
            torch.where(valid_for_norm, torch.zeros_like(features), features),
        )

        # Set missing values to 0.0
        features = features.masked_fill(missing_mask, 0.0)

        # Ensure padding remains 0.0
        padding_mask = feature_idxs == 0
        features = features.masked_fill(padding_mask, 0.0)

        return features

    def _global_to_local(self, idx: int) -> tuple:
        """Convert global index to (chunk_idx, local_idx)."""
        # Linear scan over chunk offsets (chunk counts are small)
        for chunk_idx, offset in enumerate(self.chunk_offsets[:-1]):
            next_offset = self.chunk_offsets[chunk_idx + 1]
            if offset <= idx < next_offset:
                return chunk_idx, idx - offset
        raise IndexError(f"Index {idx} out of range")

    def __len__(self) -> int:
        return self._total_events

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single event."""
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        chunk_idx, local_idx = self._global_to_local(idx)
        chunk = self._load_chunk(chunk_idx)

        return {
            "features": chunk["features"][local_idx],
            "particle_idxs": chunk["particle_idxs"][local_idx],
            "feature_idxs": chunk["feature_idxs"][local_idx],
        }
