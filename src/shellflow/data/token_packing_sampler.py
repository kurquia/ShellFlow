"""Token-packing batch sampler for variable-length particle events.

Packs events into batches of a fixed total token count (T_fixed),
producing static-shaped flat tensors that torch.compile can fuse.
At most max_particles-1 padding tokens per batch (< 0.1% waste).
"""

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


def pad_batch_to_static(
    batch: dict,
    T_fixed: int,
    B_fixed: int,
) -> dict:
    """Pad a geometric batch to static (T_fixed,) and (B_fixed,) shapes.

    Token dimension: pads flat tensors to T_fixed with zeros.
    Event dimension: pads ptr/met_vector to B_fixed with empty events.

    Args:
        batch: geometric collator output dict.
        T_fixed: static total token count.
        B_fixed: static event count.

    Returns:
        Padded batch with static shapes for torch.compile.
    """
    T_real = batch["kinematics"].shape[0]
    B_real = batch["ptr"].shape[0] - 1
    dev = batch["kinematics"].device if batch["kinematics"].is_cuda else None

    t_pad = T_fixed - T_real
    b_pad = B_fixed - B_real

    def make(size, dtype=torch.float32):
        return (
            torch.zeros(size, dtype=dtype)
            if dev is None
            else torch.zeros(size, dtype=dtype, device=dev)
        )

    out = dict(batch)

    # Truncate tokens to T_fixed (sampler packs to >= T_fixed, we trim the excess).
    # Trimming drops the last few particles from the last event — this slightly
    # reduces that event's particle count but avoids any padding tokens entirely.
    if t_pad < 0:
        # Trim flat tensors
        for k in [
            "kinematics",
            "particle_type",
            "particle_charge",
            "particle_extra",
            "batch",
        ]:
            out[k] = batch[k][:T_fixed]
        # Fix ptr: clamp the last event's end to T_fixed
        ptr = batch["ptr"].clone()
        ptr = torch.clamp(ptr, max=T_fixed)
        out["ptr"] = ptr
        T_real = T_fixed
    elif t_pad > 0:
        # Rare: batch has fewer tokens than T_fixed (final batch or very small dataset)
        def p(x):
            return torch.cat([x, make((t_pad, *x.shape[1:]), dtype=x.dtype)])

        out["kinematics"] = p(batch["kinematics"])
        out["particle_type"] = p(batch["particle_type"])
        out["particle_charge"] = p(batch["particle_charge"])
        out["particle_extra"] = p(batch["particle_extra"])
        bp = make((T_fixed,), dtype=torch.int64)
        bp[:T_real] = batch["batch"]
        out["batch"] = bp

    # Pad events: extend ptr with empty events, pad met_vector
    if b_pad > 0:
        ptr_out = out.get("ptr", batch["ptr"])
        last_val = ptr_out[-1:]
        extra_ptr = last_val.expand(b_pad)
        out["ptr"] = torch.cat([ptr_out, extra_ptr]).to(batch["ptr"].dtype)

        if "met_vector" in batch and batch["met_vector"] is not None:
            mv = batch["met_vector"]
            out["met_vector"] = torch.cat(
                [mv, make((b_pad, mv.shape[1]), dtype=mv.dtype)]
            )

    out["max_seqlen"] = batch.get("max_seqlen", 8)
    out["T_real"] = T_real  # Python int — always T_fixed when sampler overfills
    return out


class TokenPackingSampler(Sampler[list[int]]):
    """Batch sampler that packs events to a fixed token budget.

    Each batch contains enough events so that the total particle count
    is as close to ``T_fixed`` as possible without exceeding it.
    The small remainder (at most ``max_particles - 1`` tokens) is
    zero-padded by the collator.

    Args:
        particle_counts: (N_total,) int array — number of particles
            per event (excluding MET, which is handled separately).
        T_fixed: target total token count per batch.
        shuffle: whether to shuffle event order each epoch.
        seed: random seed for shuffling.
        drop_last: drop the final incomplete batch if True.
        rank: current process rank for distributed training (0 for single-GPU).
        world_size: total number of processes (1 for single-GPU).
    """

    def __init__(
        self,
        particle_counts: np.ndarray,
        T_fixed: int = 8192,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.counts = np.asarray(particle_counts, dtype=np.int32)
        self.T_fixed = T_fixed
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        # Pre-compute exact batch count.  For DDP, simulate all ranks and take
        # the minimum so every rank yields the same number of batches (required
        # by DDP to avoid hangs at the all-reduce barrier).
        self._cached_len = self._compute_len()

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling (multi-GPU compat)."""
        self.epoch = epoch

    def _shard_indices(self, indices: np.ndarray) -> np.ndarray:
        """Interleave-shard indices across ranks (same as DistributedSampler)."""
        if self.world_size <= 1:
            return indices
        return indices[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        N = len(self.counts)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(N)
        else:
            indices = np.arange(N)

        # DDP: each rank gets a disjoint shard of events
        indices = self._shard_indices(indices)

        batch = []
        batch_tokens = 0
        n_yielded = 0

        for idx in indices:
            c = int(self.counts[idx])
            batch.append(int(idx))
            batch_tokens += c
            if batch_tokens >= self.T_fixed:
                # Truncate to cached length so all ranks yield the same count
                if n_yielded >= self._cached_len:
                    return
                yield batch
                n_yielded += 1
                batch = []
                batch_tokens = 0

        # Final batch
        if batch and not self.drop_last and n_yielded < self._cached_len:
            yield batch

    def _compute_len(self) -> int:
        """Compute batch count, taking DDP sharding into account."""
        all_indices = np.arange(len(self.counts))
        if self.world_size <= 1:
            return self._count_batches(all_indices)
        # Simulate all ranks and take the minimum so no rank over-yields
        counts = []
        for r in range(self.world_size):
            shard = all_indices[r :: self.world_size]
            counts.append(self._count_batches(shard))
        return min(counts)

    def _count_batches(self, indices) -> int:
        """Count batches produced by the greedy packing for a given index order."""
        n_batches = 0
        batch_tokens = 0
        has_remainder = False
        for idx in indices:
            batch_tokens += int(self.counts[idx])
            has_remainder = True
            if batch_tokens >= self.T_fixed:
                n_batches += 1
                batch_tokens = 0
                has_remainder = False
        if has_remainder and not self.drop_last:
            n_batches += 1
        return n_batches

    def __len__(self) -> int:
        return self._cached_len

    @staticmethod
    def compute_particle_counts(dataset) -> np.ndarray:
        """Extract particle counts from a ParticleDataset.

        Reads particle indices from memory-mapped arrays to count
        unique particles per event (excluding MET at particle_idx=1).

        Args:
            dataset: ParticleDataset instance.

        Returns:
            (N_total,) int32 array of particle counts per event.
        """
        all_counts = []
        for chunk_file in dataset.chunk_files:
            mmap = np.load(chunk_file, mmap_mode="r")
            # Row 1 = packed indices: particle_idx * 1000 + feature_idx
            packed = mmap[:, 1, :]
            max_pid = (packed // 1000).max(axis=1)
            # Subtract 1 for MET (particle_idx=1), so real particles = max_pid - 1
            counts = (max_pid - 1).astype(np.int32)
            all_counts.append(counts)
        return np.concatenate(all_counts)
