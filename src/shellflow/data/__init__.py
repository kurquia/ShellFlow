"""
Data loading and processing modules for particle physics ML.

PyTorch Dataset, collator, and sampler implementations for training on
packed particle-physics event data.
"""

from shellflow.data.chunked_dataset import ChunkedParticleDataset
from shellflow.data.particle_collator import ParticleDataCollator
from shellflow.data.particle_dataset import BackgroundPrefetcher, ParticleDataset
from shellflow.data.particle_vector_collator import (
    ParticleVectorCollator,
    ParticleVectorCollatorGeometric,
    geometric_to_dense,
)
from shellflow.data.token_packing_sampler import (
    TokenPackingSampler,
    pad_batch_to_static,
)

__all__ = [
    "ParticleDataset",
    "ParticleDataCollator",
    "ParticleVectorCollator",
    "ParticleVectorCollatorGeometric",
    "geometric_to_dense",
    "TokenPackingSampler",
    "pad_batch_to_static",
    "ChunkedParticleDataset",
    "BackgroundPrefetcher",
]
