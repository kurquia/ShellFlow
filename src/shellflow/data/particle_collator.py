"""
Data collator for BERT-style masked language modeling on particle features.

This module implements a collator that batches events and applies masking
for self-supervised learning.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import yaml

from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)


class ParticleDataCollator:
    """
    Data collator for BERT-style masked feature prediction on particle data.

    This collator takes a batch of events and applies masking to a fraction of tokens:
    - `particle_idxs` are left unchanged (positional information preserved)
    - `feature_idxs` for selected tokens are replaced with mask_token_id
    - `features` for selected tokens are zeroed out
    - Creates `labels` tensor containing original feature values for masked positions

    Masking follows the 80-20 rule (configurable):
    - 80% of selected tokens: replaced with [MASK]
    - 20% of selected tokens: left unchanged

    The collator also dynamically truncates each batch to the longest non-padded
    sequence in that batch, minimizing wasted computation.

    Design Rationale for Feature Masking:
        For continuous-valued features, we zero them out at masked positions and
        train the model to predict the original values. This is similar to denoising
        autoencoders and has been shown to work well for continuous data. The model
        learns to leverage context from surrounding particles and features.

        Alternative approaches considered:
        1. Replace with random values from feature distribution (too noisy)
        2. Replace with mean value (too obvious, less effective learning signal)
        3. Zero-out (chosen): Simple, effective, model must rely on context

    Attributes:
        mask_fraction (float): Fraction of tokens to select for masking (e.g., 0.15)
        masking_prob (float): Of selected tokens, fraction to actually mask (e.g., 0.80)
        mask_token_id (int): ID to use for [MASK] token in feature_idxs
        padding_id (int): ID used for padding tokens
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        mask_fraction: Optional[float] = None,
        masking_probability: Optional[float] = None,
    ):
        """
        Initialize the ParticleDataCollator.

        Args:
            config_path: Path to config yaml file. If None, uses default location.
            mask_fraction: Override mask_fraction from config.
            masking_probability: Override masking_probability from config.
        """
        # Load configuration
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent / "config" / "dataloader_config.yaml"
            )
        else:
            config_path = Path(config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        masking_config = config["masking"]

        # Use overrides if provided, otherwise use config values
        self.mask_fraction = (
            mask_fraction
            if mask_fraction is not None
            else masking_config["mask_fraction"]
        )
        self.masking_prob = (
            masking_probability
            if masking_probability is not None
            else masking_config["masking_probability"]
        )
        self.mask_token_id = masking_config["mask_token_id"]
        self.padding_id = masking_config["padding_id"]

        logger.info(
            f"ParticleDataCollator initialized: mask_fraction={self.mask_fraction:.2f}, "
            f"masking_prob={self.masking_prob:.2f}"
        )

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of events and apply masking.

        Args:
            batch: List of event dictionaries from ParticleDataset.

        Returns:
            Dictionary containing:
                - 'features': Masked feature values, shape (batch_size, seq_len)
                - 'particle_idxs': Particle indices (unchanged), shape (batch_size, seq_len)
                - 'feature_idxs': Feature indices with masks, shape (batch_size, seq_len)
                - 'labels': Original feature values (masked positions only),
                           shape (batch_size, seq_len), with -100.0 for non-masked positions
                - 'attention_mask': Binary mask (1=valid, 0=padding), shape (batch_size, seq_len)

        Note:
            The returned tensors are dynamically truncated to the length of the
            longest non-padded sequence in this specific batch.
        """
        # Stack batch into tensors
        features = torch.stack([item["features"] for item in batch])
        particle_idxs = torch.stack([item["particle_idxs"] for item in batch])
        feature_idxs = torch.stack([item["feature_idxs"] for item in batch])

        # --- Dynamic Sequence Truncation ---
        # Find the last non-padded token position in the entire batch
        # A token is non-padded if its feature_idx is not padding_id (0)
        is_non_padding = feature_idxs != self.padding_id

        # Create attention mask (1 for valid tokens, 0 for padding)
        attention_mask = is_non_padding.long()

        # Find the maximum length needed for this batch
        # This is the position of the last non-padding token + 1
        if is_non_padding.any():
            # Get indices of non-padding tokens
            non_pad_indices = torch.where(is_non_padding)
            # Max position along sequence dimension
            max_len_in_batch = non_pad_indices[1].max().item() + 1
        else:
            # Edge case: entire batch is padding (shouldn't happen in practice)
            max_len_in_batch = 1

        # Truncate all tensors to this dynamic max length
        features = features[:, :max_len_in_batch].clone()
        particle_idxs = particle_idxs[:, :max_len_in_batch].clone()
        feature_idxs = feature_idxs[:, :max_len_in_batch].clone()
        attention_mask = attention_mask[:, :max_len_in_batch]
        is_non_padding = is_non_padding[:, :max_len_in_batch]
        # --- End Truncation ---

        # --- BERT-style Masking ---
        # Create labels tensor: will contain original feature values at masked positions,
        # and -100.0 at non-masked positions (standard ignore index for loss functions)
        labels = torch.full_like(features, fill_value=-100.0)

        # Randomly select tokens for masking (only from non-padding tokens)
        mask_selection = (
            torch.rand(features.shape) < self.mask_fraction
        ) & is_non_padding

        # Store original features at selected positions in labels
        labels[mask_selection] = features[mask_selection]

        # Of the selected tokens, mask only a fraction (80-20 rule)
        # The rest are left unchanged to help the model learn
        mask_replace = (torch.rand(features.shape) < self.masking_prob) & mask_selection

        # Apply masking:
        # 1. Zero out feature values at masked positions
        features[mask_replace] = 0.0

        # 2. Replace feature_idxs with mask_token_id at masked positions
        feature_idxs[mask_replace] = self.mask_token_id

        # Note: particle_idxs remain unchanged to preserve positional information
        # --- End Masking ---

        return {
            "features": features,
            "particle_idxs": particle_idxs,
            "feature_idxs": feature_idxs,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def get_masking_stats(
        self, batch_output: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute statistics about masking for a collated batch.

        Args:
            batch_output: Output from __call__.

        Returns:
            Dictionary with masking statistics.
        """
        labels = batch_output["labels"]
        attention_mask = batch_output["attention_mask"]

        # Count masked tokens (labels != -100)
        masked = labels != -100.0
        num_masked = masked.sum().item()

        # Count total non-padding tokens
        num_valid = attention_mask.sum().item()

        # Compute actual masking rate
        mask_rate = num_masked / num_valid if num_valid > 0 else 0.0

        return {
            "num_masked_tokens": num_masked,
            "num_valid_tokens": num_valid,
            "actual_mask_rate": mask_rate,
            "batch_size": labels.shape[0],
            "sequence_length": labels.shape[1],
        }
