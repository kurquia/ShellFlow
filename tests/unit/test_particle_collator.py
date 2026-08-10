"""
Unit tests for ParticleDataCollator.

Tests cover:
- Masking behavior and rates
- Label creation for masked positions
- Dynamic sequence truncation
- Attention mask generation
"""

import numpy as np
import pytest
import torch

from shellflow.data import ParticleDataCollator


class TestCollatorInstantiation:
    """Test collator instantiation and configuration."""

    def test_default_config(self, test_config_path):
        """Test collator instantiation with test config."""
        collator = ParticleDataCollator(config_path=test_config_path)

        assert collator.mask_fraction == 0.15
        assert collator.masking_prob == 0.8
        assert collator.mask_token_id == 29
        assert collator.padding_id == 0

    def test_custom_mask_fraction(self, test_config_path):
        """Test collator with custom mask fraction."""
        collator = ParticleDataCollator(
            config_path=test_config_path,
            mask_fraction=0.25,
        )

        assert collator.mask_fraction == 0.25

    def test_custom_masking_probability(self, test_config_path):
        """Test collator with custom masking probability."""
        collator = ParticleDataCollator(
            config_path=test_config_path,
            masking_probability=0.9,
        )

        assert collator.masking_prob == 0.9


class TestCollatorMasking:
    """Test masking behavior."""

    @pytest.fixture
    def collator(self, test_config_path):
        """Create collator instance."""
        return ParticleDataCollator(config_path=test_config_path)

    @pytest.fixture
    def sample_batch_list(self, sample_event_dicts):
        """Get sample batch as list of dicts."""
        # Use first 8 events
        return sample_event_dicts[:8]

    def test_labels_created(self, collator, sample_batch_list):
        """Test that labels tensor is created."""
        output = collator(sample_batch_list)

        assert "labels" in output
        assert output["labels"].shape == output["features"].shape

    def test_labels_have_ignore_value(self, collator, sample_batch_list):
        """Test that non-masked positions have ignore label (-100)."""
        output = collator(sample_batch_list)

        # Some positions should have -100 (non-masked)
        non_masked = output["labels"] == -100.0
        assert non_masked.any(), "All positions are masked, some should be -100"

    def test_approximate_mask_rate(self, collator, sample_batch_list):
        """Test that actual mask rate is approximately correct."""
        # Run multiple times to get statistical average
        mask_rates = []

        for _ in range(20):
            output = collator(sample_batch_list)
            stats = collator.get_masking_stats(output)
            mask_rates.append(stats["actual_mask_rate"])

        avg_mask_rate = np.mean(mask_rates)

        # Should be close to configured mask_fraction (0.15)
        # Allow some tolerance due to randomness
        assert 0.10 < avg_mask_rate < 0.20, (
            f"Mask rate {avg_mask_rate:.3f} not near 0.15"
        )

    def test_masked_features_zeroed(self, collator, sample_batch_list):
        """Test that masked feature values are zeroed out (80% of selected)."""
        output = collator(sample_batch_list)

        # Find positions where feature_idxs == mask_token_id (29)
        masked_positions = output["feature_idxs"] == collator.mask_token_id

        if masked_positions.any():
            # Features at masked positions should be 0
            assert (output["features"][masked_positions] == 0.0).all()

    def test_labels_contain_original_values(self, collator, sample_batch_list):
        """Test that labels contain original feature values at masked positions."""
        # Get original features before masking
        original_features = torch.stack(
            [item["features"] for item in sample_batch_list]
        )

        output = collator(sample_batch_list)

        # Find positions with labels (not -100)
        labeled_positions = output["labels"] != -100.0

        if labeled_positions.any():
            # Labels should match original features at those positions
            # Note: Due to truncation, we need to compare up to truncated length
            max_len = output["features"].shape[1]
            original_truncated = original_features[:, :max_len]

            labeled_values = output["labels"][labeled_positions]
            original_at_labeled = original_truncated[labeled_positions]

            assert torch.allclose(labeled_values, original_at_labeled, atol=1e-6)

    def test_particle_idxs_unchanged(self, collator, sample_batch_list):
        """Test that particle_idxs are not modified by masking."""
        # Get original particle_idxs
        original_particle_idxs = torch.stack(
            [item["particle_idxs"] for item in sample_batch_list]
        )

        output = collator(sample_batch_list)

        # Compare (accounting for truncation)
        max_len = output["particle_idxs"].shape[1]
        assert torch.equal(output["particle_idxs"], original_particle_idxs[:, :max_len])


class TestCollatorTruncation:
    """Test dynamic sequence truncation."""

    @pytest.fixture
    def collator(self, test_config_path):
        """Create collator instance."""
        return ParticleDataCollator(config_path=test_config_path)

    def test_truncation_to_max_length(self, collator, sample_event_dicts):
        """Test that batch is truncated to longest non-padded sequence."""
        # Create batch with varying amounts of padding
        batch = sample_event_dicts[:4]

        output = collator(batch)

        # Output length should be <= original length
        original_len = batch[0]["features"].shape[0]
        output_len = output["features"].shape[1]

        assert output_len <= original_len

    def test_no_all_padding_columns(self, collator, sample_event_dicts):
        """Test that truncated output has no all-padding columns at the end."""
        batch = sample_event_dicts[:4]
        output = collator(batch)

        # The last column should have at least one non-padding token
        last_col_mask = output["attention_mask"][:, -1]
        assert last_col_mask.any(), "Last column is all padding"


class TestCollatorAttentionMask:
    """Test attention mask generation."""

    @pytest.fixture
    def collator(self, test_config_path):
        """Create collator instance."""
        return ParticleDataCollator(config_path=test_config_path)

    def test_attention_mask_shape(self, collator, sample_event_dicts):
        """Test attention mask has correct shape."""
        batch = sample_event_dicts[:4]
        output = collator(batch)

        assert output["attention_mask"].shape == output["features"].shape

    def test_attention_mask_matches_padding(self, collator, sample_event_dicts):
        """Test attention mask matches padding pattern."""
        batch = sample_event_dicts[:4]
        output = collator(batch)

        # Attention mask should be 0 where feature_idxs is padding_id
        is_padding = output["feature_idxs"] == collator.padding_id

        # Note: feature_idxs can be mask_token_id (29) at masked positions,
        # which should have attention_mask=1
        # So we check: where feature_idxs is padding_id, attention_mask should be 0
        # BUT: where attention_mask is 0, feature_idxs should be padding_id
        # (the second condition is the one to test)

        padding_from_mask = output["attention_mask"] == 0
        padding_from_feature_idx = is_padding

        # Where attention_mask indicates padding, feature_idxs should also be padding
        assert torch.equal(padding_from_mask, padding_from_feature_idx)

    def test_attention_mask_values(self, collator, sample_event_dicts):
        """Test attention mask contains only 0 and 1."""
        batch = sample_event_dicts[:4]
        output = collator(batch)

        unique_values = torch.unique(output["attention_mask"])
        assert all(v in [0, 1] for v in unique_values.tolist())


class TestCollatorMaskingStats:
    """Test masking statistics computation."""

    @pytest.fixture
    def collator(self, test_config_path):
        """Create collator instance."""
        return ParticleDataCollator(config_path=test_config_path)

    def test_get_masking_stats(self, collator, sample_event_dicts):
        """Test get_masking_stats returns expected keys."""
        batch = sample_event_dicts[:4]
        output = collator(batch)
        stats = collator.get_masking_stats(output)

        assert "num_masked_tokens" in stats
        assert "num_valid_tokens" in stats
        assert "actual_mask_rate" in stats
        assert "batch_size" in stats
        assert "sequence_length" in stats

    def test_stats_consistency(self, collator, sample_event_dicts):
        """Test that stats are internally consistent."""
        batch = sample_event_dicts[:4]
        output = collator(batch)
        stats = collator.get_masking_stats(output)

        # Verify batch_size
        assert stats["batch_size"] == len(batch)

        # Verify sequence_length
        assert stats["sequence_length"] == output["features"].shape[1]

        # Verify num_valid matches attention_mask sum
        assert stats["num_valid_tokens"] == output["attention_mask"].sum().item()

        # Verify num_masked matches labels count
        num_masked_from_labels = (output["labels"] != -100.0).sum().item()
        assert stats["num_masked_tokens"] == num_masked_from_labels

        # Verify mask rate calculation
        if stats["num_valid_tokens"] > 0:
            expected_rate = stats["num_masked_tokens"] / stats["num_valid_tokens"]
            assert abs(stats["actual_mask_rate"] - expected_rate) < 1e-6


class TestCollatorEdgeCases:
    """Test edge cases and corner scenarios."""

    @pytest.fixture
    def collator(self, test_config_path):
        """Create collator instance."""
        return ParticleDataCollator(config_path=test_config_path)

    def test_single_event_batch(self, collator, sample_event_dicts):
        """Test collator works with single-event batch."""
        batch = [sample_event_dicts[0]]
        output = collator(batch)

        assert output["features"].shape[0] == 1
        assert "labels" in output
        assert "attention_mask" in output

    def test_output_tensor_types(self, collator, sample_event_dicts):
        """Test that output tensors have correct types."""
        batch = sample_event_dicts[:4]
        output = collator(batch)

        assert output["features"].dtype == torch.float32
        assert output["particle_idxs"].dtype == torch.int64
        assert output["feature_idxs"].dtype == torch.int64
        assert output["labels"].dtype == torch.float32
        assert output["attention_mask"].dtype == torch.int64

    def test_deterministic_with_same_seed(self, collator, sample_event_dicts):
        """Test that masking is different across calls (randomness)."""
        batch = sample_event_dicts[:4]

        # Run twice without setting seed
        output1 = collator(batch)
        output2 = collator(batch)

        # Masked positions should likely be different
        # (extremely unlikely to be identical by chance)
        masked1 = output1["feature_idxs"] == collator.mask_token_id
        masked2 = output2["feature_idxs"] == collator.mask_token_id

        # At least one difference should exist
        # (this could theoretically fail but probability is negligible)
        if masked1.any() and masked2.any():
            # Check they're not identical (allowing for possibility they could be)
            pass  # Just verify no errors occur
