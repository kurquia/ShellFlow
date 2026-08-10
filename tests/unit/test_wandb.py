"""
Tests for W&B integration with proper mocking.

These tests verify W&B logging workflows without actually calling W&B servers.
"""

from unittest.mock import MagicMock, patch


class TestWandbIntegration:
    """Test W&B integration with mocking."""

    def test_wandb_init_and_log(self):
        """Test basic W&B init and logging workflow."""
        with (
            patch("wandb.init") as mock_init,
            patch("wandb.log") as mock_log,
            patch("wandb.finish") as mock_finish,
        ):
            # Setup mock return values
            mock_run = MagicMock()
            mock_run.url = "https://wandb.ai/test/run/123"
            mock_init.return_value = mock_run

            # Simulate init
            import wandb

            wandb.init(project="test-project", config={"learning_rate": 0.001})

            # Simulate logging
            wandb.log({"loss": 0.5, "accuracy": 0.8})
            wandb.log({"loss": 0.3, "accuracy": 0.9})

            # Simulate finish
            wandb.finish()

            # Verify calls
            mock_init.assert_called_once()
            assert mock_log.call_count == 2
            mock_finish.assert_called_once()

    def test_wandb_config_tracking(self):
        """Test that config is properly tracked."""
        with patch("wandb.init") as mock_init:
            mock_run = MagicMock()
            mock_run.config = {}
            mock_init.return_value = mock_run

            import wandb

            config = {
                "learning_rate": 0.02,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 10,
            }

            wandb.init(project="shellflow-test", config=config)

            # Verify init was called with correct project
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["project"] == "shellflow-test"
            assert call_kwargs["config"] == config

    def test_wandb_simulated_training_loop(self):
        """Test a simulated training loop with W&B logging."""
        with (
            patch("wandb.init") as mock_init,
            patch("wandb.log") as mock_log,
            patch("wandb.finish") as mock_finish,
        ):
            mock_run = MagicMock()
            mock_init.return_value = mock_run

            import random

            import wandb

            # Seed for reproducibility
            random.seed(42)

            # Start run
            wandb.init(project="shellflow-test", config={"epochs": 5})

            # Simulate training
            epochs = 5
            offset = random.random() / 5

            logged_metrics = []
            for epoch in range(epochs):
                acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
                loss = 2**-epoch + random.random() / (epoch + 1) + offset

                metrics = {"acc": acc, "loss": loss, "epoch": epoch}
                wandb.log(metrics)
                logged_metrics.append(metrics)

            wandb.finish()

            # Verify logging
            assert mock_log.call_count == epochs
            mock_finish.assert_called_once()

            # Verify metrics trends (loss should generally decrease)
            assert logged_metrics[-1]["loss"] < logged_metrics[0]["loss"]

    def test_wandb_disabled_mode(self):
        """Test behavior when W&B is disabled (no-wandb flag)."""
        # Simulate the pattern used in training scripts
        use_wandb = False

        log_calls = []

        def maybe_log(metrics):
            if use_wandb:
                log_calls.append(metrics)

        # Should not log anything
        maybe_log({"loss": 0.5})
        maybe_log({"loss": 0.3})

        assert len(log_calls) == 0


class TestWandbMetricsLogging:
    """Test specific metrics logging patterns used in training."""

    def test_train_metrics_logging(self):
        """Test training metrics logging pattern used by the training scripts."""
        with patch("wandb.log") as mock_log:
            import wandb

            # Simulate metrics from train_epoch
            batch_idx = 0
            epoch = 0
            log_interval = 50

            metrics = {"loss": 0.5, "mae": 0.3, "num_masked": 100}

            # Interval-gated metric logging pattern
            if batch_idx % log_interval == 0:
                wandb.log(
                    {
                        "train/loss": metrics["loss"],
                        "train/mae": metrics["mae"],
                        "train/num_masked": metrics["num_masked"],
                        "train/lr": 1e-3,
                        "train/grad_norm": 0.5,
                        "train/step": epoch * 100 + batch_idx,
                    }
                )

            # Verify the logging format
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "train/loss" in call_args
            assert "train/mae" in call_args
            assert "train/step" in call_args

    def test_validation_metrics_logging(self):
        """Test validation metrics logging pattern."""
        with patch("wandb.log") as mock_log:
            import wandb

            val_metrics = {"loss": 0.4, "mae": 0.25, "avg_masked": 150}

            # Pattern from training scripts
            wandb.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/mae": val_metrics["mae"],
                    "val/avg_masked": val_metrics["avg_masked"],
                    "epoch": 1,
                }
            )

            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "val/loss" in call_args
            assert "epoch" in call_args
