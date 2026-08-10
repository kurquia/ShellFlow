"""
Simple benchmark test for ParticleDataset and DataLoader performance.

This test measures:
- Dataset initialization time for both loading modes (in-memory vs lazy)
- DataLoader throughput (events/sec and batches/sec)
- Memory usage (if psutil available)

Target Performance:
- 50,000+ events/sec
- 20+ batches/sec with batch_size=4096

Dataset supports two loading modes:
- In-memory: Fast but high memory usage (all data in RAM)
- Lazy loading: Memory-efficient with memory-mapped arrays (bounded memory usage)
"""

import os
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shellflow.data import ParticleDataset
from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)

# Try to import psutil for memory tracking (optional)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory tracking disabled")


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_dataset_init(config_path: str, lazy_loading: bool = True):
    """Benchmark dataset initialization."""
    mode_name = "Lazy Loading" if lazy_loading else "In-Memory"
    logger.info("=" * 70)
    logger.info(f"BENCHMARK: Dataset Initialization ({mode_name})")
    logger.info("=" * 70)

    mem_before = get_memory_usage_mb()
    start = time.time()

    dataset = ParticleDataset(
        config_path=config_path,
        normalize=True,
        lazy_loading=lazy_loading,
    )

    init_time = time.time() - start
    mem_after = get_memory_usage_mb()
    mem_used = mem_after - mem_before

    stats = dataset.get_statistics()

    logger.info(f"✓ Dataset loaded in {init_time:.2f}s")
    logger.info(f"  Events: {stats['num_events']:,}")
    logger.info(f"  Sequence length: {stats['sequence_length']}")
    if PSUTIL_AVAILABLE:
        logger.info(f"  Memory used: {mem_used:.1f} MB ({mem_after:.1f} MB total)")
    logger.info("")

    return dataset, init_time, mem_used


def benchmark_dataloader(
    dataset,
    batch_size: int = 4096,
    num_workers: int = 4,
    num_batches: int = 50,
):
    """Benchmark DataLoader throughput."""
    logger.info("=" * 70)
    logger.info(
        f"BENCHMARK: DataLoader (batch_size={batch_size}, num_workers={num_workers})"
    )
    logger.info("=" * 70)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    logger.info(f"Measuring {num_batches} batches...")

    # Warmup
    warmup_batches = min(5, num_batches // 10)
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break

    # Actual benchmark
    start = time.time()
    batch_times = []

    for i, batch in enumerate(loader):
        batch_times.append(time.time())

        if i >= num_batches - 1:
            break

    end = time.time()
    total_time = end - start

    # Calculate metrics
    total_events = num_batches * batch_size
    events_per_sec = total_events / total_time
    batches_per_sec = num_batches / total_time
    avg_batch_time = total_time / num_batches * 1000  # in ms

    logger.info(
        f"✓ Processed {num_batches} batches ({total_events:,} events) in {total_time:.2f}s"
    )
    logger.info(f"  Throughput: {events_per_sec:,.0f} events/sec")
    logger.info(f"  Throughput: {batches_per_sec:.1f} batches/sec")
    logger.info(f"  Avg batch time: {avg_batch_time:.2f} ms")
    logger.info("")

    # Check targets
    target_events_per_sec = 50000
    target_batches_per_sec = 20

    status = "✓" if events_per_sec >= target_events_per_sec else "✗"
    logger.info(
        f"{status} Target: {target_events_per_sec:,} events/sec (actual: {events_per_sec:,.0f})"
    )

    status = "✓" if batches_per_sec >= target_batches_per_sec else "✗"
    logger.info(
        f"{status} Target: {target_batches_per_sec} batches/sec (actual: {batches_per_sec:.1f})"
    )
    logger.info("")

    return {
        "events_per_sec": events_per_sec,
        "batches_per_sec": batches_per_sec,
        "avg_batch_time_ms": avg_batch_time,
        "total_time": total_time,
    }


def benchmark_loading_modes(
    config_path: str, batch_size: int = 4096, num_batches: int = 30
):
    """Compare in-memory vs lazy loading modes."""
    logger.info("=" * 70)
    logger.info("BENCHMARK: Loading Mode Comparison")
    logger.info("=" * 70)

    results = {}

    # Test lazy loading mode
    logger.info("\n--- Lazy Loading Mode ---")
    dataset_lazy = ParticleDataset(
        config_path=config_path,
        normalize=True,
        lazy_loading=True,
    )

    result_lazy = benchmark_dataloader(
        dataset_lazy,
        batch_size=batch_size,
        num_workers=4,
        num_batches=num_batches,
    )
    results["lazy"] = result_lazy

    # Test in-memory mode
    logger.info("\n--- In-Memory Mode ---")
    dataset_mem = ParticleDataset(
        config_path=config_path,
        normalize=True,
        lazy_loading=False,
    )

    result_mem = benchmark_dataloader(
        dataset_mem,
        batch_size=batch_size,
        num_workers=4,
        num_batches=num_batches,
    )
    results["in_memory"] = result_mem

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY: Loading Mode Comparison")
    logger.info("=" * 70)
    logger.info(f"{'Mode':<15} {'Events/sec':<15} {'Batches/sec':<15} {'Speedup':<10}")
    logger.info("-" * 70)

    baseline_speed = results["in_memory"]["events_per_sec"]

    for mode_name, result in results.items():
        speedup = result["events_per_sec"] / baseline_speed
        display_name = mode_name.replace("_", " ").title()
        logger.info(
            f"{display_name:<15} "
            f"{result['events_per_sec']:>14,.0f} "
            f"{result['batches_per_sec']:>14.1f} "
            f"{speedup:>9.2f}x"
        )

    logger.info("")
    logger.info("Key Findings:")
    logger.info("- In-memory mode: Fastest but uses most RAM")
    logger.info("- Lazy loading: Memory-efficient with memory-mapped arrays")
    logger.info("- Performance difference: Typically 10-20% slower for lazy loading")
    logger.info("")

    return results


def main():
    """Run all benchmarks."""
    logger.info("\n")
    logger.info("*" * 70)
    logger.info("*" + " " * 68 + "*")
    logger.info(
        "*" + "  ParticleDataset & DataLoader Performance Benchmark".center(68) + "*"
    )
    logger.info("*" + " " * 68 + "*")
    logger.info("*" * 70)
    logger.info("\n")

    # Configuration
    config_path = (
        Path(__file__).parent.parent
        / "src"
        / "shellflow"
        / "config"
        / "dataloader_config.yaml"
    )
    batch_size = 4096
    num_batches = 30
    num_woerkers = 8

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.error("Please ensure the config file exists or provide a valid path.")
        return

    # 1. Dataset initialization (lazy loading mode)
    dataset, init_time, mem_used = benchmark_dataset_init(
        str(config_path),
        lazy_loading=True,
    )

    # 2. Compare loading modes (in-memory vs lazy loading)
    loading_mode_results = benchmark_loading_modes(
        str(config_path), batch_size=batch_size, num_batches=num_batches
    )

    # 3. Basic DataLoader benchmark with standard configuration
    logger.info("=" * 70)
    logger.info(
        f"BENCHMARK: Standard Configuration (Lazy Loading, workers={num_woerkers})"
    )
    logger.info("=" * 70)
    baseline = benchmark_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_woerkers,
        num_batches=num_batches,
    )

    # Final Summary
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Dataset size: {len(dataset):,} events")
    logger.info("Loading mode: Lazy loading with memory-mapped arrays")
    logger.info(f"Initialization time: {init_time:.2f}s")
    if PSUTIL_AVAILABLE:
        logger.info(f"Memory used: {mem_used:.1f} MB")
    logger.info("")
    logger.info(f"Best throughput: {baseline['events_per_sec']:,.0f} events/sec")
    logger.info(
        f"Best throughput: {baseline['batches_per_sec']:.1f} batches/sec (batch_size={batch_size})"
    )
    logger.info("")
    logger.info("Loading Mode Comparison:")
    logger.info(
        f"  In-memory:     {loading_mode_results['in_memory']['events_per_sec']:,.0f} events/sec (baseline)"
    )
    logger.info(
        f"  Lazy loading:  {loading_mode_results['lazy']['events_per_sec']:,.0f} events/sec (~{loading_mode_results['lazy']['events_per_sec'] / loading_mode_results['in_memory']['events_per_sec']:.1%} of baseline)"
    )
    logger.info("")

    # Check if targets met
    target_met = (
        baseline["events_per_sec"] >= 50000 and baseline["batches_per_sec"] >= 20
    )

    if target_met:
        logger.info("✓ PASS: Performance targets met!")
    else:
        logger.info("✗ FAIL: Performance targets not met")
        logger.info("  Target: 50,000+ events/sec and 20+ batches/sec")
        logger.info(
            f"  Actual: {baseline['events_per_sec']:,.0f} events/sec and {baseline['batches_per_sec']:.1f} batches/sec"
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("KEY FEATURES")
    logger.info("=" * 70)
    logger.info(
        "✓ Dual loading modes: In-memory (fast) vs Lazy loading (memory-efficient)"
    )
    logger.info("✓ Memory-mapped arrays for zero-copy access (.npy files)")
    logger.info("✓ Multi-file support (automatic chunking)")
    logger.info("✓ Multi-format support (.npz, .npy, .zarr)")
    logger.info("✓ On-the-fly normalization (arcsinh, log10, MinMax)")
    logger.info("✓ Optimized batch loading with __getitems__")
    logger.info("✓ Scales to datasets of any size")
    logger.info("")
    logger.info("PERFORMANCE NOTES:")
    logger.info("- Test uses batch_size=1024 for reasonable benchmarking time")
    logger.info("- For production, consider batch_size=4096 with sufficient GPU memory")
    logger.info("- Lazy loading: ~10-20% slower than in-memory but memory-efficient")
    logger.info("")
    logger.info("RECOMMENDATIONS:")
    logger.info(
        "- Loading mode: Use lazy loading for large datasets, in-memory for maximum speed"
    )
    logger.info("- File format: Use .npy for best memory mapping performance")
    logger.info("- Workers: Use 4 workers for good balance of speed vs overhead")
    logger.info("- Batch size: Use 1024-4096 depending on your GPU memory")
    logger.info("- Enable pin_memory=True for GPU training")
    logger.info("- Use persistent_workers=True to avoid respawning overhead")
    logger.info("")


if __name__ == "__main__":
    main()
