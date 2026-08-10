import argparse
import asyncio
import os
import signal
import sys

import aiohttp
import yaml
from tqdm.asyncio import tqdm

from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)

# Load datasets from YAML config
config_path = os.path.join(os.path.dirname(__file__), "config", "datasets.yaml")
with open(config_path, "r") as f:
    DATASETS = yaml.safe_load(f)

# Build dynamic epilog for argument parser
epilog_text = """
Examples:
# Download all datasets
uv run python -m shellflow.rootfile_downloader --dataset all

# Download MC 4-lepton data
uv run python -m shellflow.rootfile_downloader --dataset mc_4lep

# Download real ATLAS 4-lepton data
uv run python -m shellflow.rootfile_downloader --dataset real_data_4lep

Available datasets:
"""
max_key_len = max(len(key) for key in DATASETS.keys())
for key, value in DATASETS.items():
    epilog_text += f"{key:<{max_key_len}} | {value['description']}\n"


async def download_file(url, download_dir, expected_size=None, max_retries=5):
    filename = url.split("/")[-1]
    local_path = os.path.join(download_dir, filename)

    # Check if file already exists and get its size
    downloaded_size = 0
    if os.path.exists(local_path):
        downloaded_size = os.path.getsize(local_path)
        if expected_size and downloaded_size == expected_size:
            logger.info(
                f"Skipping {filename} (already complete, {downloaded_size} bytes)"
            )
            return
        logger.info(f"Resuming download of {filename} from {downloaded_size} bytes...")
    else:
        logger.info(f"Downloading {filename} ...")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_read=300)
    ) as session:
        for attempt in range(max_retries):
            try:
                headers = {}
                if downloaded_size > 0:
                    headers["Range"] = f"bytes={downloaded_size}-"

                async with session.get(url, headers=headers) as response:
                    if response.status not in (200, 206):
                        logger.error(
                            f"Failed to download {url}: HTTP {response.status}"
                        )
                        return

                    total_size = int(response.headers.get("content-length", 0))
                    if response.status == 200:
                        # Server doesn't support resume, start from beginning
                        file_mode = "wb"
                        progress_total = total_size
                        downloaded_size = 0
                    else:
                        # 206 Partial Content - resuming
                        file_mode = "ab"
                        progress_total = downloaded_size + total_size

                    with open(local_path, file_mode) as f:
                        if progress_total:
                            with tqdm(
                                total=progress_total,
                                unit="B",
                                unit_scale=True,
                                initial=downloaded_size,
                                desc=filename,
                            ) as pbar:
                                while True:
                                    chunk = await response.content.read(
                                        1024 * 1024
                                    )  # 1MB
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    pbar.update(len(chunk))
                        else:
                            # Fallback without progress
                            data = await response.read()
                            f.write(data)
                            downloaded_size += len(data)

                logger.info(f"Saved to {local_path}")
                return  # Success

            except (
                aiohttp.ClientPayloadError,
                aiohttp.ClientError,
                asyncio.TimeoutError,
            ) as e:
                logger.warning(
                    f"Download interrupted (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    downloaded_size = (
                        os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    )
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to download {filename} after {max_retries} attempts"
                    )
                    raise


async def get_root_files_from_api(record_url):
    """Fetch ROOT file list from the CERN Open Data REST API."""
    record_id = record_url.rstrip("/").split("/")[-1]
    api_url = f"https://opendata.cern.ch/api/records/{record_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch record metadata: HTTP {response.status}")
                return []
            data = await response.json()

    files = data.get("metadata", {}).get("_files", [])
    root_files = [f for f in files if f["key"].endswith(".root")]
    logger.info(f"Found {len(root_files)} ROOT files via API (record {record_id})")
    return root_files


def signal_handler(sig, frame):
    logger.info("Process interrupted by user (Ctrl+C). Exiting gracefully.")
    sys.exit(0)


async def main(dataset_name):
    dataset = DATASETS[dataset_name]
    download_dir = dataset["output_dir"]
    atlas_url = dataset["url"]

    # Create output directory
    os.makedirs(download_dir, exist_ok=True)

    logger.info(f"Dataset: {dataset['description']}")
    logger.info(f"URL: {atlas_url}")
    logger.info(f"Output directory: {download_dir}")
    logger.info("-" * 70)

    root_files = await get_root_files_from_api(atlas_url)
    if not root_files:
        logger.error("No ROOT files found. Exiting.")
        return

    for file_info in root_files:
        filename = file_info["key"]
        file_size = file_info.get("size")
        record_id = atlas_url.rstrip("/").split("/")[-1]
        download_url = f"https://opendata.cern.ch/record/{record_id}/files/{filename}"
        if file_size:
            logger.info(f"Found ROOT file: {filename} ({file_size / 1e9:.2f} GB)")
        else:
            logger.info(f"Found ROOT file: {filename}")
        await download_file(download_url, download_dir, expected_size=file_size)

    logger.info("All files processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ROOT files from ATLAS Open Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to download (default: all)",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    loop = asyncio.get_event_loop()
    try:
        if args.dataset == "all":
            for dataset_name in DATASETS.keys():
                logger.info(f"Starting download for dataset: {dataset_name}")
                loop.run_until_complete(main(dataset_name))
        else:
            loop.run_until_complete(main(args.dataset))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user (Ctrl+C). Exiting gracefully.")
