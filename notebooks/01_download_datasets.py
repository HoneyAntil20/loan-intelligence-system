"""
Phase 1 — Download all three Kaggle datasets.
Requires: kaggle CLI configured with ~/.kaggle/kaggle.json
Run: python notebooks/01_download_datasets.py
"""

import subprocess
import os
import sys

# ── Download targets ─────────────────────────────────────────────────────────
DOWNLOADS = [
    {
        "name": "Lending Club (~3 GB)",
        "cmd": [
            "kaggle", "datasets", "download",
            "-d", "wordsforthewise/lending-club",
            "-p", "data/raw/lending_club/",
            "--unzip"
        ],
    },
    {
        "name": "IEEE-CIS Fraud Detection",
        "cmd": [
            "kaggle", "competitions", "download",
            "-c", "ieee-fraud-detection",
            "-p", "data/raw/ieee-fraud-detection/"
        ],
    },
    {
        "name": "Home Credit Default Risk",
        "cmd": [
            "kaggle", "competitions", "download",
            "-c", "home-credit-default-risk",
            "-p", "data/raw/home-credit-default-risk/"
        ],
    },
]


def run_download(spec: dict) -> bool:
    """Execute a kaggle download command and report result."""
    print(f"\nDownloading: {spec['name']}")
    print(f"  Command: {' '.join(spec['cmd'])}")
    result = subprocess.run(spec["cmd"], capture_output=False)
    if result.returncode != 0:
        print(f"  [FAILED] {spec['name']}")
        return False
    print(f"  [DONE] {spec['name']}")
    return True


if __name__ == "__main__":
    # Ensure output directories exist
    for d in ["data/raw/lending_club", "data/raw/ieee-fraud-detection", "data/raw/home-credit-default-risk"]:
        os.makedirs(d, exist_ok=True)

    print("=== Phase 1: Kaggle Dataset Downloads ===")
    print("Note: Ensure ~/.kaggle/kaggle.json is configured before running.\n")

    results = [run_download(dl) for dl in DOWNLOADS]
    passed = sum(results)

    print(f"\n{passed}/{len(results)} downloads completed.")
    if passed < len(results):
        sys.exit(1)
