"""
scripts/inspect_dataset.py

STEP 2 — Inspect RAW_DIR: detect classes, count images, find empty
folders and unsupported files, verify SELECTED_CLASSES exist, and run
a data-quality report (corrupted images, dimensions, duplicates).

Run from the project root:
    python scripts/inspect_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import dataset


def main():
    summary = dataset.inspect_dataset(config.RAW_DIR, config.SUPPORTED_EXTENSIONS)
    dataset.print_dataset_summary(config.RAW_DIR, summary)

    print()
    dataset.verify_selected_classes(config.RAW_DIR, config.SELECTED_CLASSES)

    print()
    report = dataset.run_quality_report(
        config.RAW_DIR,
        config.SELECTED_CLASSES,
        config.SUPPORTED_EXTENSIONS,
        min_dimension=config.MIN_IMAGE_DIMENSION,
        check_duplicates=True,
    )
    dataset.print_quality_report(report)


if __name__ == "__main__":
    main()
