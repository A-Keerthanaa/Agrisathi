"""
scripts/prepare_dataset.py

STEP 3 — Copy SELECTED_CLASSES from RAW_DIR into Selected_Classes/, then
perform a reproducible, stratified, class-aware Train/Val/Test split.
Verifies no data leakage afterward.

Run from the project root:
    python scripts/prepare_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import dataset


def main():
    dataset.verify_selected_classes(config.RAW_DIR, config.SELECTED_CLASSES)

    print("\n--- Preparing Selected_Classes dataset ---")
    prep_summary = dataset.prepare_selected_dataset(
        config.RAW_DIR, config.SELECTED_DIR, config.SELECTED_CLASSES,
        config.SUPPORTED_EXTENSIONS,
    )
    print(f"Copied: {prep_summary['copied']}, "
          f"Skipped (already existed): {prep_summary['skipped_existing']}")
    print(f"Total images in Selected_Classes: {prep_summary['total_images']} "
          f"across {prep_summary['num_classes']} classes")

    print("\n--- Splitting into Train / Val / Test ---")
    split_summary = dataset.stratified_split(
        config.SELECTED_DIR, config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        config.SELECTED_CLASSES, config.SUPPORTED_EXTENSIONS,
        train_split=config.TRAIN_SPLIT, val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT, seed=config.SEED,
    )

    print("\n--- Verifying no data leakage ---")
    leak_report = dataset.verify_no_leakage(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR, config.FIELD_TEST_DIR
    )
    if leak_report["num_leaks"] > 0:
        print("[CRITICAL] Data leakage detected — review the leaks above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
