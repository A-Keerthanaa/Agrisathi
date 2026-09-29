"""
scripts/error_analysis.py

STEP — Identify misclassified test images for a chosen model, save
Results/Error_Analysis/error_analysis.csv, and print the most common
confusion pairs.

Usage:
    python scripts/error_analysis.py --model MobileNetV2
(defaults to MobileNetV2 if --model is omitted)
"""

import os
import sys
import argparse

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import preprocessing, evaluation


MODEL_CHECKPOINTS = {
    "CNN": os.path.join(config.CNN_MODEL_DIR, "best_model.keras"),
    "MobileNetV2": os.path.join(config.MOBILENETV2_MODEL_DIR, "best_model.keras"),
    "MobileNetV3": os.path.join(config.MOBILENETV3_MODEL_DIR, "best_model.keras"),
}


def get_test_file_paths(test_ds, test_dir):
    """
    Recover the file paths in the exact order image_dataset_from_directory
    iterates them (unshuffled), so they align with y_true/y_pred/y_probs.
    """
    return test_ds.file_paths if hasattr(test_ds, "file_paths") else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MobileNetV2",
                        choices=list(MODEL_CHECKPOINTS.keys()))
    args = parser.parse_args()

    checkpoint_path = MODEL_CHECKPOINTS[args.model]
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}. "
              f"Train {args.model} first. Not evaluated yet.")
        sys.exit(1)

    _, _, test_ds, class_names = preprocessing.build_datasets(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        config.IMAGE_SIZE, config.BATCH_SIZE, config.SEED,
    )
    file_paths = get_test_file_paths(test_ds, config.TEST_DIR)

    model = tf.keras.models.load_model(checkpoint_path)
    y_true, y_pred, y_probs = evaluation.get_predictions(model, test_ds)

    evaluation.error_analysis(
        y_true, y_pred, y_probs, file_paths, class_names,
        save_path=os.path.join(config.ERROR_DIR, "error_analysis.csv"),
    )


if __name__ == "__main__":
    main()
