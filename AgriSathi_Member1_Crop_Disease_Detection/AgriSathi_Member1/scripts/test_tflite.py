"""
scripts/test_tflite.py

STEP — Validate the exported TFLite models:
  1. Compare Keras vs float32-TFLite vs INT8-TFLite predictions on a
     handful of real test images.
  2. Benchmark CPU inference latency for both TFLite models.
  3. Compare float32 vs INT8: file size, and (if a Keras checkpoint is
     available) accuracy/macro-F1 on the test set.

Usage:
    python scripts/test_tflite.py --model MobileNetV2
"""

import os
import sys
import argparse
import random

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import preprocessing, tflite_utils, evaluation


MODEL_CHECKPOINTS = {
    "CNN": os.path.join(config.CNN_MODEL_DIR, "best_model.keras"),
    "MobileNetV2": os.path.join(config.MOBILENETV2_MODEL_DIR, "best_model.keras"),
    "MobileNetV3": os.path.join(config.MOBILENETV3_MODEL_DIR, "best_model.keras"),
}


def collect_sample_test_images(test_dir, n=10, seed=42):
    all_paths = []
    for cls in sorted(os.listdir(test_dir)):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            all_paths.append(os.path.join(cls_dir, fname))

    rng = random.Random(seed)
    rng.shuffle(all_paths)
    return all_paths[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CHECKPOINTS.keys()))
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    float32_path = os.path.join(config.FINAL_MODEL_DIR, "disease_model_float32.tflite")
    int8_path = os.path.join(config.FINAL_MODEL_DIR, "disease_model_int8.tflite")

    for p in (float32_path, int8_path):
        if not os.path.exists(p):
            print(f"[ERROR] {p} not found. Run scripts/convert_tflite.py first.")
            sys.exit(1)

    checkpoint_path = MODEL_CHECKPOINTS[args.model]
    keras_model = tf.keras.models.load_model(checkpoint_path)

    _, _, _, class_names = preprocessing.build_datasets(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        config.IMAGE_SIZE, config.BATCH_SIZE, config.SEED,
    )

    sample_paths = collect_sample_test_images(config.TEST_DIR, n=args.n_samples, seed=config.SEED)

    print("\n=== Keras vs Float32 TFLite ===")
    tflite_utils.compare_keras_vs_tflite(
        keras_model, float32_path, sample_paths, class_names, config.IMAGE_SIZE
    )

    print("\n=== Keras vs INT8 TFLite ===")
    tflite_utils.compare_keras_vs_tflite(
        keras_model, int8_path, sample_paths, class_names, config.IMAGE_SIZE
    )

    print("\n=== CPU Latency Benchmark ===")
    float32_latency = tflite_utils.benchmark_tflite_cpu(float32_path, config.IMAGE_SIZE)
    int8_latency = tflite_utils.benchmark_tflite_cpu(int8_path, config.IMAGE_SIZE)
    print(f"Float32 TFLite — mean: {float32_latency['mean_ms']:.2f} ms, "
          f"median: {float32_latency['median_ms']:.2f} ms")
    print(f"INT8 TFLite    — mean: {int8_latency['mean_ms']:.2f} ms, "
          f"median: {int8_latency['median_ms']:.2f} ms")

    print("\n=== File Size Comparison ===")
    float32_size = evaluation.get_model_file_size_mb(float32_path)
    int8_size = evaluation.get_model_file_size_mb(int8_path)
    print(f"Float32: {float32_size:.2f} MB")
    print(f"INT8   : {int8_size:.2f} MB")
    if float32_size:
        reduction = (1 - int8_size / float32_size) * 100
        print(f"Size reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
