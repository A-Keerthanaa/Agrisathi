"""
scripts/convert_tflite.py

STEP — Convert the selected deployment-candidate Keras model to:
  1. Models/Final/disease_model_float32.tflite
  2. Models/Final/disease_model_int8.tflite  (INT8, calibrated on
     TRAINING images only)

Also generates class_names.txt / class_names.json /
preprocessing_config.json / model_metadata.json into Models/Final/.

Usage:
    python scripts/convert_tflite.py --model MobileNetV2

The --model argument must name a model that has already been trained
(checkpoint exists) and, ideally, chosen as the deployment candidate
after reviewing Results/Metrics/model_comparison.csv.
"""

import os
import sys
import argparse

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import preprocessing, tflite_utils, utils, evaluation


MODEL_CHECKPOINTS = {
    "CNN": os.path.join(config.CNN_MODEL_DIR, "best_model.keras"),
    "MobileNetV2": os.path.join(config.MOBILENETV2_MODEL_DIR, "best_model.keras"),
    "MobileNetV3": os.path.join(config.MOBILENETV3_MODEL_DIR, "best_model.keras"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CHECKPOINTS.keys()),
                         help="Which trained model to convert for deployment.")
    args = parser.parse_args()

    checkpoint_path = MODEL_CHECKPOINTS[args.model]
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}. Train {args.model} first.")
        sys.exit(1)

    model = tf.keras.models.load_model(checkpoint_path)

    # Recover class_names in the exact index order used during training
    _, _, _, class_names = preprocessing.build_datasets(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        config.IMAGE_SIZE, config.BATCH_SIZE, config.SEED,
    )

    os.makedirs(config.FINAL_MODEL_DIR, exist_ok=True)

    # ---- 1. Float32 conversion ----
    float32_path = os.path.join(config.FINAL_MODEL_DIR, "disease_model_float32.tflite")
    tflite_utils.convert_to_tflite_float32(model, float32_path)

    # ---- 2. INT8 conversion (calibrated on TRAIN_DIR only) ----
    rep_dataset_fn = tflite_utils.build_representative_dataset_fn(
        config.TRAIN_DIR, config.IMAGE_SIZE,
        num_samples=config.REPRESENTATIVE_DATASET_SIZE,
        supported_extensions=config.SUPPORTED_EXTENSIONS,
        seed=config.SEED,
    )
    int8_path = os.path.join(config.FINAL_MODEL_DIR, "disease_model_int8.tflite")
    tflite_utils.convert_to_tflite_int8(model, int8_path, rep_dataset_fn)

    # ---- 3. Inspect both models ----
    print("\n--- Float32 model ---")
    tflite_utils.inspect_tflite_model(float32_path)
    print("\n--- INT8 model ---")
    tflite_utils.inspect_tflite_model(int8_path)

    # ---- 4. class_names.txt / .json ----
    class_names_txt = os.path.join(config.FINAL_MODEL_DIR, "class_names.txt")
    with open(class_names_txt, "w") as f:
        f.write("\n".join(class_names))
    print(f"[OK] Saved: {class_names_txt}")

    class_names_json = {str(i): name for i, name in enumerate(class_names)}
    utils.save_json(class_names_json, os.path.join(config.FINAL_MODEL_DIR, "class_names.json"))

    # ---- 5. preprocessing_config.json ----
    preprocessing_config = {
        "image_size": list(config.IMAGE_SIZE),
        "color_mode": "RGB",
        "normalization": (
            "Model-specific preprocessing is baked into the exported "
            "graph (Rescaling/Lambda layer inside the model). Raw "
            "[0,255] uint8 pixels should be supplied at inference time."
        ),
        "model_type": args.model,
        "float32_input_dtype": "float32",
        "float32_output_dtype": "float32",
        "int8_model_note": (
            "Read input/output dtype and quantization scale/zero_point "
            "directly from the INT8 .tflite file at inference time — "
            "do not assume uint8 vs int8."
        ),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "class_names": class_names,
    }
    utils.save_json(preprocessing_config,
                     os.path.join(config.FINAL_MODEL_DIR, "preprocessing_config.json"))

    # ---- 6. model_metadata.json (only measured facts) ----
    float32_size = evaluation.get_model_file_size_mb(float32_path)
    int8_size = evaluation.get_model_file_size_mb(int8_path)

    metadata = {
        "model_name": args.model,
        "training_dataset": "PlantVillage (subset) — see config.SELECTED_CLASSES",
        "num_classes": len(class_names),
        "class_names": class_names,
        "input_shape": list(config.IMAGE_SIZE) + [config.IMAGE_CHANNELS],
        "float32_model_size_mb": round(float32_size, 3) if float32_size else None,
        "int8_model_size_mb": round(int8_size, 3) if int8_size else None,
        "note": (
            "Test-set accuracy/precision/recall/F1 for this model are in "
            "Results/Metrics/model_comparison.csv and "
            f"Results/Metrics/{args.model}_classification_report.txt. "
            "This file intentionally does not duplicate those numbers to "
            "avoid drift — refer to the Metrics folder for current values."
        ),
    }
    utils.save_json(metadata, os.path.join(config.FINAL_MODEL_DIR, "model_metadata.json"))

    print(f"\n[OK] Final deployment package ready in: {config.FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
