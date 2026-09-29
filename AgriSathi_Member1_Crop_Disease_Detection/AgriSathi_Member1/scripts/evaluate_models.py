"""
scripts/evaluate_models.py

STEP — Evaluate every trained model (CNN, MobileNetV2, MobileNetV3) on
the SAME held-out test set. Produces classification reports, confusion
matrices, and Results/Metrics/model_comparison.csv.

Only models that have actually been trained (checkpoint file exists)
are evaluated — this script never fabricates results for missing
models.

Run from the project root:
    python scripts/evaluate_models.py
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import preprocessing, evaluation


MODEL_CHECKPOINTS = {
    "CNN": os.path.join(config.CNN_MODEL_DIR, "best_model.keras"),
    "MobileNetV2": os.path.join(config.MOBILENETV2_MODEL_DIR, "best_model.keras"),
    "MobileNetV3": os.path.join(config.MOBILENETV3_MODEL_DIR, "best_model.keras"),
}


def main():
    _, _, test_ds, class_names = preprocessing.build_datasets(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        config.IMAGE_SIZE, config.BATCH_SIZE, config.SEED,
    )

    comparison_rows = []

    for model_name, checkpoint_path in MODEL_CHECKPOINTS.items():
        print(f"\n{'=' * 60}\nEVALUATING: {model_name}\n{'=' * 60}")

        if not os.path.exists(checkpoint_path):
            print(f"[SKIP] Checkpoint not found: {checkpoint_path}. "
                  "Train this model first (see scripts/train_*.py). "
                  "Not evaluated yet.")
            continue

        model = tf.keras.models.load_model(checkpoint_path)

        y_true, y_pred, y_probs = evaluation.get_predictions(model, test_ds)
        metrics = evaluation.compute_metrics(y_true, y_pred, class_names)

        print(metrics["report_text"])
        evaluation.save_classification_report(
            metrics["report_text"],
            os.path.join(config.METRICS_DIR, f"{model_name}_classification_report.txt"),
        )

        evaluation.plot_confusion_matrix(
            y_true, y_pred, class_names,
            os.path.join(config.CONF_MATRIX_DIR, f"{model_name}_confusion_matrix.png"),
            normalize=False, title=f"{model_name} — Confusion Matrix",
        )
        evaluation.plot_confusion_matrix(
            y_true, y_pred, class_names,
            os.path.join(config.CONF_MATRIX_DIR, f"{model_name}_confusion_matrix_normalized.png"),
            normalize=True, title=f"{model_name} — Normalized Confusion Matrix",
        )

        total_params, trainable_params = 0, 0
        try:
            from src.models import count_parameters
            total_params, trainable_params = count_parameters(model)
        except Exception as e:
            print(f"[WARNING] Could not count parameters: {e}")

        model_size_mb = evaluation.get_model_file_size_mb(checkpoint_path)

        sample_batch = next(iter(test_ds.take(1)))[0][:1].numpy()
        latency = evaluation.measure_inference_latency(model, sample_batch)

        comparison_rows.append({
            "Model": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Macro_Precision": round(metrics["macro_precision"], 4),
            "Macro_Recall": round(metrics["macro_recall"], 4),
            "Macro_F1": round(metrics["macro_f1"], 4),
            "Weighted_Precision": round(metrics["weighted_precision"], 4),
            "Weighted_Recall": round(metrics["weighted_recall"], 4),
            "Weighted_F1": round(metrics["weighted_f1"], 4),
            "Total_Parameters": total_params,
            "Trainable_Parameters": trainable_params,
            "Model_Size_MB": round(model_size_mb, 2) if model_size_mb else None,
            "CPU_Inference_Time_ms": round(latency["mean_ms"], 2),
        })

    if comparison_rows:
        df = evaluation.build_comparison_table(
            comparison_rows,
            save_path=os.path.join(config.METRICS_DIR, "model_comparison.csv"),
        )
        print("\n" + df.to_string(index=False))
    else:
        print("\n[INFO] No models have been trained yet — nothing to compare. "
              "Not evaluated yet.")


if __name__ == "__main__":
    main()
