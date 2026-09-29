"""
evaluation.py — metrics computation, classification reports, confusion
matrices, model comparison, and error analysis.

All metrics are computed from ACTUAL model predictions on the SAME
held-out test set. Nothing here fabricates numbers.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


def get_predictions(model, dataset):
    """
    Run model.predict over a tf.data Dataset (that yields (images, labels)
    batches) and return (y_true, y_pred, y_probs) as numpy arrays.
    """
    y_true = []
    y_probs = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_probs.append(probs)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_probs = np.concatenate(y_probs, axis=0)
    y_pred = np.argmax(y_probs, axis=1)

    return y_true, y_pred, y_probs


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute accuracy, macro/weighted precision/recall/F1, and a full
    classification report (as dict and as text).
    """
    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "report_dict": report_dict,
        "report_text": report_text,
    }


def save_classification_report(report_text, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report_text)
    print(f"[OK] Saved classification report: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path,
                           normalize=False, title=None):
    """Plot and save a confusion matrix (raw counts or row-normalized)."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or ("Normalized Confusion Matrix" if normalize else "Confusion Matrix"))

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:{fmt}}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if value > thresh else "black", fontsize=7)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix: {save_path}")

    return cm


def measure_inference_latency(model, sample_input, n_warmup=5, n_runs=30):
    """
    Measure CPU-oriented inference latency for a Keras model.
    sample_input: a single-batch numpy array, e.g. shape (1, H, W, 3).

    Returns {"mean_ms": ..., "median_ms": ...}.

    NOTE: this measures inference on whatever device TensorFlow is
    currently configured to use in this session. For a true mobile/CPU
    estimate, run this in a CPU-only runtime, or use the TFLite CPU
    benchmark in tflite_utils.py.
    """
    for _ in range(n_warmup):
        _ = model.predict(sample_input, verbose=0)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(sample_input, verbose=0)
        times.append((time.perf_counter() - start) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "n_runs": n_runs,
    }


def get_model_file_size_mb(filepath):
    """Return the size of a saved model file/directory in MB, or None if missing."""
    if os.path.isfile(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    if os.path.isdir(filepath):
        total = 0
        for root, _, files in os.walk(filepath):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)
    return None


def build_comparison_table(results_list, save_path=None):
    """
    results_list: list of dicts, one per model, each containing keys:
        Model, Accuracy, Macro_Precision, Macro_Recall, Macro_F1,
        Weighted_Precision, Weighted_Recall, Weighted_F1,
        Total_Parameters, Trainable_Parameters, Model_Size_MB,
        CPU_Inference_Time_ms

    Only include a row for a model AFTER it has actually been evaluated.
    Returns a pandas DataFrame and optionally saves it to CSV.
    """
    columns = [
        "Model", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1",
        "Weighted_Precision", "Weighted_Recall", "Weighted_F1",
        "Total_Parameters", "Trainable_Parameters", "Model_Size_MB",
        "CPU_Inference_Time_ms",
    ]
    df = pd.DataFrame(results_list, columns=columns)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[OK] Saved model comparison table: {save_path}")

    return df


def error_analysis(y_true, y_pred, y_probs, file_paths, class_names, save_path):
    """
    Build a DataFrame of misclassified test images:
    image_path, actual_class, predicted_class, confidence.

    file_paths must be in the SAME order as y_true/y_pred/y_probs
    (i.e. the order the test dataset was iterated in, WITHOUT shuffling).

    Also prints the most common confusion pairs.
    """
    rows = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            rows.append({
                "image_path": file_paths[i] if i < len(file_paths) else "unknown",
                "actual_class": class_names[y_true[i]],
                "predicted_class": class_names[y_pred[i]],
                "confidence": float(y_probs[i][y_pred[i]]),
            })

    df = pd.DataFrame(rows, columns=["image_path", "actual_class",
                                      "predicted_class", "confidence"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[OK] Saved error analysis: {save_path} ({len(df)} misclassified images)")

    if len(df) > 0:
        pair_counts = (
            df.groupby(["actual_class", "predicted_class"]).size()
            .sort_values(ascending=False)
        )
        print("\nMost common confusion pairs (actual -> predicted):")
        print(pair_counts.head(10))

    return df
