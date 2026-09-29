"""
training.py — callback construction, compile/fit wrappers, and
training-history plotting.
"""

import os
import json

import tensorflow as tf
import matplotlib.pyplot as plt


def build_callbacks(checkpoint_path, patience=6, reduce_lr_patience=3,
                     reduce_lr_factor=0.5, min_lr=1e-7, monitor="val_accuracy"):
    """
    Standard callback set: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode="max" if "accuracy" in monitor else "min",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode="max" if "accuracy" in monitor else "min",
        save_best_only=True,
        verbose=1,
    )

    return [early_stop, reduce_lr, checkpoint]


def compile_model(model, learning_rate=1e-3):
    """Compile a classification model with Adam + sparse categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_ds, val_ds, epochs, callbacks):
    """Thin wrapper around model.fit() that returns the History object."""
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_history(history, save_path):
    """Save a Keras History.history dict to JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[OK] Saved training history: {save_path}")


def plot_training_history(history_dict, model_name, save_dir):
    """
    Plot accuracy-vs-epoch and loss-vs-epoch charts from a
    history dict (as produced by save_history / loaded back with json).
    Saves two PNG files into save_dir with clear filenames.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Accuracy
    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["accuracy"], label="Train Accuracy")
        plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} — Accuracy vs Epoch")
        plt.legend()
        plt.tight_layout()
        acc_path = os.path.join(save_dir, f"{model_name}_accuracy.png")
        plt.savefig(acc_path, dpi=150)
        plt.close()
        print(f"[OK] Saved: {acc_path}")

    # Loss
    if "loss" in history_dict and "val_loss" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["loss"], label="Train Loss")
        plt.plot(history_dict["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} — Loss vs Epoch")
        plt.legend()
        plt.tight_layout()
        loss_path = os.path.join(save_dir, f"{model_name}_loss.png")
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"[OK] Saved: {loss_path}")


def compute_class_weights(class_counts):
    """
    Compute balanced class weights from a {class_name: count} dict,
    keyed by class INDEX (0..n-1) using the sorted class name order
    (must match the order used by image_dataset_from_directory).

    weight_i = total_samples / (num_classes * count_i)
    """
    classes = sorted(class_counts.keys())
    total = sum(class_counts.values())
    n_classes = len(classes)

    weights = {}
    for idx, cls in enumerate(classes):
        count = class_counts[cls]
        weights[idx] = total / (n_classes * count) if count > 0 else 0.0

    return weights
