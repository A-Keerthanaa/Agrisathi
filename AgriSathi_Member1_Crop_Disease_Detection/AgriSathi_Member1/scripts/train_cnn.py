"""
scripts/train_cnn.py

STEP — Train the baseline CNN model on Train/Val, save the best
checkpoint, training history, and accuracy/loss plots.

Run from the project root:
    python scripts/train_cnn.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import utils, preprocessing, models, training


def main():
    utils.set_global_seed(config.SEED)

    train_ds, val_ds, _, class_names = preprocessing.build_datasets(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        config.IMAGE_SIZE, config.BATCH_SIZE, config.SEED,
    )
    print(f"[OK] Classes detected: {class_names}")

    input_shape = config.IMAGE_SIZE + (config.IMAGE_CHANNELS,)
    augmentation = preprocessing.build_augmentation_layer()

    model = models.build_cnn_baseline(
        input_shape, num_classes=len(class_names), augmentation_layer=augmentation
    )
    training.compile_model(model, learning_rate=config.HEAD_LEARNING_RATE)
    model.summary()

    checkpoint_path = os.path.join(config.CNN_MODEL_DIR, "best_model.keras")
    callbacks = training.build_callbacks(
        checkpoint_path,
        patience=config.EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=config.REDUCE_LR_PATIENCE,
        reduce_lr_factor=config.REDUCE_LR_FACTOR,
        min_lr=config.MIN_LR,
    )

    history = training.train_model(
        model, train_ds, val_ds, epochs=config.MAX_EPOCHS_CNN, callbacks=callbacks
    )

    training.save_history(
        history, os.path.join(config.METRICS_DIR, "cnn_baseline_history.json")
    )
    training.plot_training_history(history.history, "cnn_baseline", config.GRAPH_DIR)

    print(f"\n[OK] CNN baseline training complete. Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
