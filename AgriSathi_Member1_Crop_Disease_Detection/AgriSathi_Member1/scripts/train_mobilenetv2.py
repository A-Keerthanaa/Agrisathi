"""
scripts/train_mobilenetv2.py

STEP — Train MobileNetV2 via two-stage transfer learning:
  Stage 1: frozen base, train classification head.
  Stage 2: unfreeze top layers, fine-tune with a low learning rate.

Run from the project root:
    python scripts/train_mobilenetv2.py
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

    model, base_model = models.build_mobilenetv2(
        input_shape, num_classes=len(class_names), augmentation_layer=augmentation
    )

    checkpoint_path = os.path.join(config.MOBILENETV2_MODEL_DIR, "best_model.keras")

    # ---- Stage 1: frozen base, train head ----
    print("\n=== STAGE 1: Training classification head (base frozen) ===")
    training.compile_model(model, learning_rate=config.HEAD_LEARNING_RATE)
    callbacks_stage1 = training.build_callbacks(
        checkpoint_path,
        patience=config.EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=config.REDUCE_LR_PATIENCE,
        reduce_lr_factor=config.REDUCE_LR_FACTOR,
        min_lr=config.MIN_LR,
    )
    history_stage1 = training.train_model(
        model, train_ds, val_ds, epochs=config.MAX_EPOCHS_HEAD, callbacks=callbacks_stage1
    )
    training.save_history(
        history_stage1, os.path.join(config.METRICS_DIR, "mobilenetv2_stage1_history.json")
    )

    # ---- Stage 2: fine-tune top layers ----
    print("\n=== STAGE 2: Fine-tuning top layers ===")
    models.unfreeze_top_layers(base_model, config.FINE_TUNE_UNFREEZE_LAYERS)
    training.compile_model(model, learning_rate=config.FINE_TUNE_LEARNING_RATE)
    callbacks_stage2 = training.build_callbacks(
        checkpoint_path,
        patience=config.EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=config.REDUCE_LR_PATIENCE,
        reduce_lr_factor=config.REDUCE_LR_FACTOR,
        min_lr=config.MIN_LR,
    )
    history_stage2 = training.train_model(
        model, train_ds, val_ds, epochs=config.MAX_EPOCHS_FINE_TUNE, callbacks=callbacks_stage2
    )
    training.save_history(
        history_stage2, os.path.join(config.METRICS_DIR, "mobilenetv2_stage2_history.json")
    )

    training.plot_training_history(history_stage2.history, "mobilenetv2", config.GRAPH_DIR)

    print(f"\n[OK] MobileNetV2 training complete. Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
