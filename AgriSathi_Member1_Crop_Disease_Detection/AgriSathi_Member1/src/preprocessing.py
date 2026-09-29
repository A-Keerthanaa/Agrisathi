"""
preprocessing.py — tf.data pipeline construction, augmentation, and
model-specific preprocessing (careful to avoid double normalization).
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation_layer():
    """
    Realistic, non-extreme augmentation for agricultural leaf images.
    This is applied ONLY to training data (as a model sub-layer that is
    active only during training, or by mapping only over the train
    dataset — see build_datasets()).
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_datasets(train_dir, val_dir, test_dir, image_size, batch_size, seed):
    """
    Build tf.data.Dataset objects for train/val/test using
    image_dataset_from_directory. Returns (train_ds, val_ds, test_ds,
    class_names).

    - Uses efficient tf.data pipeline (no full in-RAM loading).
    - Shuffling only applies to the training set.
    - class_names order is captured directly from Keras' own mapping,
      so it is guaranteed consistent with the model's output indices.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        label_mode="int",
    )
    class_names = train_ds.class_names  # authoritative order

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="int",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="int",
    )

    # Sanity check: val/test must use the exact same class order as train
    if val_ds.class_names != class_names:
        raise ValueError(
            "Class name order mismatch between train and val datasets. "
            "Ensure the same classes/folders exist in both directories."
        )
    if test_ds.class_names != class_names:
        raise ValueError(
            "Class name order mismatch between train and test datasets. "
            "Ensure the same classes/folders exist in both directories."
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def get_model_preprocessing_fn(model_type):
    """
    Return the correct Keras preprocessing function for a given model
    type, to avoid double-normalization bugs.

    model_type: one of "cnn", "mobilenetv2", "mobilenetv3"

    - "cnn": simple [0,1] rescaling (handled inside the model via a
      Rescaling layer — this function returns None so the caller knows
      not to apply anything extra).
    - "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input
      (scales pixels to [-1, 1]).
    - "mobilenetv3": tf.keras.applications.mobilenet_v3.preprocess_input
      (MobileNetV3 has preprocessing built into the model graph itself
      in Keras' implementation, so this also returns None — the raw
      [0,255] image should be passed directly to the model).
    """
    model_type = model_type.lower()

    if model_type == "cnn":
        return None  # Rescaling(1./255) is a layer inside the CNN model itself

    if model_type == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input

    if model_type == "mobilenetv3":
        # MobileNetV3 in tf.keras.applications includes rescaling/
        # normalization INSIDE the model graph. Do not apply
        # preprocess_input again outside the model, or pixels will be
        # normalized twice.
        return None

    raise ValueError(f"Unknown model_type: {model_type}")
