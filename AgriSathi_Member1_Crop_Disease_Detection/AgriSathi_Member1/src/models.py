"""
models.py — model architecture builders.

Three models are built:
    1. build_cnn_baseline()      - simple CNN from scratch
    2. build_mobilenetv2()       - MobileNetV2 transfer learning
    3. build_mobilenetv3_small() - MobileNetV3Small transfer learning

Each transfer-learning builder returns (model, base_model) so the
caller can freeze/unfreeze base_model layers for the two-stage
(head-only -> fine-tune) training procedure.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_baseline(input_shape, num_classes, augmentation_layer=None):
    """
    Lightweight baseline CNN:

    Input -> [Augmentation] -> Rescaling -> Conv/Pool x3 ->
    GlobalAveragePooling -> Dense -> Dropout -> Softmax
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cnn_baseline")
    return model


def build_mobilenetv2(input_shape, num_classes, augmentation_layer=None,
                       dropout_rate=0.3):
    """
    MobileNetV2 transfer learning model.

    Base model starts FROZEN (trainable=False). Caller is responsible
    for unfreezing top layers during the fine-tuning stage.

    Note: input pixels should be raw [0,255] float — preprocess_input
    (scaling to [-1,1]) is applied INSIDE this function via a Lambda
    layer, so the caller must NOT apply it again outside the model.
    """
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = layers.Lambda(preprocess_input, name="mobilenetv2_preprocess")(x)

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    return model, base_model


def build_mobilenetv3_small(input_shape, num_classes, augmentation_layer=None,
                             dropout_rate=0.3):
    """
    MobileNetV3Small transfer learning model.

    MobileNetV3's Keras implementation includes its own internal
    rescaling/normalization, so raw [0,255] pixels are passed directly
    to the base model — do NOT apply an additional preprocess_input
    call, or pixels will be normalized twice.
    """
    from tensorflow.keras.applications import MobileNetV3Small

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,  # handles its own normalization
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv3small_transfer")
    return model, base_model


def unfreeze_top_layers(base_model, num_layers_to_unfreeze):
    """
    Unfreeze the last `num_layers_to_unfreeze` layers of base_model for
    fine-tuning, keeping earlier layers frozen. BatchNorm layers are
    kept frozen regardless (standard fine-tuning practice) to preserve
    stable running statistics.
    """
    base_model.trainable = True

    freeze_until = max(0, len(base_model.layers) - num_layers_to_unfreeze)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[OK] Unfroze top {num_layers_to_unfreeze} layers "
          f"({trainable_count} layers now trainable out of {len(base_model.layers)}).")


def count_parameters(model):
    """Return (total_params, trainable_params) for a compiled/built Keras model."""
    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    return int(total), int(trainable)
