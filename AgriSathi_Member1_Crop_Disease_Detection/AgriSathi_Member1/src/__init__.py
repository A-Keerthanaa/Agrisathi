"""
AgriSathi Member 1 — src package.

Modules:
    utils          - environment setup, reproducibility, directory helpers
    dataset        - dataset inspection, selection, splitting, quality checks
    eda            - exploratory data analysis (charts, stats)
    preprocessing  - image preprocessing + tf.data pipeline builders
    models         - CNN / MobileNetV2 / MobileNetV3 model builders
    training        - training loop helpers (callbacks, fit wrappers)
    evaluation     - metrics, classification reports, confusion matrices
    image_quality  - lightweight heuristic image-quality checks
    prediction     - predict_disease() Keras inference API
    tflite_utils   - TFLite conversion, quantization, and inference
"""

__all__ = [
    "utils",
    "dataset",
    "eda",
    "preprocessing",
    "models",
    "training",
    "evaluation",
    "image_quality",
    "prediction",
    "tflite_utils",
]
