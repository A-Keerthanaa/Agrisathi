"""
prediction.py — predict_disease(): the full Keras-model inference API.

Pipeline:
    image -> quality check -> preprocess -> model -> label + confidence
    + top-3 predictions (+ low-confidence warning if below threshold)
"""

import numpy as np
from PIL import Image

from .image_quality import check_image_quality


def load_and_prepare_image(image_path, image_size, preprocessing_fn=None):
    """
    Load an image, convert to RGB, resize to image_size, and return a
    (1, H, W, 3) float32 numpy array ready for model.predict().

    preprocessing_fn: optional function (e.g. mobilenet_v2.preprocess_input)
    applied to the raw [0,255] pixel array. Pass None for models that
    handle their own rescaling internally (e.g. the CNN baseline with
    its Rescaling layer, or MobileNetV3 with include_preprocessing=True).
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((image_size[1], image_size[0]))
        arr = np.array(img, dtype=np.float32)

    if preprocessing_fn is not None:
        arr = preprocessing_fn(arr)

    return np.expand_dims(arr, axis=0)


def predict_disease(image_path, model, class_names, image_size,
                     confidence_threshold=0.60,
                     preprocessing_fn=None,
                     quality_check_kwargs=None):
    """
    Full prediction pipeline for a single image.

    Returns a dict:
        {
            "quality_ok": bool,
            "quality_warnings": [...],
            "prediction": class_name or None,
            "confidence": float or None,
            "top_3": [(class_name, confidence), ...] or [],
            "low_confidence_warning": str or None,
        }

    If the image fails the quality check, prediction still runs (the
    quality warnings are informational), but the low-confidence message
    is shown whenever the top prediction's confidence is below
    confidence_threshold, regardless of the quality check outcome.
    """
    quality_check_kwargs = quality_check_kwargs or {}
    quality_result = check_image_quality(image_path, **quality_check_kwargs)

    result = {
        "quality_ok": quality_result["quality_ok"],
        "quality_warnings": quality_result["warnings"],
        "prediction": None,
        "confidence": None,
        "top_3": [],
        "low_confidence_warning": None,
    }

    try:
        input_array = load_and_prepare_image(image_path, image_size, preprocessing_fn)
    except Exception as e:
        result["quality_warnings"].append(f"Could not process image: {e}")
        return result

    probs = model.predict(input_array, verbose=0)[0]  # shape (num_classes,)

    top_indices = np.argsort(probs)[::-1][:3]
    top_3 = [(class_names[i], float(probs[i])) for i in top_indices]

    top_class, top_confidence = top_3[0]

    result["prediction"] = top_class
    result["confidence"] = top_confidence
    result["top_3"] = top_3

    if top_confidence < confidence_threshold:
        result["low_confidence_warning"] = (
            "Low confidence — capture a clearer image or seek expert verification."
        )

    return result


def print_prediction_result(result):
    """Human-readable console output for a predict_disease() result."""
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    if result["quality_warnings"]:
        print("Quality warnings:")
        for w in result["quality_warnings"]:
            print(f"  - {w}")

    if result["prediction"] is None:
        print("Prediction could not be made.")
        return

    print(f"\nPrediction : {result['prediction']}")
    print(f"Confidence : {result['confidence'] * 100:.1f}%")

    print("\nTop-3 predictions:")
    for i, (cls, conf) in enumerate(result["top_3"], start=1):
        print(f"  {i}. {cls} — {conf * 100:.1f}%")

    if result["low_confidence_warning"]:
        print(f"\n[NOTICE] {result['low_confidence_warning']}")
