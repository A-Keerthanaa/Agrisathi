"""
image_quality.py — lightweight, heuristic image-quality checking.

These thresholds are simple heuristics, NOT scientifically validated
detectors. They are configurable (see config/config.py) and intended
to catch obviously unusable photos (too dark, too small, badly blurred)
before running them through the disease classifier.
"""

import numpy as np
from PIL import Image


def _laplacian_variance(gray_array):
    """
    Compute the variance of the Laplacian of a grayscale image array —
    a common, simple blur heuristic (lower variance ~ blurrier image).
    Implemented with a manual 3x3 Laplacian kernel convolution so this
    module has no hard dependency on OpenCV.
    """
    kernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=np.float32)

    h, w = gray_array.shape
    if h < 3 or w < 3:
        return 0.0

    img = gray_array.astype(np.float32)
    padded = np.pad(img, 1, mode="edge")

    out = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + h, j:j + w]

    return float(np.var(out))


def check_image_quality(image_path, min_resolution=64,
                         dark_mean_threshold=40.0,
                         bright_mean_threshold=220.0,
                         blur_variance_threshold=100.0):
    """
    Run heuristic quality checks on a single image.

    Returns:
        {
            "quality_ok": bool,
            "warnings": [str, ...],
            "quality_score": float in [0, 1] (heuristic, not validated),
            "details": {...}
        }
    """
    warnings = []
    details = {}

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            gray = np.array(img.convert("L"), dtype=np.float32)
    except Exception as e:
        return {
            "quality_ok": False,
            "warnings": [f"Could not open image: {e}"],
            "quality_score": 0.0,
            "details": {},
        }

    details["width"] = width
    details["height"] = height

    # 1. Resolution check
    if min(width, height) < min_resolution:
        warnings.append(
            f"Resolution too low ({width}x{height}); "
            f"minimum expected is {min_resolution}px on the shorter side."
        )

    # 2. Brightness checks
    mean_intensity = float(np.mean(gray))
    details["mean_intensity"] = mean_intensity

    if mean_intensity < dark_mean_threshold:
        warnings.append("Image appears too dark for reliable prediction.")
    elif mean_intensity > bright_mean_threshold:
        warnings.append("Image appears overexposed/too bright for reliable prediction.")

    # 3. Blur check (heuristic Laplacian variance)
    blur_variance = _laplacian_variance(gray)
    details["blur_variance"] = blur_variance

    if blur_variance < blur_variance_threshold:
        warnings.append("Image may be blurry; consider re-capturing with a steadier hand.")

    quality_ok = len(warnings) == 0

    # Simple heuristic score combining the three signals (NOT validated)
    resolution_score = min(1.0, min(width, height) / (min_resolution * 2))
    brightness_score = 1.0 - min(
        1.0,
        abs(mean_intensity - 127.5) / 127.5
    )
    blur_score = min(1.0, blur_variance / (blur_variance_threshold * 2))
    quality_score = float(np.clip(
        (resolution_score + brightness_score + blur_score) / 3.0, 0.0, 1.0
    ))

    return {
        "quality_ok": quality_ok,
        "warnings": warnings,
        "quality_score": quality_score,
        "details": details,
    }
