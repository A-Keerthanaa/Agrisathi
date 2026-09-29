"""
tflite_utils.py — TensorFlow Lite conversion, INT8 quantization, and
quantization-aware inference.

Handles both float32 and int8/uint8 TFLite models correctly by always
reading the interpreter's actual input/output dtype and quantization
parameters rather than assuming them.
"""

import os
import time
import random

import numpy as np
import tensorflow as tf
from PIL import Image


# ------------------------------------------------------------------
# 1. CONVERSION
# ------------------------------------------------------------------

def convert_to_tflite_float32(keras_model, save_path):
    """Convert a Keras model to a plain float32 TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"[OK] Saved float32 TFLite model: {save_path} ({size_mb:.2f} MB)")
    return save_path


def build_representative_dataset_fn(train_dir, image_size, num_samples=200,
                                     supported_extensions=(".jpg", ".jpeg", ".png", ".bmp"),
                                     seed=42):
    """
    Build a representative_dataset generator function for full-integer
    post-training quantization, sampling ONLY from train_dir (never
    val/test/field-test, to avoid calibration leakage).
    """
    all_paths = []
    for cls in sorted(os.listdir(train_dir)):
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in supported_extensions:
                all_paths.append(os.path.join(cls_dir, fname))

    if not all_paths:
        raise ValueError(f"No training images found under {train_dir} to build "
                          "the representative dataset.")

    rng = random.Random(seed)
    rng.shuffle(all_paths)
    sample_paths = all_paths[:num_samples]

    print(f"[OK] Representative dataset: {len(sample_paths)} images sampled "
          f"from TRAIN_DIR only (out of {len(all_paths)} available).")

    def representative_dataset():
        for path in sample_paths:
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB").resize((image_size[1], image_size[0]))
                    arr = np.array(img, dtype=np.float32)
                    arr = np.expand_dims(arr, axis=0)
                    yield [arr]
            except Exception:
                continue

    return representative_dataset


def convert_to_tflite_int8(keras_model, save_path, representative_dataset_fn,
                            fully_integer=True):
    """
    Convert a Keras model to a full-integer INT8 TFLite model using a
    representative dataset built from TRAINING images only.

    If fully_integer=True, forces integer-only ops (inference_input_type
    and inference_output_type set to int8) for maximum size/speed
    benefit on mobile CPUs. Some models may not fully support this if
    they contain ops without an integer kernel; in that case, retry with
    fully_integer=False, which still applies weight+activation
    quantization but allows float32 input/output.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn

    if fully_integer:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
    except Exception as e:
        if fully_integer:
            print(f"[WARNING] Fully-integer INT8 conversion failed ({e}). "
                  "Retrying with float32 input/output and INT8 weights only.")
            return convert_to_tflite_int8(
                keras_model, save_path, representative_dataset_fn,
                fully_integer=False,
            )
        raise

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"[OK] Saved INT8 TFLite model: {save_path} ({size_mb:.2f} MB)")
    return save_path


# ------------------------------------------------------------------
# 2. INSPECTION
# ------------------------------------------------------------------

def inspect_tflite_model(model_path):
    """
    Load a TFLite model and print/return its input/output shape, dtype,
    and quantization parameters. Never assumes dtype — reads it from
    the interpreter directly.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    info = {
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": str(input_details["dtype"]),
        "input_quantization": input_details["quantization"],
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": str(output_details["dtype"]),
        "output_quantization": output_details["quantization"],
    }

    print("=" * 60)
    print(f"TFLITE MODEL INSPECTION: {model_path}")
    print("=" * 60)
    print(f"Input shape              : {info['input_shape']}")
    print(f"Input dtype               : {info['input_dtype']}")
    print(f"Input quantization (scale, zero_point): {info['input_quantization']}")
    print(f"Output shape              : {info['output_shape']}")
    print(f"Output dtype              : {info['output_dtype']}")
    print(f"Output quantization (scale, zero_point): {info['output_quantization']}")

    return info


# ------------------------------------------------------------------
# 3. QUANTIZATION-AWARE INFERENCE
# ------------------------------------------------------------------

def _prepare_tflite_input(image_path, image_size, input_details):
    """
    Load and prepare an image for a TFLite interpreter, correctly
    handling float32 vs int8/uint8 input tensors using the model's
    ACTUAL quantization parameters (never blindly divides by 255).
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((image_size[1], image_size[0]))
        arr = np.array(img, dtype=np.float32)  # raw [0,255]

    dtype = input_details["dtype"]

    if dtype == np.float32:
        # Float32 models in this project expect raw [0,255] pixels;
        # any internal rescaling/preprocessing is baked into the
        # converted graph itself.
        prepared = arr
    else:
        scale, zero_point = input_details["quantization"]
        if scale == 0:
            # No quantization params available; fall back to a direct
            # cast (should not normally happen for a properly
            # quantized model).
            prepared = arr.astype(dtype)
        else:
            prepared = (arr / scale + zero_point).astype(dtype)

    return np.expand_dims(prepared, axis=0)


def _dequantize_output(raw_output, output_details):
    """Convert a quantized output tensor back to float32 probabilities."""
    dtype = output_details["dtype"]
    if dtype == np.float32:
        return raw_output.astype(np.float32)

    scale, zero_point = output_details["quantization"]
    if scale == 0:
        return raw_output.astype(np.float32)

    return (raw_output.astype(np.float32) - zero_point) * scale


def predict_tflite(image_path, model_path, class_names, image_size,
                    confidence_threshold=0.60):
    """
    Run inference using a TFLite model (float32 or INT8/uint8).

    Returns:
        {
            "prediction": class_name,
            "confidence": float,
            "top_3": [(class_name, confidence), ...],
            "low_confidence_warning": str or None,
        }
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_array = _prepare_tflite_input(image_path, image_size, input_details)

    interpreter.set_tensor(input_details["index"], input_array)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details["index"])[0]
    probs = _dequantize_output(raw_output, output_details)

    # Ensure probabilities sum to ~1 (defensive normalization in case
    # of dequantized rounding drift); does not change the argmax.
    total = np.sum(probs)
    if total > 0:
        probs = probs / total

    top_indices = np.argsort(probs)[::-1][:3]
    top_3 = [(class_names[i], float(probs[i])) for i in top_indices]
    top_class, top_confidence = top_3[0]

    low_confidence_warning = None
    if top_confidence < confidence_threshold:
        low_confidence_warning = (
            "Low confidence — capture a clearer image or seek expert verification."
        )

    return {
        "prediction": top_class,
        "confidence": top_confidence,
        "top_3": top_3,
        "low_confidence_warning": low_confidence_warning,
    }


def compare_keras_vs_tflite(keras_model, tflite_path, sample_image_paths,
                             class_names, image_size):
    """
    Compare Keras model predictions vs TFLite model predictions on the
    same set of sample images. Returns a list of per-image comparison
    dicts and prints a match-rate summary. Useful sanity check after
    any TFLite conversion (float32 or INT8).
    """
    from .prediction import load_and_prepare_image

    results = []
    matches = 0

    for path in sample_image_paths:
        keras_input = load_and_prepare_image(path, image_size, preprocessing_fn=None)
        keras_probs = keras_model.predict(keras_input, verbose=0)[0]
        keras_pred = class_names[int(np.argmax(keras_probs))]

        tflite_result = predict_tflite(path, tflite_path, class_names, image_size)
        tflite_pred = tflite_result["prediction"]

        is_match = keras_pred == tflite_pred
        matches += int(is_match)

        results.append({
            "image_path": path,
            "keras_prediction": keras_pred,
            "tflite_prediction": tflite_pred,
            "match": is_match,
        })

    match_rate = matches / len(sample_image_paths) if sample_image_paths else 0.0
    print(f"[OK] Keras vs TFLite match rate: {matches}/{len(sample_image_paths)} "
          f"({match_rate * 100:.1f}%)")

    return results


def benchmark_tflite_cpu(model_path, image_size, n_warmup=5, n_runs=30):
    """
    Measure CPU inference latency for a TFLite model using a synthetic
    random input (does not require real images — used purely for
    timing). Returns {"mean_ms": ..., "median_ms": ...}.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    dtype = input_details["dtype"]
    shape = input_details["shape"]

    if dtype == np.float32:
        dummy_input = np.random.rand(*shape).astype(np.float32) * 255.0
    else:
        info = np.iinfo(dtype)
        dummy_input = np.random.randint(info.min, info.max + 1, size=shape).astype(dtype)

    for _ in range(n_warmup):
        interpreter.set_tensor(input_details["index"], dummy_input)
        interpreter.invoke()

    times = []
    output_index = interpreter.get_output_details()[0]["index"]
    for _ in range(n_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)
        times.append((time.perf_counter() - start) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "n_runs": n_runs,
    }
