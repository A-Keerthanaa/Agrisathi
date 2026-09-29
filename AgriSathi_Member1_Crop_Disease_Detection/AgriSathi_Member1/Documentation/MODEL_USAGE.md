# Model Usage

## Predicting with the Keras model (during development)

```python
import tensorflow as tf
from src.prediction import predict_disease, print_prediction_result
from config import config

model = tf.keras.models.load_model(f"{config.MOBILENETV2_MODEL_DIR}/best_model.keras")
class_names = [...]  # from Models/Final/class_names.json, in index order

result = predict_disease(
    image_path="/path/to/leaf.jpg",
    model=model,
    class_names=class_names,
    image_size=config.IMAGE_SIZE,
    confidence_threshold=config.CONFIDENCE_THRESHOLD,
)
print_prediction_result(result)
```

## Predicting with the final TFLite model (deployment-ready)

```python
from src.tflite_utils import predict_tflite
from config import config

result = predict_tflite(
    image_path="/path/to/leaf.jpg",
    model_path=f"{config.FINAL_MODEL_DIR}/disease_model_int8.tflite",
    class_names=class_names,
    image_size=config.IMAGE_SIZE,
    confidence_threshold=config.CONFIDENCE_THRESHOLD,
)
print(result)
# {"prediction": "...", "confidence": 0.xx, "top_3": [...], "low_confidence_warning": None or str}
```

`predict_tflite()` reads the model's actual input/output dtype and
quantization scale/zero-point at runtime — it works unmodified for
both the float32 and INT8 models.

## Final deployment package contents

After running `scripts/convert_tflite.py --model <ModelName>`,
`Models/Final/` contains:

| File | Purpose |
|---|---|
| `disease_model_float32.tflite` | Unquantized TFLite model |
| `disease_model_int8.tflite` | Fully-quantized INT8 TFLite model (calibrated on training images only) |
| `class_names.txt` | One class name per line, in model output-index order |
| `class_names.json` | `{"0": "class_name", ...}` mapping |
| `preprocessing_config.json` | Image size, color mode, confidence threshold, and notes needed to reproduce preprocessing in a mobile app |
| `model_metadata.json` | Measured facts only (parameter counts, file sizes) — actual accuracy/F1 numbers live in `Results/Metrics/`, not duplicated here, to avoid drift |

## For the future mobile app (Member 1 hands off here)

The mobile team needs:
1. `disease_model_int8.tflite` (primary) or `disease_model_float32.tflite`
   (fallback if INT8 accuracy loss is unacceptable).
2. `class_names.json` to map output index → disease name.
3. `preprocessing_config.json` to reproduce the exact preprocessing
   (image size, RGB, and — for the INT8 model — reading the
   quantization scale/zero-point directly from the `.tflite` file
   rather than assuming a fixed normalization).

This project (Member 1) does **not** build the mobile app itself —
only the artifacts above.

## Important caveat on all numbers

Every metric referenced in this documentation is a **placeholder
description of where the number will appear**, not a value. Actual
accuracy, F1, model size, and inference time are only ever written to
`Results/Metrics/model_comparison.csv` and `Models/Final/model_metadata.json`
after you have run the training and evaluation pipeline against the
real dataset in your Drive.
