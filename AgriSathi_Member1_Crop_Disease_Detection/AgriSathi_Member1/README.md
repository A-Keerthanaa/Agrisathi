# AgriSathi — Member 1

**Hyperlocal Crop Disease Detection for Smallholder Farmers Without Reliable Internet**

This is the **Member 1** component of the AgriSathi college ML project:
**image-based crop disease detection**, from a leaf photo through to a
quantized, offline-ready TensorFlow Lite model.

```
Leaf/crop image → Image quality check → Preprocessing →
Disease classification → Confidence score → Top-3 predictions →
TensorFlow Lite → INT8 quantization → Offline/mobile deployment
```

## 1. Project overview

Three image classification models are built and compared on the same
held-out test set:

1. A lightweight **CNN baseline** trained from scratch.
2. **MobileNetV2** (ImageNet transfer learning, two-stage fine-tuning).
3. **MobileNetV3Small** (ImageNet transfer learning, two-stage fine-tuning).

The best deployment candidate (chosen on accuracy, macro-F1, model
size, and inference latency together — not accuracy alone) is
converted to TensorFlow Lite in both float32 and fully-quantized INT8
forms, ready for offline mobile inference.

## 2. Member 1 scope

**In scope:** image → disease classification → lightweight model → TFLite.

**Out of scope** (other team members / future work): weather-based
prediction, fertilizer/crop recommendation, multilingual treatment
advice, voice/SMS/IVR systems, community disease radar, the full
mobile app, the project report, the PPT, and viva prep.

## 3. Dataset location

The dataset is **not** included in this project — it lives in your
Google Drive and is read directly from Google Colab:

```
MyDrive/AgriSathi/Dataset/Raw/plantvillage/PlantVillage/
```

See [`Documentation/DATASET_SETUP.md`](Documentation/DATASET_SETUP.md)
for the full expected folder layout and the six classes used in this
first implementation.

## 4. Google Drive & Google Colab setup

See [`Documentation/COLAB_SETUP.md`](Documentation/COLAB_SETUP.md) for
step-by-step instructions on getting this project into Colab,
mounting Drive, enabling a GPU, and troubleshooting common issues.

## 5. How to run the notebook

Open `notebooks/AgriSathi_Member1_Full_Pipeline.ipynb` in Google Colab
and run cells top to bottom. The notebook is organized into the
following sections (matching the project's logical pipeline):

1. Environment Setup
2. Google Drive Mount
3. Configuration
4. Dataset Inspection
5. Dataset Quality Check
6. EDA
7. Selected Dataset
8. Train / Validation / Test Split
9. TensorFlow Dataset Pipeline
10. Augmentation
11. CNN Baseline
12. MobileNetV2
13. MobileNetV3
14. Model Evaluation
15. Confusion Matrices
16. Model Comparison
17. Error Analysis
18. Field Test Evaluation
19. Final Model Selection
20. TFLite Float32
21. INT8 Quantization
22. TFLite Validation
23. Final Prediction Demo
24. Final Deployment Package

## 6. How to run individual scripts

Every notebook section has an equivalent standalone script under
`scripts/`, all importing shared logic from `src/` and configuration
from `config/config.py`:

```bash
python scripts/setup_colab.py          # Step 1: environment + Drive
python scripts/inspect_dataset.py      # Step 2: inspect + quality report
python scripts/prepare_dataset.py      # Step 3: select classes + split
python scripts/train_cnn.py            # Train CNN baseline
python scripts/train_mobilenetv2.py    # Train MobileNetV2 (2-stage)
python scripts/train_mobilenetv3.py    # Train MobileNetV3Small (2-stage)
python scripts/evaluate_models.py      # Evaluate all trained models
python scripts/error_analysis.py --model MobileNetV2
python scripts/convert_tflite.py --model MobileNetV2   # after choosing a winner
python scripts/test_tflite.py --model MobileNetV2
```

Scripts are meant to be run from Google Colab against your mounted
Drive (they use the same `config/config.py` paths as the notebook),
but will also run on any machine where `RAW_DIR` points at a local
copy of the dataset.

## 7. Model architectures

- **CNN baseline** — `Conv2D → MaxPool` × 3, `GlobalAveragePooling2D`,
  `Dense(128) → Dropout(0.3) → Softmax`. See `src/models.py::build_cnn_baseline`.
- **MobileNetV2** — ImageNet-pretrained base (frozen, then top layers
  fine-tuned at `1e-5`), `GlobalAveragePooling2D → Dropout → Softmax`.
- **MobileNetV3Small** — same two-stage strategy; uses the model's
  built-in `include_preprocessing=True` rescaling (no separate
  `preprocess_input` call, to avoid double normalization).

## 8. Training pipeline

`tf.data` pipelines (via `image_dataset_from_directory`) with
shuffling on train only, batching, and `prefetch(AUTOTUNE)` — the full
8,000+ image dataset is never loaded into RAM at once. Callbacks:
`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` (best-only).

## 9. Evaluation

All three models are evaluated on the **same** test set:
accuracy, macro/weighted precision/recall/F1, full classification
reports, and confusion matrices (raw + normalized). Results land in
`Results/Metrics/` and `Results/Confusion_Matrices/`.

## 10. TFLite conversion

`scripts/convert_tflite.py --model <Name>` converts the chosen model
to `Models/Final/disease_model_float32.tflite`.

## 11. INT8 quantization

The same script also produces `Models/Final/disease_model_int8.tflite`
using full-integer post-training quantization, calibrated on a
representative sample of **training images only** (never
val/test/field-test, to avoid calibration leakage).

## 12. Final model usage

See [`Documentation/MODEL_USAGE.md`](Documentation/MODEL_USAGE.md) for
`predict_disease()` (Keras) and `predict_tflite()` (TFLite) usage
examples, and a description of every file in `Models/Final/`.

## 13. Troubleshooting

See the troubleshooting table at the bottom of
[`Documentation/COLAB_SETUP.md`](Documentation/COLAB_SETUP.md).

## 14. No fabricated results

**Nothing in this repository claims a trained-model result.** Every
metric (accuracy, F1, model size, inference time, field-test
performance) is only ever written to `Results/` and `Models/Final/`
after you actually run the pipeline against your dataset in Google
Drive. Anywhere a number would normally appear before that has
happened, the code prints `"Not evaluated yet."`

## Project structure

```
AgriSathi_Member1/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.py                  # single source of truth for all paths & hyperparameters
├── notebooks/
│   └── AgriSathi_Member1_Full_Pipeline.ipynb
├── src/                            # importable modules used by both the notebook and scripts/
│   ├── utils.py
│   ├── dataset.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── image_quality.py
│   ├── prediction.py
│   └── tflite_utils.py
├── scripts/                        # standalone CLI entry points
├── Models/{CNN,MobileNetV2,MobileNetV3,Final}/   # empty — populated by training/conversion
├── Dataset/{Selected_Classes,Train,Val,Test,Field_Test}/  # empty — populated by prepare_dataset.py
├── Results/{Graphs,Confusion_Matrices,Metrics,Predictions,Error_Analysis}/  # empty — populated by evaluation
└── Documentation/
    ├── DATASET_SETUP.md
    ├── COLAB_SETUP.md
    └── MODEL_USAGE.md
```
