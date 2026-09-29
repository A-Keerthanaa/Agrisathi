"""
AgriSathi — Member 1: Image-Based Crop Disease Detection
==========================================================
MASTER CONFIGURATION

Every script/notebook cell in this project imports paths and
hyperparameters from here. Do NOT hardcode paths elsewhere.

Edit DRIVE_MOUNT / PROJECT_DIR / RAW_DIR below only if your
Google Drive layout differs from the default.
"""

import os

# ------------------------------------------------------------------
# 1. GOOGLE DRIVE / PROJECT PATHS
# ------------------------------------------------------------------
# Google Drive is mounted at /content/drive in Google Colab.

DRIVE_MOUNT = "/content/drive"

PROJECT_DIR = (
    f"{DRIVE_MOUNT}/MyDrive/"
    "AgriSathi_Member1_Crop_Disease_Detection/"
    "AgriSathi_Member1"
)

DATASET_DIR = f"{PROJECT_DIR}/Dataset"

# Raw PlantVillage dataset — READ ONLY.
RAW_DIR = (
    f"{DRIVE_MOUNT}/MyDrive/"
    "plantvillage/"
    "PlantVillage"
)

# Working datasets derived from RAW_DIR
SELECTED_DIR = f"{DATASET_DIR}/Selected_Classes"
TRAIN_DIR = f"{DATASET_DIR}/Train"
VAL_DIR = f"{DATASET_DIR}/Val"
TEST_DIR = f"{DATASET_DIR}/Test"
FIELD_TEST_DIR = f"{DATASET_DIR}/Field_Test"

# Models
MODEL_DIR = f"{PROJECT_DIR}/Models"
CNN_MODEL_DIR = f"{MODEL_DIR}/CNN"
MOBILENETV2_MODEL_DIR = f"{MODEL_DIR}/MobileNetV2"
MOBILENETV3_MODEL_DIR = f"{MODEL_DIR}/MobileNetV3"
FINAL_MODEL_DIR = f"{MODEL_DIR}/Final"

# Results
RESULT_DIR = f"{PROJECT_DIR}/Results"
GRAPH_DIR = f"{RESULT_DIR}/Graphs"
CONF_MATRIX_DIR = f"{RESULT_DIR}/Confusion_Matrices"
PRED_DIR = f"{RESULT_DIR}/Predictions"
METRICS_DIR = f"{RESULT_DIR}/Metrics"
ERROR_DIR = f"{RESULT_DIR}/Error_Analysis"

ALL_PROJECT_DIRS = [
    DATASET_DIR,
    SELECTED_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    FIELD_TEST_DIR,
    MODEL_DIR,
    CNN_MODEL_DIR,
    MOBILENETV2_MODEL_DIR,
    MOBILENETV3_MODEL_DIR,
    FINAL_MODEL_DIR,
    RESULT_DIR,
    GRAPH_DIR,
    CONF_MATRIX_DIR,
    PRED_DIR,
    METRICS_DIR,
    ERROR_DIR,
]


def ensure_project_dirs():
    """Create every project directory if missing. Safe to call repeatedly."""
    for d in ALL_PROJECT_DIRS:
        os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------
# 2. SELECTED DISEASE CLASSES (first implementation)
# ------------------------------------------------------------------
# Known counts as of dataset handoff (VERIFY at runtime — never trust
# blindly; the actual dataset in Drive is the source of truth):
#   Pepper__bell___Bacterial_spot : 997
#   Potato___Early_blight         : 1000
#   Potato___Late_blight          : 1000
#   Tomato_Bacterial_spot         : 2127
#   Tomato_Early_blight           : 1000
#   Tomato_Late_blight            : 1909
#   Expected total                : 8033

SELECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
]

# ------------------------------------------------------------------
# 3. IMAGE / TRAINING HYPERPARAMETERS
# ------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # (height, width)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# Train / Val / Test split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Supported image extensions
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Minimum acceptable image dimension (width or height) to be
# considered usable rather than "extremely small"
MIN_IMAGE_DIMENSION = 32

# ------------------------------------------------------------------
# 4. TRAINING CONTROL
# ------------------------------------------------------------------
MAX_EPOCHS_CNN = 40
MAX_EPOCHS_HEAD = 15          # frozen-base "head only" stage (transfer models)
MAX_EPOCHS_FINE_TUNE = 20     # fine-tuning stage (transfer models)

EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

FINE_TUNE_LEARNING_RATE = 1e-5   # configurable, not claimed to be optimal
HEAD_LEARNING_RATE = 1e-3

# Number of top-most layers to unfreeze during fine-tuning
# (counted from the end of the base model)
FINE_TUNE_UNFREEZE_LAYERS = 30

# ------------------------------------------------------------------
# 5. PREDICTION / DEPLOYMENT
# ------------------------------------------------------------------
# NOT claimed to be scientifically optimal — tune later using
# validation data.
CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_MESSAGE = (
    "Low confidence — capture a clearer image or seek expert verification."
)

# Image-quality heuristic thresholds (NOT scientifically validated —
# simple heuristics only). Tune as needed.
QUALITY_MIN_RESOLUTION = 64          # min(width, height) in pixels
QUALITY_DARK_MEAN_THRESHOLD = 40.0   # mean pixel intensity (0-255) below = too dark
QUALITY_BRIGHT_MEAN_THRESHOLD = 220.0  # mean pixel intensity above = too bright
QUALITY_BLUR_LAPLACIAN_THRESHOLD = 100.0  # variance of Laplacian below = likely blurry

# ------------------------------------------------------------------
# 6. INT8 QUANTIZATION
# ------------------------------------------------------------------
# Number of TRAINING-only images sampled to build the representative
# dataset used for full-integer post-training quantization.
REPRESENTATIVE_DATASET_SIZE = 200

# ------------------------------------------------------------------
# 7. MISC
# ------------------------------------------------------------------
RANDOM_STATE = SEED  # alias, used by sklearn-style splits
