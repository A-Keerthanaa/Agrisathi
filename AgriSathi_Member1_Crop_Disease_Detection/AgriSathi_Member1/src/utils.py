"""
utils.py — environment setup, reproducibility, and small shared helpers.
"""

import os
import sys
import random
import json

import numpy as np


def mount_google_drive(drive_mount="/content/gdrive"):
    """
    Mount Google Drive inside Google Colab.

    Handles the case where Drive is already mounted (does not crash),
    and raises a clear RuntimeError if mounting fails.
    """
    if os.path.ismount(drive_mount):
        print(f"[OK] Google Drive already mounted at: {drive_mount}")
        return

    try:
        from google.colab import drive
    except ImportError as e:
        raise RuntimeError(
            "google.colab is not available. This function must be run "
            "inside Google Colab."
        ) from e

    drive.mount(drive_mount, force_remount=False)

    if not os.path.ismount(drive_mount):
        raise RuntimeError(
            f"Google Drive did not mount at {drive_mount}. "
            "Please re-run and authorize access."
        )

    print(f"[OK] Google Drive mounted at: {drive_mount}")


def set_global_seed(seed=42):
    """Set random seeds for Python, NumPy, and TensorFlow (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"[OK] Global seed set to {seed} (Python, NumPy, TensorFlow)")


def print_environment_info():
    """Print Python / TensorFlow / NumPy / Pandas versions and GPU availability."""
    print("=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)
    print(f"Python version     : {sys.version.split()[0]}")

    try:
        import tensorflow as tf
        print(f"TensorFlow version : {tf.__version__}")
        from tensorflow import keras
        print(f"Keras version      : {keras.__version__}")

        gpu_devices = tf.config.list_physical_devices("GPU")
        if gpu_devices:
            print(f"GPU available      : Yes ({len(gpu_devices)} device(s))")
            for i, gpu in enumerate(gpu_devices):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    name = details.get("device_name", "Unknown GPU")
                except Exception:
                    name = "Unknown GPU"
                print(f"  GPU {i}: {name}")
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"  Note: could not set memory growth ({e})")
        else:
            print("GPU available      : No (running on CPU)")
            print("  In Colab: Runtime > Change runtime type > GPU")
    except ImportError:
        print("TensorFlow is not installed in this environment.")

    try:
        import numpy as _np
        print(f"NumPy version      : {_np.__version__}")
    except ImportError:
        pass

    try:
        import pandas as _pd
        print(f"Pandas version     : {_pd.__version__}")
    except ImportError:
        pass

    print("=" * 60)


def ensure_dirs(dir_list):
    """Create every directory in dir_list if it does not already exist."""
    for d in dir_list:
        os.makedirs(d, exist_ok=True)


def verify_path_exists(path, description="Path", raise_on_missing=True):
    """
    Verify that `path` exists. Prints a clear message either way.
    Raises FileNotFoundError if missing and raise_on_missing=True.
    """
    if os.path.exists(path):
        print(f"[OK] {description} exists: {path}")
        return True

    msg = f"[MISSING] {description} not found at: {path}"
    print(msg)
    if raise_on_missing:
        raise FileNotFoundError(msg)
    return False


def save_json(data, path):
    """Save a Python dict/list to a JSON file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved: {path}")


def load_json(path):
    """Load a JSON file into a Python dict/list."""
    with open(path, "r") as f:
        return json.load(f)


def human_readable_size(num_bytes):
    """Convert a byte count into a human-readable string (e.g. '2.3 MB')."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"
