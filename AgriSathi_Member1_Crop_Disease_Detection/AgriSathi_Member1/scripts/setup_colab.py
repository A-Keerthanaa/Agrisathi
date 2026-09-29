"""
scripts/setup_colab.py

STEP 1 — Environment setup: mount Google Drive, print environment
info, set reproducibility seed, create project directories, and
verify RAW_DIR exists.

Run from the project root:
    python scripts/setup_colab.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src import utils


def main():
    utils.mount_google_drive(config.DRIVE_MOUNT)
    utils.print_environment_info()
    utils.set_global_seed(config.SEED)

    config.ensure_project_dirs()
    print(f"[OK] Project directories verified under: {config.PROJECT_DIR}")

    utils.verify_path_exists(config.RAW_DIR, description="RAW_DIR (raw dataset)",
                              raise_on_missing=False)

    print("\nSTEP 1 COMPLETE — environment ready.")


if __name__ == "__main__":
    main()
