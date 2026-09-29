# Google Colab Setup

## 1. Get the project into Colab

Two options:

**Option A — Upload the project folder into your Google Drive**
1. Extract `AgriSathi_Member1_Crop_Disease_Detection.zip` on your computer.
2. Upload the extracted `AgriSathi_Member1/` folder into `MyDrive/`
   (anywhere is fine, e.g. `MyDrive/AgriSathi_Member1/`).
3. In Colab, mount Drive, then `%cd /content/gdrive/MyDrive/AgriSathi_Member1`.
4. Make sure `sys.path` includes that folder so `import config` /
   `import src` work — the notebook already does this in its first cell.

**Option B — Upload just the notebook**
1. Open `notebooks/AgriSathi_Member1_Full_Pipeline.ipynb` directly in
   Colab (File > Upload notebook).
2. Also upload the `config/` and `src/` folders into the Colab
   session's `/content/` directory (or clone/copy them from Drive if
   you placed them there), so the notebook's imports resolve.

Either way, your **dataset** stays exactly where it already is:
`MyDrive/AgriSathi/Dataset/Raw/plantvillage/PlantVillage` — this
project's code and your dataset are in two different places on
purpose, and `config/config.py` is what connects them.

## 2. Enable a GPU (recommended, not required)

Runtime > Change runtime type > Hardware accelerator > GPU (T4 is fine).

The notebook detects GPU availability automatically and prints whether
one was found; CPU-only will still work, just slower to train.

## 3. Run the notebook top to bottom

Sections are numbered and match the pipeline order in the project
spec (environment setup → dataset inspection → cleaning → EDA →
selected dataset → split → tf.data pipeline → augmentation → CNN →
MobileNetV2 → MobileNetV3 → evaluation → confusion matrices →
comparison → error analysis → field test → TFLite float32 → INT8 →
TFLite validation → final prediction demo → final deployment package).

Training-dependent sections must be run in order the first time,
because later sections load checkpoints saved by earlier ones. Once
models are trained, you can re-run just the evaluation/TFLite sections
without retraining.

## 4. Re-running cells safely

Every stage is idempotent:
- Dataset copy/split steps detect existing output and skip re-copying.
- `os.makedirs(..., exist_ok=True)` is used everywhere directories are
  created.
- Model checkpoints use `save_best_only=True`, so re-running training
  will only overwrite the checkpoint if a better epoch is found.

## 5. Common Colab issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `RAW_DIR does not exist` | Wrong Drive path or Drive not mounted | Re-run the Drive-mount cell; check `config.RAW_DIR` matches your actual folder |
| Training crashes with an OOM error | Batch size too large for the assigned GPU | Lower `BATCH_SIZE` in `config/config.py` (e.g. 16) |
| `ModuleNotFoundError: No module named 'config'` | Notebook's working directory isn't the project root | Re-run the first cell, which inserts the project root into `sys.path` |
| INT8 conversion fails on first attempt | Model contains an op without a full-integer kernel | `convert_to_tflite_int8()` automatically retries with float32 input/output and INT8 weights only |
