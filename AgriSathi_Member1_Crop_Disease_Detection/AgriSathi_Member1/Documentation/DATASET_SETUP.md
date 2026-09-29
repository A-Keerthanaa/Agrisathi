# Dataset Setup

## Where the dataset must live

This project does **not** ship with the image dataset — it stays in
your Google Drive and is read directly from Google Colab. Nothing in
this ZIP contains any images.

Expected Google Drive layout:

```
MyDrive/
└── AgriSathi/
    └── Dataset/
        └── Raw/
            └── plantvillage/
                └── PlantVillage/
                    ├── Pepper__bell___Bacterial_spot/
                    ├── Pepper__bell___healthy/
                    ├── Potato___Early_blight/
                    ├── Potato___Late_blight/
                    ├── Potato___healthy/
                    ├── Tomato_Bacterial_spot/
                    ├── Tomato_Early_blight/
                    ├── Tomato_Late_blight/
                    ├── Tomato_Leaf_Mold/
                    ├── Tomato_Septoria_leaf_spot/
                    ├── Tomato_Spider_mites_Two_spotted_spider_mite/
                    ├── Tomato__Target_Spot/
                    ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
                    ├── Tomato__Tomato_mosaic_virus/
                    └── Tomato_healthy/
```

This path is controlled entirely by `config/config.py`:

```python
DRIVE_MOUNT = "/content/gdrive"
PROJECT_DIR = f"{DRIVE_MOUNT}/MyDrive/AgriSathi"
RAW_DIR = f"{PROJECT_DIR}/Dataset/Raw/plantvillage/PlantVillage"
```

If your Drive layout differs, edit these three lines — nothing else
in the codebase hardcodes a path.

## Selected classes for this first implementation

```python
SELECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
]
```

`scripts/inspect_dataset.py` verifies these folders actually exist
before anything else runs. If a folder is missing, the script stops
with a clear error rather than silently substituting another class.

## What the code creates (and never touches)

- `RAW_DIR` is **read-only** — no script in this project ever writes,
  moves, or deletes files inside it.
- `Dataset/Selected_Classes/`, `Dataset/Train/`, `Dataset/Val/`,
  `Dataset/Test/`, `Dataset/Field_Test/` are created and populated by
  the pipeline (copies of the raw images, never the originals).

## Field-test images (optional, add later)

`Dataset/Field_Test/` is created empty. If you later add real
smartphone-captured field images there (organized the same way as the
other class folders), the field-test evaluation cells in the notebook
will automatically pick them up. Until then, the pipeline prints
"No field-test images available yet." and does not fabricate results.
