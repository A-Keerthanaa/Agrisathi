"""
dataset.py — dataset inspection, quality checks, class selection, and
train/val/test splitting.

Nothing here ever modifies or deletes files inside RAW_DIR.
"""

import os
import shutil
import hashlib
from collections import defaultdict

from PIL import Image


# ------------------------------------------------------------------
# 1. DATASET INSPECTION
# ------------------------------------------------------------------

def inspect_dataset(raw_dir, supported_extensions):
    """
    Recursively inspect raw_dir and return a summary dict:

    {
        "class_counts": {class_name: image_count, ...},
        "empty_folders": [folder_path, ...],
        "unsupported_files": [file_path, ...],
        "total_images": int,
        "num_classes": int,
    }

    Does NOT assume folder names. Only reports what is actually found.
    A folder is treated as a "class" if it is a direct subdirectory of
    raw_dir that contains at least one file (recursively).
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"RAW_DIR does not exist: {raw_dir}")

    class_counts = {}
    empty_folders = []
    unsupported_files = []

    subfolders = sorted(
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    )

    for folder_name in subfolders:
        folder_path = os.path.join(raw_dir, folder_name)

        image_count = 0
        for root, _, files in os.walk(folder_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported_extensions:
                    image_count += 1
                elif not fname.startswith("."):
                    unsupported_files.append(fpath)

        if image_count == 0:
            empty_folders.append(folder_path)
        else:
            class_counts[folder_name] = image_count

    total_images = sum(class_counts.values())

    summary = {
        "class_counts": class_counts,
        "empty_folders": empty_folders,
        "unsupported_files": unsupported_files,
        "total_images": total_images,
        "num_classes": len(class_counts),
    }
    return summary


def print_dataset_summary(raw_dir, summary):
    """Pretty-print the dataset summary produced by inspect_dataset()."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Dataset path      : {raw_dir}")
    print(f"Number of classes : {summary['num_classes']}")
    print(f"Total images      : {summary['total_images']}")
    print("-" * 60)
    print(f"{'Class':<45}{'Images'}")
    print("-" * 60)
    for class_name, count in sorted(summary["class_counts"].items()):
        print(f"{class_name:<45}{count}")
    print("-" * 60)

    if summary["empty_folders"]:
        print(f"\n[WARNING] {len(summary['empty_folders'])} empty folder(s) found:")
        for f in summary["empty_folders"]:
            print(f"  - {f}")

    if summary["unsupported_files"]:
        print(f"\n[WARNING] {len(summary['unsupported_files'])} unsupported file(s) found "
              f"(showing up to 10):")
        for f in summary["unsupported_files"][:10]:
            print(f"  - {f}")


def verify_selected_classes(raw_dir, selected_classes):
    """
    Verify that every class in selected_classes exists as a folder inside
    raw_dir. Raises FileNotFoundError with a clear message if any are
    missing. Never silently substitutes another class.
    """
    missing = []
    for cls in selected_classes:
        cls_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_path):
            missing.append(cls)

    if missing:
        raise FileNotFoundError(
            "The following SELECTED_CLASSES folders were NOT found in "
            f"RAW_DIR ({raw_dir}):\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nStopping. Fix SELECTED_CLASSES or verify the dataset "
              "in Google Drive. No class was substituted automatically."
        )

    print(f"[OK] All {len(selected_classes)} selected classes verified in RAW_DIR.")
    return True


# ------------------------------------------------------------------
# 2. IMAGE QUALITY / INTEGRITY CHECKS
# ------------------------------------------------------------------

def check_image_integrity(image_path):
    """
    Try to open and verify an image file.
    Returns (is_valid, error_message_or_None).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open because verify() leaves the file in a state that
        # can't be used further without reopening.
        with Image.open(image_path) as img:
            img.load()
            _ = img.size
        return True, None
    except Exception as e:
        return False, str(e)


def get_image_mode_and_size(image_path):
    """Return (width, height, mode) for a valid image, or (None, None, None)."""
    try:
        with Image.open(image_path) as img:
            return img.size[0], img.size[1], img.mode
    except Exception:
        return None, None, None


def file_md5(path, chunk_size=8192):
    """Compute the MD5 hash of a file's bytes (for exact-duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(image_paths):
    """
    Given a list of image file paths, return a dict:
        {md5_hash: [path1, path2, ...]}
    for every hash that has more than one associated path (i.e. an
    exact byte-for-byte duplicate).
    """
    hash_map = defaultdict(list)
    for path in image_paths:
        try:
            h = file_md5(path)
            hash_map[h].append(path)
        except Exception:
            continue
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def find_near_duplicates(image_paths, hash_size=8):
    """
    Find near-duplicate images using perceptual hashing (average hash).
    Requires the `imagehash` package; if unavailable, returns an empty
    dict and prints a note instead of failing the whole pipeline.

    Returns: {phash_string: [path1, path2, ...]} for hashes shared by
    more than one image.
    """
    try:
        import imagehash
    except ImportError:
        print("[INFO] `imagehash` package not installed — skipping "
              "near-duplicate detection. Run `pip install imagehash` "
              "to enable this check.")
        return {}

    hash_map = defaultdict(list)
    for path in image_paths:
        try:
            with Image.open(path) as img:
                phash = str(imagehash.average_hash(img, hash_size=hash_size))
                hash_map[phash].append(path)
        except Exception:
            continue

    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def run_quality_report(raw_dir, selected_classes, supported_extensions,
                        min_dimension=32, check_duplicates=True):
    """
    Run the full data-quality checking stage described in the project spec:
    corrupted images, unreadable images, unsupported extensions (already
    covered in inspect_dataset), image dimensions/modes, extremely small
    images, duplicate filenames, exact duplicates, and (optionally)
    near-duplicates.

    Returns a report dict. Does NOT delete or modify any files.
    """
    report = {
        "corrupted": [],
        "too_small": [],
        "mode_counts": defaultdict(int),
        "duplicate_filenames": [],
        "exact_duplicates": {},
        "near_duplicates": {},
        "images_checked": 0,
    }

    all_paths = []
    filename_map = defaultdict(list)

    for cls in selected_classes:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_extensions:
                continue
            fpath = os.path.join(cls_dir, fname)
            all_paths.append(fpath)
            filename_map[fname].append(fpath)

    report["images_checked"] = len(all_paths)

    for fpath in all_paths:
        is_valid, err = check_image_integrity(fpath)
        if not is_valid:
            report["corrupted"].append((fpath, err))
            continue

        w, h, mode = get_image_mode_and_size(fpath)
        if w is None:
            continue
        if min(w, h) < min_dimension:
            report["too_small"].append((fpath, w, h))
        report["mode_counts"][mode] += 1

    for fname, paths in filename_map.items():
        if len(paths) > 1:
            report["duplicate_filenames"].append((fname, paths))

    if check_duplicates:
        valid_paths = [p for p in all_paths if p not in {c[0] for c in report["corrupted"]}]
        report["exact_duplicates"] = find_exact_duplicates(valid_paths)
        report["near_duplicates"] = find_near_duplicates(valid_paths)

    return report


def print_quality_report(report):
    """Pretty-print the output of run_quality_report()."""
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Images checked        : {report['images_checked']}")
    print(f"Corrupted images      : {len(report['corrupted'])}")
    print(f"Extremely small images: {len(report['too_small'])}")
    print(f"Duplicate filenames   : {len(report['duplicate_filenames'])}")
    print(f"Exact duplicate groups: {len(report['exact_duplicates'])}")
    print(f"Near-duplicate groups : {len(report['near_duplicates'])}")
    print("\nImage mode distribution:")
    for mode, count in report["mode_counts"].items():
        print(f"  {mode}: {count}")

    if report["corrupted"]:
        print("\nSample corrupted images (up to 5):")
        for fpath, err in report["corrupted"][:5]:
            print(f"  - {fpath} ({err})")


# ------------------------------------------------------------------
# 3. SELECTED-CLASS DATASET PREPARATION (copy, non-destructive)
# ------------------------------------------------------------------

def prepare_selected_dataset(raw_dir, selected_dir, selected_classes,
                              supported_extensions):
    """
    Copy only `selected_classes` from raw_dir into selected_dir.

    - Never modifies or deletes RAW_DIR.
    - Safe to re-run: skips files that already exist with the same size.
    - Verifies final class/image counts and returns a summary dict.
    """
    os.makedirs(selected_dir, exist_ok=True)

    summary = {"copied": 0, "skipped_existing": 0, "class_counts": {}}

    for cls in selected_classes:
        src_dir = os.path.join(raw_dir, cls)
        dst_dir = os.path.join(selected_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"[WARNING] Source class folder missing, skipping: {src_dir}")
            continue

        count = 0
        for fname in os.listdir(src_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_extensions:
                continue

            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            if os.path.exists(dst_path) and \
                    os.path.getsize(dst_path) == os.path.getsize(src_path):
                summary["skipped_existing"] += 1
            else:
                shutil.copy2(src_path, dst_path)
                summary["copied"] += 1

            count += 1

        summary["class_counts"][cls] = count

    summary["total_images"] = sum(summary["class_counts"].values())
    summary["num_classes"] = len(summary["class_counts"])
    return summary


# ------------------------------------------------------------------
# 4. STRATIFIED TRAIN / VAL / TEST SPLIT
# ------------------------------------------------------------------

def stratified_split(selected_dir, train_dir, val_dir, test_dir,
                      selected_classes, supported_extensions,
                      train_split=0.70, val_split=0.15, test_split=0.15,
                      seed=42):
    """
    Perform a reproducible, stratified, class-aware split of images from
    selected_dir into train_dir / val_dir / test_dir.

    - Splitting happens BEFORE any augmentation.
    - No image ever appears in more than one split (copy, not move, but
      each source image is assigned to exactly one destination split).
    - Safe to re-run: if a split already contains files for a class,
      the class is skipped (to avoid re-splitting non-deterministically
      after files have already been distributed/possibly augmented
      elsewhere). Delete the Train/Val/Test folders manually if you
      need to force a fresh split.
    """
    assert abs((train_split + val_split + test_split) - 1.0) < 1e-6, \
        "train_split + val_split + test_split must sum to 1.0"

    import random
    rng = random.Random(seed)

    for d in (train_dir, val_dir, test_dir):
        os.makedirs(d, exist_ok=True)

    split_summary = {"train": {}, "val": {}, "test": {}}

    for cls in selected_classes:
        src_dir = os.path.join(selected_dir, cls)
        if not os.path.isdir(src_dir):
            print(f"[WARNING] Missing selected class folder, skipping: {src_dir}")
            continue

        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        test_cls_dir = os.path.join(test_dir, cls)

        already_split = (
            os.path.isdir(train_cls_dir) and len(os.listdir(train_cls_dir)) > 0
        )
        if already_split:
            print(f"[SKIP] {cls}: split already exists, not re-splitting "
                  "(delete Train/Val/Test manually to force a re-split).")
            split_summary["train"][cls] = len(os.listdir(train_cls_dir))
            split_summary["val"][cls] = len(os.listdir(val_cls_dir)) if os.path.isdir(val_cls_dir) else 0
            split_summary["test"][cls] = len(os.listdir(test_cls_dir)) if os.path.isdir(test_cls_dir) else 0
            continue

        images = [
            f for f in os.listdir(src_dir)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]
        images = sorted(images)  # deterministic order before shuffling
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        # remainder goes to test to avoid dropping images due to rounding
        n_test = n - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        for fname in train_files:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(train_cls_dir, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(val_cls_dir, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(test_cls_dir, fname))

        split_summary["train"][cls] = len(train_files)
        split_summary["val"][cls] = len(val_files)
        split_summary["test"][cls] = len(test_files)

        print(f"[OK] {cls}: {n} total -> train={len(train_files)}, "
              f"val={len(val_files)}, test={len(test_files)}")

    return split_summary


def verify_no_leakage(train_dir, val_dir, test_dir, field_test_dir=None):
    """
    Verify there is no filename overlap between train/val/test (and
    optionally field_test) by comparing file MD5 hashes across splits.

    Returns a dict describing any leakage found. An empty "leaks" list
    means no leakage was detected.
    """
    def hash_all(directory):
        hashes = {}
        if not os.path.isdir(directory):
            return hashes
        for root, _, files in os.walk(directory):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    hashes[file_md5(fpath)] = fpath
                except Exception:
                    continue
        return hashes

    train_hashes = hash_all(train_dir)
    val_hashes = hash_all(val_dir)
    test_hashes = hash_all(test_dir)

    splits = {"train": train_hashes, "val": val_hashes, "test": test_hashes}
    if field_test_dir:
        splits["field_test"] = hash_all(field_test_dir)

    leaks = []
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = set(splits[a].keys()) & set(splits[b].keys())
            for h in common:
                leaks.append({
                    "hash": h,
                    "split_a": a, "path_a": splits[a][h],
                    "split_b": b, "path_b": splits[b][h],
                })

    if leaks:
        print(f"[WARNING] {len(leaks)} leaked duplicate image(s) found across splits!")
    else:
        print("[OK] No data leakage detected across splits (based on exact file hashes).")

    return {"leaks": leaks, "num_leaks": len(leaks)}
