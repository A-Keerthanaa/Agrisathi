"""
eda.py — exploratory data analysis helpers.

All chart-producing functions save to disk (GRAPH_DIR) and also return
the underlying data so nothing here fabricates numbers silently.
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image


def class_distribution_table(class_counts):
    """Return a sorted list of (class_name, count) tuples plus the total."""
    rows = sorted(class_counts.items(), key=lambda x: x[0])
    total = sum(class_counts.values())
    return rows, total


def plot_class_distribution(class_counts, save_path, title="Class Distribution"):
    """Bar chart of image counts per class, saved to save_path."""
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color="#4C7C4C")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Number of images")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved class distribution chart: {save_path}")


def collect_image_dimensions(class_dir_map, supported_extensions, max_per_class=None):
    """
    class_dir_map: {class_name: directory_path}
    Returns a list of (width, height) tuples across all images found.

    If max_per_class is set, only sample up to that many images per
    class (useful to keep EDA fast on large datasets).
    """
    dims = []
    for cls, cls_dir in class_dir_map.items():
        if not os.path.isdir(cls_dir):
            continue
        files = [
            f for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]
        if max_per_class is not None:
            files = files[:max_per_class]

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                with Image.open(fpath) as img:
                    dims.append(img.size)  # (width, height)
            except Exception:
                continue
    return dims


def image_dimension_stats(dims):
    """
    Given a list of (width, height) tuples, return summary statistics:
    unique dimension count, most common dimensions, min/max/mean width
    and height.
    """
    if not dims:
        return {
            "num_images": 0,
            "num_unique_dimensions": 0,
            "most_common": [],
            "width_stats": {},
            "height_stats": {},
        }

    dim_counter = Counter(dims)
    widths = [d[0] for d in dims]
    heights = [d[1] for d in dims]

    def stats(values):
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "num_images": len(dims),
        "num_unique_dimensions": len(dim_counter),
        "most_common": dim_counter.most_common(10),
        "width_stats": stats(widths),
        "height_stats": stats(heights),
    }


def print_dimension_stats(stats):
    print("=" * 60)
    print("IMAGE DIMENSION STATISTICS")
    print("=" * 60)
    print(f"Images analyzed          : {stats['num_images']}")
    print(f"Unique (width,height)    : {stats['num_unique_dimensions']}")
    if stats["width_stats"]:
        print(f"Width  (min/mean/max)    : {stats['width_stats']['min']} / "
              f"{stats['width_stats']['mean']:.1f} / {stats['width_stats']['max']}")
        print(f"Height (min/mean/max)    : {stats['height_stats']['min']} / "
              f"{stats['height_stats']['mean']:.1f} / {stats['height_stats']['max']}")
    print("\nMost common dimensions (dimension: count):")
    for dim, count in stats["most_common"]:
        print(f"  {dim}: {count}")


def plot_sample_images(class_dir_map, save_path, supported_extensions,
                        samples_per_class=1):
    """
    Save a grid image containing `samples_per_class` sample(s) from every
    class in class_dir_map.
    """
    classes = sorted(class_dir_map.keys())
    n_classes = len(classes)
    if n_classes == 0:
        print("[WARNING] No classes provided for sample image grid.")
        return

    fig, axes = plt.subplots(
        samples_per_class, n_classes,
        figsize=(3 * n_classes, 3 * samples_per_class)
    )
    if samples_per_class == 1:
        axes = [axes] if n_classes == 1 else axes
        axes = [axes] if not hasattr(axes, "__iter__") else list(axes)

    for col, cls in enumerate(classes):
        cls_dir = class_dir_map[cls]
        files = []
        if os.path.isdir(cls_dir):
            files = sorted([
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in supported_extensions
            ])[:samples_per_class]

        for row in range(samples_per_class):
            ax = axes[col] if samples_per_class == 1 else axes[row][col]
            if row < len(files):
                try:
                    with Image.open(os.path.join(cls_dir, files[row])) as img:
                        ax.imshow(img.convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "load error", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "no image", ha="center", va="center")
            ax.axis("off")
            if row == 0:
                ax.set_title(cls, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved sample image grid: {save_path}")
