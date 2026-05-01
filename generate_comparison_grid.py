import argparse
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from document_scanner import SUPPORTED_EXTS, read_image


def _iter_files(input_dir: str) -> list[Path]:
    inp = Path(input_dir)
    return sorted([file for file in inp.iterdir() if file.is_file() and file.suffix.lower() in SUPPORTED_EXTS])


def _prepare_for_plot(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def generate_grid(input_dir: str, output_dir: str, max_images: int = 8) -> Path | None:
    files = _iter_files(input_dir)
    if not files:
        print("    [WARN] Nu exista imagini pentru comparison grid.")
        return None

    out = Path(output_dir)
    selected_files = files[: min(max_images, 8)]
    pipeline_rows = []

    for file in selected_files:
        stem = file.stem
        pipeline_path = out / f"{stem}_pipeline.jpg"
        if pipeline_path.exists():
            pipeline = cv2.imread(str(pipeline_path))
            if pipeline is not None:
                pipeline_rows.append((file.name, pipeline))
            continue

    if pipeline_rows:
        columns = 2 if len(pipeline_rows) > 1 else 1
        rows = math.ceil(len(pipeline_rows) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(9 * columns, 6 * rows))
        if rows == 1 and columns == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = list(axes.flatten())

        for axis, (name, pipeline) in zip(axes, pipeline_rows):
            axis.imshow(_prepare_for_plot(pipeline))
            axis.set_title(name)
            axis.axis("off")

        for axis in axes[len(pipeline_rows) :]:
            axis.axis("off")

        fig.suptitle("Document scanner - pipeline grid", fontsize=16, fontweight="bold")
        plt.tight_layout()
        path = out / "comparison_grid.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print("    [OK] comparison_grid.png")
        return path

    rows = []
    for file in selected_files:
        stem = file.stem
        rectified_path = out / f"{stem}_rectified.jpg"
        binary_path = out / f"{stem}_binary.jpg"
        if not rectified_path.exists() or not binary_path.exists():
            continue

        original = read_image(str(file))
        rectified = cv2.imread(str(rectified_path))
        binary = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
        if original is None or rectified is None or binary is None:
            continue
        rows.append((file.name, original, rectified, binary))

    if not rows:
        print("    [WARN] Nu exista suficiente rezultate pentru comparison grid.")
        return None

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for axis_row, (name, original, rectified, binary) in zip(axes, rows):
        items = [
            (_prepare_for_plot(original), f"{name} - input"),
            (_prepare_for_plot(rectified), "rectified"),
            (_prepare_for_plot(binary), "best binary"),
        ]
        for axis, (image, title) in zip(axis_row, items):
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis.set_title(title)
            axis.axis("off")

    fig.suptitle("Document scanner - visual comparison grid", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = out / "comparison_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("    [OK] comparison_grid.png")
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate a visual comparison grid")
    parser.add_argument("--input", default="dataset", help="Folderul cu imaginile originale")
    parser.add_argument("--output", default="output", help="Folderul cu rezultatele pipeline-ului")
    parser.add_argument("--max-images", type=int, default=8, help="Numarul maxim de imagini afisate")
    args = parser.parse_args()
    generate_grid(args.input, args.output, args.max_images)


if __name__ == "__main__":
    main()
