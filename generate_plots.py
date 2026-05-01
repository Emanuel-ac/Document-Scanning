import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("output")
PLOTS_DIR = OUTPUT_DIR
CSV_INPUT_DIR = OUTPUT_DIR

LABEL_COLORS = {
    "Excellent": "#2ecc71",
    "Good": "#3498db",
    "Fair": "#f39c12",
    "Poor": "#e74c3c",
}

METHOD_COLORS = {
    "contour": "#9b59b6",
    "threshold": "#34495e",
    "hough": "#1abc9c",
    "corners": "#e67e22",
    "fallback": "#95a5a6",
}

BIN_METHOD_COLORS = {
    "adaptive_gaussian": "#2980b9",
    "otsu_deshadow": "#16a085",
    "sauvola_deshadow": "#8e44ad",
    "hybrid_deshadow": "#d35400",
}

NUMERIC_FIELDS = {
    "sharpness",
    "skew_angle",
    "contrast",
    "brightness",
    "shadow_level",
    "binarization_score",
    "detection_confidence",
    "overall_score",
    "lighting_variation",
    "background_edge_density",
    "document_area_ratio",
    "detection_success",
    "usable_scan",
    "rejected_input",
}


def set_output_dir(output_dir: str = "output", csv_input_dir: str | None = None) -> None:
    global OUTPUT_DIR, PLOTS_DIR, CSV_INPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    PLOTS_DIR = OUTPUT_DIR
    CSV_INPUT_DIR = Path(csv_input_dir) if csv_input_dir else OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_csv"


def _coerce_numeric_values(record: dict) -> dict:
    coerced = {}
    for key, value in record.items():
        if key in NUMERIC_FIELDS or key.startswith("binary_score_"):
            try:
                coerced[key] = float(value)
            except (TypeError, ValueError):
                coerced[key] = value
        else:
            coerced[key] = value
    return coerced


def _flatten_metrics_json(record: dict) -> dict:
    flat = {}
    for key, value in record.items():
        if key == "binarization_scores" and isinstance(value, dict):
            for method_name, score in value.items():
                flat[f"binary_score_{method_name}"] = score
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                flat[child_key] = child_value
        else:
            flat[key] = value
    return _coerce_numeric_values(flat)


def load_metrics(output_dir: str | None = None, csv_input_dir: str | None = None):
    if output_dir is not None or csv_input_dir is not None:
        set_output_dir(output_dir or str(OUTPUT_DIR), csv_input_dir)

    records = []
    for csv_path in [CSV_INPUT_DIR / "summary.csv", OUTPUT_DIR / "summary.csv"]:
        if csv_path.exists():
            with open(csv_path, encoding="utf-8") as file:
                records = [_coerce_numeric_values(row) for row in csv.DictReader(file)]
            return records

    for jf in sorted(OUTPUT_DIR.glob("*_metrics.json")):
        with open(jf, encoding="utf-8") as file:
            data = json.load(file)
        if "file" not in data:
            data["file"] = jf.stem.replace("_metrics", "")
        records.append(_flatten_metrics_json(data))
    return records


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f6fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#dcdde1",
            "axes.labelcolor": "#2f3640",
            "text.color": "#2f3640",
            "xtick.color": "#2f3640",
            "ytick.color": "#2f3640",
            "grid.color": "#dcdde1",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def label_sharpness(value):
    if value > 800:
        return "Excellent"
    if value > 300:
        return "Good"
    if value > 100:
        return "Fair"
    return "Poor"


def label_overall(value):
    if value >= 80:
        return "Excellent"
    if value >= 60:
        return "Good"
    if value >= 40:
        return "Fair"
    return "Poor"


def plot_detection_rate(records):
    methods = [r.get("detection_method", r.get("method", "unknown")) for r in records]
    count = Counter(methods)

    labels = list(count.keys())
    sizes = [count[label] for label in labels]
    colors = [METHOD_COLORS.get(label, "#bdc3c7") for label in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Plot 1 - Detection methods used", fontsize=15, fontweight="bold")

    wedges, _, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="#ffffff", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontsize(10)
    ax1.set_title("Detection method distribution")

    bars = ax2.bar(labels, sizes, color=colors, edgecolor="#ffffff", linewidth=1.2)
    for bar, value in zip(bars, sizes):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_xlabel("Method")
    ax2.set_ylabel("Images")
    ax2.set_title("Number of images per detector")
    ax2.grid(axis="y")
    ax2.set_ylim(0, max(sizes) * 1.3 if sizes else 1)

    plt.tight_layout()
    path = PLOTS_DIR / "plot_1_detection_rate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_sharpness(records):
    values = [float(r["sharpness"]) for r in records]
    labels = [r.get("sharpness_label", label_sharpness(v)) for r, v in zip(records, values)]
    files = [r.get("file", f"img_{idx}") for idx, r in enumerate(records)]
    colors = [LABEL_COLORS.get(label, "#bdc3c7") for label in labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Plot 2 - Sharpness (Laplacian variance)", fontsize=15, fontweight="bold")

    x = np.arange(len(files))
    ax1.bar(x, values, color=colors, edgecolor="#ffffff", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Laplacian variance")
    ax1.set_title("Sharpness per image")

    for threshold, label, color in [(800, "Excellent", "#2ecc71"), (300, "Good", "#3498db"), (100, "Fair", "#f39c12")]:
        ax1.axhline(threshold, color=color, linestyle="--", alpha=0.7, linewidth=1.2)
        ax1.text(len(files) - 0.5, threshold + 10, label, color=color, ha="right", fontsize=8)

    ax2.hist(values, bins=20, color="#3498db", edgecolor="#ffffff", linewidth=0.8, alpha=0.9)
    ax2.axvline(np.mean(values), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {np.mean(values):.1f}")
    ax2.axvline(np.median(values), color="#f39c12", linestyle="--", linewidth=2, label=f"Median: {np.median(values):.1f}")
    ax2.set_xlabel("Laplacian variance")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Sharpness distribution")
    ax2.legend()
    ax2.grid(axis="y")

    patches = [mpatches.Patch(color=color, label=label) for label, color in LABEL_COLORS.items()]
    ax1.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    path = PLOTS_DIR / "plot_2_sharpness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_skew(records):
    values = [float(r["skew_angle"]) for r in records]
    labels = [r.get("skew_label", "Unknown") for r in records]
    files = [r.get("file", f"img_{idx}") for idx, r in enumerate(records)]
    colors = [LABEL_COLORS.get(label, "#bdc3c7") for label in labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Plot 3 - Skew angle distribution", fontsize=15, fontweight="bold")

    x = np.arange(len(files))
    ax1.bar(x, values, color=colors, edgecolor="#ffffff", linewidth=0.8)
    ax1.axhline(0, color="#2f3640", linewidth=1)
    ax1.axhline(5, color="#f39c12", linestyle="--", alpha=0.7, linewidth=1.2, label="+/-5 deg")
    ax1.axhline(-5, color="#f39c12", linestyle="--", alpha=0.7, linewidth=1.2)
    ax1.axhline(15, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.2, label="+/-15 deg")
    ax1.axhline(-15, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Skew angle (deg)")
    ax1.set_title("Skew angle per image")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y")

    counts = Counter(labels)
    pie_items = [(counts.get(label, 0), label, color) for label, color in LABEL_COLORS.items() if counts.get(label, 0) > 0]
    if pie_items:
        ax2.pie(
            [size for size, _, _ in pie_items],
            labels=[f"{label} ({size})" for size, label, _ in pie_items],
            colors=[color for _, _, color in pie_items],
            autopct="%1.0f%%",
            startangle=140,
            wedgeprops=dict(edgecolor="#ffffff", linewidth=2),
        )
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("Skew labels")

    plt.tight_layout()
    path = PLOTS_DIR / "plot_3_skew_angle.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_overall_scores(records):
    scores = [float(r.get("overall_score", 0)) for r in records]
    labels = [r.get("overall_label", label_overall(score)) for r, score in zip(records, scores)]
    files = [r.get("file", f"img_{idx}") for idx, r in enumerate(records)]
    colors = [LABEL_COLORS.get(label, "#bdc3c7") for label in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Plot 4 - Overall quality score (0-100)", fontsize=15, fontweight="bold")

    x = np.arange(len(files))
    bars = ax.bar(x, scores, color=colors, edgecolor="#ffffff", linewidth=0.8)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{score:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.axhline(80, color="#2ecc71", linestyle="--", alpha=0.7, linewidth=1.5, label="80 Excellent")
    ax.axhline(60, color="#3498db", linestyle="--", alpha=0.7, linewidth=1.5, label="60 Good")
    ax.axhline(40, color="#f39c12", linestyle="--", alpha=0.7, linewidth=1.5, label="40 Fair")

    mean_score = np.mean(scores)
    ax.axhline(mean_score, color="#2f3640", linestyle="-", linewidth=2, label=f"Mean: {mean_score:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "plot_4_overall_scores.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_quality_breakdown(records):
    def norm_sharpness(value):
        return min(float(value) / 800.0 * 100.0, 100.0)

    def norm_skew(value):
        return max(0.0, 100.0 - abs(float(value)) * 4.0)

    def norm_contrast(value):
        return min(float(value) / 40.0 * 100.0, 100.0)

    files = [r.get("file", f"img_{idx}") for idx, r in enumerate(records)]
    sharp_norm = [norm_sharpness(r["sharpness"]) for r in records]
    skew_norm = [norm_skew(r["skew_angle"]) for r in records]
    contrast_norm = [norm_contrast(r["contrast"]) for r in records]
    binary_score = [float(r.get("binarization_score", 0)) for r in records]
    overall = [float(r.get("overall_score", 0)) for r in records]

    x = np.arange(len(files))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle("Plot 5 - Quality breakdown per image", fontsize=14, fontweight="bold")

    ax.bar(x - 2 * bar_width, sharp_norm, bar_width, label="Sharpness", color="#9b59b6", alpha=0.85)
    ax.bar(x - 1 * bar_width, skew_norm, bar_width, label="Skew OK", color="#1abc9c", alpha=0.85)
    ax.bar(x, contrast_norm, bar_width, label="Contrast", color="#e67e22", alpha=0.85)
    ax.bar(x + 1 * bar_width, binary_score, bar_width, label="Binary score", color="#3498db", alpha=0.85)
    ax.bar(x + 2 * bar_width, overall, bar_width, label="Overall", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Normalized score (0-100)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "plot_5_quality_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _aggregate_condition(records, key):
    grouped = defaultdict(lambda: {"usable": [], "detected": []})
    for record in records:
        label = record.get(key)
        if not label:
            continue
        grouped[label]["usable"].append(float(record.get("usable_scan", 0)))
        grouped[label]["detected"].append(float(record.get("detection_success", 0)))
    labels = list(grouped.keys())
    usable = [100.0 * np.mean(grouped[label]["usable"]) for label in labels]
    detected = [100.0 * np.mean(grouped[label]["detected"]) for label in labels]
    return labels, usable, detected


def plot_success_by_condition(records):
    background_labels, background_usable, background_detected = _aggregate_condition(records, "background_complexity")
    lighting_labels, lighting_usable, lighting_detected = _aggregate_condition(records, "lighting_condition")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plot 6 - Success rate vs background and lighting", fontsize=15, fontweight="bold")

    for axis, labels, usable, detected, title in [
        (axes[0], background_labels, background_usable, background_detected, "Background complexity"),
        (axes[1], lighting_labels, lighting_usable, lighting_detected, "Lighting condition"),
    ]:
        x = np.arange(len(labels))
        width = 0.35
        axis.bar(x - width / 2, detected, width, label="Detection success", color="#3498db")
        axis.bar(x + width / 2, usable, width, label="Usable scan", color="#2ecc71")
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Rate (%)")
        axis.set_title(title)
        axis.grid(axis="y")
        axis.legend(fontsize=8)

    plt.tight_layout()
    path = PLOTS_DIR / "plot_6_success_by_condition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_binarization_methods(records):
    score_columns = sorted([key for key in records[0].keys() if key.startswith("binary_score_")]) if records else []
    methods = [column.replace("binary_score_", "") for column in score_columns]
    valid_methods = []
    avg_scores = []

    for method, column in zip(methods, score_columns):
        values = [float(record[column]) for record in records if float(record[column]) >= 0]
        if not values:
            continue
        valid_methods.append(method)
        avg_scores.append(float(np.mean(values)))

    wins = Counter()
    for record in records:
        available = {}
        for method in valid_methods:
            value = float(record.get(f"binary_score_{method}", -1))
            if value >= 0:
                available[method] = value
        if available:
            winner = max(available, key=available.get)
            wins[winner] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plot 7 - Binarization method comparison", fontsize=15, fontweight="bold")

    colors = [BIN_METHOD_COLORS.get(method, "#7f8c8d") for method in valid_methods]
    axes[0].bar(valid_methods, avg_scores, color=colors, edgecolor="#ffffff", linewidth=1.2)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Average binary score")
    axes[0].set_title("Average heuristic score")
    axes[0].grid(axis="y")
    axes[0].tick_params(axis="x", rotation=20)

    win_values = [wins.get(method, 0) for method in valid_methods]
    axes[1].bar(valid_methods, win_values, color=colors, edgecolor="#ffffff", linewidth=1.2)
    axes[1].set_ylabel("Images won")
    axes[1].set_title("Best method count")
    axes[1].grid(axis="y")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    path = PLOTS_DIR / "plot_7_binarization_methods.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main(output_dir: str = "output", csv_input_dir: str | None = None):
    set_output_dir(output_dir, csv_input_dir)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    records = load_metrics()
    if not records:
        print("[WARN] Niciun fisier de metrici gasit.")
        print("       Ruleaza mai intai: python document_scanner.py <input_folder>")
        return []

    print(f"[INFO] {len(records)} inregistrari incarcate din {OUTPUT_DIR}/")
    saved = []
    saved.append(plot_detection_rate(records))
    print("  OK plot_1_detection_rate.png")
    saved.append(plot_sharpness(records))
    print("  OK plot_2_sharpness.png")
    saved.append(plot_skew(records))
    print("  OK plot_3_skew_angle.png")
    saved.append(plot_overall_scores(records))
    print("  OK plot_4_overall_scores.png")
    saved.append(plot_quality_breakdown(records))
    print("  OK plot_5_quality_breakdown.png")
    saved.append(plot_success_by_condition(records))
    print("  OK plot_6_success_by_condition.png")

    if any(key.startswith("binary_score_") for key in records[0].keys()):
        saved.append(plot_binarization_methods(records))
        print("  OK plot_7_binarization_methods.png")

    print(f"\nGraficele au fost salvate in: {PLOTS_DIR}/")
    for idx, path in enumerate(saved, 1):
        print(f"  {idx}. {path.name}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for the document scanner output")
    parser.add_argument("--output", default="output", help="Folderul care contine imaginile, metricile JSON si graficele")
    parser.add_argument(
        "--csv-input",
        default=None,
        help="Folderul din care se citeste summary.csv; implicit foloseste un folder sibling de forma <output>_csv",
    )
    args = parser.parse_args()
    main(output_dir=args.output, csv_input_dir=args.csv_input)
