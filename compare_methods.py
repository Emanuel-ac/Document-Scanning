import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

from document_scanner import DETECTION_METHODS, iter_input_files, resolve_csv_output_dir, scan_document
from generate_plots import load_metrics


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _condition_rows(records: list[dict], key: str, label: str) -> list[dict]:
    grouped = defaultdict(lambda: {"usable": [], "detected": []})
    for record in records:
        condition = record.get(key)
        if not condition:
            continue
        grouped[condition]["usable"].append(float(record.get("usable_scan", 0)))
        grouped[condition]["detected"].append(float(record.get("detection_success", 0)))

    rows = []
    for condition, values in grouped.items():
        rows.append(
            {
                "condition_type": label,
                "condition": condition,
                "count": len(values["usable"]),
                "detection_rate": round(sum(values["detected"]) / max(len(values["detected"]), 1) * 100.0, 2),
                "usable_scan_rate": round(sum(values["usable"]) / max(len(values["usable"]), 1) * 100.0, 2),
            }
        )
    return rows


def _build_detector_summary(benchmark_rows: list[dict]) -> list[dict]:
    summary_by_method = defaultdict(list)
    wins = Counter()

    for row in benchmark_rows:
        summary_by_method[row["forced_detection_method"]].append(row)

    grouped_by_file = defaultdict(list)
    for row in benchmark_rows:
        grouped_by_file[row["file"]].append(row)
    for file_rows in grouped_by_file.values():
        winner = max(file_rows, key=lambda item: float(item["overall_score"]))
        wins[winner["forced_detection_method"]] += 1

    summary_rows = []
    for method_name in DETECTION_METHODS:
        rows = summary_by_method.get(method_name, [])
        if not rows:
            continue
        summary_rows.append(
            {
                "forced_detection_method": method_name,
                "count": len(rows),
                "detection_rate": round(
                    sum(float(row["detection_success"]) for row in rows) / max(len(rows), 1) * 100.0,
                    2,
                ),
                "usable_scan_rate": round(
                    sum(float(row["usable_scan"]) for row in rows) / max(len(rows), 1) * 100.0,
                    2,
                ),
                "avg_overall_score": round(
                    sum(float(row["overall_score"]) for row in rows) / max(len(rows), 1),
                    2,
                ),
                "avg_detection_confidence": round(
                    sum(float(row["detection_confidence"]) for row in rows) / max(len(rows), 1),
                    3,
                ),
                "avg_binarization_score": round(
                    sum(float(row["binarization_score"]) for row in rows) / max(len(rows), 1),
                    2,
                ),
                "wins": wins.get(method_name, 0),
            }
        )
    return summary_rows


def _run_detector_benchmark(
    input_dir: str,
    csv_output_dir: Path | None = None,
    quality_threshold: float = 60.0,
    reject_low_quality: bool = True,
    detector_binarization_mode: str = "otsu_deshadow",
) -> tuple[list[dict], list[dict]]:
    files = iter_input_files(Path(input_dir), recursive=False)
    if not files:
        return [], []

    benchmark_rows = []
    total_jobs = len(files) * len(DETECTION_METHODS)
    start_time = time.time()

    print(
        f"    [INFO] Benchmark detectoare... {len(files)} imagini x {len(DETECTION_METHODS)} metode = {total_jobs} rulari"
    )
    print(f"    [INFO] Binarizare fixa pentru benchmark: {detector_binarization_mode}")
    for file_idx, file in enumerate(files, 1):
        file_rows = []
        image_start = time.time()
        for method_idx, method_name in enumerate(DETECTION_METHODS, 1):
            job_idx = (file_idx - 1) * len(DETECTION_METHODS) + method_idx
            result = scan_document(
                str(file),
                output_dir=None,
                detection_mode=method_name,
                binarization_mode=detector_binarization_mode,
                quality_threshold=quality_threshold,
                reject_low_quality=reject_low_quality,
                candidate_binarization_methods=(detector_binarization_mode,),
            )
            if not result.success or not result.metrics:
                continue

            row = {
                "file": file.name,
                "forced_detection_method": method_name,
                "resolved_detection_method": result.method,
                "detection_success": int(result.method != "fallback"),
                "usable_scan": int(result.accepted),
                "quality_status": "accepted" if result.accepted else "rejected",
                "rejection_reason": result.rejection_reason,
                "binarization_method": result.binarization_method,
                "benchmark_binarization_mode": detector_binarization_mode,
                "overall_score": result.metrics.overall_score,
                "detection_confidence": result.metrics.detection_confidence,
                "sharpness": result.metrics.sharpness,
                "skew_angle": result.metrics.skew_angle,
                "binarization_score": result.metrics.binarization_score,
            }
            benchmark_rows.append(row)
            file_rows.append(row)
            print(
                f"      [{job_idx:03d}/{total_jobs:03d}] {file.name:<32} "
                f"det={method_name:<9} score={result.metrics.overall_score:>5.1f} "
                f"quality={'accepted' if result.accepted else 'rejected'}",
                flush=True,
            )

        elapsed = time.time() - image_start
        if file_rows:
            winner = max(file_rows, key=lambda item: float(item["overall_score"]))
            total_elapsed = time.time() - start_time
            avg_per_image = total_elapsed / max(file_idx, 1)
            remaining_images = len(files) - file_idx
            eta_seconds = avg_per_image * remaining_images
            print(
                f"      -> best pentru {file.name}: {winner['forced_detection_method']} "
                f"(score={float(winner['overall_score']):.1f}) "
                f"[img {file_idx}/{len(files)}, {elapsed:.1f}s, ETA ~{eta_seconds/60.0:.1f} min]",
                flush=True,
            )

        if csv_output_dir:
            _write_csv(csv_output_dir / "detector_comparison.csv", benchmark_rows)
            _write_csv(csv_output_dir / "detector_summary.csv", _build_detector_summary(benchmark_rows))

    summary_rows = _build_detector_summary(benchmark_rows)
    print(f"    [INFO] Benchmark detectoare finalizat in {(time.time() - start_time) / 60.0:.1f} minute.")
    return benchmark_rows, summary_rows


def run_comparison(
    input_dir: str,
    output_dir: str,
    csv_output_dir: str | None = None,
    run_detector_benchmark: bool = True,
    quality_threshold: float = 60.0,
    reject_low_quality: bool = True,
    detector_binarization_mode: str = "otsu_deshadow",
) -> None:
    out = resolve_csv_output_dir(output_dir, csv_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_metrics(output_dir, csv_input_dir=str(out))
    if not records:
        print("    [WARN] Nu exista summary.csv sau metrici pentru comparatie.")
        return

    score_columns = sorted([key for key in records[0].keys() if key.startswith("binary_score_")])
    if not score_columns:
        print("    [WARN] Nu exista scoruri de binarizare pentru comparatie.")
        return

    methods = [column.replace("binary_score_", "") for column in score_columns]
    winner_counts = Counter()
    selected_counts = Counter(record.get("binarization_method", "unknown") for record in records)
    comparison_rows = []

    for record in records:
        scores = {}
        for method in methods:
            value = float(record.get(f"binary_score_{method}", -1))
            if value >= 0:
                scores[method] = value

        if not scores:
            continue

        winner = max(scores, key=scores.get)
        winner_counts[winner] += 1
        comparison_rows.append(
            {
                "file": record.get("file", ""),
                "selected_method": record.get("binarization_method", ""),
                "winning_method": winner,
                **{method: round(score, 2) for method, score in scores.items()},
            }
        )

    summary_rows = []
    for method in methods:
        values = [float(record.get(f"binary_score_{method}", -1)) for record in records]
        values = [value for value in values if value >= 0]
        if not values:
            continue
        summary_rows.append(
            {
                "method": method,
                "avg_score": round(sum(values) / len(values), 2),
                "wins": winner_counts.get(method, 0),
                "selected_count": selected_counts.get(method, 0),
            }
        )

    experiment_rows = []
    experiment_rows.extend(_condition_rows(records, "background_complexity", "background"))
    experiment_rows.extend(_condition_rows(records, "lighting_condition", "lighting"))

    _write_csv(out / "binarization_comparison.csv", comparison_rows)
    _write_csv(out / "binarization_summary.csv", summary_rows)
    _write_csv(out / "experiment_success_rates.csv", experiment_rows)

    print(f"    [OK] {out / 'binarization_comparison.csv'}")
    print(f"    [OK] {out / 'binarization_summary.csv'}")
    print(f"    [OK] {out / 'experiment_success_rates.csv'}")
    for row in summary_rows:
        print(
            f"    {row['method']:<18} avg={row['avg_score']:>5.1f} "
            f"wins={row['wins']:>3} selected={row['selected_count']:>3}"
        )

    if run_detector_benchmark:
        detector_rows, detector_summary = _run_detector_benchmark(
            input_dir,
            csv_output_dir=out,
            quality_threshold=quality_threshold,
            reject_low_quality=reject_low_quality,
            detector_binarization_mode=detector_binarization_mode,
        )
        _write_csv(out / "detector_comparison.csv", detector_rows)
        _write_csv(out / "detector_summary.csv", detector_summary)
        if detector_rows:
            print(f"    [OK] {out / 'detector_comparison.csv'}")
        if detector_summary:
            print(f"    [OK] {out / 'detector_summary.csv'}")
            for row in detector_summary:
                print(
                    f"    {row['forced_detection_method']:<18} det={row['detection_rate']:>6.2f}% "
                    f"usable={row['usable_scan_rate']:>6.2f}% "
                    f"avg={row['avg_overall_score']:>5.1f} wins={row['wins']:>3}"
                )


def main():
    parser = argparse.ArgumentParser(description="Compare binarization methods using summary.csv")
    parser.add_argument("--input", default="dataset", help="Folderul de input; pastrat pentru compatibilitate")
    parser.add_argument("--output", default="output", help="Folderul care contine metricile/imaginile rezultate")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Folder separat pentru CSV-urile generate; implicit foloseste un folder sibling de forma <output>_csv",
    )
    parser.add_argument(
        "--skip-detector-benchmark",
        action="store_true",
        help="Nu rula benchmark-ul comparativ pentru detectorii de document",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=60.0,
        help="Pragul minim al scorului overall pentru a considera un scan acceptat",
    )
    parser.add_argument(
        "--no-reject-low-quality",
        action="store_true",
        help="Nu marca scanurile slabe drept rejectate in benchmark",
    )
    parser.add_argument(
        "--detector-binarization",
        default="otsu_deshadow",
        choices=["adaptive_gaussian", "otsu_deshadow", "sauvola_deshadow", "hybrid_deshadow"],
        help="Metoda de binarizare folosita fix in benchmark-ul detectorilor",
    )
    args = parser.parse_args()
    run_comparison(
        args.input,
        args.output,
        args.csv_output,
        run_detector_benchmark=not args.skip_detector_benchmark,
        quality_threshold=args.quality_threshold,
        reject_low_quality=not args.no_reject_low_quality,
        detector_binarization_mode=args.detector_binarization,
    )


if __name__ == "__main__":
    main()
