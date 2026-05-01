import argparse
import time
from pathlib import Path

import numpy as np

from document_scanner import (
    BINARIZATION_METHODS,
    DETECTION_METHODS,
    resolve_csv_output_dir,
    result_to_summary_row,
    scan_document,
    write_summary,
)
from generate_plots import main as generate_plots_main

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".heic", ".heif"}


def run_pipeline(
    input_dir,
    output_dir,
    debug=False,
    skip_compare=False,
    detection="auto",
    binarization="auto",
    save_all_binaries=False,
    save_stages_separately=False,
    generate_plots=False,
    generate_grid=False,
    csv_output_dir=None,
    quality_threshold=60.0,
    reject_low_quality=True,
    skip_detector_benchmark=False,
    detector_binarization="otsu_deshadow",
):
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_out = resolve_csv_output_dir(output_dir, csv_output_dir)
    csv_out.mkdir(parents=True, exist_ok=True)

    files = sorted([file for file in inp.iterdir() if file.suffix.lower() in SUPPORTED_EXTS])
    if not files:
        print(f"[ERROR] Nicio imagine gasita in {inp}")
        return

    print(f"\n{'='*65}")
    print("  Document Scanner -- Pipeline Complet")
    print(f"  Input:  {inp}  ({len(files)} imagini)")
    print(f"  Output: {out}")
    print(f"  CSV:    {csv_out}")
    print(f"{'='*65}\n")

    print("  [PASUL 1/4] Scanare imagini...\n")
    results = []
    results_summary = []
    start_total = time.time()

    for idx, file in enumerate(files, 1):
        print(f"  [{idx:03d}/{len(files):03d}] {file.name}")
        t0 = time.time()
        result = scan_document(
            str(file),
            output_dir=str(out),
            save_debug=debug,
            detection_mode=detection,
            binarization_mode=binarization,
            save_all_binaries=save_all_binaries,
            save_stage_images=save_stages_separately,
            quality_threshold=quality_threshold,
            reject_low_quality=reject_low_quality,
        )
        elapsed = time.time() - t0

        if result.success and result.metrics:
            metrics = result.metrics
            quality_status = "accepted" if result.accepted else f"rejected [{result.rejection_reason}]"
            print(
                f"           detect={result.method:<10} "
                f"binary={result.binarization_method:<17} "
                f"sharp={metrics.sharpness:>7.1f} ({metrics.sharpness_label:<9}) "
                f"skew={metrics.skew_angle:>5.1f}deg ({metrics.skew_label:<8}) "
                f"bscore={metrics.binarization_score:>5.1f} "
                f"score={metrics.overall_score:>5.1f}/100 ({metrics.overall_label:<9}) "
                f"quality={quality_status} "
                f"[{elapsed:.2f}s]"
            )
            results.append(result)
            results_summary.append(result_to_summary_row(file.name, result))
        else:
            print(f"           [FAIL] {result.message}")

    total_scan = time.time() - start_total
    print(f"\n{'='*65}")
    print(f"  Procesate: {len(results)}/{len(files)} imagini in {total_scan:.1f}s")
    if results:
        scores = [result.metrics.overall_score for result in results]
        accepted_count = sum(1 for result in results if result.accepted)
        rejected_count = len(results) - accepted_count
        print(f"  Score mediu   : {np.mean(scores):.1f}/100")
        print(f"  Score maxim   : {np.max(scores):.1f}/100")
        print(f"  Score minim   : {np.min(scores):.1f}/100")
        print(f"  Accepted      : {accepted_count}")
        print(f"  Rejected      : {rejected_count}")
        counts = {}
        for result in results:
            label = result.metrics.overall_label
            counts[label] = counts.get(label, 0) + 1
        for label, count in sorted(counts.items()):
            print(f"  {label:<12}: {count} imagini")
        summary_path = write_summary(results_summary, str(csv_out))
        print(f"  Summary CSV   : {summary_path}")
    print(f"{'='*65}\n")

    if generate_plots:
        print("  [PASUL 2/4] Generez grafice metrici calitate...")
        generate_plots_main(output_dir=str(out), csv_input_dir=str(csv_out))
    else:
        print("  [PASUL 2/4] Graficele plot_*.png sunt sarite implicit.")
    print()

    if generate_grid:
        print("  [PASUL 3/4] Generez comparison grid pentru pipeline-uri...")
        try:
            from generate_comparison_grid import generate_grid

            generate_grid(input_dir, output_dir, max_images=min(len(files), 16))
        except Exception as error:
            print(f"    [WARN] Grid generation error: {error}")
    else:
        print("  [PASUL 3/4] comparison_grid.png este sarit implicit.")
    print()

    if not skip_compare:
        print("  [PASUL 4/4] Compar metodele de binarizare, detectorii si conditiile de test...")
        try:
            from compare_methods import run_comparison

            run_comparison(
                input_dir,
                output_dir,
                str(csv_out),
                run_detector_benchmark=not skip_detector_benchmark,
                quality_threshold=quality_threshold,
                reject_low_quality=reject_low_quality,
                detector_binarization_mode=detector_binarization,
            )
        except Exception as error:
            print(f"    [WARN] Comparison error: {error}")
    else:
        print("  [PASUL 4/4] Comparatie sarita (--skip-compare).")

    print(f"\n{'='*65}")
    print(f"  DONE -- imagini in: {out}/")
    print(f"  DONE -- CSV-uri in: {csv_out}/")
    print(f"{'='*65}")
    _print_output_summary(out, csv_out)


def _print_output_summary(out, csv_out=None):
    out = Path(out)
    csv_out = Path(csv_out or out)
    categories = {
        "Pipeline imagini": list(out.glob("*_pipeline.jpg")),
        "Rectified separat": list(out.glob("*_rectified.jpg")),
        "Binary separat": list(out.glob("*_binary.jpg")),
        "Variante binare": list(out.glob("*_binary_*.jpg")),
        "Metrici JSON": list(out.glob("*_metrics.json")),
        "Debug outlines": list(out.glob("*_debug.jpg")),
        "Grafice": list(out.glob("plot_*.png")),
        "Comparatii vizuale": list(out.glob("comparison_*.png")),
        "CSV-uri": list(csv_out.glob("*.csv")),
    }
    for label, files in categories.items():
        if files:
            print(f"  {label:<22}: {len(files)} fisiere")


def main():
    parser = argparse.ArgumentParser(description="Run full document scanner pipeline")
    parser.add_argument("--input", default="dataset", help="Folder imagini input")
    parser.add_argument("--output", default="output", help="Folder output")
    parser.add_argument("--debug", action="store_true", help="Salveaza imagini debug")
    parser.add_argument("--skip-compare", action="store_true", help="Sari comparatia metode")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Folder separat pentru fisierele CSV; implicit foloseste un folder sibling de forma <output>_csv",
    )
    parser.add_argument(
        "--detection",
        default="auto",
        choices=["auto", *DETECTION_METHODS],
        help="Fixeaza detectorul documentului sau lasa selectie automata",
    )
    parser.add_argument(
        "--binarization",
        default="auto",
        choices=["auto", *BINARIZATION_METHODS],
        help="Fixeaza metoda de binarizare sau lasa selectie automata",
    )
    parser.add_argument(
        "--save-all-binaries",
        action="store_true",
        help="Salveaza toate variantele de binarizare pentru fiecare imagine",
    )
    parser.add_argument(
        "--save-stages-separately",
        action="store_true",
        help="Salveaza si rectified/binary separat; implicit se salveaza doar imaginea compusa de pipeline",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Genereaza optional imaginile plot_*.png",
    )
    parser.add_argument(
        "--generate-grid",
        action="store_true",
        help="Genereaza optional comparison_grid.png",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=60.0,
        help="Pragul minim al scorului overall pentru a accepta un scan",
    )
    parser.add_argument(
        "--no-reject-low-quality",
        action="store_true",
        help="Nu marca scanurile slabe ca rejectate; pastraza doar scorurile",
    )
    parser.add_argument(
        "--skip-detector-benchmark",
        action="store_true",
        help="Nu mai ruleaza benchmark-ul comparativ pentru detectorii de document",
    )
    parser.add_argument(
        "--detector-binarization",
        default="otsu_deshadow",
        choices=["adaptive_gaussian", "otsu_deshadow", "sauvola_deshadow", "hybrid_deshadow"],
        help="Metoda de binarizare folosita fix in benchmark-ul detectorilor",
    )
    args = parser.parse_args()
    run_pipeline(
        args.input,
        args.output,
        args.debug,
        args.skip_compare,
        args.detection,
        args.binarization,
        args.save_all_binaries,
        args.save_stages_separately,
        args.generate_plots,
        args.generate_grid,
        args.csv_output,
        args.quality_threshold,
        not args.no_reject_low_quality,
        args.skip_detector_benchmark,
        args.detector_binarization,
    )


if __name__ == "__main__":
    main()
