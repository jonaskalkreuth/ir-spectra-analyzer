"""
Command-line entry point for the IR spectrum analyzer.

Examples
--------
# Analyze the bundled sample spectrum
python scripts/run_analysis.py

# Analyze a specific file
python scripts/run_analysis.py --input data/my_spectrum.csv

# Save results to JSON and produce a plot
python scripts/run_analysis.py --input data/my_spectrum.csv --output results.json --plot

# Input data is % Transmittance, skip baseline correction
python scripts/run_analysis.py --input data/trans.csv --transmittance --no-baseline

# Adjust peak detection sensitivity
python scripts/run_analysis.py --prominence 0.08 --min-height 0.03
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzer import analyze
from src.classifier import predict
from src.loader import load_spectrum
from src.preprocessing import convert_to_absorbance
from src.utils import print_results, save_results, plot_spectrum
from src import config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_analysis",
        description="Analyze an IR spectrum from a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        default=config.DEFAULT_DATA_PATH,
        metavar="PATH",
        help="Path to the CSV spectrum file.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Save results as JSON to this path.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Display (or save) a spectrum plot with annotated peaks.",
    )
    p.add_argument(
        "--plot-output",
        default=None,
        metavar="PATH",
        help="Save the plot to a file instead of showing interactively.",
    )
    p.add_argument(
        "--transmittance",
        action="store_true",
        help="Treat input intensity as %% Transmittance (converts to Absorbance).",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip ALS baseline correction.",
    )
    p.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip ML material classification (report only).",
    )
    p.add_argument(
        "--prominence",
        type=float,
        default=config.MIN_PROMINENCE,
        metavar="FLOAT",
        help="Minimum peak prominence (0–1 scale).",
    )
    p.add_argument(
        "--min-height",
        type=float,
        default=config.MIN_HEIGHT,
        metavar="FLOAT",
        help="Minimum peak intensity (0–1 scale).",
    )
    p.add_argument(
        "--min-distance",
        type=float,
        default=config.MIN_DISTANCE_CM,
        metavar="CM",
        help="Minimum separation between peaks in cm⁻¹.",
    )
    p.add_argument(
        "--intensity-filter",
        choices=["strong", "medium", "weak"],
        default=config.MIN_INTENSITY_FILTER,
        help="Minimum expected band intensity for functional group matches.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(name)s: %(message)s",
    )

    try:
        results = analyze(
            args.input,
            correct_baseline=not args.no_baseline,
            input_is_transmittance=args.transmittance,
            min_prominence=args.prominence,
            min_height=args.min_height,
            min_distance_cm=args.min_distance,
            min_intensity_filter=args.intensity_filter,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    if not args.no_classify:
        # Re-load the raw arrays for the ML path: analyze() consumes the file
        # internally and returns only the report dict, so predict() (which wants
        # raw arrays, not a path) needs its own read. One small CSV — negligible.
        df = load_spectrum(args.input)
        intensities = df["intensity"].values
        if args.transmittance:
            # predict() expects absorbance, same as analyze() converts to.
            intensities = convert_to_absorbance(intensities)
        try:
            results["prediction"] = predict(df["wavenumber"].values, intensities)
        except FileNotFoundError:
            # models/ is gitignored, so a fresh clone has no trained model yet.
            # Degrade to report-only with a hint, rather than crashing.
            results["prediction"] = None
            print(
                "  (No trained model found — see \"Reproduce from scratch\" "
                "in the README to enable classification.)\n"
            )
        except ValueError as exc:
            # e.g. a spectrum too short for the derivative window.
            results["prediction"] = None
            print(f"  (Classification skipped: {exc})\n")

    print_results(results)

    if args.output:
        save_results(results, args.output)
        print(f"Results saved to {args.output}")

    if args.plot:
        df = load_spectrum(args.input)
        plot_spectrum(
            df["wavenumber"].values,
            df["intensity"].values,
            results=results,
            title=f"IR Spectrum — {Path(args.input).name}",
            output_path=args.plot_output,
        )
        if args.plot_output:
            print(f"Plot saved to {args.plot_output}")


if __name__ == "__main__":
    main()
