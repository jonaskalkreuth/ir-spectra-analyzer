"""
Output formatting, export, and plotting utilities.
"""

import json
from pathlib import Path


# ── Console output ────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    """Print a human-readable analysis summary to stdout."""
    sep = "-" * 56
    print(sep)
    print(f"  IR Analysis Report")
    print(f"  Source : {results.get('source', 'unknown')}")
    print(sep)

    print(f"\n  Peaks detected: {results.get('num_peaks', 0)}\n")

    peaks = results.get("peaks", [])
    if peaks:
        print(f"  {'Wavenumber':>12}  {'Intensity':>10}  {'Prominence':>10}  "
              f"{'FWHM (cm-1)':>11}  Region")
        print("  " + "-" * 54)
        for p in peaks:
            print(
                f"  {p['wavenumber']:>10.1f}  "
                f"{p['intensity']:>10.4f}  "
                f"{p['prominence']:>10.4f}  "
                f"{p['fwhm_cm']:>11.1f}  "
                f"{p['region']}"
            )

    groups = results.get("groups", [])
    if groups:
        print(f"\n  Possible functional-group assignments (window match only) ({len(groups)} matches):\n")
        seen = set()
        for g in groups:
            key = (g["group"], round(g["wavenumber"]))
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  {g['wavenumber']:>8.1f} cm-1  "
                f"{g['group']:<40}  "
                f"[{g['intensity']}, {g['shape']}]"
            )
            print(f"           -> {g['description']}")

    classes = results.get("compound_classes", [])
    if classes:
        print(f"\n  Likely compound class(es): {', '.join(classes)}")

    prediction = results.get("prediction")
    if prediction:
        print(
            f"\n  Predicted material: {prediction['predicted_class']}  "
            f"({prediction['confidence']:.1%} confidence)"
        )
        # Show the other classes so model uncertainty is visible — a near-tie
        # (e.g. 51% vs 47%) reads very differently from a confident 94%.
        # The predicted class may not be the most probable, so filter by name.
        others = [
            o for o in prediction["top_k"]
            if o["class"] != prediction["predicted_class"]
        ]
        if others:
            ranked = ", ".join(
                f"{o['class']} {o['probability']:.1%}" for o in others
            )
            print(f"  Other classes: {ranked}")

    print(f"\n{sep}\n")


# ── File export ───────────────────────────────────────────────────────────────

def save_results(results: dict, path: str = "results.json") -> None:
    """Save analysis results as a formatted JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spectrum(
    wavenumbers,
    intensity,
    results: dict = None,
    title: str = "IR Spectrum",
    output_path: str = None,
) -> None:
    """
    Plot the IR spectrum with detected peaks annotated.

    Parameters
    ----------
    wavenumbers : array-like
    intensity   : array-like (normalized absorbance)
    results     : output dict from analyzer.analyze(); peaks are overlaid if given
    title       : plot title
    output_path : if given, save to file instead of showing interactively
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(wavenumbers, intensity, color="steelblue", linewidth=1.0, label="Spectrum")

    if results:
        for p in results.get("peaks", []):
            ax.axvline(p["wavenumber"], color="tomato", linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"{p['wavenumber']:.0f}",
                xy=(p["wavenumber"], p["intensity"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="tomato",
                rotation=90,
            )

    # Conventional IR: wavenumber decreases left-to-right
    ax.set_xlim(max(wavenumbers), min(wavenumbers))
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Normalized Absorbance", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.2)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
