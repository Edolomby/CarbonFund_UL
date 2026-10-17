#!/usr/bin/env python3
"""Summarise and plot scenario-level hedging costs exported by CarbonFund_UL.

Usage: python tools/analyze_hedging.py hist_general_CIR_mc.csv
Requires NumPy and Matplotlib. Larger costs are treated as worse outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


STRATEGIES = ("no_hedge", "static_hedge", "dynamic_hedge")
LABELS = ("No hedge", "Static hedge", "Dynamic hedge")
COLORS = ("#4477AA", "#EE7733", "#228833")
ALIASES = {
    "nohedge": "no_hedge",
    "statichedge": "static_hedge",
    "dynamichedge": "dynamic_hedge",
    "continuoushedge": "dynamic_hedge",
    "hedgingcost": "dynamic_hedge",
}
NORMAL_975 = NormalDist().inv_cdf(0.975)


def normalise_header(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def read_costs(path: Path) -> tuple[np.ndarray, list[str]]:
    """Retain CSV row alignment; blank/NaN cells remain missing, not zero."""
    columns: dict[str, str] = {}
    values: list[list[float]] = []
    ids: set[str] = set()
    warnings: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"{path}: missing CSV header")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate column names")
        id_column = None
        for name in headers:
            key = normalise_header(name)
            strategy = ALIASES.get(key)
            if strategy is not None:
                if strategy in columns:
                    raise ValueError(f"{path}: ambiguous columns for {strategy}")
                columns[strategy] = name
            if key == "scenarioid":
                if id_column is not None:
                    raise ValueError(f"{path}: ambiguous scenario_id columns")
                id_column = name
        missing = set(STRATEGIES) - columns.keys()
        if missing:
            raise ValueError(f"{path}: missing cost columns: {', '.join(sorted(missing))}")
        for row in reader:
            if None in row:
                raise ValueError(f"{path}, line {reader.line_num}: too many CSV fields")
            if all(not (value or "").strip() for value in row.values()):
                continue
            if id_column is not None:
                identifier = (row.get(id_column) or "").strip()
                if not identifier or identifier in ids:
                    raise ValueError(f"{path}, line {reader.line_num}: missing/duplicate scenario_id")
                ids.add(identifier)
            current = []
            for strategy in STRATEGIES:
                cell = (row.get(columns[strategy]) or "").strip()
                try:
                    number = float(cell) if cell else math.nan
                except ValueError as error:
                    raise ValueError(
                        f"{path}, line {reader.line_num}, {strategy}: invalid number {cell!r}"
                    ) from error
                if math.isinf(number):
                    raise ValueError(f"{path}, line {reader.line_num}: infinite cost")
                current.append(number)
            values.append(current)
    if not values:
        raise ValueError(f"{path}: no scenario rows")
    data = np.asarray(values, dtype=float)
    for i, strategy in enumerate(STRATEGIES):
        count = int(np.isnan(data[:, i]).sum())
        if count == len(data):
            raise ValueError(f"{path}: {strategy} has no finite observations")
        if count:
            warnings.append(f"{strategy}: {count} missing cells; paired comparisons keep row alignment")
    return data, warnings


def mean_statistics(x: np.ndarray) -> dict[str, float | int]:
    n = len(x)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1)) if n > 1 else math.nan
    deviation = math.sqrt(variance)
    se = deviation / math.sqrt(n)
    return {
        "n": n, "mean": mean, "variance": variance, "std": deviation,
        "mean_se": se, "mean_ci95_low": mean - NORMAL_975 * se,
        "mean_ci95_high": mean + NORMAL_975 * se,
    }


def expected_shortfall(x: np.ndarray, level: float) -> float:
    """Mean of the worst (1-level) fraction, with fractional endpoint weight."""
    ordered = np.sort(x)
    cutoff = level * len(ordered)
    index = int(math.floor(cutoff))
    total = float(np.sum(ordered[index + 1:]))
    total += (index + 1 - cutoff) * float(ordered[index])
    return total / (len(ordered) - cutoff)


def summarise(data: np.ndarray) -> list[dict]:
    result = []
    for i, strategy in enumerate(STRATEGIES):
        x = data[np.isfinite(data[:, i]), i]
        q = np.quantile(x, [0.025, 0.5, 0.90, 0.95, 0.975, 0.99], method="linear")
        result.append({
            "strategy": strategy, **mean_statistics(x),
            "min": float(np.min(x)), "q025": float(q[0]), "median": float(q[1]),
            "q90": float(q[2]), "q95": float(q[3]), "q975": float(q[4]),
            "q99": float(q[5]), "max": float(np.max(x)),
            "probability_positive": float(np.mean(x > 0)),
            "es95": expected_shortfall(x, 0.95),
            "es99": expected_shortfall(x, 0.99),
        })
    return result


def paired_comparisons(data: np.ndarray) -> list[dict]:
    result = []
    for reference, candidate in ((0, 1), (0, 2), (1, 2)):
        keep = np.isfinite(data[:, reference]) & np.isfinite(data[:, candidate])
        a, b = data[keep, reference], data[keep, candidate]
        if not len(a):
            continue
        stats = mean_statistics(b - a)
        ratio = correlation = math.nan
        if len(a) > 1:
            va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
            if va > 0:
                ratio = vb / va
            if va > 0 and vb > 0:
                correlation = float(np.corrcoef(a, b)[0, 1])
        result.append({
            "reference": STRATEGIES[reference], "candidate": STRATEGIES[candidate],
            **stats, "correlation": correlation,
            "variance_ratio_candidate_to_reference": ratio,
            "variance_reduction_percent": 100 * (1 - ratio),
        })
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in row.items()
            })


def histogram_density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Dividing by the FULL sample count preserves the probability mass when
    # the inset excludes observations outside its displayed interval.
    counts, _ = np.histogram(values, bins=edges)
    return counts / (len(values) * np.diff(edges))


def make_plots(data: np.ndarray, destination: Path, title: str,
               bins: int, zoom: list[float] | None, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [data[np.isfinite(data[:, i]), i] for i in range(3)]
    edges = np.histogram_bin_edges(np.concatenate(series), bins=bins)
    if zoom is None:
        low, high = np.quantile(np.concatenate(series[1:]), [0.01, 0.99])
        if low == high:
            half_width = max(1.0, abs(float(low))) * 0.05
            low, high = low - half_width, high + half_width
    else:
        low, high = zoom
    zoom_edges = np.linspace(low, high, bins + 1)
    xlabel = "Hedging cost"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.84, wspace=0.30)
    inset = axes[0].inset_axes([0.61, 0.50, 0.36, 0.45])
    for x, label, color in zip(series, LABELS, COLORS):
        axes[0].stairs(histogram_density(x, edges), edges, label=label, color=color)
        inset.stairs(histogram_density(x, zoom_edges), zoom_edges, color=color)
        ordered = np.sort(x)
        axes[1].step(np.r_[ordered[0], ordered],
                     np.r_[0.0, np.arange(1, len(x) + 1) / len(x)],
                     where="post", label=label, color=color)
    axes[0].set(title="Density and central zoom", xlabel=xlabel, ylabel="Density")
    axes[0].legend(loc="upper left", fontsize=8)
    inset.set_xlim(low, high)
    inset.set_ylim(bottom=0)
    inset.tick_params(labelsize=7)
    inset.set_title("Zoom", fontsize=8)
    axes[1].set(title="Empirical CDF", xlabel=xlabel, ylabel="Probability")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    boxes = axes[2].boxplot(series, showfliers=False, patch_artist=True)
    for box, color in zip(boxes["boxes"], COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.5)
    axes[2].set_xticks([1, 2, 3], LABELS, rotation=20)
    axes[2].set(title="Box plots (outliers hidden)", ylabel=xlabel)
    for ax in (*axes, inset):
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=14)
    fig.savefig(destination / "distributions.png", dpi=180)
    fig.savefig(destination / "distributions.pdf")
    if not show:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("hedging_analysis"))
    parser.add_argument("--bins", type=int, default=70)
    parser.add_argument("--zoom", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--divide-by", type=float, default=1.0,
                        help="Divide every cost by this positive value before analysis")
    parser.add_argument("--title", help="Custom plot title (one input file only)")
    parser.add_argument("--show", action="store_true", help="Also open Matplotlib windows")
    args = parser.parse_args()
    if args.bins < 1 or not math.isfinite(args.divide_by) or args.divide_by <= 0:
        parser.error("bins must be positive and divide-by positive and finite")
    if args.zoom and (not all(map(math.isfinite, args.zoom)) or args.zoom[0] >= args.zoom[1]):
        parser.error("zoom requires finite MIN < MAX")
    if args.title and len(args.csv_files) != 1:
        parser.error("title is supported for one input file at a time")
    stems = [path.stem for path in args.csv_files]
    if len(stems) != len(set(stems)):
        parser.error("input filenames must have distinct stems to avoid overwriting results")
    try:
        for source in args.csv_files:
            data, warnings = read_costs(source)
            data /= args.divide_by
            if np.isinf(data).any():
                raise ValueError("Cost scaling overflowed; choose a larger divide-by value")
            summary = summarise(data)
            paired = paired_comparisons(data)
            out = args.output / source.stem
            out.mkdir(parents=True, exist_ok=True)
            write_rows(out / "summary.csv", summary)
            write_rows(out / "paired_comparisons.csv", paired)
            metadata = {
                "source": str(source.resolve()), "rows": len(data),
                "divide_by": args.divide_by, "quantile_method": "linear",
                "cost_basis": "as stored; no discounting or rebasing performed",
                "mean_ci": "95% normal CLT approximation; independent outer scenarios assumed",
                "paired_difference": "candidate minus reference, matching CSV rows",
                "expected_shortfall": "mean of worst 5%/1%, with fractional endpoint weight",
                "numpy_version": np.__version__, "warnings": warnings,
            }
            (out / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
            title = args.title or source.stem
            if args.divide_by != 1:
                title += f" (costs divided by {args.divide_by:g})"
            make_plots(data, out, title, args.bins, args.zoom, args.show)
            print(f"\n{source}: {len(data)} scenario rows")
            print(f"{'Strategy':<16} {'N':>9} {'Mean':>13} {'Std':>13} {'Mean SE':>13} {'q95':>13}")
            for row in summary:
                print(f"{row['strategy']:<16} {row['n']:>9d} {row['mean']:>13.6g} "
                      f"{row['std']:>13.6g} {row['mean_se']:>13.6g} {row['q95']:>13.6g}")
            for warning in warnings:
                print(f"Note: {warning}")
            print(f"Saved results: {out}")
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Analysis error: {error}\n")


if __name__ == "__main__":
    main()
