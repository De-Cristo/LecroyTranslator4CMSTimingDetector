#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone walk-fit refit from saved CSV data"
    )
    p.add_argument("input_csv", help="Input event-level walk-fit CSV with columns invE,delta_t")
    p.add_argument("--output-prefix", default=None,
                   help="Output prefix for plots and CSVs (default: derived from input path)")
    p.add_argument("--nbins", type=int, default=12,
                   help="Number of horizontal bins in 1/E for occupancy and profile building (default: 12)")
    p.add_argument("--min-entries", type=int, default=5,
                   help="Minimum entries per bin to keep that bin in the fit (default: 5)")
    p.add_argument("--poly-order", type=int, default=2,
                   help="Polynomial order for the fit (default: 2)")
    return p.parse_args()


def _default_output_prefix(input_csv):
    path = Path(input_csv)
    stem = path.stem
    if stem.endswith("_data"):
        stem = stem[:-5]
    return str(path.with_name(f"{stem}_refit"))


def _load_walk_fit_csv(path):
    x = []
    y = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["invE"]))
            y.append(float(row["delta_t"]))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _bin_edges(x, nbins):
    x = np.asarray(x, dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        x_max = x_min + 1e-12
    return np.linspace(x_min, x_max, nbins + 1)


def _bin_occupancy_mask(x, nbins=12, min_entries=5):
    x = np.asarray(x, dtype=float)
    edges = _bin_edges(x, nbins)
    bin_idx = np.searchsorted(edges, x, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)
    counts = np.bincount(bin_idx, minlength=nbins)
    keep = counts[bin_idx] >= min_entries
    return keep, edges, counts


def _profile_points_from_bins(x, y, nbins=12, min_entries=5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    keep, edges, counts = _bin_occupancy_mask(x, nbins=nbins, min_entries=min_entries)
    bin_idx = np.searchsorted(edges, x, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    x_prof = []
    y_prof = []
    y_err = []
    kept_counts = []
    for i in range(nbins):
        if counts[i] < min_entries:
            continue
        in_bin = bin_idx == i
        x_bin = x[in_bin]
        y_bin = y[in_bin]
        n_bin = len(x_bin)
        x_prof.append(float(np.mean(x_bin)))
        y_prof.append(float(np.mean(y_bin)))
        kept_counts.append(n_bin)
        if n_bin >= 2:
            y_err.append(float(np.std(y_bin, ddof=1) / np.sqrt(n_bin)))
        else:
            y_err.append(0.0)

    return (
        np.asarray(x_prof, dtype=float),
        np.asarray(y_prof, dtype=float),
        np.asarray(y_err, dtype=float),
        np.asarray(kept_counts, dtype=int),
    )


def _mad_fit_mask(y):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.zeros(0, dtype=bool)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad <= 0:
        return np.ones(len(y), dtype=bool)
    cut = 5 * 1.4826 * mad
    return np.abs(y - med) < cut


def _write_fit_coeffs_csv(prefix, p_event, p_binned):
    with open(f"{prefix}_refit_coeffs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fit_kind", "p2", "p1", "p0"])
        writer.writerow(["event", *np.asarray(p_event, dtype=float).tolist()])
        writer.writerow(["binned", *np.asarray(p_binned, dtype=float).tolist()])


def _write_profile_csv(prefix, x_prof, y_prof, y_err, counts):
    with open(f"{prefix}_refit_profile.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["invE_mean", "delta_t_mean", "delta_t_sem", "count"])
        for row in zip(x_prof, y_prof, y_err, counts):
            writer.writerow(row)


def _parabola_label(coeffs):
    p2, p1, p0 = np.asarray(coeffs, dtype=float)
    return f"y = {p2:.2e}x^2 + {p1:.2e}x + {p0:.2e}"


def _plot_scatter_refit(prefix, x, y, fit_mask, p_event, poly_order):
    fig, ax = plt.subplots(figsize=(8, 6))
    excluded = ~fit_mask
    if np.any(excluded):
        ax.scatter(x[excluded], y[excluded], s=6, alpha=0.25, color="lightgray", label=f"Excluded ({excluded.sum()})")
    ax.scatter(x[fit_mask], y[fit_mask], s=6, alpha=0.35, color="steelblue", label=f"Fit sample ({fit_mask.sum()})")
    x_line = np.linspace(float(np.min(x[fit_mask])), float(np.max(x[fit_mask])), 200)
    y_line = np.polyval(p_event, x_line)
    ax.plot(x_line, y_line, color="red", linewidth=2, label=f"Poly{poly_order} fit")
    ax.set_xlabel("1 / Energy")
    ax.set_ylabel("Delta t (ps)")
    ax.set_title("Walk fit from CSV (scatter)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_refit_scatter.png", dpi=150)
    plt.close(fig)


def _plot_binned_refit(prefix, x_prof, y_prof, y_err, p_binned, poly_order):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        x_prof, y_prof, yerr=y_err,
        fmt="o", color="steelblue", ecolor="steelblue",
        elinewidth=1, capsize=3, label=f"Binned profile ({len(x_prof)} bins)"
    )
    x_line = np.linspace(float(np.min(x_prof)), float(np.max(x_prof)), 200)
    y_line = np.polyval(p_binned, x_line)
    ax.plot(x_line, y_line, color="red", linewidth=2, label=f"Poly{poly_order} fit")
    ax.set_xlabel("1 / Energy")
    ax.set_ylabel("Delta t (ps)")
    ax.set_title("Walk fit from CSV (binned)")
    ax.text(
        0.03, 0.97, _parabola_label(p_binned),
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray")
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_refit_binned.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    output_prefix = args.output_prefix or _default_output_prefix(args.input_csv)

    x, y = _load_walk_fit_csv(args.input_csv)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < max(3, args.poly_order + 1):
        raise SystemExit("Not enough valid points to fit.")

    occ_mask, _edges, _counts = _bin_occupancy_mask(x, nbins=args.nbins, min_entries=args.min_entries)
    mad_mask = _mad_fit_mask(y)
    fit_mask = occ_mask & mad_mask
    if int(np.sum(fit_mask)) < max(3, args.poly_order + 1):
        raise SystemExit("Not enough retained event-level points after occupancy/MAD selection.")

    p_event = np.polyfit(x[fit_mask], y[fit_mask], args.poly_order)

    x_prof, y_prof, y_err, counts = _profile_points_from_bins(
        x, y, nbins=args.nbins, min_entries=args.min_entries
    )
    if len(x_prof) < max(3, args.poly_order + 1):
        raise SystemExit("Not enough retained profile bins to fit.")
    p_binned = np.polyfit(x_prof, y_prof, args.poly_order)

    _plot_scatter_refit(output_prefix, x, y, fit_mask, p_event, args.poly_order)
    _plot_binned_refit(output_prefix, x_prof, y_prof, y_err, p_binned, args.poly_order)
    _write_fit_coeffs_csv(output_prefix, p_event, p_binned)
    _write_profile_csv(output_prefix, x_prof, y_prof, y_err, counts)


if __name__ == "__main__":
    main()
