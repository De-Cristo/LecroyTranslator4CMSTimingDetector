import os

import matplotlib.pyplot as plt
import numpy as np

from clock_study_calc import filter_range


def save_detected_edges_plot(t_rel, a, edges, out_png, evt, zero_line):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(t_rel, a, label="waveform")
    ax.axhline(zero_line, color="gray", linestyle="--", label="zero_line")
    for t_edge, _, _ in edges:
        ax.axvline(t_edge, color="red", linestyle="--")
        ax.scatter([t_edge], [np.interp(t_edge, t_rel, a)], color="red")
    ax.set_title(f"Event {evt} clock edges (count={len(edges)})")
    ax.set_xlabel("Time (ns, rel)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def save_linear_fit_plot(
    edge_indices,
    edge_times,
    fit_vals,
    out_png,
    title,
    point_label,
    fit_label=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(edge_indices, edge_times, color="C0", label=point_label)
    if fit_label is not None and len(fit_vals) > 0 and np.any(~np.isnan(fit_vals)):
        ax.plot(edge_indices, fit_vals, color="C1", label=fit_label)
    ax.set_xlabel("Edge index n_j")
    ax.set_ylabel("Time t_j (ns)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def save_template_artifact(template, t_axis, plots_dir, out_dir, base, polarity, save_array):
    out_png = os.path.join(plots_dir, f"clock_template_{polarity}_{base}.png")
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(t_axis, template, lw=1)
    ax.set_xlabel("Time (ns, rel to edge)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"Template ({polarity}) for {base}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    if save_array:
        np.save(os.path.join(out_dir, f"clock_template_{polarity}_{base}.npy"), template)
        print(f"[ok] Saved {polarity} template png and numpy: {out_png}")
    else:
        print(f"[ok] Saved {polarity} template png: {out_png}")


def save_template_overlay_plot(
    t_axis,
    template,
    snippet_norm,
    shift_ns,
    out_png,
    evt,
    edge_idx,
    edge_type,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(t_axis, template, label=f"template ({edge_type})", alpha=0.8)
    ax.plot(t_axis - shift_ns, snippet_norm, label="snippet (aligned)", alpha=0.9)
    ax.axvline(0.0, color="gray", linestyle="--", label="edge reference")
    ax.set_xlabel("Time (ns, rel to template reference)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"Event {evt} edge {edge_idx} template alignment")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def save_high_jitter_plot(
    t_rel,
    a,
    precise_times,
    edge_indices_used,
    precise_times_used,
    fit_vals,
    slope,
    sigma_t0,
    out_png,
    evt,
):
    fig, (ax_wave, ax_fit_plot) = plt.subplots(1, 2, figsize=(12, 4))
    ax_wave.plot(t_rel, a, label="waveform")
    edge_amp = np.interp(precise_times, t_rel, a, left=np.nan, right=np.nan)
    ax_wave.scatter(precise_times, edge_amp, color="red", zorder=5, label="precise edges")
    for tt in precise_times:
        ax_wave.axvline(tt, color="red", alpha=0.3)
    ax_wave.set_xlabel("Time (ns, rel)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(f"Event {evt} waveform (σ_t0={sigma_t0 * 1e3:.1f} ps)")
    ax_wave.legend(loc="best")
    ax_wave.grid(True, alpha=0.3)

    ax_fit_plot.scatter(edge_indices_used, precise_times_used, color="C0", label="precise times (fit)")
    if len(fit_vals) > 0 and np.any(~np.isnan(fit_vals)):
        ax_fit_plot.plot(edge_indices_used, fit_vals, color="C1", label=f"fit Tclk={slope:.4f} ns")
    ax_fit_plot.set_xlabel("Edge index n_j")
    ax_fit_plot.set_ylabel("Time t_j (ns)")
    ax_fit_plot.set_title("Linear fit diagnostics")
    ax_fit_plot.grid(True, alpha=0.3)
    ax_fit_plot.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def save_simple_histogram(values, out_png, title, xlabel, bins=100, figsize=(6, 4)):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return False

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(values, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def save_histogram_with_gaussian(
    values,
    out_png,
    title,
    xlabel,
    color,
    bins=100,
    xlim=None,
    legend_template="Gaussian μ={mean:.3f} σ={std:.3f}",
    alpha=0.75,
    figsize=(6, 4),
):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _, bin_edges, _ = ax.hist(values, bins=bins, color=color, alpha=alpha, label="data")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float(np.std(values))

    if std > 0 and np.isfinite(std):
        if xlim is None:
            plot_min = bin_edges[0]
            plot_max = bin_edges[-1]
        else:
            plot_min, plot_max = xlim
        x = np.linspace(plot_min, plot_max, 400)
        bin_width = bin_edges[1] - bin_edges[0]
        scale = len(values) * bin_width
        gauss = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax.plot(
            x,
            scale * gauss,
            color="black" if color != "black" else "red",
            lw=2,
            label=legend_template.format(mean=mean, std=std),
        )
    else:
        ax.axvline(mean, color="black", lw=2, label=legend_template.format(mean=mean, std=std))

    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return {"count": len(values), "mean": mean, "std": std}


def save_interval_histograms(
    intervals,
    plots_dir,
    out_dir,
    base,
    hist_prefix,
    stats_prefix,
    color,
    title_prefix,
    save_stats=False,
    zoom_range=(6.0, 6.5),
):
    intervals = np.asarray(intervals, dtype=float)
    intervals = intervals[~np.isnan(intervals)]
    if len(intervals) == 0:
        return None

    out_hist = os.path.join(plots_dir, f"{hist_prefix}_{base}.png")
    stats = save_histogram_with_gaussian(
        intervals,
        out_hist,
        f"{title_prefix} — {base} (N={len(intervals)})",
        "Inter-edge interval (ns)",
        color=color,
        bins=100,
        legend_template="Gaussian μ={mean:.3f} ns σ={std:.3f} ns",
    )

    xmin, xmax = zoom_range
    in_range = filter_range(intervals, xmin, xmax)
    zoom_stats = None
    if len(in_range) > 0:
        out_zoom = os.path.join(plots_dir, f"{hist_prefix}_zoom_{base}.png")
        zoom_stats = save_histogram_with_gaussian(
            in_range,
            out_zoom,
            f"{title_prefix} zoom {xmin}-{xmax} ns — {base} (N={len(in_range)})",
            "Inter-edge interval (ns)",
            color=color,
            bins=50,
            xlim=(xmin, xmax),
            alpha=0.8,
            legend_template="Gaussian μ={mean:.6f} ns σ={std:.6f} ns",
        )
        print(f"[ok] Saved zoomed histogram: {out_zoom}")
        if save_stats and zoom_stats is not None:
            zoom_stats_path = os.path.join(out_dir, f"{stats_prefix}_zoom_{base}.csv")
            with open(zoom_stats_path, "w") as handle:
                handle.write("n_intervals_zoom,mean_ns_zoom,std_ns_zoom\n")
                handle.write(
                    f"{zoom_stats['count']},{zoom_stats['mean']:.12g},{zoom_stats['std']:.12g}\n"
                )
            print(f"[ok] Saved zoom stats: {zoom_stats_path}")
    else:
        print(f"[warn] No inter-edge intervals in {xmin}-{xmax} ns for {base}")

    if save_stats and stats is not None:
        stats_path = os.path.join(out_dir, f"{stats_prefix}_{base}.csv")
        with open(stats_path, "w") as handle:
            handle.write("n_intervals,mean_ns,std_ns\n")
            handle.write(f"{stats['count']},{stats['mean']:.9g},{stats['std']:.9g}\n")

    return {"full": stats, "zoom": zoom_stats}


def save_jitter_comparison_histogram(
    sigma_single_ps,
    sigma_t0_ps,
    sigma_t_ave_ps,
    out_png,
    title,
):
    sigma_single_zoom = filter_range(sigma_single_ps, 0.0, 100.0)
    sigma_t0_zoom = filter_range(sigma_t0_ps, 0.0, 100.0)
    sigma_t_ave_zoom = filter_range(sigma_t_ave_ps, 0.0, 100.0)

    if (
        len(sigma_single_zoom) == 0
        and len(sigma_t0_zoom) == 0
        and len(sigma_t_ave_zoom) == 0
    ):
        return False

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bins = np.linspace(0.0, 100.0, 100)
    if len(sigma_single_zoom) > 0:
        mu_single = float(np.mean(sigma_single_zoom))
        ax.hist(
            sigma_single_zoom,
            bins=bins,
            color="C0",
            alpha=0.5,
            label=f"Single edge $\\sigma_{{single}}$ (mean={mu_single:.1f} ps)",
        )
    if len(sigma_t0_zoom) > 0:
        mu_t0 = float(np.mean(sigma_t0_zoom))
        ax.hist(
            sigma_t0_zoom,
            bins=bins,
            color="C3",
            alpha=0.7,
            label=f"Fit $\\sigma_{{t0}}$ (mean={mu_t0:.1f} ps)",
        )
    if len(sigma_t_ave_zoom) > 0:
        mu_t_ave = float(np.mean(sigma_t_ave_zoom))
        ax.hist(
            sigma_t_ave_zoom,
            bins=bins,
            color="C2",
            alpha=0.7,
            label=f"Fit $\\sigma_{{t\\_ave}}$ (mean={mu_t_ave:.1f} ps)",
        )
    ax.set_xlabel("Jitter σ (ps)")
    ax.set_ylabel("Counts")
    ax.set_xlim(0.0, 100.0)
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def save_zero_summary_outputs(df_zero, df_zero_fit, out_dir, save_csv_details=False):
    if not df_zero.empty:
        if save_csv_details:
            out_csv_zero = os.path.join(out_dir, "clock_edges_zero_cross.csv")
            df_zero.to_csv(out_csv_zero, index=False, float_format="%.9g")
            print(f"[ok] Wrote summary CSV (zero-cross): {out_csv_zero} (rows={len(df_zero)})")

        out_hist = os.path.join(out_dir, "clock_edge_hist_zero_cross.png")
        if save_simple_histogram(
            df_zero["edge_time_ns_rel"].dropna().values,
            out_hist,
            "Clock edge times (relative)",
            "Edge time (ns, rel)",
            bins=100,
        ):
            print("[ok] Saved histogram of zero-cross edges")
        else:
            print("[warn] No detected edges to plot")

    if df_zero_fit.empty:
        return

    if save_csv_details:
        out_csv_zero_fit = os.path.join(out_dir, "clock_zero_fit_results.csv")
        df_zero_fit.to_csv(out_csv_zero_fit, index=False, float_format="%.9g")
        print(f"[ok] Wrote zero-cross fit CSV: {out_csv_zero_fit} (rows={len(df_zero_fit)})")

    sigma_vals = df_zero_fit["sigma_t0_ns"].dropna().to_numpy()
    sigma_zoom = filter_range(sigma_vals * 1e3, 0.0, 50.0)
    if len(sigma_zoom) > 0:
        stats = save_histogram_with_gaussian(
            sigma_zoom,
            os.path.join(out_dir, "clock_zero_fit_sigma_t0_hist.png"),
            f"Zero-cross event jitter (σ_t0) 0-50 ps — entries={len(sigma_zoom)}",
            "Event t0 jitter σ (ps)",
            color="C3",
            bins=60,
            xlim=(0.0, 50.0),
            legend_template="Gaussian μ={mean:.3f} ps σ={std:.3f} ps",
        )
        if stats is not None:
            print(
                f"[ok] Saved zero-cross t0 jitter histogram. Mean={stats['mean']:.3f} ps, "
                f"σ={stats['std']:.3f} ps, entries={stats['count']}"
            )
    else:
        print("[warn] No sigma_t0 entries in 0-50 ps range for zero-cross histogram")

    sigma_single_vals = df_zero_fit["sigma_single_edge_ns"].dropna().to_numpy()
    sigma_t_ave_vals = df_zero_fit["sigma_t_ave_ns"].dropna().to_numpy()
    if len(sigma_single_vals) > 0 and len(sigma_vals) > 0:
        if save_jitter_comparison_histogram(
            sigma_single_vals * 1e3,
            sigma_vals * 1e3,
            sigma_t_ave_vals * 1e3,
            os.path.join(out_dir, "clock_zero_fit_jitter_comparison.png"),
            "Comparison: Zero-cross single-edge precision vs Multi-edge fit jitter",
        ):
            print("[ok] Saved zero-cross jitter improvement comparison histogram.")


def save_template_summary_outputs(
    df_template_edges,
    df_template_fit,
    out_dir,
    save_csv_details=False,
):
    if not df_template_edges.empty and save_csv_details:
        out_precise_csv = os.path.join(out_dir, "clock_edges_template_precise.csv")
        df_template_edges.to_csv(out_precise_csv, index=False, float_format="%.9g")
        print(
            f"[ok] Wrote precise edge CSV (template): {out_precise_csv} "
            f"(rows={len(df_template_edges)})"
        )

    if df_template_fit.empty:
        return

    if save_csv_details:
        out_fit_csv = os.path.join(out_dir, "clock_template_fit_results.csv")
        df_template_fit.to_csv(out_fit_csv, index=False, float_format="%.9g")
        print(f"[ok] Wrote template fit CSV: {out_fit_csv} (rows={len(df_template_fit)})")

    tclk_vals = df_template_fit["tclk_ns"].dropna().to_numpy()
    tclk_zoom = filter_range(tclk_vals, 6.1, 6.4)
    if len(tclk_zoom) > 0:
        stats = save_histogram_with_gaussian(
            tclk_zoom,
            os.path.join(out_dir, "clock_template_fit_tclk_hist.png"),
            f"Template fit clock period (6.1-6.4 ns) — entries={len(tclk_zoom)}",
            "Fitted clock period Tclk (ns)",
            color="C1",
            bins=60,
            xlim=(6.1, 6.4),
            legend_template="Gaussian μ={mean:.6f} ns σ={std:.6f} ns",
        )
        if stats is not None:
            print(
                f"[ok] Saved clock period histogram (template fit). Mean={stats['mean']:.6f} ns, "
                f"σ={stats['std']:.6f} ns, entries={stats['count']}"
            )
    else:
        print("[warn] No Tclk entries in 6.1-6.4 ns range for histogram")

    sigma_vals = df_template_fit["sigma_t0_ns"].dropna().to_numpy()
    sigma_zoom = filter_range(sigma_vals * 1e3, 0.0, 50.0)
    if len(sigma_zoom) > 0:
        stats = save_histogram_with_gaussian(
            sigma_zoom,
            os.path.join(out_dir, "clock_template_fit_sigma_t0_hist.png"),
            f"Template fit event jitter (σ_t0) 0-50 ps — entries={len(sigma_zoom)}",
            "Event t0 jitter σ (ps)",
            color="C3",
            bins=60,
            xlim=(0.0, 50.0),
            legend_template="Gaussian μ={mean:.3f} ps σ={std:.3f} ps",
        )
        if stats is not None:
            print(
                f"[ok] Saved t0 jitter histogram (template fit). Mean={stats['mean']:.3f} ps, "
                f"σ={stats['std']:.3f} ps, entries={stats['count']}"
            )
    else:
        print("[warn] No sigma_t0 entries in 0-50 ps range for histogram")

    sigma_single_vals = df_template_fit["sigma_single_edge_ns"].dropna().to_numpy()
    sigma_t_ave_vals = df_template_fit["sigma_t_ave_ns"].dropna().to_numpy()
    if len(sigma_single_vals) > 0 and len(sigma_vals) > 0:
        if save_jitter_comparison_histogram(
            sigma_single_vals * 1e3,
            sigma_vals * 1e3,
            sigma_t_ave_vals * 1e3,
            os.path.join(out_dir, "clock_template_fit_jitter_comparison.png"),
            "Comparison: Single-edge precision vs Multi-edge fit jitter",
        ):
            print("[ok] Saved jitter improvement comparison histogram.")
