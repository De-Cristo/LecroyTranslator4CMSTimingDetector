#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


def split_by_time_gaps(values, gap_factor=100):
    """
    Split 1D time values into clusters based on large gaps.
    """
    values = np.asarray(values)
    values = np.sort(values)

    gaps = np.diff(values)
    threshold = np.median(gaps) * gap_factor

    split_idxs = np.where(gaps > threshold)[0]
    clusters = np.split(values, split_idxs + 1)

    return clusters


def find_missing_by_shift(dt_trigger, dt_root, ratio_threshold=2.0, eps=1e-13):
    missing_trigger_indices = []
    missing_root_indices = []
    aligned_ratios = []
    root_to_trigger = []

    shift = 0
    i_root = 0

    while i_root < len(dt_root):
        j = i_root + shift
        if j >= len(dt_trigger):
            break

        r = dt_root[i_root]
        t = dt_trigger[j]
        ratio = r / (t + eps)

        if ratio > ratio_threshold and j + 1 < len(dt_trigger):
            t2 = t + dt_trigger[j + 1]
            ratio2 = r / (t2 + eps)

            if 0.99 < ratio2 < 1.01:
                missing_trigger_indices.append(j)
                missing_root_indices.append(i_root)

                shift += 1
                j = i_root + shift
                if j >= len(dt_trigger):
                    break

                t = dt_trigger[j]
                ratio = r / (t + eps)

        aligned_ratios.append(ratio)
        root_to_trigger.append(j)

        i_root += 1

    trigger_to_root = np.full(len(dt_trigger), np.nan)
    for i_root, j_trigger in enumerate(root_to_trigger):
        trigger_to_root[j_trigger] = i_root

    return (
        np.array(missing_trigger_indices),
        np.array(missing_root_indices),
        np.array(root_to_trigger),
        trigger_to_root,
        np.array(aligned_ratios),
    )


def read_trigger_values(meta_path):
    with open(meta_path, "r") as f:
        lines = f.readlines()

    trigger_values = []
    for line in lines:
        if "trigger_time" in line.lower():
            match = re.search(r"\[(.*?)\]", line)
            if not match:
                continue
            parts = match.group(1).split(";")
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                try:
                    trigger_values.append(float(p))
                except ValueError:
                    print("Failed to parse trigger_time value:", repr(p))

    return trigger_values


def save_plot(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Intro.ipynb synchronization workflow with saved outputs."
    )
    parser.add_argument("--csv-path", default="4237_1_e.csv")
    parser.add_argument("--scope-dir", default="4237_scope")
    parser.add_argument(
        "--meta-path",
        default=None,
        help="Override full path to meta CSV (defaults to scope-dir/raw_C1_0004237_0000001_6347_meta.csv).",
    )
    parser.add_argument("--gap-factor", type=float, default=500.0)
    parser.add_argument("--ratio-threshold", type=float, default=1.01)
    parser.add_argument("--output-dir", default="sync_outputs")
    parser.add_argument("--window-start", type=int, default=2400)
    parser.add_argument("--window-end", type=int, default=2500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    df = pd.read_csv(args.csv_path)
    df["time_list"] = df["time"].apply(parse_list)

    df["time_last"] = df["time_list"].apply(
        lambda lst: lst[-1] if len(lst) > 0 else None
    )
    vals = df["time_last"].dropna().values

    fig = plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=80)
    plt.xlabel("Last time value")
    plt.ylabel("Count")
    plt.title("Distribution of last values in 'time' lists")
    plt.tight_layout()
    save_plot(fig, output_dir / "hist_time_last.png")

    if args.meta_path:
        meta_path = args.meta_path
    else:
        meta_path = str(Path(args.scope_dir) / "raw_C1_0004237_0000001_6347_meta.csv")
    trigger_values = read_trigger_values(meta_path)

    fig = plt.figure(figsize=(7, 4))
    plt.hist(trigger_values, bins=80)
    plt.xlabel("trigger_time")
    plt.ylabel("Count")
    plt.title("Distribution of trigger_time values")
    plt.tight_layout()
    save_plot(fig, output_dir / "hist_trigger_time.png")

    trigger_ps = np.array(trigger_values) * 1e12

    clusters = split_by_time_gaps(vals, args.gap_factor)
    if len(clusters) == 0:
        raise RuntimeError("No clusters found in time values.")
    first_bunch = clusters[0]

    fig = plt.figure(figsize=(7, 4))
    plt.hist(first_bunch, bins=80)
    plt.xlabel("Last time value")
    plt.ylabel("Count")
    plt.title("Distribution of last values in 'time' lists (first bunch)")
    plt.tight_layout()
    save_plot(fig, output_dir / "hist_first_bunch.png")

    # Pad trigger values with zeros if shorter (for scatter/fit)
    N = len(first_bunch)
    M = len(trigger_ps)
    if M < N:
        padded_trigger = np.concatenate([trigger_ps, np.zeros(N - M)])
    else:
        padded_trigger = trigger_ps[:N]

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(padded_trigger, first_bunch, s=12, alpha=0.6)
    plt.xlabel("Trigger time (ps)")
    plt.ylabel("Last time value (ps)")
    plt.title("Scatter: First Bunch vs Trigger Time (missing padded = 0)")
    plt.grid(True, alpha=0.3)
    save_plot(fig, output_dir / "scatter_padded.png")

    x = np.array(padded_trigger)
    y = np.array(first_bunch)
    mask = x > 0
    x_fit = x[mask]
    y_fit = y[mask]

    a_pad, b_pad = np.polyfit(x_fit, y_fit, 1)
    x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_line = a_pad * x_line + b_pad

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=12, alpha=0.6, label="data")
    plt.plot(
        x_line,
        y_line,
        color="red",
        linewidth=2,
        label=f"fit: y = {a_pad:.4f} x + {b_pad:.2e}",
    )
    plt.xlabel("Trigger time (ps)")
    plt.ylabel("Last time value (ps)")
    plt.title("Linear Fit: First Bunch vs Trigger Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(fig, output_dir / "fit_padded.png")

    trigger = np.array(trigger_ps)
    root = np.array(first_bunch)

    dt_trigger = np.diff(trigger)
    dt_root = np.diff(root)

    L = min(len(dt_trigger), len(dt_root))
    dt_trunc_trigger = dt_trigger[:L]
    dt_trunc_root = dt_root[:L]
    eps = 1e-13
    ratio = dt_trunc_root / (dt_trunc_trigger + eps)

    x_idx = np.arange(L)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(x_idx, ratio, ".", markersize=3)
    plt.axhline(1.0, color="red", linestyle="--", label="ratio = 1")
    plt.xlabel("step index (n -> n+1)")
    plt.ylabel("Δt_trigger / Δt_root")
    plt.title("Ratio of consecutive time differences")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.yscale("log")
    save_plot(fig, output_dir / "ratio_log.png")

    (
        missing_trig_idx,
        missing_root_idx,
        root_to_trigger,
        trigger_to_root,
        aligned_ratio,
    ) = find_missing_by_shift(dt_trigger, dt_root, args.ratio_threshold, eps)

    L_aligned = len(aligned_ratio)
    x_aligned = np.arange(L_aligned)

    fig = plt.figure(figsize=(12, 5))
    plt.scatter(x_aligned, aligned_ratio, s=20, color="blue", label="aligned ratio")
    missing_in_range = [i for i in missing_root_idx if i < L_aligned]
    if len(missing_in_range) > 0:
        plt.scatter(
            missing_in_range,
            aligned_ratio[missing_in_range],
            s=40,
            color="red",
            label="missing (detected)",
        )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ratio = 1")
    plt.yscale("log")
    plt.xlabel("step index (ROOT Δt index)")
    plt.ylabel("Δt_root / Δt_trigger (aligned, log)")
    plt.title("Aligned Ratio with Missing Points Highlighted (ROOT indices)")
    plt.legend()
    plt.tight_layout()
    save_plot(fig, output_dir / "aligned_ratio_missing.png")

    mapped_trigger = []
    mapped_root = []
    for j_trigger, i_root in enumerate(trigger_to_root):
        if not np.isnan(i_root):
            mapped_trigger.append(trigger_ps[j_trigger])
            mapped_root.append(first_bunch[int(i_root)])

    mapped_trigger = np.array(mapped_trigger)
    mapped_root = np.array(mapped_root)

    a_map, b_map = np.polyfit(mapped_trigger, mapped_root, 1)
    x_line = np.linspace(mapped_trigger.min(), mapped_trigger.max(), 400)
    y_line = a_map * x_line + b_map

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(mapped_trigger, mapped_root, s=12, alpha=0.6, label="mapped data")
    plt.plot(
        x_line,
        y_line,
        color="red",
        linewidth=2,
        label=f"fit: y = {a_map:.10f} x + {b_map:.3e}",
    )
    plt.xlabel("Trigger time (ps)")
    plt.ylabel("First bunch last time (ps)")
    plt.title("Mapped ROOT vs Trigger Correlation (Aligned)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(fig, output_dir / "mapped_fit.png")

    start = args.window_start
    end = min(args.window_end, L_aligned)
    x_window = np.arange(start, end)
    y_window = aligned_ratio[start:end]
    missing_in_window = [i for i in missing_root_idx if start <= i < end]

    fig = plt.figure(figsize=(10, 4))
    plt.scatter(x_window, y_window, s=40, color="blue", label="aligned ratio")
    if len(missing_in_window) > 0:
        plt.scatter(
            missing_in_window,
            aligned_ratio[missing_in_window],
            s=60,
            color="red",
            label="missing (detected)",
        )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.xlabel("step index")
    plt.ylabel("Δt_root / Δt_trigger (aligned, log)")
    plt.title(f"Aligned Ratio (index {start} to {end})")
    plt.legend()
    plt.tight_layout()
    save_plot(fig, output_dir / "aligned_ratio_window.png")

    np.savez(
        output_dir / "mapping_results.npz",
        trigger_ps=trigger_ps,
        first_bunch=first_bunch,
        missing_trig_idx=missing_trig_idx,
        missing_root_idx=missing_root_idx,
        root_to_trigger=root_to_trigger,
        trigger_to_root=trigger_to_root,
        aligned_ratio=aligned_ratio,
        mapped_trigger=mapped_trigger,
        mapped_root=mapped_root,
        slope_padded=a_pad,
        offset_padded=b_pad,
        slope_mapped=a_map,
        offset_mapped=b_map,
    )

    mapped_df = pd.DataFrame(
        {
            "trigger_ps": mapped_trigger,
            "root_ps": mapped_root,
        }
    )
    mapped_df.to_csv(output_dir / "mapped_points.csv", index=False)

    stats = {
        "csv_path": args.csv_path,
        "meta_path": meta_path,
        "num_vals": int(len(vals)),
        "num_trigger": int(len(trigger_ps)),
        "num_first_bunch": int(len(first_bunch)),
        "num_mapped": int(len(mapped_trigger)),
        "gap_factor": args.gap_factor,
        "ratio_threshold": args.ratio_threshold,
        "slope_padded": float(a_pad),
        "offset_padded": float(b_pad),
        "slope_mapped": float(a_map),
        "offset_mapped": float(b_map),
    }
    (output_dir / "summary.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
