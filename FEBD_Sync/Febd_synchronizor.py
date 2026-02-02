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
import csv


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


# New helper: preserve original indices while splitting by gaps
def split_by_time_gaps_with_indices(values, indices, gap_factor=100):
    """
    Split values into clusters based on large gaps while preserving corresponding original indices.
    Returns (clusters_values, clusters_indices) where each is a list of numpy arrays.
    """
    # Ensure numpy arrays
    vals = np.asarray(values)
    idxs = np.asarray(indices)

    # sort by value and apply the same ordering to indices
    order = np.argsort(vals)
    vals_sorted = vals[order]
    idxs_sorted = idxs[order]

    # find gaps on sorted values
    gaps = np.diff(vals_sorted)
    threshold = np.median(gaps) * gap_factor
    split_idxs = np.where(gaps > threshold)[0]

    vals_clusters = np.split(vals_sorted, split_idxs + 1)
    idxs_clusters = np.split(idxs_sorted, split_idxs + 1)

    return vals_clusters, idxs_clusters


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
    # keep the original dataframe indices for each non-null time_last so we can
    # report the original CSV row indices for matched entries
    # prefer the original 'entry' column from the CSV if present; otherwise fall back
    # to the dataframe row index. These values will be stored with matched points so
    # you can trace matches back to ../trc_out_root_reco/4237_1_e.csv
    time_last_series = df["time_last"].dropna()
    vals = time_last_series.values
    if 'entry' in df.columns:
        vals_indices = df.loc[time_last_series.index, 'entry'].values
    else:
        vals_indices = time_last_series.index.values

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

    # split into clusters, preserving original row indices
    clusters, clusters_idx = split_by_time_gaps_with_indices(vals, vals_indices, args.gap_factor)
    if len(clusters) == 0:
        raise RuntimeError("No clusters found in time values.")
    first_bunch = clusters[0]
    first_bunch_idx = clusters_idx[0]

    fig = plt.figure(figsize=(7, 4))
    plt.hist(first_bunch, bins=80)
    plt.xlabel("Last time value")
    plt.ylabel("Count")
    plt.title("Distribution of last values in 'time' lists (first bunch)")
    plt.tight_layout()
    save_plot(fig, output_dir / "hist_first_bunch.png")

    # Determine meta files to use for each cluster/segment. Support single file, glob pattern,
    # or a provided file whose numeric suffix (before _meta.csv) can be replaced to find siblings.
    import glob
    import os

    if args.meta_path:
        # If meta_path contains glob tokens use it directly
        if any(ch in args.meta_path for ch in "*?["):
            meta_files = sorted(glob.glob(args.meta_path))
        else:
            # try to discover sibling meta files by turning the last numeric group into a wildcard
            m = re.search(r"_(\d+)_meta\.csv$", args.meta_path)
            if m:
                pattern = args.meta_path[: m.start(1)] + "*" + args.meta_path[m.end(1) :]
                meta_files = sorted(glob.glob(pattern))
                if not meta_files and os.path.exists(args.meta_path):
                    meta_files = [args.meta_path]
            else:
                meta_files = [args.meta_path] if os.path.exists(args.meta_path) else []
    else:
        # default: use scope-dir and the expected naming convention raw_C1_..._meta.csv
        default_pattern = str(Path(args.scope_dir) / "raw_C1_*_*_meta.csv")
        meta_files = sorted(glob.glob(default_pattern))

    if not meta_files:
        print(f'No meta files found (pattern/source). Skipping per-segment synchronization.')
    else:
        n_clusters = len(clusters)
        n_meta = len(meta_files)
        use_n = min(n_clusters, n_meta)
        print(f'Found {n_meta} meta files and {n_clusters} clusters; processing {use_n} segment(s)')

        for seg_idx in range(use_n):
            seg_num = seg_idx + 1
            meta_file = meta_files[seg_idx]
            first_bunch = clusters[seg_idx]
            first_bunch_idx = clusters_idx[seg_idx]
            print(f'Processing segment {seg_num}: meta={os.path.basename(meta_file)} (cluster len={len(first_bunch)})')

            # derive a stable suffix from the meta filename so outputs are named consistently
            base = os.path.basename(meta_file)
            mnum = re.search(r'_(\d+(?:_\d+)+)_meta\.csv$', base)
            if mnum:
                suffix = mnum.group(1)
            else:
                # fallback to segment number
                suffix = f'seg{seg_num}'

            trigger_values = read_trigger_values(meta_file)
            trigger_ps = np.array(trigger_values) * 1e12

            # Pad trigger values with zeros if shorter (for scatter/fit)
            N = len(first_bunch)
            M = len(trigger_ps)
            if M < N:
                padded_trigger = np.concatenate([trigger_ps, np.zeros(N - M)])
            else:
                padded_trigger = trigger_ps[:N]

            # scatter and fit for this segment
            fig = plt.figure(figsize=(7, 5))
            plt.scatter(padded_trigger, first_bunch, s=12, alpha=0.6)
            plt.xlabel("Trigger time (ps)")
            plt.ylabel("Last time value (ps)")
            plt.title(f"Scatter: {base} First Bunch vs Trigger Time (missing padded = 0)")
            plt.grid(True, alpha=0.3)
            save_plot(fig, output_dir / f"scatter_padded_{suffix}.png")

            x = np.array(padded_trigger)
            y = np.array(first_bunch)
            mask = x > 0
            x_fit = x[mask]
            y_fit = y[mask]

            if len(x_fit) < 2:
                print(f'Not enough non-zero trigger points to fit for segment {seg_num}; skipping fit and mapping')
                continue

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
            plt.title(f"Linear Fit: {base} First Bunch vs Trigger Time")
            plt.legend()
            plt.grid(True, alpha=0.3)
            save_plot(fig, output_dir / f"fit_padded_{suffix}.png")

            trigger = np.array(trigger_ps)
            root = np.array(first_bunch)
            dt_trigger = np.diff(trigger)
            dt_root = np.diff(root)

            L = min(len(dt_trigger), len(dt_root))
            dt_trunc_trigger = dt_trigger[:L]
            dt_trunc_root = dt_root[:L]
            eps = 1e-13

            (
                missing_trig_idx,
                missing_root_idx,
                root_to_trigger,
                trigger_to_root,
                aligned_ratio,
            ) = find_missing_by_shift(dt_trigger, dt_root, args.ratio_threshold, eps)

            # Prepare mapped arrays and include original CSV entry indices for root
            mapped_trigger = []
            mapped_root = []
            mapped_root_idx = []
            for j_trigger, i_root in enumerate(trigger_to_root):
                if not np.isnan(i_root):
                    i_root_int = int(i_root)
                    mapped_trigger.append(trigger_ps[j_trigger])
                    mapped_root.append(first_bunch[i_root_int])
                    # map to original row index from the input CSV
                    mapped_root_idx.append(int(first_bunch_idx[i_root_int]))

            mapped_trigger = np.array(mapped_trigger)
            mapped_root = np.array(mapped_root)
            mapped_root_idx = np.array(mapped_root_idx)

            if len(mapped_trigger) >= 2:
                a_map, b_map = np.polyfit(mapped_trigger, mapped_root, 1)
            else:
                a_map, b_map = np.nan, np.nan

            # Save mapping results for this segment as CSV (json-serialized values)
            mapping = {
                "trigger_ps": np.array(trigger_ps).tolist(),
                "first_bunch": np.array(first_bunch).tolist(),
                "missing_trig_idx": np.array(missing_trig_idx).tolist(),
                "missing_root_idx": np.array(missing_root_idx).tolist(),
                "root_to_trigger": np.array(root_to_trigger).tolist(),
                "trigger_to_root": np.array(trigger_to_root).tolist(),
                "aligned_ratio": np.array(aligned_ratio).tolist(),
                "mapped_trigger": np.array(mapped_trigger).tolist(),
                "mapped_root": np.array(mapped_root).tolist(),
                "mapped_root_idx": np.array(mapped_root_idx).tolist(),
                "slope_padded": float(a_pad),
                "offset_padded": float(b_pad),
                "slope_mapped": float(a_map) if not np.isnan(a_map) else None,
                "offset_mapped": float(b_map) if not np.isnan(b_map) else None,
            }

            out_csv = output_dir / f"mapping_results_{suffix}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(["key", "value_json"])
                for k, v in mapping.items():
                    writer.writerow([k, json.dumps(v)])
            print(f'Saved mapping results to {out_csv}')

            # Save mapped points for this segment
            mapped_df = pd.DataFrame({"trigger_ps": mapped_trigger, "root_ps": mapped_root, "root_idx": mapped_root_idx})
            mapped_df.to_csv(output_dir / f"mapped_points_{suffix}.csv", index=False)
            print(f'Saved mapped points to {output_dir / f"mapped_points_{suffix}.csv"}')

            # Save mapped-fit plot if mapping exists
            if len(mapped_trigger) > 0:
                x_line = np.linspace(mapped_trigger.min(), mapped_trigger.max(), 400) if len(mapped_trigger) >= 2 else np.array([0, 1])
                y_line = (a_map * x_line + b_map) if not np.isnan(a_map) else np.zeros_like(x_line)

                fig = plt.figure(figsize=(7, 5))
                plt.scatter(mapped_trigger, mapped_root, s=12, alpha=0.6, label="mapped data")
                if not np.isnan(a_map):
                    plt.plot(x_line, y_line, color="red", linewidth=2, label=f"fit: y = {a_map:.10f} x + {b_map:.3e}")
                plt.xlabel("Trigger time (ps)")
                plt.ylabel("First bunch last time (ps)")
                plt.title(f"Mapped ROOT vs Trigger Correlation (Aligned) - {base}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                save_plot(fig, output_dir / f"mapped_fit_{suffix}.png")
    # Final summary:
    print('\nInferred branches:')
    print('  channel branch :', ch_branch)
    print('  channelIdx branch :', idx_branch)
    print('  time branch :', time_branch)
    print('  energy branch :', energy_branch)

    if getattr(args, 'dump_df', None):
        print('\nDataFrame saved to:', args.dump_df)
    else:
        print('\nDump options removed except --dump-df. Use --dump-df to export DataFrame as pickle.')

    return


if __name__ == '__main__':
    main()
