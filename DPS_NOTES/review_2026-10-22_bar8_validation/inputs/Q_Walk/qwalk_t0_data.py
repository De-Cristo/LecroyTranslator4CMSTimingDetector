"""Synchronized TOFHIR/MCP ingestion for the Run-4399 Q-walk comparison."""

import re
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from Q_Walk.qwalk_channels import (
    TRIGGER_CHANNEL,
    UPSTREAM_BAR_CHANNELS,
    reference_topology,
)
from Q_Walk.qwalk_core import (
    build_calibration_table,
    calculate_slope,
    decode_thresholds,
    lookup_calibration,
    reconstruct_time2_safe,
)


TOFHIR_BRANCHES = (
    "step1",
    "step2",
    "tot",
    "t1coarse",
    "t1fine",
    "t2coarse",
    "t2fine",
    "time",
    "channelID",
    "energy",
    "tacID",
)
MCP_BRANCHES = ("index", "peak_time", "trigger_time", "peak_amp", "phi_peak")

MATCHED_COLUMNS = (
    "file_id",
    "event_id",
    "entry",
    "segment",
    "bar",
    "channelID",
    "energy",
    "time",
    "time2",
    "delta_t12_ps",
    "delta_i_ua",
    "slope_ua_per_ns",
    "inv_slope_ns_per_ua",
    "ch192_time",
    "mcp_trigger_time",
    "mcp_peak_time",
    "mcp_peak_amp",
)
MATCHED_DTYPES = {
    "file_id": "string",
    "event_id": "string",
    "entry": "int64",
    "segment": "int64",
    "bar": "int64",
    "channelID": "int64",
    "energy": "float64",
    "time": "float64",
    "time2": "float64",
    "delta_t12_ps": "float64",
    "delta_i_ua": "float64",
    "slope_ua_per_ns": "float64",
    "inv_slope_ns_per_ua": "float64",
    "ch192_time": "float64",
    "mcp_trigger_time": "float64",
    "mcp_peak_time": "float64",
    "mcp_peak_amp": "float64",
}

REFERENCE_ALIGNMENT_COLUMNS = (
    "file_id",
    "segment",
    "mcp_trigger_time",
    "mcp_peak_time",
    "ch192_time",
    "time",
)
EVENT_ALIGNMENT_COLUMNS = (
    "mcp_trigger_time",
    "mcp_peak_time",
    "ch192_time",
)


def _validate_lowess_parameters(lowess_frac, lowess_n_iter):
    integer_types = (int, np.integer)
    real_types = (int, float, np.integer, np.floating)
    if (
        isinstance(lowess_n_iter, (bool, np.bool_))
        or not isinstance(lowess_n_iter, integer_types)
        or lowess_n_iter < 0
    ):
        raise ValueError("lowess_n_iter must be a non-negative integer")
    if (
        isinstance(lowess_frac, (bool, np.bool_))
        or not isinstance(lowess_frac, real_types)
        or not np.isfinite(lowess_frac)
        or not 0.0 < lowess_frac <= 1.0
    ):
        raise ValueError("lowess_frac must be finite and in (0, 1]")


def _centered_line_fit(x, y, mask):
    x_fit = x[mask]
    y_fit = y[mask]
    x_anchor = float(x_fit[0])
    y_anchor = float(y_fit[0])
    x_relative = x_fit - x_anchor
    y_relative = y_fit - y_anchor
    x_center = float(np.mean(x_relative))
    y_center = float(np.mean(y_relative))
    dx = x_relative - x_center
    denominator = float(np.dot(dx, dx))
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0, x_anchor, y_anchor, y_center
    slope = float(np.dot(dx, y_relative - y_center) / denominator)
    relative_intercept = y_center - slope * x_center
    return slope, x_anchor, y_anchor, relative_intercept


def robust_centered_line(x, y, n_iter=3, mad_nsigma=3.0):
    """Fit a robust line after centering timestamps at every iteration."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    if not np.isfinite(mad_nsigma) or mad_nsigma <= 0.0:
        raise ValueError("mad_nsigma must be finite and positive")

    finite = np.isfinite(x) & np.isfinite(y)
    prediction = np.full(x.shape, np.nan, dtype=float)
    residual = np.full(x.shape, np.nan, dtype=float)
    mask = finite.copy()
    if not mask.any():
        return prediction, residual, mask, np.nan

    slope = 0.0
    for iteration in range(n_iter + 1):
        slope, x_anchor, y_anchor, relative_intercept = _centered_line_fit(
            x, y, mask
        )
        finite_x = np.isfinite(x)
        prediction[finite_x] = y_anchor + (
            slope * (x[finite_x] - x_anchor) + relative_intercept
        )
        residual[finite] = y[finite] - prediction[finite]

        if iteration == n_iter:
            break
        active_residual = residual[mask]
        residual_center = float(np.median(active_residual))
        absolute_deviation = np.abs(active_residual - residual_center)
        mad = float(np.median(absolute_deviation))
        if not np.isfinite(mad):
            break
        if mad == 0.0:
            value_scale = max(
                1.0,
                float(np.max(np.abs(y[mask]))),
                float(np.max(np.abs(prediction[mask]))),
            )
            tolerance = 32.0 * np.finfo(float).eps * value_scale
            updated = finite & (
                np.abs(residual - residual_center) <= tolerance
            )
        else:
            updated = finite & (
                np.abs(residual - residual_center) <= mad_nsigma * 1.4826 * mad
            )
        if updated.sum() < 2 or np.array_equal(updated, mask):
            break
        mask = updated

    return prediction, residual, mask, slope


def _tricube(values):
    values = np.asarray(values, dtype=float)
    weights = np.zeros_like(values)
    inside = np.abs(values) < 1.0
    weights[inside] = (1.0 - np.abs(values[inside]) ** 3) ** 3
    return weights


def _bisquare(values):
    values = np.asarray(values, dtype=float)
    weights = np.zeros_like(values)
    inside = np.abs(values) < 1.0
    weights[inside] = (1.0 - values[inside] ** 2) ** 2
    return weights


def _weighted_local_linear_prediction(x, y, weights, x_eval):
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.nan
    x_relative = x - x_eval
    x_center = float(np.sum(weights * x_relative) / weight_sum)
    y_center = float(np.sum(weights * y) / weight_sum)
    dx = x_relative - x_center
    denominator = float(np.sum(weights * dx * dx))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return y_center
    slope = float(np.sum(weights * dx * (y - y_center)) / denominator)
    return y_center - slope * x_center


def _nearest_window_bounds(x_sorted, position, neighbor_count):
    left = position
    right = position
    while right - left + 1 < neighbor_count:
        if left == 0:
            right += 1
        elif right == len(x_sorted) - 1:
            left -= 1
        else:
            left_distance = x_sorted[position] - x_sorted[left - 1]
            right_distance = x_sorted[right + 1] - x_sorted[position]
            if left_distance <= right_distance:
                left -= 1
            else:
                right += 1
    return left, right + 1


def _robust_lowess_at_training_points(x, y, frac, n_iter):
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]
    point_count = len(x_sorted)
    neighbor_count = min(point_count, max(2, int(np.ceil(frac * point_count))))
    robust_weights = np.ones(point_count, dtype=float)
    fitted = np.empty(point_count, dtype=float)

    for iteration in range(n_iter + 1):
        for position in range(point_count):
            left, right = _nearest_window_bounds(
                x_sorted, position, neighbor_count
            )
            x_window = x_sorted[left:right]
            y_window = y_sorted[left:right]
            distances = np.abs(x_window - x_sorted[position])
            bandwidth = float(np.max(distances))
            if bandwidth == 0.0:
                local = distances == 0.0
                weights = robust_weights[left:right] * local
            else:
                weights = _tricube(distances / np.nextafter(bandwidth, np.inf))
                weights *= robust_weights[left:right]
            estimate = _weighted_local_linear_prediction(
                x_window, y_window, weights, x_sorted[position]
            )
            fitted[position] = y_sorted[position] if not np.isfinite(estimate) else estimate

        if iteration == n_iter:
            break
        fit_residual = y_sorted - fitted
        mad = float(np.median(np.abs(fit_residual - np.median(fit_residual))))
        if mad == 0.0 or not np.isfinite(mad):
            break
        robust_weights = _bisquare(fit_residual / (6.0 * mad))

    return x_sorted, fitted


def _has_nondegenerate_alignment_axis(x):
    if len(x) < 2 or len(np.unique(x)) < 2:
        return False
    x_relative = x - x[0]
    x_centered = x_relative - np.mean(x_relative)
    centered_sum_squares = float(np.dot(x_centered, x_centered))
    return np.isfinite(centered_sum_squares) and centered_sum_squares > 0.0


def _robust_lowess_correction(x, residual, inlier, x_eval, frac, n_iter=3):
    training = inlier & np.isfinite(x) & np.isfinite(residual)
    training_x = x[training]
    if len(training_x) >= 2 and not _has_nondegenerate_alignment_axis(training_x):
        return np.zeros(len(x_eval), dtype=float), "degenerate_alignment_axis"
    if training.sum() < 10:
        return np.zeros(len(x_eval), dtype=float), "insufficient_alignment_points"
    if not np.isfinite(frac) or not 0.0 < frac <= 1.0:
        raise ValueError("lowess_frac must be finite and in (0, 1]")

    x_sorted, fitted = _robust_lowess_at_training_points(
        training_x, residual[training], frac, n_iter
    )
    unique_x = np.unique(x_sorted)
    if len(unique_x) == len(x_sorted):
        unique_fit = fitted
    else:
        unique_fit = np.array(
            [np.mean(fitted[x_sorted == value]) for value in unique_x], dtype=float
        )
    correction = np.full(len(x_eval), np.nan, dtype=float)
    finite_eval = np.isfinite(x_eval)
    correction[finite_eval] = np.interp(
        x_eval[finite_eval], unique_x, unique_fit, left=unique_fit[0], right=unique_fit[-1]
    )
    return correction, "ok"


def _unique_alignment_events(segment):
    if "event_id" in segment.columns:
        identity_column = "event_id"
    elif "entry" in segment.columns:
        identity_column = "entry"
    else:
        row_count = len(segment)
        positions = np.arange(row_count, dtype=int)
        return positions, positions

    event_groups = segment.groupby(
        identity_column, sort=False, dropna=False
    ).indices
    event_rows = np.empty(len(event_groups), dtype=int)
    row_to_event = np.empty(len(segment), dtype=int)
    for event_position, (event_key, row_positions) in enumerate(
        event_groups.items()
    ):
        row_positions = np.asarray(row_positions, dtype=int)
        event_rows[event_position] = row_positions[0]
        row_to_event[row_positions] = event_position
        for column in EVENT_ALIGNMENT_COLUMNS:
            values = segment.iloc[row_positions][column].to_numpy(dtype=float)
            reference = np.full(values.shape, values[0], dtype=float)
            if not np.array_equal(values, reference, equal_nan=True):
                raise ValueError(
                    f"Inconsistent {column} within {identity_column}={event_key!r}"
                )
    return event_rows, row_to_event


def reconstruct_reference_delta_t(frame, lowess_frac=0.15, lowess_n_iter=3):
    """Construct the direct MCP-t0 label independently within each segment."""
    _validate_lowess_parameters(lowess_frac, lowess_n_iter)
    missing = [column for column in REFERENCE_ALIGNMENT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = frame.copy()
    row_count = len(result)
    expected = np.full(row_count, np.nan, dtype=float)
    line_slope = np.full(row_count, np.nan, dtype=float)
    alignment_residual = np.full(row_count, np.nan, dtype=float)
    alignment_inlier = np.zeros(row_count, dtype=bool)
    lowess_correction = np.full(row_count, np.nan, dtype=float)
    status = np.full(row_count, "", dtype=object)

    grouped = result.groupby(["file_id", "segment"], sort=False, dropna=False)
    for positions in grouped.indices.values():
        positions = np.asarray(positions, dtype=int)
        segment = result.iloc[positions]
        event_rows, row_to_event = _unique_alignment_events(segment)
        events = segment.iloc[event_rows]
        trigger = events["mcp_trigger_time"].to_numpy(dtype=float)
        ch192 = events["ch192_time"].to_numpy(dtype=float)
        prediction, residual, inlier, slope = robust_centered_line(trigger, ch192)
        correction, segment_status = _robust_lowess_correction(
            trigger,
            residual,
            inlier,
            trigger,
            frac=lowess_frac,
            n_iter=lowess_n_iter,
        )
        expected[positions] = (prediction + correction)[row_to_event]
        line_slope[positions] = slope
        alignment_residual[positions] = residual[row_to_event]
        alignment_inlier[positions] = inlier[row_to_event]
        lowess_correction[positions] = correction[row_to_event]
        status[positions] = segment_status

    result["expected_ch192_time"] = expected
    result["ch192_line_slope"] = line_slope
    result["ch192_alignment_residual"] = alignment_residual
    result["ch192_alignment_inlier"] = alignment_inlier
    result["ch192_lowess_correction"] = lowess_correction
    result["reference_alignment_status"] = status
    result["delta_t_ref_ps"] = (
        result["time"].to_numpy(dtype=float) - expected
    ) - (
        result["mcp_peak_time"].to_numpy(dtype=float)
        - result["mcp_trigger_time"].to_numpy(dtype=float)
    )
    return result


def _empty_matched_frame():
    return pd.DataFrame(
        {column: pd.Series(dtype=MATCHED_DTYPES[column]) for column in MATCHED_COLUMNS}
    )


def _coerce_matched_frame(frame):
    return frame.loc[:, MATCHED_COLUMNS].astype(MATCHED_DTYPES)


def match_mcp_events(events, mcp):
    """Inner-join TOFHIR rows to unique MCP entries in entry order."""
    return (
        events.merge(mcp, on="entry", how="inner", validate="many_to_one")
        .sort_values("entry", kind="mergesort")
        .reset_index(drop=True)
    )


def _event_topologies(hits):
    return hits.groupby("entry")["channelID"].agg(
        hit_count="size", channel_set=lambda values: frozenset(map(int, values))
    )


def select_exact_reference_topology(hits, bar):
    """Return all rows from events containing exactly one of each required hit."""
    required = reference_topology(bar)
    topology = _event_topologies(hits)
    accepted = topology.index[
        topology["hit_count"].eq(len(required)) & topology["channel_set"].eq(required)
    ]
    return hits.loc[hits["entry"].isin(accepted)].copy()

def select_exact_tofhir_topology(hits: pd.DataFrame, channels: frozenset[int]) -> pd.DataFrame:
    """Return all rows from events containing exactly one of each required hit."""
    topology = _event_topologies(hits)
    accepted = topology.index[
        topology["hit_count"].eq(len(channels)) & topology["channel_set"].eq(channels)
    ]
    return hits.loc[hits["entry"].isin(accepted)].copy()


def assign_scope_segments(trigger_time, gap_factor=500.0):
    """Label trigger-time regions separated by gaps or clock resets.

    Empty, singleton, constant, and decreasing-only inputs raise ``ValueError``
    because they contain no positive spacing from which to estimate a baseline.
    """
    trigger_time = np.asarray(trigger_time, dtype=float)
    gaps = np.diff(trigger_time)
    positive = gaps[gaps > 0.0]
    if len(positive) == 0:
        raise ValueError("Cannot segment trigger times without positive spacing")
    threshold = float(gap_factor) * np.median(positive)
    boundaries = (gaps < 0.0) | (gaps > threshold)
    return np.concatenate(([0], np.cumsum(boundaries))).astype(int)


def _read_tofhir_hits(tree):
    hits = ak.to_dataframe(tree.arrays(TOFHIR_BRANCHES, library="ak")).reset_index()
    calibration = build_calibration_table(
        hits["channelID"].to_numpy(),
        hits["tacID"].to_numpy(),
        hits["t1fine"].to_numpy(),
        hits["time"].to_numpy(),
    )
    calib_t1, _ = lookup_calibration(
        calibration,
        hits["channelID"].to_numpy(),
        hits["tacID"].to_numpy(),
        hits["t1fine"].to_numpy(),
    )
    calib_t2, _ = lookup_calibration(
        calibration,
        hits["channelID"].to_numpy(),
        hits["tacID"].to_numpy(),
        hits["t2fine"].to_numpy(),
    )
    hits["time2"] = reconstruct_time2_safe(
        hits["time"].to_numpy(),
        hits["t1coarse"].to_numpy(),
        hits["t2coarse"].to_numpy(),
        calib_t1,
        calib_t2,
    )
    hits["delta_t12_ps"] = hits["time2"] - hits["time"]
    _, _, hits["delta_i_ua"] = decode_thresholds(hits["step2"].to_numpy())
    hits["slope_ua_per_ns"] = calculate_slope(
        hits["time"].to_numpy(),
        hits["time2"].to_numpy(),
        hits["delta_i_ua"].to_numpy(),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        hits["inv_slope_ns_per_ua"] = 1.0 / hits["slope_ua_per_ns"]
    return hits


def _read_mcp_events(tree):
    events = pd.DataFrame(tree.arrays(MCP_BRANCHES, library="np"))
    events = events.rename(
        columns={
            "index": "entry",
            "trigger_time": "mcp_trigger_time",
            "peak_time": "mcp_peak_time",
            "peak_amp": "mcp_peak_amp",
            "phi_peak": "mcp_phi_peak",
        }
    )
    events = events.sort_values("entry", kind="mergesort").reset_index(drop=True)
    events["segment"] = assign_scope_segments(events["mcp_trigger_time"].to_numpy())
    return events


def _select_reference_hits(hits, bars):
    requested_bars = tuple(dict.fromkeys(map(int, bars)))
    topology_to_bar = {reference_topology(bar): bar for bar in requested_bars}
    topology = _event_topologies(hits)
    accepted = topology.loc[topology["hit_count"].eq(5), ["channel_set"]].copy()
    accepted["bar"] = accepted["channel_set"].map(topology_to_bar)
    accepted = accepted.dropna(subset=["bar"])
    if accepted.empty:
        return pd.DataFrame()

    accepted["bar"] = accepted["bar"].astype(np.int64)
    topology_hits = hits.merge(
        accepted[["bar"]],
        left_on="entry",
        right_index=True,
        how="inner",
        validate="many_to_one",
    )
    ch192 = topology_hits.loc[
        topology_hits["channelID"].eq(TRIGGER_CHANNEL), ["entry", "time"]
    ].rename(columns={"time": "ch192_time"})

    channel_to_bar = {
        channel: bar
        for bar in requested_bars
        for channel in UPSTREAM_BAR_CHANNELS[bar]
    }
    upstream_bar = topology_hits["channelID"].map(channel_to_bar)
    upstream = topology_hits.loc[upstream_bar.eq(topology_hits["bar"])].copy()
    return upstream.merge(ch192, on="entry", how="inner", validate="many_to_one")


def read_synchronized_file(path, bars=tuple(range(16))):
    """Build the matched reference-hit table from the latest data and MCP cycles."""
    with uproot.open(path) as root_file:
        hits = _read_tofhir_hits(root_file["data"])
        mcp = _read_mcp_events(root_file["MCP"])

    reference_hits = _select_reference_hits(hits, bars)
    if reference_hits.empty:
        return _empty_matched_frame()

    matched = match_mcp_events(reference_hits, mcp)
    if matched.empty:
        return _empty_matched_frame()

    file_id = Path(path).stem
    matched["file_id"] = file_id
    matched["event_id"] = matched["entry"].map(lambda entry: f"{file_id}:{entry}")
    matched = matched.sort_values(
        ["entry", "bar", "channelID"], kind="mergesort"
    ).reset_index(drop=True)
    return _coerce_matched_frame(matched)


def _numeric_path_key(path):
    name_parts = re.split(r"(\d+)", Path(path).name.lower())
    natural_name = tuple(int(part) if part.isdigit() else part for part in name_parts)
    return natural_name, str(path)


def load_synchronized_files(paths, bars=tuple(range(16)), max_files=None):
    """Read synchronized files in deterministic natural filename order."""
    sorted_paths = sorted(list(paths), key=_numeric_path_key)
    if not sorted_paths:
        raise ValueError("paths must contain at least one synchronized ROOT file")
    if max_files is not None:
        if max_files < 0:
            raise ValueError("max_files must be non-negative")
        sorted_paths = sorted_paths[:max_files]
    if not sorted_paths:
        return _empty_matched_frame()

    frames = [read_synchronized_file(path, bars=bars) for path in sorted_paths]
    if all(frame.empty for frame in frames):
        return _empty_matched_frame()
    return pd.concat(frames, ignore_index=True)

from Q_Walk.qwalk_channels import exact_four_channel_topology

def read_synchronized_topology_file(path, upstream_bar, downstream_bar):
    topology = exact_four_channel_topology(upstream_bar, downstream_bar)
    with uproot.open(path) as root_file:
        hits = _read_tofhir_hits(root_file["data"])
        mcp = _read_mcp_events(root_file["MCP"])

    events = select_exact_tofhir_topology(hits, topology)
    if events.empty:
        return _empty_matched_frame()

    ch192 = events.loc[events["channelID"].eq(TRIGGER_CHANNEL), ["entry", "time"]].rename(columns={"time": "ch192_time"})
    signal = events.loc[events["channelID"].ne(TRIGGER_CHANNEL)].copy()
    signal = signal.merge(ch192, on="entry", how="inner", validate="many_to_one")

    matched = match_mcp_events(signal, mcp)
    if matched.empty:
        return _empty_matched_frame()

    file_id = Path(path).stem
    matched["file_id"] = file_id
    matched["event_id"] = matched["entry"].map(lambda entry: f"{file_id}:{entry}")
    matched["segment"] = matched["segment"].astype(np.int64)
    matched["bar"] = upstream_bar
    matched = matched.sort_values(
        ["entry", "channelID"], kind="mergesort"
    ).reset_index(drop=True)
    return _coerce_matched_frame(matched)


def count_topology_candidates(path, upstream_bar, downstream_bar):
    topology = exact_four_channel_topology(upstream_bar, downstream_bar)
    with uproot.open(path) as root_file:
        hits = _read_tofhir_hits(root_file["data"])
        mcp = _read_mcp_events(root_file["MCP"])
    events = select_exact_tofhir_topology(hits, topology)
    if events.empty:
        return 0
    signal = events.loc[events["channelID"].ne(TRIGGER_CHANNEL)].copy()
    matched = match_mcp_events(signal, mcp)
    return len(matched["entry"].unique())

def load_synchronized_topology_files(paths, upstream_bar, downstream_bar, max_files=None):
    sorted_paths = sorted(list(paths), key=_numeric_path_key)
    if not sorted_paths:
        raise ValueError("paths must contain at least one synchronized ROOT file")
    if max_files is not None:
        sorted_paths = sorted_paths[:max_files]
    if not sorted_paths:
        return _empty_matched_frame()

    frames = [read_synchronized_topology_file(path, upstream_bar, downstream_bar) for path in sorted_paths]
    if all(frame.empty for frame in frames):
        return _empty_matched_frame()
    return pd.concat(frames, ignore_index=True)
