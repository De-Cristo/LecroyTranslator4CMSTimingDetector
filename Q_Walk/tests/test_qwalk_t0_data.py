from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from Q_Walk.qwalk_channels import (
    CMSNOTE_WALK_COEFFICIENTS_NS,
    FIXED_DOWNSTREAM_PAIR,
    REFERENCE_CHANNELS,
    UPSTREAM_BAR_CHANNELS,
    reference_topology,
)
from Q_Walk import qwalk_t0_data
from Q_Walk.qwalk_t0_data import (
    MATCHED_COLUMNS,
    assign_scope_segments,
    load_synchronized_files,
    match_mcp_events,
    read_synchronized_file,
    reconstruct_reference_delta_t,
    robust_centered_line,
    select_exact_reference_topology,
)


@pytest.mark.parametrize(
    ("bar", "expected_channels"),
    [
        (0, (141, 145)),
        (1, (143, 144)),
        (2, (142, 146)),
        (3, (138, 147)),
        (4, (140, 148)),
        (5, (136, 151)),
        (6, (139, 149)),
        (7, (133, 154)),
        (8, (137, 150)),
        (9, (135, 155)),
        (10, (134, 156)),
        (11, (130, 159)),
        (12, (131, 158)),
        (13, (128, 152)),
        (14, (129, 153)),
        (15, (132, 157)),
    ],
)
def test_upstream_bar_channel_mapping(bar, expected_channels):
    assert UPSTREAM_BAR_CHANNELS[bar] == expected_channels


def test_reference_channel_metadata():
    assert FIXED_DOWNSTREAM_PAIR == (234, 243)
    assert REFERENCE_CHANNELS == (133, 137, 150, 154)


@pytest.mark.parametrize(
    ("channel", "expected_coefficients"),
    [
        (133, (0.0796, 238.3, -21560.0)),
        (154, (-0.5155, 422.6, -55830.0)),
        (137, (0.1218, 198.5, -15230.0)),
        (150, (-0.2292, 388.9, -51320.0)),
    ],
)
def test_cmsnote_walk_coefficients(channel, expected_coefficients):
    assert CMSNOTE_WALK_COEFFICIENTS_NS[channel] == expected_coefficients


def test_reference_topology_includes_trigger_and_downstream_pair():
    assert reference_topology(8) == frozenset({137, 150, 192, 234, 243})


def test_match_mcp_events_uses_mcp_index_as_data_entry():
    events = pd.DataFrame({"entry": [10, 11, 12], "value": [1, 2, 3]})
    mcp = pd.DataFrame({"entry": [12, 10], "mcp_peak_time": [8.0, 9.0]})

    matched = match_mcp_events(events, mcp)

    assert matched["entry"].tolist() == [10, 12]
    assert matched["mcp_peak_time"].tolist() == [9.0, 8.0]


def test_match_mcp_events_requires_unique_mcp_entries():
    events = pd.DataFrame({"entry": [10, 10]})
    mcp = pd.DataFrame({"entry": [10, 10], "mcp_peak_time": [8.0, 9.0]})

    with pytest.raises(pd.errors.MergeError, match="many-to-one"):
        match_mcp_events(events, mcp)


def test_reference_delta_t_removes_clock_relation_and_mcp_trigger_walk():
    trigger = np.arange(20, dtype=float) * 1_000.0
    ch192 = 50_000.0 + 1.00001 * trigger
    frame = pd.DataFrame(
        {
            "file_id": "f",
            "segment": 0,
            "mcp_trigger_time": trigger,
            "mcp_peak_time": trigger + 200.0,
            "ch192_time": ch192,
            "time": ch192 + 700.0,
        }
    )

    result = reconstruct_reference_delta_t(frame, lowess_frac=0.3)

    assert np.allclose(result["delta_t_ref_ps"], 500.0, atol=1e-6)


def test_robust_centered_line_rejects_outlier_when_mad_is_zero():
    x = np.full(5, 7.0)
    y = np.array([11.0, 11.0, 11.0, 11.0, 99.0])

    prediction, residual, mask, slope = robust_centered_line(x, y)

    assert slope == 0.0
    assert mask.tolist() == [True, True, True, True, False]
    assert np.allclose(prediction, 11.0)
    assert np.allclose(residual, [0.0, 0.0, 0.0, 0.0, 88.0])


def test_robust_centered_line_predicts_in_relative_timestamp_coordinates():
    offset = 1.0e20
    step = np.spacing(offset) * 8.0
    x = offset + np.arange(20, dtype=float) * step
    y = 3.0 * offset + (x - offset)

    prediction, residual, mask, slope = robust_centered_line(x, y)

    assert slope == pytest.approx(1.0, abs=1e-12)
    assert mask.all()
    assert np.array_equal(prediction, y)
    assert np.array_equal(residual, np.zeros_like(y))


def test_robust_centered_line_excludes_nonfinite_pairs_without_infinities():
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    y = np.array([10.0, 12.0, 14.0, np.inf, 16.0])

    prediction, residual, mask, slope = robust_centered_line(x, y)

    assert slope == pytest.approx(2.0)
    assert mask.tolist() == [True, True, False, False, True]
    assert not np.isinf(prediction).any()
    assert not np.isinf(residual).any()
    assert np.allclose(residual[mask], 0.0)


def test_robust_centered_line_handles_no_finite_pairs_explicitly():
    prediction, residual, mask, slope = robust_centered_line(
        [np.nan, np.inf], [1.0, np.nan]
    )

    assert np.isnan(prediction).all()
    assert np.isnan(residual).all()
    assert not mask.any()
    assert np.isnan(slope)


def _reference_frame(file_id, segment, trigger, ch192):
    trigger = np.asarray(trigger, dtype=float)
    ch192 = np.asarray(ch192, dtype=float)
    return pd.DataFrame(
        {
            "file_id": file_id,
            "segment": segment,
            "mcp_trigger_time": trigger,
            "mcp_peak_time": trigger + 200.0,
            "ch192_time": ch192,
            "time": ch192 + 700.0,
        }
    )


def _duplicated_reference_frame(unique_event_count, include_event_id=True):
    trigger = 1.0e11 + np.arange(unique_event_count, dtype=float) * 1_000.0
    ch192 = 1.0e13 + 1.00001 * (trigger - trigger[0])
    events = _reference_frame("duplicated", 0, trigger, ch192)
    events["entry"] = np.arange(unique_event_count)
    if include_event_id:
        events["event_id"] = [f"duplicated:{entry}" for entry in events["entry"]]
    hits = events.loc[events.index.repeat(2)].reset_index(drop=True)
    hits["channelID"] = np.tile([137, 150], unique_event_count)
    return hits


def _realistic_duplicated_reference_frame(unique_event_count, outliers=None):
    trigger = 1.0e11 + np.arange(unique_event_count, dtype=float) * 20_000.0
    phase = np.linspace(0.0, 3.0 * np.pi, unique_event_count)
    curvature = 55.0 * np.sin(phase)
    noise_pattern = np.array([0.0, 3.0, -2.0, 4.0, -3.0])
    noise = np.resize(noise_pattern, unique_event_count)
    clean_ch192 = 1.0e13 + 1.00001 * (trigger - trigger[0]) + curvature
    measured_ch192 = clean_ch192 + noise
    for event_position, shift in (outliers or {}).items():
        measured_ch192[event_position] += shift

    events = _reference_frame("realistic", 2, trigger, measured_ch192)
    events["time"] = clean_ch192 + 700.0
    events["entry"] = np.arange(unique_event_count)
    events["event_id"] = [f"realistic:{entry}" for entry in events["entry"]]
    hits = events.loc[events.index.repeat(2)].reset_index(drop=True)
    hits["channelID"] = np.tile([137, 150], unique_event_count)
    return hits


@pytest.mark.parametrize("lowess_n_iter", [-1, 1.5, "3", True])
def test_reference_reconstruction_rejects_invalid_lowess_iterations(
    lowess_n_iter,
):
    trigger = 1.0e11 + np.arange(5, dtype=float) * 1_000.0
    frame = _reference_frame("short", 0, trigger, 1.0e13 + trigger)

    with pytest.raises(ValueError, match="lowess_n_iter"):
        reconstruct_reference_delta_t(frame, lowess_n_iter=lowess_n_iter)


@pytest.mark.parametrize("lowess_frac", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_reference_reconstruction_rejects_invalid_lowess_fraction_for_short_group(
    lowess_frac,
):
    trigger = 1.0e11 + np.arange(5, dtype=float) * 1_000.0
    frame = _reference_frame("short", 0, trigger, 1.0e13 + trigger)

    with pytest.raises(ValueError, match="lowess_frac"):
        reconstruct_reference_delta_t(frame, lowess_frac=lowess_frac)


def test_reference_reconstruction_isolates_file_segments():
    trigger = np.arange(12, dtype=float) * 1_000.0
    first = _reference_frame("f", 0, trigger, 50_000.0 + 1.1 * trigger)
    second = _reference_frame("f", 1, trigger, -70_000.0 + 0.9 * trigger)
    interleaved = pd.concat([first, second], ignore_index=True).sort_index(
        key=lambda index: index % len(trigger), kind="mergesort"
    )

    result = reconstruct_reference_delta_t(interleaved, lowess_frac=0.3)

    assert np.allclose(result["delta_t_ref_ps"], 500.0, atol=1e-6)
    assert result.groupby("segment")["ch192_line_slope"].first().to_dict() == (
        {0: pytest.approx(1.1), 1: pytest.approx(0.9)}
    )
    assert set(result["reference_alignment_status"]) == {"ok"}


def test_reference_reconstruction_reports_insufficient_alignment_points():
    trigger = np.arange(9, dtype=float) * 1_000.0
    frame = _reference_frame("short", 3, trigger, 25_000.0 + trigger)

    result = reconstruct_reference_delta_t(frame, lowess_frac=0.3)

    assert (result["reference_alignment_status"] == "insufficient_alignment_points").all()
    assert np.array_equal(result["ch192_lowess_correction"], np.zeros(9))
    assert np.allclose(result["expected_ch192_time"], frame["ch192_time"])
    assert np.allclose(result["delta_t_ref_ps"], 500.0)


def test_reference_reconstruction_counts_unique_events_not_hit_rows():
    frame = _duplicated_reference_frame(5)
    frame["entry"] = np.arange(len(frame))

    result = reconstruct_reference_delta_t(frame, lowess_frac=0.3)

    assert (
        result["reference_alignment_status"]
        == "insufficient_alignment_points"
    ).all()
    assert np.array_equal(result["ch192_lowess_correction"], np.zeros(10))


@pytest.mark.parametrize(
    "column", ["mcp_trigger_time", "mcp_peak_time", "ch192_time"]
)
def test_reference_reconstruction_rejects_inconsistent_event_alignment_fields(
    column,
):
    frame = _duplicated_reference_frame(10)
    frame.loc[1, column] += 20.0

    with pytest.raises(ValueError, match=rf"{column}.*event_id"):
        reconstruct_reference_delta_t(frame, lowess_frac=0.3)


def test_reference_reconstruction_uses_entry_and_broadcasts_event_diagnostics():
    frame = _duplicated_reference_frame(10, include_event_id=False)

    result = reconstruct_reference_delta_t(frame, lowess_frac=0.3)

    diagnostic_columns = [
        "expected_ch192_time",
        "ch192_line_slope",
        "ch192_alignment_residual",
        "ch192_alignment_inlier",
        "ch192_lowess_correction",
        "reference_alignment_status",
    ]
    for column in diagnostic_columns:
        values = result[column].to_numpy().reshape(-1, 2)
        assert np.array_equal(values[:, 0], values[:, 1])
    assert set(result["reference_alignment_status"]) == {"ok"}
    assert np.allclose(result["delta_t_ref_ps"], 500.0, atol=1e-6)


def test_reference_reconstruction_threshold_uses_realistic_unique_event_count():
    five_events = _realistic_duplicated_reference_frame(5)
    ten_events = _realistic_duplicated_reference_frame(10)

    short_result = reconstruct_reference_delta_t(five_events, lowess_frac=0.4)
    full_result = reconstruct_reference_delta_t(ten_events, lowess_frac=0.4)

    assert set(short_result["reference_alignment_status"]) == {
        "insufficient_alignment_points"
    }
    assert np.array_equal(
        short_result["ch192_lowess_correction"], np.zeros(10)
    )
    assert set(full_result["reference_alignment_status"]) == {"ok"}
    assert np.isfinite(full_result["delta_t_ref_ps"]).all()
    assert np.max(np.abs(full_result["ch192_lowess_correction"])) > 10.0


def test_lowess_bisquare_handles_moderate_outliers_kept_by_line_fit():
    frame = _realistic_duplicated_reference_frame(
        40, outliers={12: 75.0, 27: -75.0}
    )

    plain = reconstruct_reference_delta_t(
        frame, lowess_frac=0.2, lowess_n_iter=0
    )
    robust = reconstruct_reference_delta_t(
        frame, lowess_frac=0.2, lowess_n_iter=3
    )

    event_rows = np.arange(0, len(frame), 2)
    assert robust.loc[event_rows[[12, 27]], "ch192_alignment_inlier"].all()
    plain_error = np.mean(np.abs(plain.loc[event_rows, "delta_t_ref_ps"] - 500.0))
    robust_error = np.mean(
        np.abs(robust.loc[event_rows, "delta_t_ref_ps"] - 500.0)
    )
    assert robust_error < 0.8 * plain_error


def test_reference_reconstruction_preserves_row_order_and_index():
    trigger = np.arange(12, dtype=float) * 1_000.0
    frame = _reference_frame("ordered", 0, trigger, 40_000.0 + trigger)
    frame["row_marker"] = np.arange(len(frame))[::-1]
    frame = frame.iloc[[7, 1, 9, 0, 11, 4, 2, 10, 3, 8, 5, 6]]
    frame.index = pd.Index([4, 4, 2, 8, 1, 1, 9, 3, 3, 7, 6, 0], name="source")

    result = reconstruct_reference_delta_t(frame, lowess_frac=0.3)

    assert result.index.equals(frame.index)
    assert result["row_marker"].tolist() == frame["row_marker"].tolist()


def test_reference_reconstruction_is_finite_for_constant_trigger_segment():
    trigger = np.full(12, 8.0e19)
    ch192 = np.full(12, 9.0e19)
    frame = _reference_frame("constant", 0, trigger, ch192)

    result = reconstruct_reference_delta_t(frame, lowess_frac=0.3)

    diagnostic_columns = [
        "expected_ch192_time",
        "ch192_line_slope",
        "ch192_alignment_residual",
        "ch192_lowess_correction",
        "delta_t_ref_ps",
    ]
    assert np.isfinite(result[diagnostic_columns].to_numpy()).all()
    assert set(result["reference_alignment_status"]) == {
        "degenerate_alignment_axis"
    }
    assert np.array_equal(
        result["ch192_lowess_correction"], np.zeros(len(result))
    )


def test_reference_reconstruction_robustly_smooths_nonlinear_clock_residual():
    trigger = np.arange(60, dtype=float) * 1_000.0
    clock_curve = 80.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(trigger)))
    clean_ch192 = 50_000.0 + 1.00001 * trigger + clock_curve
    measured_ch192 = clean_ch192.copy()
    measured_ch192[25] += 1_000.0
    frame = _reference_frame("curved", 0, trigger, measured_ch192)
    frame["time"] = clean_ch192 + 700.0

    result = reconstruct_reference_delta_t(frame, lowess_frac=0.25)

    assert not result.loc[25, "ch192_alignment_inlier"]
    assert np.max(np.abs(result["ch192_lowess_correction"])) > 20.0
    assert np.max(np.abs(result["delta_t_ref_ps"] - 500.0)) < 12.0


def test_lowess_local_fits_use_fixed_size_sorted_neighbor_windows(monkeypatch):
    x = np.arange(40, dtype=float)
    y = np.sin(x / 5.0) + 0.01 * (x % 3)
    window_sizes = []
    real_predict = qwalk_t0_data._weighted_local_linear_prediction

    def recording_predict(local_x, local_y, weights, x_eval):
        window_sizes.append(len(local_x))
        return real_predict(local_x, local_y, weights, x_eval)

    monkeypatch.setattr(
        qwalk_t0_data, "_weighted_local_linear_prediction", recording_predict
    )

    qwalk_t0_data._robust_lowess_at_training_points(x, y, frac=0.2, n_iter=1)

    assert window_sizes
    assert max(window_sizes) == 8


def test_lowess_windows_match_nearest_timestamps_for_irregular_spacing(monkeypatch):
    x = np.array([0.0, 1.0, 2.0, 50.0, 51.0, 52.0, 53.0, 200.0, 1_000.0, 1_001.0])
    y = np.sin(x / 20.0)
    observed_windows = []
    real_predict = qwalk_t0_data._weighted_local_linear_prediction

    def recording_predict(local_x, local_y, weights, x_eval):
        observed_windows.append(local_x.copy())
        return real_predict(local_x, local_y, weights, x_eval)

    monkeypatch.setattr(
        qwalk_t0_data, "_weighted_local_linear_prediction", recording_predict
    )

    qwalk_t0_data._robust_lowess_at_training_points(x, y, frac=0.4, n_iter=0)

    expected_windows = []
    indices = np.arange(len(x))
    for x_eval in x:
        nearest = np.lexsort((indices, np.abs(x - x_eval)))[:4]
        expected_windows.append(x[np.sort(nearest)])
    assert len(observed_windows) == len(expected_windows)
    for observed, expected in zip(observed_windows, expected_windows):
        assert np.array_equal(observed, expected)


def test_exact_bar8_topology_rejects_missing_extra_and_duplicate_hits():
    hits = pd.DataFrame(
        {
            "entry": [1] * 5 + [2] * 4 + [3] * 6 + [4] * 6,
            "channelID": [
                137,
                150,
                192,
                234,
                243,
                137,
                150,
                192,
                234,
                137,
                150,
                192,
                234,
                243,
                244,
                137,
                150,
                192,
                234,
                243,
                243,
            ],
        }
    )

    selected = select_exact_reference_topology(hits, 8)

    assert selected["entry"].unique().tolist() == [1]
    assert selected["channelID"].tolist() == [137, 150, 192, 234, 243]


def test_scope_segments_split_at_large_trigger_gap():
    trigger = np.array([0.0, 10.0, 20.0, 10_000.0, 10_010.0])

    assert assign_scope_segments(trigger, gap_factor=500.0).tolist() == [0, 0, 0, 1, 1]


def test_scope_segments_split_when_trigger_clock_resets():
    trigger = np.array([0.0, 10.0, 20.0, 5.0, 15.0, 25.0])

    assert assign_scope_segments(trigger, gap_factor=500.0).tolist() == [0, 0, 0, 1, 1, 1]


def test_scope_segments_detect_reset_across_missing_timestamps():
    trigger = [np.nan, 100.0, 200.0, np.nan, np.inf, 10.0, 20.0, np.nan]
    assert assign_scope_segments(trigger).tolist() == [-1, 0, 0, -1, -1, 1, 1, -1]


def test_reference_rejects_clock_failure_without_signal_timing_selection():
    trigger = np.arange(100, dtype=float) * 1e9
    bad = _reference_frame("bad", 0, trigger, trigger + np.tile([-1e7, 1e7], 50))
    good = _reference_frame("good", 0, trigger, trigger + 100)
    result = reconstruct_reference_delta_t(pd.concat([bad, good], ignore_index=True))
    rejected = result.file_id.eq("bad")
    assert result.loc[rejected, "delta_t_ref_ps"].isna().all()
    assert result.loc[rejected, "reference_alignment_status"].eq("rejected_clock_quality").all()
    assert (result.loc[rejected, "clock_line_scaled_mad_ps"] > 1000).all()
    assert np.isfinite(result.loc[~rejected, "delta_t_ref_ps"]).all()
    good["time"] += 1e12
    shifted = reconstruct_reference_delta_t(good)
    assert np.isfinite(shifted.delta_t_ref_ps).all()


def test_clock_quality_threshold_can_only_be_positive_or_explicitly_disabled():
    frame = _reference_frame("a", 0, np.arange(10.0), np.arange(10.0))
    for threshold in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="clock_max_mad_ps"):
            reconstruct_reference_delta_t(frame, clock_max_mad_ps=threshold)


@pytest.mark.parametrize("trigger", [[], [10.0]])
def test_scope_segments_reject_input_without_a_positive_spacing(trigger):
    with pytest.raises(ValueError, match="positive spacing"):
        assign_scope_segments(trigger)


def test_load_synchronized_files_sorts_numeric_filename_components(monkeypatch):
    seen = []

    def fake_read(path, bars):
        seen.append(Path(path).name)
        return pd.DataFrame({"source": [Path(path).name]})

    monkeypatch.setattr(qwalk_t0_data, "read_synchronized_file", fake_read)

    loaded = load_synchronized_files(
        ["run4399_file10.root", "run4399_file2.root", "run4399_file1.root"],
        bars=(8,),
    )

    assert seen == ["run4399_file1.root", "run4399_file2.root", "run4399_file10.root"]
    assert loaded["source"].tolist() == seen


def test_load_synchronized_files_applies_max_files_after_sorting(monkeypatch):
    seen = []

    def fake_read(path, bars):
        seen.append(Path(path).name)
        return pd.DataFrame({"source": [Path(path).name]})

    monkeypatch.setattr(qwalk_t0_data, "read_synchronized_file", fake_read)

    load_synchronized_files(
        ["sync10.root", "sync2.root", "sync1.root"], bars=(8,), max_files=2
    )

    assert seen == ["sync1.root", "sync2.root"]


def test_load_synchronized_files_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        load_synchronized_files([])


def test_matched_columns_have_the_required_order():
    assert MATCHED_COLUMNS == (
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


EXPECTED_MATCHED_DTYPES = {
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


def _assert_matched_schema_and_dtypes(frame):
    assert tuple(frame.columns) == MATCHED_COLUMNS
    assert frame.dtypes.astype(str).to_dict() == EXPECTED_MATCHED_DTYPES


class FakeTree:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def arrays(self, branches, library):
        self.calls.append((tuple(branches), library))
        return self.result


class FakeRootFile:
    def __init__(self, data_tree, mcp_tree):
        self.trees = {"data": data_tree, "MCP": mcp_tree}
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __getitem__(self, key):
        self.requested.append(key)
        return self.trees[key]


def _synthetic_data_hits(exact_entries=range(5)):
    rows = []
    channels = (137, 150, 192, 234, 243)
    for entry in range(5):
        event_channels = channels if entry in exact_entries else (*channels, 244)
        for subentry, channel in enumerate(event_channels):
            rows.append(
                {
                    "entry": entry,
                    "subentry": subentry,
                    "step1": 0.0,
                    "step2": 113621.0,
                    "tot": 0.0,
                    "t1coarse": 0,
                    "t1fine": 10,
                    "t2coarse": 1,
                    "t2fine": 10,
                    "time": 1_000.0 + 100.0 * entry + channel,
                    "channelID": channel,
                    "energy": 200.0 + channel,
                    "tacID": 0,
                }
            )
    return pd.DataFrame(rows).set_index(["entry", "subentry"])


def test_read_synchronized_file_segments_complete_mcp_before_sparse_selection(
    monkeypatch,
):
    data_token = object()
    data_tree = FakeTree(data_token)
    mcp_tree = FakeTree(
        {
            "index": np.arange(5),
            "peak_time": np.array([200.0, 210.0, 220.0, 10_200.0, 10_210.0]),
            "trigger_time": np.array([0.0, 10.0, 20.0, 10_000.0, 10_010.0]),
            "peak_amp": np.arange(40.0, 45.0),
            "phi_peak": np.arange(5.0),
        }
    )
    monkeypatch.setattr(
        qwalk_t0_data.uproot,
        "open",
        lambda path: FakeRootFile(data_tree, mcp_tree),
    )
    monkeypatch.setattr(
        qwalk_t0_data.ak,
        "to_dataframe",
        lambda arrays: _synthetic_data_hits(exact_entries={0, 4}),
    )

    result = read_synchronized_file("sparse.root", bars=(8,))

    assert result.groupby("entry")["segment"].first().to_dict() == {0: 0, 4: 1}


def test_all_bar_topology_selection_groups_events_once(monkeypatch):
    hits = _synthetic_data_hits().reset_index()
    real_groupby = pd.DataFrame.groupby
    entry_groupbys = 0

    def counting_groupby(frame, by=None, *args, **kwargs):
        nonlocal entry_groupbys
        if by == "entry":
            entry_groupbys += 1
        return real_groupby(frame, by=by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", counting_groupby)

    result = qwalk_t0_data._select_reference_hits(hits, bars=range(16))

    assert entry_groupbys == 1
    assert result["bar"].unique().tolist() == [8]
    assert len(result) == 10


def test_read_synchronized_file_returns_typed_empty_frame(monkeypatch):
    data_token = object()
    mcp_tree = FakeTree(
        {
            "index": np.arange(5),
            "peak_time": np.arange(200.0, 250.0, 10.0),
            "trigger_time": np.array([0.0, 10.0, 20.0, 10_000.0, 10_010.0]),
            "peak_amp": np.arange(40.0, 45.0),
            "phi_peak": np.arange(5.0),
        }
    )
    monkeypatch.setattr(
        qwalk_t0_data.uproot,
        "open",
        lambda path: FakeRootFile(FakeTree(data_token), mcp_tree),
    )
    monkeypatch.setattr(
        qwalk_t0_data.ak,
        "to_dataframe",
        lambda arrays: _synthetic_data_hits(exact_entries=set()),
    )

    result = read_synchronized_file("empty.root", bars=(8,))

    assert result.empty
    _assert_matched_schema_and_dtypes(result)


def test_load_synchronized_files_preserves_typed_schema_when_all_files_are_empty(
    monkeypatch,
):
    monkeypatch.setattr(
        qwalk_t0_data,
        "read_synchronized_file",
        lambda path, bars: pd.DataFrame(columns=MATCHED_COLUMNS),
    )

    result = load_synchronized_files(["sync2.root", "sync1.root"], bars=(8,))

    assert result.empty
    _assert_matched_schema_and_dtypes(result)


def test_read_synchronized_file_transforms_and_matches_fake_root_data(monkeypatch):
    data_token = object()
    data_tree = FakeTree(data_token)
    mcp_tree = FakeTree(
        {
            "index": np.array([4, 0, 3, 1, 2]),
            "peak_time": np.array([10_210.0, 200.0, 10_200.0, 210.0, 220.0]),
            "trigger_time": np.array([10_010.0, 0.0, 10_000.0, 10.0, 20.0]),
            "peak_amp": np.array([44.0, 40.0, 43.0, 41.0, 42.0]),
            "phi_peak": np.array([4.0, 0.0, 3.0, 1.0, 2.0]),
        }
    )
    fake_file = FakeRootFile(data_tree, mcp_tree)
    monkeypatch.setattr(qwalk_t0_data.uproot, "open", lambda path: fake_file)
    monkeypatch.setattr(
        qwalk_t0_data.ak,
        "to_dataframe",
        lambda arrays: _synthetic_data_hits() if arrays is data_token else None,
    )

    result = read_synchronized_file("/input/run4399_sync2.root", bars=(8,))

    assert fake_file.requested == ["data", "MCP"]
    assert data_tree.calls == [(qwalk_t0_data.TOFHIR_BRANCHES, "ak")]
    assert mcp_tree.calls == [(qwalk_t0_data.MCP_BRANCHES, "np")]
    assert tuple(result.columns) == MATCHED_COLUMNS
    assert len(result) == 10
    assert result["channelID"].tolist() == [137, 150] * 5
    assert result["bar"].tolist() == [8] * 10
    assert result["file_id"].unique().tolist() == ["run4399_sync2"]
    assert result["event_id"].unique().tolist() == [
        "run4399_sync2:0",
        "run4399_sync2:1",
        "run4399_sync2:2",
        "run4399_sync2:3",
        "run4399_sync2:4",
    ]
    assert result.groupby("entry")["segment"].first().tolist() == [0, 0, 0, 1, 1]
    assert np.allclose(result["delta_t12_ps"], 6_250.0)
    assert np.allclose(result["delta_i_ua"], 7.8125)
    assert np.allclose(result["slope_ua_per_ns"], 1.25)
    assert np.allclose(result["inv_slope_ns_per_ua"], 0.8)
    assert result.groupby("entry")["mcp_peak_amp"].first().tolist() == [40, 41, 42, 43, 44]
    expected_ch192 = [1_192.0, 1_292.0, 1_392.0, 1_492.0, 1_592.0]
    assert result.groupby("entry")["ch192_time"].first().tolist() == expected_ch192


def _write_synchronized_root(path, exact_entries):
    branch_values = {branch: [] for branch in qwalk_t0_data.TOFHIR_BRANCHES}
    for entry in range(5):
        channels = [137, 150, 192, 234, 243]
        if entry not in exact_entries:
            channels.append(244)
        n_hits = len(channels)
        event_values = {
            "step1": [0.0] * n_hits,
            "step2": [113621.0] * n_hits,
            "tot": [0.0] * n_hits,
            "t1coarse": [0] * n_hits,
            "t1fine": [10] * n_hits,
            "t2coarse": [1] * n_hits,
            "t2fine": [10] * n_hits,
            "time": [1_000.0 + 100.0 * entry + channel for channel in channels],
            "channelID": channels,
            "energy": [200.0 + channel for channel in channels],
            "tacID": [0] * n_hits,
        }
        for branch, values in event_values.items():
            branch_values[branch].append(values)

    with uproot.recreate(path) as root_file:
        root_file.mktree("data", {
            branch: ak.Array(values) for branch, values in branch_values.items()
        })
        root_file.mktree("MCP", {
            "index": np.arange(5, dtype=np.int64),
            "peak_time": np.array([200.0, 210.0, 220.0, 10_200.0, 10_210.0]),
            "trigger_time": np.array([0.0, 10.0, 20.0, 10_000.0, 10_010.0]),
            "peak_amp": np.arange(40.0, 45.0),
            "phi_peak": np.arange(5.0),
        })


def test_read_synchronized_file_from_temporary_root_preserves_contract(tmp_path):
    root_path = tmp_path / "run4399_sync2.root"
    _write_synchronized_root(root_path, exact_entries={0, 4})

    result = read_synchronized_file(root_path, bars=range(16))

    _assert_matched_schema_and_dtypes(result)
    assert result.groupby("entry")["segment"].first().to_dict() == {0: 0, 4: 1}
    keys = ["event_id", "bar", "channelID"]
    assert not result.duplicated(keys).any()
    ordering = list(
        result[["entry", "bar", "channelID"]].itertuples(index=False, name=None)
    )
    assert ordering == sorted(ordering)
    assert result["channelID"].tolist() == [137, 150, 137, 150]
