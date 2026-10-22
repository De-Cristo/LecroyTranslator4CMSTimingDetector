import numpy as np
import pandas as pd
import pytest
from Q_Walk import qwalk_core
from Q_Walk.qwalk_core import (
    decode_thresholds,
    reconstruct_time2,
    reconstruct_time2_safe,
    calculate_slope,
    walk_empirical_137,
    walk_slope_model_137,
    build_calibration_table,
    lookup_calibration,
    read_root_hits,
    select_notebook_hits,
    select_notebook_fit_hits,
    select_upstream_hits_with_fixed_downstream_bar,
    fit_slope_vs_q,
)


class FakeRootFile:
    def __init__(self, tree):
        self.tree = tree

    def __enter__(self):
        return {"data": self.tree}

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeTree:
    def arrays(self, branches, library):
        return object()


def test_notebook_threshold_and_slope_formulae():
    vth1, vth2, delta_i = decode_thresholds(np.array([113621.0]))
    assert (vth1[0], vth2[0]) == (10, 35)
    assert delta_i[0] == 7.8125
    assert calculate_slope(np.array([10_000.0]), np.array([11_000.0]), delta_i)[0] == 7.8125


def test_time2_wraparound_matches_notebook_formula():
    actual = reconstruct_time2(
        time1=np.array([1_000.0]),
        t1coarse=np.array([1023]),
        t2coarse=np.array([0]),
        calib_t1=np.array([100.0]),
        calib_t2=np.array([200.0]),
    )
    assert actual[0] == 7_350.0


def test_safe_time2_does_not_overflow_large_coarse_difference():
    actual = reconstruct_time2_safe(
        time1=np.array([1_000.0]),
        t1coarse=np.array([0]),
        t2coarse=np.array([12]),
        calib_t1=np.array([100.0]),
        calib_t2=np.array([200.0]),
    )
    assert actual[0] == 76_100.0


def test_notebook_and_safe_modes_match_without_overflow():
    args = {
        "time1": np.array([1_000.0]),
        "t1coarse": np.array([0]),
        "t2coarse": np.array([1]),
        "calib_t1": np.array([100.0]),
        "calib_t2": np.array([200.0]),
    }
    assert reconstruct_time2(**args)[0] == reconstruct_time2_safe(**args)[0]


def test_read_root_hits_dispatches_explicit_time2_modes(monkeypatch):
    flattened_hits = pd.DataFrame({
        "entry": [0],
        "subentry": [0],
        "channelID": [137],
        "tacID": [0],
        "t1fine": [10],
        "t2fine": [10],
        "t1coarse": [0],
        "t2coarse": [12],
        "time": [1_000.0],
    })
    monkeypatch.setattr(qwalk_core.uproot, "open", lambda path: FakeRootFile(FakeTree()))
    monkeypatch.setattr(qwalk_core.ak, "to_dataframe", lambda arrays: flattened_hits.copy())
    monkeypatch.setattr(qwalk_core, "reconstruct_time2", lambda *args: np.array([111.0]))
    monkeypatch.setattr(qwalk_core, "reconstruct_time2_safe", lambda *args: np.array([222.0]))

    notebook_hits = read_root_hits("hits.root", time2_mode="notebook")
    safe_hits = read_root_hits("hits.root", time2_mode="safe")

    assert notebook_hits["time2"].tolist() == [111.0]
    assert safe_hits["time2"].tolist() == [222.0]


def test_read_root_hits_rejects_invalid_time2_mode_before_io(monkeypatch):
    def fail_if_opened(path):
        raise AssertionError("invalid mode must be rejected before opening ROOT input")

    monkeypatch.setattr(qwalk_core.uproot, "open", fail_if_opened)
    with pytest.raises(ValueError, match="time2_mode"):
        read_root_hits("hits.root", time2_mode="invalid")


def test_documented_channel_137_walk_curves():
    assert np.isclose(walk_empirical_137(np.array([200.0]))[0], 0.73355)
    assert np.isclose(walk_slope_model_137(np.array([200.0]))[0], 0.727401, atol=1e-6)


def test_calibration_table_uses_channel_tac_and_fine_code():
    table = build_calibration_table(
        channel=np.array([137, 137, 137]),
        tac=np.array([0, 0, 1]),
        fine=np.array([10, 10, 10]),
        time=np.array([6_300.0, 6_500.0, 6_700.0]),
    )
    assert table[(137, 0, 10)] == 150.0
    values, missing = lookup_calibration(table, np.array([137, 137]), np.array([0, 3]), np.array([10, 10]))
    assert np.allclose(values, [150.0, 0.0])
    assert missing.tolist() == [False, True]


def test_notebook_selection_requires_four_hits_and_anchor_channel():
    hits = pd.DataFrame({
        "entry": [1, 1, 1, 1, 2, 2, 2, 2],
        "channelID": [133, 137, 150, 154, 140, 137, 150, 154],
        "subentry": [0, 1, 2, 3, 0, 1, 2, 3]
    })
    selected = select_notebook_hits(hits)
    assert selected["entry"].unique().tolist() == [1, 2]


def test_upstream_selection_requires_exact_fixed_downstream_four_hit_topology():
    hits = pd.DataFrame({
        "entry": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "subentry": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4],
        "channelID": [128, 152, 234, 243, 128, 152, 234, 242, 128, 152, 234, 243, 244],
    })
    selected = select_upstream_hits_with_fixed_downstream_bar(hits)
    assert selected["entry"].unique().tolist() == [1]
    assert selected["channelID"].tolist() == [128, 152]


def test_linear_slope_fit_recovers_known_coefficients():
    q = np.array([100.0, 200.0, 300.0])
    result = fit_slope_vs_q(q, 11.0 + 0.02 * q)
    assert np.allclose(result.linear_coefficients, [11.0, 0.02])


def test_local_trend_uses_elementwise_weights_not_a_dense_diagonal(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("local fit must not construct a dense diagonal weight matrix")

    monkeypatch.setattr(np, "diag", fail_if_called)
    q = np.linspace(100.0, 700.0, 1000)
    result = fit_slope_vs_q(q, 11.0 + 0.01 * q)
    assert np.isfinite(result.local_trend["slope"]).all()


def test_notebook_fit_hits_apply_channel_specific_q_and_slope_windows():
    hits = pd.DataFrame({
        "energy": [100.0, 100.1, 649.9, 650.0, 200.0],
        "slope_ua_per_ns": [12.0, 10.1, 23.9, 12.0, 25.0],
    })
    selected = select_notebook_fit_hits(hits, 137)
    assert selected.index.tolist() == [1, 2]
