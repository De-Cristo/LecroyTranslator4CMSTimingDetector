from Q_Walk.qwalk_extended_t0_reference import (
    DOWNSTREAM_CHANNELS,
    UPSTREAM_CHANNELS,
    build_topology_specs,
    selected_inv_q_edges,
)
import numpy as np


def test_extended_reference_specs_cover_requested_channels_once():
    specs = build_topology_specs()
    channels = [channel for spec in specs.values() for channel in spec["channels"]]
    assert sorted(channels) == sorted((*UPSTREAM_CHANNELS, *DOWNSTREAM_CHANNELS))
    assert len(channels) == len(set(channels))


def test_extended_reference_uses_expected_fixed_coincidence_bars():
    specs = build_topology_specs()
    upstream = [spec for spec in specs.values() if spec["module"] == "upstream"]
    downstream = [spec for spec in specs.values() if spec["module"] == "downstream"]
    assert {tuple(spec["coincidence_pair"]) for spec in upstream} == {(234, 243)}
    assert {tuple(spec["coincidence_pair"]) for spec in downstream} == {(137, 150)}


def test_selected_inverse_q_edges_cover_full_selected_q_range():
    edges = selected_inv_q_edges([100.0, 120.0, 400.0], 4)
    assert len(edges) == 5
    assert np.all(np.diff(edges) > 0.0)
    assert np.isclose(1.0 / edges[-1], 100.0)
    assert np.isclose(1.0 / edges[0], 400.0)
