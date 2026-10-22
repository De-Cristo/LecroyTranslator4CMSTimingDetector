"""Focused tests of the frozen profile-selection semantics."""
import numpy as np
import pandas as pd

from Q_Walk.qwalk_channels import UPSTREAM_BAR_CHANNELS
from Q_Walk.qwalk_run4076_peak_right_profile import _build_profile, _fit_linear_profile
from Q_Walk.qwalk_two_hit_pair_profile import select_pair_with_exactly_two_module_hits


def test_pair_selection_ignores_other_module_and_trigger_but_rejects_duplicates():
    events = [[137, 150], [137, 150, 192, 234, 243, 238],
              [137, 150, 133], [137, 137], [137, 150, 150]]
    hits = pd.DataFrame([(i, ch) for i, channels in enumerate(events) for ch in channels],
                        columns=["entry", "channelID"])
    selected = select_pair_with_exactly_two_module_hits(hits, (137, 150), UPSTREAM_BAR_CHANNELS)
    assert selected.entry.tolist() == [0, 0, 1, 1]
    assert selected.channelID.tolist() == [137, 150, 137, 150]


def test_profile_includes_last_edge_and_excludes_singleton_sem_from_fit():
    profile = _build_profile(np.array([0., 0., 1., 1., 3.]),
                             np.array([1., 3., 3., 5., 99.]), edges=[0., 1., 2., 3.])
    assert [p["count"] for p in profile] == [2, 2, 1]
    assert profile[-1]["q_mean_adc"] == 3.
    assert np.isnan(profile[-1]["slope_sem_ua_per_ns"])
    assert profile[0]["slope_sd_ua_per_ns"] == np.sqrt(2)
    assert profile[0]["slope_sem_ua_per_ns"] == 1.
    fit = _fit_linear_profile(profile)
    assert fit["profile_bins_used"] == 2
    np.testing.assert_allclose([fit["intercept_ua_per_ns"], fit["coefficient_ua_per_ns_per_adc"]], [2., 2.])
