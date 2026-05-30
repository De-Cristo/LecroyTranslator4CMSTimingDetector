import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


fake_matplotlib = types.ModuleType("matplotlib")
fake_matplotlib.use = lambda *_args, **_kwargs: None
sys.modules.setdefault("matplotlib", fake_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("uproot", types.ModuleType("uproot"))
sys.modules.setdefault("awkward", types.ModuleType("awkward"))

fake_bar_helpers = types.ModuleType("bar_helpers")
fake_bar_helpers.find_data_tree = lambda *_args, **_kwargs: None
fake_bar_helpers.gauss = lambda *_args, **_kwargs: None
fake_bar_helpers.log = lambda *_args, **_kwargs: None
sys.modules.setdefault("bar_helpers", fake_bar_helpers)

fake_bar_processing = types.ModuleType("bar_processing")
fake_bar_processing._mcp_internal_dt_selector = lambda *_args, **_kwargs: ([], None)
sys.modules.setdefault("bar_processing", fake_bar_processing)

fake_bar_plotting = types.ModuleType("bar_plotting")
fake_bar_plotting.plot_t_diff = lambda *_args, **_kwargs: None
sys.modules.setdefault("bar_plotting", fake_bar_plotting)

fake_ch192 = types.ModuleType("ch192_vs_trigger")
fake_ch192.build_mcp_map = lambda *_args, **_kwargs: {}
fake_ch192.extract_ch_times = lambda *_args, **_kwargs: {}
fake_ch192.detect_segments = lambda *_args, **_kwargs: []
fake_ch192._process_one_file = lambda *_args, **_kwargs: None
fake_ch192._gauss_fit_hist = lambda *_args, **_kwargs: None
sys.modules.setdefault("ch192_vs_trigger", fake_ch192)

import ch192_vs_trigger_lowess


class LowessHelperTest(unittest.TestCase):
    def test_lowess_preserves_linear_trend(self):
        x = np.linspace(0.0, 10.0, 21)
        y = 3.0 + 2.0 * x

        yhat = ch192_vs_trigger_lowess._lowess_smooth(
            x, y, frac=0.3, it=0, delta=0.0
        )

        self.assertEqual(len(yhat), len(y))
        self.assertTrue(np.allclose(yhat, y, atol=1e-8))

    def test_lowess_downweights_outlier_with_robust_iterations(self):
        x = np.linspace(0.0, 10.0, 21)
        y = 1.5 * x
        y[10] += 8.0

        yhat = ch192_vs_trigger_lowess._lowess_smooth(
            x, y, frac=0.7, it=3, delta=0.0
        )

        self.assertTrue(math.isfinite(yhat[10]))
        self.assertLess(abs(yhat[10] - 1.5 * x[10]), 2.0)
        self.assertLess(abs(yhat[10] - 1.5 * x[10]), abs(y[10] - 1.5 * x[10]))

    def test_validation_fit_uses_fixed_20_bins(self):
        self.assertEqual(ch192_vs_trigger_lowess._validation_fit_nbins(), 20)

    def test_force_unit_slope_makes_fit_slope_one(self):
        trigger = np.linspace(0.0, 10.0, 21)
        corrected = 5.0 + 1.02 * trigger
        fit_mask = np.ones(len(trigger), dtype=bool)

        adjusted = ch192_vs_trigger_lowess._force_unit_slope(
            corrected, trigger, fit_mask
        )

        m, _ = np.polyfit(trigger[fit_mask], adjusted[fit_mask], 1)
        self.assertAlmostEqual(m, 1.0, places=10)

    def test_stage_builder_returns_lowess_then_unit_slope(self):
        trigger = np.linspace(0.0, 10.0, 21)
        predicted = 5.0 + 1.02 * trigger
        lowess_resid = 0.2 * np.sin(trigger)
        fit_mask = np.ones(len(trigger), dtype=bool)

        lowess_only, final = ch192_vs_trigger_lowess._build_corrected_stages(
            predicted, lowess_resid, trigger, fit_mask
        )

        self.assertTrue(np.allclose(lowess_only, predicted - lowess_resid, atol=1e-12))
        m_lowess, _ = np.polyfit(trigger[fit_mask], lowess_only[fit_mask], 1)
        m_final, _ = np.polyfit(trigger[fit_mask], final[fit_mask], 1)
        self.assertGreater(abs(m_lowess - 1.0), 1e-3)
        self.assertAlmostEqual(m_final, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
