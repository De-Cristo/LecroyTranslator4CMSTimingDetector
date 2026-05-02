import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _FakeDataFrame:
    def __init__(self, rows=None, columns=None):
        self._rows = [dict(r) for r in (rows or [])]
        if columns is not None:
            self.columns = list(columns)
        elif self._rows:
            self.columns = list(self._rows[0].keys())
        else:
            self.columns = []

    def sort_values(self, columns):
        self._rows.sort(key=lambda row: tuple(row.get(c) for c in columns))
        return self

    def reset_index(self, drop=True):
        return self

    @property
    def loc(self):
        return _FakeLoc(self)

    def __len__(self):
        return len(self._rows)


class _FakeLoc:
    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        row, col = key
        return self._df._rows[row][col]


fake_pd = types.ModuleType("pandas")
fake_pd.DataFrame = _FakeDataFrame
sys.modules.setdefault("pandas", fake_pd)
sys.modules.setdefault("awkward", types.ModuleType("awkward"))
sys.modules.setdefault("uproot", types.ModuleType("uproot"))

fake_matplotlib = types.ModuleType("matplotlib")
fake_matplotlib.use = lambda *_args, **_kwargs: None
sys.modules.setdefault("matplotlib", fake_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

fake_scipy_optimize = types.ModuleType("scipy.optimize")
fake_scipy_optimize.curve_fit = lambda *args, **kwargs: None
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.optimize", fake_scipy_optimize)

import simple_sync


class MatchedCsvFormatTest(unittest.TestCase):
    def test_build_matched_dataframe_restores_legacy_columns(self):
        rows = [
            {
                "entry": 7,
                "mcp_index": 7,
                "mcp_peak_time": 12300.0,
                "mcp_peak_amp": -0.5,
                "mcp_peak_sigma_ps": 42.0,
                "mcp_peak_phase": 50.0,
                "mcp_trigger_time": 12000.0,
                "mcp_trigger_offset_ps": 10.0,
                "root_time_ps": 999.0,
                "meta_file": "raw_C2_meta.csv",
                "peaks_file": "peaks.csv",
                "segment": 3,
                "trigger_ps_from_meta": 12000.0,
                "phi_peak_from_edge": 75.0,
                "phi_trigger_from_edge": 25.0,
                "t_ave_ps": 12250.0,
                "prev_edge_ps": 12225.0,
            }
        ]
        root_arrays = {
            "channelID": {7: "[192, 160]"},
            "t1coarse": {7: "[11, 22]"},
            "time": {7: "[999.0, 1001.0]"},
            "energy": {7: "[1.5, 2.5]"},
        }

        df = simple_sync.build_legacy_matched_dataframe(rows, root_arrays)

        self.assertEqual(list(df.columns), simple_sync.LEGACY_MATCHED_CSV_COLUMNS)
        self.assertEqual(df.loc[0, "channelID"], "[192, 160]")
        self.assertEqual(df.loc[0, "t1coarse"], "[11, 22]")
        self.assertEqual(df.loc[0, "time"], "[999.0, 1001.0]")
        self.assertEqual(df.loc[0, "energy"], "[1.5, 2.5]")
        self.assertEqual(df.loc[0, "t0_abs_ps"], 12250.0)
        self.assertEqual(df.loc[0, "prev_edge_ps"], 12225.0)
        self.assertEqual(df.loc[0, "phi_peak_from_trigger"], 300.0)
        self.assertEqual(df.loc[0, "peak_minus_t0_ps"], 50.0)
        self.assertEqual(df.loc[0, "peak_minus_prev_edge_ps"], 75.0)
        self.assertEqual(df.loc[0, "trigger_minus_t0_ps"], -250.0)
        self.assertNotIn("t_ave_ps", df.columns)

    def test_build_matched_dataframe_handles_missing_values(self):
        df = simple_sync.build_legacy_matched_dataframe([{"entry": 1}], {})

        self.assertEqual(list(df.columns), simple_sync.LEGACY_MATCHED_CSV_COLUMNS)
        self.assertEqual(df.loc[0, "channelID"], "[]")
        self.assertTrue(math.isnan(df.loc[0, "t0_abs_ps"]))


if __name__ == "__main__":
    unittest.main()
