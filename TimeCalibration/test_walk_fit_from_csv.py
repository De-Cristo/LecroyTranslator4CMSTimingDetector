from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import walk_fit_from_csv


class WalkFitFromCsvTest(unittest.TestCase):
    def test_bin_occupancy_mask_excludes_sparse_tail_bin(self):
        x = np.array([0.10, 0.11, 0.12, 0.20, 0.21, 0.29])

        mask, edges, counts = walk_fit_from_csv._bin_occupancy_mask(
            x, nbins=3, min_entries=2
        )

        self.assertTrue(np.array_equal(counts, np.array([3, 2, 1])))
        self.assertTrue(np.array_equal(mask, np.array([True, True, True, True, True, False])))
        self.assertEqual(len(edges), 4)

    def test_profile_points_keep_only_occupied_bins(self):
        x = np.array([0.10, 0.11, 0.12, 0.20, 0.21, 0.29])
        y = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 100.0])

        x_prof, y_prof, y_err, counts = walk_fit_from_csv._profile_points_from_bins(
            x, y, nbins=3, min_entries=2
        )

        self.assertTrue(np.allclose(x_prof, np.array([0.11, 0.205])))
        self.assertTrue(np.allclose(y_prof, np.array([2.0, 5.0])))
        self.assertTrue(np.array_equal(counts, np.array([3, 2])))
        self.assertEqual(len(y_err), 2)

    def test_parse_args_accepts_occupancy_cli(self):
        with mock.patch.object(
            sys, "argv",
            ["prog", "input.csv", "--nbins", "9", "--min-entries", "7", "--output-prefix", "out"]
        ):
            args = walk_fit_from_csv.parse_args()

        self.assertEqual(args.input_csv, "input.csv")
        self.assertEqual(args.nbins, 9)
        self.assertEqual(args.min_entries, 7)
        self.assertEqual(args.output_prefix, "out")

    def test_write_fit_coeffs_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = str(Path(tmpdir) / "test_walk")
            p_event = np.array([1.0, 2.0, 3.0])
            p_binned = np.array([4.0, 5.0, 6.0])

            walk_fit_from_csv._write_fit_coeffs_csv(prefix, p_event, p_binned)

            lines = Path(f"{prefix}_refit_coeffs.csv").read_text().strip().splitlines()
            self.assertEqual(lines[0], "fit_kind,p2,p1,p0")
            self.assertEqual(lines[1], "event,1.0,2.0,3.0")
            self.assertEqual(lines[2], "binned,4.0,5.0,6.0")

    def test_parabola_label_contains_coefficients(self):
        label = walk_fit_from_csv._parabola_label(np.array([1.0, 2.0, 3.0]))
        self.assertIn("y =", label)
        self.assertIn("1.00e+00", label)
        self.assertIn("2.00e+00", label)
        self.assertIn("3.00e+00", label)


if __name__ == "__main__":
    unittest.main()
