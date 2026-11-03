"""Checks for the sampling unit and weighting used in the MWR revision."""
import unittest
import numpy as np
from mwr_sounding_uncertainty import cluster_replicates


class ClusterBootstrapTests(unittest.TestCase):
    def test_case_weighting_and_joint_day_resampling(self):
        # Day 0 has two T storms and one N storm; day 1 has one of each.
        values = np.array([1., 3., 10., 7., 20.])
        modes = np.array(['T', 'T', 'N', 'T', 'N'])
        days = np.array([0, 0, 0, 1, 1])
        draws = np.array([[1, 1], [2, 0], [0, 2]])
        actual = cluster_replicates(values, modes, days, draws)
        # Explicitly duplicate every case on each drawn day, then average.
        expected = []
        for weights in draws:
            indices = np.concatenate([np.tile(np.flatnonzero(days == d), count)
                                      for d, count in enumerate(weights)])
            t = values[indices[modes[indices] == 'T']].mean()
            n = values[indices[modes[indices] == 'N']].mean()
            expected.append([t, n, t-n])
        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(actual[0, 0], 11/3)  # not an equal-day mean of 4.5

    def test_missing_values_change_counts_without_becoming_zero(self):
        actual = cluster_replicates(np.array([np.nan, 4., 10., 20.]),
            np.array(['T','T','N','N']), np.array([0,1,0,1]), np.array([[1,1]]))
        np.testing.assert_allclose(actual, [[4.,15.,-11.]])

    def test_no_group_support_fails_explicitly(self):
        with self.assertRaises(ValueError):
            cluster_replicates(np.array([1.,2.]),np.array(['T','N']),
                               np.array([0,1]),np.array([[2,0]]))


if __name__ == '__main__':
    unittest.main()
