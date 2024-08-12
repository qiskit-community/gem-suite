# GEM experiment suite
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""GEM Analysis test."""

import unittest

import numpy as np
from gem_suite.plaquettes import PlaquetteLattice

from .utils import process_counts_debug


class TestAnalysis(unittest.TestCase):
    """Test case for GEM analysis with Falcon processor mock."""

    def setUp(self):
        super().setUp()

        falcon_cmap = [
            (0, 1),
            (1, 0),
            (1, 2),
            (1, 4),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 5),
            (4, 1),
            (4, 7),
            (5, 3),
            (5, 8),
            (6, 7),
            (7, 4),
            (7, 6),
            (7, 10),
            (8, 5),
            (8, 9),
            (8, 11),
            (9, 8),
            (10, 7),
            (10, 12),
            (11, 8),
            (11, 14),
            (12, 10),
            (12, 13),
            (12, 15),
            (13, 12),
            (13, 14),
            (14, 11),
            (14, 13),
            (14, 16),
            (15, 12),
            (15, 18),
            (16, 14),
            (16, 19),
            (17, 18),
            (18, 15),
            (18, 17),
            (18, 21),
            (19, 16),
            (19, 20),
            (19, 22),
            (20, 19),
            (21, 18),
            (21, 23),
            (22, 19),
            (22, 25),
            (23, 21),
            (23, 24),
            (24, 23),
            (24, 25),
            (25, 22),
            (25, 24),
            (25, 26),
            (26, 25),
        ]

        self.falcon_lattice = PlaquetteLattice.from_coupling_map(falcon_cmap)

    def test_fusion_blossom_decode(self):
        """Test count decode with fusion blossom solver."""
        rng = np.random.default_rng(123)
        counts = {"".join(rng.choice(["0", "1"], size=21)): 100 for _ in range(10)}
        test_outcomes, test_dict = self.falcon_lattice.decode_outcomes(
            counts=counts,
            decoder="fusion-blossom",
            return_counts=True,
        )
        ref = process_counts_debug(counts)
        self.assertDictEqual(test_dict, ref[0])
        np.testing.assert_array_almost_equal(
            np.array(test_outcomes.w_ops, dtype=float),
            np.array(ref[1], dtype=float),
        )
        np.testing.assert_array_almost_equal(
            np.array(test_outcomes.zxz_ops, dtype=float),
            np.array(ref[2], dtype=float),
        )
        self.assertAlmostEqual(test_outcomes.f[0], ref[3][0])
        self.assertAlmostEqual(test_outcomes.f[1], ref[3][1])
        self.assertAlmostEqual(test_outcomes.g[0], ref[4][0])
        self.assertAlmostEqual(test_outcomes.g[1], ref[4][1])

    def test_pymatching_decode(self):
        """Test count decode with pymatching solver."""
        rng = np.random.default_rng(123)
        counts = {"".join(rng.choice(["0", "1"], size=21)): 100 for _ in range(10)}
        test_outcomes, test_dict = self.falcon_lattice.decode_outcomes(
            counts=counts,
            decoder="pymatching",
            return_counts=True,
        )
        ref = process_counts_debug(counts)
        self.assertDictEqual(test_dict, ref[0])
        np.testing.assert_array_almost_equal(
            np.array(test_outcomes.w_ops, dtype=float),
            np.array(ref[1], dtype=float),
        )
        np.testing.assert_array_almost_equal(
            np.array(test_outcomes.zxz_ops, dtype=float),
            np.array(ref[2], dtype=float),
        )
        self.assertAlmostEqual(test_outcomes.f[0], ref[3][0])
        self.assertAlmostEqual(test_outcomes.f[1], ref[3][1])
        self.assertAlmostEqual(test_outcomes.g[0], ref[4][0])
        self.assertAlmostEqual(test_outcomes.g[1], ref[4][1])
