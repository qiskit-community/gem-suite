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
"""GEM circuit test."""

import unittest
from math import pi

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeGeneva
from qiskit import QuantumCircuit

from gem_suite.experiments import GemExperiment


class TestCircuit(unittest.TestCase):
    """Test that generates GEM circuit."""

    def test_generate_circuit(self):
        """Test comparing virtual circuits."""
        exp = GemExperiment(range(2), backend=FakeGeneva())

        exp.set_experiment_options(schedule_idx=10)
        test_circ = exp.parameterized_circuits()[0]

        theta = exp.parameter

        ref_circ = QuantumCircuit(21, 21)
        ref_circ.h(range(21))
        ref_circ.barrier()
        ref_circ.rzz(theta, [4, 7, 13, 16], [6, 9, 15, 18])
        ref_circ.rzz(pi / 2, [3, 8, 12, 17], [5, 11, 14, 20])
        ref_circ.barrier()
        ref_circ.rzz(theta, [0, 6, 9, 15], [3, 8, 12, 17])
        ref_circ.rzz(pi / 2, [1, 10, 19], [2, 11, 20])
        ref_circ.barrier()
        ref_circ.rzz(theta, [0, 9, 18], [1, 10, 19])
        ref_circ.rzz(pi / 2, [2, 5, 11, 14], [4, 7, 13, 16])
        ref_circ.barrier()
        ref_circ.h([1, 3, 4, 7, 8, 10, 12, 13, 16, 17, 19])
        ref_circ.barrier()
        ref_circ.measure(range(21), range(21))

        self.assertEqual(test_circ, ref_circ)

    def test_qubit_layout(self):
        """Test physical qubit index list."""
        exp = GemExperiment(range(2), backend=FakeGeneva())
        self.assertListEqual(
            list(exp.physical_qubits),
            [
                1,
                2,
                3,
                4,
                5,
                7,
                8,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                18,
                19,
                21,
                22,
                23,
                24,
                25,
            ],
        )

    def test_parameter_list(self):
        """Test parameter (theta) list to scan."""
        exp = GemExperiment(range(2), backend=FakeGeneva())

        np.testing.assert_array_almost_equal(
            exp.parameters(),
            np.linspace(0, np.pi / 2, 21),
        )
