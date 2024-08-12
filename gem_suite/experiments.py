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
"""Qiskit Experiments implementation of GEM experiment protocol."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit.providers.backend import BackendV2
from qiskit.circuit import QuantumCircuit, Parameter

from .analysis import GemAnalysis
from .plaquettes import PlaquetteLattice


class GemExperiment(BaseExperiment):
    """GEM experiment in 2D protocol."""

    parameter = Parameter("θ")

    def __init__(
        self,
        plaquettes: PlaquetteLattice | list[int],
        backend: BackendV2 | None,
    ):
        """Create new GEM experiment.

        .. notes::
            Current implementation supports only Heavy Hexagon lattice
            and measurement-based 2D protocol.

        Args:
            plaquettes: List of plaquette indices or configuired :class:`PlaquetteLattice`
                instance to build GEM circuits on.
            backend: Qiskit Backend to run experiments.

        Raises:
            RuntimeError: When plaquettes are selected by index but backend is not provided.
        """
        if not isinstance(plaquettes, PlaquetteLattice):
            plaquettes = PlaquetteLattice.from_backend(backend).filter(
                includes=plaquettes
            )
        qubits = [q.index for q in plaquettes.qubits()]
        super().__init__(
            physical_qubits=qubits,
            analysis=GemAnalysis(plaquettes=plaquettes),
            backend=backend,
        )
        self._plaquettes = plaquettes

    @property
    def plaquettes(self) -> PlaquetteLattice:
        """Plaquette lattice to run experiemnts on."""
        return self._plaquettes

    def parameters(self) -> np.ndarray:
        """Create numpy array of parameters scanned in this experiment.

        Returns:
            Angle of ZZ gate determining tA.
        """
        if self.experiment_options.angles is None:
            return np.linspace(
                self.experiment_options.min_angle,
                self.experiment_options.max_angle,
                self.experiment_options.num_angles,
            )
        return np.asarray(self.experiment_options.angles, dtype=float)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values for GEM experiment.

        Experiment Options:
            schedule_idx (int): Select from 0 to 11 if any. If not provided
                the experiment builds circuits for all possible scheduling patterns
                and performs the analysis on averaged outcomes.
            min_angle (float): Minimum angle to scan.
            max_angle (float): Maximum angle to scan.
            num_angles (int): Number of angles to scan.
            angles (Sequence[float]): The list of angles that will be scanned in
                the experiment. If not set, then ``num_angles``
                evenly spaced delays between ``min_angle`` and ``max_angle``
                are used. If ``angles`` is set, these parameters are ignored.
        """
        options = super()._default_experiment_options()
        options.schedule_idx = None
        options.sweep_type = "A"

        options.min_angle = 0
        options.max_angle = np.pi / 2
        options.num_angles = 21
        options.angles = None

        options.set_validator("sweep_type", ["A", "B"])

        return options

    def parameterized_circuits(self) -> tuple[QuantumCircuit, ...]:
        """Return GEM circuit with unbound parameters.

        Returns:
            Parameterized circuits.
        """
        bond_idxs = [
            self.physical_qubits.index(q.index)
            for q in self._plaquettes.qubits()
            if q.role == "Bond"
        ]

        # Create virtual circuit
        # Virtual qubit index is based off of list index of self.physical_qubits
        if usr_index := self.experiment_options.schedule_idx:
            sched_idx = [usr_index]
        else:
            sched_idx = list(range(12))
        circs = []
        for i in sched_idx:
            sched_iter = self._plaquettes.build_gate_schedule(i)
            circ = QuantumCircuit(self.num_qubits, self.num_qubits)
            circ.metadata = {"schedule_index": i}
            circ.h(range(self.num_qubits))
            for gate_group in sched_iter:
                circ.barrier()
                for gate in gate_group:
                    if gate.group == self.experiment_options.sweep_type:
                        angle = self.parameter
                    else:
                        angle = np.pi / 2
                    circ.rzz(
                        angle,
                        self.physical_qubits.index(gate.index0),
                        self.physical_qubits.index(gate.index1),
                    )
            circ.barrier()
            circ.h(bond_idxs)
            circ.barrier()
            circ.measure(range(self.num_qubits), range(self.num_qubits))
            circs.append(circ)
        return tuple(circs)

    def circuits(self) -> list[QuantumCircuit]:
        circs_bound = []
        for tmp_circ, param in product(
            self.parameterized_circuits(), self.parameters()
        ):
            assigned = tmp_circ.assign_parameters(
                {self.parameter: param}, inplace=False
            )
            assigned.metadata["theta"] = np.round(param / np.pi, 5)
            circs_bound.append(assigned)
        return circs_bound

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata["connectivity"] = self.plaquettes.connectivity()
        metadata["plaquette_qubit_map"] = {
            p.index: p.qubits for p in self.plaquettes.plaquettes()
        }
        return metadata
