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
"""Plaquette representation of the Qiskit Backend."""

from __future__ import annotations

from typing import Iterator, Callable, NamedTuple

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix
from qiskit_experiments.framework import FigureData
from qiskit.providers import BackendV2

from gem_suite.gem_core import PyHeavyHexLattice, PyQubit, PyPlaquette, PyScheduledGate
from .plot_utils import dot_to_mplfigure


class ScheduledGate(NamedTuple):
    q0: int
    q1: int
    group: str


class DecodeOutcome(NamedTuple):
    w_ops: list[float]
    zxz_ops: list[float]
    f: tuple[float, float]
    g: tuple[float, float]


class PlaquetteLattice:
    """Plaquette representation of Qiskit Backend."""

    def __init__(self, lattice: PyHeavyHexLattice):
        """Create new plaquette lattice.

        Args:
            lattice: Rust plaquette core.
        """
        self._core = lattice

    @classmethod
    def from_backend(cls, backend: BackendV2):
        """Create new instance from Qiskit Backend."""
        if hasattr(backend, "configuration"):
            cmap = [tuple(qs) for qs in backend.configuration().coupling_map]
        else:
            cmap = list(backend.coupling_map)
        return PlaquetteLattice(PyHeavyHexLattice(cmap))

    @classmethod
    def from_coupling_map(cls, coupling_map: list[tuple[int, int]]):
        """Create new instance from device coupling map."""
        return PlaquetteLattice(PyHeavyHexLattice([tuple(qs) for qs in coupling_map]))

    def qubits(self) -> Iterator[PyQubit]:
        """Yield annotated qubit dataclasses."""
        yield from self._core.qubits()

    def plaquettes(self) -> Iterator[PyPlaquette]:
        """Yield plaquette dataclasses."""
        yield from self._core.plaquettes()

    def connectivity(self) -> list[tuple[int, int]]:
        """Return unidirectional qubit connectivity."""
        return self._core.connectivity()

    def draw_qubits(self) -> FigureData:
        """Draw coupling graph with qubits in the lattice."""
        svg_fig = dot_to_mplfigure(self._core.qubit_graph_dot(), "fdp", 300)
        return FigureData(svg_fig, name="qubit_graph")

    def draw_plaquettes(self) -> FigureData:
        """Draw coupling graph with plaquette in the lattice."""
        svg_fig = dot_to_mplfigure(self._core.plaquette_graph_dot(), "neato", 300)
        return FigureData(svg_fig, name="plaquette_graph")

    def draw_decode_graph(self) -> FigureData:
        """Draw qubit graph with annotation for decoding."""
        svg_fig = dot_to_mplfigure(self._core.decode_graph_dot(), "fdp", 300)
        return FigureData(svg_fig, name="decode_graph")

    def filter(self, includes: list[int]) -> PlaquetteLattice:
        """Create new plaquette lattice instance with subset of plaquettes.

        Args:
            includes: Index of plaquettes to include.
                This cannot include disconnected groups.
                All qubits must be connected to build GEM circuit.

        Returns:
            New plaquette lattice instance.
        """
        return PlaquetteLattice(self._core.filter(includes))

    def build_gate_schedule(self, index: int) -> Iterator[list[PyScheduledGate]]:
        """Yield list of entangling gates that can be simultaneously applied.

        Args:
            index: Index of gate schedule. There might be multiple scheduling patterns.

        Yields:
            List of :class:`.PyScheduledGate` dataclass representing
                an entangling gate, and all gates in a list can be
                applied simultaneously without qubit overlapping.
        """
        yield from self._core.build_gate_schedule(index)

    def check_matrix_csc(self) -> csc_matrix:
        """Create check matrix from the plaquette lattice."""
        hvec, dims = self._core.check_matrix_csc()
        data = len(hvec[0])
        return csc_matrix((data, hvec), dims, dtype=bool)

    def decode_outcomes(
        self,
        counts: dict[str, int],
        decoder: str = "pymatching",
        return_counts: bool = False,
    ) -> DecodeOutcome | tuple[DecodeOutcome, dict[str, int]]:
        """Decode count dictionary of the experiment result and analyze.

        Args:
            counts: Count dictionary of single circuit.
            decoder: MWPM decoder to decode count string. Either "pymatching" or "fusion-blossom".
            return_counts: Set True to return count dictionary.

        Returns:
            Decoded outcomes including plaquette and ZXZ bond operators,
            and f and g quantities associated with the prepared state magnetization.
            When the return_counts is set, this returns a tuple of
            outcome and count dictionary keyed on decoded site qubit bitstring.
        """
        if decoder == "fusion-blossom":
            out = self._core.decode_outcomes_fb(counts, return_counts)
            outcome = DecodeOutcome(*out[1:])
        elif decoder == "pymatching":
            hvec, dims = self._core.check_matrix_csc()
            data = np.ones(len(hvec[0]))
            solver = _build_batch_decoder(csc_matrix((data, hvec), dims, dtype=np.bool_))
            out = self._core.decode_outcomes_pm(solver, counts, return_counts)
            outcome = DecodeOutcome(*out[1:])
        else:
            raise ValueError(f"MWPM decoder {decoder} is not supported.")
        if return_counts:
            return outcome, out[0]
        return outcome


def _build_batch_decoder(check_matrix) -> Callable:

    solver = Matching.from_check_matrix(check_matrix)

    def _batched_solver(shots: list[bool], shape: tuple[int, int]):
        batch_in = np.array(shots, dtype=np.bool_).reshape(shape)
        out = solver.decode_batch(shots=batch_in)
        return out.astype(np.bool_).flatten().tolist()

    return _batched_solver
