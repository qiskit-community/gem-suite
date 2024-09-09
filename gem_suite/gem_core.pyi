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

from typing import Callable

def visualize_plaquette_with_noise(
    plaquette_qubits_map: dict[int, list[int]],
    noise_map: dict[int, float],
) -> str: ...

class PyQubit:
    index: int
    role: str
    group: str
    coordinate: tuple[int, int] | None
    neighbors: list[int]

class PyPlaquette:
    index: int
    qubits: list[int]
    neighbors: list[int]

class PyScheduledGate:
    index0: int
    index1: int
    group: str

class PyHeavyHexLattice:
    def __init__(self, coupling_map: list[tuple[int, int]]) -> PyHeavyHexLattice: ...
    @classmethod
    def from_plaquettes(
        cls,
        plaquette_qubits_map: dict[int, list[int]],
        connectivity: list[tuple[int, int]],
    ) -> PyHeavyHexLattice: ...
    def qubit_graph_dot(self) -> str: ...
    def plaquette_graph_dot(self) -> str: ...
    def decode_graph_dot(self) -> str: ...
    def qubits(self) -> list[int]: ...
    def plaquettes(self) -> list[int]: ...
    def connectivity(self) -> list[tuple[int, int]]: ...
    def filter(self, includes: list[int]) -> PyHeavyHexLattice: ...
    def build_gate_schedule(self, index: int) -> list[list[PyScheduledGate]]: ...
    def decode_outcomes_fb(
        self, counts: dict[str, int], return_counts: bool
    ) -> tuple[
        dict[str, int] | None,
        list[float],
        list[float],
        tuple[float, float],
        tuple[float, float],
    ]: ...
    def decode_outcomes_pm(
        self, solver: Callable, counts: dict[str, int], return_counts: bool
    ) -> tuple[
        dict[str, int] | None,
        list[float],
        list[float],
        tuple[float, float],
        tuple[float, float],
    ]: ...
    def check_matrix_csc(
        self,
    ) -> tuple[tuple[list[int], list[int]], tuple[float, float]]: ...
