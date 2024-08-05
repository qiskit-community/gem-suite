# This code is part of Qiskit.
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

import subprocess
import tempfile
import io

from collections import namedtuple
from typing import cast, TYPE_CHECKING, Iterator
try:
    from PIL import Image  # type: ignore
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

if TYPE_CHECKING:
    from PIL import Image  # type: ignore

from gem_suite.gem_core import PyHeavyHexLattice, PyQubit, PyPlaquette, PyScheduledGate
from qiskit.providers import BackendV2


ScheduledGate = namedtuple("ScheduledGate", ["q0", "q1", "group"])


class PlaquetteLattice:
    """Plaquette representation of Qiskit Backend."""

    def __init__(self, backend: BackendV2):
        """Create new plaquette lattice from backend.
        
        Args:
            backend: Qiskit Backend.
        """
        if hasattr(backend, "configuration"):
            cmap = backend.configuration().coupling_map
        else:
            cmap = list(backend.coupling_map)
        self.core = PyHeavyHexLattice(cmap)
        self.coupling_map = cmap

    @classmethod
    def from_coupling_map(cls, coupling_map: list[tuple[int, int]]):
        """Build plaquette lattice from coupling map.
        
        Args:
            coupling_map: List of connected qubit pair.
        
        Returns:
            New PlaquetteLattice instance.
        """
        new_lattice = PyHeavyHexLattice(coupling_map)
        instance = object.__new__(PlaquetteLattice)
        instance.core = new_lattice
        return instance
    
    def qubits(self) -> Iterator[PyQubit]:
        """Yield annotated qubit dataclasses."""
        yield from self.core.qubits()
        
    def plaquettes(self) -> Iterator[PyPlaquette]:
        """Yield plaquette dataclasses."""
        yield from self.core.plaquettes()
    
    def draw_qubits(self) -> Image:
        """Draw coupling graph with qubits in the lattice."""
        return _to_image(self.core.qubit_graph_dot(), "fdp")
        
    def draw_plaquettes(self) -> Image:
        """Draw coupling graph with plaquette in the lattice."""
        return _to_image(self.core.plaquette_graph_dot(), "neato")
    
    def draw_sites(self) -> Image:
        """Draw coupling graph with site qubits in the lattice."""
        return _to_image(self.core.site_graph_dot(), "fdp")

    def draw_snake(self) -> Image:
        """Draw snake coupling graph with site qubits in the lattice."""
        return _to_image(self.core.snake_graph_dot(), "fdp")
    
    def filter(self, includes: list[int]) -> PlaquetteLattice:
        """Create new plaquette lattice instance with subset of plaquettes.
        
        Args:
            includes: Index of plaquettes to include.
                This cannot include disconnected groups.
                All qubits must be connected to build GEM circuit.
        
        Returns:
            New plaquette lattice instance.
        """
        new_lattice = self.core.filter(includes)
        instance = object.__new__(PlaquetteLattice)
        instance.core = new_lattice
        return instance
    
    def build_gate_schedule(self, index: int) -> Iterator[list[PyScheduledGate]]:
        """Yield list of entangling gates that can be simultaneously applied.
        
        Args:
            index: Index of gate schedule. There might be multiple scheduling patterns.
        
        Yields:
            List of :class:`.PyScheduledGate` dataclass representing
                an entangling gate, and all gates in a list can be 
                applied simultaneously without qubit overlapping. 
        """
        yield from self.core.build_gate_schedule(index)
    

def _to_image(dot_data: str, method: str) -> Image:
        if not HAS_PILLOW:
            raise ImportError(
                "Pillow is necessary to use draw(). "
                "It can be installed with 'pip install pydot pillow'"
            )
        try:
            subprocess.run(
                ["dot", "-V"],
                cwd=tempfile.gettempdir(),
                check=True,
                capture_output=True,
            )
        except Exception:
            raise RuntimeError(
                "Graphviz could not be found or run. "
                "This function requires that Graphviz is installed."
            )
        dot_result = subprocess.run(
            [method, "-T", "png"],
            input=cast(str, dot_data).encode("utf-8"),
            capture_output=True,
            encoding=None,
            check=True,
            text=False,
        )
        dot_bytes_image = io.BytesIO(dot_result.stdout)
        return Image.open(dot_bytes_image)
