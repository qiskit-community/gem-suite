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

from typing import cast, TYPE_CHECKING
try:
    from PIL import Image  # type: ignore
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

if TYPE_CHECKING:
    from PIL import Image  # type: ignore

from gem_suite.gem_core import PyHeavyHexPlaquette
from qiskit.providers import BackendV2


class PlaquetteLattice:

    def __init__(self, backend: BackendV2):
        if hasattr(backend, "configuration"):
            cmap = backend.configuration().coupling_map
        else:
            cmap = list(backend.coupling_map)
        self.core = PyHeavyHexPlaquette(cmap)
        
    def plaquette(self, index: int | None) -> list[int]:
        return self.core.plaquettes[index]
    
    def draw(self) -> Image:
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
            ["fdp", "-T", "png"],
            input=cast(str, self.core.to_dot()).encode("utf-8"),
            capture_output=True,
            encoding=None,
            check=True,
            text=False,
        )
        dot_bytes_image = io.BytesIO(dot_result.stdout)
        return Image.open(dot_bytes_image)
