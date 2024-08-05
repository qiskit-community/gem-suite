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
"""Qiskit Experiments implementation of GEM outcome decoding and analysis."""

from __future__ import annotations
from typing import List, Tuple

from qiskit_experiments.framework import (
    BaseAnalysis,
    ExperimentData,
    AnalysisResultData,
    ArtifactData,
    FigureData,
    Options,
)

from .plaquettes import PlaquetteLattice


class GemAnalysis(BaseAnalysis):
    """GEM analysis in 2D protocol."""
    
    def __init__(
        self,
        plaquettes: PlaquetteLattice | None = None,
    ):
        super().__init__()
        self._plaquettes = plaquettes
    
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> tuple[list[AnalysisResultData | ArtifactData], list[FigureData]]:
        if self._plaquettes is None:
            cmap = experiment_data.metadata.get("coupling_map", None)
            if cmap is None:
                raise RuntimeError(
                    "Analysis is not initialized with PlaquetteLattice and "
                    "coupling map is not provided by the experiment metadata. "
                    "This analysis cannot be performed because of "
                    "the missing lattice configuration to decode bitstrings."
                )
            self._plaquettes = PlaquetteLattice.from_coupling_map(cmap)
        
