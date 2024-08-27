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

import pandas as pd
from uncertainties import ufloat
from qiskit_experiments.framework import (
    BaseAnalysis,
    ExperimentData,
    AnalysisResultData,
    ArtifactData,
    FigureData,
    Options,
)

from gem_suite.gem_core import visualize_plaquette_with_noise, PyHeavyHexLattice
from .sub_analysis import (
    analyze_magnetization,
    analyze_operators,
    analyze_individual_bonds,
    analyze_clifford_limit,
)
from .plaquettes import PlaquetteLattice
from .plot_utils import dot_to_mplfigure


class GemAnalysis(BaseAnalysis):
    """GEM analysis for 2D protocol."""

    def __init__(
        self,
        plaquettes: PlaquetteLattice | None = None,
    ):
        """Create new GEM analysis instance.

        Args:
            plaquettes: PlaquetteLattice instance of target device.
                This can be automatically instantiated from the
                GEM Experiment metadata if not given.
        """
        super().__init__()
        self._plaquettes = plaquettes

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options for GEM analysis.

        Analysis Options:
            analyze_individual_bond (bool): Perform analysis on individual
                ZXZ operator of each bond qubits to analyze
                local p_bond and p_site quantities.
                This may create large amount of analysis result entries
                when you run experiment with many plaquettes.
            analyze_clifford_limit (bool): Perform analysis at Clifford limit,
                i.e. theta = 0.5 pi. This data point must be included.
                Enabling this option will add a new figure illustrating
                the error distribution in the plaquette view along with
                several analysis results of plaquette-wise qualities.
            decoder (str): Minimum weight perfect matching decoder.
                "fusion-blossom" and "pymatching" are supported.
        """
        options = super()._default_options()
        options.update_options(
            analyze_individual_bond=False,
            analyze_clifford_limit=False,
            decoder="pymatching",
        )
        options.set_validator("decoder", ["pymatching", "fusion-blossom"])
        return options

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> tuple[list[AnalysisResultData | ArtifactData], list[FigureData]]:
        fig_data = []
        analysis_results = []

        # Check if lattice object exists.
        # The lattice information is necessary for decoding experiment outcomes.
        if self._plaquettes is None:
            plaquette_qubit_map = experiment_data.metadata.get(
                "plaquette_qubit_map", None
            )
            connectivity = experiment_data.metadata.get("connectivity", None)
            if plaquette_qubit_map is None or connectivity is None:
                raise RuntimeError(
                    "Analysis is not initialized with PlaquetteLattice instance and "
                    "its description is not provided by the experiment metadata. "
                    "This analysis cannot be performed because of "
                    "the missing lattice information to decode bitstrings."
                )
            lattice = PyHeavyHexLattice.from_plaquettes(
                plaquette_qubit_map, connectivity
            )
            self._plaquettes = PlaquetteLattice(lattice)

        # Decode count dictionaries and build intermediate data collection.
        records = []
        for data in experiment_data.data():
            result = self._plaquettes.decode_outcomes(
                counts=data["counts"],
                decoder=self.options.decoder,
                return_counts=False,
            )
            records.extend(
                [
                    {
                        "name": "f",
                        "theta": data["metadata"]["theta"],
                        "schedule": data["metadata"]["schedule_index"],
                        "value": ufloat(result.f[0], result.f[1]),
                        "component": pd.NA,
                    },
                    {
                        "name": "g",
                        "theta": data["metadata"]["theta"],
                        "schedule": data["metadata"]["schedule_index"],
                        "value": ufloat(result.g[0], result.g[1]),
                        "component": pd.NA,
                    },
                ]
            )
            for i, expv in enumerate(result.w_ops):
                records.append(
                    {
                        "name": "w",
                        "theta": data["metadata"]["theta"],
                        "schedule": data["metadata"]["schedule_index"],
                        "value": expv,
                        "component": i,
                    }
                )
            for i, expv in enumerate(result.zxz_ops):
                records.append(
                    {
                        "name": "zxz",
                        "theta": data["metadata"]["theta"],
                        "schedule": data["metadata"]["schedule_index"],
                        "value": expv,
                        "component": i,
                    }
                )
        dataframe = pd.DataFrame.from_records(records)
        del records
        
        # Analyze magnetizations
        num_site = 0
        for qubit in self._plaquettes.qubits():
            if qubit.role == "Site":
                num_site += 1
        mag_figs, mag_results = analyze_magnetization(dataframe, num_site)
        fig_data.extend(mag_figs)
        analysis_results.extend(mag_results)

        # Analyze operators
        op_figs, op_results = analyze_operators(dataframe)
        fig_data.extend(op_figs)
        analysis_results.extend(op_results)

        # Analyze local bond parameters (Optional)
        if self.options.analyze_individual_bond:
            local_params = analyze_individual_bonds(
                data=dataframe,
                qubits=self._plaquettes.qubits(),
            )
            analysis_results.extend(local_params)

        # Analyze plaquette quality (Optional)
        if self.options.analyze_clifford_limit:
            plaquette_qualities = analyze_clifford_limit(
                data=dataframe,
                plaquettes=self._plaquettes.plaquettes(),
                qubits=self._plaquettes.qubits(),
            )
            analysis_results.extend(plaquette_qualities)
            if plaquette_qualities:
                p_map = {p.index: p.qubits for p in self._plaquettes.plaquettes()}
                noise_map = {
                    q.extra["plaquette"]: q.value.n
                    for q in plaquette_qualities if q.value is not None
                }
                dot = visualize_plaquette_with_noise(p_map, noise_map)
                fig_data.append(
                    FigureData(
                        dot_to_mplfigure(dot, "neato", 300), name="plaquette_quality"
                    )
                )

        return analysis_results, fig_data
