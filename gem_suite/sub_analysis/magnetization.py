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
"""Analysis for prepared state magnetization."""
from __future__ import annotations

import pandas as pd
import numpy as np

from uncertainties import unumpy as unp
from qiskit_experiments.framework import AnalysisResultData, FigureData

from gem_suite.plot_utils import plot_schedules


def analyze_magnetization(
    data: pd.DataFrame,
) -> tuple[list[FigureData], list[AnalysisResultData]]:
    """Analyze magnetization of the prepared state.

    Args:
        data: Dataframe contaning f and g entries.

    Returns:
        FigureData and AnalysisResultData for Qiskit Experiments.
    """
    figures = []
    analysis_results = []

    # Analyze two point classical correlation
    f_data = data[data.name == "f"]
    axis, (xvals, yvals) = plot_schedules(f_data)
    nominal_yvals = unp.nominal_values(yvals)
    argmax = np.argmax(nominal_yvals)
    fmax = nominal_yvals[argmax]
    axis.set_ylabel(r"Average two-point correlation $f$")
    axis.text(
        xvals[argmax],
        nominal_yvals[argmax],
        f"max_tpc = {fmax:.4f}",
        ha="right",
        va="bottom",
    )
    figures.append(
        FigureData(
            figure=axis.get_figure(),
            name="two_point_correlation",
        )
    )
    analysis_results.append(
        AnalysisResultData(
            name="max_tpc",
            value=yvals[argmax],
        )
    )

    # Analyze variance
    g_data = data[data.name == "g"]
    axis, (xvals, yvals) = plot_schedules(g_data)
    nominal_yvals = unp.nominal_values(yvals)
    argmax = np.argmax(nominal_yvals)
    critical = xvals[argmax]
    axis.set_ylabel(r"Normalized variance $g$")
    axis.text(
        critical,
        nominal_yvals[argmax],
        f"critical_angle = {critical:.4f}",
        ha="right",
        va="bottom",
    )
    figures.append(
        FigureData(
            figure=axis.get_figure(),
            name="normalized_variance",
        )
    )
    analysis_results.append(
        AnalysisResultData(
            name="critical_angle",
            value=critical,
        )
    )

    return figures, analysis_results
