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

import warnings
from typing import Callable

import pandas as pd
import numpy as np

from scipy.optimize import least_squares
from uncertainties import ufloat, UFloat, unumpy as unp, correlated_values
from qiskit_experiments.framework import AnalysisResultData, FigureData
from qiskit_experiments.database_service.device_component import Qubit

from gem_suite.plot_utils import plot_schedules
from gem_suite.gem_core import PyQubit, PyPlaquette


def analyze_operators(
    data: pd.DataFrame,
) -> tuple[list[FigureData], list[AnalysisResultData]]:
    """Analyze plaquette and bond operators.
    
    .. notes::
        This analysis is performed on mean quantities
        averaged over all components, i.e. plaquettes and bonds.
    
    Args:
        data: Dataframe contaning w and zxz entries.
    
    Returns:
        FigureData and AnalysisResultData for Qiskit Experiments.
    """
    figures = []
    analysis_results = []

    # Average over all components
    w_op_data_mean = data[data.name == "w"].groupby(
        ["theta", "schedule"],
        as_index=False,
    ).value.agg(lambda v: ufloat(np.average(v), np.std(v)))
    zxz_op_data_mean = data[data.name == "zxz"].groupby(
        ["theta", "schedule"],
        as_index=False,
    ).value.agg(lambda v: ufloat(np.average(v), np.std(v)))

    # Fit data
    # Here we consider multi-objective fitting for both plaquette and bond models
    # since they share the p_bond parameter describing the
    # readout error on bond qubits and noise during entangling process.
    ax_w_op, (xvals, yvals_w) = plot_schedules(w_op_data_mean)
    ax_w_op.set_ylabel(r"Mean plaquette observables $\langle W \rangle$")

    ax_zxz_op, (_, yvals_zxz) = plot_schedules(zxz_op_data_mean)
    ax_zxz_op.set_ylabel(r"Mean bond observables $\langle ZXZ \rangle$")

    yvals_w_nominal = unp.nominal_values(yvals_w)
    yvals_w_std = unp.std_devs(yvals_w)
    yvals_w_std = np.where(np.isclose(yvals_w_std, 0.0), np.finfo(float).eps, yvals_w_std)
    w_weights = 1 / yvals_w_std
    w_weights = np.clip(w_weights, 0.0, np.percentile(w_weights, 90))

    yvals_zxz_nominal = unp.nominal_values(yvals_zxz)
    yvals_zxz_std = unp.std_devs(yvals_zxz)
    yvals_zxz_std = np.where(np.isclose(yvals_zxz_std, 0.0), np.finfo(float).eps, yvals_zxz_std)
    zxz_weights = 1 / yvals_zxz_std
    zxz_weights = np.clip(zxz_weights, 0.0, np.percentile(zxz_weights, 90))
    
    def multi_residuals(params):
        r_w = (w_model(xvals, params[0]) - yvals_w_nominal) * w_weights
        r_zxz = (zxz_model(xvals, params[0], params[1]) - yvals_zxz_nominal) * zxz_weights
        return np.concatenate([r_w, r_zxz])
    
    if ret := fit_util(
        multi_residuals,
        x0=[0.01, 0.01],
        bounds=([0.0, 0.0], [1.0, 1.0]),
        n_data=len(xvals),
    ):
        fit_vals, red_chisq, aic = ret
        
        # Plot fit curves
        x_interp = np.linspace(0, 0.5, 100)
        y_interp_w = w_model(x_interp, fit_vals[0].n)
        y_interp_zxz = zxz_model(x_interp, fit_vals[0].n, fit_vals[1].n)
        ax_w_op.plot(
            x_interp,
            y_interp_w,
            color="blue",
            lw=2.0,
        )
        ax_w_op.text(
            x_interp[50],
            y_interp_w[50],
            f"p_bond = {fit_vals[0].n:.4f}",
            ha="left",
            va="top",
        )
        ax_zxz_op.plot(
            x_interp,
            y_interp_zxz,
            color="blue",
            lw=2.0,
        )
        ax_zxz_op.text(
            x_interp[50],
            y_interp_zxz[50],
            f"p_bond = {fit_vals[0].n:.4f}, p_site = {fit_vals[1].n:.4f}",
            ha="left",
            va="top",
        )
        
        # Create results
        analysis_results = [
            AnalysisResultData(
                "p_bond_mean",
                value=fit_vals[0],
                chisq=red_chisq,
                quality="Good" if red_chisq < 3.0 else "Bad",
                extra={"aic": aic},
            ),
            AnalysisResultData(
                "p_site_mean",
                value=fit_vals[1],
                chisq=red_chisq,
                quality="Good" if red_chisq < 3.0 else "Bad",
                extra={"aic": aic},
            ),
        ]
    else:
        warnings.warn(
            "Fitting for global p_bond and p_site is not successful. ",
            RuntimeWarning,
        )
        # Create results
        analysis_results = [
            AnalysisResultData(
                "p_bond_mean",
                value=None,
                quality="Bad",
            ),
            AnalysisResultData(
                "p_site_mean",
                value=None,
                quality="Bad",
            ),
        ]
    figures.extend(
        [
            FigureData(
                figure=ax_w_op.get_figure(),
                name="plaquette_ops",
            ),
            FigureData(
                figure=ax_zxz_op.get_figure(),
                name="bond_ops",
            )
        ]
    )
    
    return figures, analysis_results


def analyze_individual_bonds(
    data: pd.DataFrame,
    qubits: list[PyQubit],
) -> list[AnalysisResultData]:
    """Analyze bond operators individually.
    
    Args:
        data: Dataframe contaning zxz entries.
    
    Returns:
        AnalysisResultData for Qiskit Experiments.
    """
    analysis_results = []
    
    # Create map from bond index to qubit
    bond_qubits = sorted([q for q in qubits if q.role == "Bond"], key=lambda q: q.index)
    
    # ZXZ data averaged over all schedules
    zxz_op_data = data[data.name == "zxz"].groupby(
        ["theta", "component"], as_index=False
    ).value.agg(lambda v: ufloat(np.average(v), np.std(v)))
    bonds = sorted(zxz_op_data.component.unique())
    for bond, qubit in zip(bonds, bond_qubits):
        qk_components = [Qubit(qubit.index), Qubit(qubit.neighbors[0]), Qubit(qubit.neighbors[1])]
        subdata = zxz_op_data[zxz_op_data.component == bond]
        xvals = subdata.theta.to_numpy()
        yvals = subdata.value.to_numpy()
        yvals_nominal = unp.nominal_values(yvals)
        yvals_std = unp.std_devs(yvals)
        yvals_std = np.where(np.isclose(yvals_std, 0.0), np.finfo(float).eps, yvals_std)
        weights = 1 / yvals_std
        weights = np.clip(weights, 0.0, np.percentile(weights, 90))
        
        def objective(params):
            return zxz_model(xvals, params[0], params[1] - yvals_nominal) * weights
        
        if ret := fit_util(
            objective,
            x0=[0.01, 0.01],
            bounds=([0.0, 0.0], [1.0, 1.0]),
            n_data=len(xvals),
        ):
            fit_vals, red_chisq, aic = ret
            # Plotting for every bond generates too many figures.
            # Just create results entries.
            this_results = [
                AnalysisResultData(
                    "p_bond",
                    value=fit_vals[0],
                    chisq=red_chisq,
                    quality="Good" if red_chisq < 3.0 else "Bad",
                    device_components=qk_components,
                    extra={"aic": aic},
                ),
                AnalysisResultData(
                    "p_site",
                    value=fit_vals[1],
                    chisq=red_chisq,
                    quality="Good" if red_chisq < 3.0 else "Bad",
                    device_components=qk_components,
                    extra={"aic": aic},
                ),
            ]
        else:
            warnings.warn(
                f"Fitting for p_bond and p_site on bond {bond} is not successful. ",
                RuntimeWarning,
            )
            this_results = [
                AnalysisResultData(
                    "p_bond",
                    value=None,
                    quality="Bad",
                    device_components=qk_components,
                ),
                AnalysisResultData(
                    "p_site",
                    value=None,
                    quality="Bad",
                    device_components=qk_components,
                ),
            ]
        analysis_results.extend(this_results)
    return analysis_results        


def analyze_clifford_limit(
    data: pd.DataFrame,
    plaquettes: list[PyPlaquette],
    qubits: list[PyQubit],
) -> list[AnalysisResultData]:
    """Visualize quality distribution at Clifford limit."""
    analysis_results = []

    clif_data = data[np.isclose(data.theta - 0.5, np.finfo(float).eps)]
    if len(clif_data) == 0:
        warnings.warn(
            "Experiment data at theta = 0.5 pi of the Clifford limit is not included in the result.",
            RuntimeWarning,
        )
        return []
    plaquettes = sorted(plaquettes, key=lambda p: p.index)
    bond_to_qubits = {i: q for i, q in enumerate(sorted([q for q in qubits if q.role == "Bond"], key=lambda q: q.index))}
    for plaquette in plaquettes:
        sub_qubits = plaquette.qubits
        plq_index = plaquette.index
        w_val = clif_data[
            (clif_data.name == "w") & (clif_data.component == plq_index)
        ].value.agg(lambda v: ufloat(np.average(v), np.std(v)))
        zxz_vals = []
        for bond_bit, bond_qubit in bond_to_qubits.items():
            # Take bond qubits in this plaquette
            if bond_qubit.index not in sub_qubits:
                continue
            zxz_val = clif_data[
                (clif_data.name == "zxz") & (clif_data.component == bond_bit)
            ].value.agg(lambda v: ufloat(np.average(v), np.std(v)))
            zxz_vals.append(zxz_val)
        zxz_mean_val = np.average(zxz_vals)
        analysis_results.append(
            AnalysisResultData(
                "plaquette_quality",
                value=w_val * zxz_mean_val,
                device_components=[Qubit(q) for q in sub_qubits],
                extra={"plaquette": plq_index}
            )
        )
    return analysis_results


def fit_util(
    objective: Callable, 
    x0: list[float], 
    bounds: tuple[list[float], list[float]], 
    n_data: int,
) -> None | tuple[tuple[UFloat, ...], float, float]:
    """Wrapper of scipy optmizer function.
    
    Args:
        objective: An objective function to minimize.
        x0: Initial parameters.
        bounds: Minimum and maximum range of parameters.
        n_data: Number of data points to fit.
    
    Returns:
        Parameters in ufloat, reduced Chi-squared, and AIC when successful.
    """
    if n_data < len(x0):
        warnings.warn(
            "Number of experiment data points is smaller than free parameters to fit. "
            "Fit cannot be performed on this experiment result."
        )
        return None
    ret = least_squares(
        fun=objective,
        x0=x0,
        bounds=bounds,
        method="trf",
        loss="linear",
    )
    if ret.success:
        # Compute statistics
        dof = n_data - len(x0)
        chisq = np.sum(ret.fun ** 2)
        red_chisq = chisq / dof
        aic = n_data * np.log(chisq / n_data) + 2 * len(x0)
        # Compute error of the parameters
        hess = np.matmul(ret.jac.T, ret.jac)
        fit_vals = None
        try:
            covar = np.linalg.inv(hess)
            if not any(np.diag(covar) < 0):
                fit_vals = correlated_values(ret.x, covariance_mat=covar)
        except np.linalg.LinAlgError:
            pass
        return (fit_vals or tuple(ufloat(x, np.nan) for x in ret.x), red_chisq, aic)
    return None


def w_model(theta, p_bond):
    """Fit model for plaquette observable.
    
    Args:
        theta: Experiment parameter.
        p_bond: Readout error on bond qubits and noise during entangling process.
    """
    return np.sin(np.pi * theta) ** 6 * (1 - 2 * p_bond) ** 6

def zxz_model(theta, p_bond, p_site):
    """Fit model for bond observable.
    
    Args:
        theta: Experiment parameter.
        p_bond: Readout error on bond qubits and noise during entangling process.
        p_site: Readout error on site qubits.
    """
    return np.sin(np.pi * theta) * (1 - 2 * p_site) ** 2 * (1 - 2 * p_bond)
