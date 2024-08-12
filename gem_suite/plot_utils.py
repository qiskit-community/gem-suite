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
"""Sub module for visualization."""
from __future__ import annotations

import subprocess
import tempfile
import io

from typing import cast, TYPE_CHECKING

import pandas as pd
import numpy as np
from uncertainties import unumpy as unp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

try:
    from PIL import Image  # type: ignore

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

if TYPE_CHECKING:
    from PIL import Image  # type: ignore


def plot_schedules(
    data: pd.DataFrame,
) -> tuple[Axes, tuple[np.ndarray, np.ndarray]]:
    """Plot individual data and average of them in dataframe.

    Args:
        data: Filtered dataframe containing data to visualize.

    Returns:
        Matplotlib figure axis object and averaged data in ufloat format.
    """
    axis: Axes = get_non_gui_ax()
    data = data.sort_values("theta")
    schedules = sorted(data.schedule.unique())
    if len(schedules) > 1:
        for sched in schedules:
            subdata = data[data.schedule == sched]
            nominal = unp.nominal_values(subdata.value)
            stdev = unp.std_devs(subdata.value)
            axis.plot(
                subdata.theta,
                nominal,
                color="grey",
                alpha=0.4,
                lw=0.7,
            )
            axis.fill_between(
                subdata.theta,
                nominal + stdev,
                nominal - stdev,
                color="grey",
                alpha=0.1,
            )
        avg = data.groupby("theta", as_index=False).value.agg(np.average)
        x_mean = avg.theta.to_numpy()
        y_mean = avg.value.to_numpy()
    else:
        x_mean = data.theta.to_numpy()
        y_mean = data.value.to_numpy()
    nominal = unp.nominal_values(y_mean)
    stdev = unp.std_devs(y_mean)
    axis.errorbar(
        x_mean,
        nominal,
        color="blue",
        alpha=1.0,
        fmt="o",
    )
    axis.set_xlabel(r"$\theta$ ($\pi$)")
    axis.set_xlim(0, max(0.5, x_mean.max()))
    return axis, (x_mean.astype(float), y_mean)


def dot_to_mplfigure(
    dot_data: str,
    method: str,
    dpi: int = 300,
    rescale: int = 4,
) -> Figure:
    """Render dot script and return in matplotlib Figure format.

    Args:
        dot_data: Input dot script.
        method: Drawing method.
        dpi: Dot per inch.
        rescale: Scaling factor of image to load PIL Image data in the matplotlib canvas.

    Returns:
        Matplotlib Figure data written in SVG Backend.
        This backend doesn't support automatic rendering in Jupyter notebook environment.
    """
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
    except Exception as ex:
        raise RuntimeError(
            "Graphviz could not be found or run. "
            "This function requires that Graphviz is installed."
        ) from ex
    dot_result = subprocess.run(
        [method, "-T", "png"],
        input=cast(str, dot_data).encode("utf-8"),
        capture_output=True,
        encoding=None,
        check=True,
        text=False,
    )
    dot_bytes_image = io.BytesIO(dot_result.stdout)
    img = Image.open(dot_bytes_image)

    # Convert into matplotlib axis object
    axis: Axes = get_non_gui_ax()
    fig = axis.get_figure()
    fig.set_size_inches(img.width / dpi * rescale, img.height / dpi * rescale)
    fig.dpi = dpi

    axis.imshow(img)
    axis.set_axis_off()
    return fig
