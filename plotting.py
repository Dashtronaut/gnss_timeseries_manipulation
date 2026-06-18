"""
gnss_tools.plotting
===================
Publication-ready time-series plots for GNSS displacement data.

Vertical annotations
--------------------
* Red dashed  lines  — earthquake times  (``eqtimes``).
* Green dotted lines — offset / antenna-change times  (``offsets``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .timeseries import TimeSeries

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_COMP_STYLES = {
    "east":  dict(color="#2166ac", label="East"),
    "north": dict(color="#d6604d", label="North"),
    "up":    dict(color="#4dac26", label="Up"),
}
_SIGMA_ALPHA  = 0.18
_EQ_STYLE     = dict(color="red",   linestyle="--", linewidth=0.9, alpha=0.7)
_OFF_STYLE    = dict(color="green", linestyle=":",  linewidth=0.9, alpha=0.8)

_SIGMA_COLS = {
    "east":  "sigma_east",
    "north": "sigma_north",
    "up":    "sigma_up",
}


# ---------------------------------------------------------------------------
# Single-station plot
# ---------------------------------------------------------------------------

def plot_station(
    station:          "TimeSeries",
    components:       List[str] = None,
    show_uncertainty: bool  = True,
    figsize:          tuple = (12, 8),
    title:            Optional[str]  = None,
    save_path:        Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot one station's East / North / Up time series.

    Parameters
    ----------
    station          : :class:`~gnss_tools.TimeSeries`
    components       : which components to plot (default: all three).
    show_uncertainty : draw ±1σ shading.
    figsize          : ``(width, height)`` in inches.
    title            : figure suptitle.  Defaults to ``station.name``.
    save_path        : if given, save before returning.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    if components is None:
        components = ["east", "north", "up"]

    n_panels = len(components)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    df    = station.data
    times = df.index

    for ax, comp in zip(axes, components):
        style = _COMP_STYLES.get(comp, dict(color="steelblue",
                                            label=comp.capitalize()))
        y = df[comp].values

        ax.plot(times, y, ".", markersize=2.5,
                color=style["color"], rasterized=True, zorder=3)

        if show_uncertainty:
            scol = _SIGMA_COLS.get(comp)
            if scol and scol in df.columns:
                s = df[scol].values
                ax.fill_between(
                    times, y - s, y + s,
                    color=style["color"], alpha=_SIGMA_ALPHA, zorder=2,
                )

        # Earthquake annotations (label only the first line so the legend
        # shows "earthquake" once, not once per event)
        for i, eq in enumerate(station.eqtimes):
            ax.axvline(eq, label="earthquake" if (ax is axes[0] and i == 0) else "",
                       **_EQ_STYLE)

        # Offset / antenna-change annotations (same single-label treatment)
        for i, off in enumerate(station.offsets):
            ax.axvline(off, label="offset" if (ax is axes[0] and i == 0) else "",
                       **_OFF_STYLE)

        ax.set_ylabel(f"{style['label']} (mm)", fontsize=9)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.35)
        ax.grid(True, linewidth=0.4, alpha=0.4)

    # Legend (only if there are annotated events)
    if station.eqtimes or station.offsets:
        axes[0].legend(fontsize=7, loc="upper left", framealpha=0.7)

    _format_xaxis(axes[-1])

    ref = f" ({station.reference_frame})" if station.reference_frame else ""
    fig.suptitle(title or f"{station.name}{ref}", fontsize=12, fontweight="bold")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Multi-station comparison plot
# ---------------------------------------------------------------------------

def plot_multi(
    stations:         Sequence["TimeSeries"],
    component:        str   = "east",
    show_uncertainty: bool  = False,
    offset_mm:        float = 0.0,
    title:            Optional[str]  = None,
    figsize:          tuple = (12, 6),
    save_path:        Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Overlay one displacement component from multiple stations.

    Parameters
    ----------
    stations         : iterable of :class:`~gnss_tools.TimeSeries`.
    component        : ``"east"``, ``"north"``, or ``"up"``.
    show_uncertainty : draw ±1σ shading per station.
    offset_mm        : artificial vertical offset between traces (mm),
                       useful for stacked comparison.
    title            : figure title.
    figsize          : ``(width, height)`` in inches.
    save_path        : if given, save before returning.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    label_map = {"east": "East", "north": "North", "up": "Up"}
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for k, sta in enumerate(stations):
        df     = sta.data
        times  = df.index
        y      = df[component].values + k * offset_mm
        color  = cmap(k % 10)

        ax.plot(times, y, ".", markersize=2, color=color,
                label=sta.name, rasterized=True)

        if show_uncertainty:
            scol = _SIGMA_COLS.get(component)
            if scol and scol in df.columns:
                s = df[scol].values
                ax.fill_between(times, y - s, y + s,
                                color=color, alpha=0.15)

    ax.set_ylabel(f"{label_map.get(component, component)} (mm)", fontsize=10)
    _format_xaxis(ax)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=7, ncol=min(4, len(list(stations))),
              loc="upper left", framealpha=0.7)
    ax.set_title(
        title or f"GNSS — {label_map.get(component, component)}",
        fontweight="bold",
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_xaxis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=8)