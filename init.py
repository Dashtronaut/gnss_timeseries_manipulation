"""
Supported file formats
----------------------
* USGS  .rneu   — decimal-year  N  E  U  σN  σE  σU  (mm)
* UNR   .tenv3  — decimal-year + YMD, E  N  U  σE  σN  σU  (m → mm)
* PBO   .pos    — YYYYMMDD      N  E  U  σN  σE  σU  (m → mm)

Quick-start
-----------
    from gnss_tools import TimeSeries

    sta = TimeSeries.from_file("albh.rneu", name="ALBH")

    sta_clean = (
        sta
        .remove_outliers()                        # Hampel identifier
        .remove_offsets(auto=True)                # estimate & subtract all known offsets
        .detrend()
        .plot()
    )
"""

from .timeseries import TimeSeries
from .io import load_rneu, load_tenv3, load_pos, load_file
from .processing import (
    hampel_identifier,
    detrend_series,
    estimate_offset,
    estimate_interval_offset,
)
from .plotting import plot_station, plot_multi

__all__ = [
    "TimeSeries",
    "load_rneu", "load_tenv3", "load_pos", "load_file",
    "hampel_identifier", "detrend_series",
    "estimate_offset", "estimate_interval_offset",
    "plot_station", "plot_multi",
]
