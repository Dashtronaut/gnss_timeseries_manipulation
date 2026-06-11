"""
gnss_timeseries_manipulation.timeseries
=====================
Immutable container for a single GNSS station's displacement time series.

* Index is a ``pd.DatetimeIndex``; column order is east / north / up.
* Offset handling offers both automatic (median-window estimation) and
  manual (user-supplied mm values) paths, both chainable.

Outlier removal
---------------
Uses the Hampel identifier (rolling median ± k × MAD, default k = 3,
window = 11 samples), applied independently per component (Klos et al. 2015,
*IAG Symposia* 143; Langbein & Bock 2004 for the IQR predecessor).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .io import load_file
from . import processing, plotting


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DateLike = Union[str, dt.datetime, pd.Timestamp]
OffsetDict = Dict[str, Dict[str, float]]   # ISO-date → {component: mm}


# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeSeries:
    """
    Immutable GNSS displacement time series for one station.

    Parameters
    ----------
    name : str
        Four-character station identifier (e.g. ``"ALBH"``).
    data : pd.DataFrame
        Time series with columns ``east``, ``north``, ``up``,
        ``sigma_east``, ``sigma_north``, ``sigma_up`` and a
        ``pd.DatetimeIndex``.  Values in **millimetres**.
    latitude, longitude : float
        Station coordinates in decimal degrees.
    reference_frame : str
        Reference frame label (e.g. ``"NA-fixed"``, ``"ITRF2014"``).
    eqtimes : list of datetime
        Known earthquake times; drawn as red dashed lines in plots.
    offsets : list of datetime
        Known equipment / antenna change times; drawn as green dotted
        lines in plots.
    fmt : str
        Original file format (``"rneu"``, ``"tenv3"``, ``"pos"``).
    """

    name:            str
    data:            pd.DataFrame
    latitude:        float = float("nan")
    longitude:       float = float("nan")
    reference_frame: str   = ""
    eqtimes:         List[dt.datetime] = field(default_factory=list)
    offsets:         List[dt.datetime] = field(default_factory=list)
    fmt:             str   = ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        required = ["east", "north", "up",
                    "sigma_east", "sigma_north", "sigma_up"]
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pd.DatetimeIndex")
        if len(self.data) == 0:
            raise ValueError(f"{self.name} contains no data")
        # Sort index (bypass frozen via object.__setattr__)
        object.__setattr__(self, "data", self.data.sort_index())

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        fmt:             str   = "auto",
        name:            Optional[str]   = None,
        latitude:        float = float("nan"),
        longitude:       float = float("nan"),
        reference_frame: str   = "",
        eqtimes:         Optional[List[DateLike]] = None,
        offsets:         Optional[List[DateLike]] = None,
    ) -> "TimeSeries":
        """
        Load a GNSS displacement file and return a :class:`TimeSeries`.

        Parameters
        ----------
        path : str or Path
        fmt  : ``"auto"``, ``"rneu"``, ``"tenv3"``, or ``"pos"``.
        name : station name; inferred from filename stem if omitted.
        latitude, longitude : station coordinates (decimal degrees).
        reference_frame : e.g. ``"NA-fixed"``, ``"ITRF2014"``.
        eqtimes : known earthquake datetimes to annotate.
        offsets : known equipment-change datetimes to annotate.
        """
        path = Path(path)
        df, detected_fmt = load_file(path, fmt=fmt)
        return cls(
            name=name or path.stem.upper(),
            data=df,
            latitude=latitude,
            longitude=longitude,
            reference_frame=reference_frame,
            eqtimes=[pd.Timestamp(t) for t in (eqtimes or [])],
            offsets=[pd.Timestamp(t) for t in (offsets or [])],
            fmt=detected_fmt,
        )

    @classmethod
    def from_arrays(
        cls,
        dtarray:         Sequence,
        east:            Sequence[float],
        north:           Sequence[float],
        up:              Sequence[float],
        sigma_east:      Sequence[float],
        sigma_north:     Sequence[float],
        sigma_up:        Sequence[float],
        name:            str   = "",
        latitude:        float = float("nan"),
        longitude:       float = float("nan"),
        reference_frame: str   = "",
        eqtimes:         Optional[List[DateLike]] = None,
        offsets:         Optional[List[DateLike]] = None,
    ) -> "TimeSeries":
        """
        Construct a :class:`TimeSeries` from raw arrays / lists.

        Parameters
        ----------
        dtarray : array-like of datetime-like objects or strings.
        east, north, up : displacement arrays in mm.
        sigma_east, sigma_north, sigma_up : uncertainty arrays in mm.
        """
        df = pd.DataFrame(
            {
                "east":        np.asarray(east,  dtype=float),
                "north":       np.asarray(north, dtype=float),
                "up":          np.asarray(up,    dtype=float),
                "sigma_east":  np.asarray(sigma_east,  dtype=float),
                "sigma_north": np.asarray(sigma_north, dtype=float),
                "sigma_up":    np.asarray(sigma_up,    dtype=float),
            },
            index=pd.to_datetime(dtarray),
        )
        return cls(
            name=name,
            data=df,
            latitude=latitude,
            longitude=longitude,
            reference_frame=reference_frame,
            eqtimes=[pd.Timestamp(t) for t in (eqtimes or [])],
            offsets=[pd.Timestamp(t) for t in (offsets or [])],
        )

    # ------------------------------------------------------------------
    # Internal helper — create a mutated copy
    # ------------------------------------------------------------------

    def _new(
        self,
        data:     Optional[pd.DataFrame]     = None,
        eqtimes:  Optional[List]             = None,
        offsets:  Optional[List]             = None,
        **kwargs,
    ) -> "TimeSeries":
        """Return a new :class:`TimeSeries` preserving all metadata."""
        return TimeSeries(
            name=self.name,
            data=data if data is not None else self.data.copy(),
            latitude=self.latitude,
            longitude=self.longitude,
            reference_frame=self.reference_frame,
            eqtimes=eqtimes if eqtimes is not None else list(self.eqtimes),
            offsets=offsets if offsets is not None else list(self.offsets),
            fmt=self.fmt,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dtarray(self) -> np.ndarray:
        """Datetime index as a numpy array."""
        return self.data.index.to_numpy()

    @property
    def east(self) -> np.ndarray:
        return self.data["east"].to_numpy()

    @property
    def north(self) -> np.ndarray:
        return self.data["north"].to_numpy()

    @property
    def up(self) -> np.ndarray:
        return self.data["up"].to_numpy()

    @property
    def sigma_east(self) -> np.ndarray:
        return self.data["sigma_east"].to_numpy()

    @property
    def sigma_north(self) -> np.ndarray:
        return self.data["sigma_north"].to_numpy()

    @property
    def sigma_up(self) -> np.ndarray:
        return self.data["sigma_up"].to_numpy()

    @property
    def decimal_years(self) -> np.ndarray:
        """Time index as decimal years (e.g. 1993.635)."""
        idx = self.data.index
        days_in_year = np.where(
            idx.is_leap_year, 366.0, 365.0
        )
        return idx.year + (idx.day_of_year - 1) / days_in_year

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        t0 = self.data.index[0].strftime("%Y-%m-%d") if len(self) else "—"
        t1 = self.data.index[-1].strftime("%Y-%m-%d") if len(self) else "—"
        return (
            f"TimeSeries(name={self.name!r}, n={len(self)}, "
            f"span={t0}→{t1}, ref={self.reference_frame!r}, "
            f"fmt={self.fmt!r})"
        )

    # ------------------------------------------------------------------
    # Time windowing
    # ------------------------------------------------------------------

    def impose_time_limits(
        self,
        starttime: DateLike,
        endtime:   DateLike,
    ) -> "TimeSeries":
        """
        Return a new :class:`TimeSeries` restricted to [starttime, endtime].
        Metadata lists (eqtimes, offsets) are also trimmed.
        """
        s, e = pd.Timestamp(starttime), pd.Timestamp(endtime)
        df = self.data.loc[s:e]
        new_eq  = [t for t in self.eqtimes if s <= t <= e]
        new_off = [t for t in self.offsets  if s <= t <= e]
        return self._new(data=df, eqtimes=new_eq, offsets=new_off)

    # Alias
    def trim(self, start: DateLike, end: DateLike) -> "TimeSeries":
        return self.impose_time_limits(start, end)

    # ------------------------------------------------------------------
    # NaN removal
    # ------------------------------------------------------------------

    def remove_nans(self) -> "TimeSeries":
        """Drop rows where any of east / north / up is NaN."""
        df = self.data.dropna(subset=["east", "north", "up"])
        return self._new(data=df)

    # ------------------------------------------------------------------
    # Outlier removal  (Hampel identifier — see module docstring)
    # ------------------------------------------------------------------

    def remove_outliers(
        self,
        window:       int   = 11,
        k:            float = 3.0,
        up_scale:     float = 2.0,
        components:   Sequence[str] = ("east", "north", "up"),
    ) -> "TimeSeries":
        """
        Remove outliers using the Hampel identifier.

        Parameters
        ----------
        window : int
            Half-width of the rolling window in *samples*.  The full
            window is ``2 * window + 1`` so the current sample is centred.
            Default 11 (≈ 11 days for daily data).
        k : float
            MAD multiplier.  Samples beyond ``k × 1.4826 × MAD`` of the
            local median are flagged.  Default 3.0.
        up_scale : float
            Additional multiplier applied only to the Up component to
            account for its larger noise level (Langbein & Svarc 2019).
            Default 2.0 (so Up uses k_eff = 6.0 by default).
        components : sequence of str
            Which components to clean.  Default: all three.

        Returns
        -------
        TimeSeries
            New object with outliers set to NaN.
        """
        df = self.data.copy()
        for comp in components:
            k_eff = k * up_scale if comp == "up" else k
            df[comp] = processing.hampel_identifier(
                df[comp].values, window=window, k=k_eff
            )
        return self._new(data=df)

    # ------------------------------------------------------------------
    # Offset handling
    # ------------------------------------------------------------------

    def add_offset_times(
        self,
        times:    Sequence[DateLike],
        kind:     str = "offset",
    ) -> "TimeSeries":
        """
        Register known event times without applying any correction.

        Parameters
        ----------
        times : sequence of datetime-like
        kind  : ``"offset"`` (antenna / equipment) or ``"earthquake"``.
        """
        ts_list = [pd.Timestamp(t) for t in times]
        if kind == "earthquake":
            return self._new(eqtimes=list(self.eqtimes) + ts_list)
        return self._new(offsets=list(self.offsets) + ts_list)

    def remove_offsets(
        self,
        auto:          bool = True,
        manual:        Optional[OffsetDict] = None,
        window_days:   int  = 30,
        components:    Sequence[str] = ("east", "north", "up"),
    ) -> "TimeSeries":
        """
        Remove step offsets from the time series.

        Two complementary paths are available and can be combined:

        Automatic (``auto=True``)
            For every datetime registered in ``self.offsets``, estimate
            the jump magnitude from the median of the data in a
            ``window_days``-day window immediately before and after the
            event, then subtract it.  Works for both instantaneous
            (antenna swap) and extended (slow-slip) offsets.

        Manual (``manual={...}``)
            Subtract user-supplied offsets in mm.  Useful when USGS
            publishes offset magnitudes on the station page, or when an
            automatic estimate is unreliable (e.g. data gaps near the
            event).

            Format::

                manual = {
                    "1994-04-15": {"east": 11.72, "north": 2.11, "up": -8.12},
                    "2015-09-16": {"east": -1.46, "north": 1.01, "up": 12.65},
                }

        Parameters
        ----------
        auto         : apply automatic estimation for all stored offset times.
        manual       : dict of {ISO-date: {component: offset_mm}}.
        window_days  : half-window in days for automatic estimation.
        components   : which components to correct.

        Returns
        -------
        TimeSeries
        """
        df = self.data.copy()

        # --- automatic path ---
        if auto and self.offsets:
            for ts in sorted(self.offsets):
                for comp in components:
                    jump = processing.estimate_offset(
                        df[comp], ts, window_days=window_days
                    )
                    mask = df.index >= ts
                    df.loc[mask, comp] -= jump

        # --- manual path ---
        if manual:
            for date_str, comp_offsets in manual.items():
                ts = pd.Timestamp(date_str)
                mask = df.index >= ts
                for comp, val_mm in comp_offsets.items():
                    if comp in df.columns:
                        df.loc[mask, comp] -= val_mm

        return self._new(data=df)

    def fit_offset(
        self,
        offset_time:  DateLike,
        window_days:  int = 30,
        component:    str = "east",
    ) -> float:
        """
        Estimate an instantaneous step offset at ``offset_time``.

        Uses a ``window_days``-day median window on each side.

        Returns
        -------
        float : estimated jump in mm (positive = series rises after event).
        """
        return processing.estimate_offset(
            self.data[component],
            pd.Timestamp(offset_time),
            window_days=window_days,
        )

    def fit_interval_offset(
        self,
        start_time:   DateLike,
        end_time:     DateLike,
        window_days:  int = 30,
        component:    str = "east",
    ) -> float:
        """
        Estimate an offset occurring over an extended time interval
        (e.g. a slow-slip event or antenna height campaign).

        Returns
        -------
        float : estimated jump in mm.
        """
        return processing.estimate_interval_offset(
            self.data[component],
            pd.Timestamp(start_time),
            pd.Timestamp(end_time),
            window_days=window_days,
        )

    def apply_offset(
        self,
        offset_time:  DateLike,
        offset_mm:    float,
        component:    str  = "east",
        subtract:     bool = True,
    ) -> "TimeSeries":
        """
        Apply a pre-computed offset correction to a single component.

        Parameters
        ----------
        offset_time : event datetime.
        offset_mm   : magnitude in mm.
        component   : ``"east"``, ``"north"``, or ``"up"``.
        subtract    : if True (default) subtract the offset; if False add it.
        """
        df = self.data.copy()
        sign = -1 if subtract else 1
        df.loc[df.index >= pd.Timestamp(offset_time), component] += sign * offset_mm
        return self._new(data=df)

    # ------------------------------------------------------------------
    # Detrending
    # ------------------------------------------------------------------

    def detrend(
        self,
        method:           str   = "linear",
        components:       Sequence[str] = ("east", "north", "up"),
        reference_epoch:  Optional[float] = None,
    ) -> "TimeSeries":
        """
        Remove a secular trend from one or more components.

        Parameters
        ----------
        method : ``"linear"`` (OLS) or ``"robust"`` (Theil-Sen, from
                 ``scipy.stats``; resistant to remaining outliers).
        components : components to detrend.
        reference_epoch : decimal year used as t = 0 for the fit.
                          Defaults to temporal midpoint.

        Returns
        -------
        TimeSeries
        """
        df = self.data.copy()
        t  = self.decimal_years
        t0 = reference_epoch if reference_epoch is not None else 0.5 * (t[0] + t[-1])
        dt = t - t0
        for comp in components:
            df[comp] = processing.detrend_series(
                df[comp].values, dt, method=method
            )
        return self._new(data=df)

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    def get_velocity(self, component: str = "east") -> float:
        """
        Return the linear velocity (mm/yr) for *component* via OLS.
        """
        t = self.decimal_years
        y = self.data[component].values
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return float("nan")
        from scipy.stats import linregress
        slope, *_ = linregress(t[mask], y[mask])
        return float(slope)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        components:       Sequence[str] = ("east", "north", "up"),
        show_uncertainty: bool  = True,
        figsize:          Tuple[float, float] = (12, 8),
        title:            Optional[str] = None,
        save_path:        Optional[Union[str, Path]] = None,
    ):
        """
        Plot the time series (returns ``matplotlib.figure.Figure``).

        Earthquake times (``eqtimes``) are shown as red dashed lines;
        offset times (``offsets``) as green dotted lines.
        """
        return plotting.plot_station(
            self,
            components=list(components),
            show_uncertainty=show_uncertainty,
            figsize=figsize,
            title=title,
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_csv(self, path: Union[str, Path], **kwargs) -> None:
        """Write the time series DataFrame to CSV."""
        self.data.to_csv(path, **kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self.data.copy()
