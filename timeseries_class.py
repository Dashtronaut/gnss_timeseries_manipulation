from dataclasses import dataclass, field
from typing import List
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

# Pandas dataframe storing all gnss data indexed by datetime 
# Immutable just to make sure the original doesn't get altered in some way
@dataclass(frozen=True)
class TimeSeries:
    name: str

    data: pd.DataFrame

    latitude: float
    longitude: float

    reference_frame: str

    eqtimes: List[dt.datetime] = field(default_factory=list)
    offsets: List[dt.datetime] = field(default_factory=list)

    def __post_init__(self):

        required_columns = [
            "east",
            "north",
            "up",
            "sigma_east",
            "sigma_north",
            "sigma_up"
        ]

        # Checking for required columns and expected values
        missing = [
            c for c in required_columns
            if c not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if not isinstance(
            self.data.index,
            pd.DatetimeIndex
        ):
            raise TypeError(
                "DataFrame index must be a pandas DatetimeIndex"
            )

        if len(self.data) == 0:
            raise ValueError(
                f"{self.name} contains no data"
            )

        object.__setattr__(
            self,
            "data",
            self.data.sort_index()
        )

    
    # Helper function to create new timeseries object, preserves metadata
    def _new(self, data):

        return TimeSeries(
            name=self.name,
            data=data,

            latitude=self.latitude,
            longitude=self.longitude,

            reference_frame=self.reference_frame,

            eqtimes=self.eqtimes,
            offsets=self.offsets
        )

    @property
    def dtarray(self):
        """
        Return datetime array.
        """

        return self.data.index.to_numpy()

    @property
    def east(self):
        return self.data["east"].to_numpy()

    @property
    def north(self):
        return self.data["north"].to_numpy()

    @property
    def up(self):
        return self.data["up"].to_numpy()

    @property
    def sigma_east(self):
        return self.data["sigma_east"].to_numpy()

    @property
    def sigma_north(self):
        return self.data["sigma_north"].to_numpy()

    @property
    def sigma_up(self):
        return self.data["sigma_up"].to_numpy()

    # Converting datetime notation into decimal for more flexibility
    @property
    def decimal_years(self):
        idx = self.data.index

        return (
            idx.year +
            (idx.dayofyear - 1) / 365.25
        )

   
    # Returns new timeseries object within given date range
    def impose_time_limits(
        self,
        starttime,
        endtime
    ):

        df = self.data.loc[starttime:endtime]

        # Keeps only metadata within range
        new_eqtimes = [
            t for t in self.eqtimes
            if starttime <= t <= endtime
        ]

        new_offsets = [
            t for t in self.offsets
            if starttime <= t <= endtime
        ]

        return TimeSeries(
            name=self.name,
            data=df,

            latitude=self.latitude,
            longitude=self.longitude,

            reference_frame=self.reference_frame,

            eqtimes=new_eqtimes,
            offsets=new_offsets
        )

    # Removing NaN values; importantly, returns a new object instead of altering original
    def remove_nans(self):

        cols = ["east", "north", "up"]

        df = self.data.dropna(subset=cols)

        return self._new(df)

  
    # Removes outliers by deviation from the median (filter in mm deviation), window of samples
    def remove_outliers(
        self,
        threshold=10.0,
        window=35
    ):

        df = self.data.copy()

        keep = np.ones(len(df), dtype=bool)

        for comp in ["east", "north", "up"]:

            med = signal.medfilt(
                df[comp],
                kernel_size=window
            )

            residual = np.abs(df[comp] - med)

            if comp == "up":
                keep &= residual < threshold * 2
            else:
                keep &= residual < threshold

        return self._new(df.loc[keep])

  
    # Returns linear velocity in mm/year of specified component
    def get_velocity(
        self,
        component="east"
    ):

        t = self.decimal_years

        y = self.data[component]

        mask = np.isfinite(y)

        coeffs = np.polyfit(
            t[mask],
            y[mask],
            1
        )

        velocity = coeffs[0]

        return velocity

    # Simple linear detrending
    def detrend(self):

        t = self.decimal_years

        df = self.data.copy()

        for comp in ["east", "north", "up"]:

            y = df[comp]

            mask = np.isfinite(y)

            coeffs = np.polyfit(
                t[mask],
                y[mask],
                1
            )

            trend = np.polyval(coeffs, t)

            df[comp] = y - trend

        return self._new(df)


    # Removes offsets at specified time index, automatically computes offset using median of data before and after offset
    def remove_offset(
        self,
        offset_time
    ):

        df = self.data.copy()

        before = df.loc[df.index < offset_time]
        after = df.loc[df.index >= offset_time]

        if len(before) == 0 or len(after) == 0:
            raise ValueError(
                "Offset time must split dataset into "
                "before and after segments."
            )

        for comp in ["east", "north", "up"]:

            jump = (
                after[comp].median() -
                before[comp].median()
            )

            df.loc[
                df.index >= offset_time,
                comp
            ] -= jump

        return self._new(df)

    # Removes offsets in internal data object list
    def remove_offsets(self):

        ts = self

        all_offsets = (
            list(self.eqtimes) +
            list(self.offsets)
        )

        for offset in sorted(all_offsets):
            ts = ts.remove_offset(offset)

        return ts

    
    # Basic pandas plot
    def plot(
        self,
        show_uncertainty=False,
        figsize=(10, 8)
    ):
        """
        Plot east/north/up time series.
        """

        fig, axes = plt.subplots(
            3,
            1,
            figsize=figsize,
            sharex=True
        )

        components = [
            ("east", "sigma_east"),
            ("north", "sigma_north"),
            ("up", "sigma_up")
        ]

        for ax, (comp, sigma) in zip(
            axes,
            components
        ):

            ax.plot(
                self.data.index,
                self.data[comp],
                ".-",
                linewidth=1
            )

            # Optional uncertainty shading
            if show_uncertainty:

                y = self.data[comp]
                s = self.data[sigma]

                ax.fill_between(
                    self.data.index,
                    y - s,
                    y + s,
                    alpha=0.3
                )

            # Plotting earthquake times
            for eq in self.eqtimes:
                ax.axvline(
                    eq,
                    linestyle="--",
                    alpha=0.5
                )

            # Plotting offset times
            for off in self.offsets:
                ax.axvline(
                    off,
                    linestyle=":",
                    alpha=0.5
                )

            ax.set_ylabel(comp)

        axes[-1].set_xlabel("Date")

        fig.suptitle(
            f"{self.name} ({self.reference_frame})"
        )

        plt.tight_layout()

        plt.show()

# Constructor helper function to create dataframe object from numpy arrays or lists
def from_arrays(
    name,
    dtarray,

    east,
    north,
    up,

    sigma_east,
    sigma_north,
    sigma_up,

    latitude,
    longitude,

    reference_frame,

    eqtimes=None,
    offsets=None
):

    if eqtimes is None:
        eqtimes = []

    if offsets is None:
        offsets = []

    df = pd.DataFrame(
        {
            "east": east,
            "north": north,
            "up": up,

            "sigma_east": sigma_east,
            "sigma_north": sigma_north,
            "sigma_up": sigma_up
        },
        index=pd.to_datetime(dtarray)
    )

    return TimeSeries(
        name=name,
        data=df,

        latitude=latitude,
        longitude=longitude,

        reference_frame=reference_frame,

        eqtimes=eqtimes,
        offsets=offsets
    )
    
@classmethod
def from_csv(cls, filepath, **kwargs):

    from read_write_csv.py import read_csv_timeseries

    return read_csv_timeseries(filepath, **kwargs)


@classmethod
def from_pos(cls, filepath, **kwargs):

    from read_pos.py import read_pos_timeseries

    return read_pos_timeseries(filepath, **kwargs)


@classmethod
def from_tenv(cls, filepath, **kwargs):

    from read_tenv.py import read_tenv_timeseries

    return read_tenv_timeseries(filepath, **kwargs)


@classmethod
def from_rneu(cls, filepath, **kwargs):

    from read_rneu.py import read_rneu_timeseries

    return read_rneu_timeseries(filepath, **kwargs)
