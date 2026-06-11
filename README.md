# gnss_timeseries_manipulation

A quick Python toolkit for loading, cleaning, detrending, and plotting GNSS
displacement time series.  Built around an **immutable** `TimeSeries` dataclass.

**Supported formats**

| Format | Extension | Source |
|--------|-----------|--------|
| USGS   | `.rneu`   | USGS Earthquake Hazards Program |
| UNR    | `.tenv3`  | University of Nevada Reno |
| PBO    | `.pos`    | Plate Boundary Observatory / NOTA |


## Installation

```bash
git clone https://github.com/Dashtronaut/gnss_timeseries_manipulation.git
```

**Dependencies:** `numpy`, `pandas`, `scipy`, `matplotlib`.

---

## Quick-start

```python
from gnss_timeseries_manipulation import TimeSeries

sta = TimeSeries.from_file(
    "albh.rneu",
    fmt="rneu",
    name="ALBH",
    latitude=48.39,
    longitude=-123.49,
    reference_frame="NA-fixed",
    offsets=["1994-04-15", "1995-01-12", "2015-09-16"],  # antenna changes
)

sta_processed = (
    sta
    .remove_outliers()       
    .remove_offsets(auto=True)       # auto-estimate & subtract all offsets
    .detrend(method="linear")
)

sta_processed.plot(save_path="albh.png")
print(sta_processed.get_velocity("east"), "mm/yr")
```

---

## Core API

### `TimeSeries` — immutable dataclass

Every method returns a **new** `TimeSeries`; the original is never mutated.

| Method | Description |
|--------|-------------|
| `TimeSeries.from_file(path, fmt, name, …)` | Load from `.rneu` / `.tenv3` / `.pos` |
| `TimeSeries.from_arrays(dtarray, east, north, up, …)` | Construct from NumPy arrays |
| `.remove_outliers(window, k, up_scale)` | Hampel identifier per component |
| `.remove_offsets(auto, manual, window_days)` | Auto and/or manual offset correction |
| `.add_offset_times(times, kind)` | Creates a list of automatically identifed offsets |
| `.fit_offset(time, window_days, component)` | Estimate an instantaneous (mm) |
| `.fit_interval_offset(start, end, …)` | Estimate a long-term offset |
| `.apply_offset(time, offset_mm, component)` | Subtract a known offset |
| `.detrend(method, components)` | Remove linear/robust secular trend |
| `.get_velocity(component)` | OLS velocity in mm/yr |
| `.impose_time_limits(start, end)` | Trim to date range |
| `.trim(start, end)` | Alias for `impose_time_limits` |
| `.remove_nans()` | Drop rows with NaN in E/N/U |
| `.plot(components, show_uncertainty, …)` | Three-panel time-series figure |
| `.to_csv(path)` | Export to CSV |
| `.to_dataframe()` | Return copy of underlying DataFrame |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.east`, `.north`, `.up` | `np.ndarray` | Displacement (mm) |
| `.sigma_east`, `.sigma_north`, `.sigma_up` | `np.ndarray` | Uncertainty (mm) |
| `.decimal_years` | `np.ndarray` | Time in decimal years |
| `.dtarray` | `np.ndarray` | DatetimeIndex as array |
| `.eqtimes` | `list[Timestamp]` | Registered earthquake times |
| `.offsets` | `list[Timestamp]` | Registered offset times |

### Standalone functions

```python
from gnss_timeseries_manipulation import load_rneu, load_tenv3, load_pos   # → pd.DataFrame
from gnss_timeseries_manipulation import plot_multi                          # multi-station overlay
```

---

## Outlier removal: Outliers are detected using the **Hampel identifier**.

A sample is flagged when it deviates from the local rolling median by more
than `k × 1.4826 × MAD` (default k = 3, full window = 23 samples for daily
data).  Outliers are **set to NaN**, not interpolated.

The Up component uses a doubled MAD multiplier (`k_eff = k × up_scale`,
default `up_scale = 2`) to account for its ~4× larger coloured-noise
amplitude (Langbein & Svarc 2019, *JGR Solid Earth*).

---

## Offset removal: automatic vs manual

```python
# Automatic — estimates jump magnitude from ±30-day median windows
sta2 = sta.remove_offsets(auto=True, window_days=30)

# Manual — use known values (mm)
sta2 = sta.remove_offsets(
    auto=False,
    manual={
        "1994-04-15": {"east": 11.72, "north": 2.11, "up": -8.12},
        "2015-09-16": {"east": -1.46, "north": 1.01, "up": 12.65},
    }
)

# Both combined (manual overrides only the listed dates)
sta2 = sta.remove_offsets(auto=True, manual={"2003-09-06": {"up": -25.89}})
```

Event times stored in `sta.offsets` (and `sta.eqtimes`) are automatically
drawn as vertical lines in `sta.plot()`.
