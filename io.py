"""
gnss_tools.io
=============
Parsers for the three most common GNSS displacement time-series text formats.

All parsers return ``(df, fmt_str)`` where *df* has:

    Index : pd.DatetimeIndex
    Columns : east  north  up  sigma_east  sigma_north  sigma_up

All values are in **millimetres**.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_maybe_gzip(path: Path):
    """
    Transparently open plain or gzipped text files.

    Tries UTF-8 first, then falls back to latin-1 (ISO-8859-1), which covers
    all single-byte encodings including Windows-1252.  Windows line endings
    (\\r\\n) are handled automatically by Python's universal newline mode.
    """
    if path.suffix.lower() == ".gz":
        # Try UTF-8; fall back to latin-1 for gzipped files
        try:
            import io
            raw = gzip.open(path, "rb").read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            return io.StringIO(text)
        except Exception:
            return gzip.open(path, "rt", encoding="latin-1", errors="replace")
    # Plain file — latin-1 accepts every byte value, so this never fails.
    # Universal newlines (the default) strips \r so \r\n files parse cleanly.
    return open(path, "r", encoding="latin-1")


def _decyr_to_datetime(decyr: np.ndarray) -> pd.DatetimeIndex:
    """Convert a decimal-year array to a DatetimeIndex."""
    years       = decyr.astype(int)
    remainder   = decyr - years
    is_leap     = ((years % 4 == 0) & (years % 100 != 0)) | (years % 400 == 0)
    days_in_yr  = np.where(is_leap, 366.0, 365.0)
    day_of_year = remainder * days_in_yr          # 0-based fractional
    timestamps  = [
        pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(days=float(d))
        for y, d in zip(years, day_of_year)
    ]
    return pd.DatetimeIndex(timestamps)


def _make_df(
    decyr: np.ndarray,
    east:  np.ndarray,
    north: np.ndarray,
    up:    np.ndarray,
    se:    np.ndarray,
    sn:    np.ndarray,
    su:    np.ndarray,
) -> pd.DataFrame:
    """Assemble, sort, and de-duplicate the canonical DataFrame."""
    idx = _decyr_to_datetime(decyr)
    df = pd.DataFrame(
        {
            "east":        east,
            "north":       north,
            "up":          up,
            "sigma_east":  se,
            "sigma_north": sn,
            "sigma_up":    su,
        },
        index=idx,
    ).sort_index()
    return df[~df.index.duplicated(keep="first")]


# ---------------------------------------------------------------------------
# USGS .rneu
# ---------------------------------------------------------------------------
# Whitespace-separated ASCII; comment lines start with #.
# Columns: decimal_year  N(mm)  E(mm)  U(mm)  σN(mm)  σE(mm)  σU(mm)
#          [optional extra columns ignored]
#
# Note: USGS orders N before E in the file; we store east/north.
# ---------------------------------------------------------------------------

def load_rneu(path) -> pd.DataFrame:
    """
    Load a USGS ``.rneu`` time-series file.

    Column order in file: ``decyr  N  E  U  σN  σE  σU``
    (North before East, all in mm).

    Returns
    -------
    pd.DataFrame with canonical columns and DatetimeIndex.
    """
    path = Path(path)
    rows = []
    first_lines: list[str] = []   # kept for diagnostics
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            raw = line.rstrip("\r\n")   # explicit strip in case universal newlines missed it
            line = raw.strip()
            if len(first_lines) < 8:
                first_lines.append(repr(raw))
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            rows.append(vals)

    if not rows:
        preview = "\n  ".join(first_lines) if first_lines else "(file appears empty)"
        raise ValueError(
            f"No valid data found in {path}\n\n"
            f"Expected whitespace-separated columns:\n"
            f"  decimal_year  N(mm)  E(mm)  U(mm)  σN(mm)  σE(mm)  σU(mm)\n\n"
            f"First lines of file:\n  {preview}\n\n"
            f"Common causes:\n"
            f"  • Wrong file downloaded (e.g. a detrended .data.gz instead of .rneu)\n"
            f"  • File is HTML/XML (server returned an error page)\n"
            f"  • Encoding issue — try opening the file in a text editor to verify\n"
        )

    arr = np.array(rows)
    #              decyr      N          E          U          σN         σE         σU
    return _make_df(arr[:,0], arr[:,2], arr[:,1], arr[:,3], arr[:,5], arr[:,4], arr[:,6])


# ---------------------------------------------------------------------------
# UNR .tenv3
# ---------------------------------------------------------------------------
# Header line begins with "Sta" or "____".
# Columns (0-based):
#   0=sta  1=YYMMMDD  2=decyr  3=MJD  4=GPSweek  5=dow
#   6=reflon  7=dE(m)  8=dN(m)  9=dU(m)  10=σE(m)  11=σN(m)  12=σU(m) …
#
# Displacements in metres → multiply by 1000 for mm.
# ---------------------------------------------------------------------------

def load_tenv3(path) -> pd.DataFrame:
    """
    Load a University of Nevada Reno ``.tenv3`` time-series file.

    Returns
    -------
    pd.DataFrame with canonical columns and DatetimeIndex.
    """
    path = Path(path)
    rows = []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Sta") or line.startswith("____"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                decyr = float(parts[2])
                e_m   = float(parts[7])
                n_m   = float(parts[8])
                u_m   = float(parts[9])
                se_m  = float(parts[10])
                sn_m  = float(parts[11])
                su_m  = float(parts[12])
            except (ValueError, IndexError):
                continue
            rows.append([decyr,
                         e_m * 1e3, n_m * 1e3, u_m * 1e3,
                         se_m * 1e3, sn_m * 1e3, su_m * 1e3])

    if not rows:
        raise ValueError(f"No valid data found in {path}")

    arr = np.array(rows)
    return _make_df(arr[:,0], arr[:,1], arr[:,2], arr[:,3],
                    arr[:,4], arr[:,5], arr[:,6])


# ---------------------------------------------------------------------------
# PBO / NOTA .pos
# ---------------------------------------------------------------------------
# Header section ends with a line starting "YYYYMMDD" or "*".
# Data columns (0-based):
#   0=YYYYMMDD  1=YYMMDD  2=MJD  3=week  4=dow  5=refN  6=refE  7=refU
#   8=dN(m)  9=dE(m)  10=dU(m)  11=σN(m)  12=σE(m)  13=σU(m)
#
# Displacements in metres → mm.
# ---------------------------------------------------------------------------

def load_pos(path) -> pd.DataFrame:
    """
    Load a PBO / NOTA ``.pos`` time-series file.

    Returns
    -------
    pd.DataFrame with canonical columns and DatetimeIndex.
    """
    path = Path(path)
    rows = []
    in_data = False
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("YYYYMMDD") or line.startswith("*"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                ds   = parts[0]            # YYYYMMDD
                year, month, day = int(ds[:4]), int(ds[4:6]), int(ds[6:8])
                n_m  = float(parts[8])
                e_m  = float(parts[9])
                u_m  = float(parts[10])
                sn_m = float(parts[11])
                se_m = float(parts[12])
                su_m = float(parts[13])
            except (ValueError, IndexError):
                continue
            ts    = pd.Timestamp(year=year, month=month, day=day)
            doy   = ts.day_of_year
            diy   = 366 if ts.is_leap_year else 365
            decyr = year + (doy - 1) / diy
            rows.append([decyr,
                         e_m * 1e3, n_m * 1e3, u_m * 1e3,
                         se_m * 1e3, sn_m * 1e3, su_m * 1e3])

    if not rows:
        raise ValueError(f"No valid data found in {path}")

    arr = np.array(rows)
    return _make_df(arr[:,0], arr[:,1], arr[:,2], arr[:,3],
                    arr[:,4], arr[:,5], arr[:,6])


# ---------------------------------------------------------------------------
# Auto-detect dispatcher
# ---------------------------------------------------------------------------

_EXT_TO_FMT = {
    ".rneu":  "rneu",
    ".tenv3": "tenv3",
    ".pos":   "pos",
}

_LOADERS = {
    "rneu":  load_rneu,
    "tenv3": load_tenv3,
    "pos":   load_pos,
}


def load_file(path, fmt: str = "auto") -> Tuple[pd.DataFrame, str]:
    """
    Load any supported GNSS time-series file with optional auto-detection.

    Parameters
    ----------
    path : str or Path
    fmt  : ``"auto"``, ``"rneu"``, ``"tenv3"``, or ``"pos"``.

    Returns
    -------
    ``(df, fmt_string)``
    """
    path = Path(path)

    if fmt == "auto":
        # Strip .gz suffix first
        bare = Path(path.stem) if path.suffix.lower() == ".gz" else path
        fmt  = _EXT_TO_FMT.get(bare.suffix.lower())
        if fmt is None:
            fmt = _sniff_format(path)

    if fmt not in _LOADERS:
        raise ValueError(
            f"Unknown format {fmt!r}. "
            f"Supported: {list(_LOADERS)}"
        )

    return _LOADERS[fmt](path), fmt


def _sniff_format(path: Path) -> str:
    """Inspect the first non-comment data line to guess the file format."""
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if re.match(r"^\d{8}$", parts[0]):
                return "pos"
            if re.match(r"^[A-Z0-9]{4}$", parts[0]) and len(parts) >= 13:
                return "tenv3"
            if re.match(r"^\d{4}\.\d+$", parts[0]):
                return "rneu"
            break
    raise ValueError(
        f"Could not auto-detect format of {path}. "
        "Pass fmt='rneu', 'tenv3', or 'pos' explicitly."
    )    remainder   = decyr - years
    is_leap     = ((years % 4 == 0) & (years % 100 != 0)) | (years % 400 == 0)
    days_in_yr  = np.where(is_leap, 366.0, 365.0)
    day_of_year = remainder * days_in_yr          # 0-based fractional
    timestamps  = [
        pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(days=float(d))
        for y, d in zip(years, day_of_year)
    ]
    return pd.DatetimeIndex(timestamps)


def _make_df(
    decyr: np.ndarray,
    east:  np.ndarray,
    north: np.ndarray,
    up:    np.ndarray,
    se:    np.ndarray,
    sn:    np.ndarray,
    su:    np.ndarray,
) -> pd.DataFrame:
    """Assemble, sort, and de-duplicate the canonical DataFrame."""
    idx = _decyr_to_datetime(decyr)
    df = pd.DataFrame(
        {
            "east":        east,
            "north":       north,
            "up":          up,
            "sigma_east":  se,
            "sigma_north": sn,
            "sigma_up":    su,
        },
        index=idx,
    ).sort_index()
    return df[~df.index.duplicated(keep="first")]


# ---------------------------------------------------------------------------
# USGS .rneu
# ---------------------------------------------------------------------------
# Whitespace-separated ASCII; comment lines start with #.
# Columns: decimal_year  N(mm)  E(mm)  U(mm)  σN(mm)  σE(mm)  σU(mm)
#          [optional extra columns ignored]
#
# Note: USGS orders N before E in the file; this tool uses east/north.
# ---------------------------------------------------------------------------

path = Path(path)
rows = []
first_lines: list[str] = []   # kept for diagnostics
with _open_maybe_gzip(path) as fh:
    for line in fh:
        raw = line.rstrip("\r\n")   # explicit strip in case universal newlines missed it
        line = raw.strip()
        if len(first_lines) < 8:
            first_lines.append(repr(raw))
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(p) for p in parts[:7]]
        except ValueError:
            continue
        rows.append(vals)

    if not rows:
        preview = "\n  ".join(first_lines) if first_lines else "(file appears empty)"
        raise ValueError(
            f"No valid data found in {path}\n\n"
            f"Expected whitespace-separated columns:\n"
            f"  decimal_year  N(mm)  E(mm)  U(mm)  σN(mm)  σE(mm)  σU(mm)\n\n"
            f"First lines of file:\n  {preview}\n\n"
            f"Common causes:\n"
            f"  • Wrong file downloaded (e.g. a detrended .data.gz instead of .rneu)\n"
            f"  • File is HTML/XML (server returned an error page)\n"
            f"  • Encoding issue — try opening the file in a text editor to verify\n"
        )
    
    arr = np.array(rows)
    #              decyr      N          E          U          σN         σE         σU
    return _make_df(arr[:,0], arr[:,2], arr[:,1], arr[:,3], arr[:,5], arr[:,4], arr[:,6])




# ---------------------------------------------------------------------------
# UNR .tenv3
# ---------------------------------------------------------------------------
# Header line begins with "Sta" or "____".
# Columns (0-based):
#   0=sta  1=YYMMMDD  2=decyr  3=MJD  4=GPSweek  5=dow
#   6=reflon  7=dE(m)  8=dN(m)  9=dU(m)  10=σE(m)  11=σN(m)  12=σU(m) …
#
# Displacements in metres → multiply by 1000 for mm.
# ---------------------------------------------------------------------------

def load_tenv3(path) -> pd.DataFrame:
    """
    Load a University of Nevada Reno ``.tenv3`` time-series file.

    Returns
    -------
    pd.DataFrame with canonical columns and DatetimeIndex.
    """
    path = Path(path)
    rows = []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Sta") or line.startswith("____"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                decyr = float(parts[2])
                e_m   = float(parts[7])
                n_m   = float(parts[8])
                u_m   = float(parts[9])
                se_m  = float(parts[10])
                sn_m  = float(parts[11])
                su_m  = float(parts[12])
            except (ValueError, IndexError):
                continue
            rows.append([decyr,
                         e_m * 1e3, n_m * 1e3, u_m * 1e3,
                         se_m * 1e3, sn_m * 1e3, su_m * 1e3])

    if not rows:
        raise ValueError(f"No valid data found in {path}")

    arr = np.array(rows)
    return _make_df(arr[:,0], arr[:,1], arr[:,2], arr[:,3],
                    arr[:,4], arr[:,5], arr[:,6])


# ---------------------------------------------------------------------------
# PBO / NOTA .pos
# ---------------------------------------------------------------------------
# Header section ends with a line starting "YYYYMMDD" or "*".
# Data columns (0-based):
#   0=YYYYMMDD  1=YYMMDD  2=MJD  3=week  4=dow  5=refN  6=refE  7=refU
#   8=dN(m)  9=dE(m)  10=dU(m)  11=σN(m)  12=σE(m)  13=σU(m)
#
# Displacements in metres → mm.
# ---------------------------------------------------------------------------

def load_pos(path) -> pd.DataFrame:
    """
    Load a PBO / NOTA ``.pos`` time-series file.

    Returns
    -------
    pd.DataFrame with canonical columns and DatetimeIndex.
    """
    path = Path(path)
    rows = []
    in_data = False
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("YYYYMMDD") or line.startswith("*"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                ds   = parts[0]            # YYYYMMDD
                year, month, day = int(ds[:4]), int(ds[4:6]), int(ds[6:8])
                n_m  = float(parts[8])
                e_m  = float(parts[9])
                u_m  = float(parts[10])
                sn_m = float(parts[11])
                se_m = float(parts[12])
                su_m = float(parts[13])
            except (ValueError, IndexError):
                continue
            ts    = pd.Timestamp(year=year, month=month, day=day)
            doy   = ts.day_of_year
            diy   = 366 if ts.is_leap_year else 365
            decyr = year + (doy - 1) / diy
            rows.append([decyr,
                         e_m * 1e3, n_m * 1e3, u_m * 1e3,
                         se_m * 1e3, sn_m * 1e3, su_m * 1e3])

    if not rows:
        raise ValueError(f"No valid data found in {path}")

    arr = np.array(rows)
    return _make_df(arr[:,0], arr[:,1], arr[:,2], arr[:,3],
                    arr[:,4], arr[:,5], arr[:,6])


# ---------------------------------------------------------------------------
# Auto-detect dispatcher
# ---------------------------------------------------------------------------

_EXT_TO_FMT = {
    ".rneu":  "rneu",
    ".tenv3": "tenv3",
    ".pos":   "pos",
}

_LOADERS = {
    "rneu":  load_rneu,
    "tenv3": load_tenv3,
    "pos":   load_pos,
}


def load_file(path, fmt: str = "auto") -> Tuple[pd.DataFrame, str]:
    """
    Load any supported GNSS time-series file with optional auto-detection.

    Parameters
    ----------
    path : str or Path
    fmt  : ``"auto"``, ``"rneu"``, ``"tenv3"``, or ``"pos"``.

    Returns
    -------
    ``(df, fmt_string)``
    """
    path = Path(path)

    if fmt == "auto":
        # Strip .gz suffix first
        bare = Path(path.stem) if path.suffix.lower() == ".gz" else path
        fmt  = _EXT_TO_FMT.get(bare.suffix.lower())
        if fmt is None:
            fmt = _sniff_format(path)

    if fmt not in _LOADERS:
        raise ValueError(
            f"Unknown format {fmt!r}. "
            f"Supported: {list(_LOADERS)}"
        )

    return _LOADERS[fmt](path), fmt


def _sniff_format(path: Path) -> str:
    """Inspect the first non-comment data line to guess the file format."""
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if re.match(r"^\d{8}$", parts[0]):
                return "pos"
            if re.match(r"^[A-Z0-9]{4}$", parts[0]) and len(parts) >= 13:
                return "tenv3"
            if re.match(r"^\d{4}\.\d+$", parts[0]):
                return "rneu"
            break
    raise ValueError(
        f"Could not auto-detect format of {path}. "
        "Pass fmt='rneu', 'tenv3', or 'pos' explicitly."
    )
