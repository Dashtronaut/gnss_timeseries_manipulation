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
# Two variants exist in the wild:
#
# Variant A — "nafixed" / detrended download (decimal-year first):
#   decyr  N(mm)  E(mm)  U(mm)  σN(mm)  σE(mm)  σU(mm)  [extra cols ignored]
#   First token is a decimal year, e.g. "1993.6377"
#
# Variant B — "stacov" / full download (date first, tab-separated):
#   YYYYMMDD  decyr  N(mm)  E(mm)  U(mm)  flag  σN(mm)  σE(mm)  σU(mm)  wrms  filename
#   First token is an 8-digit date, e.g. "19930821"
#   Column 5 ("flag", e.g. "rrr") is a non-numeric quality string, not data.
#
# Both variants put North before East in the file; we store east/north
# internally, so columns are swapped on load.
# ---------------------------------------------------------------------------

def _parse_rneu_line(parts: list) -> list | None:
    """
    Try to extract [decyr, N, E, U, sigN, sigE, sigU] from a split line.
    Returns None if the line matches neither known .rneu variant.
    """
    # Variant A: first token is a decimal year (e.g. "1993.6377")
    if re.match(r"^\d{4}\.\d+$", parts[0]):
        if len(parts) < 7:
            return None
        try:
            return [float(p) for p in parts[:7]]
        except ValueError:
            return None

    # Variant B: first token is an 8-digit YYYYMMDD date (e.g. "19930821")
    if re.match(r"^\d{8}$", parts[0]) and len(parts) >= 9:
        try:
            decyr   = float(parts[1])
            n, e, u = float(parts[2]), float(parts[3]), float(parts[4])
            # parts[5] is the quality flag string ("rrr") — skip it
            sn, se, su = float(parts[6]), float(parts[7]), float(parts[8])
            return [decyr, n, e, u, sn, se, su]
        except (ValueError, IndexError):
            return None

    return None


def load_rneu(path) -> pd.DataFrame:
    """
    Load a USGS ``.rneu`` time-series file.

    Automatically detects and handles both known variants:

    * **Variant A** (NA-fixed / detrended): ``decyr  N  E  U  σN  σE  σU``
    * **Variant B** (stacov / full):  ``YYYYMMDD  decyr  N  E  U  flag  σN  σE  σU  …``

    All displacement and sigma values are in mm.  North/East are swapped to
    the internal (east, north) column order.

    Returns
    -------
    pd.DataFrame with canonical columns and DatetimeIndex.
    """
    path = Path(path)
    rows: list = []
    first_lines: list = []   # kept for diagnostics
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            raw = line.rstrip("\r\n")
            stripped = raw.strip()
            if len(first_lines) < 8:
                first_lines.append(repr(raw))
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            parsed = _parse_rneu_line(parts)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        preview = "\n  ".join(first_lines) if first_lines else "(file appears empty)"
        raise ValueError(
            f"No valid data found in {path}\n\n"
            f"Recognised .rneu layouts:\n"
            f"  Variant A: decyr  N(mm)  E(mm)  U(mm)  σN  σE  σU\n"
            f"  Variant B: YYYYMMDD  decyr  N  E  U  flag  σN  σE  σU  …\n\n"
            f"First lines of file:\n  {preview}\n\n"
            f"Common causes:\n"
            f"  • File is HTML (server returned an error page) — open in a browser to check\n"
            f"  • An unrecognised column layout — please share the first few lines above\n"
        )

    arr = np.array(rows)
    # arr columns: [decyr, N, E, U, sigN, sigE, sigU]
    # internal storage order is east, north → swap columns 1 and 2 (and 4,5)
    return _make_df(arr[:, 0], arr[:, 2], arr[:, 1], arr[:, 3],
                    arr[:, 5], arr[:, 4], arr[:, 6])


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
            # rneu Variant A: decimal year first
            if re.match(r"^\d{4}\.\d+$", parts[0]):
                return "rneu"
            # UNR tenv3: 4-char station code first, 13+ columns
            if re.match(r"^[A-Z0-9]{4}$", parts[0]) and len(parts) >= 13:
                return "tenv3"
            # Both rneu Variant B (stacov) and pos start with YYYYMMDD.
            # Disambiguate: rneu stacov has a non-numeric quality flag
            # (e.g. "rrr") in column 5; pos's column 5 is a numeric refN.
            if re.match(r"^\d{8}$", parts[0]):
                if len(parts) >= 9:
                    try:
                        float(parts[5])
                    except ValueError:
                        return "rneu"   # column 5 isn't numeric → stacov flag
                if len(parts) >= 14:
                    return "pos"
                return "rneu"
            break
    raise ValueError(
        f"Could not auto-detect format of {path}. "
        "Pass fmt='rneu', 'tenv3', or 'pos' explicitly."
    )