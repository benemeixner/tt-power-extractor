# analysis_core.py
"""Core utilities to parse cycling time-trial CSV exports and extract the last N minutes of Power."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import pandas as pd


@dataclass
class SegmentSummary:
    minutes_requested: float
    seconds_available: float
    mean_power_w: float
    median_power_w: float
    max_power_w: float
    min_power_w: float


def _detect_header_idx(lines) -> int:
    for i, line in enumerate(lines):
        s = line.strip()
        if ("Time" in s) and ("Power" in s):
            return i
    raise ValueError("Could not find a header line containing both 'Time' and 'Power'.")


def _detect_sep(header_line: str) -> str:
    return ";" if header_line.count(";") >= header_line.count(",") else ","


def _to_seconds(time_series: pd.Series) -> np.ndarray:
    ts = pd.to_numeric(time_series, errors="coerce")
    if ts.notna().mean() > 0.95:
        vals = ts.to_numpy(dtype=float)
        vmax = np.nanmax(vals)
        # heuristic: very large -> milliseconds
        if vmax > 5000:
            return vals / 1000.0
        return vals

    s = time_series.astype(str).str.strip()

    def parse_one(x: str) -> float:
        if not x or x.lower() in {"nan", "none"}:
            return np.nan
        x = x.replace(",", ".")
        parts = x.split(":")
        try:
            if len(parts) == 3:  # HH:MM:SS
                h, m, sec = parts
                return float(h) * 3600 + float(m) * 60 + float(sec)
            if len(parts) == 2:  # MM:SS(.sss)
                m, sec = parts
                return float(m) * 60 + float(sec)
            return float(x)
        except Exception:
            return np.nan

    out = np.array([parse_one(v) for v in s.to_list()], dtype=float)
    if np.isfinite(out).mean() < 0.95:
        raise ValueError("Time column could not be parsed as numeric seconds or HH:MM:SS.")
    return out


def read_tt_file(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = _detect_header_idx(lines)
    sep = _detect_sep(lines[header_idx])

    # Skip everything before header + the units row right after header
    skiprows = list(range(header_idx)) + [header_idx + 1]
    decimal = "," if sep == ";" else "."

    df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skiprows, decimal=decimal)
    df.columns = [c.strip() for c in df.columns]

    # Case-insensitive lookup
    colmap = {c.lower(): c for c in df.columns}
    if "time" not in colmap or "power" not in colmap:
        raise ValueError(f"Expected columns 'Time' and 'Power'. Found: {list(df.columns)}")

    # Ensure numeric power
    df[colmap["power"]] = pd.to_numeric(df[colmap["power"]], errors="coerce")
    df = df.dropna(subset=[colmap["power"]]).reset_index(drop=True)
    return df


def extract_last_minutes(df: pd.DataFrame, minutes: Union[int, float]) -> Tuple[pd.DataFrame, SegmentSummary]:
    minutes = float(minutes)
    if minutes <= 0:
        raise ValueError("minutes must be > 0")

    colmap = {c.lower(): c for c in df.columns}
    time_col = colmap["time"]
    power_col = colmap["power"]

    t_s = _to_seconds(df[time_col])
    p = pd.to_numeric(df[power_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(t_s) & np.isfinite(p)
    t_s = t_s[mask]
    p = p[mask]

    end_s = float(np.nanmax(t_s))
    start_s = end_s - minutes * 60.0

    seg_mask = t_s >= start_s
    t_seg = t_s[seg_mask]
    p_seg = p[seg_mask]
    if len(t_seg) < 2:
        raise ValueError("Not enough samples in the requested segment.")

    t_rel = t_seg - float(np.nanmin(t_seg))

    seg = pd.DataFrame({"t_s": t_seg, "t_rel_s": t_rel, "Power": p_seg})

    seconds_available = float(np.nanmax(t_rel) - np.nanmin(t_rel))
    summ = SegmentSummary(
        minutes_requested=minutes,
        seconds_available=seconds_available,
        mean_power_w=float(np.nanmean(p_seg)),
        median_power_w=float(np.nanmedian(p_seg)),
        max_power_w=float(np.nanmax(p_seg)),
        min_power_w=float(np.nanmin(p_seg)),
    )
    return seg, summ


def to_1hz(seg: pd.DataFrame) -> pd.DataFrame:
    out = seg.copy()
    out["sec"] = np.floor(out["t_rel_s"]).astype(int)
    g = out.groupby("sec", as_index=False)["Power"].mean()
    g["t_rel_s"] = g["sec"].astype(float)
    return g[["t_rel_s", "Power"]]
