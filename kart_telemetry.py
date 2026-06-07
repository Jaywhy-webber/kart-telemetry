#!/usr/bin/env python3
"""
kart_telemetry.py
=================
Lap-by-lap analysis for AiM CSV kart telemetry (GPS speed + RPM, e.g. KF1 Kranji).

Built up in stages so you can run/trust each layer before moving on:

    Stage 1  lap times          -> times_from_beacons(), flying_laps()
    Stage 2  consistency        -> consistency_stats()
    Stage 3  sector breakdown   -> sector_table()
    Stage 4  finer detail       -> top_speed(), rpm_per_kmh()
    Stage 5  driver dynamics    -> braking_zones(), acceleration_zones(),
                                   rotation_events(), corner_speed_profile(),
                                   gg_stats()

Works on one driver or two. Give it one CSV for a solo report, or two CSVs
(e.g. Jamie.csv Joshua.csv) for a head-to-head.

Usage:
    python kart_telemetry.py Jamie.csv
    python kart_telemetry.py Jamie.csv Joshua.csv
    python kart_telemetry.py Jamie.csv --sectors 6
    python kart_telemetry.py Jamie.csv --plots
    python kart_telemetry.py Jamie.csv Joshua.csv --plots
    python kart_telemetry.py Jamie.csv --no-beacon
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Loading / parsing the AiM CSV
# ----------------------------------------------------------------------------
@dataclass
class Session:
    racer: str
    rate: float                  # logging rate, Hz
    beacons: list                # beacon-crossing timestamps, s
    seg_strings: list            # embedded "Segment Times" strings (for cross-check)
    df: pd.DataFrame             # cleaned sample data, numeric
    path: str = ""
    gps_beacons: bool = False    # True if beacons were reconstructed from GPS


def load_session(path, no_beacon=False, sf_latlon=None):
    """Parse an AiM CSV into a Session (metadata + clean numeric DataFrame).

    no_beacon : ignore the file's beacons and reconstruct lap crossings from
                GPS instead (also triggers automatically if the file has none).
    sf_latlon : optional (lat, lon) start/finish point for the GPS method.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    meta = {"racer": "?", "rate": 20.0, "beacons": [], "seg_strings": []}
    data_hdr_idx = None
    for i, row in enumerate(rows):
        if not row:
            continue
        key = row[0]
        if key == "Racer":
            meta["racer"] = row[1] if len(row) > 1 else "?"
        elif key == "Sample Rate":
            meta["rate"] = float(row[1])
        elif key == "Beacon Markers":
            meta["beacons"] = [float(x) for x in row[1:] if x.strip()]
        elif key == "Segment Times":
            meta["seg_strings"] = [x for x in row[1:] if x.strip()]
        elif key == "Time" and "GPS Speed" in row:
            data_hdr_idx = i          # channel-name row; units row follows
            break

    if data_hdr_idx is None:
        raise ValueError(f"{path}: could not find data header row")

    # Read data; first data row is the units row -> coerce to numeric drops it.
    df = pd.read_csv(path, skiprows=data_hdr_idx, header=0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Time"])
    df = df.reset_index(drop=True)

    sess = Session(
        racer=meta["racer"], rate=meta["rate"], beacons=meta["beacons"],
        seg_strings=meta["seg_strings"], df=df, path=path,
    )

    # Reconstruct beacons from GPS if asked, or if the file has none.
    sess.gps_beacons = False
    if no_beacon or len(sess.beacons) < 2:
        sess.beacons = beacons_from_gps(sess, sf_latlon=sf_latlon)
        sess.seg_strings = []          # no file segment times to cross-check
        sess.gps_beacons = True

    _annotate_laps(sess)
    return sess


def _annotate_laps(sess):
    """Tag every sample with its lap index and a per-lap distance (m from S/F)."""
    t = sess.df["Time"].values
    dist = sess.df["Distance on GPS Speed"].values
    lap_idx = np.full(len(t), -1, dtype=int)
    b = sess.beacons
    for k in range(len(b) - 1):
        m = (t >= b[k]) & (t < b[k + 1])
        lap_idx[m] = k
    sess.df["lap"] = lap_idx

    # per-lap distance, reset to 0 at each beacon crossing
    rel = np.full(len(t), np.nan)
    for k in range(len(b) - 1):
        m = lap_idx == k
        if m.any():
            rel[m] = dist[m] - np.interp(b[k], t, dist)
    sess.df["lap_dist"] = rel


# ----------------------------------------------------------------------------
# Beacon fallback: detect lap crossings from GPS when no beacon data exists
# ----------------------------------------------------------------------------
def beacons_from_gps(sess, sf_latlon=None, nsat_min=6, gate_m=8.0,
                     speed_floor=20.0, min_lap_s=5.0):
    """Reconstruct beacon-crossing timestamps from the GPS track.

    Builds a virtual start/finish line and times every crossing, interpolating
    between the two samples either side of the line for sub-sample accuracy.
    Validated against the real beacons on Jamie.csv to within ~50 ms/lap.

    sf_latlon : (lat, lon) of the start/finish point. If None, uses the median
                of positions at the file's own beacons (cross-check mode); in
                real beacon-less use, pass the S/F coordinate explicitly.
    gate_m    : only count a crossing if the kart passes within this many metres
                of the S/F point laterally (rejects far-away sign flips).
    """
    d = sess.df[sess.df["GPS Nsat"] >= nsat_min]
    t = d["Time"].values
    lat = d["GPS Latitude"].values
    lon = d["GPS Longitude"].values
    spd = d["GPS Speed"].values

    if sf_latlon is None:
        if len(sess.beacons) < 3:
            raise ValueError("No beacons to derive S/F from; pass sf_latlon=(lat,lon)")
        sf_lat = np.median([np.interp(b, t, lat) for b in sess.beacons[1:-1]])
        sf_lon = np.median([np.interp(b, t, lon) for b in sess.beacons[1:-1]])
    else:
        sf_lat, sf_lon = sf_latlon

    # local flat-earth metres around S/F
    R = 6371000.0
    x = np.radians(lon - sf_lon) * np.cos(np.radians(sf_lat)) * R
    y = np.radians(lat - sf_lat) * R

    # direction of travel through S/F (from the sample nearest the point)
    i0 = int(np.argmin(np.hypot(x, y)))
    i0 = min(max(i0, 1), len(x) - 2)
    dx, dy = x[i0 + 1] - x[i0 - 1], y[i0 + 1] - y[i0 - 1]
    norm = np.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm

    s = x * dx + y * dy          # along-track: 0 on the S/F line
    perp = -x * dy + y * dx      # lateral offset from S/F point

    crossings = []
    for k in range(1, len(s)):
        if s[k - 1] < 0 <= s[k] and abs(perp[k]) < gate_m and spd[k] > speed_floor:
            frac = -s[k - 1] / (s[k] - s[k - 1])
            crossings.append(t[k - 1] + frac * (t[k] - t[k - 1]))

    # de-duplicate near-simultaneous detections
    clean = []
    for c in crossings:
        if not clean or c - clean[-1] > min_lap_s:
            clean.append(round(float(c), 3))
    return clean


# ----------------------------------------------------------------------------
# Stage 1 - lap times
# ----------------------------------------------------------------------------
def times_from_beacons(sess):
    """Lap time of every beacon-to-beacon segment (s). Cross-checks the file's
    own Segment Times so we know the parse is right."""
    b = sess.beacons
    return [round(b[i + 1] - b[i], 3) for i in range(len(b) - 1)]


def flying_laps(sess, drop_outliers=True):
    """Return (indices, times) of clean flying laps.

    Beacon mode: drop the first segment (out/warm-up) and last (in lap).
    GPS mode: the slow out/in laps usually aren't detected as crossings at all,
    so we don't blindly trim both ends -- we rely on the outlier filter instead.
    Then optionally drop anything >3 s slower than the median (traffic / spin)."""
    times = times_from_beacons(sess)
    idx = list(range(len(times)))
    if not getattr(sess, "gps_beacons", False) and len(idx) > 2:
        idx, times = idx[1:-1], times[1:-1]      # drop out + in laps
    if drop_outliers and times:
        med = np.median(times)
        keep = [(i, t) for i, t in zip(idx, times) if t <= med + 3.0]
        idx, times = [i for i, _ in keep], [t for _, t in keep]
    return idx, times


# ----------------------------------------------------------------------------
# Stage 2 - consistency
# ----------------------------------------------------------------------------
def consistency_stats(sess):
    idx, times = flying_laps(sess)
    t = np.array(times)
    return {
        "racer": sess.racer,
        "n_flying": len(t),
        "best": round(t.min(), 3),
        "mean": round(t.mean(), 3),
        "median": round(float(np.median(t)), 3),
        "worst": round(t.max(), 3),
        "std": round(t.std(ddof=1), 3),        # sample std -> consistency
        "range": round(t.max() - t.min(), 3),
        "lap_idx": idx,
        "lap_times": [round(x, 3) for x in times],
    }


# ----------------------------------------------------------------------------
# Stage 3 - sector breakdown (equal-distance sectors)
# ----------------------------------------------------------------------------
def sector_table(sess, n_sectors=6):
    """Split each flying lap into N equal-distance sectors and time them.

    Returns a DataFrame: rows = flying laps, cols = S1..Sn (seconds in sector).
    Distance-based (not beacon-based) so it works without extra split beacons;
    swap in real corner distances later for the 'finer detail' stage."""
    idx, _ = flying_laps(sess)
    out = {}
    for k in idx:
        lap = sess.df[sess.df["lap"] == k]
        d = lap["lap_dist"].values
        tt = lap["Time"].values
        # use this lap's own length so small GPS distance drift cancels out
        L = d[-1]
        edges = np.linspace(0, L, n_sectors + 1)
        cross_t = np.interp(edges, d, tt)          # time at each sector boundary
        out[k] = np.diff(cross_t)
    table = pd.DataFrame(out, index=[f"S{i+1}" for i in range(n_sectors)]).T
    table.index.name = "lap"
    return table


# ----------------------------------------------------------------------------
# Stage 4 - finer detail hooks (top speed, gearing)
# ----------------------------------------------------------------------------
def top_speed(sess, nsat_min=6):
    d = sess.df[sess.df["GPS Nsat"] >= nsat_min]
    return round(d["GPS Speed"].max(), 2)


def rpm_per_kmh(sess, speed_floor=40, nsat_min=6):
    """km/h gained per 1000 rpm at steady throttle - a gearing signature.
    Higher km/h-per-1000rpm = taller gearing."""
    d = sess.df[(sess.df["GPS Nsat"] >= nsat_min) &
                (sess.df["GPS Speed"] >= speed_floor) &
                (sess.df["RPM"] > 0)]
    if len(d) < 50:
        return None
    slope = np.polyfit(d["RPM"], d["GPS Speed"], 1)[0]  # km/h per rpm
    return round(slope * 1000, 3)


# ----------------------------------------------------------------------------
# Stage 5 - driver dynamics
# ----------------------------------------------------------------------------
def _detect_zones(sess, col, thresh, above=False, min_dur_s=0.1, nsat_min=6):
    """Detect contiguous events where `col` crosses `thresh`. Returns list of dicts."""
    d = sess.df[
        (sess.df["GPS Nsat"] >= nsat_min) & (sess.df["lap"] >= 0)
    ].reset_index(drop=True)
    min_samples = max(1, int(min_dur_s * sess.rate))
    vals = d[col].values
    mask = (vals > thresh) if above else (vals < thresh)

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_samples:
            seg = d.iloc[s:e]
            events.append({
                "lap": int(d.iloc[s]["lap"]),
                "dist": round(float(d.iloc[s]["lap_dist"]), 1),
                "end_dist": round(float(d.iloc[e - 1]["lap_dist"]), 1),
                "peak": round(float(seg[col].min() if not above else seg[col].max()), 3),
                "duration": round((e - s) / sess.rate, 3),
                "entry_speed": round(float(d.iloc[s]["GPS Speed"]), 1),
            })
    return events


def braking_zones(sess, lon_thresh=-0.3, min_dur_s=0.1):
    """Braking events: GPS LonAcc < lon_thresh sustained >= min_dur_s seconds."""
    return _detect_zones(sess, "GPS LonAcc", lon_thresh, above=False, min_dur_s=min_dur_s)


def acceleration_zones(sess, lon_thresh=0.15, min_dur_s=0.1):
    """Acceleration events: GPS LonAcc > lon_thresh sustained >= min_dur_s seconds."""
    return _detect_zones(sess, "GPS LonAcc", lon_thresh, above=True, min_dur_s=min_dur_s)


def rotation_events(sess, gyro_thresh=80.0, nsat_min=6):
    """Detect kart rotation: actual yaw rate exceeds expected (from speed/radius)
    by more than gyro_thresh deg/s, indicating oversteer or snap."""
    d = sess.df[
        (sess.df["GPS Nsat"] >= nsat_min) & (sess.df["lap"] >= 0) &
        (sess.df["GPS Radius"] > 0) & (sess.df["GPS Radius"] < 500)
    ].reset_index(drop=True)
    if len(d) < 10:
        return []

    spd_ms = d["GPS Speed"].values / 3.6
    expected_yaw = (spd_ms / d["GPS Radius"].values) * (180.0 / np.pi)
    oversteer = np.abs(d["GPS Gyro"].values) - expected_yaw
    mask = oversteer > gyro_thresh

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        if (e - s) >= 2:
            events.append({
                "lap": int(d.iloc[s]["lap"]),
                "dist": round(float(d.iloc[s]["lap_dist"]), 1),
                "peak_oversteer": round(float(oversteer[s:e].max()), 1),
                "duration": round((e - s) / sess.rate, 3),
            })
    return events


def corner_speed_profile(sess, n_sectors=6):
    """Per sector per flying lap: apex (min), entry, and exit speed in km/h."""
    idx, _ = flying_laps(sess)
    records = []
    for k in idx:
        lap = sess.df[sess.df["lap"] == k]
        d = lap["lap_dist"].values
        spd = lap["GPS Speed"].values
        L = d[-1]
        edges = np.linspace(0, L, n_sectors + 1)
        row = {"lap": k}
        for i in range(n_sectors):
            m = (d >= edges[i]) & (d < edges[i + 1])
            if m.any():
                s_spd = spd[m]
                row[f"S{i+1}_apex"] = round(float(s_spd.min()), 1)
                row[f"S{i+1}_entry"] = round(float(s_spd[0]), 1)
                row[f"S{i+1}_exit"] = round(float(s_spd[-1]), 1)
        records.append(row)
    return pd.DataFrame(records).set_index("lap")


def gg_stats(sess, nsat_min=6):
    """Lateral/longitudinal g envelope and combined peak."""
    d = sess.df[sess.df["GPS Nsat"] >= nsat_min]
    lat = d["GPS LatAcc"].values
    lon = d["GPS LonAcc"].values
    combined = np.hypot(lat, lon)
    return {
        "lat_pos": round(float(lat.max()), 3),
        "lat_neg": round(float(lat.min()), 3),
        "lon_accel": round(float(lon.max()), 3),
        "lon_brake": round(float(lon.min()), 3),
        "combined_peak": round(float(combined.max()), 3),
    }


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_session(sess, n_sectors=6):
    """4-panel analysis figure saved alongside the CSV."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{sess.racer} — Session Analysis", fontsize=14, fontweight="bold")

    idx, times = flying_laps(sess)
    best_lap_idx = idx[int(np.argmin(times))]

    # --- Panel 1: Speed trace with braking / accel zones ---
    ax = axes[0, 0]
    for k in idx:
        lap = sess.df[sess.df["lap"] == k]
        if k == best_lap_idx:
            ax.plot(lap["lap_dist"], lap["GPS Speed"], color="tab:blue", lw=1.8, zorder=3)
        else:
            ax.plot(lap["lap_dist"], lap["GPS Speed"], color="#bbbbbb", lw=0.6, zorder=1)

    best_lap = sess.df[sess.df["lap"] == best_lap_idx]
    dist_bl = best_lap["lap_dist"].values
    spd_bl = best_lap["GPS Speed"].values
    lon_bl = best_lap["GPS LonAcc"].values
    spd_ceil = spd_bl.max() * 1.05
    ax.fill_between(dist_bl, 0, spd_ceil, where=(lon_bl < -0.3),
                    alpha=0.18, color="red", zorder=2)
    ax.fill_between(dist_bl, 0, spd_ceil, where=(lon_bl > 0.15),
                    alpha=0.18, color="green", zorder=2)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed Trace  (blue = best lap)")
    ax.legend(handles=[
        mpatches.Patch(color="red", alpha=0.4, label="braking"),
        mpatches.Patch(color="green", alpha=0.4, label="acceleration"),
    ], fontsize=8)

    # --- Panel 2: G-G diagram ---
    ax = axes[0, 1]
    d = sess.df[sess.df["GPS Nsat"] >= 6]
    sc = ax.scatter(d["GPS LatAcc"], d["GPS LonAcc"],
                    c=d["GPS Speed"], cmap="plasma", s=2, alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Speed (km/h)")
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.25)
    ax.axhline(0, color="k", lw=0.5, alpha=0.25)
    ax.axvline(0, color="k", lw=0.5, alpha=0.25)
    ax.set_xlabel("Lateral g")
    ax.set_ylabel("Longitudinal g")
    ax.set_title("G-G Diagram  (colour = speed)")
    ax.set_aspect("equal")

    # --- Panel 3: RPM vs Speed ---
    ax = axes[1, 0]
    d2 = sess.df[
        (sess.df["GPS Nsat"] >= 6) &
        (sess.df["GPS Speed"] > 20) &
        (sess.df["RPM"] > 0)
    ]
    sc2 = ax.scatter(d2["RPM"], d2["GPS Speed"],
                     c=d2["GPS LonAcc"], cmap="RdYlGn",
                     s=2, alpha=0.4, vmin=-0.5, vmax=0.5)
    plt.colorbar(sc2, ax=ax, label="LonAcc (g)")
    if len(d2) >= 50:
        coeffs = np.polyfit(d2["RPM"], d2["GPS Speed"], 1)
        rpm_range = np.array([d2["RPM"].min(), d2["RPM"].max()])
        ax.plot(rpm_range, np.polyval(coeffs, rpm_range), "b-", lw=1.5,
                label=f"{coeffs[0]*1000:.3f} km/h per 1000 rpm")
        ax.legend(fontsize=8)
    ax.set_xlabel("RPM")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("RPM vs Speed  (colour = LonAcc)")

    # --- Panel 4: Lap time bar chart ---
    ax = axes[1, 1]
    lap_nums = list(range(1, len(times) + 1))
    best_i = int(np.argmin(times))
    colors = ["tab:blue" if i == best_i else "tab:gray" for i in range(len(times))]
    ax.bar(lap_nums, times, color=colors, edgecolor="white", linewidth=0.5)
    mean_t = float(np.mean(times))
    ax.axhline(mean_t, color="tab:orange", lw=1.5, linestyle="--",
               label=f"mean {mean_t:.3f}s")
    # Lap time trend
    if len(times) > 2:
        slope = np.polyfit(range(len(times)), times, 1)[0]
        ax.set_xlabel(f"Lap  (trend {slope*1000:+.0f} ms/lap)")
    else:
        ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Lap Times  (blue = best)")
    ax.legend(fontsize=8)
    ax.set_xticks(lap_nums)

    plt.tight_layout()
    save_path = _plot_path(sess.path, f"{sess.racer.replace(' ', '_')}_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"          saved: {save_path}")
    plt.close()


def plot_compare(a, b, n_sectors=6):
    """3-panel head-to-head figure: speed trace, sector heatmap, G-G comparison."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{a.racer}  vs  {b.racer}", fontsize=14, fontweight="bold")

    idx_a, ta = flying_laps(a)
    idx_b, tb = flying_laps(b)
    best_a = idx_a[int(np.argmin(ta))]
    best_b = idx_b[int(np.argmin(tb))]

    # --- Panel 1: Best lap speed trace overlay ---
    ax = axes[0]
    lap_a = a.df[a.df["lap"] == best_a]
    lap_b = b.df[b.df["lap"] == best_b]
    ax.plot(lap_a["lap_dist"], lap_a["GPS Speed"], color="tab:blue", lw=1.5, label=a.racer)
    ax.plot(lap_b["lap_dist"], lap_b["GPS Speed"], color="tab:orange", lw=1.5, label=b.racer)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Best Lap Speed Trace")
    ax.legend(fontsize=9)

    # --- Panel 2: Sector delta heatmap (each lap vs own session mean) ---
    ax = axes[1]
    tbl_a = sector_table(a, n_sectors)
    tbl_b = sector_table(b, n_sectors)
    delta_a = tbl_a - tbl_a.mean()
    delta_b = tbl_b - tbl_b.mean()
    delta_a.index = [f"{a.racer[:5]} L{i+1}" for i in range(len(delta_a))]
    delta_b.index = [f"{b.racer[:5]} L{i+1}" for i in range(len(delta_b))]
    combined = pd.concat([delta_a, delta_b])
    vals = combined.values
    vmax = max(float(np.nanmax(np.abs(vals))), 0.3)
    im = ax.imshow(vals, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Delta from own mean (s)")
    ax.set_xticks(range(n_sectors))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_sectors)])
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined.index, fontsize=7)
    ax.set_title("Sector Delta Heatmap\n(green = faster than own avg)")

    # --- Panel 3: G-G comparison ---
    ax = axes[2]
    da = a.df[a.df["GPS Nsat"] >= 6]
    db = b.df[b.df["GPS Nsat"] >= 6]
    ax.scatter(da["GPS LatAcc"], da["GPS LonAcc"],
               s=1, alpha=0.25, color="tab:blue", label=a.racer)
    ax.scatter(db["GPS LatAcc"], db["GPS LonAcc"],
               s=1, alpha=0.25, color="tab:orange", label=b.racer)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.25)
    ax.axhline(0, color="k", lw=0.5, alpha=0.25)
    ax.axvline(0, color="k", lw=0.5, alpha=0.25)
    ax.set_xlabel("Lateral g")
    ax.set_ylabel("Longitudinal g")
    ax.set_title("G-G Comparison")
    ax.legend(fontsize=9, markerscale=6)
    ax.set_aspect("equal")

    plt.tight_layout()
    name_a = a.racer.replace(" ", "_")
    name_b = b.racer.replace(" ", "_")
    save_path = _plot_path(a.path, f"compare_{name_a}_{name_b}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"          saved: {save_path}")
    plt.close()


def _plot_path(csv_path, filename):
    directory = os.path.dirname(os.path.abspath(csv_path))
    return os.path.join(directory, filename)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(sess, n_sectors=6):
    print(f"\n===== {sess.racer}  ({sess.path}) =====")
    src = "GPS-derived" if sess.gps_beacons else "file beacon"
    raw = times_from_beacons(sess)
    print(f"[stage 1] {len(raw)} {src} segments: "
          f"{', '.join(f'{x:.3f}' for x in raw)}")
    if sess.gps_beacons:
        print("          (lap boundaries reconstructed from GPS start/finish line)")
    else:
        print(f"          file Segment Times match parse: "
              f"{_segments_match(sess)}")

    s = consistency_stats(sess)
    print(f"[stage 2] flying laps n={s['n_flying']}  "
          f"best={s['best']}  mean={s['mean']}  "
          f"std={s['std']}  range={s['range']}")
    print(f"          lap times: {s['lap_times']}")

    tbl = sector_table(sess, n_sectors)
    print(f"[stage 3] sector times (s), {tbl.shape[1]} equal-distance sectors:")
    display = tbl.round(3).astype(object)
    display.loc["mean"] = tbl.mean().round(3)
    display.loc["std"]  = tbl.std(ddof=1).round(3)
    print(display.to_string())

    print(f"[stage 4] top speed (>=6 sat): {top_speed(sess)} km/h | "
          f"gearing: {rpm_per_kmh(sess)} km/h per 1000 rpm")

    _report_stage5(sess, n_sectors)


def _report_stage5(sess, n_sectors=6):
    idx, times = flying_laps(sess)
    n_flying = len(idx)

    brakes = braking_zones(sess)
    accels = acceleration_zones(sess)
    rot    = rotation_events(sess)
    gg     = gg_stats(sess)
    csp    = corner_speed_profile(sess, n_sectors)

    # Braking
    if brakes:
        n_per_lap = len(brakes) / n_flying if n_flying else 0
        peak_dec  = min(e["peak"] for e in brakes)
        mean_dur  = float(np.mean([e["duration"] for e in brakes]))
        bp_by_lap = {}
        for e in brakes:
            bp_by_lap.setdefault(e["lap"], []).append(e["dist"])
        first_bp = [v[0] for v in bp_by_lap.values()]
        bp_std = round(float(np.std(first_bp, ddof=1)), 1) if len(first_bp) > 1 else None
        brake_str = (f"{len(brakes)} zones ({n_per_lap:.1f}/lap)  "
                     f"peak {peak_dec:.2f}g  dur {mean_dur:.2f}s"
                     + (f"  brake-pt std {bp_std}m" if bp_std is not None else ""))
    else:
        brake_str = "none detected"

    # Acceleration
    if accels:
        n_per_lap_a = len(accels) / n_flying if n_flying else 0
        peak_acc    = max(e["peak"] for e in accels)
        accel_str   = (f"{len(accels)} zones ({n_per_lap_a:.1f}/lap)  "
                       f"peak +{peak_acc:.2f}g")
    else:
        accel_str = "none detected"

    # Rotation
    if rot:
        worst = max(rot, key=lambda e: e["peak_oversteer"])
        rot_str = (f"{len(rot)} events  "
                   f"max +{worst['peak_oversteer']:.0f} deg/s  "
                   f"(lap {worst['lap']+1}, dist {worst['dist']:.0f}m)")
    else:
        rot_str = "none detected"

    # G-G
    gg_str = (f"lat {gg['lat_neg']:.2f}/{gg['lat_pos']:+.2f}g  "
              f"lon {gg['lon_brake']:.2f}/{gg['lon_accel']:+.2f}g  "
              f"combined peak {gg['combined_peak']:.2f}g")

    # Corner speed — show lowest-apex sector
    apex_cols = [c for c in csp.columns if c.endswith("_apex")]
    if apex_cols:
        apex_means = {c.replace("_apex", ""): csp[c].mean() for c in apex_cols}
        slowest = min(apex_means, key=apex_means.get)
        fastest = max(apex_means, key=apex_means.get)
        cs_str = (f"{slowest} apex {apex_means[slowest]:.1f} km/h (slowest)  "
                  f"{fastest} apex {apex_means[fastest]:.1f} km/h (fastest)")
    else:
        cs_str = "N/A"

    # Lap time trend
    if len(times) > 2:
        slope = np.polyfit(range(len(times)), times, 1)[0]
        trend_str = f"{slope*1000:+.0f} ms/lap"
    else:
        trend_str = "N/A"

    print(f"[stage 5] braking:    {brake_str}")
    print(f"          accel:      {accel_str}")
    print(f"          rotation:   {rot_str}")
    print(f"          G-G:        {gg_str}")
    print(f"          corner spd: {cs_str}")
    print(f"          lap trend:  {trend_str}")


def compare(a, b, n_sectors=6):
    sa, sb = consistency_stats(a), consistency_stats(b)
    print("\n========== HEAD-TO-HEAD ==========")
    print(f"{'metric':<14}{a.racer:>12}{b.racer:>12}{'delta':>10}")
    for key, label in [("best", "best lap"), ("mean", "avg lap"),
                       ("std", "consistency"), ("range", "spread")]:
        d = round(sb[key] - sa[key], 3)
        print(f"{label:<14}{sa[key]:>12}{sb[key]:>12}{d:>+10}")
    print(f"{'top speed':<14}{top_speed(a):>12}{top_speed(b):>12}"
          f"{round(top_speed(b)-top_speed(a),2):>+10}")
    print(f"{'km/h/1000rpm':<14}{rpm_per_kmh(a):>12}{rpm_per_kmh(b):>12}"
          f"{round(rpm_per_kmh(b)-rpm_per_kmh(a),3):>+10}")

    ta_tbl, tb_tbl = sector_table(a, n_sectors), sector_table(b, n_sectors)
    ta, tb = ta_tbl.mean(), tb_tbl.mean()
    ta_std, tb_std = ta_tbl.std(ddof=1), tb_tbl.std(ddof=1)
    ra, rb = a.racer[:6], b.racer[:6]
    print("\nsector breakdown (mean / std, s):")
    print(f"{'sector':<8}{ra+' mean':>10}  {rb+' mean':<10}{'delta':>8}"
          f"    {ra+' std':>8}  {rb+' std':<8}")
    print("-" * 62)
    for s in ta.index:
        d = round(tb[s] - ta[s], 3)
        print(f"{s:<8}{ta[s]:>10.3f}  {tb[s]:<10.3f}{d:>+8.3f}"
              f"    {ta_std[s]:>8.3f}  {tb_std[s]:<8.3f}")

    # Corner apex speed
    csp_a = corner_speed_profile(a, n_sectors)
    csp_b = corner_speed_profile(b, n_sectors)
    apex_cols = [c for c in csp_a.columns if c.endswith("_apex")]
    if apex_cols:
        sectors = [c.replace("_apex", "") for c in apex_cols]
        print("\ncorner apex speed (mean km/h):")
        print(f"{'sector':<8}{a.racer[:8]:>10}  {b.racer[:8]:<10}{'delta':>8}")
        print("-" * 42)
        for s in sectors:
            avg_a = csp_a[f"{s}_apex"].mean()
            avg_b = csp_b[f"{s}_apex"].mean()
            d = round(avg_b - avg_a, 1)
            print(f"{s:<8}{avg_a:>10.1f}  {avg_b:<10.1f}{d:>+8.1f}")

    # G-G comparison
    gg_a, gg_b = gg_stats(a), gg_stats(b)
    print("\nG-G envelope:")
    print(f"{'metric':<20}{a.racer[:8]:>10}  {b.racer[:8]:<10}{'delta':>8}")
    print("-" * 52)
    for key, label in [("lat_pos", "lat peak +g"), ("lat_neg", "lat peak -g"),
                       ("lon_brake", "lon brake"), ("lon_accel", "lon accel"),
                       ("combined_peak", "combined peak")]:
        va, vb = gg_a[key], gg_b[key]
        d = round(vb - va, 3)
        print(f"{label:<20}{va:>10.3f}  {vb:<10.3f}{d:>+8.3f}")


def _segments_match(sess):
    """Compare beacon-diff lap times to the file's printed Segment Times."""
    def parse_mmss(s):
        m, sec = s.split(":")
        return round(int(m) * 60 + float(sec), 3)
    printed = [parse_mmss(s) for s in sess.seg_strings]
    # segment_times[0] is the time to first beacon; beacon diffs start after it
    diffs = times_from_beacons(sess)
    return printed[1:] == [round(x, 3) for x in diffs]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AiM kart telemetry lap analysis")
    ap.add_argument("files", nargs="+", help="one or two AiM CSV files")
    ap.add_argument("--sectors", type=int, default=6)
    ap.add_argument("--no-beacon", action="store_true",
                    help="ignore file beacons; reconstruct laps from GPS")
    ap.add_argument("--plots", action="store_true",
                    help="generate matplotlib analysis plots (saved as PNG)")
    args = ap.parse_args()

    sessions = [load_session(p, no_beacon=args.no_beacon) for p in args.files]
    for s in sessions:
        report(s, n_sectors=args.sectors)
    if len(sessions) == 2:
        compare(sessions[0], sessions[1], n_sectors=args.sectors)

    if args.plots:
        print("\n[plots]")
        for s in sessions:
            plot_session(s, n_sectors=args.sectors)
        if len(sessions) == 2:
            plot_compare(sessions[0], sessions[1], n_sectors=args.sectors)


if __name__ == "__main__":
    main()
