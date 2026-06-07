#!/usr/bin/env python3
"""
dashboard.py — multi-tab HTML dashboard for AiM kart telemetry.

Usage:
    python dashboard.py Jamie.csv
    python dashboard.py Jamie.csv Joshua.csv
    python dashboard.py Jamie.csv Joshua.csv ThirdDriver.csv
    python dashboard.py Jamie.csv --sectors 8
"""

import argparse, os
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kart_telemetry as kt

# ── colour palette ────────────────────────────────────────────────────────────
_COLORS = ["#E8002D", "#0090D0", "#FF8000", "#00D2BE",
           "#9B59B6", "#39B54A", "#FF87BC", "#B6BABD"]
_BG  = "#15151E"   # chart background — matches --card
_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor=_BG, plot_bgcolor=_BG,
    font=dict(color="#E8E8F0", size=11, family="'Titillium Web','Helvetica Neue',Arial,sans-serif"),
    margin=dict(l=60, r=60, t=88, b=55),
)
_MAP_LAYOUT = {k: v for k, v in _LAYOUT.items() if k != "margin"}
_GRID = dict(gridcolor="#252535", zerolinecolor="#2E2E42", tickfont=dict(size=10))

_ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services"
         "/World_Imagery/MapServer/tile/{z}/{y}/{x}")

# GPS registration offset — corrects systematic displacement vs satellite imagery.
# Set via --lat-offset / --lon-offset CLI flags; zero by default.
_OFF = [0.0, 0.0]          # [lat_offset_degrees, lon_offset_degrees]
def _la(a): return a + _OFF[0]   # apply lat offset
def _lo(a): return a + _OFF[1]   # apply lon offset

_TILE_CACHE: dict = {}   # (zoom, ty, tx) → PIL.Image — avoids re-downloading per build

def _best_tile_zoom(lats, lons, max_tiles=64, pad=1):
    """Highest zoom where the GPS extent (plus pad tiles each side) fits in max_tiles."""
    for z in range(19, 13, -1):
        n = 2 ** z
        tx0 = int((float(np.min(lons)) + 180) / 360 * n) - pad
        tx1 = int((float(np.max(lons)) + 180) / 360 * n) + pad
        lat_r_hi = np.radians(float(np.max(lats)))
        lat_r_lo = np.radians(float(np.min(lats)))
        ty0 = int((1 - np.log(np.tan(lat_r_hi) + 1/np.cos(lat_r_hi)) / np.pi) / 2 * n) - pad
        ty1 = int((1 - np.log(np.tan(lat_r_lo) + 1/np.cos(lat_r_lo)) / np.pi) / 2 * n) + pad
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles:
            return z
    return 14

def _fetch_tile(zoom, ty, tx):
    key = (zoom, ty, tx)
    if key not in _TILE_CACHE:
        try:
            import requests
            from PIL import Image
            import io as _io2
            url = _ESRI.format(z=zoom, y=ty, x=tx)
            r = requests.get(url, timeout=10,
                             headers={"User-Agent": "kart-telemetry/1.0"})
            _TILE_CACHE[key] = (Image.open(_io2.BytesIO(r.content)).convert("RGB")
                                if r.status_code == 200 else None)
        except Exception:
            _TILE_CACHE[key] = None
    return _TILE_CACHE[key]

def _sat_snapshot(lats, lons, zoom, pad=1):
    """
    Download + stitch ESRI tiles.  Returns (b64_jpeg, ox_gpx, oy_gpx, W, H).
    ox_gpx / oy_gpx are the global-pixel offsets of the top-left corner.
    """
    import base64, io as _io2
    from PIL import Image
    TILE = 256
    n = 2 ** zoom

    def _dt(lat, lon):
        tx = int((lon + 180) / 360 * n)
        lat_r = np.radians(lat)
        ty = int((1 - np.log(np.tan(lat_r) + 1/np.cos(lat_r)) / np.pi) / 2 * n)
        return tx, max(0, min(ty, n - 1))

    tx0, ty0 = _dt(float(np.max(lats)), float(np.min(lons)))  # top-left tile
    tx1, ty1 = _dt(float(np.min(lats)), float(np.max(lons)))  # bot-right tile
    tx0 -= pad; tx1 += pad; ty0 -= pad; ty1 += pad

    W = (tx1 - tx0 + 1) * TILE
    H = (ty1 - ty0 + 1) * TILE
    canvas = Image.new("RGB", (W, H), (60, 60, 60))

    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = _fetch_tile(zoom, ty, tx)
            if tile:
                canvas.paste(tile, ((tx - tx0) * TILE, (ty - ty0) * TILE))

    buf = _io2.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, tx0 * TILE, ty0 * TILE, W, H

def _gps2px(lats, lons, ox, oy, zoom):
    """GPS lat/lon arrays → (xs, ys) in stitched-image pixel space (y increases down)."""
    n = (2 ** zoom) * 256
    xs = (lons + 180) / 360 * n - ox
    lat_r = np.radians(lats)
    ys = (1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / np.pi) / 2 * n - oy
    return xs, ys


def find_offset(sess, zoom=18):
    """
    Automatically estimate the GPS registration offset by cross-correlating
    the GPS trace image with edges detected in ESRI satellite tiles.

    Requires: requests, Pillow, scipy
    Returns: (lat_offset, lon_offset) in degrees — add these to GPS coordinates.
    """
    try:
        import requests
        from PIL import Image, ImageDraw
        from scipy.ndimage import gaussian_filter, sobel as _sobel
        from scipy.signal import fftconvolve
        import io as _io
    except ImportError as exc:
        print(f"[auto-align] Missing dependency: {exc}")
        print("  pip install requests Pillow scipy")
        return 0.0, 0.0

    TILE = 256
    n_global = (2 ** zoom) * TILE   # total pixels along one axis at this zoom

    # ── tile / pixel helpers ──────────────────────────────────────────────────
    def _deg2tile(lat, lon):
        n = 2 ** zoom
        tx = int((lon + 180) / 360 * n)
        lat_r = np.radians(lat)
        ty = int((1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / np.pi) / 2 * n)
        return tx, max(0, min(ty, n - 1))

    def _deg2gpx(lat, lon):
        """Global pixel coordinates (float)."""
        gx = (lon + 180) / 360 * n_global
        lat_r = np.radians(lat)
        gy = (1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / np.pi) / 2 * n_global
        return gx, gy

    # ── bounding box ─────────────────────────────────────────────────────────
    d = sess.df[sess.df["GPS Nsat"] >= 6]
    lats = d["GPS Latitude"].values
    lons = d["GPS Longitude"].values

    PAD = 2   # extra tiles around bounding box
    tx0, ty1 = _deg2tile(lats.max(), lons.min())   # top-left  (high lat = low ty)
    tx1, ty0 = _deg2tile(lats.min(), lons.max())   # bot-right
    tx0 -= PAD; tx1 += PAD; ty0 -= PAD; ty1 += PAD
    W = (tx1 - tx0 + 1) * TILE
    H = (ty1 - ty0 + 1) * TILE

    if W > 5120 or H > 5120:
        print(f"[auto-align] Coverage too large ({W}×{H} px). Try --align-zoom 17.")
        return 0.0, 0.0

    # ── download + stitch satellite tiles ────────────────────────────────────
    sat_img = Image.new("RGB", (W, H), (100, 100, 100))
    sess_req = requests.Session()
    sess_req.headers["User-Agent"] = "kart-telemetry/1.0"
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f"[auto-align] Downloading {n_tiles} tiles at zoom {zoom} …")
    failed = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            url = _ESRI.format(z=zoom, y=ty, x=tx)
            try:
                r = sess_req.get(url, timeout=10)
                if r.status_code == 200:
                    tile = Image.open(_io.BytesIO(r.content)).convert("RGB")
                    sat_img.paste(tile, ((tx - tx0) * TILE, (ty - ty0) * TILE))
                else:
                    failed += 1
            except Exception:
                failed += 1
    if failed:
        print(f"[auto-align] {failed}/{n_tiles} tiles failed — alignment may be less accurate.")

    # ── satellite: greyscale → Sobel edges → normalise ───────────────────────
    sat_g = np.array(sat_img.convert("L"), dtype=np.float32)
    sat_g = gaussian_filter(sat_g, sigma=1.5)
    ex = _sobel(sat_g, axis=1)
    ey = _sobel(sat_g, axis=0)
    sat_edges = np.hypot(ex, ey).astype(np.float32)
    sat_edges /= sat_edges.max() + 1e-9

    # ── GPS: draw ALL flying laps as thick lines → Gaussian smear ────────────
    ox = tx0 * TILE   # pixel-space origin of the stitched image
    oy = ty0 * TILE

    bk, all_idx, _ = _best_k(sess)
    gps_img = Image.new("L", (W, H), 0)
    drw = ImageDraw.Draw(gps_img)
    for k in all_idx:
        lap = d[d["lap"] == k].sort_values("lap_dist")
        pts = []
        for _, row in lap.iterrows():
            gx, gy = _deg2gpx(row["GPS Latitude"], row["GPS Longitude"])
            pts.append((int(gx - ox), int(gy - oy)))
        if len(pts) > 1:
            drw.line(pts, fill=255, width=6)

    gps_arr = gaussian_filter(np.array(gps_img, dtype=np.float32), sigma=4)
    gps_arr /= gps_arr.max() + 1e-9

    # ── 2-D cross-correlation within a ±SEARCH pixel search window ───────────
    SEARCH = 100   # ≈ 27 m at zoom 18
    cy, cx = H // 2, W // 2
    # Crop the GPS kernel to the search window to keep FFT fast
    gps_crop = gps_arr[cy - SEARCH: cy + SEARCH, cx - SEARCH: cx + SEARCH]
    if gps_crop.max() == 0:
        print("[auto-align] GPS trace produced a blank image — check GPS quality.")
        return 0.0, 0.0

    # fftconvolve(A, B_flipped) == cross-correlation(A, B)
    corr = fftconvolve(sat_edges, gps_crop[::-1, ::-1], mode="same")

    # Look only in the central ±SEARCH region of the full correlation map
    cy_c, cx_c = corr.shape[0] // 2, corr.shape[1] // 2
    region = corr[cy_c - SEARCH: cy_c + SEARCH, cx_c - SEARCH: cx_c + SEARCH]
    py_rel, px_rel = np.unravel_index(np.argmax(region), region.shape)

    # Pixel shift needed to move GPS onto satellite
    dy_px = py_rel - SEARCH   # + = GPS must move south  (increase pixel y)
    dx_px = px_rel - SEARCH   # + = GPS must move east   (increase pixel x)

    # ── pixel shift → degrees ─────────────────────────────────────────────────
    lat_c = float(np.median(lats))
    # lon: uniform Mercator scaling
    dlon = dx_px * 360.0 / n_global
    # lat: Mercator derivative at lat_c  →  dlat ≈ -dy * 360 * cos(lat_c) / n
    dlat = -dy_px * 360.0 * np.cos(np.radians(lat_c)) / n_global

    # ── sanity check ─────────────────────────────────────────────────────────
    dlat_m = abs(dlat) * 111320
    dlon_m = abs(dlon) * 111320 * np.cos(np.radians(lat_c))
    if dlat_m > 80 or dlon_m > 80:
        print(f"[auto-align] Offset too large ({dlat_m:.0f} m N, {dlon_m:.0f} m E) — "
              "correlation likely confused by non-track features.")
        print("  Measure manually and use --lat-offset / --lon-offset.")
        return 0.0, 0.0

    dlat = round(dlat, 6)
    dlon = round(dlon, 6)
    ns = "N" if dlat >= 0 else "S"
    ew = "E" if dlon >= 0 else "W"
    print(f"[auto-align] Offset detected: lat {dlat:+.6f}°  lon {dlon:+.6f}°")
    print(f"             ≈ {dlat_m:.1f} m {ns},  {dlon_m:.1f} m {ew}")
    return dlat, dlon

def _align_controls_html(div_id, lat_c, zoom, group=None, key=None):
    """Compact GPS nudge pad rendered below a satellite map figure.
    group: comma-separated list of ALL satellite map div IDs that move together.
    key:   shared state key — all bars with the same key share one offset accumulator.
    """
    grp = group if group else div_id
    k   = key   if key   else div_id
    return (
        f'<div class="align-bar" data-chart="{div_id}" '
        f'data-lat-c="{lat_c:.6f}" data-zoom="{zoom}" '
        f'data-group="{grp}" data-key="{k}">'
        f'<span class="align-label">GPS ALIGN</span>'
        f'<div class="align-pad">'
        f'  <button class="align-arrow" onclick="nudgeGPS(this,\'N\')" title="North">↑</button>'
        f'  <div class="align-horiz">'
        f'    <button class="align-arrow" onclick="nudgeGPS(this,\'W\')" title="West">←</button>'
        f'    <button class="align-arrow align-reset-xy" onclick="resetAlignGPS(this)" title="Reset">○</button>'
        f'    <button class="align-arrow" onclick="nudgeGPS(this,\'E\')" title="East">→</button>'
        f'  </div>'
        f'  <button class="align-arrow" onclick="nudgeGPS(this,\'S\')" title="South">↓</button>'
        f'</div>'
        f'<select class="align-step">'
        f'  <option value="0.3">0.3 m</option>'
        f'  <option value="0.5">0.5 m</option>'
        f'  <option value="1" selected>1 m</option>'
        f'  <option value="2">2 m</option>'
        f'  <option value="5">5 m</option>'
        f'</select>'
        f'<span class="align-readout" id="{div_id}_readout">'
        f'lat +0.000000°&ensp;lon +0.000000°</span>'
        f'<button class="align-copy-btn" onclick="copyAlignOffset(this)">Copy offset</button>'
        f'</div>'
    )


# ── state for single Plotly JS embed ─────────────────────────────────────────
_plotly_embedded = [False]

def _fig_html(fig, div_id=None):
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    first = not _plotly_embedded[0]
    _plotly_embedded[0] = True
    kw = dict(
        include_plotlyjs=first, full_html=False,
        config={"responsive": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
    )
    if div_id:
        kw["div_id"] = div_id
    return fig.to_html(**kw)

# ── data helpers ──────────────────────────────────────────────────────────────
def _best_k(sess):
    idx, times = kt.flying_laps(sess)
    return idx[int(np.argmin(times))], idx, times

def _gps_xy(sess, ref_lat=None, ref_lon=None, nsat_min=6):
    d = sess.df[sess.df["GPS Nsat"] >= nsat_min].copy()
    if ref_lat is None:
        ref_lat = float(d["GPS Latitude"].median())
        ref_lon = float(d["GPS Longitude"].median())
    R = 6371000.
    d["x"] = np.radians(d["GPS Longitude"].values - ref_lon) * np.cos(np.radians(ref_lat)) * R
    d["y"] = np.radians(d["GPS Latitude"].values - ref_lat) * R
    return d, ref_lat, ref_lon

def _delta_curve(ref_d, ref_s, cmp_d, cmp_s, n=2000):
    """Cumulative time delta: positive means comparison is slower than reference."""
    L = min(ref_d[-1], cmp_d[-1])
    dc = np.linspace(0, L, n)
    sa = np.interp(dc, ref_d, ref_s) / 3.6
    sb = np.interp(dc, cmp_d, cmp_s) / 3.6
    dd = np.diff(dc)
    return dc[1:], np.cumsum(dd / np.maximum(sb[:-1], 0.1) - dd / np.maximum(sa[:-1], 0.1))

# ── solo charts ───────────────────────────────────────────────────────────────
def _track_fig(sess, n_sectors, color):
    d   = sess.df[sess.df["GPS Nsat"] >= 6]
    bk, idx, times = _best_k(sess)

    all_lats = _la(d["GPS Latitude"].values)
    all_lons = _lo(d["GPS Longitude"].values)
    zoom  = _best_tile_zoom(all_lats, all_lons)
    lat_c = float((np.max(all_lats) + np.min(all_lats)) / 2)
    print(f"  [track map] downloading tiles zoom={zoom} …", flush=True)
    b64, ox, oy, W, H = _sat_snapshot(all_lats, all_lons, zoom)

    def _px(la, lo):
        return _gps2px(_la(la), _lo(lo), ox, oy, zoom)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=f"data:image/jpeg;base64,{b64}",
        xref="x", yref="y",
        x=0, y=0, sizex=W, sizey=H,
        xanchor="left", yanchor="top",
        sizing="stretch", layer="below"))

    # all laps — dim outline
    for k in idx:
        lap = d[d["lap"] == k]
        xs, ys = _px(lap["GPS Latitude"].values, lap["GPS Longitude"].values)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(color="rgba(200,200,200,0.22)", width=1.5),
            showlegend=False, hoverinfo="skip"))

    # best lap — speed heatmap
    best = d[d["lap"] == bk].sort_values("lap_dist")
    sp   = best["GPS Speed"].values
    bxs, bys = _px(best["GPS Latitude"].values, best["GPS Longitude"].values)
    fig.add_trace(go.Scatter(x=bxs, y=bys, mode="markers",
        marker=dict(color=sp, colorscale="RdYlGn", size=5, showscale=True,
                    cmin=float(np.percentile(sp, 5)), cmax=float(np.percentile(sp, 95)),
                    colorbar=dict(title="km/h", thickness=12, len=0.75,
                                  tickfont=dict(size=10))),
        name="Speed (best lap)",
        text=[f"{v:.1f} km/h" for v in sp],
        hovertemplate="%{text}<extra></extra>"))

    bl_d  = best["lap_dist"].values
    bl_la = best["GPS Latitude"].values
    bl_lo = best["GPS Longitude"].values

    # braking markers
    brakes = [e for e in kt.braking_zones(sess) if e["lap"] == bk and e["duration"] >= 0.25]
    if brakes:
        brlat = np.array([np.interp(e["dist"], bl_d, bl_la) for e in brakes])
        brlon = np.array([np.interp(e["dist"], bl_d, bl_lo) for e in brakes])
        brx, bry = _px(brlat, brlon)
        fig.add_trace(go.Scatter(x=brx, y=bry, mode="markers",
            marker=dict(color="#e05c5c", size=10, symbol="circle"),
            name="Braking",
            text=[f"Brake {e['peak']:.2f}g" for e in brakes],
            hovertemplate="%{text}<extra></extra>"))

    # throttle-on markers
    accels = [e for e in kt.acceleration_zones(sess) if e["lap"] == bk and e["duration"] >= 0.5]
    if accels:
        aclat = np.array([np.interp(e["dist"], bl_d, bl_la) for e in accels])
        aclon = np.array([np.interp(e["dist"], bl_d, bl_lo) for e in accels])
        acx, acy = _px(aclat, aclon)
        fig.add_trace(go.Scatter(x=acx, y=acy, mode="markers",
            marker=dict(color="#3EC97A", size=8, symbol="circle"),
            name="Throttle-on",
            text=[f"Accel +{e['peak']:.2f}g" for e in accels],
            hovertemplate="%{text}<extra></extra>"))

    # sector boundary markers
    L = bl_d[-1]
    edges = np.linspace(0, L, n_sectors + 1)[1:-1]
    elat = np.array([np.interp(e, bl_d, bl_la) for e in edges])
    elon = np.array([np.interp(e, bl_d, bl_lo) for e in edges])
    ex, ey = _px(elat, elon)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers+text",
        marker=dict(color="white", size=10),
        text=[f"S{i+1}" for i in range(len(edges))],
        textposition="top right", textfont=dict(color="white"),
        name="Sectors"))

    fig.update_layout(
        **_MAP_LAYOUT,
        title=dict(text=(f"Track Map — {sess.racer}  (best lap · L{bk+1})<br>"
                         f"<sup>Esri World Imagery  ·  speed heatmap  ·  ● braking  ● throttle</sup>"),
                   font=dict(size=13)),
        xaxis=dict(range=[0, W], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        yaxis=dict(range=[H, 0], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=65, b=0),
        height=540,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig, lat_c, zoom


def _self_delta_fig(sess, color):
    """All laps vs personal best: speed trace (top) + delta curves (bottom)."""
    bk, idx, times = _best_k(sess)
    best = sess.df[sess.df["lap"] == bk].sort_values("lap_dist")
    bl_d = best["lap_dist"].values
    bl_s = best["GPS Speed"].values

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.58, 0.42], vertical_spacing=0.07)
    others = [k for k in idx if k != bk]
    alt_colors = [c for c in _COLORS if c != color] + [color]

    # Grey background laps
    for k in others:
        lap = sess.df[sess.df["lap"] == k].sort_values("lap_dist")
        fig.add_trace(go.Scatter(x=lap["lap_dist"].values, y=lap["GPS Speed"].values,
            mode="lines", line=dict(color="rgba(130,130,130,0.3)", width=1),
            showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Best lap
    fig.add_trace(go.Scatter(x=bl_d, y=bl_s, mode="lines",
        line=dict(color=color, width=2.5),
        name=f"Best  L{bk+1} · {min(times):.3f}s"), row=1, col=1)

    # Delta curves
    for ci, k in enumerate(others):
        t = times[idx.index(k)]
        c = alt_colors[ci % len(alt_colors)]
        lap = sess.df[sess.df["lap"] == k].sort_values("lap_dist")
        dd, delta = _delta_curve(bl_d, bl_s,
                                  lap["lap_dist"].values, lap["GPS Speed"].values)
        fig.add_trace(go.Scatter(x=dd, y=np.clip(delta, 0, None),
            fill="tozeroy", fillcolor="rgba(224,92,92,0.18)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dd, y=np.clip(delta, None, 0),
            fill="tozeroy", fillcolor="rgba(105,181,120,0.18)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dd, y=delta, mode="lines",
            line=dict(color=c, width=1.4),
            name=f"L{k+1} · {t:.3f}s",
            hovertemplate=f"L{k+1}: %{{y:+.3f}}s @ %{{x:.0f}}m<extra></extra>"),
            row=2, col=1)

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.12)", width=1), row=2, col=1)
    fig.update_layout(**_LAYOUT,
        title=f"{sess.racer} — All Laps vs Personal Best",
        height=530,
        legend=dict(x=1.01, y=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(
        title_text="Δ vs best (s)<br><sub>red=slower  green=faster</sub>",
        row=2, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=2, col=1)
    return fig


def _lap_time_fig(sess, color):
    """Delta-from-best dot chart — each lap shown as its gap to the session best.
    Best lap lands at 0; all others show exactly how much time was lost.
    Colour gradient: green at 0 → red at the worst lap.
    """
    _, idx, times = _best_k(sess)
    best  = float(min(times))
    deltas = [round(t - best, 3) for t in times]
    worst_d = max(deltas) or 1.0
    mean_d  = float(np.mean(deltas))
    std_t   = float(np.std(times, ddof=1))
    lap_labels = [f"L{i+1}" for i in range(len(times))]

    # colour: interpolate accent (red) → green by delta fraction
    def _dot_color(d):
        frac = min(d / worst_d, 1.0)
        # 0 → #39B54A (green), 1 → #E8002D (red)
        r = int(0x39 + frac * (0xE8 - 0x39))
        g = int(0xB5 + frac * (0x00 - 0xB5))
        b = int(0x4A + frac * (0x2D - 0x4A))
        return f"#{r:02X}{g:02X}{b:02X}"

    dot_colors = [_dot_color(d) for d in deltas]

    fig = go.Figure()

    # Trend line (thin, neutral)
    if len(times) > 2:
        slope, intercept = np.polyfit(range(len(deltas)), deltas, 1)
        trend_y = [slope * i + intercept for i in range(len(deltas))]
        fig.add_trace(go.Scatter(
            x=lap_labels, y=trend_y, mode="lines",
            line=dict(color="rgba(200,200,200,0.25)", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip"))

    # Stem lines from 0 down to each dot
    for i, (lbl, d) in enumerate(zip(lap_labels, deltas)):
        fig.add_shape(type="line",
            x0=i, x1=i, y0=0, y1=d,
            line=dict(color="rgba(200,200,200,0.18)", width=1))

    # Dots
    fig.add_trace(go.Scatter(
        x=lap_labels, y=deltas, mode="markers",
        marker=dict(color=dot_colors, size=13, line=dict(color="rgba(0,0,0,.3)", width=1)),
        text=[f"L{i+1}  {times[i]:.3f}s  (+{d:.3f}s)" for i, d in enumerate(deltas)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False))

    # Mean delta line
    fig.add_hline(y=mean_d,
        line=dict(color="rgba(200,200,200,0.35)", width=1, dash="dash"),
        annotation_text=f"avg +{mean_d:.3f}s · std {std_t:.3f}s",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#9090A8"))

    # Best lap = 0 baseline
    fig.add_hline(y=0, line=dict(color=color, width=1.2, dash="dot"), opacity=0.5)

    trend_note = ""
    if len(times) > 2:
        trend_note = f"  ·  trend {slope*1000:+.0f} ms/lap"

    fig.update_layout(**_LAYOUT,
        title=f"Lap Delta from Best — {sess.racer}{trend_note}",
        yaxis=dict(title="Δ from best (s)", rangemode="tozero", ticksuffix="s"),
        xaxis_title="Lap",
        height=320, showlegend=False)
    return fig


def _gg_fig(sess, color, suffix=""):
    d = sess.df[sess.df["GPS Nsat"] >= 6]
    th = np.linspace(0, 2*np.pi, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
        line=dict(color="rgba(200,200,200,0.15)", width=1, dash="dash"),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["GPS LatAcc"].values, y=d["GPS LonAcc"].values,
        mode="markers",
        marker=dict(color=d["GPS Speed"].values, colorscale="plasma", size=3,
                    opacity=.45, showscale=True,
                    cmin=float(d["GPS Speed"].quantile(.05)),
                    cmax=float(d["GPS Speed"].quantile(.95)),
                    colorbar=dict(title="Speed<br>(km/h)", thickness=14, len=0.8)),
        name="G-G",
        hovertemplate="Lat %{x:.2f}g  Lon %{y:.2f}g<extra></extra>"))
    fig.update_layout(**_LAYOUT,
        title=f"G-G Diagram{suffix}",
        xaxis=dict(title="Lateral g", scaleanchor="y", scaleratio=1),
        yaxis_title="Longitudinal g", height=400)
    return fig


def _rpm_fig(sess, color, suffix=""):
    d = sess.df[(sess.df["GPS Nsat"] >= 6) &
                (sess.df["GPS Speed"] > 20) & (sess.df["RPM"] > 0)]
    fig = go.Figure(go.Scatter(x=d["RPM"].values, y=d["GPS Speed"].values,
        mode="markers",
        marker=dict(color=d["GPS LonAcc"].values, colorscale="RdYlGn",
                    size=3, opacity=.4, cmin=-.5, cmax=.5, showscale=True,
                    colorbar=dict(title="LonAcc<br>(g)", thickness=14, len=0.8)),
        hovertemplate="RPM %{x:.0f}  Speed %{y:.1f} km/h<extra></extra>"))
    if len(d) >= 50:
        cf = np.polyfit(d["RPM"].values, d["GPS Speed"].values, 1)
        rr = np.array([d["RPM"].min(), d["RPM"].max()])
        fig.add_trace(go.Scatter(x=rr, y=np.polyval(cf, rr), mode="lines",
            line=dict(color="white", width=2),
            name=f"{cf[0]*1000:.3f} km/h / 1000 rpm"))
    fig.update_layout(**_LAYOUT, title=f"RPM vs Speed{suffix}",
        xaxis_title="RPM", yaxis_title="Speed (km/h)", height=400,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig

# ── combined charts ────────────────────────────────────────────────────────────
def _combined_delta_fig(sessions):
    ref = min(sessions, key=lambda s: kt.consistency_stats(s)["best"])
    bk_r, _, _ = _best_k(ref)
    ref_lap = ref.df[ref.df["lap"] == bk_r].sort_values("lap_dist")
    rd = ref_lap["lap_dist"].values
    rs = ref_lap["GPS Speed"].values

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.58, 0.42], vertical_spacing=0.07)
    c0 = _COLORS[0]
    fig.add_trace(go.Scatter(x=rd, y=rs, mode="lines",
        line=dict(color=c0, width=2.5),
        name=ref.racer, legendgroup=ref.racer,
        hovertemplate=f"{ref.racer}: %{{y:.1f}} km/h<extra></extra>"),
        row=1, col=1)

    others = [s for s in sessions if s is not ref]
    for i, sess in enumerate(others, 1):
        c = _COLORS[i % len(_COLORS)]
        bk, _, _ = _best_k(sess)
        lap = sess.df[sess.df["lap"] == bk].sort_values("lap_dist")
        ld, ls = lap["lap_dist"].values, lap["GPS Speed"].values
        fig.add_trace(go.Scatter(x=ld, y=ls, mode="lines",
            line=dict(color=c, width=2.5),
            name=sess.racer, legendgroup=sess.racer,
            hovertemplate=f"{sess.racer}: %{{y:.1f}} km/h<extra></extra>"),
            row=1, col=1)
        dd, delta = _delta_curve(rd, rs, ld, ls)
        fig.add_trace(go.Scatter(x=dd, y=np.clip(delta, 0, None),
            fill="tozeroy", fillcolor="rgba(224,92,92,0.18)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=sess.racer), row=2, col=1)
        fig.add_trace(go.Scatter(x=dd, y=np.clip(delta, None, 0),
            fill="tozeroy", fillcolor="rgba(105,181,120,0.18)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=sess.racer), row=2, col=1)
        fig.add_trace(go.Scatter(x=dd, y=delta, mode="lines",
            line=dict(color=c, width=1.8),
            name=f"Δ {sess.racer}−{ref.racer}", legendgroup=sess.racer,
            hovertemplate=f"{sess.racer}: %{{y:+.3f}}s @ %{{x:.0f}}m<extra></extra>"),
            row=2, col=1)

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1), row=2, col=1)
    fig.update_layout(**_LAYOUT,
        title=(f"Speed Trace + Time Delta  (reference: {ref.racer})<br>"
               f"<sup>Red = slower than {ref.racer} · Green = faster · "
               f"Click legend to toggle drivers</sup>"),
        height=560,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text=f"Δ vs {ref.racer} (s)", row=2, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=2, col=1)
    return fig


def _combined_track_fig(sessions, n_sectors):
    all_lats, all_lons = [], []
    for sess in sessions:
        d = sess.df[sess.df["GPS Nsat"] >= 6]
        all_lats.extend(_la(d["GPS Latitude"].values))
        all_lons.extend(_lo(d["GPS Longitude"].values))
    all_lats = np.array(all_lats)
    all_lons = np.array(all_lons)

    zoom  = _best_tile_zoom(all_lats, all_lons)
    lat_c = float((np.max(all_lats) + np.min(all_lats)) / 2)
    print(f"  [combined map] downloading tiles zoom={zoom} …", flush=True)
    b64, ox, oy, W, H = _sat_snapshot(all_lats, all_lons, zoom)

    def _px(la, lo):
        return _gps2px(_la(la), _lo(lo), ox, oy, zoom)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=f"data:image/jpeg;base64,{b64}",
        xref="x", yref="y",
        x=0, y=0, sizex=W, sizey=H,
        xanchor="left", yanchor="top",
        sizing="stretch", layer="below"))

    for i, sess in enumerate(sessions):
        c  = _COLORS[i % len(_COLORS)]
        d  = sess.df[sess.df["GPS Nsat"] >= 6]
        bk, _, times = _best_k(sess)
        # dim outline — all laps
        for k in kt.flying_laps(sess)[0]:
            lap = d[d["lap"] == k]
            xs, ys = _px(lap["GPS Latitude"].values, lap["GPS Longitude"].values)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                line=dict(color="rgba(200,200,200,0.12)", width=1),
                showlegend=False, hoverinfo="skip"))
        # best lap — driver colour
        lap = d[d["lap"] == bk].sort_values("lap_dist")
        xs, ys = _px(lap["GPS Latitude"].values, lap["GPS Longitude"].values)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(color=c, width=3.5),
            name=f"{sess.racer}  {min(times):.3f}s", legendgroup=sess.racer,
            text=[f"{v:.1f} km/h" for v in lap["GPS Speed"].values],
            hovertemplate=f"{sess.racer}: %{{text}}<extra></extra>"))

    fig.update_layout(
        **_MAP_LAYOUT,
        title="Track Map — Best Lap Overlay<br>"
              "<sup>Esri World Imagery  ·  Click legend to toggle drivers</sup>",
        xaxis=dict(range=[0, W], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        yaxis=dict(range=[H, 0], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=65, b=0),
        height=520,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig, lat_c, zoom


def _combined_laptimes_fig(sessions):
    """Delta-from-own-best chart for multiple drivers on the same axes.
    Each driver's best lap = 0; dots show gap per lap in their team colour.
    Enables instant cross-driver consistency comparison on a shared scale.
    """
    fig = go.Figure()
    overall_worst = 0.0
    for sess in sessions:
        _, _, times = _best_k(sess)
        best = min(times)
        worst_d = max(t - best for t in times)
        overall_worst = max(overall_worst, worst_d)

    for i, sess in enumerate(sessions):
        c = _COLORS[i % len(_COLORS)]
        _, idx, times = _best_k(sess)
        best   = float(min(times))
        deltas = [round(t - best, 3) for t in times]
        mean_d = float(np.mean(deltas))
        std_t  = float(np.std(times, ddof=1))
        lap_labels = [f"L{j+1}" for j in range(len(times))]

        # Trend
        if len(times) > 2:
            slope, intercept = np.polyfit(range(len(deltas)), deltas, 1)
            trend_y = [slope * j + intercept for j in range(len(deltas))]
            fig.add_trace(go.Scatter(
                x=lap_labels, y=trend_y, mode="lines",
                line=dict(color=c, width=1, dash="dot"), opacity=0.35,
                showlegend=False, hoverinfo="skip", legendgroup=sess.racer))

        # Stems
        for j, (lbl, d) in enumerate(zip(lap_labels, deltas)):
            fig.add_shape(type="line", x0=j, x1=j, y0=0, y1=d,
                line=dict(color=c, width=1), opacity=0.25)

        # Dots
        fig.add_trace(go.Scatter(
            x=lap_labels, y=deltas, mode="markers",
            marker=dict(color=c, size=11, line=dict(color="rgba(0,0,0,.3)", width=1)),
            name=f"{sess.racer}  best {best:.3f}s  std {std_t:.3f}s",
            legendgroup=sess.racer,
            text=[f"{sess.racer}  L{j+1}  {times[j]:.3f}s  (+{d:.3f}s)"
                  for j, d in enumerate(deltas)],
            hovertemplate="%{text}<extra></extra>"))

        # Per-driver mean
        fig.add_hline(y=mean_d,
            line=dict(color=c, width=1, dash="dash"), opacity=0.45,
            annotation_text=f"{sess.racer[:6]} avg +{mean_d:.3f}s",
            annotation_position="top right",
            annotation_font=dict(size=9, color=c))

    fig.add_hline(y=0,
        line=dict(color="rgba(200,200,200,0.25)", width=1))

    fig.update_layout(**_LAYOUT,
        title="Lap Delta from Own Best — All Drivers",
        yaxis=dict(title="Δ from own best (s)", rangemode="tozero", ticksuffix="s"),
        xaxis_title="Lap",
        height=360,
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def _combined_gg_fig(sessions):
    th = np.linspace(0, 2*np.pi, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
        line=dict(color="rgba(200,200,200,0.15)", width=1, dash="dash"),
        showlegend=False, hoverinfo="skip"))
    for i, sess in enumerate(sessions):
        c = _COLORS[i % len(_COLORS)]
        d = sess.df[sess.df["GPS Nsat"] >= 6]
        fig.add_trace(go.Scatter(x=d["GPS LatAcc"].values, y=d["GPS LonAcc"].values,
            mode="markers", marker=dict(color=c, size=2, opacity=0.22),
            name=sess.racer, legendgroup=sess.racer))
    fig.update_layout(**_LAYOUT,
        title="G-G Comparison<br><sup>Click legend to toggle drivers</sup>",
        xaxis=dict(title="Lateral g", scaleanchor="y", scaleratio=1),
        yaxis_title="Longitudinal g", height=420,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)",
                    itemsizing="constant"))
    return fig

# ── HTML helpers ───────────────────────────────────────────────────────────────
def _kpi(label, value, sub="", hi=False):
    cls = "kpi hi" if hi else "kpi"
    return (f'<div class="{cls}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def _apex_table_html(sessions, n_sectors):
    ref = min(sessions, key=lambda s: kt.consistency_stats(s)["best"])
    csps = {s.racer: kt.corner_speed_profile(s, n_sectors) for s in sessions}
    rows = []
    for i in range(n_sectors):
        sn = f"S{i+1}"
        ref_apex = csps[ref.racer][f"{sn}_apex"].mean()
        row = {"sector": sn, "_gap": 0.0}
        for sess in sessions:
            apex = csps[sess.racer][f"{sn}_apex"].mean()
            row[sess.racer] = round(apex, 1)
            if sess is not ref:
                row["_gap"] = max(row["_gap"], ref_apex - apex)
        rows.append(row)
    rows.sort(key=lambda r: r["_gap"], reverse=True)

    html = ('<table class="st"><thead><tr>'
            '<th>Rank</th><th>Sector</th>'
            f'<th>{ref.racer} ★ (ref)</th>')
    for s in sessions:
        if s is not ref:
            html += f'<th>{s.racer}</th><th>Gap (km/h)</th>'
    html += '<th>Coaching focus</th></tr></thead><tbody>'
    for rank, row in enumerate(rows, 1):
        html += f'<tr><td class="lbl">#{rank}</td><td><strong>{row["sector"]}</strong></td>'
        html += f'<td>{row[ref.racer]}</td>'
        for sess in sessions:
            if sess is ref:
                continue
            apex = row[sess.racer]
            gap = round(apex - row[ref.racer], 1)
            gcls = "neg" if gap >= 0 else "pos"
            html += f'<td>{apex}</td><td class="{gcls}">{gap:+.1f}</td>'
        g = row["_gap"]
        note = ("Entry — over-slowing at corner peak" if g > 5
                else "Moderate loss — technique or line" if g > 2
                else "Small — clean up exits")
        html += f'<td class="lbl">{note}</td></tr>'
    return html + '</tbody></table>'


def _stage5_html(sess, n_sectors):
    brakes = kt.braking_zones(sess)
    accels = kt.acceleration_zones(sess)
    rot    = kt.rotation_events(sess)
    gg     = kt.gg_stats(sess)
    csp    = kt.corner_speed_profile(sess, n_sectors)
    _, idx, times = _best_k(sess)
    n = len(idx)
    rows = []
    if brakes:
        by_lap = {}
        for e in brakes:
            by_lap.setdefault(e["lap"], []).append(e["dist"])
        bp_std = round(float(np.std([v[0] for v in by_lap.values()], ddof=1)), 1) if len(by_lap) > 1 else "—"
        rows.append(("Braking",
            f"{len(brakes)} zones · {len(brakes)/n:.1f}/lap · "
            f"peak {min(e['peak'] for e in brakes):.2f}g · "
            f"brake-point std <strong>{bp_std}m</strong>"))
    if accels:
        rows.append(("Acceleration",
            f"{len(accels)} zones · {len(accels)/n:.1f}/lap · "
            f"peak +{max(e['peak'] for e in accels):.2f}g"))
    rot_val = (f"{len(rot)} events · max +{max(e['peak_oversteer'] for e in rot):.0f} deg/s"
               if rot else "none detected")
    rows.append(("Rotation", rot_val))
    rows.append(("G-G envelope",
        f"Lat {gg['lat_neg']:.2f}/{gg['lat_pos']:+.2f}g · "
        f"Lon {gg['lon_brake']:.2f}/{gg['lon_accel']:+.2f}g · "
        f"peak {gg['combined_peak']:.2f}g"))
    ac = [c for c in csp.columns if c.endswith("_apex")]
    if ac:
        am = {c.replace("_apex", ""): csp[c].mean() for c in ac}
        slow, fast = min(am, key=am.get), max(am, key=am.get)
        rows.append(("Corner apex",
            f"Slowest: {slow} @ {am[slow]:.1f} km/h · "
            f"Fastest: {fast} @ {am[fast]:.1f} km/h"))
    if len(times) > 2:
        slope = np.polyfit(range(len(times)), times, 1)[0]
        tcls = "neg" if slope < 0 else "pos"
        rows.append(("Lap trend",
            f'<span class="{tcls}">{slope*1000:+.0f} ms/lap</span> '
            f'({"improving ↓" if slope < 0 else "plateauing →"})'))
    html = '<table class="st"><thead><tr><th>Metric</th><th>Detail</th></tr></thead><tbody>'
    for lbl, val in rows:
        html += f'<tr><td class="lbl">{lbl}</td><td>{val}</td></tr>'
    return html + '</tbody></table>'


def _hth_html(sessions):
    stats = {s.racer: kt.consistency_stats(s) for s in sessions}
    ggs   = {s.racer: kt.gg_stats(s) for s in sessions}
    def _bp(sess):
        evs = kt.braking_zones(sess)
        bl = {}
        for e in evs:
            bl.setdefault(e["lap"], []).append(e["dist"])
        pts = [v[0] for v in bl.values()]
        return round(float(np.std(pts, ddof=1)), 1) if len(pts) > 1 else None
    metrics = [
        ("Best lap (s)",      [stats[s.racer]["best"] for s in sessions], "lower"),
        ("Avg lap (s)",       [stats[s.racer]["mean"] for s in sessions], "lower"),
        ("Std dev (s)",       [stats[s.racer]["std"]  for s in sessions], "lower"),
        ("Top speed (km/h)",  [kt.top_speed(s) for s in sessions],        "higher"),
        ("Gearing km/h/1krpm",[kt.rpm_per_kmh(s) for s in sessions],      "higher"),
        ("G-G peak (g)",      [ggs[s.racer]["combined_peak"] for s in sessions], "higher"),
        ("Brake-pt std (m)",  [_bp(s) for s in sessions],                 "lower"),
    ]
    html = ('<table class="st"><thead><tr><th>Metric</th>'
            + ''.join(f'<th>{s.racer}</th>' for s in sessions)
            + '</tr></thead><tbody>')
    for label, vals, better in metrics:
        num = [v for v in vals if v is not None]
        if not num:
            continue
        best_val = min(num) if better == "lower" else max(num)
        html += f'<tr><td class="lbl">{label}</td>'
        for v in vals:
            if v is None:
                html += '<td>—</td>'; continue
            cls = 'neg' if v == best_val else ''
            html += f'<td class="{cls}">{v}</td>'
        html += '</tr>'
    return html + '</tbody></table>'

# ── CSS / JS ──────────────────────────────────────────────────────────────────
_CSS = """
:root{
  --bg:#15151E;--card:#1E1E2E;--elevated:#252535;--border:#2E2E42;
  --border-hi:#44445A;--accent:#E8002D;--accent-dim:rgba(232,0,45,.09);
  --accent-glow:rgba(232,0,45,.10);--red:#FF4040;--green:#39B54A;
  --text:#FFFFFF;--dim:#9090A8;--muted:#2E2E42;
  --dot-color:rgba(255,255,255,.025);
  --font:'Titillium Web','Helvetica Neue',Arial,sans-serif;
  --font-display:'Titillium Web','Helvetica Neue',Arial,sans-serif;
  --font-bold:'Titillium Web','Helvetica Neue',Arial,sans-serif;
}
[data-theme="light"]{
  --bg:#F5F5F7;--card:#FFFFFF;--elevated:#EBEBF2;--border:#D5D5E0;
  --border-hi:#ABABC0;--accent:#E8002D;--accent-dim:rgba(232,0,45,.07);
  --accent-glow:rgba(232,0,45,.06);--red:#D0021B;--green:#1A8F2D;
  --text:#15151E;--dim:#66667A;--muted:#E5E5EF;
  --dot-color:rgba(0,0,0,.04);
}
*{box-sizing:border-box;margin:0;padding:0;font-family:var(--font)}
html{scroll-behavior:smooth}

body{
  background:
    radial-gradient(ellipse 70% 50% at 0% 0%,rgba(232,0,45,.06) 0%,transparent 45%),
    radial-gradient(ellipse 50% 50% at 100% 100%,rgba(0,144,208,.04) 0%,transparent 45%),
    var(--bg);
  color:var(--text);font-size:14px;min-height:100vh;
  transition:background .3s,color .3s;
}

/* fine-grid technical atmosphere */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(var(--dot-color) 1px,transparent 1px),
    linear-gradient(90deg,var(--dot-color) 1px,transparent 1px);
  background-size:40px 40px;
}

.page{max-width:1460px;margin:0 auto;padding:36px 26px;position:relative;z-index:1}

/* ── header ─────────────────────────────────────────────────────── */
.header{margin-bottom:36px;padding-bottom:28px;position:relative}
.header::after{
  content:'';position:absolute;bottom:0;left:0;
  width:100%;height:2px;
  background:linear-gradient(to right,var(--accent) 0%,rgba(232,0,45,.15) 35%,transparent 70%);
}
.header-eyebrow{
  font-size:.68rem;letter-spacing:4px;color:var(--accent);
  text-transform:uppercase;margin-bottom:14px;
  font-family:var(--font-display);
}
.header h1{
  font-size:3rem;font-weight:700;line-height:1;
  letter-spacing:-1px;color:var(--text);
  font-family:var(--font-bold);
  animation:slideIn .5s cubic-bezier(.22,1,.36,1) both;
}
.header h1 .vs{color:var(--dim);font-weight:400;font-size:1.8rem;
                letter-spacing:0;margin:0 14px;vertical-align:.12em}
.header-meta{
  margin-top:14px;font-size:.88rem;color:var(--dim);
  animation:fadeUp .5s .1s ease both;
}
.header-meta .hi{color:var(--accent);font-weight:700}

/* ── theme toggle ───────────────────────────────────────────────── */
.theme-toggle{
  position:fixed;top:20px;right:24px;z-index:999;
  background:var(--card);border:1px solid var(--border);border-radius:3px;
  padding:7px 13px;cursor:pointer;
  display:flex;align-items:center;gap:8px;
  font-size:.72rem;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
  color:var(--dim);
  transition:color .2s,background .2s,border-color .2s,box-shadow .2s;
  box-shadow:0 2px 12px rgba(0,0,0,.18);
}
.theme-toggle:hover{color:var(--accent);border-color:var(--accent);
  box-shadow:0 2px 18px rgba(0,0,0,.28)}
.theme-toggle svg{width:15px;height:15px;flex-shrink:0;
  stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

/* ── tabs ───────────────────────────────────────────────────────── */
.tab-bar{
  display:flex;gap:0;flex-wrap:wrap;margin-bottom:36px;
  background:var(--card);border:1px solid var(--border);
  border-radius:3px;padding:0;overflow:hidden;
  animation:fadeUp .4s .15s ease both;
}
.tab-btn{
  background:none;border:none;border-right:1px solid var(--border);
  color:var(--dim);font-size:.72rem;font-weight:700;
  font-family:var(--font-display);
  letter-spacing:2px;text-transform:uppercase;
  padding:11px 26px;cursor:pointer;
  transition:color .18s,background .18s;white-space:nowrap;
  position:relative;
}
.tab-btn:last-child{border-right:none}
.tab-btn:hover{color:var(--text);background:var(--elevated)}
.tab-btn.active{background:var(--accent);color:#fff;}
.tab-content{display:none}
.tab-content.active{display:block}

/* ── KPI row ────────────────────────────────────────────────────── */
.kpi-row{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
  gap:10px;margin-bottom:36px;
}
.kpi{
  background:var(--card);border:1px solid var(--border);border-radius:3px;
  padding:20px 20px 18px;border-top:3px solid transparent;
  transition:border-color .25s,box-shadow .25s;
  animation:fadeUp .5s ease both;
}
.kpi:hover{border-color:var(--border-hi)}
.kpi.hi{
  border-top-color:var(--accent);
  box-shadow:0 0 28px var(--accent-glow);
}
.kpi.hi:hover{border-color:var(--border-hi);border-top-color:var(--accent)}
.kpi:nth-child(1){animation-delay:.05s}
.kpi:nth-child(2){animation-delay:.10s}
.kpi:nth-child(3){animation-delay:.15s}
.kpi:nth-child(4){animation-delay:.20s}
.kpi:nth-child(5){animation-delay:.25s}
.kpi:nth-child(6){animation-delay:.30s}
.kpi-label{
  font-size:.64rem;color:var(--dim);text-transform:uppercase;
  letter-spacing:2px;margin-bottom:11px;font-family:var(--font-display);
}
.kpi-value{font-size:1.55rem;font-weight:700;color:var(--accent);line-height:1.1;
  font-family:var(--font-bold);}
.kpi-sub{font-size:.72rem;color:var(--dim);margin-top:7px}

/* ── sections ───────────────────────────────────────────────────── */
.section{margin-bottom:40px;animation:fadeUp .55s ease both;animation-delay:.12s}
.sec-title{
  display:flex;align-items:center;gap:14px;
  font-size:.78rem;color:var(--text);text-transform:uppercase;
  letter-spacing:3px;font-weight:700;
  font-family:var(--font-display);
  margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--border);
}
.sec-title::before{
  content:'';display:block;width:4px;height:22px;
  background:var(--accent);flex-shrink:0;
}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.box{
  background:var(--card);border:1px solid var(--border);
  border-radius:3px;overflow:hidden;
  transition:border-color .2s;
}
.box:hover{border-color:var(--border-hi)}
.mt{margin-top:14px}

/* ── tables ─────────────────────────────────────────────────────── */
table.st{
  width:100%;border-collapse:collapse;
  background:var(--card);border-radius:3px;overflow:hidden;
}
table.st th{
  background:var(--elevated);color:var(--dim);
  font-size:.64rem;text-transform:uppercase;
  letter-spacing:2px;padding:13px 18px;text-align:left;font-weight:700;
  font-family:var(--font-display);
}
table.st td{
  font-size:.82rem;padding:11px 18px;
  border-top:1px solid var(--border);line-height:1.55;
}
table.st tr:hover td{background:var(--elevated)}
.lbl{color:var(--dim);width:22%}
.pos{color:var(--red)}.neg{color:var(--green)}

/* ── info box ───────────────────────────────────────────────────── */
.info-box{
  background:var(--card);border:1px solid var(--border);
  border-left:3px solid var(--accent);border-radius:10px;
  padding:26px 30px;
}
.info-title{
  font-size:.64rem;text-transform:uppercase;letter-spacing:2px;
  color:var(--accent);margin-bottom:16px;font-weight:700;
}
.info-box p,.info-box li{
  color:var(--dim);font-size:.86rem;line-height:1.8;margin-bottom:10px;
}
.info-box ul{padding-left:20px;margin-bottom:12px}
.info-box strong{color:var(--text)}
.info-box em{color:var(--text);font-style:normal;font-weight:600}

/* ── animations ─────────────────────────────────────────────────── */
@keyframes fadeUp{
  from{opacity:0;transform:translateY(12px)}
  to  {opacity:1;transform:translateY(0)}
}
@keyframes slideIn{
  from{opacity:0;transform:translateX(-16px)}
  to  {opacity:1;transform:translateX(0)}
}

/* ── sector inner-tabs ──────────────────────────────────────────── */
.stab-bar{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:12px}
.stab-btn{
  background:var(--elevated);border:1px solid var(--border);border-radius:2px;
  color:var(--dim);font-size:.68rem;font-weight:700;
  font-family:var(--font-display);
  letter-spacing:2px;text-transform:uppercase;
  padding:7px 16px;cursor:pointer;
  transition:color .15s,background .15s,border-color .15s;
}
.stab-btn:hover{color:var(--text);border-color:var(--border-hi)}
.stab-btn.active{
  background:var(--accent);color:#fff;border-color:var(--accent);
}
.stab-content{display:none}
.stab-content.active{display:block}
.sec-stats-bar{
  display:flex;gap:28px;flex-wrap:wrap;
  padding:14px 18px;background:var(--elevated);
  border:1px solid var(--border);border-radius:8px;margin-bottom:10px;
  font-size:.8rem;color:var(--dim);
}
.sec-stats-bar span{display:flex;flex-direction:column;gap:4px}
.sec-stats-bar .slabel{font-size:.63rem;text-transform:uppercase;
  letter-spacing:1.6px;color:var(--dim)}
.sec-stats-bar .sval{color:var(--text);font-weight:700;font-size:.95rem}
.sec-stats-bar .sval.good{color:var(--green)}
.sec-stats-bar .sval.warn{color:var(--red)}
table.st tr.best-row td{background:rgba(46,213,115,.06)}
.sector-detail-grid{display:grid;grid-template-columns:1fr 360px;gap:14px;align-items:start}
.sector-detail-right{display:flex;flex-direction:column;gap:10px}
/* ── sector lap selection table rows ────────────────────────────── */
.lap-sel-table tbody tr.lap-sel-row{
  cursor:pointer;transition:background .12s,opacity .15s;
}
.lap-sel-table tbody tr.lap-sel-row:hover{background:var(--elevated) !important}
.lap-sel-table tbody tr.lap-sel-dim{opacity:.18}
/* sel-controls: gradient switch + reset in one bar */
.sel-controls{
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 2px 7px;
}
.lap-reset-btn{
  background:none;border:1px solid var(--border);border-radius:2px;
  color:var(--dim);font-size:.64rem;font-weight:700;
  font-family:var(--font-display);letter-spacing:1.5px;text-transform:uppercase;
  padding:4px 10px;cursor:pointer;
  transition:color .15s,border-color .15s;
}
.lap-reset-btn:hover{color:var(--accent);border-color:var(--accent)}
.grad-switch-row{display:flex;align-items:center;padding:4px 2px}
.grad-switch{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.grad-switch input{display:none}
.grad-track{
  position:relative;width:32px;height:18px;border-radius:9px;
  background:var(--border);transition:background .2s;flex-shrink:0}
.grad-switch input:checked ~ .grad-track{background:var(--accent)}
.grad-thumb{
  position:absolute;top:3px;left:3px;width:12px;height:12px;
  border-radius:50%;background:#fff;transition:transform .2s}
.grad-switch input:checked ~ .grad-track .grad-thumb{transform:translateX(14px)}
.grad-label{font-size:.72rem;color:var(--dim);letter-spacing:.5px}
@media(max-width:1000px){.sector-detail-grid{grid-template-columns:1fr}}
/* theoretical best */
.theo-best-wrap{display:flex;flex-direction:column;gap:16px}
.theo-kpi-row{display:flex;gap:20px;flex-wrap:wrap}
.theo-kpi{display:flex;flex-direction:column;gap:3px;
  background:var(--elevated);border:1px solid var(--border);border-radius:3px;
  border-top:3px solid var(--border);
  padding:12px 20px;min-width:160px}
.theo-kpi:first-child{border-top-color:var(--accent)}
.theo-label{font-size:.64rem;letter-spacing:2px;color:var(--dim);text-transform:uppercase;
  font-family:var(--font-display)}
.theo-val{font-size:1.45rem;font-weight:700;letter-spacing:.5px;font-family:var(--font-bold)}
.theo-val.hi{color:var(--accent)}
.theo-val.pos{color:var(--green)}
.theo-body{display:grid;grid-template-columns:1fr 340px;gap:20px;align-items:start}
.theo-map{min-height:0}
.theo-table-col{display:flex;flex-direction:column;gap:8px}
.sector-dot{display:inline-block;width:10px;height:10px;border-radius:50%;
  margin-right:6px;vertical-align:middle}
@media(max-width:1100px){.theo-body{grid-template-columns:1fr}}

@media(max-width:820px){.grid2{grid-template-columns:1fr}}

/* ── GPS align bar ──────────────────────────────────────────────── */
.align-bar{
  display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  padding:8px 14px;margin-top:2px;
  background:var(--elevated);border:1px solid var(--border);
  border-top:none;border-radius:0 0 3px 3px;
}
.align-label{
  font-family:var(--font-display);font-size:.64rem;
  letter-spacing:2.5px;color:var(--accent);text-transform:uppercase;
  white-space:nowrap;flex-shrink:0;
}
.align-pad{display:grid;grid-template-rows:auto auto auto;grid-template-columns:1fr;
  gap:2px;align-items:center;justify-items:center;flex-shrink:0}
.align-horiz{display:flex;gap:2px}
.align-arrow{
  width:28px;height:28px;border-radius:2px;
  background:var(--card);border:1px solid var(--border);
  color:var(--dim);font-size:.85rem;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:color .15s,border-color .15s,background .15s;
  line-height:1;
}
.align-arrow:hover{color:var(--accent);border-color:var(--accent);background:var(--accent-dim)}
.align-reset-xy{font-size:.7rem;color:var(--dim)}
.align-step{
  background:var(--card);border:1px solid var(--border);border-radius:2px;
  color:var(--text);font-family:var(--font-display);font-size:.68rem;
  padding:4px 6px;cursor:pointer;flex-shrink:0;
}
.align-readout{
  font-family:var(--font-display);font-size:.7rem;color:var(--dim);
  letter-spacing:.5px;white-space:nowrap;flex:1;min-width:240px;
}
.align-readout.dirty{color:var(--accent)}
.align-copy-btn{
  background:none;border:1px solid var(--border);border-radius:2px;
  color:var(--dim);font-family:var(--font-display);font-size:.64rem;
  font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  padding:4px 10px;cursor:pointer;white-space:nowrap;
  transition:color .15s,border-color .15s;flex-shrink:0;
}
.align-copy-btn:hover{color:var(--accent);border-color:var(--accent)}
"""

_JS = """
/* ── theme toggle ────────────────────────────────────────────────────────── */
var _PLOT_DARK  = {paper_bgcolor:'#15151E',plot_bgcolor:'#15151E',
                   font:{color:'#E8E8F0'},
                   'xaxis.gridcolor':'#252535','yaxis.gridcolor':'#252535'};
var _PLOT_LIGHT = {paper_bgcolor:'#FFFFFF',plot_bgcolor:'#FFFFFF',
                   font:{color:'#15151E'},
                   'xaxis.gridcolor':'#D5D5E0','yaxis.gridcolor':'#D5D5E0'};

function _replotAll(light) {
    var upd = light ? _PLOT_LIGHT : _PLOT_DARK;
    document.querySelectorAll('.js-plotly-plot').forEach(function(el) {
        try { Plotly.relayout(el, upd); } catch(e) {}
    });
}

function initTheme() {
    var saved = localStorage.getItem('kt-theme') || 'dark';
    setTheme(saved, true);
}

function setTheme(theme, init) {
    var light = theme === 'light';
    document.documentElement.setAttribute('data-theme', light ? 'light' : '');
    localStorage.setItem('kt-theme', theme);
    var btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.querySelector('.theme-label').textContent = light ? 'Dark' : 'Light';
        btn.querySelector('.theme-icon').innerHTML = light
            ? '<circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.22" y1="4.22" x2="7.05" y2="7.05"/><line x1="16.95" y1="16.95" x2="19.78" y2="19.78"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.22" y1="19.78" x2="7.05" y2="16.95"/><line x1="16.95" y1="7.05" x2="19.78" y2="4.22"/>'
            : '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    }
    if (!init) _replotAll(light);
}

function toggleTheme() {
    var cur = document.documentElement.getAttribute('data-theme');
    setTheme(cur === 'light' ? 'dark' : 'light', false);
}

document.addEventListener('DOMContentLoaded', initTheme);

function showTab(id, btn) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
    setTimeout(function() {
        var plots = document.getElementById(id).querySelectorAll('.js-plotly-plot');
        plots.forEach(function(p) { if (window.Plotly) Plotly.Plots.resize(p); });
    }, 60);
}
function showSectorTab(id, btn) {
    var wrap = btn.closest('.sector-tab-group');
    wrap.querySelectorAll('.stab-content').forEach(t => t.classList.remove('active'));
    wrap.querySelectorAll('.stab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
}

/* ── GPS alignment nudge ─────────────────────────────────────────────────────
   All bars sharing data-key share one offset accumulator (_gpsOff[key]).
   Each bar has its own data-zoom / data-lat-c so pixel conversion is correct
   per figure even when sector maps use a different zoom than the track map.
   Nudging from any bar updates every figure in data-group, and syncs all
   readouts across every bar that shares the same data-key.
*/
var _gpsOff     = {};  // { key:   { dLat: deg, dLon: deg } }
var _origLayout = {};  // { divId: { xr0, xr1, yr0, yr1, ix, iy } }

function _saveOrig(divId, div) {
    if (_origLayout[divId]) return;
    var xa = (div.layout.xaxis  || {}).range || [0, 1];
    var ya = (div.layout.yaxis  || {}).range || [1, 0];
    var im = (div.layout.images || [])[0]    || {};
    _origLayout[divId] = {
        xr0: xa[0], xr1: xa[1],
        yr0: ya[0], yr1: ya[1],
        ix:  im.x  || 0,
        iy:  im.y  || 0
    };
}

function _getGroup(bar) {
    return (bar.dataset.group || bar.dataset.chart).split(',').map(function(s){return s.trim();});
}

// Apply incremental (dLat°, dLon°) shift to one figure, using that figure's own zoom.
function _applyViewportOne(divId, dLat, dLon) {
    var div = document.getElementById(divId);
    if (!div || !div.layout) return;
    _saveOrig(divId, div);

    // Look up this figure's zoom and latC from its own align bar
    var bar2  = document.querySelector('.align-bar[data-chart="' + divId + '"]');
    var zoom2 = bar2 ? parseInt(bar2.dataset.zoom)      : 18;
    var latC2 = bar2 ? parseFloat(bar2.dataset.latC)    : 1.4;
    var cosL2 = Math.cos(latC2 * Math.PI / 180);
    var n2    = Math.pow(2, zoom2) * 256;
    var dx    = dLon * n2 / 360;
    var dy    = -dLat * n2 / (360 * cosL2);

    var xa = (div.layout.xaxis  || {}).range || [0, 1];
    var ya = (div.layout.yaxis  || {}).range || [1, 0];
    var im = (div.layout.images || [])[0]    || {};
    Plotly.relayout(div, {
        'xaxis.range[0]': xa[0] - dx, 'xaxis.range[1]': xa[1] - dx,
        'yaxis.range[0]': ya[0] - dy, 'yaxis.range[1]': ya[1] - dy,
        'images[0].x':    (im.x || 0) - dx,
        'images[0].y':    (im.y || 0) - dy
    });
}

function _updateAllReadouts(key) {
    var off   = _gpsOff[key] || {dLat:0, dLon:0};
    var dirty = (off.dLat !== 0 || off.dLon !== 0);
    var fmt   = function(v) { return (v >= 0 ? '+' : '') + v.toFixed(6) + '°'; };
    var text  = 'lat ' + fmt(off.dLat) + ' lon ' + fmt(off.dLon);
    document.querySelectorAll('.align-bar[data-key="' + key + '"] .align-readout')
        .forEach(function(rd) {
            rd.textContent = text;
            rd.classList.toggle('dirty', dirty);
        });
}

function nudgeGPS(btn, dir) {
    var bar   = btn.closest('.align-bar');
    var key   = bar.dataset.key   || bar.dataset.chart;
    var grp   = _getGroup(bar);
    var latC  = parseFloat(bar.dataset.latC);
    var stepM = parseFloat(bar.querySelector('.align-step').value);
    if (!_gpsOff[key]) _gpsOff[key] = {dLat:0, dLon:0};

    var cosL = Math.cos(latC * Math.PI / 180);
    var dLat = 0, dLon = 0;
    if (dir === 'N') dLat =  stepM / 111320;
    if (dir === 'S') dLat = -stepM / 111320;
    if (dir === 'E') dLon =  stepM / (111320 * cosL);
    if (dir === 'W') dLon = -stepM / (111320 * cosL);
    _gpsOff[key].dLat += dLat;
    _gpsOff[key].dLon += dLon;

    grp.forEach(function(gid) { _applyViewportOne(gid, dLat, dLon); });
    _updateAllReadouts(key);
}

function resetAlignGPS(btn) {
    var bar = btn.closest('.align-bar');
    var key = bar.dataset.key || bar.dataset.chart;
    var grp = _getGroup(bar);
    var off = _gpsOff[key];
    if (!off || (off.dLat === 0 && off.dLon === 0)) return;

    grp.forEach(function(gid) {
        var div  = document.getElementById(gid);
        var orig = _origLayout[gid];
        if (!div || !div.layout || !orig) return;
        Plotly.relayout(div, {
            'xaxis.range[0]': orig.xr0, 'xaxis.range[1]': orig.xr1,
            'yaxis.range[0]': orig.yr0, 'yaxis.range[1]': orig.yr1,
            'images[0].x':    orig.ix,  'images[0].y':    orig.iy
        });
        delete _origLayout[gid];
    });
    _gpsOff[key] = {dLat:0, dLon:0};
    _updateAllReadouts(key);
}

function copyAlignOffset(btn) {
    var bar = btn.closest('.align-bar');
    var key = bar.dataset.key || bar.dataset.chart;
    var off = _gpsOff[key] || {dLat:0, dLon:0};
    var txt = '--lat-offset ' + off.dLat.toFixed(6) + ' --lon-offset ' + off.dLon.toFixed(6);
    navigator.clipboard.writeText(txt).then(function() {
        btn.textContent = 'Copied!';
        setTimeout(function() { btn.textContent = 'Copy offset'; }, 2000);
    });
}

/* ── sector lap highlight state ─────────────────────────────────────────────
   _sLS  { chartId: { traceIdx: colorHex } }   highlighted non-best laps
   _sBV  { chartId: bool }                      best lap visible?
   _sGV  { chartId: bool }                      speed gradient on?
   Trace layout per chart:
     0 .. greyCount-1  →  non-best laps  (lines)
     greyCount          →  best lap speed-gradient markers
     greyCount+1        →  best lap solid line (hidden by default)
   Table rows (class .lap-sel-row) ARE the selection controls.
*/
var _sLS = {}, _sBV = {}, _sGV = {};
var _sLC = ['#E8002D','#0090D0','#FF8000','#00D2BE','#9B59B6','#39B54A','#FF87BC','#B6BABD'];

function _bestVisible(cid) { return _sBV.hasOwnProperty(cid) ? _sBV[cid] : true; }
function _gradOn(cid)      { return _sGV.hasOwnProperty(cid) ? _sGV[cid] : true; }

function _redrawSL(cid, gc, st) {
    var any = Object.keys(st).length > 0;
    for (var i = 0; i < gc; i++) {
        if (st.hasOwnProperty(i))
            Plotly.restyle(cid, {opacity:1.0,'line.width':3.0,'line.color':st[i]}, [i]);
        else
            Plotly.restyle(cid, {opacity:any?0.05:0.4,'line.width':1.5,'line.color':'#9090A8'}, [i]);
    }
    var bv = _bestVisible(cid), grad = _gradOn(cid);
    Plotly.restyle(cid, {opacity: bv && grad  ? 1.0 : 0.0}, [gc]);
    Plotly.restyle(cid, {opacity: bv && !grad ? 1.0 : 0.0}, [gc + 1]);
}

function _applyRowColor(tr, c) {
    tr.style.background = c + '16';
    if (tr.cells[0]) tr.cells[0].style.borderLeft = '3px solid ' + c;
    tr.classList.add('lap-sel-active');
}
function _clearRowColor(tr) {
    tr.style.background = '';
    if (tr.cells[0]) tr.cells[0].style.borderLeft = '';
    tr.classList.remove('lap-sel-active');
}
function _syncRowDim(table, st) {
    if (!table) return;
    var any = Object.keys(st).length > 0;
    table.querySelectorAll('.lap-sel-row:not(.lap-sel-best)').forEach(function(row) {
        var ri = parseInt(row.dataset.traceIdx);
        row.classList.toggle('lap-sel-dim', any && !st.hasOwnProperty(ri));
    });
}

function toggleSectorLap(chartId, traceIdx, greyCount, evt) {
    if (!_sLS[chartId]) _sLS[chartId] = {};
    var st  = _sLS[chartId];
    var tr  = evt.currentTarget;
    var tbl = tr.closest('.lap-sel-table');
    if (st.hasOwnProperty(traceIdx)) {
        delete st[traceIdx];
        _clearRowColor(tr);
    } else {
        var used = Object.values(st);
        var c = _sLC.find(function(x){return used.indexOf(x)<0;}) || _sLC[used.length%_sLC.length];
        st[traceIdx] = c;
        _applyRowColor(tr, c);
    }
    _syncRowDim(tbl, st);
    _redrawSL(chartId, greyCount, st);
}

function toggleBestLap(chartId, greyCount, tr) {
    _sBV[chartId] = !_bestVisible(chartId);
    var on = _bestVisible(chartId);
    if (on) {
        _applyRowColor(tr, '#E8002D');
    } else {
        _clearRowColor(tr);
        tr.classList.add('lap-sel-dim');
    }
    _redrawSL(chartId, greyCount, _sLS[chartId] || {});
}

function toggleGradient(chartId, greyCount, sw) {
    _sGV[chartId] = sw.checked;
    _redrawSL(chartId, greyCount, _sLS[chartId] || {});
}

function resetSectorLaps(chartId, greyCount, btn) {
    _sLS[chartId] = {};
    _sBV[chartId] = true;
    var gi = Array.from({length:greyCount},function(_,i){return i;});
    if (gi.length) Plotly.restyle(chartId,{opacity:0.4,'line.width':1.5,'line.color':'#9090A8'},gi);
    var grad = _gradOn(chartId);
    Plotly.restyle(chartId, {opacity: grad ? 1.0 : 0.0}, [greyCount]);
    Plotly.restyle(chartId, {opacity: grad ? 0.0 : 1.0}, [greyCount + 1]);
    // Reset all row styles
    var tbl = btn.closest('.sector-detail-right').querySelector('.lap-sel-table');
    if (tbl) {
        tbl.querySelectorAll('.lap-sel-row').forEach(function(row) {
            row.classList.remove('lap-sel-active','lap-sel-dim');
            row.style.background = '';
            if (row.cells[0]) row.cells[0].style.borderLeft = '';
        });
        // Restore best lap row to active state
        var bestRow = tbl.querySelector('.lap-sel-best');
        if (bestRow) _applyRowColor(bestRow, '#E8002D');
    }
}
"""

# ── sector race-lines (single driver, all laps) ───────────────────────────────
def _sector_laps_fig(sector_idx, n_sectors, sess):
    """GPS race-lines for every flying lap of one driver, zoomed to one sector.

    Returns (fig, others, bk, idx, times) so the caller can generate
    highlight buttons that target each trace by index.

    Trace layout:
      0 .. len(others)-1  →  non-best laps (grey lines, solid colour so JS restyle works)
      len(others)         →  best lap speed heatmap
    """
    bk, idx, times = _best_k(sess)
    others = [k for k in idx if k != bk]
    _GREY  = "#8890A8"
    d      = sess.df[sess.df["GPS Nsat"] >= 6]

    # Determine sector window from best lap for snapshot extent
    best  = d[d["lap"] == bk].sort_values("lap_dist")
    bl_d  = best["lap_dist"].values
    L     = float(bl_d[-1])
    e     = np.linspace(0, L, n_sectors + 1)
    s0, s1 = e[sector_idx], e[sector_idx + 1]
    buf   = (s1 - s0) * 0.25   # 25 % context each side
    win_best = best[(bl_d >= max(0, s0 - buf)) & (bl_d <= min(L, s1 + buf))]
    z_lats = _la(win_best["GPS Latitude"].values  if len(win_best) > 1 else d["GPS Latitude"].values)
    z_lons = _lo(win_best["GPS Longitude"].values if len(win_best) > 1 else d["GPS Longitude"].values)

    zoom   = _best_tile_zoom(z_lats, z_lons, max_tiles=36)
    lat_c  = float((np.max(z_lats) + np.min(z_lats)) / 2)
    b64, ox, oy, W, H = _sat_snapshot(z_lats, z_lons, zoom)

    def _px(la, lo):
        return _gps2px(_la(la), _lo(lo), ox, oy, zoom)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=f"data:image/jpeg;base64,{b64}",
        xref="x", yref="y",
        x=0, y=0, sizex=W, sizey=H,
        xanchor="left", yanchor="top",
        sizing="stretch", layer="below"))

    for k in others:
        lap = d[d["lap"] == k].sort_values("lap_dist")
        ld  = lap["lap_dist"].values
        li  = idx.index(k)
        t   = times[li]
        if len(ld) < 3:
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                line=dict(color=_GREY, width=1.5), opacity=0.4,
                name=f"L{li+1}", showlegend=False))
            continue
        L_k = float(ld[-1])
        ek  = np.linspace(0, L_k, n_sectors + 1)
        s0k, s1k = ek[sector_idx], ek[sector_idx + 1]
        bufk = (s1k - s0k) * 0.25
        win = lap[(ld >= max(0, s0k - bufk)) & (ld <= min(L_k, s1k + bufk))]
        if len(win) < 2:
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                line=dict(color=_GREY, width=1.5), opacity=0.4,
                name=f"L{li+1}", showlegend=False))
            continue
        xs, ys = _px(win["GPS Latitude"].values, win["GPS Longitude"].values)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(color=_GREY, width=2), opacity=0.4,
            name=f"L{li+1} · {t:.3f}s", showlegend=False,
            text=[f"{v:.1f} km/h" for v in win["GPS Speed"].values],
            hovertemplate=f"L{li+1}: %{{text}}<extra></extra>"))

    # best lap — two traces: speed-gradient markers (grad on) + solid line (grad off)
    bi = idx.index(bk)
    if len(win_best) >= 2:
        sp = win_best["GPS Speed"].values
        xs, ys = _px(win_best["GPS Latitude"].values, win_best["GPS Longitude"].values)
        # trace greyCount: gradient markers (visible by default)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
            marker=dict(color=sp, colorscale="RdYlGn", size=6, showscale=True,
                        cmin=float(np.percentile(sp, 5)),
                        cmax=float(np.percentile(sp, 95)),
                        colorbar=dict(title="km/h", thickness=12, len=0.7,
                                      tickfont=dict(size=10))),
            name=f"L{bi+1} ★ {min(times):.3f}s", showlegend=False,
            text=[f"{v:.1f} km/h" for v in sp],
            hovertemplate="%{text}<extra></extra>"))
        # trace greyCount+1: solid red line (hidden by default — shown when gradient off)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(color="#E8002D", width=3),
            opacity=0.0,
            name=f"L{bi+1} ★ solid", showlegend=False,
            text=[f"{v:.1f} km/h" for v in sp],
            hovertemplate="%{text}<extra></extra>"))
    else:
        # placeholder traces so index arithmetic stays consistent
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False))
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",   showlegend=False, opacity=0.0))

    fig.update_layout(
        **_MAP_LAYOUT,
        title=dict(
            text=(f"Race Lines — S{sector_idx+1}<br>"
                  f"<sup>Esri World Imagery  ·  ★ best lap speed  ·  use buttons to highlight laps</sup>"),
            font=dict(size=13)),
        xaxis=dict(range=[0, W], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        yaxis=dict(range=[H, 0], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=65, b=0),
        showlegend=False,
        height=420,
    )
    return fig, others, bk, idx, times, zoom, lat_c


# ── sector zoom chart (combined tab — all sessions) ───────────────────────────
def _sector_zoom_fig(sector_idx, n_sectors, all_sessions, focus_sess, rlat, rlon):
    """GPS race-line for one sector, all sessions overlaid.

    The focus driver is drawn as a speed-heatmap scatter; others as coloured lines.
    A 15% distance buffer is added each side so the corner entry/exit has context.
    """
    fig = go.Figure()

    for i, sess in enumerate(all_sessions):
        c = _COLORS[i % len(_COLORS)]
        bk, _, _ = _best_k(sess)
        d_xy, _, _ = _gps_xy(sess, ref_lat=rlat, ref_lon=rlon)
        best = d_xy[d_xy["lap"] == bk].sort_values("lap_dist")
        if len(best) < 3:
            continue

        bl_d = best["lap_dist"].values
        L    = float(bl_d[-1])
        edges  = np.linspace(0, L, n_sectors + 1)
        s0, s1 = edges[sector_idx], edges[sector_idx + 1]
        buf    = (s1 - s0) * 0.18

        # buffered window for context
        win = best[(bl_d >= max(0, s0 - buf)) & (bl_d <= min(L, s1 + buf))]
        # exact sector window
        sec = best[(bl_d >= s0) & (bl_d <= s1)]
        if len(win) < 2:
            continue

        is_focus = (sess is focus_sess)

        if is_focus:
            # speed-heatmap on exact sector
            sp = win["GPS Speed"].values
            fig.add_trace(go.Scatter(
                x=win["x"].values, y=win["y"].values,
                mode="markers",
                marker=dict(
                    color=sp, colorscale="RdYlGn", size=6, showscale=True,
                    cmin=float(np.percentile(sp, 5)),
                    cmax=float(np.percentile(sp, 95)),
                    colorbar=dict(title="Speed<br>(km/h)", thickness=12, len=0.75,
                                  tickfont=dict(size=10)),
                ),
                name=f"{sess.racer} ★",
                hovertemplate="%{marker.color:.1f} km/h<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=win["x"].values, y=win["y"].values,
                mode="lines",
                line=dict(color=c, width=2.5),
                opacity=0.72,
                name=sess.racer,
                hovertemplate=f"{sess.racer}: %{{text}} km/h<extra></extra>",
                text=[f"{v:.1f}" for v in win["GPS Speed"].values],
            ))

        # sector entry / exit markers
        for pt, sym, label in [(sec.iloc[0], "triangle-right", "entry"),
                                (sec.iloc[-1], "square",        "exit")]:
            fig.add_trace(go.Scatter(
                x=[float(pt["x"])], y=[float(pt["y"])],
                mode="markers",
                marker=dict(symbol=sym, color=c, size=9,
                            line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate=f"{sess.racer} {label}<extra></extra>",
            ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=(f"Race Line — S{sector_idx+1}"
                  f"<br><sup>★ speed heatmap · ▶ entry · ■ exit</sup>"),
            font=dict(size=13),
        ),
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False,
                   zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        height=400,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── theoretical best ─────────────────────────────────────────────────────────
def _theo_best_fig(sess, tbl, n_sectors):
    """
    Track map where each sector is drawn from the lap that was fastest in that sector.
    Each sector gets a distinct colour; all laps shown as dim background.
    """
    d = sess.df[sess.df["GPS Nsat"] >= 6]
    bk, idx, times = _best_k(sess)

    all_lats = _la(d["GPS Latitude"].values)
    all_lons = _lo(d["GPS Longitude"].values)
    zoom  = _best_tile_zoom(all_lats, all_lons)
    lat_c = float((np.max(all_lats) + np.min(all_lats)) / 2)
    b64, ox, oy, W, H = _sat_snapshot(all_lats, all_lons, zoom)

    def _px(la, lo):
        return _gps2px(_la(la), _lo(lo), ox, oy, zoom)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=f"data:image/jpeg;base64,{b64}",
        xref="x", yref="y",
        x=0, y=0, sizex=W, sizey=H,
        xanchor="left", yanchor="top",
        sizing="stretch", layer="below"))

    # All laps — dim background
    for k in idx:
        lap = d[d["lap"] == k]
        xs, ys = _px(lap["GPS Latitude"].values, lap["GPS Longitude"].values)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(color="rgba(200,200,200,0.13)", width=1.2),
            showlegend=False, hoverinfo="skip"))

    # Per-sector best GPS segments
    for si, col in enumerate(tbl.columns):
        best_lap_k = int(tbl[col].idxmin())
        st = float(tbl[col].min())
        c  = _COLORS[si % len(_COLORS)]
        li = idx.index(best_lap_k) if best_lap_k in idx else -1
        lap_label = f"L{li+1}" if li >= 0 else "?"

        lap_data = d[d["lap"] == best_lap_k].sort_values("lap_dist")
        ld = lap_data["lap_dist"].values
        if len(ld) < 3:
            continue
        L_k = float(ld[-1])
        edges = np.linspace(0, L_k, n_sectors + 1)
        s0, s1 = edges[si], edges[si + 1]
        win = lap_data[(ld >= s0) & (ld <= s1)]
        if len(win) < 2:
            continue
        xs, ys = _px(win["GPS Latitude"].values, win["GPS Longitude"].values)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
            line=dict(color=c, width=4.5),
            name=f"{col} · {lap_label} · {st:.3f}s",
            text=[f"{v:.1f} km/h" for v in win["GPS Speed"].values],
            hovertemplate=f"{col} ({lap_label}): %{{text}}<extra></extra>"))

    # Sector boundary markers (from best actual lap for consistent geometry)
    best_lap = d[d["lap"] == bk].sort_values("lap_dist")
    bl_d = best_lap["lap_dist"].values
    L    = float(bl_d[-1])
    edges = np.linspace(0, L, n_sectors + 1)[1:-1]
    elat = np.array([np.interp(e, bl_d, best_lap["GPS Latitude"].values)  for e in edges])
    elon = np.array([np.interp(e, bl_d, best_lap["GPS Longitude"].values) for e in edges])
    ex, ey = _px(elat, elon)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers+text",
        marker=dict(color="white", size=9, opacity=0.75),
        text=[f"S{i+1}" for i in range(len(edges))],
        textposition="top right", textfont=dict(color="white"),
        showlegend=False))

    fig.update_layout(
        **_MAP_LAYOUT,
        title=dict(
            text="Theoretical Best — Composite Race Line<br>"
                 "<sup>Each colour = that sector's GPS trace from its personal-best lap</sup>",
            font=dict(size=13)),
        xaxis=dict(range=[0, W], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        yaxis=dict(range=[H, 0], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=65, b=0),
        height=500,
        legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)))
    return fig, zoom, lat_c


def _theo_best_html(sess, n_sectors, racer_slug="", sat_group=None, group_key=None):
    tbl        = kt.sector_table(sess, n_sectors)
    theo_time  = float(tbl.min().sum())
    actual_best = float(tbl.sum(axis=1).min())
    gain       = actual_best - theo_time

    # KPI banner
    kpi_html = (
        f'<div class="theo-kpi-row">'
        f'<div class="theo-kpi">'
        f'  <span class="theo-label">Theoretical Best</span>'
        f'  <span class="theo-val hi">{theo_time:.3f}s</span>'
        f'</div>'
        f'<div class="theo-kpi">'
        f'  <span class="theo-label">Actual Best</span>'
        f'  <span class="theo-val">{actual_best:.3f}s</span>'
        f'</div>'
        f'<div class="theo-kpi">'
        f'  <span class="theo-label">Potential Gain</span>'
        f'  <span class="theo-val pos">{gain:.3f}s</span>'
        f'</div>'
        f'</div>'
    )

    # Per-sector breakdown table
    _, idx, _ = _best_k(sess)
    rows = ""
    for si, col in enumerate(tbl.columns):
        best_lap_k = int(tbl[col].idxmin())
        st = float(tbl[col].min())
        col_mean = float(tbl[col].mean())
        li = idx.index(best_lap_k) if best_lap_k in idx else -1
        lap_label = f"L{li+1}" if li >= 0 else "?"
        c = _COLORS[si % len(_COLORS)]
        rows += (
            f'<tr>'
            f'<td><span class="sector-dot" style="background:{c}"></span>{col}</td>'
            f'<td class="lbl">{lap_label}</td>'
            f'<td class="neg">{st:.3f}s</td>'
            f'<td class="dim">{col_mean:.3f}s</td>'
            f'<td class="pos">+{col_mean - st:.3f}s</td>'
            f'</tr>'
        )
    breakdown_table = (
        '<table class="st"><thead><tr>'
        '<th>Sector</th><th>Source Lap</th>'
        '<th>Best Time</th><th>Mean Time</th><th>vs Mean</th>'
        '</tr></thead><tbody>' + rows + '</tbody></table>'
    )

    theo_div = f"{racer_slug}_theo_map" if racer_slug else None
    theo_fig, theo_zoom, theo_lat_c = _theo_best_fig(sess, tbl, n_sectors)
    fig_html  = _fig_html(theo_fig, div_id=theo_div)
    theo_align = _align_controls_html(
        theo_div, theo_lat_c, theo_zoom,
        group=sat_group, key=group_key) if theo_div else ""

    return (
        f'<div class="theo-best-wrap">'
        f'{kpi_html}'
        f'<div class="theo-body">'
        f'<div class="theo-map-col">'
        f'<div class="box theo-map">{fig_html}</div>'
        f'{theo_align}'
        f'</div>'
        f'<div class="theo-table-col">{breakdown_table}</div>'
        f'</div>'
        f'</div>'
    )


# ── sector breakdown ──────────────────────────────────────────────────────────
def _sector_breakdown_html(sess, n_sectors, sat_group=None, group_key=None):
    tbl = kt.sector_table(sess, n_sectors)   # rows=laps, cols=S1..Sn
    racer_slug = sess.racer.replace(" ", "_").replace(".", "")

    # ── summary table ──
    summary_rows = ""
    for col in tbl.columns:
        vals = tbl[col]
        mean = vals.mean()
        std  = vals.std(ddof=1)
        best = vals.min()
        worst= vals.max()
        rng  = worst - best
        std_cls = ("neg" if std < 0.08 else "pos" if std > 0.25 else "")
        consistency = ("Consistent" if std < 0.08
                       else "Variable"   if std < 0.25
                       else "Scattered")
        summary_rows += (
            f'<tr>'
            f'<td class="lbl">{col}</td>'
            f'<td>{mean:.3f}</td>'
            f'<td class="{std_cls}">{std:.3f}</td>'
            f'<td class="neg">{best:.3f}</td>'
            f'<td>{worst:.3f}</td>'
            f'<td>{rng:.3f}</td>'
            f'<td class="lbl {std_cls}">{consistency}</td>'
            f'</tr>'
        )
    summary_table = (
        '<table class="st"><thead><tr>'
        '<th>Sector</th><th>Mean (s)</th><th>Std (s)</th>'
        '<th>Best (s)</th><th>Worst (s)</th><th>Range (s)</th>'
        '<th>Consistency</th>'
        '</tr></thead><tbody>' + summary_rows + '</tbody></table>'
    )

    # ── per-sector tabs ──
    tab_btns = ""
    tab_divs = ""
    for si, col in enumerate(tbl.columns):
        vals  = tbl[col]
        mean  = vals.mean()
        std   = vals.std(ddof=1)
        best_t= vals.min()
        worst_t = vals.max()
        tab_id = f"{racer_slug}_sec_{col}"
        active = " active" if si == 0 else ""

        tab_btns += (
            f'<button class="stab-btn{active}" '
            f'onclick="showSectorTab(\'{tab_id}\', this)">{col}</button>'
        )

        std_cls = "good" if std < 0.08 else "warn" if std > 0.25 else ""
        stats_bar = (
            f'<div class="sec-stats-bar">'
            f'<span><span class="slabel">Mean</span>'
            f'<span class="sval">{mean:.3f}s</span></span>'
            f'<span><span class="slabel">Std</span>'
            f'<span class="sval {std_cls}">{std:.3f}s</span></span>'
            f'<span><span class="slabel">Best</span>'
            f'<span class="sval good">{best_t:.3f}s</span></span>'
            f'<span><span class="slabel">Worst</span>'
            f'<span class="sval">{worst_t:.3f}s</span></span>'
            f'<span><span class="slabel">Range</span>'
            f'<span class="sval">{worst_t - best_t:.3f}s</span></span>'
            f'</div>'
        )

        chart_id = f"{racer_slug}_ch_{col}"
        fig, others, bk_k, laps_idx, laps_times, sec_zoom, sec_lat_c = \
            _sector_laps_fig(si, n_sectors, sess)
        zoom_html   = _fig_html(fig, div_id=chart_id)
        sector_align = _align_controls_html(
            chart_id, sec_lat_c, sec_zoom,
            group=sat_group, key=group_key)

        grey_count = len(others)
        gc_s       = str(grey_count)
        bi         = laps_idx.index(bk_k)
        bt         = min(laps_times)

        # ── controls bar (gradient switch + reset) ──
        grad_switch = (
            f'<label class="grad-switch" title="Toggle speed gradient">'
            f'<input type="checkbox" checked '
            f'onchange="toggleGradient(\'{chart_id}\',{gc_s},this)">'
            f'<span class="grad-track"><span class="grad-thumb"></span></span>'
            f'<span class="grad-label">Speed gradient</span>'
            f'</label>'
        )
        controls_bar = (
            f'<div class="sel-controls">'
            f'{grad_switch}'
            f'<button class="lap-reset-btn" '
            f'onclick="resetSectorLaps(\'{chart_id}\',{gc_s},this)">Reset</button>'
            f'</div>'
        )

        # ── interactive lap table — rows ARE the selection buttons ──
        lap_rows = ""
        for li, (lap_num, lap_t) in enumerate(vals.items()):
            d_best = lap_t - best_t
            d_mean = lap_t - mean
            is_best = (lap_t == best_t)
            dm_cls  = "neg" if d_mean < -0.001 else ("pos" if d_mean > std else "")

            if is_best:
                # Best lap row — toggleable via toggleBestLap, starts active
                lap_rows += (
                    f'<tr class="lap-sel-row lap-sel-best lap-sel-active" '
                    f'style="background:rgba(232,0,45,.08)" '
                    f'onclick="toggleBestLap(\'{chart_id}\',{gc_s},this)">'
                    f'<td class="lbl" style="border-left:3px solid #E8002D;padding-left:9px">'
                    f'L{li+1}&nbsp;★</td>'
                    f'<td class="neg">{lap_t:.3f}</td>'
                    f'<td>—</td><td>—</td>'
                    f'</tr>'
                )
            else:
                try:
                    oi = others.index(lap_num)
                except ValueError:
                    oi = -1
                db_cls = "neg" if d_best < 0.001 else ""
                lap_rows += (
                    f'<tr class="lap-sel-row" data-trace-idx="{oi}" '
                    f'onclick="toggleSectorLap(\'{chart_id}\',{oi},{gc_s},event)">'
                    f'<td class="lbl" style="padding-left:12px">L{li+1}</td>'
                    f'<td>{lap_t:.3f}</td>'
                    f'<td class="{db_cls}">{d_best:+.3f}</td>'
                    f'<td class="{dm_cls}">{d_mean:+.3f}</td>'
                    f'</tr>'
                )

        tab_divs += (
            f'<div id="{tab_id}" class="stab-content{active}">'
            f'<div class="sector-detail-grid">'
            f'<div class="sector-map-col">'
            f'<div class="box sector-map">{zoom_html}</div>'
            f'{sector_align}'
            f'</div>'
            f'<div class="sector-detail-right">'
            f'{stats_bar}'
            f'{controls_bar}'
            f'<table class="st lap-sel-table"><thead><tr>'
            f'<th>Lap</th><th>Time (s)</th><th>Δ Best</th><th>Δ Mean</th>'
            f'</tr></thead><tbody>{lap_rows}</tbody></table>'
            f'</div></div>'
            f'</div>'
        )

    sector_tabs = (
        f'<div class="sector-tab-group mt">'
        f'<div class="stab-bar">{tab_btns}</div>'
        f'{tab_divs}'
        f'</div>'
    )
    return summary_table + sector_tabs


# ── tab builders ──────────────────────────────────────────────────────────────
def _driver_tab(sess, color, n_sectors, sat_group=None, group_key="all_sat"):
    s = kt.consistency_stats(sess)
    ts = kt.top_speed(sess)
    grp = kt.rpm_per_kmh(sess)
    _, idx, times = _best_k(sess)
    trend = ""
    if len(times) > 2:
        slope = np.polyfit(range(len(times)), times, 1)[0]
        trend = f"  trend {slope*1000:+.0f} ms/lap"

    kpis = "".join([
        _kpi("Best Lap", f"{s['best']}s", hi=True),
        _kpi("Avg Lap", f"{s['mean']}s", f"std {s['std']}s"),
        _kpi("Top Speed", f"{ts} km/h"),
        _kpi("Gearing", f"{grp}" if grp else "—", "km/h / 1000 rpm"),
        _kpi("Flying Laps", str(s["n_flying"])),
        _kpi("Spread", f"{s['range']}s", "worst − best"),
    ])

    _track_f, _lat_c, _zoom = _track_fig(sess, n_sectors, color)
    _racer_slug = sess.racer.replace(" ", "_").replace(".", "")
    _track_div  = f"{_racer_slug}_trackmap"
    _theo_div   = f"{_racer_slug}_theo_map"
    # Use the universal sat_group and group_key passed in from build_dashboard
    _sat_group  = sat_group or _track_div
    track    = _fig_html(_track_f, div_id=_track_div)
    track_align = _align_controls_html(_track_div, _lat_c, _zoom,
                                        group=_sat_group, key=group_key)
    delta    = _fig_html(_self_delta_fig(sess, color))
    laps     = _fig_html(_lap_time_fig(sess, color))
    gg_h     = _fig_html(_gg_fig(sess, color))
    rpm_h    = _fig_html(_rpm_fig(sess, color))
    s5       = _stage5_html(sess, n_sectors)
    sec_brkd = _sector_breakdown_html(sess, n_sectors,
                                       sat_group=_sat_group, group_key=group_key)
    theo_h   = _theo_best_html(sess, n_sectors, _racer_slug,
                                sat_group=_sat_group, group_key=group_key)

    # corner apex table (entry / apex / exit)
    csp = kt.corner_speed_profile(sess, n_sectors)
    apex_cols = [f"S{i+1}" for i in range(n_sectors)]
    apex_rows = "".join(
        f'<tr><td class="lbl">{s_}</td>'
        f'<td>{round(csp[f"{s_}_apex"].mean(),1)}</td>'
        f'<td>{round(csp[f"{s_}_entry"].mean(),1)}</td>'
        f'<td>{round(csp[f"{s_}_exit"].mean(),1)}</td></tr>'
        for s_ in apex_cols if f"{s_}_apex" in csp.columns
    )
    apex_html = (
        '<table class="st"><thead><tr>'
        '<th>Sector</th><th>Apex km/h</th><th>Entry km/h</th><th>Exit km/h</th>'
        '</tr></thead><tbody>' + apex_rows + '</tbody></table>'
    )

    return f"""
<div class="kpi-row">{kpis}</div>

<div class="section">
  <div class="sec-title">Track Map</div>
  <div class="box">{track}</div>
  {track_align}
</div>

<div class="section">
  <div class="sec-title">All Laps vs Personal Best — Speed Trace + Time Delta</div>
  <div class="box">{delta}</div>
</div>

<div class="section">
  <div class="sec-title">Sector Breakdown</div>
  {sec_brkd}
</div>

<div class="section">
  <div class="sec-title">Theoretical Best Lap</div>
  {theo_h}
</div>

<div class="section">
  <div class="sec-title">Lap Times &amp; Corner Apex Speed</div>
  <div class="grid2">
    <div class="box">{laps}</div>
    <div>
      <div class="sec-title" style="margin-top:0">Corner Apex Speed (mean, km/h)</div>
      {apex_html}
    </div>
  </div>
</div>

<div class="section">
  <div class="sec-title">Driver Dynamics</div>
  <div class="grid2">
    <div class="box">{gg_h}</div>
    <div class="box">{rpm_h}</div>
  </div>
</div>

<div class="section">
  <div class="sec-title">Stage 5 Summary</div>
  {s5}
</div>
"""


def _combined_tab(sessions, n_sectors, sat_group=None, group_key="all_sat"):
    ref = min(sessions, key=lambda s: kt.consistency_stats(s)["best"])
    delta  = _fig_html(_combined_delta_fig(sessions))
    _ctrack_f, _clat_c, _czoom = _combined_track_fig(sessions, n_sectors)
    track  = _fig_html(_ctrack_f, div_id="combined_trackmap")
    ctrack_align = _align_controls_html("combined_trackmap", _clat_c, _czoom,
                                         group=sat_group or "combined_trackmap",
                                         key=group_key)
    laps   = _fig_html(_combined_laptimes_fig(sessions))
    gg_h   = _fig_html(_combined_gg_fig(sessions))
    apex   = _apex_table_html(sessions, n_sectors)
    hth    = _hth_html(sessions)

    return f"""
<div class="section">
  <div class="sec-title">Speed Trace + Time Delta  (reference: {ref.racer})</div>
  <div class="box">{delta}</div>
</div>

<div class="section">
  <div class="sec-title">Track Map — Best Lap Overlay</div>
  <div class="box">{track}</div>
  {ctrack_align}
</div>

<div class="section">
  <div class="sec-title">Apex Speed Table — Ranked by Coaching Priority</div>
  {apex}
</div>

<div class="section">
  <div class="sec-title">Lap Times — All Drivers</div>
  <div class="grid2">
    <div class="box">{laps}</div>
    <div class="box">{gg_h}</div>
  </div>
</div>

<div class="section">
  <div class="sec-title">Head-to-Head Summary</div>
  {hth}
</div>
"""

# ── main assembly ──────────────────────────────────────────────────────────────
def _build_header(sessions):
    if len(sessions) == 1:
        s = kt.consistency_stats(sessions[0])
        ts = kt.top_speed(sessions[0])
        h1 = sessions[0].racer
        meta = (f'Best lap <span class="hi">{s["best"]}s</span>'
                f'&ensp;·&ensp;{s["n_flying"]} flying laps'
                f'&ensp;·&ensp;Top speed <span class="hi">{ts} km/h</span>'
                f'&ensp;·&ensp;Avg <span class="hi">{s["mean"]}s</span>'
                f'&ensp;·&ensp;Std <span class="hi">{s["std"]}s</span>')
    else:
        parts = [f'<span class="hi">{s.racer}</span>'
                 f'&thinsp;<span style="color:var(--dim);font-weight:300;'
                 f'font-size:.55em;vertical-align:.2em"> '
                 f'{kt.consistency_stats(s)["best"]}s</span>'
                 for s in sessions]
        h1_inner = '<span class="vs">vs</span>'.join(parts)
        h1 = h1_inner
        ref = min(sessions, key=lambda s: kt.consistency_stats(s)["best"])
        gap = round(
            max(kt.consistency_stats(s)["best"] for s in sessions) -
            kt.consistency_stats(ref)["best"], 3)
        meta = (f'Reference: <span class="hi">{ref.racer}</span>'
                f'&ensp;·&ensp;Gap: <span class="hi">{gap:+}s</span>'
                f'&ensp;·&ensp;{len(sessions)} drivers')
    return f"""<div class="header">
  <div class="header-eyebrow">Kart Telemetry &nbsp;/&nbsp; AiM CSV Analysis</div>
  <h1>{h1}</h1>
  <div class="header-meta">{meta}</div>
</div>"""


def build_dashboard(sessions, n_sectors):
    _plotly_embedded[0] = False   # reset so Plotly JS embeds on first figure

    # Build a single universal satellite-map group so ALL align bars share one offset.
    _all_sat = []
    for sess in sessions:
        slug = sess.racer.replace(" ", "_").replace(".", "")
        _all_sat.append(f"{slug}_trackmap")
        _all_sat.append(f"{slug}_theo_map")
        for i in range(n_sectors):
            _all_sat.append(f"{slug}_ch_S{i+1}")
    if len(sessions) > 1:
        _all_sat.append("combined_trackmap")
    _universal_group = ",".join(_all_sat)
    _universal_key   = "all_sat"

    tabs = []
    for i, sess in enumerate(sessions):
        tid = f"tab_{sess.racer.replace(' ', '_')}"
        tabs.append((tid, sess.racer,
                     _driver_tab(sess, _COLORS[i % len(_COLORS)], n_sectors,
                                 sat_group=_universal_group, group_key=_universal_key)))
    if len(sessions) > 1:
        tabs.append(("tab_combined", "Combined",
                     _combined_tab(sessions, n_sectors,
                                   sat_group=_universal_group, group_key=_universal_key)))

    btn_html = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" '
        f'onclick="showTab(\'{tid}\',this)">{label}</button>'
        for i, (tid, label, _) in enumerate(tabs)
    )
    div_html = "".join(
        f'<div id="{tid}" class="tab-content{" active" if i==0 else ""}">{body}</div>'
        for i, (tid, _, body) in enumerate(tabs)
    )
    header = _build_header(sessions)
    page_title = " vs ".join(s.racer for s in sessions)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title} — Kart Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<button id="theme-toggle" class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme">
  <svg class="theme-icon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
  <span class="theme-label">Light</span>
</button>
<div class="page">
  {header}
  <div class="tab-bar">{btn_html}</div>
  {div_html}
</div>
<script>{_JS}</script>
</body>
</html>"""


def _out_path(ref_path, filename):
    return os.path.join(os.path.dirname(os.path.abspath(ref_path)), filename)


def main():
    ap = argparse.ArgumentParser(description="Generate kart telemetry HTML dashboard")
    ap.add_argument("files", nargs="+", help="AiM CSV files (one or more drivers)")
    ap.add_argument("--sectors", type=int, default=6)
    ap.add_argument("--no-beacon", action="store_true")
    ap.add_argument("--lat-offset", type=float, default=0.0,
                    help="Latitude offset in degrees to align GPS trace with satellite imagery")
    ap.add_argument("--lon-offset", type=float, default=0.0,
                    help="Longitude offset in degrees to align GPS trace with satellite imagery")
    ap.add_argument("--auto-align", action="store_true",
                    help="Auto-detect GPS registration offset via satellite tile cross-correlation")
    ap.add_argument("--align-zoom", type=int, default=18,
                    help="Tile zoom level used for --auto-align (default 18; try 17 for large tracks)")
    args = ap.parse_args()

    sessions = [kt.load_session(p, no_beacon=args.no_beacon) for p in args.files]

    if args.auto_align:
        dlat, dlon = find_offset(sessions[0], zoom=args.align_zoom)
        _OFF[0] = dlat
        _OFF[1] = dlon
    else:
        _OFF[0] = args.lat_offset
        _OFF[1] = args.lon_offset
        if args.lat_offset or args.lon_offset:
            print(f"GPS offset applied: lat {args.lat_offset:+.6f}°  lon {args.lon_offset:+.6f}°")

    html = build_dashboard(sessions, args.sectors)

    if len(sessions) == 1:
        name = sessions[0].racer.replace(" ", "_")
        out = _out_path(sessions[0].path, f"{name}_dashboard.html")
    else:
        names = "_vs_".join(s.racer.replace(" ", "_") for s in sessions)
        out = _out_path(sessions[0].path, f"{names}_dashboard.html")

    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out}")
    print(f"Open:  file:///{out.replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
