# Kart Telemetry Dashboard

A self-contained kart telemetry analysis tool that reads AiM CSV data and produces an interactive HTML dashboard — no server, no login, no cloud. Open the output file in any browser.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [AiM CSV Format](#aim-csv-format)
4. [Data Cleaning](#data-cleaning)
5. [Analysis Engine — `kart_telemetry.py`](#analysis-engine)
6. [Dashboard — `dashboard.py`](#dashboard)
   - [Solo Driver Tab](#solo-driver-tab)
   - [Sector Breakdown](#sector-breakdown)
   - [Theoretical Best Lap](#theoretical-best-lap)
   - [Combined / Head-to-Head Tab](#combined--head-to-head-tab)
7. [Satellite Imagery](#satellite-imagery)
8. [GPS Registration Offset](#gps-registration-offset)
9. [Dark / Light Mode](#dark--light-mode)
10. [CLI Reference](#cli-reference)
11. [Architecture](#architecture)

---

## Requirements

```
pip install pandas numpy plotly requests Pillow scipy
```

| Package   | Used for                                              |
|-----------|-------------------------------------------------------|
| pandas    | CSV parsing, sector table, lap annotation             |
| numpy     | All numerical computation                             |
| plotly    | Interactive charts and maps embedded in HTML          |
| requests  | Downloading ESRI satellite tiles at build time        |
| Pillow    | Stitching tile images into a single JPEG              |
| scipy     | FFT cross-correlation for auto GPS alignment          |

`scipy` is only needed if you use `--auto-align`. Everything else runs without it.

---

## Quick Start

```bash
# Single driver
python dashboard.py Jamie.csv

# Head-to-head (two or more drivers)
python dashboard.py Jamie.csv Joshua.csv

# Custom sector count (default 6)
python dashboard.py Jamie.csv --sectors 8

# GPS-only lap detection (no beacon data in file)
python dashboard.py Jamie.csv --no-beacon

# Correct satellite registration offset manually
python dashboard.py Jamie.csv --lat-offset 0.000135 --lon-offset -0.000090

# Auto-detect and correct satellite registration offset
python dashboard.py Jamie.csv --auto-align
```

The output is a single self-contained HTML file (e.g. `Jamie_dashboard.html` or `Jamie_vs_Joshua_dashboard.html`) written alongside the CSV files. Satellite tile images are embedded as base64 JPEG — the file works fully offline after generation. An internet connection is only required during the build step to download the tiles.

---

## AiM CSV Format

AiM loggers export a CSV with two distinct regions:

**Metadata block** (top of file, before the channel data):

| Field            | Description                                              |
|------------------|----------------------------------------------------------|
| `Racer`          | Driver name — used as the tab label in the dashboard     |
| `Sample Rate`    | Hz (typically 20 Hz for GPS-based loggers)               |
| `Beacon Markers` | Comma-separated timestamps (seconds) of S/F crossings    |
| `Segment Times`  | Pre-computed lap times from the logger — used to verify  |

**Channel data** (below the metadata):

The row immediately after the column headers is a units row (strings like `"km/h"`, `"rpm"`). The loader coerces everything to numeric and drops this row automatically.

Key channels used by the analysis:

| Channel                  | Used for                                          |
|--------------------------|---------------------------------------------------|
| `Time`                   | Sample timestamps (seconds from session start)    |
| `GPS Latitude/Longitude` | All geographic visualisations                     |
| `GPS Speed`              | Speed heatmap, braking/throttle detection         |
| `GPS Nsat`               | Quality filter — samples with `Nsat < 6` excluded |
| `GPS LatAcc`             | Lateral G (G-G diagram)                           |
| `GPS LonAcc`             | Longitudinal G (braking / throttle detection)     |
| `GPS Radius`             | Turn radius (rotation event detection)            |
| `GPS Gyro`               | Yaw rate (rotation event detection)               |
| `RPM`                    | RPM vs Speed scatter (gearing analysis)           |
| `Distance on GPS Speed`  | Cumulative distance (used for `lap_dist`)         |

---

## Data Cleaning

Multiple cleaning and filtering steps are applied at different stages of the pipeline. They are documented here in execution order.

### 1. CSV parse — units row removal

The AiM CSV places a units row (e.g. `"km/h"`, `"rpm"`, `"deg"`) immediately after the channel-name header. `load_session` reads from the header row downward, then calls:

```python
df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Time"])
```

`errors="coerce"` turns every non-numeric cell (the units row, any stray strings) into `NaN`. `dropna(subset=["Time"])` removes any row that has no valid timestamp, which includes the units row and any trailing empty rows. The result is a fully numeric DataFrame.

### 2. GPS quality filter — `GPS Nsat >= 6`

Every function that touches GPS channels (`GPS Latitude`, `GPS Longitude`, `GPS Speed`, `GPS LatAcc`, `GPS LonAcc`) first filters:

```python
d = sess.df[sess.df["GPS Nsat"] >= 6]
```

`GPS Nsat` (number of satellites locked) below 6 indicates poor fix quality. Samples with fewer satellites produce unreliable position and speed values, especially noticeable as speed spikes or position jumps. All map traces, speed heatmaps, braking/throttle markers, G-G diagrams, and the gearing regression use this filtered subset.

### 3. Lap detection — outlier lap removal

`flying_laps()` applies two filtering steps:

**Out-lap and in-lap trimming (beacon mode only):** In beacon mode the first segment is always the warm-up/out-lap (slow, tyres cold) and the last is the in-lap (slow, heading to pits). Both are unconditionally removed before any statistics are computed:

```python
if not sess.gps_beacons and len(idx) > 2:
    idx, times = idx[1:-1], times[1:-1]
```

In GPS mode this trimming is skipped because slow laps are rarely detected as valid crossings by the beacon reconstruction algorithm.

**Outlier lap filter (both modes):** Any lap more than 3 seconds slower than the median is removed. This catches laps affected by traffic, spins, mechanical issues, or a red flag:

```python
med = np.median(times)
keep = [(i, t) for i, t in zip(idx, times) if t <= med + 3.0]
```

The 3-second threshold is intentionally generous — it removes genuine outliers while retaining laps that are slightly slow due to a minor mistake or traffic.

### 4. GPS beacon reconstruction — lateral gate and speed floor

When no beacon data is available, `beacons_from_gps()` detects start/finish line crossings from the GPS track. Two filters prevent false detections:

**Lateral gate (`gate_m = 8.0 m`):** A crossing is only counted if the kart passes within 8 metres laterally of the S/F reference point. Sign changes in the along-track coordinate that happen away from the physical line (e.g. the kart going around a corner that happens to cross the projected line) are rejected:

```python
if s[k-1] < 0 <= s[k] and abs(perp[k]) < gate_m and spd[k] > speed_floor:
```

**Speed floor (`speed_floor = 20.0 km/h`):** Crossings at low speed are ignored. This prevents the kart being pushed or crawling through the pits from triggering a false lap boundary.

**Minimum lap duration (`min_lap_s = 5.0 s`):** After crossing detection, consecutive detections within 5 seconds of each other are deduplicated. This handles cases where the GPS position oscillates around the S/F line at low sample rates:

```python
for c in crossings:
    if not clean or c - clean[-1] > min_lap_s:
        clean.append(c)
```

### 5. Zone detection — minimum duration filter

`braking_zones()` and `acceleration_zones()` use `_detect_zones()`, which applies a minimum duration threshold before recording an event:

```python
min_samples = max(1, int(min_dur_s * sess.rate))
```

- Braking events: default `min_dur_s = 0.1 s` → at 20 Hz, at least 2 consecutive samples below the deceleration threshold
- Throttle-on events: default `min_dur_s = 0.1 s` — same logic

Events shorter than this are sensor noise or momentary weight transfer, not genuine braking or throttle application zones. The dashboard additionally filters displayed markers to braking events ≥ 0.25 s and throttle events ≥ 0.5 s, which removes all but the deliberate corner-related events.

Only samples within flying laps (`lap >= 0`) are considered — out-laps and in-laps are excluded by the `lap >= 0` condition (the out-lap is tagged `lap = -1` by `_annotate_laps`).

### 6. Rotation event detection — radius bounds

`rotation_events()` computes expected yaw rate as `speed / radius` and flags samples where the actual gyro reading exceeds the expected by more than 80°/s. To avoid division errors and spurious results from near-straight sections, it pre-filters to samples with a valid, meaningful turn radius:

```python
(sess.df["GPS Radius"] > 0) & (sess.df["GPS Radius"] < 500)
```

`GPS Radius = 0` is a sentinel for a straight or undefined radius. `GPS Radius > 500 m` is effectively a straight for a kart circuit and the yaw-rate mismatch on a straight is meaningless.

### 7. Gearing regression — speed and RPM floors

`rpm_per_kmh()` fits a linear regression of RPM vs GPS Speed. It excludes:

- Samples below 40 km/h (`speed_floor = 40`) — low speeds mix gear-change transients and corner exits where the relationship between RPM and speed is non-linear
- Samples with `RPM = 0` — engine off or logger artefact
- Requires at least 50 valid samples before running the regression — prevents a meaningless fit on very short or heavily filtered sessions

### 8. Sector timing — per-lap normalisation

`sector_table()` computes sector boundaries as equal fractions of **each lap's own total distance**, not a global fixed distance:

```python
L = d[-1]                          # this lap's total distance
edges = np.linspace(0, L, n_sectors + 1)
```

GPS odometry accumulates small errors each lap, so a lap that records 410 m will have slightly different sector boundaries than one recording 412 m. Normalising per-lap prevents these cumulative drift errors from causing sector-boundary misalignment between laps, which would otherwise make the first and last sector times inconsistent even on identical driven lines.

---

## Analysis Engine

`kart_telemetry.py` is the pure analysis layer. It has no dependencies on the dashboard and can be run directly:

```bash
python kart_telemetry.py Jamie.csv            # solo report
python kart_telemetry.py Jamie.csv Joshua.csv # head-to-head
python kart_telemetry.py Jamie.csv --plots    # also save PNG analysis figures
```

### Data flow

```
load_session(path)
  → parse metadata block (Racer, Rate, Beacon Markers, Segment Times)
  → read channel data, coerce to numeric, drop units row
  → beacons_from_gps() if no beacon data or --no-beacon
  → _annotate_laps()   tag every sample with lap index and lap_dist
  → Session dataclass  immutable snapshot passed to all stage functions
```

### Stage functions

| Stage | Function | Output |
|-------|----------|--------|
| 1 | `flying_laps(sess)` | `(idx, times)` — list of flying lap indices and their times |
| 2 | `consistency_stats(sess)` | Dict: best / mean / median / worst / std / range |
| 3 | `sector_table(sess, n)` | DataFrame — rows=laps, columns=S1…Sn, values=sector times (s) |
| 4 | `top_speed(sess)` | Peak GPS Speed across flying laps (km/h) |
| 4 | `rpm_per_kmh(sess)` | Gearing ratio: km/h per 1000 RPM |
| 5 | `braking_zones(sess)` | List of dicts: `{lap, dist, end_dist, peak, duration, entry_speed}` |
| 5 | `acceleration_zones(sess)` | Same shape as braking, throttle-on events |
| 5 | `rotation_events(sess)` | `{lap, dist, peak_oversteer, duration}` — oversteer/snap events |
| 5 | `corner_speed_profile(sess, n)` | DataFrame: `Sx_entry`, `Sx_apex`, `Sx_exit` speeds per lap |
| 5 | `gg_stats(sess)` | Peak lateral/longitudinal/combined G envelope |

`sector_table` computes sector time using wall-clock timestamps interpolated at the sector boundary distances. This makes sector times independent of GPS speed accuracy — the time between two positions is always correct even if the instantaneous speed reading has noise.

`braking_zones` and `acceleration_zones` detect sustained events where `GPS LonAcc` crosses a threshold (−0.3 g for braking, +0.15 g for throttle) for at least `min_dur_s` seconds. Peak deceleration/acceleration and entry speed are recorded per event.

`rotation_events` computes the expected yaw rate from `GPS Speed / GPS Radius` and compares it to the actual `GPS Gyro` reading. A large positive difference (actual > expected) indicates oversteer or snap rotation beyond what the cornering line requires.

---

## Dashboard

`dashboard.py` imports `kart_telemetry` and builds a single HTML file using Plotly for all charts. Plotly's JavaScript bundle (~3 MB) is embedded inline on first use; subsequent charts reuse it, keeping total file size reasonable.

The visual theme uses Courier New throughout, a dark background (`#07080D`), and a warm amber accent (`#D4A843`). A light theme is also available — see [Dark / Light Mode](#dark--light-mode).

### Solo Driver Tab

One tab per driver. Contains:

#### Track Map

Satellite-backed GPS race-line for the best lap, with:

- **Speed heatmap** — the best lap drawn as a scatter of dots coloured by GPS Speed using a Red→Yellow→Green colorscale. Red = slow (corners), green = fast (straights).
- **Braking markers** — red circles at the start of each braking event on the best lap (≥ 0.25 s duration). Hover for peak deceleration in g.
- **Throttle-on markers** — green circles at the point where full throttle is applied after each corner (≥ 0.5 s duration). Hover for peak acceleration in g.
- **Sector boundary markers** — white dots labelled S1–S(n−1) marking where each sector ends. The label convention is "you crossed the S1 line" = sector 1 complete.
- **Background outlines** — all flying laps in dim grey behind the best lap, showing the range of driven lines.

#### Speed Trace + Delta

Two-panel chart (shared distance axis):

- **Top** — GPS Speed vs distance. All laps in transparent grey; the best lap in the driver's accent colour.
- **Bottom** — Cumulative time delta of every lap vs the best lap. Positive = slower. Area above zero shaded red, below zero shaded green. Each lap gets a distinct colour.

#### Sector Breakdown

Per-sector tabs (S1…Sn). Each tab contains:

**Stats bar** — Mean, Std, Best, Worst, Range across all flying laps for that sector.

**Sector race-line map** — a zoomed-in satellite view covering only that sector (plus 25% buffer each side for entry/exit context):

- All laps as dim grey lines
- Best lap as speed-gradient scatter (or solid amber line — see below)

**Lap selection buttons** — multi-select toggle per lap. Each selected lap is highlighted in a unique colour; unselected laps dim to near-invisible. The **All** button resets.

**Best lap button** — the gold ★ button is toggleable. Click it to hide/show the best lap, useful when comparing two non-best laps directly.

**Speed gradient switch** — toggles the best lap between the speed-heatmap dot cloud (on) and a solid amber line (off). Use "off" when comparing racing lines without colour distraction.

**Lap table** — each lap's sector time, delta vs best, delta vs mean.

**Summary table** — all-lap statistics with a consistency rating: Consistent (std < 0.08 s), Variable (< 0.25 s), or Scattered (≥ 0.25 s).

### Theoretical Best Lap

Located below the Sector Breakdown. Answers: *if you combined the best sector time from each lap into one run, what would the time be?*

**KPI banner:**
- **Theoretical Best** — sum of each sector's minimum time across all laps
- **Actual Best** — fastest complete lap actually driven
- **Potential Gain** — the gap between the two (time theoretically still available)

**Composite race-line map** — each sector drawn in a distinct colour from the lap that was fastest in that sector. All other laps shown as dim grey outlines. Sector colours match the breakdown table. Hover for lap source and speed.

Note: the composite trace may have small visual discontinuities at sector boundaries because consecutive sectors can come from different laps with slightly different lines through the transition. This is expected — the theoretical best is a timing construct, not a physically continuous trajectory.

**Breakdown table** — per sector: source lap, best sector time, mean sector time, saving vs mean.

### Combined / Head-to-Head Tab

Only present when multiple CSV files are provided:

- **Speed Trace + Delta** — best lap speed overlaid for all drivers with cumulative delta curves, showing exactly where on track each driver gains or loses time relative to the fastest driver
- **Track Map** — all drivers' best laps on one satellite map in distinct colours, toggleable via legend
- **Apex Speed Table** — entry/apex/exit speed per sector averaged across all laps, ranked by inter-driver gap to surface the highest-priority coaching areas
- **Lap Times** — grouped bar chart with mean dashed line per driver
- **G-G Diagram** — lateral vs longitudinal G per driver, coloured by speed
- **Head-to-Head Table** — best lap, mean lap, std, top speed, gearing side by side with deltas

---

## Satellite Imagery

Maps use ESRI World Imagery, a free public tile service:

```
https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
```

Tiles are downloaded **once** at build time, stitched into a single JPEG per map view, and base64-embedded in the HTML. The file then works fully offline.

**Zoom selection** — `_best_tile_zoom()` iterates from zoom 19 downward, picking the highest level where the GPS bounding box fits within the tile budget (64 tiles for full-track views, 25 tiles for sector zooms). At zoom 19 and Singapore's latitude (~1.4°), one pixel ≈ 0.3 m.

**Tile cache** — `_TILE_CACHE` is an in-memory dict keyed by `(zoom, ty, tx)`. Sector maps cover subsets of the same area as the full-track view, so tiles are reused without re-fetching.

**Coordinate conversion** — GPS lat/lon are converted to pixel coordinates using the Web Mercator projection:

```
global_pixel_x = (lon + 180) / 360 × 2^zoom × 256
global_pixel_y = (1 − log(tan(lat) + sec(lat)) / π) / 2 × 2^zoom × 256
```

Subtracting the top-left tile's pixel origin gives coordinates in the stitched image space, used directly as x/y values for Plotly Scatter traces. The y-axis is inverted (`range=[H, 0]`) so north is up, with `scaleanchor="x"` enforcing a 1:1 aspect ratio.

---

## GPS Registration Offset

Consumer GPS units have a systematic position error of a few metres. Combined with georeferencing uncertainty in the satellite imagery, the GPS trace may appear displaced from the actual track surface. Two correction methods are available.

### Manual correction

```bash
python dashboard.py Jamie.csv --lat-offset 0.000135 --lon-offset -0.000090
```

To measure the offset:
1. Open the dashboard in a browser and zoom into a tight, recognisable corner.
2. Hover over the GPS trace — the Plotly toolbar shows the current lat/lon. Note the value.
3. Open Google Maps, navigate to the same corner in satellite view, right-click the exact point on the tarmac → **"What's here?"**
4. Calculate: `lat_offset = google_lat − plotly_lat`, `lon_offset = google_lon − plotly_lon`

Offsets are in decimal degrees. A 10 m northward correction is approximately `+0.0000898°` latitude.

### Automatic alignment (`--auto-align`)

```bash
python dashboard.py Jamie.csv --auto-align
# for large tracks where zoom 18 exceeds tile budget:
python dashboard.py Jamie.csv --auto-align --align-zoom 17
```

Requires `scipy` and `Pillow`. The algorithm:

1. Downloads satellite tiles covering the GPS bounding box at the specified zoom level
2. Applies Gaussian blur then Sobel edge detection to produce an edge-magnitude image
3. Renders all flying lap GPS traces as thick lines into a matching pixel canvas, then Gaussian-smears them into a density heatmap
4. Computes the 2D FFT cross-correlation between the GPS density and the satellite edges — the peak gives the pixel shift that maximally overlaps the track trace with the satellite edge map
5. Converts the pixel shift to degrees using the Mercator scale factor at the track's latitude: `dlon = dx_px × 360 / (2^zoom × 256)` and `dlat = −dy_px × 360 × cos(lat) / (2^zoom × 256)`
6. Rejects offsets larger than 80 m (likely a false correlation to buildings or roads) and falls back to zero

Auto-align works best when the track surface has clear contrast against its surroundings (dark tarmac vs grass or gravel). If the correlation fails or produces a large offset, it prints a warning and the manual flags remain the reliable fallback.

---

## Dark / Light Mode

A theme toggle button is fixed to the top-right corner of the dashboard, visible regardless of scroll position.

- **Dark mode** (default) — near-black background (`#07080D`), dark card surfaces, warm amber accent (`#D4A843`)
- **Light mode** — warm off-white background (`#F2F0EB`), cream card surfaces, darkened amber accent (`#B8860B`), adjusted text and border colours for accessible contrast

Clicking the toggle also calls `Plotly.relayout` on every chart in the page, updating paper/plot backgrounds and grid colours to match. This keeps charts visually consistent with the surrounding page.

The chosen theme is saved in `localStorage` under the key `kt-theme` and restored on every subsequent page load, so the preference persists across sessions and browser restarts.

---

## CLI Reference

### `dashboard.py`

```
python dashboard.py <files...> [options]
```

| Argument | Default | Description |
|---|---|---|
| `files` | required | One or more AiM CSV files. First file = reference driver in combined tab. |
| `--sectors N` | `6` | Number of equal-distance sectors per lap |
| `--no-beacon` | off | Force GPS-only lap detection even if beacon data is present |
| `--lat-offset F` | `0.0` | Latitude correction in decimal degrees (+ = north) |
| `--lon-offset F` | `0.0` | Longitude correction in decimal degrees (+ = east) |
| `--auto-align` | off | Auto-detect GPS/satellite offset via cross-correlation |
| `--align-zoom N` | `18` | Tile zoom level for auto-align computation |

### `kart_telemetry.py` (standalone)

```
python kart_telemetry.py <files...> [options]
```

| Argument | Default | Description |
|---|---|---|
| `files` | required | One or two AiM CSV files |
| `--sectors N` | `6` | Sector count for sector table |
| `--no-beacon` | off | Force GPS lap detection |
| `--plots` | off | Save matplotlib PNG analysis figures alongside the CSV |

---

## Architecture

```
kart_telemetry.py          Pure analysis engine — no UI dependencies
dashboard.py               Dashboard generator — imports kart_telemetry
  _TILE_CACHE              In-memory tile cache shared across all map builds
  _OFF                     Global GPS offset state set by --lat-offset / --auto-align
  _best_tile_zoom()        Selects highest zoom fitting GPS extent in tile budget
  _sat_snapshot()          Downloads + stitches tiles → base64 JPEG + pixel origin
  _gps2px()                Vectorised GPS lat/lon → image pixel coordinates (Web Mercator)
  _track_fig()             Solo track map (best lap + all laps + markers)
  _combined_track_fig()    Multi-driver best-lap overlay
  _sector_laps_fig()       Per-sector race-line figure (JS-driven multi-select)
  _theo_best_fig()         Composite race-line from per-sector best laps
  _sector_breakdown_html() Full sector analysis section with tabs
  _theo_best_html()        Theoretical best KPI banner + composite map + table
  find_offset()            Auto-align via satellite cross-correlation
  build_dashboard()        Top-level assembler — returns full HTML string
  main()                   CLI entry point
```

All three map figure functions share the same pattern:

1. Compute GPS extent from the relevant data subset
2. `_best_tile_zoom()` → choose zoom level
3. `_sat_snapshot()` → fetch tiles, get `(b64, ox, oy, W, H)`
4. Create `go.Figure`, add JPEG via `add_layout_image`
5. `_gps2px(lats, lons, ox, oy, zoom)` → pixel coordinates
6. Add `go.Scatter` traces in pixel space
7. `xaxis range=[0, W]`, `yaxis range=[H, 0]` (y inverted = north up), `scaleanchor="x"` for 1:1 aspect

The JavaScript in `_JS` manages interactive state for sector maps:

| Variable | Type | Purpose |
|---|---|---|
| `_sLS` | `{chartId: {traceIdx: colorHex}}` | Highlighted lap selections per chart |
| `_sBV` | `{chartId: bool}` | Best lap visibility per chart |
| `_sGV` | `{chartId: bool}` | Speed gradient switch state per chart |

`_redrawSL()` is the single function that translates this state into `Plotly.restyle` calls. Every control (lap button, best-lap button, gradient switch, All/reset) calls `_redrawSL` after updating the relevant state variable, so there is one code path for all visual updates.
