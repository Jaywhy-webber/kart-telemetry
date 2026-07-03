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
8. [GPS Alignment](#gps-alignment)
   - [Interactive Nudge Pad](#interactive-nudge-pad)
   - [Manual CLI Offset](#manual-cli-offset)
   - [Auto-Align](#auto-align)
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
| scipy     | FFT cross-correlation for `--auto-align` only         |

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

# Bake in a known GPS/satellite offset
python dashboard.py Jamie.csv --lat-offset 0.000135 --lon-offset -0.000090

# Auto-detect GPS/satellite offset
python dashboard.py Jamie.csv --auto-align
```

The output is a single self-contained HTML file (e.g. `Jamie_dashboard.html` or `Jamie_vs_Joshua_Tan_dashboard.html`) written to the same directory as the CSVs. Satellite tile images are embedded as base64 JPEG — the file works fully offline after generation. An internet connection is only required during the build step to download tiles.

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

Multiple cleaning and filtering steps are applied at different stages of the pipeline.

### 1. CSV parse — units row removal

The AiM CSV places a units row (e.g. `"km/h"`, `"rpm"`, `"deg"`) immediately after the channel-name header. `load_session` reads from the header row downward, then calls:

```python
df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Time"])
```

`errors="coerce"` turns every non-numeric cell into `NaN`. `dropna(subset=["Time"])` removes any row with no valid timestamp — this covers the units row and any trailing empty rows.

### 2. GPS quality filter — `GPS Nsat >= 6`

Every function that touches GPS channels first filters:

```python
d = sess.df[sess.df["GPS Nsat"] >= 6]
```

Fewer than 6 satellites locked indicates poor fix quality, producing unreliable position and speed values. All map traces, speed heatmaps, braking/throttle markers, G-G diagrams, and the gearing regression use this filtered subset.

### 3. Lap detection — outlier lap removal

`flying_laps()` applies two filtering steps:

**Out-lap and in-lap trimming (beacon mode only):** The first and last segments are unconditionally removed — they are the warm-up and cool-down laps.

**Outlier lap filter (both modes):** Any lap more than 3 seconds slower than the median is removed — catches traffic, spins, red flags:

```python
med = np.median(times)
keep = [(i, t) for i, t in zip(idx, times) if t <= med + 3.0]
```

### 4. GPS beacon reconstruction — lateral gate and speed floor

When no beacon data is available, `beacons_from_gps()` detects S/F crossings from GPS. Two filters prevent false detections:

**Lateral gate (8 m):** Crossings are only counted if the kart passes within 8 m of the S/F reference point.

**Speed floor (20 km/h):** Crossings at low speed are ignored (kart being pushed, pits).

**Minimum lap duration (5 s):** Consecutive crossings within 5 seconds are deduplicated.

### 5. Zone detection — minimum duration filter

`braking_zones()` and `acceleration_zones()` require at least 0.1 s of sustained event before recording. Dashboard markers are filtered further: braking ≥ 0.25 s, throttle ≥ 0.5 s.

### 6. Rotation event detection — radius bounds

`rotation_events()` pre-filters to `0 < GPS Radius < 500 m` — avoids division errors on straights and spurious results from near-straight sections.

### 7. Gearing regression — speed and RPM floors

`rpm_per_kmh()` excludes samples below 40 km/h and `RPM = 0`, and requires at least 50 valid samples before fitting.

### 8. Sector timing — per-lap normalisation

`sector_table()` computes sector boundaries as equal fractions of **each lap's own total distance**:

```python
L = d[-1]
edges = np.linspace(0, L, n_sectors + 1)
```

GPS odometry drifts per lap. Normalising per-lap prevents cumulative distance errors from causing sector-boundary misalignment across laps.

---

## Analysis Engine

`kart_telemetry.py` is the pure analysis layer with no UI dependencies:

```bash
python kart_telemetry.py Jamie.csv            # solo report
python kart_telemetry.py Jamie.csv Joshua.csv # head-to-head
```

### Data flow

```
load_session(path)
  → parse metadata (Racer, Rate, Beacon Markers, Segment Times)
  → read channel data, coerce to numeric, drop units row
  → beacons_from_gps() if no beacon data or --no-beacon
  → _annotate_laps()   tag every sample with lap index and lap_dist
  → Session dataclass  passed to all stage functions
```

### Stage functions

| Stage | Function | Output |
|-------|----------|--------|
| 1 | `flying_laps(sess)` | `(idx, times)` — flying lap indices and times |
| 2 | `consistency_stats(sess)` | Dict: best / mean / median / worst / std / range |
| 3 | `sector_table(sess, n)` | DataFrame — rows=laps, cols=S1…Sn, values=sector times (s) |
| 4 | `top_speed(sess)` | Peak GPS Speed across flying laps (km/h) |
| 4 | `rpm_per_kmh(sess)` | Gearing ratio: km/h per 1000 RPM |
| 5 | `braking_zones(sess)` | List: `{lap, dist, end_dist, peak, duration, entry_speed}` |
| 5 | `acceleration_zones(sess)` | Same shape, throttle-on events |
| 5 | `rotation_events(sess)` | `{lap, dist, peak_oversteer, duration}` |
| 5 | `corner_speed_profile(sess, n)` | DataFrame: `Sx_entry`, `Sx_apex`, `Sx_exit` per lap |
| 5 | `gg_stats(sess)` | Peak lateral / longitudinal / combined G envelope |

`sector_table` computes timing from wall-clock timestamps interpolated at sector boundary distances — independent of GPS speed accuracy.

`rotation_events` computes expected yaw rate as `speed / radius` and flags samples where actual `GPS Gyro` exceeds that by more than 80°/s.

---

## Dashboard

`dashboard.py` imports `kart_telemetry` and builds a single HTML file. Plotly's JS bundle (~3 MB) is embedded inline on the first figure; subsequent charts reuse it.

**Visual theme:** Titillium Web font, dark background (`#15151E`), F1-inspired red accent (`#E8002D`). A light theme is available via the toggle — see [Dark / Light Mode](#dark--light-mode).

### Solo Driver Tab

One tab per driver. Sections:

#### KPI Row
Best lap / Avg lap / Top speed / Gearing / Flying laps / Spread.

#### Track Map
Satellite-backed GPS race-line with:
- **Speed heatmap** — best lap as dots coloured Red→Yellow→Green by GPS Speed
- **Braking markers** — red circles at start of each braking event (≥ 0.25 s)
- **Throttle-on markers** — green circles at throttle application point (≥ 0.5 s)
- **Sector boundary markers** — white dots labelled S1–S(n−1)
- **Background outlines** — all flying laps in dim grey
- **GPS Align bar** — interactive nudge pad below the map (see [GPS Alignment](#gps-alignment))

#### Delta Chart
All laps plotted as offset from the best lap time. Y=0 is best; other laps show how many seconds off best they were. Colour gradient green→red. Trend line and per-driver mean annotation.

#### Speed Trace
GPS Speed vs lap distance. All laps in transparent grey; best lap in the driver's accent colour.

#### Sector Breakdown
Per-sector tabs S1…Sn. Each tab:
- **Zoomed satellite map** — covers only that sector (25% context buffer each side). All laps as dim grey; best lap as speed-heatmap dots. GPS Align bar below.
- **Stats bar** — mean, std, best, worst, range across flying laps
- **Lap table** — click any row to highlight that lap on the map in a unique colour. Multi-select. Non-selected laps dim out. Click a highlighted row again to deselect.
- **Controls bar** — speed gradient toggle (dots vs solid line), reset button
- **Summary table** — consistency rating per sector: Consistent (std < 0.08 s), Variable (< 0.25 s), Scattered (≥ 0.25 s)

#### Theoretical Best Lap
- **KPI banner** — Theoretical Best / Actual Best / Potential Gain
- **Composite race-line map** — each sector drawn from its personal-best lap in a distinct colour; all other laps as grey outlines. GPS Align bar below.
- **Breakdown table** — source lap, best time, mean time, saving vs mean per sector

#### G-G Diagram
Lateral vs longitudinal G scatter coloured by speed. Shows the driver's grip envelope.

#### RPM vs Speed
Scatter coloured by longitudinal G. Linear fit shows gearing ratio.

#### Stage 5 Table
Braking consistency, acceleration zones, rotation events, corner apex speeds, lap trend.

### Combined / Head-to-Head Tab

Present when multiple CSVs are provided:

- **Speed Trace + Delta** — best lap speed overlaid for all drivers; cumulative delta vs fastest driver
- **Track Map** — all drivers' best laps on one satellite map in distinct colours. GPS Align bar below.
- **Apex Speed Table** — entry/apex/exit speed per sector, ranked by inter-driver gap
- **Lap Times** — delta-from-best scatter per driver with mean lines
- **G-G Diagram** — lateral vs longitudinal G per driver
- **Head-to-Head Table** — best lap, mean, std, top speed, gearing with deltas

---

## Satellite Imagery

Maps use ESRI World Imagery, a free public tile service:

```
https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
```

Tiles are downloaded **once** at build time, stitched into a single JPEG per map view, and base64-embedded in the HTML. The file then works fully offline.

**Zoom selection** — `_best_tile_zoom()` iterates from zoom 19 downward, picking the highest level where the GPS bounding box (plus `pad=1` tile on each side) fits within the tile budget. Budget: 64 tiles for full-track views, 36 tiles for sector zooms. At zoom 19 near Singapore (~1.4°N), one pixel ≈ 0.3 m.

**Padding** — `pad=1` (default) adds one tile on each side of the GPS bounding box. This ensures the track is fully visible even when GPS samples land near a tile boundary or some laps don't reach every corner.

**Tile cache** — `_TILE_CACHE` is an in-memory dict keyed by `(zoom, ty, tx)`. Sector maps re-use tiles fetched for the full-track view without re-downloading.

**Coordinate conversion** — GPS lat/lon → pixel coordinates via Web Mercator:

```
global_px_x = (lon + 180) / 360 × 2^zoom × 256
global_px_y = (1 − log(tan(lat) + sec(lat)) / π) / 2 × 2^zoom × 256
```

Subtracting the top-left tile's pixel origin gives coordinates in the stitched image space, used directly as x/y values for Plotly Scatter traces. The y-axis is inverted (`range=[H, 0]`) so north is up, with `scaleanchor="x"` enforcing a 1:1 aspect ratio.

---

## GPS Alignment

Consumer GPS units have a systematic position error of a few metres. Combined with georeferencing uncertainty in the satellite imagery, the GPS trace may appear displaced from the actual track surface. Three correction methods are available.

### Interactive Nudge Pad

Every satellite map in the dashboard has a **GPS ALIGN** bar directly below it. All align bars across all maps — every driver's track map, all sector maps, all theoretical best maps, and the combined map — share a single universal offset. Adjusting from any bar updates all maps simultaneously.

**Controls:**
- **↑ ↓ ← →** — nudge GPS traces N/S/E/W by the selected step size
- **○** — reset to original position
- **Step size selector** — 0.3 / 0.5 / 1 / 2 / 5 metres per click
- **Readout** — shows accumulated offset in decimal degrees; turns red when non-zero
- **Copy offset** — copies `--lat-offset X.XXXXXX --lon-offset X.XXXXXX` to clipboard

**Technique:** zoom into a tight corner you recognise, switch to 0.3 m steps, and nudge until the GPS trace sits on the tarmac. The readout shows the total correction. Click **Copy offset** and paste the flags into your next build command to bake the alignment in permanently.

**How it works internally:** instead of moving trace data, the align pad shifts the Plotly axis range (viewport) and the layout image origin in opposite directions by the same pixel amount. The GPS traces appear to move while the satellite image stays fixed. Each figure computes its own pixel-per-metre conversion using its own zoom level.

### Manual CLI Offset

```bash
python dashboard.py Jamie.csv --lat-offset 0.000135 --lon-offset -0.000090
```

Offsets are in decimal degrees. A 10 m northward correction ≈ `+0.0000898°` latitude. Positive lat = north, positive lon = east.

To measure:
1. Open the dashboard, zoom into a recognisable corner
2. Hover over the GPS trace — Plotly shows lat/lon
3. Find the same point in Google Maps satellite view, right-click → **"What's here?"**
4. `lat_offset = google_lat − plotly_lat`, `lon_offset = google_lon − plotly_lon`

### Auto-Align

```bash
python dashboard.py Jamie.csv --auto-align
python dashboard.py Jamie.csv --auto-align --align-zoom 17   # for large tracks
```

Requires `scipy`. The algorithm:

1. Downloads satellite tiles at the specified zoom
2. Applies Gaussian blur then Sobel edge detection → edge-magnitude image
3. Renders GPS traces as thick lines into a matching canvas, smears into a density heatmap
4. Computes 2D FFT cross-correlation between GPS density and satellite edges — peak = optimal pixel shift
5. Converts pixel shift to degrees using Mercator scale factor at track latitude
6. Rejects offsets > 80 m (likely false correlation) and falls back to zero

Works best when the track surface has clear contrast against surroundings (dark tarmac vs grass/gravel).

---

## Dark / Light Mode

A theme toggle button is fixed to the top-right corner of the dashboard.

- **Dark mode** (default) — near-black background (`#15151E`), F1 red accent (`#E8002D`)
- **Light mode** — warm off-white background, adjusted text and borders for accessible contrast

The toggle also calls `Plotly.relayout` on every chart to update paper/plot backgrounds and grid colours. The chosen theme is saved in `localStorage` under `kt-theme` and restored on every subsequent page load.

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
| `--auto-align` | off | Auto-detect GPS/satellite offset via cross-correlation (requires scipy) |
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
| `--plots` | off | Save matplotlib PNG analysis figures |

---

## Architecture

```
kart_telemetry.py               Pure analysis engine — no UI dependencies
dashboard.py                    Dashboard generator — imports kart_telemetry
  _TILE_CACHE                   In-memory tile cache shared across all map builds in one run
  _OFF                          Global [lat_offset, lon_offset] set by CLI flags
  _la() / _lo()                 Apply _OFF to raw lat/lon arrays
  _best_tile_zoom()             Highest zoom fitting GPS extent + padding in tile budget
  _sat_snapshot()               Download + stitch ESRI tiles → base64 JPEG + pixel origin
  _gps2px()                     Vectorised GPS lat/lon → pixel coords (Web Mercator)
  _align_controls_html()        GPS nudge pad HTML — data-key / data-group for universal sync
  _track_fig()                  Returns (fig, lat_c, zoom) — solo track map
  _combined_track_fig()         Returns (fig, lat_c, zoom) — multi-driver overlay
  _sector_laps_fig()            Returns (fig, others, bk, idx, times, zoom, lat_c)
  _theo_best_fig()              Returns (fig, zoom, lat_c) — composite best-sector map
  _sector_breakdown_html()      Full sector analysis section with tabs + align bars
  _theo_best_html()             Theoretical best KPI + composite map + table + align bar
  _driver_tab()                 Complete solo driver tab HTML
  _combined_tab()               Complete combined/head-to-head tab HTML
  find_offset()                 Auto-align via satellite cross-correlation (--auto-align)
  build_dashboard()             Computes universal sat group → assembles all tabs → HTML string
  main()                        CLI entry point
```

### Map figure pattern (all three map types follow this)

1. Compute GPS extent from the data subset
2. `_best_tile_zoom()` → choose zoom, accounting for `pad=1`
3. `_sat_snapshot()` → fetch/stitch tiles → `(b64, ox, oy, W, H)`
4. Create `go.Figure`, embed JPEG via `fig.add_layout_image()`
5. `_gps2px(lats, lons, ox, oy, zoom)` → pixel coordinates
6. Add `go.Scatter` traces in pixel space
7. `xaxis.range=[0,W]`, `yaxis.range=[H,0]` (y inverted = north up), `scaleanchor="x"` (1:1 aspect)

### Universal GPS alignment

`build_dashboard()` computes `_universal_group` (comma-separated list of every satellite div ID on the page) and `_universal_key = "all_sat"`. Both are passed to every tab and section builder. Every align bar on the page shares these values, so `_gpsOff["all_sat"]` is the single source of truth for the current offset.

`_applyViewportOne(divId, dLat, dLon)` looks up the figure's own zoom from its align bar (`data-zoom`) and computes `dx, dy` in pixels independently per figure. It then calls `Plotly.relayout` to shift both the axis ranges and the layout image origin — GPS traces appear to move while the satellite stays fixed.
