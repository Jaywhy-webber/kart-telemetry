# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Generate dashboard (single driver)
python dashboard.py Jamie.csv

# Head-to-head comparison
python dashboard.py Jamie.csv Joshua.csv

# Custom sector count (default 6)
python dashboard.py Jamie.csv --sectors 8

# GPS-only lap detection
python dashboard.py Jamie.csv --no-beacon

# Manual GPS/satellite offset correction
python dashboard.py Jamie.csv --lat-offset 0.000135 --lon-offset -0.000090

# Auto-detect offset via cross-correlation (requires scipy)
python dashboard.py Jamie.csv --auto-align

# Run analysis engine directly (no dashboard)
python kart_telemetry.py Jamie.csv
python kart_telemetry.py Jamie.csv Joshua.csv
```

Dependencies: `pip install pandas numpy plotly requests Pillow scipy`

`scipy` is optional — only required for `--auto-align`.

## Architecture

Two files:

- **`kart_telemetry.py`** — pure analysis engine, no UI dependencies. Stages: lap detection → consistency stats → sector timing → top speed/gearing → braking/accel/rotation zones → corner speed profile → G-G stats. `load_session()` → `Session` dataclass → stage functions.
- **`dashboard.py`** — imports `kart_telemetry`, builds a self-contained HTML file with embedded Plotly charts and satellite imagery. No server required.

### dashboard.py key functions

| Function | Purpose |
|---|---|
| `_best_tile_zoom(lats, lons, max_tiles, pad)` | Highest zoom where GPS extent + padding fits in tile budget |
| `_sat_snapshot(lats, lons, zoom, pad)` | Download + stitch ESRI tiles → base64 JPEG + pixel origin. `pad=1` adds one tile buffer on each side |
| `_gps2px(lats, lons, ox, oy, zoom)` | Vectorised lat/lon → pixel coords via Web Mercator |
| `_track_fig(sess, n_sectors, color)` | Returns `(fig, lat_c, zoom)` — solo track map with satellite |
| `_combined_track_fig(sessions, n_sectors)` | Returns `(fig, lat_c, zoom)` — multi-driver overlay |
| `_sector_laps_fig(sector_idx, n_sectors, sess)` | Returns `(fig, others, bk, idx, times, zoom, lat_c)` — sector zoom with all laps |
| `_theo_best_fig(sess, tbl, n_sectors)` | Returns `(fig, zoom, lat_c)` — composite best-sector race line |
| `_align_controls_html(div_id, lat_c, zoom, group, key)` | GPS nudge pad HTML below a satellite map |
| `build_dashboard(sessions, n_sectors)` | Top-level assembler — computes universal sat group, builds all tabs |

### GPS alignment system

All satellite map figures share a **single universal offset** (`data-key="all_sat"`, `data-group` = every satellite div ID on the page). Nudging from any align bar updates every map simultaneously.

JS state: `_gpsOff["all_sat"]` stores `{dLat, dLon}` in degrees. `_applyViewportOne(divId, dLat, dLon)` shifts the Plotly axis range and layout image origin in opposite directions — traces appear to move over a stationary satellite image. Each figure uses its own zoom (read from its align bar's `data-zoom`) for the pixel conversion.

Div ID naming convention: `{racer_slug}_trackmap`, `{racer_slug}_theo_map`, `{racer_slug}_ch_S1..Sn`, `combined_trackmap`.

### JavaScript state variables (`_JS`)

| Variable | Purpose |
|---|---|
| `_sLS` | `{chartId: {traceIdx: colorHex}}` — highlighted lap selections per sector chart |
| `_sBV` | `{chartId: bool}` — best lap visibility per sector chart |
| `_sGV` | `{chartId: bool}` — speed gradient switch state |
| `_gpsOff` | `{key: {dLat, dLon}}` — accumulated GPS alignment offset in degrees |
| `_origLayout` | `{divId: {xr0,xr1,yr0,yr1,ix,iy}}` — saved original axis ranges + image pos for reset |

`_redrawSL()` is the single function that translates lap-selection state into `Plotly.restyle` calls.

## AiM CSV format

- Metadata header block (Racer, Sample Rate, Beacon Markers, Segment Times) precedes channel data
- Row immediately after channel-name headers is a units row — coerce to numeric + `dropna` removes it
- `GPS Nsat >= 6` filter applied before any GPS channel is used
- Beacon-diff lap times match the file's own Segment Times exactly

## Verified facts — do not re-derive

- Jamie: best lap 50.135 s, mean 50.68 s, 10 flying laps
- Consistency std uses `ddof=1` (sample std) throughout
- GPS-only detection agrees with beacons within ~50 ms/lap
- ESRI tile max zoom is 19 — `maxzoom: 19` enforced in tile config
- Satellite JPEG is embedded base64 as a Plotly `layout_image`; GPS traces are `go.Scatter` in pixel space on top


<frontend_aesthetics>
You tend to converge toward generic, "on distribution" outputs. In frontend design, this creates what users call the "AI slop" aesthetic. Avoid this: make creative, distinctive frontends that surprise and delight. Focus on:
 
Typography: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics.
 
Color & Theme: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes. Draw from IDE themes and cultural aesthetics for inspiration.
 
Motion: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions.
 
Backgrounds: Create atmosphere and depth rather than defaulting to solid colors. Layer CSS gradients, use geometric patterns, or add contextual effects that match the overall aesthetic.
 
Avoid generic AI-generated aesthetics:
- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white backgrounds)
- Predictable layouts and component patterns
- Cookie-cutter design that lacks context-specific character
 
Interpret creatively and make unexpected choices that feel genuinely designed for the context. Vary between light and dark themes, different fonts, different aesthetics. You still tend to converge on common choices (Space Grotesk, for example) across generations. Avoid this: it is critical that you think outside the box!
</frontend_aesthetics>
