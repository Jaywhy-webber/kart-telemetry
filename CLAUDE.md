# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run solo report
python kart_telemetry.py Jamie.csv

# Head-to-head comparison
python kart_telemetry.py Jamie.csv Joshua.csv

# Custom sector count
python kart_telemetry.py Jamie.csv --sectors 8

# GPS-only lap detection (no beacon data)
python kart_telemetry.py Jamie.csv --no-beacon
```

Dependencies: `pip install pandas numpy`

## Architecture

Single-file script (`kart_telemetry.py`) built in four layered stages:

| Stage | Functions | Output |
|-------|-----------|--------|
| 1 | `times_from_beacons()`, `flying_laps()` | Lap times from beacon markers |
| 2 | `consistency_stats()` | best/mean/median/worst/std/range |
| 3 | `sector_table()` | Equal-distance sector times per lap |
| 4 | `top_speed()`, `rpm_per_kmh()` | Gearing and top speed hooks |

**Data flow:** `load_session()` → `Session` dataclass → stage functions → `report()` / `compare()`

`_annotate_laps()` tags every sample with `lap` index and `lap_dist` (metres from S/F), enabling all distance-based analysis downstream.

`beacons_from_gps()` is the fallback when no beacon data exists: builds a virtual S/F line and interpolates sub-sample crossing times. Validated to within ~50 ms/lap against real beacons.

## AiM CSV format

- Metadata header block (Racer, Sample Rate, Beacon Markers, Segment Times) precedes the channel data
- `Beacon Markers` = timestamps (s) of S/F crossings; lap time = consecutive beacon diff
- First segment = out-lap (slow), last = in-lap (slow) — `flying_laps()` trims these in beacon mode
- Row immediately after the channel-name header is a units row (strings); coercing to numeric + `dropna` removes it
- Filter GPS quality: `GPS Nsat >= 6` before using GPS Speed / Latitude / Longitude

## Verified facts — do not re-derive

- Jamie: best lap 50.135 s, mean 50.68 s, 10 flying laps
- Beacon-diff lap times match the file's own Segment Times exactly
- GPS-only detection agrees with beacons within ~50 ms/lap
- Consistency std uses `ddof=1` (sample std) throughout — keep this consistent

## Planned extensions

- Add Joshua's CSV for head-to-head (`compare()` already implemented)
- Replace equal-distance sectors with real corner-distance boundaries (T1–T7, Kranji)
- Speed-trace and RPM-distribution plots (matplotlib)
- `--sf-latlon` CLI flag for genuinely beacon-less files
- `--save` flag to write results to CSV/txt


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
"""