# Security Audit Report — Kart Telemetry App

**Date:** 2026-06-08  
**Scope:** `kart_telemetry.py`, `dashboard.py`  
**Auditor:** Claude Code (automated)

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH     | 0 |
| MEDIUM   | 3 |
| LOW      | 3 |

No secrets, hardcoded credentials, shell injection, `eval`, `exec`, `subprocess`, or `pickle` usage was found.  
No known CVEs detected in installed dependencies (`pandas 3.0.3`, `numpy 2.4.6`, `plotly 6.7.0`, `requests 2.32.3`, `Pillow 12.2.0`).

---

## Findings

---

### [MEDIUM-1] XSS via unescaped `Racer` field in HTML output

**File:** `dashboard.py` — lines 482, 557, 621, 633, 646, 652, 704, 2240–2244, 2250  
**Source:** `kart_telemetry.py:70` — `meta["racer"] = row[1]`

**What was found:**  
`sess.racer` is read verbatim from the CSV `Racer` row and injected into the HTML output without HTML-escaping in dozens of places — chart titles, hover templates, table cells, and the page `<h1>`. If a CSV file with a crafted `Racer` value such as `<script>alert(document.cookie)</script>` is passed to `dashboard.py`, the generated HTML will execute that script when opened in a browser.

**Impact:**  
Low in solo use, but dashboards are designed to be shared (coaches, teammates). A maliciously crafted CSV delivered to a user produces an HTML file that runs arbitrary JS in the recipient's browser.

**Recommended fix:**  
Escape all user-supplied strings before embedding in HTML. Add a helper and use it wherever `sess.racer` enters HTML context:

```python
import html as _html

def _h(s: str) -> str:
    return _html.escape(str(s))
```

Replace e.g.:
```python
title=f"{sess.racer} — All Laps vs Personal Best"
```
with:
```python
title=f"{_h(sess.racer)} — All Laps vs Personal Best"
```

Apply to all f-strings that embed `sess.racer`, `ref.racer`, `a.racer`, `b.racer`, racer names in `_build_header`, table cells in `_hth_html`, `_apex_table_html`, and the page `<title>` tag (`dashboard.py:2311`).

---

### [MEDIUM-2] HTML attribute injection via unescaped `racer_slug` in div IDs and data attributes

**File:** `dashboard.py` — lines 270–293, 2106, 2273, 2285

**What was found:**  
`racer_slug` is derived from `sess.racer` with only spaces and dots stripped:
```python
racer_slug = sess.racer.replace(" ", "_").replace(".", "")
```
This slug is then embedded directly into HTML attributes (`data-chart`, `data-group`, `data-key`, `onclick`, `id`):
```python
f'onclick="showTab(\'{tid}\',this)">{label}</button>'
```
A racer name containing `'` or `"` would break out of the attribute and allow attribute or JS injection. For example, `Racer'; alert(1); x='` would corrupt the `onclick` handler.

**Impact:**  
Same vector as MEDIUM-1 — crafted CSV corrupts generated HTML.

**Recommended fix:**  
Sanitise the slug more aggressively before use in attributes. Strip everything except alphanumerics and underscores:

```python
import re
racer_slug = re.sub(r'[^\w]', '_', sess.racer)
```

For the `onclick` string context, additionally HTML-escape single quotes or use `data-*` attributes and read them in JS instead of interpolating directly into inline handlers.

---

### [MEDIUM-3] Unvalidated image data from external tile server parsed by Pillow

**File:** `dashboard.py` — lines 65–68, 179–183

**What was found:**  
Tile images are fetched from the ESRI CDN and passed directly to `Image.open()` without size or format validation:
```python
r = requests.get(url, timeout=10, headers={"User-Agent": "kart-telemetry/1.0"})
_TILE_CACHE[key] = Image.open(_io2.BytesIO(r.content)).convert("RGB")
```
While the ESRI CDN is trusted, if the request is intercepted (e.g. MITM on a public network, DNS spoofing, or the tile URL is swapped to a local/custom server via future CLI flags), a crafted image could trigger Pillow parsing vulnerabilities.

**Impact:**  
Low likelihood given ESRI's CDN reliability, but offline/custom tile setups or compromised network paths could exploit this.

**Recommended fix:**  
Add a size guard and explicit format validation before decoding:

```python
MAX_TILE_BYTES = 2 * 1024 * 1024  # 2 MB per tile
if len(r.content) > MAX_TILE_BYTES:
    _TILE_CACHE[key] = None
else:
    img = Image.open(_io2.BytesIO(r.content))
    if img.format not in ("JPEG", "PNG", "WEBP"):
        _TILE_CACHE[key] = None
    else:
        _TILE_CACHE[key] = img.convert("RGB")
```

---

### [LOW-1] Temporary debug file committed to git

**File:** `App/_tmp_matches.txt`  
**Severity:** LOW

**What was found:**  
A scratch file (`_tmp_matches.txt`) is tracked in the git repository. It contains CSS/JS string excerpts that appear to be search output from a prior development session.

**Impact:**  
No credentials or sensitive data detected in this file, but leaking internal development artifacts is poor hygiene. The `.gitignore` does not cover `_tmp_matches.txt`.

**Recommended fix:**  
Delete the file and add the pattern to `.gitignore`:
```
_tmp_*.txt
```

---

### [LOW-2] No bounds check on tile count before downloading

**File:** `dashboard.py` — lines 42–53, 88–99

**What was found:**  
`_best_tile_zoom()` enforces a `max_tiles=64` budget for the main track map, but `find_offset()` (auto-align) uses its own tiling loop with a separate `W > 5120 or H > 5120` pixel check (line 164). A GPS trace covering an unusually large area could trigger a large tile download before the check triggers — up to ~100+ tiles at the default zoom=18. Each tile is a network request.

**Impact:**  
Excessive network requests / slow builds. Not a security vulnerability but a denial-of-service risk on large GPS datasets.

**Recommended fix:**  
Add an explicit tile count cap at the start of `find_offset()`:
```python
n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
if n_tiles > 200:
    print(f"[auto-align] Too many tiles ({n_tiles}). Try a higher --align-zoom value.")
    return 0.0, 0.0
```

---

### [LOW-3] Optional dependency `scipy` not installed; `--auto-align` will fail silently-ish

**File:** `dashboard.py` — lines 123–132  
**Severity:** LOW / Informational

**What was found:**  
`scipy` is listed as optional but is not installed in the current environment. The `find_offset()` function catches the `ImportError` and returns `(0.0, 0.0)` with a printed message. However, this means `--auto-align` silently produces no offset with only a console message — no exception, no non-zero exit code.

**Impact:**  
Users who rely on `--auto-align` may not notice the fallback if stdout is not monitored.

**Recommended fix:**  
Either install `scipy` (`pip install scipy`) or raise a more visible warning and exit non-zero when `--auto-align` is requested but the dependency is missing:

```python
if args.auto_align:
    try:
        import scipy  # noqa: F401
    except ImportError:
        sys.exit("[error] --auto-align requires scipy: pip install scipy")
    dlat, dlon = find_offset(sessions[0], zoom=args.align_zoom)
```

---

## Dependency Versions

| Package  | Version | Status |
|----------|---------|--------|
| pandas   | 3.0.3   | OK — no known CVEs |
| numpy    | 2.4.6   | OK — no known CVEs |
| plotly   | 6.7.0   | OK — no known CVEs |
| requests | 2.32.3  | OK — no known CVEs |
| Pillow   | 12.2.0  | OK — no known CVEs |
| scipy    | not installed | Optional; needed for `--auto-align` |

---

## Patterns Not Found (Confirmed Clean)

- No hardcoded tokens, API keys, passwords, or secrets
- No `eval()`, `exec()`, `__import__()`, or `compile()` usage
- No `subprocess`, `os.system()`, or shell injection paths
- No `pickle` / `marshal` deserialization
- No `verify=False` on HTTPS requests (requests defaults to `verify=True`)
- No SQL queries of any kind
- No `input()` calls in production paths
- No debug backdoors or intentional bypasses
