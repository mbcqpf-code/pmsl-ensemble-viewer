import os
import json
import warnings
import tempfile
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy.ndimage import minimum_filter, maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from herbie import Herbie

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION & DOMAIN SETUP
# ==============================================================================
OUTPUT_DIR = "pmsl_output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
SPREAD_DIR = os.path.join(OUTPUT_DIR, "spread")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPREAD_DIR, exist_ok=True)

# Forecast leads: 0 to 240 hours every 12 hours
FXX_STEPS = list(range(0, 241, 12))

# Standard meteorological MSLP isobars (4 hPa intervals)
PMSL_CONTOURS = list(range(960, 1052, 4))

CANVAS_W, CANVAS_H = 1200, 800

http_session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)
http_session.headers.update({"User-Agent": "Mozilla/5.0 (PMSL-Viewer; Operational Met Research)"})

# 0.5-degree synoptic CONUS grid
lon_min, lon_max = -130.0, -60.0
lat_min, lat_max = 20.0, 55.0
grid_lats = np.arange(lat_min, lat_max + 0.5, 0.5)
grid_lons = np.arange(lon_min, lon_max + 0.5, 0.5)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)


# ==============================================================================
# 2. CYCLE DETECTION & BASEMAP INITIALIZATION
# ==============================================================================
def detect_latest_complete_cycle():
    now = datetime.utcnow()
    latest_hour = 12 if now.hour >= 12 else 0
    anchor = now.replace(hour=latest_hour, minute=0, second=0, microsecond=0)
    candidates = [anchor - timedelta(hours=12 * i) for i in range(4)]
    
    print("🔍 Probing for newest complete 240h ensemble cycle...")
    for cand in candidates:
        cycle_str = cand.strftime('%Y-%m-%d %H:00')
        print(f"  • Checking cycle: {cycle_str}Z...")
        try:
            H_ifs = Herbie(cycle_str, model="ifs", product="enfo", fxx=240)
            if H_ifs.grib is not None:
                print(f"🎯 Target Cycle Verified: {cycle_str}Z\n")
                return cycle_str
        except Exception:
            continue
    return candidates[1].strftime('%Y-%m-%d %H:00')

global_init_time = detect_latest_complete_cycle()

print("🗺️ Rendering Cartopy basemap...")
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="#f8f9fa")
ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
ax.add_feature(cfeature.COASTLINE, linewidth=1.1, edgecolor="#333333")
ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor="#333333")
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#888888", linestyle=":")
ax.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#888888", linewidth=0.4)

fig.savefig(os.path.join(OUTPUT_DIR, "basemap.png"), dpi=100)
geo_trans = ccrs.PlateCarree()._as_mpl_transform(ax)


# ==============================================================================
# 3. GRID STANDARDIZATION & VECTOR EXTRACTION
# ==============================================================================
def standardize_pmsl_grid(ds):
    """Interpolates to standard grid and converts Pascals to hPa."""
    if ds is None:
        return None
    if isinstance(ds, list):
        if len(ds) == 0:
            return None
        ds = ds[0]
    try:
        ds = ds.squeeze()
        var_key = list(ds.data_vars)[0]
        da = ds[var_key]
        if da.longitude.max() > 180:
            da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
        da = da.sortby(['latitude', 'longitude'])
        da_interp = da.interp(latitude=grid_lats, longitude=grid_lons, method='linear')
        arr = da_interp.values
        # GRIB sea level pressure is typically stored in Pa (~101325 Pa)
        if np.nanmean(arr) > 2000:
            arr = arr / 100.0
        return arr
    except Exception:
        return None

def extract_contours(grid):
    """Extracts vector isobars transformed to screen pixel coordinates."""
    if grid is None or np.all(np.isnan(grid)):
        return {}
    try:
        cs = ax.contour(target_lons, target_lats, grid, levels=PMSL_CONTOURS, transform=ccrs.PlateCarree())
        result = {}
        for level, segs in zip(cs.levels, cs.allsegs):
            if not segs:
                continue
            lvl_key = str(int(level))
            lvl_lines = []
            for s in segs:
                if len(s) < 2:
                    continue
                disp_coords = geo_trans.transform(s)
                fig_coords = fig.transFigure.inverted().transform(disp_coords)
                px = np.round(fig_coords[:, 0] * CANVAS_W).astype(int)
                py = np.round((1.0 - fig_coords[:, 1]) * CANVAS_H).astype(int)
                pts = [[int(x), int(y)] for x, y in zip(px, py) if -50 <= x <= CANVAS_W + 50 and -50 <= y <= CANVAS_H + 50]
                if len(pts) > 1:
                    lvl_lines.append(pts)
            if lvl_lines:
                result[lvl_key] = lvl_lines

        if hasattr(cs, 'remove'):
            cs.remove()
        elif hasattr(cs, 'collections'):
            for coll in cs.collections:
                coll.remove()
        return result
    except Exception:
        return {}


# ==============================================================================
# 4. EXTREMA DETECTION & OBJECT-BASED MEDIAN CLUSTERING
# ==============================================================================
def detect_member_centers(grid):
    """
    Identifies local Lows and Highs using spatial neighborhood filters.
    Returns pixel coordinates and 2-digit pressure labels matching Tropical Tidbits.
    """
    if grid is None or np.all(np.isnan(grid)):
        return {"lows": [], "highs": []}
    
    # 9x9 neighborhood (~4.5 deg x 4.5 deg ~ 450 km)
    window = 9
    min_flt = minimum_filter(grid, size=window, mode='nearest')
    max_flt = maximum_filter(grid, size=window, mode='nearest')
    
    low_mask = (grid == min_flt) & (grid < 1020.0)
    high_mask = (grid == max_flt) & (grid > 1012.0)
    
    # Exclude boundary margins (5 grid cells)
    low_mask[:5, :] = low_mask[-5:, :] = low_mask[:, :5] = low_mask[:, -5:] = False
    high_mask[:5, :] = high_mask[-5:, :] = high_mask[:, :5] = high_mask[:, -5:] = False
    
    lows = []
    highs = []
    
    for r, c in zip(*np.where(low_mask)):
        lat, lon = grid_lats[r], grid_lons[c]
        disp_pt = geo_trans.transform(np.array([[lon, lat]]))
        fig_pt = fig.transFigure.inverted().transform(disp_pt)[0]
        px = int(round(fig_pt[0] * CANVAS_W))
        py = int(round((1.0 - fig_pt[1]) * CANVAS_H))
        
        if 0 <= px <= CANVAS_W and 0 <= py <= CANVAS_H:
            p_val = float(grid[r, c])
            lbl = f"{int(round(p_val)) % 100:02d}"
            lows.append({"x": px, "y": py, "val": round(p_val, 1), "lbl": lbl, "lat": float(lat), "lon": float(lon)})
            
    for r, c in zip(*np.where(high_mask)):
        lat, lon = grid_lats[r], grid_lons[c]
        disp_pt = geo_trans.transform(np.array([[lon, lat]]))
        fig_pt = fig.transFigure.inverted().transform(disp_pt)[0]
        px = int(round(fig_pt[0] * CANVAS_W))
        py = int(round((1.0 - fig_pt[1]) * CANVAS_H))
        
        if 0 <= px <= CANVAS_W and 0 <= py <= CANVAS_H:
            p_val = float(grid[r, c])
            lbl = f"{int(round(p_val)) % 100:02d}"
            highs.append({"x": px, "y": py, "val": round(p_val, 1), "lbl": lbl, "lat": float(lat), "lon": float(lon)})
            
    return {"lows": lows, "highs": highs}

def cluster_extrema_medians(member_extrema_list, total_members, center_type="lows", radius_km=450.0):
    """
    Clusters individual member centers using spatial graph grouping.
    Calculates median central pressure, coordinates, and member support percentage.
    """
    all_centers = []
    for m_idx, centers in enumerate(member_extrema_list):
        for pt in centers.get(center_type, []):
            all_centers.append({**pt, "member": m_idx})
            
    if not all_centers:
        return []
        
    n = len(all_centers)
    visited = [False] * n
    clusters = []
    
    def approx_dist_km(lat1, lon1, lat2, lon2):
        dlat = (lat2 - lat1) * 111.0
        dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians(0.5 * (lat1 + lat2)))
        return np.sqrt(dlat**2 + dlon**2)
    
    for i in range(n):
        if visited[i]:
            continue
        cluster = [all_centers[i]]
        visited[i] = True
        
        # Spatial breadth-first expansion
        queue = [all_centers[i]]
        while queue:
            curr = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    dist = approx_dist_km(curr["lat"], curr["lon"], all_centers[j]["lat"], all_centers[j]["lon"])
                    if dist <= radius_km:
                        visited[j] = True
                        cluster.append(all_centers[j])
                        queue.append(all_centers[j])
                        
        # Ensure distinct members per cluster
        member_ids = set(item["member"] for item in cluster)
        support_count = len(member_ids)
        support_pct = (support_count / total_members) * 100.0 if total_members > 0 else 0
        
        # Aggregate stats
        pressures = [item["val"] for item in cluster]
        med_p = float(np.median(pressures))
        med_lat = float(np.median([item["lat"] for item in cluster]))
        med_lon = float(np.median([item["lon"] for item in cluster]))
        
        disp_pt = geo_trans.transform(np.array([[med_lon, med_lat]]))
        fig_pt = fig.transFigure.inverted().transform(disp_pt)[0]
        px = int(round(fig_pt[0] * CANVAS_W))
        py = int(round((1.0 - fig_pt[1]) * CANVAS_H))
        
        clusters.append({
            "type": "L" if center_type == "lows" else "H",
            "x": px,
            "y": py,
            "lat": round(med_lat, 2),
            "lon": round(med_lon, 2),
            "median_p": round(med_p, 1),
            "lbl": f"{int(round(med_p)) % 100:02d}",
            "p10": round(float(np.percentile(pressures, 10)), 1),
            "p90": round(float(np.percentile(pressures, 90)), 1),
            "min_p": round(float(np.min(pressures)), 1),
            "max_p": round(float(np.max(pressures)), 1),
            "count": support_count,
            "total_members": total_members,
            "pct": round(support_pct, 1)
        })
        
    return clusters


# ==============================================================================
# 5. DATA INGESTION (PMSL WORKERS)
# ==============================================================================
def fetch_gefs_pmsl(fxx):
    members = ["c00"] + [f"p{str(i).zfill(2)}" for i in range(1, 31)]
    def _worker(m):
        for var in [":PRMSL:mean sea level:", ":MSLET:mean sea level:"]:
            try:
                H = Herbie(global_init_time, model="gefs", product="atmos.5", member=m, fxx=fxx)
                grid = standardize_pmsl_grid(H.xarray(var))
                if grid is not None and grid.ndim == 2:
                    return grid
            except Exception:
                continue
        return None
    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_worker, members):
            if res is not None:
                grids.append(res)
    return grids

def fetch_aigefs_pmsl(fxx):
    members = [f"mem{str(i).zfill(3)}" for i in range(31)]
    def _worker(m):
        for var in [":PRMSL:mean sea level:", ":PRES:mean sea level:"]:
            try:
                H = Herbie(global_init_time, model="aigefs", product="pres", member=m, fxx=fxx)
                grid = standardize_pmsl_grid(H.xarray(var))
                if grid is not None and grid.ndim == 2:
                    return grid
            except Exception:
                continue
        return None
    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_worker, members):
            if res is not None:
                grids.append(res)
    return grids

def fetch_ifs_pmsl(fxx):
    grids = []
    try:
        H = Herbie(global_init_time, model="ifs", product="enfo", fxx=fxx)
        ds_list = H.xarray(":msl:")
        if not isinstance(ds_list, list):
            ds_list = [ds_list]
        for ds in ds_list:
            g = standardize_pmsl_grid(ds)
            if g is not None:
                if g.ndim == 3:
                    for i in range(g.shape[0]):
                        grids.append(g[i])
                elif g.ndim == 2:
                    grids.append(g)
    except Exception:
        pass
    return grids

def fetch_aifs_pmsl(fxx):
    grids = []
    try:
        H = Herbie(global_init_time, model="aifs", product="enfo", fxx=fxx)
        for data_type in ["pf", "cf"]:
            try:
                ds_list = H.xarray(":msl:", backend_kwargs={"filter_by_keys": {"dataType": data_type}})
                if not isinstance(ds_list, list):
                    ds_list = [ds_list]
                for ds in ds_list:
                    var_key = list(ds.data_vars)[0]
                    da = ds[var_key].squeeze()
                    dim_names = list(da.dims)
                    if any(d in dim_names for d in ['number', 'realization']):
                        m_dim = 'number' if 'number' in dim_names else 'realization'
                        for m_idx in range(len(da[m_dim])):
                            g = standardize_pmsl_grid(da.isel({m_dim: m_idx}).to_dataset(name=var_key))
                            if g is not None and g.ndim == 2:
                                grids.append(g)
                    else:
                        g = standardize_pmsl_grid(da.to_dataset(name=var_key))
                        if g is not None and g.ndim == 2:
                            grids.append(g)
            except Exception:
                continue
    except Exception:
        pass
    return grids

def fetch_det_pmsl(model_name, product, search_str, fxx):
    try:
        H = Herbie(global_init_time, model=model_name, product=product, fxx=fxx)
        ds = H.xarray(search_str)
        if isinstance(ds, list):
            ds = ds[0]
        g = standardize_pmsl_grid(ds)
        if g is not None and g.ndim == 2:
            return g
    except Exception:
        pass
    return None


# ==============================================================================
# 6. SPREAD ENGINE (MSLP STANDARD DEVIATION)
# ==============================================================================
PMSL_SPREAD_LEVELS = np.arange(1.0, 13.0, 1.0)  # 1 hPa to 12 hPa
SPREAD_CMAP = plt.cm.YlOrRd

def compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems):
    def calc_spread(mem_list):
        return np.std(np.stack(mem_list), axis=0, ddof=1) if len(mem_list) > 1 else None

    spr_gefs = calc_spread(gefs_mems)
    spr_aigefs = calc_spread(aigefs_mems)
    spr_ifs = calc_spread(ifs_mems)
    spr_aifs = calc_spread(aifs_mems)

    means = [np.mean(np.stack(m), axis=0) for m in [gefs_mems, aigefs_mems, ifs_mems, aifs_mems] if m]
    spr_inter = np.std(np.stack(means), axis=0, ddof=1) if len(means) > 1 else np.zeros_like(target_lats)

    all_members = gefs_mems + aigefs_mems + ifs_mems + aifs_mems
    spr_total = np.std(np.stack(all_members), axis=0, ddof=1) if len(all_members) > 1 else np.zeros_like(target_lats)

    return {
        "total": spr_total,
        "inter": spr_inter,
        "gefs": spr_gefs,
        "aigefs": spr_aigefs,
        "ifs": spr_ifs,
        "aifs": spr_aifs
    }

def render_spread_underlay(spread_grid, filename):
    if spread_grid is None or np.all(np.isnan(spread_grid)):
        return
    fig_spr = plt.figure(figsize=(12, 8), dpi=100)
    ax_spr = fig_spr.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0))
    ax_spr.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax_spr.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax_spr.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    
    ax_spr.contourf(
        target_lons, target_lats, spread_grid,
        levels=PMSL_SPREAD_LEVELS, cmap=SPREAD_CMAP,
        transform=ccrs.PlateCarree(), extend='max', alpha=0.85
    )
    
    ax_spr.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax_spr.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax_spr.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax_spr.add_feature(cfeature.LAKES, facecolor="#eef2f5", edgecolor="#666666", linewidth=0.4)

    fig_spr.savefig(os.path.join(SPREAD_DIR, filename), dpi=100)
    plt.close(fig_spr)


# ==============================================================================
# 7. BATCH EXECUTION LOOP
# ==============================================================================
for fxx in FXX_STEPS:
    valid_time = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M') + timedelta(hours=fxx)
    print(f"\n📦 Processing PMSL Lead F{fxx:03d} (Valid: {valid_time.strftime('%Y-%m-%d %H:%MZ')})...")
    
    # 1. Fetch ensembles
    gefs_mems   = fetch_gefs_pmsl(fxx)
    aigefs_mems = fetch_aigefs_pmsl(fxx)
    ifs_mems    = fetch_ifs_pmsl(fxx)
    aifs_mems   = fetch_aifs_pmsl(fxx)

    all_members = gefs_mems + aigefs_mems + ifs_mems + aifs_mems
    print(f"  ↳ Members loaded: GEFS={len(gefs_mems)}, AIGEFS={len(aigefs_mems)}, IFS={len(ifs_mems)}, AIFS={len(aifs_mems)} (Total: {len(all_members)})")

    # 2. Compute means
    gefs_mean   = np.mean(np.stack(gefs_mems), axis=0) if gefs_mems else None
    aigefs_mean = np.mean(np.stack(aigefs_mems), axis=0) if aigefs_mems else None
    ifs_mean    = np.mean(np.stack(ifs_mems), axis=0) if ifs_mems else None
    aifs_mean   = np.mean(np.stack(aifs_mems), axis=0) if aifs_mems else None
    super_mean  = np.mean(np.stack(all_members), axis=0) if all_members else None

    # 3. Spread underlays
    spread_diags = compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems)
    for spr_key, spr_grid in spread_diags.items():
        if spr_grid is not None:
            render_spread_underlay(spr_grid, f"{spr_key}_f{fxx:03d}.png")

    # 4. Fetch deterministics
    det_gfs  = fetch_det_pmsl("gfs", "pgrb2.0p25", ":PRMSL:mean sea level:", fxx)
    det_ifs  = fetch_det_pmsl("ifs", "oper", ":msl:", fxx)
    det_aifs = fetch_det_pmsl("aifs", "oper", ":msl:", fxx)

    # 5. Detect member Low/High centers
    gefs_extrema   = [detect_member_centers(m) for m in gefs_mems]
    aigefs_extrema = [detect_member_centers(m) for m in aigefs_mems]
    ifs_extrema    = [detect_member_centers(m) for m in ifs_mems]
    aifs_extrema   = [detect_member_centers(m) for m in aifs_mems]
    all_extrema    = gefs_extrema + aigefs_extrema + ifs_extrema + aifs_extrema

    # 6. Object-based median clusters
    total_mem_count = len(all_members)
    median_low_clusters  = cluster_extrema_medians(all_extrema, total_mem_count, center_type="lows")
    median_high_clusters = cluster_extrema_medians(all_extrema, total_mem_count, center_type="highs")

    step_data = {
        "fxx": fxx,
        "init": global_init_time,
        "valid": valid_time.strftime('%Y-%m-%d %H:%MZ'),
        "total_members": total_mem_count,
        "super_mean": extract_contours(super_mean),
        "gefs_mean": extract_contours(gefs_mean),
        "aigefs_mean": extract_contours(aigefs_mean),
        "ifs_mean": extract_contours(ifs_mean),
        "aifs_mean": extract_contours(aifs_mean),
        "gfs_det": extract_contours(det_gfs),
        "ifs_det": extract_contours(det_ifs),
        "aifs_det": extract_contours(det_aifs),
        "gefs_extrema": gefs_extrema,
        "aigefs_extrema": aigefs_extrema,
        "ifs_extrema": ifs_extrema,
        "aifs_extrema": aifs_extrema,
        "median_lows": median_low_clusters,
        "median_highs": median_high_clusters
    }

    with open(os.path.join(DATA_DIR, f"f{fxx:03d}.json"), "w") as f:
        json.dump(step_data, f, separators=(',', ':'))

plt.close(fig)


# ==============================================================================
# 8. DYNAMIC FRONTEND GENERATION
# ==============================================================================
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PMSL Multi-Ensemble & Extrema Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --bg: #0b1120;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; background: var(--bg); color: var(--text); display: flex; flex-direction: column; align-items: center;
    }}
    .header {{ width: 100%; max-width: 1250px; padding: 14px 20px 6px 20px; box-sizing: border-box; text-align: center; }}
    .header h2 {{ margin: 0 0 4px 0; font-size: 1.35rem; }}
    .header .subtitle {{ font-size: 0.85rem; color: var(--text-muted); }}

    .toolbar {{
      width: 100%; max-width: 1250px; background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 10px; padding: 10px 16px; margin: 8px 0; box-sizing: border-box;
      display: flex; flex-direction: column; gap: 10px;
    }}
    .row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 12px; justify-content: space-between; }}
    .btn {{
      background: var(--accent); color: white; border: none; padding: 5px 12px; border-radius: 5px;
      cursor: pointer; font-size: 12px; font-weight: 600;
    }}
    .btn:hover {{ filter: brightness(1.15); }}
    .btn-secondary {{ background: #475569; }}
    .btn-micro {{ background: #334155; padding: 2px 6px; font-size: 11px; border-radius: 4px; }}
    select {{
      padding: 5px 9px; border-radius: 6px; background: #0f172a; color: white;
      border: 1px solid var(--border); font-size: 13px;
    }}
    
    .panel-box {{
      border-top: 1px solid var(--border); padding-top: 8px; display: flex; flex-direction: column; gap: 8px;
    }}
    .group-line {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-size: 13px; }}
    .group-label {{ font-weight: bold; min-width: 140px; display: flex; align-items: center; gap: 6px; }}
    .checkbox-item {{ display: flex; align-items: center; gap: 5px; cursor: pointer; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}

    .viewer-card {{
      position: relative; width: 100%; max-width: 1200px; aspect-ratio: 1200 / 800;
      background: #000; border-radius: 10px; overflow: hidden; box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }}
    .viewer-card img, .viewer-card canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
    #contourCanvas {{ pointer-events: none; }}
    
    .legend-box {{
      position: absolute; top: 12px; right: 12px; background: rgba(255, 255, 255, 0.94);
      color: #111; padding: 8px 12px; border-radius: 6px; font-size: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 4px;
      pointer-events: none;
    }}
    .title-stamp {{
      position: absolute; top: 12px; left: 12px; background: rgba(255, 255, 255, 0.94);
      color: #111; padding: 6px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;
      pointer-events: none; line-height: 1.3;
    }}
    .spread-colorbar {{ display: none; align-items: center; gap: 8px; font-size: 11px; }}
    .colorbar-gradient {{
      width: 120px; height: 10px; border-radius: 3px;
      background: linear-gradient(to right, #ffffcc, #fed976, #feb24c, #fd8d3c, #f03b20, #bd0026);
      border: 1px solid var(--border);
    }}
  </style>
</head>
<body>

  <div class="header">
    <h2>PMSL Multi-Ensemble & Pressure Extrema Viewer</h2>
    <div class="subtitle">Cycle: {global_init_time}Z | Tropical Tidbits Extrema & Object-Based Medians</div>
  </div>

  <div class="toolbar">
    <!-- Row 1: Lead Scrubbing & Underlay -->
    <div class="row">
      <div style="display: flex; gap: 6px; align-items: center;">
        <button class="btn btn-secondary" onclick="step(-1)">◀ Prev</button>
        <select id="leadSelect" onchange="loadStepData()">
          {"".join([f'<option value="{f:03d}">F{f:03d} (+{f}h)</option>' for f in FXX_STEPS])}
        </select>
        <button class="btn btn-secondary" onclick="step(1)">Next ▶</button>
      </div>

      <div style="display: flex; gap: 8px; align-items: center;">
        <b>Underlay:</b>
        <select id="bgSelect" onchange="updateBackground()">
          <option value="none">Basemap Only</option>
          <option value="total">Total Superensemble Spread (σ)</option>
          <option value="inter">Inter-Model Disagreement (σ_inter)</option>
          <option value="gefs">GEFS Spread</option>
          <option value="aigefs">AIGEFS Spread</option>
          <option value="ifs">IFS Spread</option>
          <option value="aifs">AIFS Spread</option>
        </select>
        <div class="spread-colorbar" id="cbarBox">
          <span>1 hPa</span>
          <div class="colorbar-gradient"></div>
          <span>12+ hPa</span>
        </div>
      </div>
    </div>

    <!-- Row 2: Ensemble Isobars & Deterministics -->
    <div class="panel-box">
      <div class="group-line">
        <span class="group-label">Isobars (Mean):</span>
        <label class="checkbox-item"><input type="checkbox" id="chk_super" checked onchange="draw()"><span class="swatch" style="background:#000;"></span> Super Mean</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_gefs_mean" checked onchange="draw()"><span class="swatch" style="background:#0044ff;"></span> GEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aigefs_mean" checked onchange="draw()"><span class="swatch" style="background:#8e44ad;"></span> AIGEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ifs_mean" checked onchange="draw()"><span class="swatch" style="background:#d63031;"></span> IFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs_mean" checked onchange="draw()"><span class="swatch" style="background:#00b894;"></span> AIFS</label>
        <span style="border-left: 1px solid var(--border); height: 16px; margin: 0 4px;"></span>
        <label class="checkbox-item"><input type="checkbox" id="chk_gfs_det" onchange="draw()"> GFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ifs_det" onchange="draw()"> IFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs_det" onchange="draw()"> AIFS Det</label>
      </div>

      <!-- Row 3: Member Centers (Tropical Tidbits Style) -->
      <div class="group-line">
        <span class="group-label">Member Centers:</span>
        <label class="checkbox-item"><input type="checkbox" id="chk_show_lows" checked onchange="draw()"><span style="color:#e74c3c; font-weight:bold;">Lows (Red)</span></label>
        <label class="checkbox-item"><input type="checkbox" id="chk_show_highs" checked onchange="draw()"><span style="color:#2980b9; font-weight:bold;">Highs (Blue)</span></label>
        <span style="border-left: 1px solid var(--border); height: 16px; margin: 0 4px;"></span>
        <label class="checkbox-item"><input type="checkbox" id="chk_gefs_centers" checked onchange="draw()"> GEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aigefs_centers" checked onchange="draw()"> AIGEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ifs_centers" checked onchange="draw()"> IFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs_centers" checked onchange="draw()"> AIFS</label>
      </div>

      <!-- Row 4: Median Clusters (Anti-Washout Engine) -->
      <div class="group-line" style="background: rgba(59, 130, 246, 0.08); padding: 4px 8px; border-radius: 6px;">
        <span class="group-label">Ensemble Median:</span>
        <label class="checkbox-item"><input type="checkbox" id="chk_median_lows" checked onchange="draw()"> <b>Plot Median Lows</b></label>
        <label class="checkbox-item"><input type="checkbox" id="chk_median_highs" onchange="draw()"> <b>Plot Median Highs</b></label>
        <span style="display: flex; align-items: center; gap: 6px; margin-left: 10px;">
          <span>Min Consensus:</span>
          <select id="minPctSelect" onchange="draw()">
            <option value="15">≥ 15% of members</option>
            <option value="20" selected>≥ 20% of members</option>
            <option value="30">≥ 30% of members</option>
            <option value="50">≥ 50% (High Confidence)</option>
          </select>
        </span>
      </div>
    </div>
  </div>

  <div class="viewer-card">
    <img id="mapUnderlay" src="basemap.png" alt="Basemap">
    <canvas id="contourCanvas" width="1200" height="800"></canvas>
    
    <div class="title-stamp" id="mapTitleStamp">
      PMSL (hPa) & Extrema Centers<br>Loading...
    </div>

    <div class="legend-box" id="legendBox"></div>
  </div>

  <script>
    let dataCache = {{}};
    let currentStepData = null;

    function updateBackground() {{
      const bg = document.getElementById('bgSelect').value;
      const fxx = document.getElementById('leadSelect').value;
      const img = document.getElementById('mapUnderlay');
      const cbar = document.getElementById('cbarBox');

      if (bg === 'none') {{
        img.src = 'basemap.png';
        cbar.style.display = 'none';
      }} else {{
        img.src = `spread/${{bg}}_f${{fxx}}.png`;
        cbar.style.display = 'flex';
      }}
    }}

    async function loadStepData() {{
      const fxx = document.getElementById('leadSelect').value;
      updateBackground();

      if (dataCache[fxx]) {{
        currentStepData = dataCache[fxx];
        draw();
        return;
      }}
      try {{
        const resp = await fetch(`data/f${{fxx}}.json`);
        currentStepData = await resp.json();
        dataCache[fxx] = currentStepData;
        draw();
      }} catch (err) {{
        console.error("Failed loading lead F" + fxx, err);
      }}
    }}

    function step(delta) {{
      const select = document.getElementById('leadSelect');
      let idx = select.selectedIndex + delta;
      if (idx >= 0 && idx < select.options.length) {{
        select.selectedIndex = idx;
        loadStepData();
      }}
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
    }});

    function draw() {{
      if (!currentStepData) return;
      const canvas = document.getElementById('contourCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      document.getElementById('mapTitleStamp').innerHTML = 
        `PMSL (hPa) | Lead: F${{String(currentStepData.fxx).padStart(3, '0')}}<br>` +
        `Init: ${{currentStepData.init}}Z | Valid: ${{currentStepData.valid}} | Members: ${{currentStepData.total_members}}`;

      const legendItems = [];

      // 1. Draw Isobars
      function drawLines(layerDict, color, width, dashed = false) {{
        if (!layerDict) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dashed ? [6, 4] : []);
        ctx.globalAlpha = 1.0;

        for (const [lvl, lines] of Object.entries(layerDict)) {{
          lines.forEach(pts => {{
            if (pts.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
            ctx.stroke();
          }});
        }}
      }}

      // Deterministics
      if (document.getElementById('chk_gfs_det').checked && currentStepData.gfs_det) {{
        drawLines(currentStepData.gfs_det, '#001155', 1.8, true);
        legendItems.push({{ label: 'GFS Det', color: '#001155', dash: true }});
      }}
      if (document.getElementById('chk_ifs_det').checked && currentStepData.ifs_det) {{
        drawLines(currentStepData.ifs_det, '#78281f', 1.8, true);
        legendItems.push({{ label: 'IFS Det', color: '#78281f', dash: true }});
      }}
      if (document.getElementById('chk_aifs_det').checked && currentStepData.aifs_det) {{
        drawLines(currentStepData.aifs_det, '#145a32', 1.8, true);
        legendItems.push({{ label: 'AIFS Det', color: '#145a32', dash: true }});
      }}

      // Ensemble Means
      if (document.getElementById('chk_gefs_mean').checked && currentStepData.gefs_mean) {{
        drawLines(currentStepData.gefs_mean, '#0044ff', 2.2);
        legendItems.push({{ label: 'GEFS Mean', color: '#0044ff', dash: false }});
      }}
      if (document.getElementById('chk_aigefs_mean').checked && currentStepData.aigefs_mean) {{
        drawLines(currentStepData.aigefs_mean, '#8e44ad', 2.2);
        legendItems.push({{ label: 'AIGEFS Mean', color: '#8e44ad', dash: false }});
      }}
      if (document.getElementById('chk_ifs_mean').checked && currentStepData.ifs_mean) {{
        drawLines(currentStepData.ifs_mean, '#d63031', 2.2);
        legendItems.push({{ label: 'IFS Mean', color: '#d63031', dash: false }});
      }}
      if (document.getElementById('chk_aifs_mean').checked && currentStepData.aifs_mean) {{
        drawLines(currentStepData.aifs_mean, '#00b894', 2.2);
        legendItems.push({{ label: 'AIFS Mean', color: '#00b894', dash: false }});
      }}

      // Superensemble Mean (Bold Black with Contour Labels)
      if (document.getElementById('chk_super').checked && currentStepData.super_mean) {{
        drawLines(currentStepData.super_mean, '#000000', 3.0);
        legendItems.push({{ label: 'Superensemble Mean', color: '#000000', dash: false }});

        ctx.font = 'bold 10px sans-serif';
        ctx.fillStyle = '#000';
        for (const [lvl, lines] of Object.entries(currentStepData.super_mean)) {{
          lines.forEach(pts => {{
            if (pts.length > 12) {{
              const mid = pts[Math.floor(pts.length / 2)];
              ctx.fillText(lvl, mid[0] + 3, mid[1] - 3);
            }}
          }});
        }}
      }}

      // 2. Member Extrema Centers (Tropical Tidbits Style)
      const showLows = document.getElementById('chk_show_lows').checked;
      const showHighs = document.getElementById('chk_show_highs').checked;
      
      const activeEnsembles = [];
      if (document.getElementById('chk_gefs_centers').checked) activeEnsembles.push(currentStepData.gefs_extrema);
      if (document.getElementById('chk_aigefs_centers').checked) activeEnsembles.push(currentStepData.aigefs_extrema);
      if (document.getElementById('chk_ifs_centers').checked) activeEnsembles.push(currentStepData.ifs_extrema);
      if (document.getElementById('chk_aifs_centers').checked) activeEnsembles.push(currentStepData.aifs_extrema);

      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      activeEnsembles.forEach(ens => {{
        if (!ens) return;
        ens.forEach(member => {{
          if (showLows && member.lows) {{
            ctx.fillStyle = '#e74c3c'; // Tropical Tidbits Red
            member.lows.forEach(pt => ctx.fillText(pt.lbl, pt.x, pt.y));
          }}
          if (showHighs && member.highs) {{
            ctx.fillStyle = '#2980b9'; // Tropical Tidbits Blue
            member.highs.forEach(pt => ctx.fillText(pt.lbl, pt.x, pt.y));
          }}
        }});
      }});

      // 3. Object-Based Median Lows & Highs
      const minPct = parseFloat(document.getElementById('minPctSelect').value);
      const plotMedLows = document.getElementById('chk_median_lows').checked;
      const plotMedHighs = document.getElementById('chk_median_highs').checked;

      function renderClusterBadges(clusters, color) {{
        clusters.forEach(c => {{
          if (c.pct < minPct) return;
          
          // Outer halo
          ctx.beginPath();
          ctx.arc(c.x, c.y, 16, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
          ctx.fill();
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = color;
          ctx.stroke();

          // Value and consensus text
          ctx.font = 'bold 11px sans-serif';
          ctx.fillStyle = color;
          ctx.fillText(c.median_p, c.x, c.y - 2);

          ctx.font = '8px sans-serif';
          ctx.fillStyle = '#333333';
          ctx.fillText(`${{c.pct}}% (${{c.count}}/${{c.total_members}})`, c.x, c.y + 7);
        }});
      }}

      if (plotMedLows && currentStepData.median_lows) {{
        renderClusterBadges(currentStepData.median_lows, '#c0392b');
      }}
      if (plotMedHighs && currentStepData.median_highs) {{
        renderClusterBadges(currentStepData.median_highs, '#2471a3');
      }}

      // Dynamic Legend
      document.getElementById('legendBox').innerHTML = legendItems.map(item => `
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="width:20px; height:3px; background:${{item.color}}; border-bottom:${{item.dash ? '2px dashed #000' : 'none'}}"></span>
          <span>${{item.label}}</span>
        </div>
      `).join('');
    }}

    loadStepData();
  </script>
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
    f.write(html_content)

print(f"🎉 Production PMSL build complete! Files written to {OUTPUT_DIR}")
