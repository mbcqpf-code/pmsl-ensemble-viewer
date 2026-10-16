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
# 1. DOMAIN CONFIGURATION & DIRECTORIES
# ==============================================================================
OUTPUT_DIR = "pmsl_output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
SPREAD_DIR = os.path.join(OUTPUT_DIR, "spread")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPREAD_DIR, exist_ok=True)

# 0 to 240 hours every 12 hours (21 lead times)
FXX_STEPS = list(range(0, 241, 12))
PMSL_CONTOURS = list(range(960, 1052, 4))
CANVAS_W, CANVAS_H = 1200, 800

http_session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)
http_session.headers.update({"User-Agent": "Mozilla/5.0 (PMSL-Viewer; Operational Met Research)"})

lon_min, lon_max = -130.0, -60.0
lat_min, lat_max = 20.0, 55.0
grid_lats = np.arange(lat_min, lat_max + 0.5, 0.5)
grid_lons = np.arange(lon_min, lon_max + 0.5, 0.5)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)


# ==============================================================================
# 2. CYCLE DETECTOR (IFS & AIGEFS SFC AT F240)
# ==============================================================================
def detect_latest_complete_cycle():
    now = datetime.utcnow()
    latest_hour = 12 if now.hour >= 12 else 0
    anchor = now.replace(hour=latest_hour, minute=0, second=0, microsecond=0)
    candidates = [anchor - timedelta(hours=12 * i) for i in range(4)]
    
    print("🔍 Probing for newest complete 240h ensemble cycle...")
    for cand in candidates:
        cycle_str = cand.strftime('%Y-%m-%d %H:00')
        print(f"  • Checking candidate cycle: {cycle_str}Z...")
        
        ifs_ready = False
        try:
            H_ifs = Herbie(cycle_str, model="ifs", product="enfo", fxx=240)
            if H_ifs.grib is not None:
                ifs_ready = True
        except Exception:
            pass

        if not ifs_ready:
            continue

        aigefs_ready = False
        date_nomads = cand.strftime('%Y%m%d')
        cycle_nomads = cand.strftime('%H')
        url_aigefs_sfc = (
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/"
            f"aigefs.{date_nomads}/{cycle_nomads}/mem000/model/atmos/grib2/"
            f"aigefs.t{cycle_nomads}z.sfc.f240.grib2.idx"
        )
        try:
            resp = http_session.head(url_aigefs_sfc, timeout=5)
            if resp.status_code == 200:
                aigefs_ready = True
            else:
                H_ai = Herbie(cycle_str, model="aigefs", product="sfc", member="mem000", fxx=240)
                if H_ai.grib is not None:
                    aigefs_ready = True
        except Exception:
            pass

        if ifs_ready and aigefs_ready:
            print(f"🎯 Target Cycle Verified: {cycle_str}Z (All global suites complete out to 240h)\n")
            return cycle_str

    fallback_str = candidates[1].strftime('%Y-%m-%d %H:00')
    print(f"⚠️ Falling back to verified baseline: {fallback_str}Z\n")
    return fallback_str

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
# 3. GRID STANDARDIZATION & ISOBAR EXTRACTION
# ==============================================================================
def standardize_pmsl_grid(ds):
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
        if np.nanmean(arr) > 2000:
            arr = arr / 100.0  # Convert Pa -> hPa
        return arr
    except Exception:
        return None

def extract_contours(grid):
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
# 4. EXTREMA EXTRACTION & DENSITY-SEEDED RADIAL CLUSTERING
# ==============================================================================
def detect_member_centers(grid):
    if grid is None or np.all(np.isnan(grid)):
        return {"lows": [], "highs": []}
    
    window = 9  # ~4.5 degrees radius
    min_flt = minimum_filter(grid, size=window, mode='nearest')
    max_flt = maximum_filter(grid, size=window, mode='nearest')
    
    low_mask = (grid == min_flt) & (grid < 1020.0)
    high_mask = (grid == max_flt) & (grid > 1012.0)
    
    low_mask[:5, :] = low_mask[-5:, :] = low_mask[:, :5] = low_mask[:, -5:] = False
    high_mask[:5, :] = high_mask[-5:, :] = high_mask[:, :5] = high_mask[:, -5:] = False
    
    lows, highs = [], []
    for r, c in zip(*np.where(low_mask)):
        lat, lon = grid_lats[r], grid_lons[c]
        disp_pt = geo_trans.transform(np.array([[lon, lat]]))
        fig_pt = fig.transFigure.inverted().transform(disp_pt)[0]
        px = int(round(fig_pt[0] * CANVAS_W))
        py = int(round((1.0 - fig_pt[1]) * CANVAS_H))
        if 0 <= px <= CANVAS_W and 0 <= py <= CANVAS_H:
            p_val = float(grid[r, c])
            lows.append({"x": px, "y": py, "val": round(p_val, 1), "lbl": f"{int(round(p_val)) % 100:02d}", "lat": float(lat), "lon": float(lon)})
            
    for r, c in zip(*np.where(high_mask)):
        lat, lon = grid_lats[r], grid_lons[c]
        disp_pt = geo_trans.transform(np.array([[lon, lat]]))
        fig_pt = fig.transFigure.inverted().transform(disp_pt)[0]
        px = int(round(fig_pt[0] * CANVAS_W))
        py = int(round((1.0 - fig_pt[1]) * CANVAS_H))
        if 0 <= px <= CANVAS_W and 0 <= py <= CANVAS_H:
            p_val = float(grid[r, c])
            highs.append({"x": px, "y": py, "val": round(p_val, 1), "lbl": f"{int(round(p_val)) % 100:02d}", "lat": float(lat), "lon": float(lon)})
            
    return {"lows": lows, "highs": highs}

def cluster_extrema_medians(member_extrema_list, total_members, center_type="lows", radius_km=350.0, model_tag="SUPER", color="#000000"):
    """
    Groups member centers using density-seeded radial clustering.
    Supports both superensemble and suite-specific cluster calculation.
    """
    all_centers = []
    for m_idx, centers in enumerate(member_extrema_list):
        for pt in centers.get(center_type, []):
            all_centers.append({**pt, "member": m_idx})
            
    if not all_centers or total_members == 0:
        return []

    def approx_dist_km(lat1, lon1, lat2, lon2):
        dlat = (lat2 - lat1) * 111.0
        dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians(0.5 * (lat1 + lat2)))
        return np.sqrt(dlat**2 + dlon**2)

    unassigned = list(all_centers)
    clusters = []
    min_members = max(2, int(total_members * 0.10))

    while unassigned:
        best_seed = None
        max_unique_members = -1
        
        for cand in unassigned:
            nearby = [p for p in unassigned if approx_dist_km(cand["lat"], cand["lon"], p["lat"], p["lon"]) <= radius_km]
            unique_count = len(set(p["member"] for p in nearby))
            
            if unique_count > max_unique_members:
                max_unique_members = unique_count
                best_seed = cand
            elif unique_count == max_unique_members and best_seed is not None:
                if (center_type == "lows" and cand["val"] < best_seed["val"]) or \
                   (center_type == "highs" and cand["val"] > best_seed["val"]):
                    best_seed = cand

        if max_unique_members < min_members or best_seed is None:
            break

        cluster_pts = [p for p in unassigned if approx_dist_km(best_seed["lat"], best_seed["lon"], p["lat"], p["lon"]) <= radius_km]

        member_dict = {}
        for p in cluster_pts:
            m = p["member"]
            if m not in member_dict:
                member_dict[m] = p
            else:
                if (center_type == "lows" and p["val"] < member_dict[m]["val"]) or \
                   (center_type == "highs" and p["val"] > member_dict[m]["val"]):
                    member_dict[m] = p

        final_pts = list(member_dict.values())
        support_count = len(final_pts)
        support_pct = (support_count / total_members) * 100.0 if total_members > 0 else 0

        pressures = [p["val"] for p in final_pts]
        med_p = float(np.median(pressures))
        med_lat = float(np.median([p["lat"] for p in final_pts]))
        med_lon = float(np.median([p["lon"] for p in final_pts]))

        disp_pt = geo_trans.transform(np.array([[med_lon, med_lat]]))
        fig_pt = fig.transFigure.inverted().transform(disp_pt)[0]
        px = int(round(fig_pt[0] * CANVAS_W))
        py = int(round((1.0 - fig_pt[1]) * CANVAS_H))

        clusters.append({
            "model": model_tag,
            "color": color,
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

        assigned_keys = set((p["lat"], p["lon"], p["member"]) for p in cluster_pts)
        unassigned = [p for p in unassigned if (p["lat"], p["lon"], p["member"]) not in assigned_keys]

    return clusters


# ==============================================================================
# 5. DATA INGESTION ENGINE
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

def fetch_aigefs_direct_http(date_str, cycle_str, member_str, fxx):
    fxx_str = f"f{fxx:03d}"
    url_base = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/aigefs.{date_str}/{cycle_str}/{member_str}/model/atmos/grib2/"
    fname = f"aigefs.t{cycle_str}z.sfc.{fxx_str}.grib2"
    try:
        r_idx = http_session.get(f"{url_base}{fname}.idx", timeout=8)
        if r_idx.status_code != 200:
            return None
        idx_lines = r_idx.text.splitlines()
        target_idx = None
        for i, line in enumerate(idx_lines):
            if ":PRMSL:mean sea level:" in line:
                target_idx = i
                break
        if target_idx is None:
            return None
            
        start_byte = int(idx_lines[target_idx].split(':')[1])
        end_byte = int(idx_lines[target_idx + 1].split(':')[1]) - 1 if target_idx + 1 < len(idx_lines) else ""
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        
        resp = http_session.get(f"{url_base}{fname}", headers=headers, timeout=12)
        if resp.status_code not in [200, 206]:
            return None
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
            
        ds = xr.open_dataset(tmp_path, engine="cfgrib")
        grid = standardize_pmsl_grid(ds)
        ds.close()
        os.remove(tmp_path)
        return grid
    except Exception:
        return None

def fetch_aigefs_pmsl(fxx):
    date_dt = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M')
    date_str = date_dt.strftime('%Y%m%d')
    cycle_str = date_dt.strftime('%H')
    members = [f"mem{str(i).zfill(3)}" for i in range(31)]
    
    def _worker(m_str):
        try:
            H = Herbie(global_init_time, model="aigefs", product="sfc", member=m_str, fxx=fxx, source="nomads")
            grid = standardize_pmsl_grid(H.xarray(":PRMSL:mean sea level:"))
            if grid is not None and grid.ndim == 2:
                return grid
        except Exception:
            pass
        return fetch_aigefs_direct_http(date_str, cycle_str, m_str, fxx)

    grids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_worker, members):
            if res is not None and res.ndim == 2:
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

def fetch_aigfs_pmsl(fxx):
    date_dt = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M')
    date_str = date_dt.strftime('%Y%m%d')
    cycle_str = date_dt.strftime('%H')
    fxx_str = f"f{fxx:03d}"
    
    try:
        H = Herbie(global_init_time, model="aigfs", product="sfc", fxx=fxx, source="nomads")
        grid = standardize_pmsl_grid(H.xarray(":PRMSL:mean sea level:"))
        if grid is not None and grid.ndim == 2:
            return grid
    except Exception:
        pass
        
    url_base = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{date_str}/{cycle_str}/model/atmos/grib2/"
    fname = f"aigfs.t{cycle_str}z.sfc.{fxx_str}.grib2"
    try:
        r_idx = http_session.get(f"{url_base}{fname}.idx", timeout=8)
        if r_idx.status_code == 200:
            idx_lines = r_idx.text.splitlines()
            target_idx = None
            for i, line in enumerate(idx_lines):
                if ":PRMSL:mean sea level:" in line:
                    target_idx = i
                    break
            if target_idx is not None:
                start_byte = int(idx_lines[target_idx].split(':')[1])
                end_byte = int(idx_lines[target_idx + 1].split(':')[1]) - 1 if target_idx + 1 < len(idx_lines) else ""
                headers = {"Range": f"bytes={start_byte}-{end_byte}"}
                resp = http_session.get(f"{url_base}{fname}", headers=headers, timeout=12)
                if resp.status_code in [200, 206]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name
                    ds = xr.open_dataset(tmp_path, engine="cfgrib")
                    grid = standardize_pmsl_grid(ds)
                    ds.close()
                    os.remove(tmp_path)
                    return grid
    except Exception:
        pass
    return None

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
# 6. ENSEMBLE SPREAD UNDERLAYS
# ==============================================================================
PMSL_SPREAD_LEVELS = np.arange(1.0, 13.0, 1.0)
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
# 7. BATCH EXTRACTION & PROCESSING LOOP
# ==============================================================================
for fxx in FXX_STEPS:
    valid_time = datetime.strptime(global_init_time, '%Y-%m-%d %H:%M') + timedelta(hours=fxx)
    print(f"\n📦 Processing PMSL Lead F{fxx:03d} (Valid: {valid_time.strftime('%Y-%m-%d %H:%MZ')})...")
    
    gefs_mems   = fetch_gefs_pmsl(fxx)
    aigefs_mems = fetch_aigefs_pmsl(fxx)
    ifs_mems    = fetch_ifs_pmsl(fxx)
    aifs_mems   = fetch_aifs_pmsl(fxx)

    all_members = gefs_mems + aigefs_mems + ifs_mems + aifs_mems
    print(f"  ↳ Members loaded: GEFS={len(gefs_mems)}, AIGEFS={len(aigefs_mems)}, IFS={len(ifs_mems)}, AIFS={len(aifs_mems)} (Total: {len(all_members)})")

    det_gfs   = fetch_det_pmsl("gfs", "pgrb2.0p25", ":PRMSL:mean sea level:", fxx)
    det_aigfs = fetch_aigfs_pmsl(fxx)
    det_ifs   = fetch_det_pmsl("ifs", "oper", ":msl:", fxx)
    det_aifs  = fetch_det_pmsl("aifs", "oper", ":msl:", fxx)

    gefs_mean   = np.mean(np.stack(gefs_mems), axis=0) if gefs_mems else None
    aigefs_mean = np.mean(np.stack(aigefs_mems), axis=0) if aigefs_mems else None
    ifs_mean    = np.mean(np.stack(ifs_mems), axis=0) if ifs_mems else None
    aifs_mean   = np.mean(np.stack(aifs_mems), axis=0) if aifs_mems else None
    super_mean  = np.mean(np.stack(all_members), axis=0) if all_members else None

    spread_diags = compute_spread_diagnostics(gefs_mems, aigefs_mems, ifs_mems, aifs_mems)
    for spr_key, spr_grid in spread_diags.items():
        if spr_grid is not None:
            render_spread_underlay(spr_grid, f"{spr_key}_f{fxx:03d}.png")

    gefs_extrema   = [detect_member_centers(m) for m in gefs_mems]
    aigefs_extrema = [detect_member_centers(m) for m in aigefs_mems]
    ifs_extrema    = [detect_member_centers(m) for m in ifs_mems]
    aifs_extrema   = [detect_member_centers(m) for m in aifs_mems]
    all_extrema    = gefs_extrema + aigefs_extrema + ifs_extrema + aifs_extrema

    det_extrema = {
        "gfs": detect_member_centers(det_gfs),
        "aigfs": detect_member_centers(det_aigfs),
        "ifs": detect_member_centers(det_ifs),
        "aifs": detect_member_centers(det_aifs)
    }

    total_mem_count = len(all_members)

    # 1. Superensemble Median Clusters
    super_med_lows  = cluster_extrema_medians(all_extrema, total_mem_count, "lows", 350.0, "SUPER", "#000000")
    super_med_highs = cluster_extrema_medians(all_extrema, total_mem_count, "highs", 350.0, "SUPER", "#000000")

    # 2. Individual Ensemble Suite Median Clusters
    gefs_med_lows   = cluster_extrema_medians(gefs_extrema, len(gefs_mems), "lows", 350.0, "GEFS", "#0044ff") if gefs_mems else []
    gefs_med_highs  = cluster_extrema_medians(gefs_extrema, len(gefs_mems), "highs", 350.0, "GEFS", "#0044ff") if gefs_mems else []

    aigefs_med_lows  = cluster_extrema_medians(aigefs_extrema, len(aigefs_mems), "lows", 350.0, "AIGEFS", "#8e44ad") if aigefs_mems else []
    aigefs_med_highs = cluster_extrema_medians(aigefs_extrema, len(aigefs_mems), "highs", 350.0, "AIGEFS", "#8e44ad") if aigefs_mems else []

    ifs_med_lows    = cluster_extrema_medians(ifs_extrema, len(ifs_mems), "lows", 350.0, "IFS", "#d63031") if ifs_mems else []
    ifs_med_highs   = cluster_extrema_medians(ifs_extrema, len(ifs_mems), "highs", 350.0, "IFS", "#d63031") if ifs_mems else []

    aifs_med_lows   = cluster_extrema_medians(aifs_extrema, len(aifs_mems), "lows", 350.0, "AIFS", "#00b894") if aifs_mems else []
    aifs_med_highs  = cluster_extrema_medians(aifs_extrema, len(aifs_mems), "highs", 350.0, "AIFS", "#00b894") if aifs_mems else []

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
        "aigfs_det": extract_contours(det_aigfs),
        "ifs_det": extract_contours(det_ifs),
        "aifs_det": extract_contours(det_aifs),
        "gefs_extrema": gefs_extrema,
        "aigefs_extrema": aigefs_extrema,
        "ifs_extrema": ifs_extrema,
        "aifs_extrema": aifs_extrema,
        "det_extrema": det_extrema,
        "median_lows": {
            "super": super_med_lows,
            "gefs": gefs_med_lows,
            "aigefs": aigefs_med_lows,
            "ifs": ifs_med_lows,
            "aifs": aifs_med_lows
        },
        "median_highs": {
            "super": super_med_highs,
            "gefs": gefs_med_highs,
            "aigefs": aigefs_med_highs,
            "ifs": ifs_med_highs,
            "aifs": aifs_med_highs
        }
    }

    with open(os.path.join(DATA_DIR, f"f{fxx:03d}.json"), "w") as f:
        json.dump(step_data, f, separators=(',', ':'))

plt.close(fig)


# ==============================================================================
# 8. FRONTEND GENERATOR (SUITE MEDIANS & COMPACT CONTROLS)
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
      --map-w: min(98vw, calc((100vh - 180px) * 1.5));
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; padding: 4px 10px 12px 10px; box-sizing: border-box;
      background: var(--bg); color: var(--text);
      display: flex; flex-direction: column; align-items: center; min-height: 100vh;
    }}
    .header {{ width: 100%; max-width: var(--map-w); padding: 4px 6px; box-sizing: border-box; text-align: center; }}
    .header h2 {{ margin: 0; font-size: 1.25rem; }}
    .header .subtitle {{ font-size: 0.8rem; color: var(--text-muted); margin-top: 2px; }}

    .toolbar {{
      width: 100%; max-width: var(--map-w); background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 6px 12px; margin: 4px 0 6px 0; box-sizing: border-box;
      display: flex; flex-direction: column; gap: 6px;
    }}
    .row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 10px; justify-content: space-between; }}
    .btn {{
      background: var(--accent); color: white; border: none; padding: 4px 10px; border-radius: 5px;
      cursor: pointer; font-size: 12px; font-weight: 600; transition: filter 0.15s;
    }}
    .btn:hover {{ filter: brightness(1.15); }}
    .btn-secondary {{ background: #475569; }}
    .btn-micro {{
      background: #334155; color: #cbd5e1; border: 1px solid #475569; padding: 1px 6px;
      font-size: 10px; font-weight: 600; border-radius: 4px; cursor: pointer; line-height: 1.2;
    }}
    .btn-micro:hover {{ background: #475569; color: white; }}
    select {{
      padding: 4px 8px; border-radius: 5px; background: #0f172a; color: white;
      border: 1px solid var(--border); font-size: 12px;
    }}
    
    .panel-box {{
      border-top: 1px solid var(--border); padding-top: 6px; display: flex; flex-direction: column; gap: 6px;
    }}
    .group-line {{ display: flex; align-items: center; gap: 8px 12px; flex-wrap: wrap; font-size: 12.5px; }}
    .group-label {{
      font-weight: bold; min-width: 175px; display: flex; align-items: center; gap: 4px;
    }}
    .checkbox-item {{ display: flex; align-items: center; gap: 4px; cursor: pointer; user-select: none; }}
    .swatch {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
    
    .badge-tag {{
      display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10.5px; font-weight: bold;
    }}

    .viewer-card {{
      position: relative; width: 100%; max-width: var(--map-w); aspect-ratio: 1200 / 800;
      background: #000; border-radius: 8px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      cursor: grab; user-select: none;
    }}
    .viewer-card:active {{ cursor: grabbing; }}
    #contourCanvas {{ width: 100%; height: 100%; display: block; }}
    
    .legend-box {{
      position: absolute; top: 10px; right: 10px; background: rgba(255, 255, 255, 0.94);
      color: #111; padding: 6px 10px; border-radius: 6px; font-size: 11px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 3px;
      pointer-events: none;
    }}
    .title-stamp {{
      position: absolute; top: 10px; left: 10px; background: rgba(255, 255, 255, 0.94);
      color: #111; padding: 5px 8px; border-radius: 6px; font-size: 11px; font-weight: bold;
      pointer-events: none; line-height: 1.3;
    }}
    
    .zoom-controls {{
      position: absolute; bottom: 12px; right: 12px; display: flex; flex-direction: column; gap: 5px; z-index: 10;
    }}
    .zoom-btn {{
      background: rgba(30, 41, 59, 0.92); color: white; border: 1px solid var(--border);
      width: 30px; height: 30px; border-radius: 5px; font-size: 15px; font-weight: bold;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 10px rgba(0,0,0,0.3); transition: background 0.15s;
    }}
    .zoom-btn:hover {{ background: #3b82f6; }}
    .zoom-indicator {{
      background: rgba(15, 23, 42, 0.85); color: #cbd5e1; font-size: 10px;
      padding: 1px 3px; border-radius: 3px; text-align: center; border: 1px solid var(--border);
    }}

    .hover-tooltip {{
      position: fixed; display: none; pointer-events: none; z-index: 1000;
      background: rgba(15, 23, 42, 0.96); border: 1px solid #3b82f6; color: white;
      padding: 8px 12px; border-radius: 6px; font-size: 12px; line-height: 1.4;
      box-shadow: 0 8px 24px rgba(0,0,0,0.7); max-width: 290px;
    }}
    .spread-colorbar {{ display: none; align-items: center; gap: 6px; font-size: 11px; }}
    .colorbar-gradient {{
      width: 110px; height: 10px; border-radius: 2px;
      background: linear-gradient(to right, #ffffcc, #fed976, #feb24c, #fd8d3c, #f03b20, #bd0026);
      border: 1px solid var(--border);
    }}
  </style>
</head>
<body>

  <div class="header">
    <h2>PMSL Multi-Ensemble & Extrema Center Viewer</h2>
    <div class="subtitle">Cycle: {global_init_time}Z | Multi-Suite Medians & Interactive Synoptic Inspection</div>
  </div>

  <div class="toolbar">
    <!-- Row 1: Lead Scrubbing & Underlay -->
    <div class="row">
      <div style="display: flex; gap: 5px; align-items: center;">
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

    <!-- Row 2: Isobars -->
    <div class="panel-box">
      <div class="group-line">
        <span class="group-label">
          Isobars:
          <button class="btn-micro" onclick="setGroup('isobars', true)">All</button>
          <button class="btn-micro" onclick="setGroup('isobars', false)">None</button>
        </span>
        <label class="checkbox-item"><input type="checkbox" id="chk_super" checked onchange="draw()"><span class="swatch" style="background:#000;"></span> Super Mean</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_gefs_mean" checked onchange="draw()"><span class="swatch" style="background:#0044ff;"></span> GEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aigefs_mean" checked onchange="draw()"><span class="swatch" style="background:#8e44ad;"></span> AIGEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ifs_mean" checked onchange="draw()"><span class="swatch" style="background:#d63031;"></span> IFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs_mean" checked onchange="draw()"><span class="swatch" style="background:#00b894;"></span> AIFS</label>
        <span style="border-left: 1px solid var(--border); height: 14px; margin: 0 2px;"></span>
        <label class="checkbox-item"><input type="checkbox" id="chk_gfs_det" onchange="draw()"> GFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aigfs_det" onchange="draw()"> AIGFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ifs_det" onchange="draw()"> IFS Det</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs_det" onchange="draw()"> AIFS Det</label>
      </div>

      <!-- Row 3: Member Centers -->
      <div class="group-line">
        <span class="group-label">
          Member Centers:
          <button class="btn-micro" onclick="setGroup('members', true)">All</button>
          <button class="btn-micro" onclick="setGroup('members', false)">None</button>
        </span>
        <label class="checkbox-item"><input type="checkbox" id="chk_show_lows" checked onchange="draw()"><span style="color:#e74c3c; font-weight:bold;">Lows (Red)</span></label>
        <label class="checkbox-item"><input type="checkbox" id="chk_show_highs" checked onchange="draw()"><span style="color:#2980b9; font-weight:bold;">Highs (Blue)</span></label>
        <span style="border-left: 1px solid var(--border); height: 14px; margin: 0 2px;"></span>
        <label class="checkbox-item"><input type="checkbox" id="chk_gefs_centers" checked onchange="draw()"> GEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aigefs_centers" checked onchange="draw()"> AIGEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_ifs_centers" checked onchange="draw()"> IFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_aifs_centers" checked onchange="draw()"> AIFS</label>
      </div>

      <!-- Row 4: Deterministic Centers -->
      <div class="group-line">
        <span class="group-label">
          Deterministic Centers:
          <button class="btn-micro" onclick="setGroup('det_centers', true)">All</button>
          <button class="btn-micro" onclick="setGroup('det_centers', false)">None</button>
        </span>
        <label class="checkbox-item"><input type="checkbox" id="chk_det_gfs" checked onchange="draw()"><span class="badge-tag" style="background:#001155; color:#fff;">GFS</span></label>
        <label class="checkbox-item"><input type="checkbox" id="chk_det_aigfs" checked onchange="draw()"><span class="badge-tag" style="background:#4a235a; color:#fff;">AIGFS</span></label>
        <label class="checkbox-item"><input type="checkbox" id="chk_det_ifs" checked onchange="draw()"><span class="badge-tag" style="background:#78281f; color:#fff;">IFS</span></label>
        <label class="checkbox-item"><input type="checkbox" id="chk_det_aifs" checked onchange="draw()"><span class="badge-tag" style="background:#145a32; color:#fff;">AIFS</span></label>
      </div>

      <!-- Row 5: Ensemble Medians (Super & Individual Suites) -->
      <div class="group-line" style="background: rgba(59, 130, 246, 0.08); padding: 4px 8px; border-radius: 5px;">
        <span class="group-label">
          Ensemble Medians:
          <button class="btn-micro" onclick="setGroup('medians', true)">All</button>
          <button class="btn-micro" onclick="setGroup('medians', false)">None</button>
        </span>
        <label class="checkbox-item"><input type="checkbox" id="chk_med_super" checked onchange="draw()"><span class="swatch" style="background:#000;"></span> Super</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_med_gefs" checked onchange="draw()"><span class="swatch" style="background:#0044ff;"></span> GEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_med_aigefs" checked onchange="draw()"><span class="swatch" style="background:#8e44ad;"></span> AIGEFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_med_ifs" checked onchange="draw()"><span class="swatch" style="background:#d63031;"></span> IFS</label>
        <label class="checkbox-item"><input type="checkbox" id="chk_med_aifs" checked onchange="draw()"><span class="swatch" style="background:#00b894;"></span> AIFS</label>
        
        <span style="border-left: 1px solid var(--border); height: 14px; margin: 0 4px;"></span>
        <label class="checkbox-item"><input type="checkbox" id="chk_med_type_lows" checked onchange="draw()"><span style="color:#e74c3c; font-weight:bold;">Lows</span></label>
        <label class="checkbox-item"><input type="checkbox" id="chk_med_type_highs" onchange="draw()"><span style="color:#2980b9; font-weight:bold;">Highs</span></label>
        
        <span style="display: flex; align-items: center; gap: 5px; margin-left: 6px;">
          <span>Min Consensus:</span>
          <select id="minPctSelect" onchange="draw()">
            <option value="15">≥ 15%</option>
            <option value="20" selected>≥ 20%</option>
            <option value="30">≥ 30%</option>
            <option value="50">≥ 50%</option>
          </select>
        </span>
      </div>
    </div>
  </div>

  <div class="viewer-card" id="viewerCard">
    <canvas id="contourCanvas" width="1200" height="800"></canvas>
    
    <div class="title-stamp" id="mapTitleStamp">
      PMSL (hPa) & Extrema Centers<br>Loading...
    </div>

    <div class="legend-box" id="legendBox"></div>

    <div class="zoom-controls">
      <button class="zoom-btn" onclick="zoomDelta(1.3)" title="Zoom In">+</button>
      <button class="zoom-btn" onclick="zoomDelta(0.77)" title="Zoom Out">−</button>
      <button class="zoom-btn" onclick="resetZoom()" title="Reset / Fit Screen" style="font-size:10px;">Fit</button>
      <div class="zoom-indicator" id="zoomLevelText">1.0x</div>
    </div>
  </div>

  <div class="hover-tooltip" id="hoverTooltip"></div>

  <script>
    const CANVAS_W = 1200;
    const CANVAS_H = 800;
    let dataCache = {{}};
    let currentStepData = null;
    let underlayImg = new Image();
    underlayImg.src = 'basemap.png';

    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startX = 0;
    let startY = 0;

    const card = document.getElementById('viewerCard');
    const canvas = document.getElementById('contourCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('hoverTooltip');

    underlayImg.onload = () => draw();

    function setGroup(grp, state) {{
      if (grp === 'isobars') {{
        ['chk_super', 'chk_gefs_mean', 'chk_aigefs_mean', 'chk_ifs_mean', 'chk_aifs_mean',
         'chk_gfs_det', 'chk_aigfs_det', 'chk_ifs_det', 'chk_aifs_det'].forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.checked = state;
        }});
      }} else if (grp === 'members') {{
        ['chk_gefs_centers', 'chk_aigefs_centers', 'chk_ifs_centers', 'chk_aifs_centers'].forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.checked = state;
        }});
        if (state) {{
          document.getElementById('chk_show_lows').checked = true;
          document.getElementById('chk_show_highs').checked = true;
        }}
      }} else if (grp === 'det_centers') {{
        ['chk_det_gfs', 'chk_det_aigfs', 'chk_det_ifs', 'chk_det_aifs'].forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.checked = state;
        }});
      }} else if (grp === 'medians') {{
        ['chk_med_super', 'chk_med_gefs', 'chk_med_aigefs', 'chk_med_ifs', 'chk_med_aifs'].forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.checked = state;
        }});
      }}
      draw();
    }}

    function updateBackground() {{
      const bg = document.getElementById('bgSelect').value;
      const fxx = document.getElementById('leadSelect').value;
      const cbar = document.getElementById('cbarBox');

      if (bg === 'none') {{
        underlayImg.src = 'basemap.png';
        cbar.style.display = 'none';
      }} else {{
        underlayImg.src = `spread/${{bg}}_f${{fxx}}.png`;
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

    // Pan & Zoom
    function clampPan() {{
      const minX = CANVAS_W * (1 - zoom);
      const minY = CANVAS_H * (1 - zoom);
      panX = Math.min(0, Math.max(minX, panX));
      panY = Math.min(0, Math.max(minY, panY));
    }}

    function zoomAt(mx, my, factor) {{
      const oldZoom = zoom;
      zoom = Math.min(Math.max(1.0, zoom * factor), 6.0);
      if (zoom === 1.0) {{
        panX = 0; panY = 0;
      }} else {{
        panX = mx - (mx - panX) * (zoom / oldZoom);
        panY = my - (my - panY) * (zoom / oldZoom);
        clampPan();
      }}
      document.getElementById('zoomLevelText').textContent = zoom.toFixed(1) + 'x';
      draw();
    }}

    function zoomDelta(factor) {{
      zoomAt(CANVAS_W / 2, CANVAS_H / 2, factor);
    }}

    function resetZoom() {{
      zoom = 1.0;
      panX = 0;
      panY = 0;
      document.getElementById('zoomLevelText').textContent = '1.0x';
      draw();
    }}

    card.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const rect = card.getBoundingClientRect();
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;
      zoomAt(mx, my, e.deltaY < 0 ? 1.18 : 0.85);
    }}, {{ passive: false }});

    card.addEventListener('mousedown', (e) => {{
      if (zoom <= 1.0) return;
      isDragging = true;
      const rect = card.getBoundingClientRect();
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      startX = (e.clientX - rect.left) * scaleX - panX;
      startY = (e.clientY - rect.top) * scaleY - panY;
    }});

    window.addEventListener('mousemove', (e) => {{
      if (isDragging) {{
        const rect = card.getBoundingClientRect();
        const scaleX = CANVAS_W / rect.width;
        const scaleY = CANVAS_H / rect.height;
        panX = (e.clientX - rect.left) * scaleX - startX;
        panY = (e.clientY - rect.top) * scaleY - startY;
        clampPan();
        draw();
      }}
      handleHoverInspection(e);
    }});

    window.addEventListener('mouseup', () => {{ isDragging = false; }});

    // Synced Multi-Suite Hover Inspection
    function handleHoverInspection(e) {{
      if (!currentStepData) return;
      const rect = card.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {{
        tooltip.style.display = 'none';
        return;
      }}
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      const mapX = (mx - panX) / zoom;
      const mapY = (my - panY) / zoom;

      const minPct = parseFloat(document.getElementById('minPctSelect').value);
      const showLows = document.getElementById('chk_med_type_lows').checked;
      const showHighs = document.getElementById('chk_med_type_highs').checked;

      const suiteActive = {{
        super: document.getElementById('chk_med_super').checked,
        gefs: document.getElementById('chk_med_gefs').checked,
        aigefs: document.getElementById('chk_med_aigefs').checked,
        ifs: document.getElementById('chk_med_ifs').checked,
        aifs: document.getElementById('chk_med_aifs').checked
      }};

      const activeClusters = [];
      Object.keys(suiteActive).forEach(suiteKey => {{
        if (!suiteActive[suiteKey]) return;
        if (showLows && currentStepData.median_lows && currentStepData.median_lows[suiteKey]) {{
          activeClusters.push(...currentStepData.median_lows[suiteKey]);
        }}
        if (showHighs && currentStepData.median_highs && currentStepData.median_highs[suiteKey]) {{
          activeClusters.push(...currentStepData.median_highs[suiteKey]);
        }}
      }});

      // Snapping distance
      const nearby = activeClusters.filter(c => {{
        if (c.pct < minPct) return false;
        return Math.hypot(c.x - mapX, c.y - mapY) < (30 / zoom);
      }});

      if (nearby.length > 0) {{
        let content = `<div style="font-weight:bold; font-size:12px; margin-bottom:4px; border-bottom:1px solid #334155; padding-bottom:3px;">
          Ensemble Median Centers (${{nearby.length}})
        </div>`;

        nearby.forEach(c => {{
          const isL = c.type === 'L';
          content += `
            <div style="margin-bottom:5px;">
              <span style="color:${{c.color}}; font-weight:bold;">${{c.model}} Median ${{isL ? 'Low' : 'High'}}:</span>
              <b>${{c.median_p}} hPa</b>
              <div style="font-size:11px; color:#cbd5e1;">
                Consensus: <b>${{c.pct}}%</b> (${{c.count}}/${{c.total_members}} members)<br>
                Spread (P10–P90): ${{c.p10}} – ${{c.p90}} hPa (${{c.min_p}}–${{c.max_p}} abs)<br>
                Loc: ${{c.lat.toFixed(1)}}°N, ${{Math.abs(c.lon).toFixed(1)}}°W
              </div>
            </div>
          `;
        }});

        tooltip.innerHTML = content;
        tooltip.style.left = (e.clientX + 14) + 'px';
        tooltip.style.top = (e.clientY + 14) + 'px';
        tooltip.style.display = 'block';
      }} else {{
        tooltip.style.display = 'none';
      }}
    }}

    // =========================================================================
    // RENDERING ENGINE
    // =========================================================================
    function draw() {{
      if (!currentStepData) return;
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

      ctx.save();
      ctx.translate(panX, panY);
      ctx.scale(zoom, zoom);

      if (underlayImg.complete && underlayImg.naturalWidth !== 0) {{
        ctx.drawImage(underlayImg, 0, 0, CANVAS_W, CANVAS_H);
      }}

      document.getElementById('mapTitleStamp').innerHTML = 
        `PMSL (hPa) | Lead: F${{String(currentStepData.fxx).padStart(3, '0')}}<br>` +
        `Init: ${{currentStepData.init}}Z | Valid: ${{currentStepData.valid}} | Members: ${{currentStepData.total_members}}`;

      const legendItems = [];

      function drawLines(layerDict, color, width, dashed = false) {{
        if (!layerDict) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width / Math.pow(zoom, 0.4);
        ctx.setLineDash(dashed ? [6 / zoom, 4 / zoom] : []);
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

      // Deterministic Isobars
      if (document.getElementById('chk_gfs_det').checked && currentStepData.gfs_det) {{
        drawLines(currentStepData.gfs_det, '#001155', 1.8, true);
        legendItems.push({{ label: 'GFS Det', color: '#001155', dash: true }});
      }}
      if (document.getElementById('chk_aigfs_det').checked && currentStepData.aigfs_det) {{
        drawLines(currentStepData.aigfs_det, '#4a235a', 1.8, true);
        legendItems.push({{ label: 'AIGFS Det', color: '#4a235a', dash: true }});
      }}
      if (document.getElementById('chk_ifs_det').checked && currentStepData.ifs_det) {{
        drawLines(currentStepData.ifs_det, '#78281f', 1.8, true);
        legendItems.push({{ label: 'IFS Det', color: '#78281f', dash: true }});
      }}
      if (document.getElementById('chk_aifs_det').checked && currentStepData.aifs_det) {{
        drawLines(currentStepData.aifs_det, '#145a32', 1.8, true);
        legendItems.push({{ label: 'AIFS Det', color: '#145a32', dash: true }});
      }}

      // Ensemble Mean Isobars
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

      // Superensemble Mean
      if (document.getElementById('chk_super').checked && currentStepData.super_mean) {{
        drawLines(currentStepData.super_mean, '#000000', 3.0);
        legendItems.push({{ label: 'Superensemble Mean', color: '#000000', dash: false }});

        ctx.font = `bold ${{Math.round(10 / zoom)}}px sans-serif`;
        ctx.fillStyle = '#000';
        for (const [lvl, lines] of Object.entries(currentStepData.super_mean)) {{
          lines.forEach(pts => {{
            if (pts.length > 12) {{
              const mid = pts[Math.floor(pts.length / 2)];
              ctx.fillText(lvl, mid[0] + 3 / zoom, mid[1] - 3 / zoom);
            }}
          }});
        }}
      }}

      // Member Extrema Centers
      const showLows = document.getElementById('chk_show_lows').checked;
      const showHighs = document.getElementById('chk_show_highs').checked;
      
      const activeEnsembles = [];
      if (document.getElementById('chk_gefs_centers').checked) activeEnsembles.push(currentStepData.gefs_extrema);
      if (document.getElementById('chk_aigefs_centers').checked) activeEnsembles.push(currentStepData.aigefs_extrema);
      if (document.getElementById('chk_ifs_centers').checked) activeEnsembles.push(currentStepData.ifs_extrema);
      if (document.getElementById('chk_aifs_centers').checked) activeEnsembles.push(currentStepData.aifs_extrema);

      ctx.font = `bold ${{Math.round(11 / zoom)}}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      activeEnsembles.forEach(ens => {{
        if (!ens) return;
        ens.forEach(member => {{
          if (showLows && member.lows) {{
            ctx.fillStyle = '#e74c3c';
            member.lows.forEach(pt => ctx.fillText(pt.lbl, pt.x, pt.y));
          }}
          if (showHighs && member.highs) {{
            ctx.fillStyle = '#2980b9';
            member.highs.forEach(pt => ctx.fillText(pt.lbl, pt.x, pt.y));
          }}
        }});
      }});

      // Deterministic Centers
      const detConfigs = [
        {{ key: 'gfs', chk: 'chk_det_gfs', color: '#001155', tag: 'GFS' }},
        {{ key: 'aigfs', chk: 'chk_det_aigfs', color: '#4a235a', tag: 'AIGFS' }},
        {{ key: 'ifs', chk: 'chk_det_ifs', color: '#78281f', tag: 'IFS' }},
        {{ key: 'aifs', chk: 'chk_det_aifs', color: '#145a32', tag: 'AIFS' }}
      ];

      detConfigs.forEach(d => {{
        if (!document.getElementById(d.chk).checked || !currentStepData.det_extrema) return;
        const extrema = currentStepData.det_extrema[d.key];
        if (!extrema) return;

        function renderDetBadge(pts, isLow) {{
          pts.forEach(pt => {{
            const badgeW = 44 / zoom;
            const badgeH = 18 / zoom;
            const rx = pt.x - badgeW / 2;
            const ry = pt.y - badgeH / 2;

            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
            ctx.strokeStyle = d.color;
            ctx.lineWidth = 2.0 / zoom;
            ctx.beginPath();
            ctx.roundRect(rx, ry, badgeW, badgeH, 4 / zoom);
            ctx.fill();
            ctx.stroke();

            ctx.font = `bold ${{Math.round(9 / zoom)}}px sans-serif`;
            ctx.fillStyle = isLow ? '#c0392b' : '#2471a3';
            ctx.fillText(`${{d.tag}} ${{pt.lbl}}`, pt.x, pt.y);
          }});
        }}

        if (showLows && extrema.lows) renderDetBadge(extrema.lows, true);
        if (showHighs && extrema.highs) renderDetBadge(extrema.highs, false);
      }});

      // Ensemble Medians (Super & Individual Suites)
      const minPct = parseFloat(document.getElementById('minPctSelect').value);
      const plotMedLows = document.getElementById('chk_med_type_lows').checked;
      const plotMedHighs = document.getElementById('chk_med_type_highs').checked;

      const suiteActive = {{
        super: document.getElementById('chk_med_super').checked,
        gefs: document.getElementById('chk_med_gefs').checked,
        aigefs: document.getElementById('chk_med_aigefs').checked,
        ifs: document.getElementById('chk_med_ifs').checked,
        aifs: document.getElementById('chk_med_aifs').checked
      }};

      function renderClusterBadge(c) {{
        if (c.pct < minPct) return;
        const isSuper = c.model === 'SUPER';
        const r = (isSuper ? 18 : 15.5) / zoom;

        // Base badge
        ctx.beginPath();
        ctx.arc(c.x, c.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.fill();
        ctx.lineWidth = (isSuper ? 3.0 : 2.2) / zoom;
        ctx.strokeStyle = c.color;
        ctx.stroke();

        // Model Tag
        ctx.font = `bold ${{Math.round((isSuper ? 7.5 : 7.0) / zoom)}}px sans-serif`;
        ctx.fillStyle = c.color;
        ctx.fillText(c.model, c.x, c.y - 7.5 / zoom);

        // Median Pressure
        ctx.font = `bold ${{Math.round((isSuper ? 10.5 : 9.5) / zoom)}}px sans-serif`;
        ctx.fillStyle = c.type === 'L' ? '#b91c1c' : '#1d4ed8';
        ctx.fillText(c.median_p, c.x, c.y + 0.5 / zoom);

        // Member Consensus
        ctx.font = `${{Math.round((isSuper ? 7.5 : 7.0) / zoom)}}px sans-serif`;
        ctx.fillStyle = '#475569';
        ctx.fillText(`${{Math.round(c.pct)}}% (${{c.count}})`, c.x, c.y + 8.5 / zoom);
      }}

      // Render suite medians first, super median on top
      ['gefs', 'aigefs', 'ifs', 'aifs', 'super'].forEach(k => {{
        if (!suiteActive[k]) return;
        if (plotMedLows && currentStepData.median_lows && currentStepData.median_lows[k]) {{
          currentStepData.median_lows[k].forEach(renderClusterBadge);
        }}
        if (plotMedHighs && currentStepData.median_highs && currentStepData.median_highs[k]) {{
          currentStepData.median_highs[k].forEach(renderClusterBadge);
        }}
      }});

      ctx.restore();

      // Legend Items
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
