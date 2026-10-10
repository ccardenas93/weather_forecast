"""INAMHI WRF 2 m temperature from the INAMHI-GEOGLOWS GeoTIFF service.

The service returns one GeoTIFF (EPSG:4326, about 0.027 degree, Ecuador domain)
per layer and valid hour:

    GET {WRF_SERVICE_URL}?layer=wrf_temperature&datetime=<init YYYYMMDD>_<valid YYYYMMDDHHMM>

Valid hours in the file names are local time (America/Guayaquil, UTC-5) every
3 hours from 01:00 on the initialization day out to +70 h. A missing file comes
back as a short "GeoTIFF not found" text with HTTP 200, so size is the test.

Everything here is best effort: any failure returns an empty frame and the
forecast continues on ECMWF alone.
"""
import io
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from forecast_config import (
    STATION_LAT,
    STATION_LON,
    WRF_DOWNLOAD_THREADS,
    WRF_LOCAL_UTC_OFFSET_HOURS,
    WRF_MAX_LOOKBACK_DAYS,
    WRF_MIN_FILE_BYTES,
    WRF_REQUEST_TIMEOUT_SECONDS,
    WRF_SERVICE_URL,
    WRF_TEMPERATURE_LAYER,
    WRF_VALID_HOURS,
)
from time_utils import to_utc_naive


WRF_COLUMNS = ["valid_time", "wrf_temp_c", "wrf_init_time"]


def wrf_enabled():
    return bool(WRF_SERVICE_URL)


def _request_geotiff(layer, init_date, valid_local, attempts=3):
    params = {
        "layer": layer,
        "datetime": f"{init_date:%Y%m%d}_{valid_local:%Y%m%d%H%M}",
    }
    for attempt in range(attempts):
        try:
            response = requests.get(WRF_SERVICE_URL, params=params, timeout=WRF_REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code == 200:
            content = response.content
            return content if len(content) >= WRF_MIN_FILE_BYTES else None
        time.sleep(3 * (attempt + 1))
    return None


def _sample_geotiff(content, lon, lat):
    import rasterio

    with rasterio.open(io.BytesIO(content)) as raster:
        band = raster.read(1).astype(float)
        if raster.nodata is not None:
            band[band == raster.nodata] = np.nan
        transform = raster.transform
        col = int(np.clip(np.floor((lon - transform.c) / transform.a), 0, raster.width - 1))
        row = int(np.clip(np.floor((lat - transform.f) / transform.e), 0, raster.height - 1))
        value = float(band[row, col])
    if not np.isfinite(value):
        return np.nan
    # Some exports are in Kelvin; the station never sees anything near 100 C.
    if value > 100:
        value -= 273.15
    return value


def latest_initialization(run_time):
    """Most recent initialization date (local) that has a first temperature file."""
    local_now = to_utc_naive(run_time) + pd.Timedelta(hours=WRF_LOCAL_UTC_OFFSET_HOURS)
    for days_back in range(WRF_MAX_LOOKBACK_DAYS + 1):
        init_date = (local_now - pd.Timedelta(days=days_back)).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        first_valid = init_date + timedelta(hours=WRF_VALID_HOURS[0])
        if _request_geotiff(WRF_TEMPERATURE_LAYER, init_date, first_valid) is not None:
            return init_date
    return None


def download_wrf_temperature(run_time):
    """Return a frame with UTC valid_time, wrf_temp_c and wrf_init_time for the station point."""
    empty = pd.DataFrame(columns=WRF_COLUMNS)
    if not wrf_enabled():
        return empty

    try:
        init_date = latest_initialization(run_time)
    except Exception as exc:  # noqa: BLE001 - WRF is optional
        print(f"WARNING: WRF initialization lookup failed: {exc}", flush=True)
        return empty
    if init_date is None:
        print("WARNING: no WRF initialization found in the lookback window", flush=True)
        return empty

    valid_locals = [init_date + timedelta(hours=hour) for hour in WRF_VALID_HOURS]
    try:
        with ThreadPoolExecutor(max_workers=WRF_DOWNLOAD_THREADS) as executor:
            contents = list(
                executor.map(
                    lambda valid_local: _request_geotiff(WRF_TEMPERATURE_LAYER, init_date, valid_local),
                    valid_locals,
                )
            )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: WRF download failed: {exc}", flush=True)
        return empty

    rows = []
    for valid_local, content in zip(valid_locals, contents):
        if content is None:
            continue
        try:
            value = _sample_geotiff(content, STATION_LON, STATION_LAT)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not read WRF GeoTIFF for {valid_local}: {exc}", flush=True)
            continue
        if np.isnan(value):
            continue
        valid_utc = pd.Timestamp(valid_local) - pd.Timedelta(hours=WRF_LOCAL_UTC_OFFSET_HOURS)
        rows.append({
            "valid_time": valid_utc,
            "wrf_temp_c": value,
            "wrf_init_time": pd.Timestamp(init_date) - pd.Timedelta(hours=WRF_LOCAL_UTC_OFFSET_HOURS),
        })

    if not rows:
        print("WARNING: WRF files found but no usable temperature values", flush=True)
        return empty
    wrf = pd.DataFrame(rows).sort_values("valid_time").drop_duplicates("valid_time")
    print(
        f"WRF init {init_date:%Y-%m-%d} local: {len(wrf)} temperature step(s) "
        f"from {wrf['valid_time'].min()} to {wrf['valid_time'].max()} UTC",
        flush=True,
    )
    return wrf[WRF_COLUMNS]


def merge_wrf_into_forecast(forecast, wrf):
    """Attach WRF temperature to ECMWF rows on exact valid time (both are 3-hourly on the UTC grid)."""
    merged = forecast.copy()
    if wrf is None or wrf.empty:
        merged["wrf_temp_c"] = np.nan
        merged["wrf_init_time"] = pd.NaT
        return merged
    wrf = wrf.copy()
    wrf["valid_time"] = pd.to_datetime(wrf["valid_time"])
    merged = merged.merge(wrf[WRF_COLUMNS], on="valid_time", how="left")
    merged.attrs = forecast.attrs
    return merged
