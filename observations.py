import time

import pandas as pd
import requests

from forecast_config import FETCH_RETRIES, FETCH_RETRY_SECONDS, FORECAST_TARGETS, NAME_MAP


def fetch_weather_data(start_date, end_date, station_id, table_names):
    """Retrieve hourly observations from the INAMHI API."""

    url = "https://inamhi.gob.ec/api_rest/station_data_hour/get_data_hour/"
    payload = {
        "id_estacion": station_id,
        "table_names": table_names,
        "id_aplication": 1,
        "start_date": start_date,
        "end_date": end_date,
    }
    last_error = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code != 200:
                raise RuntimeError(f"INAMHI returned HTTP {response.status_code}")

            flattened_data = []
            for measurement in response.json():
                code = measurement.get("nemonico")
                variable_name = NAME_MAP.get(code, code)
                for entry in measurement.get("data", []):
                    flattened_data.append({
                        "fecha": entry["fecha_toma_dato"],
                        variable_name: entry["valor"],
                    })

            if not flattened_data:
                raise RuntimeError(f"No weather data returned from {start_date} to {end_date}")

            return pd.DataFrame(flattened_data).groupby("fecha").first().reset_index()
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < FETCH_RETRIES:
                print(
                    f"INAMHI fetch attempt {attempt}/{FETCH_RETRIES} failed: {exc}; retrying",
                    flush=True,
                )
                time.sleep(FETCH_RETRY_SECONDS)

    raise RuntimeError(f"Failed to retrieve INAMHI data after {FETCH_RETRIES} attempt(s): {last_error}")


def load_station_targets(raw_data_path):
    data = pd.read_csv(raw_data_path)
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    missing_columns = [
        spec["station_column"]
        for spec in FORECAST_TARGETS.values()
        if spec["station_column"] not in data.columns
    ]
    if missing_columns:
        raise RuntimeError(f"Missing required target columns: {', '.join(missing_columns)}")

    data["fecha"] = pd.to_datetime(data["fecha"], utc=True).dt.tz_convert(None)
    data = data.sort_values("fecha").set_index("fecha")
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    hourly = data.asfreq("h")
    targets = {}
    targets["observed_max_c"] = hourly[FORECAST_TARGETS["max"]["station_column"]].rolling(
        "3h",
        min_periods=3,
    ).max()
    targets["observed_prom_c"] = hourly[FORECAST_TARGETS["prom"]["station_column"]]
    targets["observed_min_c"] = hourly[FORECAST_TARGETS["min"]["station_column"]].rolling(
        "3h",
        min_periods=3,
    ).min()

    targets = pd.DataFrame(targets).dropna(how="all")
    if targets.empty:
        raise RuntimeError("No usable temperature observations for max/prom/min targets")

    return targets
