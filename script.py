import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from ecmwf.opendata import Client
import cfgrib


FETCH_DAYS = int(os.getenv("FETCH_DAYS", "730"))
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "3"))
FETCH_RETRY_SECONDS = float(os.getenv("FETCH_RETRY_SECONDS", "20"))
ALLOW_STALE_OBS = os.getenv("ALLOW_STALE_OBS", "1").lower() not in {"0", "false", "no"}
BIAS_DECAY_HOURS = float(os.getenv("BIAS_DECAY_HOURS", "36"))
ECMWF_RUN_HOUR = int(os.getenv("ECMWF_RUN_HOUR", "6"))
ECMWF_STEPS = [
    int(step.strip())
    for step in os.getenv("ECMWF_STEPS", "0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48").split(",")
    if step.strip()
]

STATION_ID = 63777
STATION_NAME = "Inaquito"
STATION_LAT = -0.178300
STATION_LON = -78.487700
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "TEMPERATURA AIRE PROM")

RAW_DATA_PATH = Path("merged_data_export.csv")
ECMWF_GRIB_PATH = Path("surface_temp_multi.grib2")
FORECAST_CSV_PATH = Path("forecast_output.csv")
FORECAST_HTML_PATH = Path("forecast_Inaquito.html")

TABLE_NAMES = [
    "009010101h", "009010201h", "009010401h",  # HUMEDAD RELATIVA DEL AIRE (MAX, MIN, PROM)
    "017140801h",  # PRECIPITACION (SUM)
    "018070201h", "018070401h", "018070101h",  # PRESION ATMOSFERICA (MIN, PROM, MAX)
    "021200201h", "021200101h", "021200401h", "021200801h",  # RADIACION SOLAR GLOBAL (MIN, MAX, PROM, SUM)
    "022200201h", "022200401h", "022200101h",  # RADIACION SOLAR REFLEJADA (MIN, PROM, MAX)
    "029030101h", "029030401h", "029030201h",  # TEMPERATURA AIRE (MAX, PROM, MIN)
]

NAME_MAP = {
    "009010101h": "HUMEDAD RELATIVA DEL AIRE MAX",
    "009010201h": "HUMEDAD RELATIVA DEL AIRE MIN",
    "009010401h": "HUMEDAD RELATIVA DEL AIRE PROM",
    "017140801h": "PRECIPITACION SUM",
    "018070201h": "PRESION ATMOSFERICA MIN",
    "018070401h": "PRESION ATMOSFERICA PROM",
    "018070101h": "PRESION ATMOSFERICA MAX",
    "021200201h": "RADIACION SOLAR GLOBAL MIN",
    "021200101h": "RADIACION SOLAR GLOBAL MAX",
    "021200401h": "RADIACION SOLAR GLOBAL PROM",
    "021200801h": "RADIACION SOLAR GLOBAL SUM",
    "022200201h": "RADIACION SOLAR REFLEJADA MIN",
    "022200401h": "RADIACION SOLAR REFLEJADA MAX",
    "022200101h": "RADIACION SOLAR REFLEJADA PROM",
    "029030101h": "TEMPERATURA AIRE MAX",
    "029030401h": "TEMPERATURA AIRE PROM",
    "029030201h": "TEMPERATURA AIRE MIN",
}


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


def load_station_target(raw_data_path):
    data = pd.read_csv(raw_data_path)
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    if TARGET_COLUMN not in data.columns:
        raise RuntimeError(f"Missing required target column: {TARGET_COLUMN}")

    data["fecha"] = pd.to_datetime(data["fecha"], utc=True).dt.tz_convert(None)
    data = data.sort_values("fecha").set_index("fecha")
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    hourly = data.asfreq("h")
    target = hourly[TARGET_COLUMN].interpolate(method="time", limit=3)
    target = target.ffill(limit=3).bfill(limit=3).dropna()
    if target.empty:
        raise RuntimeError(f"No usable observations for {TARGET_COLUMN}")

    return target


def download_ecmwf_temperature():
    client = Client(source="ecmwf")
    client.retrieve(
        time=ECMWF_RUN_HOUR,
        type="fc",
        step=ECMWF_STEPS,
        param="2t",
        stream="oper",
        target=str(ECMWF_GRIB_PATH),
    )

    ds = cfgrib.open_dataset(str(ECMWF_GRIB_PATH))
    try:
        temperature = ds["t2m"].sel(latitude=STATION_LAT, longitude=STATION_LON, method="nearest")
        values_c = np.asarray(temperature.values).reshape(-1) - 273.15
        valid_times = pd.to_datetime(np.asarray(ds["valid_time"].values).reshape(-1))

        forecast = pd.DataFrame({
            "valid_time": valid_times,
            "ecmwf_2t_c": values_c,
        }).dropna()
        forecast = forecast.sort_values("valid_time").drop_duplicates("valid_time")

        if "step" in ds.coords:
            steps = pd.to_timedelta(np.asarray(ds["step"].values).reshape(-1)).total_seconds() / 3600
            if len(steps) == len(forecast):
                forecast["lead_hours"] = steps
            else:
                forecast["lead_hours"] = (
                    forecast["valid_time"] - forecast["valid_time"].min()
                ).dt.total_seconds() / 3600
        else:
            forecast["lead_hours"] = (
                forecast["valid_time"] - forecast["valid_time"].min()
            ).dt.total_seconds() / 3600

        if forecast.empty:
            raise RuntimeError("ECMWF returned no usable 2t forecast values")
        return forecast
    finally:
        ds.close()


def align_observations_to_forecast(forecast, target):
    observations = target.rename("observed_temp_c").reset_index()
    observations = observations.rename(columns={observations.columns[0]: "valid_time"})
    return forecast.merge(observations, on="valid_time", how="left")


def compute_local_bias(forecast, target):
    latest_observation_time = target.index.max()
    latest_observation_c = float(target.loc[latest_observation_time])

    overlap = align_observations_to_forecast(forecast, target)
    overlap = overlap[overlap["valid_time"] <= latest_observation_time].dropna(subset=["observed_temp_c"])

    if not overlap.empty:
        bias_samples = overlap["observed_temp_c"] - overlap["ecmwf_2t_c"]
        bias_c = float(bias_samples.median())
        raw_mae_c = float(bias_samples.abs().mean())
        raw_rmse_c = float(np.sqrt(np.mean(np.square(bias_samples))))
        uncertainty_c = max(1.5, raw_mae_c)
        source = f"median of {len(overlap)} current-run overlap point(s)"
    else:
        nearest_index = (forecast["valid_time"] - latest_observation_time).abs().idxmin()
        nearest_forecast = forecast.loc[nearest_index]
        hours_apart = abs((nearest_forecast["valid_time"] - latest_observation_time).total_seconds()) / 3600
        if hours_apart <= 6:
            bias_c = latest_observation_c - float(nearest_forecast["ecmwf_2t_c"])
            source = f"latest observation vs nearest ECMWF valid time ({hours_apart:.1f} h apart)"
        else:
            bias_c = 0.0
            source = "no recent overlap; raw ECMWF used"
        raw_mae_c = np.nan
        raw_rmse_c = np.nan
        uncertainty_c = 2.0

    return {
        "bias_c": bias_c,
        "bias_source": source,
        "raw_mae_c": raw_mae_c,
        "raw_rmse_c": raw_rmse_c,
        "uncertainty_c": uncertainty_c,
        "latest_observation_time": latest_observation_time,
        "latest_observation_c": latest_observation_c,
    }


def apply_bias_correction(forecast, bias_info):
    corrected = forecast.copy()
    hours_after_latest_obs = (
        corrected["valid_time"] - bias_info["latest_observation_time"]
    ).dt.total_seconds() / 3600
    hours_after_latest_obs = hours_after_latest_obs.clip(lower=0)

    decay = np.exp(-hours_after_latest_obs / BIAS_DECAY_HOURS)
    corrected["bias_correction_c"] = bias_info["bias_c"] * decay
    corrected["forecast_temp_c"] = corrected["ecmwf_2t_c"] + corrected["bias_correction_c"]
    corrected["persistence_temp_c"] = bias_info["latest_observation_c"]
    corrected["uncertainty_c"] = bias_info["uncertainty_c"]
    corrected["forecast_lower_c"] = corrected["forecast_temp_c"] - corrected["uncertainty_c"]
    corrected["forecast_upper_c"] = corrected["forecast_temp_c"] + corrected["uncertainty_c"]
    return corrected


def build_plot(target, forecast, bias_info):
    future = forecast[forecast["valid_time"] > bias_info["latest_observation_time"]].copy()
    if future.empty:
        future = forecast.copy()

    if pd.notna(bias_info["raw_rmse_c"]):
        rmse_text = f"{bias_info['raw_rmse_c']:.2f} C"
    else:
        rmse_text = "n/a"
    diagnostic = (
        f"Bias correction: {bias_info['bias_c']:.2f} C ({bias_info['bias_source']}). "
        f"Raw ECMWF recent RMSE: {rmse_text}"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=target.index[-120:],
        y=target.iloc[-120:],
        mode="lines",
        name=f"Observed {TARGET_COLUMN}",
        line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast["valid_time"],
        y=forecast["ecmwf_2t_c"],
        mode="lines+markers",
        name="Raw ECMWF 2m temperature",
        line=dict(color="#2ca02c", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future["valid_time"],
        y=future["forecast_upper_c"],
        mode="lines",
        name="Forecast upper band",
        showlegend=False,
        line=dict(color="rgba(214, 39, 40, 0.15)"),
    ))
    fig.add_trace(go.Scatter(
        x=future["valid_time"],
        y=future["forecast_lower_c"],
        mode="lines",
        name="Forecast uncertainty band",
        fill="tonexty",
        fillcolor="rgba(214, 39, 40, 0.15)",
        line=dict(color="rgba(214, 39, 40, 0.15)"),
    ))
    fig.add_trace(go.Scatter(
        x=future["valid_time"],
        y=future["forecast_temp_c"],
        mode="lines+markers",
        name="Bias-corrected ECMWF forecast",
        line=dict(color="#d62728"),
    ))
    fig.add_trace(go.Scatter(
        x=future["valid_time"],
        y=future["persistence_temp_c"],
        mode="lines",
        name="Persistence baseline",
        line=dict(color="#7f7f7f", dash="dot"),
    ))

    fig.add_vline(
        x=bias_info["latest_observation_time"],
        line_dash="dot",
        line_color="#444444",
        annotation_text="latest observation",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"{STATION_NAME} Temperature Forecast: ECMWF with Local Bias Correction",
        xaxis_title="UTC time",
        yaxis_title="Temperature (C)",
        legend=dict(x=0.01, y=0.99),
        annotations=[
            dict(
                text=diagnostic,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.18,
                showarrow=False,
                align="left",
            )
        ],
        margin=dict(b=110),
    )
    fig.write_html(str(FORECAST_HTML_PATH))


def main():
    end_timestamp = pd.Timestamp.now(tz="UTC")
    end_date = os.getenv("END_DATE", end_timestamp.strftime("%Y-%m-%dT%H:%M:%S"))
    start_date = os.getenv(
        "START_DATE",
        (pd.Timestamp(end_date) - pd.Timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%dT%H:%M:%S"),
    )

    print(f"Fetching INAMHI observations from {start_date} to {end_date}", flush=True)
    try:
        raw_weather = fetch_weather_data(start_date, end_date, STATION_ID, TABLE_NAMES)
        raw_weather.to_csv(RAW_DATA_PATH, index=False)
    except RuntimeError as exc:
        if not ALLOW_STALE_OBS or not RAW_DATA_PATH.exists():
            raise
        print(
            f"WARNING: {exc}. Using cached observations from {RAW_DATA_PATH} so the forecast can continue.",
            flush=True,
        )

    target = load_station_target(RAW_DATA_PATH)
    print(
        f"Loaded {len(target)} usable hourly {TARGET_COLUMN} observations; "
        f"latest at {target.index.max()} UTC",
        flush=True,
    )

    print(f"Retrieving ECMWF 2t forecast for {STATION_NAME}", flush=True)
    raw_forecast = download_ecmwf_temperature()
    bias_info = compute_local_bias(raw_forecast, target)
    corrected_forecast = apply_bias_correction(raw_forecast, bias_info)
    corrected_forecast.to_csv(FORECAST_CSV_PATH, index=False)
    build_plot(target, corrected_forecast, bias_info)

    print(
        f"Saved {FORECAST_CSV_PATH} and {FORECAST_HTML_PATH}; "
        f"local bias correction {bias_info['bias_c']:.2f} C",
        flush=True,
    )


if __name__ == "__main__":
    main()
