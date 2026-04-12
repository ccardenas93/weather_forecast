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
BIAS_TRAINING_DAYS = int(os.getenv("BIAS_TRAINING_DAYS", "90"))
MIN_BIAS_SAMPLES = int(os.getenv("MIN_BIAS_SAMPLES", "8"))
ARCHIVE_RETENTION_DAYS = int(os.getenv("ARCHIVE_RETENTION_DAYS", "180"))
ECMWF_RUN_HOUR = int(os.getenv("ECMWF_RUN_HOUR", "6"))
ECMWF_STEPS = [
    int(step.strip())
    for step in os.getenv("ECMWF_STEPS", "3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48").split(",")
    if step.strip()
]
ECMWF_PARAMS = [
    param.strip()
    for param in os.getenv("ECMWF_PARAMS", "mx2t3,2t,mn2t3").split(",")
    if param.strip()
]
ECMWF_SOURCES = [
    source.strip()
    for source in os.getenv("ECMWF_SOURCES", os.getenv("ECMWF_SOURCE", "azure,google,aws")).split(",")
    if source.strip()
]
if not ECMWF_SOURCES:
    ECMWF_SOURCES = ["azure", "google", "aws"]
ECMWF_SOURCE_RETRY_SECONDS = float(os.getenv("ECMWF_SOURCE_RETRY_SECONDS", "15"))

STATION_ID = 63777
STATION_NAME = "Inaquito"
STATION_LAT = -0.178300
STATION_LON = -78.487700

RAW_DATA_PATH = Path("merged_data_export.csv")
ECMWF_GRIB_PATH = Path("surface_temp_multi.grib2")
FORECAST_CSV_PATH = Path("forecast_output.csv")
FORECAST_HTML_PATH = Path("forecast_Inaquito.html")
FORECAST_ARCHIVE_PATH = Path("forecast_archive.csv")
FORECAST_VERIFICATION_PATH = Path("forecast_verification.csv")
FORECAST_METRICS_PATH = Path("forecast_metrics.csv")

FORECAST_TARGETS = {
    "max": {
        "label": "Max",
        "station_column": "TEMPERATURA AIRE MAX",
        "ecmwf_param": "mx2t3",
        "raw_column": "ecmwf_max_c",
        "forecast_column": "forecast_max_c",
        "persistence_column": "persistence_max_c",
        "lower_column": "forecast_max_lower_c",
        "upper_column": "forecast_max_upper_c",
        "bias_column": "bias_correction_max_c",
    },
    "prom": {
        "label": "Prom",
        "station_column": "TEMPERATURA AIRE PROM",
        "ecmwf_param": "2t",
        "raw_column": "ecmwf_prom_c",
        "forecast_column": "forecast_prom_c",
        "persistence_column": "persistence_prom_c",
        "lower_column": "forecast_prom_lower_c",
        "upper_column": "forecast_prom_upper_c",
        "bias_column": "bias_correction_prom_c",
    },
    "min": {
        "label": "Min",
        "station_column": "TEMPERATURA AIRE MIN",
        "ecmwf_param": "mn2t3",
        "raw_column": "ecmwf_min_c",
        "forecast_column": "forecast_min_c",
        "persistence_column": "persistence_min_c",
        "lower_column": "forecast_min_lower_c",
        "upper_column": "forecast_min_upper_c",
        "bias_column": "bias_correction_min_c",
    },
}
DEFAULT_TARGET_KEY = "prom"
ECMWF_VARIABLE_CANDIDATES = {
    "2t": ["t2m", "2t"],
    "mx2t3": ["mx2t3", "mx2t"],
    "mn2t3": ["mn2t3", "mn2t"],
}

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
    for key, spec in FORECAST_TARGETS.items():
        target = hourly[spec["station_column"]].interpolate(method="time", limit=3)
        target = target.ffill(limit=3).bfill(limit=3)
        targets[f"observed_{key}_c"] = target

    targets = pd.DataFrame(targets).dropna(how="all")
    if targets.empty:
        raise RuntimeError("No usable temperature observations for max/prom/min targets")

    return targets


def find_ecmwf_variable(ds, param):
    for variable in ECMWF_VARIABLE_CANDIDATES.get(param, [param]):
        if variable in ds.data_vars:
            return variable
    for variable, data_array in ds.data_vars.items():
        if data_array.attrs.get("GRIB_shortName") == param:
            return variable
    return None


def extract_station_series(ds, variable, value_column):
    temperature = ds[variable].sel(latitude=STATION_LAT, longitude=STATION_LON, method="nearest")
    values_c = np.asarray(temperature.values).reshape(-1) - 273.15
    valid_times = pd.to_datetime(np.asarray(ds["valid_time"].values).reshape(-1))

    series = pd.DataFrame({
        "valid_time": valid_times,
        value_column: values_c,
    }).dropna()
    series = series.sort_values("valid_time").drop_duplicates("valid_time")

    if "step" in ds.coords:
        steps = pd.to_timedelta(np.asarray(ds["step"].values).reshape(-1)).total_seconds() / 3600
        if len(steps) == len(series):
            series["lead_hours"] = steps

    return series


def retrieve_ecmwf_grib(source):
    if ECMWF_GRIB_PATH.exists():
        ECMWF_GRIB_PATH.unlink()

    client = Client(source=source)
    client.retrieve(
        time=ECMWF_RUN_HOUR,
        type="fc",
        step=ECMWF_STEPS,
        param=ECMWF_PARAMS,
        stream="oper",
        target=str(ECMWF_GRIB_PATH),
    )


def read_ecmwf_temperature_grib():
    datasets = cfgrib.open_datasets(str(ECMWF_GRIB_PATH))
    forecast = None
    found_params = set()
    try:
        for ds in datasets:
            for key, spec in FORECAST_TARGETS.items():
                param = spec["ecmwf_param"]
                if param in found_params:
                    continue
                variable = find_ecmwf_variable(ds, param)
                if not variable:
                    continue
                series = extract_station_series(ds, variable, spec["raw_column"])
                if series.empty:
                    continue

                merge_columns = ["valid_time", spec["raw_column"]]
                if "lead_hours" in series.columns:
                    merge_columns.append("lead_hours")
                series = series[merge_columns]
                if forecast is None:
                    forecast = series
                else:
                    forecast = forecast.merge(series, on="valid_time", how="outer")
                    if "lead_hours_x" in forecast.columns and "lead_hours_y" in forecast.columns:
                        forecast["lead_hours"] = forecast["lead_hours_x"].combine_first(forecast["lead_hours_y"])
                        forecast = forecast.drop(columns=["lead_hours_x", "lead_hours_y"])
                found_params.add(param)

        required_params = {spec["ecmwf_param"] for spec in FORECAST_TARGETS.values()}
        missing_params = sorted(required_params - found_params)
        if missing_params:
            raise RuntimeError(f"ECMWF forecast missing required parameters: {', '.join(missing_params)}")
        if forecast is None or forecast.empty:
            raise RuntimeError("ECMWF returned no usable temperature forecast values")

        forecast = forecast.sort_values("valid_time").drop_duplicates("valid_time")
        if "lead_hours" not in forecast.columns:
            forecast["lead_hours"] = (
                forecast["valid_time"] - forecast["valid_time"].min()
            ).dt.total_seconds() / 3600
        return forecast
    finally:
        for ds in datasets:
            ds.close()


def cleanup_ecmwf_grib():
    if ECMWF_GRIB_PATH.exists():
        try:
            ECMWF_GRIB_PATH.unlink()
        except OSError as exc:
            print(f"WARNING: Could not remove partial ECMWF GRIB {ECMWF_GRIB_PATH}: {exc}", flush=True)


def download_ecmwf_temperatures():
    last_error = None
    for source_index, source in enumerate(ECMWF_SOURCES, start=1):
        try:
            print(
                f"Trying ECMWF source {source} ({source_index}/{len(ECMWF_SOURCES)})",
                flush=True,
            )
            retrieve_ecmwf_grib(source)
            forecast = read_ecmwf_temperature_grib()
            forecast.attrs["ecmwf_source"] = source
            return forecast
        except Exception as exc:
            last_error = exc
            print(f"ECMWF source {source} failed: {exc}", flush=True)
            cleanup_ecmwf_grib()
            if source_index < len(ECMWF_SOURCES):
                print(
                    f"Retrying ECMWF from next source in {ECMWF_SOURCE_RETRY_SECONDS:.0f} seconds",
                    flush=True,
                )
                time.sleep(ECMWF_SOURCE_RETRY_SECONDS)

    source_list = ", ".join(ECMWF_SOURCES)
    raise RuntimeError(f"Failed to retrieve ECMWF forecast from sources: {source_list}") from last_error


def align_observations_to_forecast(forecast, station_targets, observed_column):
    observations = station_targets[[observed_column]].rename(columns={observed_column: "observed_temp_c"}).reset_index()
    observations = observations.rename(columns={observations.columns[0]: "valid_time"})
    return forecast.merge(observations, on="valid_time", how="left")


def to_utc_naive(timestamp):
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert(None)


def parse_timestamp_column(data, column):
    if column in data.columns:
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce").dt.tz_convert(None)
    return data


def load_forecast_archive(path):
    if not path.exists():
        return pd.DataFrame()

    archive = pd.read_csv(path)
    if archive.empty:
        return archive

    for column in ["run_time", "valid_time", "latest_observation_time"]:
        archive = parse_timestamp_column(archive, column)

    numeric_columns = [
        "station_id",
        "lead_hours",
        "raw_forecast_c",
        "bias_correction_c",
        "forecast_c",
        "persistence_c",
        "forecast_lower_c",
        "forecast_upper_c",
        "uncertainty_c",
        "latest_observation_c",
    ]
    for column in numeric_columns:
        if column in archive.columns:
            archive[column] = pd.to_numeric(archive[column], errors="coerce")

    return archive


def build_forecast_archive_rows(forecast, bias_by_target, run_time):
    run_time = to_utc_naive(run_time)
    target_frames = []
    for target_key, spec in FORECAST_TARGETS.items():
        bias_info = bias_by_target[target_key]
        target_forecast = pd.DataFrame({
            "run_time": run_time,
            "station_id": STATION_ID,
            "station_name": STATION_NAME,
            "target": target_key,
            "target_label": spec["label"],
            "ecmwf_param": spec["ecmwf_param"],
            "valid_time": forecast["valid_time"],
            "lead_hours": forecast["lead_hours"],
            "raw_forecast_c": forecast[spec["raw_column"]],
            "bias_correction_c": forecast[spec["bias_column"]],
            "forecast_c": forecast[spec["forecast_column"]],
            "persistence_c": forecast[spec["persistence_column"]],
            "forecast_lower_c": forecast[spec["lower_column"]],
            "forecast_upper_c": forecast[spec["upper_column"]],
            "uncertainty_c": forecast[f"uncertainty_{target_key}_c"],
            "bias_source": bias_info["bias_source"],
            "latest_observation_time": bias_info["latest_observation_time"],
            "latest_observation_c": bias_info["latest_observation_c"],
        })
        target_frames.append(target_forecast)

    return pd.concat(target_frames, ignore_index=True)


def update_forecast_archive(existing_archive, forecast, bias_by_target, run_time):
    new_rows = build_forecast_archive_rows(forecast, bias_by_target, run_time)
    archive = pd.concat([existing_archive, new_rows], ignore_index=True)
    archive = parse_timestamp_column(archive, "run_time")
    archive = parse_timestamp_column(archive, "valid_time")
    archive = parse_timestamp_column(archive, "latest_observation_time")

    cutoff = to_utc_naive(run_time) - pd.Timedelta(days=ARCHIVE_RETENTION_DAYS)
    archive = archive[archive["run_time"] >= cutoff]
    archive = archive.drop_duplicates(["run_time", "target", "valid_time"], keep="last")
    archive = archive.sort_values(["run_time", "target", "valid_time"]).reset_index(drop=True)
    archive.to_csv(FORECAST_ARCHIVE_PATH, index=False)
    return archive


def station_observations_long(station_targets):
    observations = []
    for target_key in FORECAST_TARGETS:
        observed_column = f"observed_{target_key}_c"
        target_observations = (
            station_targets[[observed_column]]
            .dropna()
            .rename_axis("valid_time")
            .reset_index()
            .rename(columns={observed_column: "observed_c"})
        )
        target_observations["target"] = target_key
        observations.append(target_observations[["target", "valid_time", "observed_c"]])

    return pd.concat(observations, ignore_index=True)


def verify_forecast_archive(archive, station_targets):
    columns = [
        "run_time",
        "station_id",
        "station_name",
        "target",
        "target_label",
        "ecmwf_param",
        "valid_time",
        "lead_hours",
        "raw_forecast_c",
        "bias_correction_c",
        "forecast_c",
        "persistence_c",
        "forecast_lower_c",
        "forecast_upper_c",
        "uncertainty_c",
        "bias_source",
        "latest_observation_time",
        "latest_observation_c",
        "observed_c",
        "raw_error_c",
        "forecast_error_c",
        "persistence_error_c",
        "within_band",
    ]
    if archive.empty:
        return pd.DataFrame(columns=columns)

    observations = station_observations_long(station_targets)
    verified = archive.merge(observations, on=["target", "valid_time"], how="inner")
    verified = verified.dropna(subset=["observed_c", "raw_forecast_c", "forecast_c"])
    if verified.empty:
        return pd.DataFrame(columns=columns)

    verified["raw_error_c"] = verified["raw_forecast_c"] - verified["observed_c"]
    verified["forecast_error_c"] = verified["forecast_c"] - verified["observed_c"]
    verified["persistence_error_c"] = verified["persistence_c"] - verified["observed_c"]
    verified["within_band"] = (
        (verified["observed_c"] >= verified["forecast_lower_c"])
        & (verified["observed_c"] <= verified["forecast_upper_c"])
    )
    verified = verified.sort_values(["valid_time", "target", "run_time"]).reset_index(drop=True)
    return verified[columns]


def summarize_verification_metrics(verified):
    columns = [
        "target",
        "target_label",
        "lead_hour",
        "sample_count",
        "forecast_bias_c",
        "forecast_mae_c",
        "forecast_rmse_c",
        "raw_bias_c",
        "raw_mae_c",
        "raw_rmse_c",
        "persistence_mae_c",
        "coverage_rate",
        "mae_improvement_vs_raw_c",
        "mae_improvement_vs_persistence_c",
    ]
    if verified.empty:
        return pd.DataFrame(columns=columns)

    scored = verified.dropna(subset=["lead_hours", "forecast_error_c", "raw_error_c"]).copy()
    if scored.empty:
        return pd.DataFrame(columns=columns)

    scored["lead_hour"] = scored["lead_hours"].round().astype(int)
    metrics = scored.groupby(["target", "target_label", "lead_hour"]).agg(
        sample_count=("forecast_error_c", "count"),
        forecast_bias_c=("forecast_error_c", "mean"),
        forecast_mae_c=("forecast_error_c", lambda errors: errors.abs().mean()),
        forecast_rmse_c=("forecast_error_c", lambda errors: np.sqrt(np.mean(np.square(errors)))),
        raw_bias_c=("raw_error_c", "mean"),
        raw_mae_c=("raw_error_c", lambda errors: errors.abs().mean()),
        raw_rmse_c=("raw_error_c", lambda errors: np.sqrt(np.mean(np.square(errors)))),
        persistence_mae_c=("persistence_error_c", lambda errors: errors.abs().mean()),
        coverage_rate=("within_band", "mean"),
    ).reset_index()
    metrics["mae_improvement_vs_raw_c"] = metrics["raw_mae_c"] - metrics["forecast_mae_c"]
    metrics["mae_improvement_vs_persistence_c"] = (
        metrics["persistence_mae_c"] - metrics["forecast_mae_c"]
    )
    return metrics[columns].sort_values(["target", "lead_hour"]).reset_index(drop=True)


def historical_bias_profile(verification, target_key, run_time):
    if verification.empty:
        return {}, {}, 0

    history = verification[verification["target"] == target_key].copy()
    if history.empty:
        return {}, {}, 0

    cutoff = to_utc_naive(run_time) - pd.Timedelta(days=BIAS_TRAINING_DAYS)
    history = parse_timestamp_column(history, "run_time")
    history = history[history["run_time"] >= cutoff]
    history = history.dropna(subset=["lead_hours", "raw_error_c", "forecast_error_c"])
    if history.empty:
        return {}, {}, 0

    history["lead_hour"] = history["lead_hours"].round().astype(int)
    lead_bias = {}
    lead_uncertainty = {}
    for lead_hour, group in history.groupby("lead_hour"):
        if len(group) < MIN_BIAS_SAMPLES:
            continue

        correction_c = float((-group["raw_error_c"]).median())
        forecast_abs_error = group["forecast_error_c"].abs()
        uncertainty_c = float(max(1.2, forecast_abs_error.quantile(0.80)))
        lead_bias[int(lead_hour)] = correction_c
        lead_uncertainty[int(lead_hour)] = uncertainty_c

    return lead_bias, lead_uncertainty, len(history)


def compute_local_bias(forecast, station_targets, target_key, verification=None, run_time=None):
    spec = FORECAST_TARGETS[target_key]
    observed_column = f"observed_{target_key}_c"
    target = station_targets[observed_column].dropna()
    latest_observation_time = target.index.max()
    latest_observation_c = float(target.loc[latest_observation_time])

    overlap = align_observations_to_forecast(forecast, station_targets, observed_column)
    overlap = overlap[overlap["valid_time"] <= latest_observation_time].dropna(subset=["observed_temp_c"])

    if not overlap.empty:
        bias_samples = overlap["observed_temp_c"] - overlap[spec["raw_column"]]
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
            bias_c = latest_observation_c - float(nearest_forecast[spec["raw_column"]])
            source = f"latest observation vs nearest ECMWF valid time ({hours_apart:.1f} h apart)"
        else:
            bias_c = 0.0
            source = "no recent overlap; raw ECMWF used"
        raw_mae_c = np.nan
        raw_rmse_c = np.nan
        uncertainty_c = 2.0

    if run_time is None:
        run_time = pd.Timestamp.now(tz="UTC")
    if verification is None:
        verification = pd.DataFrame()
    lead_bias, lead_uncertainty, historical_sample_count = historical_bias_profile(
        verification,
        target_key,
        run_time,
    )
    if lead_bias:
        source = (
            f"historical lead-time bias for {len(lead_bias)} lead(s); "
            f"current fallback from {source}"
        )

    return {
        "bias_c": bias_c,
        "bias_source": source,
        "raw_mae_c": raw_mae_c,
        "raw_rmse_c": raw_rmse_c,
        "uncertainty_c": uncertainty_c,
        "lead_bias_c": lead_bias,
        "lead_uncertainty_c": lead_uncertainty,
        "historical_sample_count": historical_sample_count,
        "latest_observation_time": latest_observation_time,
        "latest_observation_c": latest_observation_c,
    }


def apply_bias_correction(forecast, bias_by_target):
    corrected = forecast.copy()
    for target_key, bias_info in bias_by_target.items():
        spec = FORECAST_TARGETS[target_key]
        hours_after_latest_obs = (
            corrected["valid_time"] - bias_info["latest_observation_time"]
        ).dt.total_seconds() / 3600
        hours_after_latest_obs = hours_after_latest_obs.clip(lower=0)

        decay = np.exp(-hours_after_latest_obs / BIAS_DECAY_HOURS)
        fallback_bias = bias_info["bias_c"] * decay
        lead_bias = corrected["lead_hours"].round().map(
            lambda lead_hour: bias_info["lead_bias_c"].get(int(lead_hour))
            if pd.notna(lead_hour)
            else np.nan
        )
        lead_bias = pd.to_numeric(lead_bias, errors="coerce")
        corrected[spec["bias_column"]] = lead_bias.where(lead_bias.notna(), fallback_bias)
        corrected[spec["forecast_column"]] = corrected[spec["raw_column"]] + corrected[spec["bias_column"]]
        corrected[spec["persistence_column"]] = bias_info["latest_observation_c"]
        lead_uncertainty = corrected["lead_hours"].round().map(
            lambda lead_hour: bias_info["lead_uncertainty_c"].get(int(lead_hour))
            if pd.notna(lead_hour)
            else np.nan
        )
        uncertainty_column = f"uncertainty_{target_key}_c"
        lead_uncertainty = pd.to_numeric(lead_uncertainty, errors="coerce")
        corrected[uncertainty_column] = lead_uncertainty.where(
            lead_uncertainty.notna(),
            bias_info["uncertainty_c"],
        )
        corrected[spec["lower_column"]] = corrected[spec["forecast_column"]] - corrected[uncertainty_column]
        corrected[spec["upper_column"]] = corrected[spec["forecast_column"]] + corrected[uncertainty_column]
    return corrected


def diagnostic_text(bias_info):
    if pd.notna(bias_info["raw_rmse_c"]):
        rmse_text = f"{bias_info['raw_rmse_c']:.2f} C"
    else:
        rmse_text = "n/a"
    history_text = ""
    if bias_info["historical_sample_count"]:
        history_text = (
            f" Verified archive samples: {bias_info['historical_sample_count']}; "
            f"calibrated leads: {len(bias_info['lead_bias_c'])}."
        )
    return (
        f"Bias correction: {bias_info['bias_c']:.2f} C ({bias_info['bias_source']}). "
        f"Raw ECMWF recent RMSE: {rmse_text}.{history_text}"
    )


def target_title(target_key):
    spec = FORECAST_TARGETS[target_key]
    return (
        f"{STATION_NAME} {spec['label']} Temperature Forecast: "
        "ECMWF with Local Bias Correction"
    )


def target_annotations(target_key, bias_info):
    latest_observation_time = bias_info["latest_observation_time"].to_pydatetime()
    return [
        dict(
            text="latest observation",
            x=latest_observation_time,
            y=1,
            xref="x",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        ),
        dict(
            text=diagnostic_text(bias_info),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.18,
            showarrow=False,
            align="left",
        ),
    ]


def build_plot(station_targets, forecast, bias_by_target):
    target_order = ["max", "prom", "min"]
    default_index = target_order.index(DEFAULT_TARGET_KEY)
    fig = go.Figure()

    for target_index, target_key in enumerate(target_order):
        spec = FORECAST_TARGETS[target_key]
        bias_info = bias_by_target[target_key]
        observed_column = f"observed_{target_key}_c"
        observed = station_targets[observed_column].dropna()
        future = forecast[forecast["valid_time"] > bias_info["latest_observation_time"]].copy()
        if future.empty:
            future = forecast.copy()

        visible = target_index == default_index
        plot_values = pd.concat([
            observed.iloc[-120:],
            forecast[spec["raw_column"]],
            future[spec["lower_column"]],
            future[spec["upper_column"]],
            future[spec["forecast_column"]],
        ]).dropna()
        if plot_values.empty:
            marker_min, marker_max = 0, 1
        else:
            marker_min, marker_max = float(plot_values.min()), float(plot_values.max())
        latest_observation_time = bias_info["latest_observation_time"].to_pydatetime()

        fig.add_trace(go.Scatter(
            x=observed.index[-120:],
            y=observed.iloc[-120:],
            mode="lines",
            name=f"Observed {spec['label']}",
            visible=visible,
            line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=forecast["valid_time"],
            y=forecast[spec["raw_column"]],
            mode="lines+markers",
            name=f"Raw ECMWF {spec['label']}",
            visible=visible,
            line=dict(color="#2ca02c", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["upper_column"]],
            mode="lines",
            name=f"{spec['label']} upper band",
            visible=visible,
            showlegend=False,
            line=dict(color="rgba(214, 39, 40, 0.15)"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["lower_column"]],
            mode="lines",
            name=f"{spec['label']} uncertainty band",
            visible=visible,
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.15)",
            line=dict(color="rgba(214, 39, 40, 0.15)"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["forecast_column"]],
            mode="lines+markers",
            name=f"Bias-corrected {spec['label']}",
            visible=visible,
            line=dict(color="#d62728"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["persistence_column"]],
            mode="lines",
            name=f"{spec['label']} persistence",
            visible=visible,
            line=dict(color="#7f7f7f", dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[latest_observation_time, latest_observation_time],
            y=[marker_min, marker_max],
            mode="lines",
            name="latest observation",
            visible=visible,
            showlegend=False,
            line=dict(color="#444444", dash="dot"),
        ))

    buttons = []
    traces_per_target = 7
    for target_index, target_key in enumerate(target_order):
        visible = [False] * len(fig.data)
        start = target_index * traces_per_target
        for trace_index in range(start, start + traces_per_target):
            visible[trace_index] = True
        buttons.append({
            "label": FORECAST_TARGETS[target_key]["label"],
            "method": "update",
            "args": [
                {"visible": visible},
                {
                    "title": target_title(target_key),
                    "annotations": target_annotations(target_key, bias_by_target[target_key]),
                },
            ],
        })

    fig.update_layout(
        title=target_title(DEFAULT_TARGET_KEY),
        xaxis_title="UTC time",
        yaxis_title="Temperature (C)",
        legend=dict(x=0.01, y=0.99),
        annotations=target_annotations(DEFAULT_TARGET_KEY, bias_by_target[DEFAULT_TARGET_KEY]),
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "right",
                "showactive": True,
                "active": default_index,
                "x": 0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        margin=dict(b=110),
    )
    fig.write_html(str(FORECAST_HTML_PATH))


def main():
    run_time = pd.Timestamp.now(tz="UTC")
    end_date = os.getenv("END_DATE", run_time.strftime("%Y-%m-%dT%H:%M:%S"))
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

    station_targets = load_station_targets(RAW_DATA_PATH)
    latest_observation_time = max(
        station_targets[f"observed_{target_key}_c"].dropna().index.max()
        for target_key in FORECAST_TARGETS
    )
    print(
        f"Loaded {len(station_targets)} usable hourly temperature observations; "
        f"latest at {latest_observation_time} UTC",
        flush=True,
    )

    forecast_archive = load_forecast_archive(FORECAST_ARCHIVE_PATH)
    historical_verification = verify_forecast_archive(forecast_archive, station_targets)
    print(
        f"Loaded {len(forecast_archive)} archived forecast row(s); "
        f"{len(historical_verification)} verified row(s) available for calibration",
        flush=True,
    )

    print(
        f"Retrieving ECMWF max/prom/min 2m temperature forecast for {STATION_NAME} "
        f"from sources {', '.join(ECMWF_SOURCES)}",
        flush=True,
    )
    raw_forecast = download_ecmwf_temperatures()
    print(f"Using ECMWF source {raw_forecast.attrs.get('ecmwf_source')}", flush=True)
    bias_by_target = {
        target_key: compute_local_bias(
            raw_forecast,
            station_targets,
            target_key,
            verification=historical_verification,
            run_time=run_time,
        )
        for target_key in FORECAST_TARGETS
    }
    corrected_forecast = apply_bias_correction(raw_forecast, bias_by_target)
    corrected_forecast.to_csv(FORECAST_CSV_PATH, index=False)
    build_plot(station_targets, corrected_forecast, bias_by_target)

    forecast_archive = update_forecast_archive(
        forecast_archive,
        corrected_forecast,
        bias_by_target,
        run_time,
    )
    verification = verify_forecast_archive(forecast_archive, station_targets)
    metrics = summarize_verification_metrics(verification)
    verification.to_csv(FORECAST_VERIFICATION_PATH, index=False)
    metrics.to_csv(FORECAST_METRICS_PATH, index=False)

    bias_summary = ", ".join(
        f"{FORECAST_TARGETS[target_key]['label']}: {bias_info['bias_c']:.2f} C"
        for target_key, bias_info in bias_by_target.items()
    )
    print(
        f"Saved {FORECAST_CSV_PATH} and {FORECAST_HTML_PATH}; "
        f"archived {len(forecast_archive)} row(s), verified {len(verification)} row(s), "
        f"metrics {len(metrics)} row(s); local bias corrections {bias_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
