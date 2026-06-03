import time

import numpy as np
import pandas as pd

from forecast_config import (
    ECMWF_GRIB_PATH,
    ECMWF_PARAMS,
    ECMWF_RUN_HOUR,
    ECMWF_SOURCES,
    ECMWF_SOURCE_RETRY_SECONDS,
    ECMWF_STEPS,
    ECMWF_VARIABLE_CANDIDATES,
    FORECAST_TARGETS,
    STATION_LAT,
    STATION_LON,
)


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

    from ecmwf.opendata import Client

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
    import cfgrib

    datasets = cfgrib.open_datasets(str(ECMWF_GRIB_PATH))
    forecast = None
    found_params = set()
    try:
        for ds in datasets:
            for spec in FORECAST_TARGETS.values():
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
