import pandas as pd

from forecast_config import (
    ARCHIVE_RETENTION_DAYS,
    FORECAST_ARCHIVE_PATH,
    FORECAST_TARGETS,
    STATION_ID,
    STATION_NAME,
)
from time_utils import parse_timestamp_column, to_utc_naive


def filter_operational_archive_rows(archive):
    if archive.empty or not {"run_time", "valid_time"}.issubset(archive.columns):
        return archive

    archive = parse_timestamp_column(archive.copy(), "run_time")
    archive = parse_timestamp_column(archive, "valid_time")
    archive = archive.dropna(subset=["run_time", "valid_time"])
    return archive[archive["valid_time"] > archive["run_time"]].copy()


def future_forecast_rows(forecast, run_time, latest_observation_time):
    cutoff = max(to_utc_naive(run_time), to_utc_naive(latest_observation_time))
    future = forecast[forecast["valid_time"] > cutoff].copy()
    if future.empty:
        raise RuntimeError(
            "ECMWF returned no forecast rows later than both the run time and latest observation"
        )
    return future


def mark_operational_forecast_rows(forecast, run_time, latest_observation_time):
    marked = forecast.copy()
    cutoff = max(to_utc_naive(run_time), to_utc_naive(latest_observation_time))
    marked["is_operational_forecast"] = marked["valid_time"] > cutoff
    return marked


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

    return filter_operational_archive_rows(archive)


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


def update_forecast_archive(existing_archive, forecast, bias_by_target, run_time, latest_observation_time):
    operational_forecast = future_forecast_rows(forecast, run_time, latest_observation_time)
    new_rows = build_forecast_archive_rows(operational_forecast, bias_by_target, run_time)
    archive = pd.concat([filter_operational_archive_rows(existing_archive), new_rows], ignore_index=True)
    archive = parse_timestamp_column(archive, "run_time")
    archive = parse_timestamp_column(archive, "valid_time")
    archive = parse_timestamp_column(archive, "latest_observation_time")
    archive = filter_operational_archive_rows(archive)

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

    archive = filter_operational_archive_rows(archive)
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
