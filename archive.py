import numpy as np
import pandas as pd

from calibration import climatology_for_times
from forecast_config import (
    ARCHIVE_RETENTION_DAYS,
    FORECAST_ARCHIVE_PATH,
    FORECAST_TARGETS,
    STATION_ID,
    STATION_NAME,
    VERIFICATION_RETENTION_DAYS,
)
from time_utils import parse_timestamp_column, to_utc_naive


CSV_FLOAT_FORMAT = "%.4f"

ARCHIVE_NUMERIC_COLUMNS = [
    "station_id",
    "lead_hours",
    "raw_forecast_c",
    "bias_correction_c",
    "forecast_c",
    "persistence_c",
    "climatology_c",
    "wrf_forecast_c",
    "forecast_lower_c",
    "forecast_upper_c",
    "uncertainty_c",
    "latest_observation_c",
    "latest_observation_anomaly_c",
]

VERIFICATION_COLUMNS = [
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
    "climatology_c",
    "wrf_forecast_c",
    "wrf_init_time",
    "correction_method",
    "forecast_lower_c",
    "forecast_upper_c",
    "uncertainty_c",
    "bias_source",
    "latest_observation_time",
    "latest_observation_c",
    "latest_observation_anomaly_c",
    "observed_c",
    "raw_error_c",
    "forecast_error_c",
    "persistence_error_c",
    "climatology_error_c",
    "wrf_error_c",
    "within_band",
]


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


def ensure_archive_columns(archive):
    defaults = {
        "climatology_c": np.nan,
        "wrf_forecast_c": np.nan,
        "wrf_init_time": pd.NaT,
        "latest_observation_anomaly_c": np.nan,
        "correction_method": "legacy_lead_offset",
    }
    for column, default in defaults.items():
        if column not in archive.columns:
            archive[column] = default
    return archive


def load_forecast_archive(path):
    if not path.exists():
        return pd.DataFrame()

    archive = pd.read_csv(path)
    if archive.empty:
        return archive

    for column in ["run_time", "valid_time", "latest_observation_time", "wrf_init_time"]:
        archive = parse_timestamp_column(archive, column)

    archive = ensure_archive_columns(archive)
    for column in ARCHIVE_NUMERIC_COLUMNS:
        if column in archive.columns:
            archive[column] = pd.to_numeric(archive[column], errors="coerce")

    return filter_operational_archive_rows(archive)


def backfill_archive_climatology(archive, climatology_by_target):
    """Fill climatology for archive rows written before the climatology column existed.

    Only rows that are still unverified need it, and the current trailing
    climatology is the best available estimate for those recent valid times.
    """
    if archive.empty:
        return archive

    archive = ensure_archive_columns(archive.copy())
    for target_key, climatology_by_hour in climatology_by_target.items():
        if not climatology_by_hour:
            continue
        mask = (archive["target"] == target_key) & archive["climatology_c"].isna()
        if not mask.any():
            continue
        filled = climatology_for_times(archive.loc[mask, "valid_time"], climatology_by_hour)
        archive.loc[mask, "climatology_c"] = filled.to_numpy()
    return archive


def build_forecast_archive_rows(forecast, bias_by_target, run_time):
    run_time = to_utc_naive(run_time)
    target_frames = []
    for target_key, spec in FORECAST_TARGETS.items():
        bias_info = bias_by_target[target_key]
        climatology_column = f"climatology_{target_key}_c"
        method_column = f"correction_method_{target_key}"
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
            "climatology_c": forecast[climatology_column] if climatology_column in forecast else np.nan,
            "wrf_forecast_c": forecast["wrf_temp_c"] if "wrf_temp_c" in forecast else np.nan,
            "wrf_init_time": forecast["wrf_init_time"] if "wrf_init_time" in forecast else pd.NaT,
            "correction_method": forecast[method_column] if method_column in forecast else "unknown",
            "forecast_lower_c": forecast[spec["lower_column"]],
            "forecast_upper_c": forecast[spec["upper_column"]],
            "uncertainty_c": forecast[f"uncertainty_{target_key}_c"],
            "bias_source": bias_info["bias_source"],
            "latest_observation_time": bias_info["latest_observation_time"],
            "latest_observation_c": bias_info["latest_observation_c"],
            "latest_observation_anomaly_c": bias_info.get("latest_observation_anomaly_c", np.nan),
        })
        target_frames.append(target_forecast)

    return pd.concat(target_frames, ignore_index=True)


def update_forecast_archive(existing_archive, forecast, bias_by_target, run_time, latest_observation_time):
    operational_forecast = future_forecast_rows(forecast, run_time, latest_observation_time)
    new_rows = build_forecast_archive_rows(operational_forecast, bias_by_target, run_time)
    existing_archive = ensure_archive_columns(filter_operational_archive_rows(existing_archive))
    archive = pd.concat([existing_archive, new_rows], ignore_index=True)
    archive = parse_timestamp_column(archive, "run_time")
    archive = parse_timestamp_column(archive, "valid_time")
    archive = parse_timestamp_column(archive, "latest_observation_time")
    archive = parse_timestamp_column(archive, "wrf_init_time")
    archive = filter_operational_archive_rows(archive)

    cutoff = to_utc_naive(run_time) - pd.Timedelta(days=ARCHIVE_RETENTION_DAYS)
    archive = archive[archive["run_time"] >= cutoff]
    archive = archive.drop_duplicates(["run_time", "target", "valid_time"], keep="last")
    archive = archive.sort_values(["run_time", "target", "valid_time"]).reset_index(drop=True)
    archive.to_csv(FORECAST_ARCHIVE_PATH, index=False, float_format=CSV_FLOAT_FORMAT)
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
    if archive.empty:
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)

    archive = ensure_archive_columns(filter_operational_archive_rows(archive))
    if archive.empty:
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)

    observations = station_observations_long(station_targets)
    verified = archive.merge(observations, on=["target", "valid_time"], how="inner")
    verified = verified.dropna(subset=["observed_c", "raw_forecast_c", "forecast_c"])
    if verified.empty:
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)

    verified["climatology_c"] = pd.to_numeric(verified["climatology_c"], errors="coerce")
    verified["wrf_forecast_c"] = pd.to_numeric(verified["wrf_forecast_c"], errors="coerce")
    verified["wrf_error_c"] = verified["wrf_forecast_c"] - verified["observed_c"]
    verified["raw_error_c"] = verified["raw_forecast_c"] - verified["observed_c"]
    verified["forecast_error_c"] = verified["forecast_c"] - verified["observed_c"]
    verified["persistence_error_c"] = verified["persistence_c"] - verified["observed_c"]
    verified["climatology_error_c"] = verified["climatology_c"] - verified["observed_c"]
    verified["within_band"] = (
        (verified["observed_c"] >= verified["forecast_lower_c"])
        & (verified["observed_c"] <= verified["forecast_upper_c"])
    )
    verified = verified.sort_values(["valid_time", "target", "run_time"]).reset_index(drop=True)
    return verified[VERIFICATION_COLUMNS]


def load_verification_history(path):
    """Load the cumulative verification file.

    The observation cache only ever covers about 30 days, so verification must
    accumulate across runs instead of being rebuilt from the cache each time.
    """
    if not path.exists():
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)

    history = pd.read_csv(path)
    if history.empty:
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)

    for column in ["run_time", "valid_time", "latest_observation_time", "wrf_init_time"]:
        history = parse_timestamp_column(history, column)
    for column in VERIFICATION_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan
    numeric_columns = ARCHIVE_NUMERIC_COLUMNS + [
        "observed_c",
        "raw_error_c",
        "forecast_error_c",
        "persistence_error_c",
        "climatology_error_c",
        "wrf_error_c",
    ]
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history["within_band"] = history["within_band"].astype(str).str.lower().isin(["true", "1", "1.0"])
    return history[VERIFICATION_COLUMNS]


def merge_verification_history(history, new_verification, run_time):
    frames = [frame for frame in [history, new_verification] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=VERIFICATION_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)
    merged = parse_timestamp_column(merged, "run_time")
    merged = parse_timestamp_column(merged, "valid_time")
    merged = merged.dropna(subset=["run_time", "valid_time", "observed_c", "forecast_c"])
    cutoff = to_utc_naive(run_time) - pd.Timedelta(days=VERIFICATION_RETENTION_DAYS)
    merged = merged[merged["run_time"] >= cutoff]
    merged = merged.drop_duplicates(["run_time", "target", "valid_time"], keep="last")
    merged = merged.sort_values(["valid_time", "target", "run_time"]).reset_index(drop=True)
    return merged[VERIFICATION_COLUMNS]


def save_verification_history(verification, path):
    verification.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)
