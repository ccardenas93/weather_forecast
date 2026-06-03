import numpy as np
import pandas as pd

from forecast_config import BIAS_DECAY_HOURS, BIAS_TRAINING_DAYS, FORECAST_TARGETS, MIN_BIAS_SAMPLES
from time_utils import parse_timestamp_column, to_utc_naive


def align_observations_to_forecast(forecast, station_targets, observed_column):
    observations = station_targets[[observed_column]].rename(columns={observed_column: "observed_temp_c"}).reset_index()
    observations = observations.rename(columns={observations.columns[0]: "valid_time"})
    return forecast.merge(observations, on="valid_time", how="left")


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
