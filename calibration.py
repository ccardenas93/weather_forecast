import numpy as np
import pandas as pd

from forecast_config import (
    BIAS_DECAY_HOURS,
    BIAS_TRAINING_DAYS,
    CLIMATOLOGY_DAYS,
    FORECAST_TARGETS,
    MIN_BIAS_SAMPLES,
    MIN_CLIMATOLOGY_SAMPLES,
    MIN_MOS_SAMPLES,
    MOS_TRAINING_DAYS,
    MOS_UNCERTAINTY_INFLATION,
)
from time_utils import parse_timestamp_column, to_utc_naive


METHOD_MOS_WRF = "mos_wrf"
METHOD_MOS = "mos"
METHOD_MOS_ANOMALY = "mos_anomaly"
METHOD_MOS_BASIC = "mos_basic"
METHOD_HOUR_OFFSET = "hour_offset"
METHOD_OVERLAP_FALLBACK = "overlap_fallback"
MIN_UNCERTAINTY_C = 1.2
COEFFICIENT_BOUNDS = (-1.0, 2.0)

# Predictor sets tried in order. Each regression is fitted per UTC hour of the
# valid time: observed = intercept + sum(coef * feature).
#   raw_forecast_c      raw ECMWF value for the target (mx2t3, 2t, mn2t3)
#   climatology_c       trailing 30-day mean observation for that hour of day
#   anomaly_feature     latest observed anomaly vs climatology, decayed with lead
#   wrf_forecast_c      INAMHI WRF 2 m temperature at the same valid time
MOS_FEATURE_SETS = [
    (METHOD_MOS_WRF, ["raw_forecast_c", "climatology_c", "anomaly_feature", "wrf_forecast_c"]),
    (METHOD_MOS, ["raw_forecast_c", "climatology_c", "anomaly_feature"]),
    # Climatology at a fixed hour barely moves inside the training window, so it can
    # become collinear with the intercept and produce unphysical coefficients that
    # fail the bounds check. Dropping it keeps the anomaly predictor in play.
    (METHOD_MOS_ANOMALY, ["raw_forecast_c", "anomaly_feature"]),
    (METHOD_MOS_BASIC, ["raw_forecast_c", "climatology_c"]),
]


def align_observations_to_forecast(forecast, station_targets, observed_column):
    observations = station_targets[[observed_column]].rename(columns={observed_column: "observed_temp_c"}).reset_index()
    observations = observations.rename(columns={observations.columns[0]: "valid_time"})
    return forecast.merge(observations, on="valid_time", how="left")


def diurnal_climatology(station_targets, target_key, run_time, days=CLIMATOLOGY_DAYS):
    """Mean observed value by UTC hour over the trailing window that ends at run_time.

    This is the no-skill reference for an equatorial station: most of the
    temperature signal is the diurnal cycle, so any model has to beat it.
    """
    observed_column = f"observed_{target_key}_c"
    if observed_column not in station_targets.columns:
        return {}
    series = station_targets[observed_column].dropna()
    if series.empty:
        return {}

    end = to_utc_naive(run_time)
    start = end - pd.Timedelta(days=days)
    window = series[(series.index > start) & (series.index <= end)]
    if window.empty:
        return {}

    grouped = window.groupby(window.index.hour)
    counts = grouped.count()
    means = grouped.mean()
    return {
        int(hour): float(means[hour])
        for hour in means.index
        if counts[hour] >= MIN_CLIMATOLOGY_SAMPLES
    }


def climatology_for_times(valid_times, climatology_by_hour):
    valid_times = pd.Series(pd.to_datetime(valid_times))
    hours = valid_times.dt.hour
    return pd.to_numeric(
        hours.map(lambda hour: climatology_by_hour.get(int(hour), np.nan) if pd.notna(hour) else np.nan),
        errors="coerce",
    )


def anomaly_decay(valid_times, latest_observation_time):
    hours_since_obs = (
        pd.Series(pd.to_datetime(valid_times)) - pd.Timestamp(to_utc_naive(latest_observation_time))
    ).dt.total_seconds() / 3600
    return np.exp(-hours_since_obs.clip(lower=0) / BIAS_DECAY_HOURS)


def independent_history(history):
    """Keep one row per target, valid time and ECMWF cycle.

    The workflow reruns hourly on the same 06Z ECMWF cycle, so those rows are
    not independent samples and must not get extra weight in training or scoring.
    """
    history = history.copy()
    history["base_time"] = history["valid_time"] - pd.to_timedelta(history["lead_hours"], unit="h")
    history = history.sort_values("run_time").drop_duplicates(
        ["target", "valid_time", "base_time"],
        keep="last",
    )
    return history


def training_history(verification, target_key, run_time, days):
    if verification is None or verification.empty:
        return pd.DataFrame()

    history = verification[verification["target"] == target_key].copy()
    if history.empty:
        return history

    history = parse_timestamp_column(history, "run_time")
    history = parse_timestamp_column(history, "valid_time")
    history = parse_timestamp_column(history, "latest_observation_time")
    run_time = to_utc_naive(run_time)
    cutoff = run_time - pd.Timedelta(days=days)
    history = history[(history["run_time"] >= cutoff) & (history["valid_time"] <= run_time)]
    numeric_columns = [
        "lead_hours",
        "raw_forecast_c",
        "observed_c",
        "raw_error_c",
        "forecast_error_c",
        "climatology_c",
        "latest_observation_anomaly_c",
        "wrf_forecast_c",
    ]
    for column in numeric_columns:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
        else:
            history[column] = np.nan
    history = history.dropna(subset=["lead_hours", "raw_error_c", "forecast_error_c"])
    if history.empty:
        return history

    history["valid_hour"] = history["valid_time"].dt.hour
    hours_since_obs = (
        history["valid_time"] - history["latest_observation_time"]
    ).dt.total_seconds() / 3600
    history["anomaly_feature"] = history["latest_observation_anomaly_c"] * np.exp(
        -hours_since_obs.clip(lower=0) / BIAS_DECAY_HOURS
    )
    return independent_history(history)


def hourly_bias_profile(verification, target_key, run_time):
    """Median raw ECMWF error by UTC hour of the valid time (second-tier correction)."""
    history = training_history(verification, target_key, run_time, BIAS_TRAINING_DAYS)
    if history.empty:
        return {}, {}, 0

    hour_bias = {}
    hour_uncertainty = {}
    for hour, group in history.groupby("valid_hour"):
        if len(group) < MIN_BIAS_SAMPLES:
            continue
        hour_bias[int(hour)] = float((-group["raw_error_c"]).median())
        hour_uncertainty[int(hour)] = float(
            max(MIN_UNCERTAINTY_C, group["forecast_error_c"].abs().quantile(0.80))
        )
    return hour_bias, hour_uncertainty, len(history)


def fit_linear_profile(history, feature_columns):
    """Per valid-hour least squares fit. Returns {hour: {"intercept", "coefficients", "uncertainty_c", "samples"}}."""
    required = ["observed_c", *feature_columns]
    history = history.dropna(subset=required)
    profile = {}
    for hour, group in history.groupby("valid_hour"):
        if len(group) < MIN_MOS_SAMPLES:
            continue
        design = np.column_stack([
            np.ones(len(group)),
            *[group[column].to_numpy(dtype=float) for column in feature_columns],
        ])
        observed = group["observed_c"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(design, observed, rcond=None)
        if not np.all(np.isfinite(beta)):
            continue
        if not all(COEFFICIENT_BOUNDS[0] <= coefficient <= COEFFICIENT_BOUNDS[1] for coefficient in beta[1:]):
            continue
        residuals = observed - design @ beta
        uncertainty_c = float(
            max(MIN_UNCERTAINTY_C, np.quantile(np.abs(residuals), 0.80) * MOS_UNCERTAINTY_INFLATION)
        )
        profile[int(hour)] = {
            "intercept": float(beta[0]),
            "coefficients": {column: float(value) for column, value in zip(feature_columns, beta[1:])},
            "uncertainty_c": uncertainty_c,
            "samples": int(len(group)),
        }
    return profile


def fit_mos_profiles(verification, target_key, run_time):
    """Fit every MOS predictor set that has enough history.

    Returns {method: {hour: fit}} for the methods in MOS_FEATURE_SETS. Six months
    of Inaquito verification: ECMWF + climatology cut MAE from 1.13 to 1.01 C
    against the old lead-offset correction, and adding the decayed observed
    anomaly cut it further to 0.84 C.
    """
    history = training_history(verification, target_key, run_time, MOS_TRAINING_DAYS)
    if history.empty:
        return {}
    profiles = {}
    for method, feature_columns in MOS_FEATURE_SETS:
        profile = fit_linear_profile(history, feature_columns)
        if profile:
            profiles[method] = profile
    return profiles


def fit_mos_profile(verification, target_key, run_time):
    """Backwards compatible helper: the best available ECMWF + climatology (+ anomaly) fit."""
    profiles = fit_mos_profiles(verification, target_key, run_time)
    for method, _ in MOS_FEATURE_SETS:
        if method in profiles and method != METHOD_MOS_WRF:
            return profiles[method]
    return {}


def compute_local_bias(
    forecast,
    station_targets,
    target_key,
    verification=None,
    run_time=None,
    climatology_by_hour=None,
):
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
    if climatology_by_hour is None:
        climatology_by_hour = diurnal_climatology(station_targets, target_key, run_time)

    latest_climatology = climatology_by_hour.get(int(pd.Timestamp(latest_observation_time).hour), np.nan)
    latest_observation_anomaly_c = (
        float(latest_observation_c - latest_climatology) if pd.notna(latest_climatology) else np.nan
    )

    hour_bias, hour_uncertainty, historical_sample_count = hourly_bias_profile(
        verification,
        target_key,
        run_time,
    )
    mos_profiles = fit_mos_profiles(verification, target_key, run_time)

    if mos_profiles:
        fitted = ", ".join(f"{method} {len(profile)} h" for method, profile in mos_profiles.items())
        source = (
            f"MOS regression per hour ({fitted}); "
            f"hour-of-day offset for {len(hour_bias)} hour(s); current fallback from {source}"
        )
    elif hour_bias:
        source = f"hour-of-day offset for {len(hour_bias)} hour(s); current fallback from {source}"

    return {
        "bias_c": bias_c,
        "bias_source": source,
        "raw_mae_c": raw_mae_c,
        "raw_rmse_c": raw_rmse_c,
        "uncertainty_c": uncertainty_c,
        "climatology_by_hour": climatology_by_hour,
        "latest_observation_anomaly_c": latest_observation_anomaly_c,
        "mos_profiles": mos_profiles,
        "mos_profile": fit_best_non_wrf(mos_profiles),
        "hour_bias_c": hour_bias,
        "hour_uncertainty_c": hour_uncertainty,
        "historical_sample_count": historical_sample_count,
        "latest_observation_time": latest_observation_time,
        "latest_observation_c": latest_observation_c,
    }


def fit_best_non_wrf(mos_profiles):
    for method, _ in MOS_FEATURE_SETS:
        if method != METHOD_MOS_WRF and method in mos_profiles:
            return mos_profiles[method]
    return {}


def _evaluate_profile(profile, valid_hours, features):
    """Vectorised prediction for one fitted profile; NaN where the hour or a feature is missing."""
    fits = valid_hours.map(lambda hour: profile.get(int(hour)) if pd.notna(hour) else None)
    prediction = pd.to_numeric(fits.map(lambda fit: fit["intercept"] if fit else np.nan), errors="coerce")
    for column, values in features.items():
        coefficient = pd.to_numeric(
            fits.map(lambda fit, column=column: fit["coefficients"].get(column, np.nan) if fit else np.nan),
            errors="coerce",
        )
        prediction = prediction + coefficient * values
    uncertainty = pd.to_numeric(fits.map(lambda fit: fit["uncertainty_c"] if fit else np.nan), errors="coerce")
    return prediction, uncertainty


def apply_bias_correction(forecast, bias_by_target):
    corrected = forecast.copy()
    valid_hours = corrected["valid_time"].dt.hour
    wrf = pd.to_numeric(corrected["wrf_temp_c"], errors="coerce") if "wrf_temp_c" in corrected else pd.Series(
        np.nan, index=corrected.index
    )
    for target_key, bias_info in bias_by_target.items():
        spec = FORECAST_TARGETS[target_key]
        raw = pd.to_numeric(corrected[spec["raw_column"]], errors="coerce")

        decay = anomaly_decay(corrected["valid_time"], bias_info["latest_observation_time"])
        decay.index = corrected.index
        fallback_forecast = raw + bias_info["bias_c"] * decay

        hour_bias = pd.to_numeric(
            valid_hours.map(lambda hour: bias_info["hour_bias_c"].get(int(hour), np.nan)),
            errors="coerce",
        )
        offset_forecast = raw + hour_bias

        climatology = climatology_for_times(corrected["valid_time"], bias_info["climatology_by_hour"])
        climatology.index = corrected.index
        anomaly_feature = decay * bias_info.get("latest_observation_anomaly_c", np.nan)
        features = {
            "raw_forecast_c": raw,
            "climatology_c": climatology,
            "anomaly_feature": anomaly_feature,
            "wrf_forecast_c": wrf,
        }

        forecast_c = pd.Series(np.nan, index=corrected.index)
        uncertainty = pd.Series(np.nan, index=corrected.index)
        method = pd.Series(METHOD_OVERLAP_FALLBACK, index=corrected.index, dtype=object)
        assigned = pd.Series(False, index=corrected.index)
        profiles = bias_info.get("mos_profiles", {})
        for method_name, feature_columns in MOS_FEATURE_SETS:
            profile = profiles.get(method_name)
            if not profile:
                continue
            prediction, prediction_uncertainty = _evaluate_profile(
                profile,
                valid_hours,
                {column: features[column] for column in feature_columns},
            )
            use = ~assigned & prediction.notna()
            forecast_c = forecast_c.where(~use, prediction)
            uncertainty = uncertainty.where(~use, prediction_uncertainty)
            method = method.where(~use, method_name)
            assigned = assigned | use

        use_offset = ~assigned & offset_forecast.notna()
        forecast_c = forecast_c.where(~use_offset, offset_forecast)
        hour_uncertainty = pd.to_numeric(
            valid_hours.map(lambda hour: bias_info["hour_uncertainty_c"].get(int(hour), np.nan)),
            errors="coerce",
        )
        uncertainty = uncertainty.where(~use_offset, hour_uncertainty)
        method = method.where(~use_offset, METHOD_HOUR_OFFSET)
        assigned = assigned | use_offset

        forecast_c = forecast_c.where(assigned, fallback_forecast)
        uncertainty = uncertainty.where(uncertainty.notna(), bias_info["uncertainty_c"])

        corrected[spec["forecast_column"]] = forecast_c
        corrected[spec["bias_column"]] = forecast_c - raw
        corrected[spec["persistence_column"]] = bias_info["latest_observation_c"]
        corrected[f"climatology_{target_key}_c"] = climatology
        corrected[f"correction_method_{target_key}"] = method.to_numpy()
        corrected[f"uncertainty_{target_key}_c"] = uncertainty
        corrected[spec["lower_column"]] = forecast_c - uncertainty
        corrected[spec["upper_column"]] = forecast_c + uncertainty
    return corrected
