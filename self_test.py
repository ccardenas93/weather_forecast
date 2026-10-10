import numpy as np
import pandas as pd

from archive import (
    build_forecast_archive_rows,
    future_forecast_rows,
    merge_verification_history,
    verify_forecast_archive,
)
from calibration import (
    METHOD_MOS,
    METHOD_MOS_ANOMALY,
    METHOD_MOS_BASIC,
    METHOD_MOS_WRF,
    METHOD_OVERLAP_FALLBACK,
    apply_bias_correction,
    compute_local_bias,
    diurnal_climatology,
    fit_mos_profiles,
)
from forecast_config import FORECAST_TARGETS
from metrics import summarize_verification_metrics, weighted_summary
from time_utils import to_utc_naive
from wrf_forecast import merge_wrf_into_forecast


def _synthetic_forecast_and_observations():
    valid_time = pd.date_range("2026-04-12T06:00:00", periods=5, freq="3h")
    forecast = pd.DataFrame({
        "valid_time": valid_time,
        "lead_hours": [0, 3, 6, 9, 12],
        "ecmwf_max_c": [19.5, 20.0, 21.0, 22.0, 21.5],
        "ecmwf_prom_c": [13.5, 14.0, 15.0, 16.0, 15.5],
        "ecmwf_min_c": [9.5, 10.0, 10.5, 11.0, 10.8],
    })
    station_targets = pd.DataFrame({
        "observed_max_c": [20.0, 21.0, 22.0, 22.5, 21.0],
        "observed_prom_c": [14.5, 15.0, 15.5, 16.5, 15.2],
        "observed_min_c": [8.8, 9.0, 10.0, 10.6, 10.4],
    }, index=valid_time)
    return forecast, station_targets


def _test_fallback_pipeline():
    run_time = pd.Timestamp("2026-04-12T06:00:00Z")
    forecast, station_targets = _synthetic_forecast_and_observations()
    forecast = merge_wrf_into_forecast(forecast, None)
    assert forecast["wrf_temp_c"].isna().all()

    bias_by_target = {
        target_key: compute_local_bias(
            forecast,
            station_targets,
            target_key,
            verification=pd.DataFrame(),
            run_time=run_time,
        )
        for target_key in FORECAST_TARGETS
    }
    corrected = apply_bias_correction(forecast, bias_by_target)
    assert (corrected["correction_method_prom"] == METHOD_OVERLAP_FALLBACK).all()
    operational = future_forecast_rows(corrected, run_time, pd.Timestamp("2026-04-12T06:00:00Z"))
    archive = build_forecast_archive_rows(operational, bias_by_target, run_time)
    verified = verify_forecast_archive(archive, station_targets)
    metrics = summarize_verification_metrics(verified)

    assert len(archive) == 12, f"expected 12 archive rows, got {len(archive)}"
    assert {"climatology_c", "correction_method", "wrf_forecast_c", "latest_observation_anomaly_c"}.issubset(archive.columns)
    assert len(verified) == 12, f"expected 12 verified rows, got {len(verified)}"
    assert len(metrics) == 12, f"expected 12 metric rows, got {len(metrics)}"
    assert set(metrics["target"]) == {"max", "prom", "min"}
    assert set(metrics["lead_hour"]) == {3, 6, 9, 12}
    assert archive["valid_time"].min() > to_utc_naive(run_time)
    assert metrics["forecast_mae_c"].notna().all()
    assert metrics["raw_mae_c"].notna().all()
    assert metrics["wrf_mae_c"].isna().all()
    assert metrics["coverage_rate"].between(0, 1).all()

    merged = merge_verification_history(verified, verified, run_time)
    assert len(merged) == 12, "cumulative verification must deduplicate identical rows"
    summary = weighted_summary(metrics)
    assert summary["samples"] == 12


def _synthetic_verification(with_wrf):
    rng = np.random.default_rng(7)
    days = 40
    valid_times = pd.date_range("2026-04-20T00:00:00", periods=days * 8, freq="3h")
    base_times = valid_times.normalize() + pd.Timedelta(hours=6)
    lead_hours = ((valid_times - base_times).total_seconds() / 3600).astype(float)
    lead_hours = np.where(lead_hours <= 0, lead_hours + 24, lead_hours)
    hour = valid_times.hour.to_numpy()
    climatology = 15 + 4 * np.sin((hour - 9) / 24 * 2 * np.pi)
    raw = climatology - 2.5 + rng.normal(0, 1.0, len(valid_times))
    anomaly = rng.normal(0, 1.0, len(valid_times))
    run_times = base_times + pd.Timedelta(hours=2)
    latest_obs_times = run_times - pd.Timedelta(hours=2)
    hours_since_obs = ((valid_times - latest_obs_times).total_seconds() / 3600).astype(float)
    decayed = anomaly * np.exp(-np.clip(hours_since_obs, 0, None) / 36)
    observed = 1.0 + 0.6 * raw + 0.4 * climatology + 0.5 * decayed + rng.normal(0, 0.3, len(valid_times))
    wrf = observed + rng.normal(0, 0.8, len(valid_times)) if with_wrf else np.nan

    verification = pd.DataFrame({
        "run_time": run_times,
        "target": "prom",
        "valid_time": valid_times,
        "lead_hours": lead_hours,
        "raw_forecast_c": raw,
        "forecast_c": raw + 2.5,
        "climatology_c": climatology,
        "latest_observation_time": latest_obs_times,
        "latest_observation_anomaly_c": anomaly,
        "wrf_forecast_c": wrf,
        "observed_c": observed,
    })
    verification["raw_error_c"] = verification["raw_forecast_c"] - verification["observed_c"]
    verification["forecast_error_c"] = verification["forecast_c"] - verification["observed_c"]
    return verification, {int(h): float(c) for h, c in zip(hour[:8], climatology[:8])}


def _test_mos_fit_and_apply():
    run_time = pd.Timestamp("2026-06-01T12:00:00Z")
    verification, climatology_by_hour = _synthetic_verification(with_wrf=False)

    profiles = fit_mos_profiles(verification, "prom", run_time)
    assert {METHOD_MOS, METHOD_MOS_ANOMALY, METHOD_MOS_BASIC} <= set(profiles), profiles.keys()
    assert METHOD_MOS_WRF not in profiles, "no WRF history must mean no WRF fit"
    assert len(profiles[METHOD_MOS]) == 8, f"expected a MOS fit for 8 valid hours, got {len(profiles[METHOD_MOS])}"
    for fit in profiles[METHOD_MOS].values():
        assert abs(fit["coefficients"]["raw_forecast_c"] - 0.6) < 0.15, fit
        assert abs(fit["coefficients"]["anomaly_feature"] - 0.5) < 0.25, fit
        assert fit["uncertainty_c"] >= 1.2

    forecast, station_targets = _synthetic_forecast_and_observations()
    station_targets.index = pd.date_range("2026-06-01T06:00:00", periods=5, freq="3h")
    forecast["valid_time"] = station_targets.index
    forecast = merge_wrf_into_forecast(forecast, None)
    bias_info = compute_local_bias(
        forecast,
        station_targets,
        "prom",
        verification=verification,
        run_time=run_time,
        climatology_by_hour=climatology_by_hour,
    )
    assert np.isfinite(bias_info["latest_observation_anomaly_c"])
    corrected = apply_bias_correction(forecast, {"prom": bias_info})
    assert (corrected["correction_method_prom"] == METHOD_MOS).all(), corrected["correction_method_prom"].tolist()
    assert corrected["forecast_prom_c"].notna().all()
    assert (corrected["forecast_prom_upper_c"] > corrected["forecast_prom_lower_c"]).all()

    # With WRF history and a WRF value on the forecast rows, the WRF fit must take over.
    verification_wrf, _ = _synthetic_verification(with_wrf=True)
    profiles_wrf = fit_mos_profiles(verification_wrf, "prom", run_time)
    assert METHOD_MOS_WRF in profiles_wrf, profiles_wrf.keys()
    wrf = pd.DataFrame({
        "valid_time": forecast["valid_time"].iloc[:3],
        "wrf_temp_c": [15.0, 16.0, 17.0],
        "wrf_init_time": pd.Timestamp("2026-06-01T05:00:00"),
    })
    forecast_wrf = merge_wrf_into_forecast(forecast.drop(columns=["wrf_temp_c", "wrf_init_time"]), wrf)
    bias_info_wrf = compute_local_bias(
        forecast_wrf,
        station_targets,
        "prom",
        verification=verification_wrf,
        run_time=run_time,
        climatology_by_hour=climatology_by_hour,
    )
    corrected_wrf = apply_bias_correction(forecast_wrf, {"prom": bias_info_wrf})
    methods = corrected_wrf["correction_method_prom"].tolist()
    assert methods[:3] == [METHOD_MOS_WRF] * 3, methods
    assert methods[3:] == [METHOD_MOS] * 2, methods

    empty_climatology = diurnal_climatology(station_targets, "prom", run_time)
    assert empty_climatology == {}, "five observations must not produce a climatology"


def run_self_test():
    _test_fallback_pipeline()
    _test_mos_fit_and_apply()
    return "passed"
