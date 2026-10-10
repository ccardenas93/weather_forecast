import os
import sys

import pandas as pd

from archive import (
    backfill_archive_climatology,
    load_forecast_archive,
    load_verification_history,
    mark_operational_forecast_rows,
    merge_verification_history,
    save_verification_history,
    update_forecast_archive,
    verify_forecast_archive,
)
from calibration import apply_bias_correction, compute_local_bias, diurnal_climatology
from ecmwf_forecast import download_ecmwf_temperatures
from forecast_config import (
    ALLOW_STALE_OBS,
    ECMWF_SOURCES,
    FETCH_DAYS,
    FORECAST_ARCHIVE_PATH,
    FORECAST_CSV_PATH,
    FORECAST_HTML_PATH,
    FORECAST_METRICS_PATH,
    FORECAST_TARGETS,
    FORECAST_VERIFICATION_PATH,
    METRICS_HTML_PATH,
    RAW_DATA_PATH,
    RECENT_METRICS_DAYS,
    STATION_ID,
    STATION_NAME,
    SYSTEM_STATUS_PATH,
    TABLE_NAMES,
)
from metrics import summarize_by_method, summarize_verification_metrics
from observations import fetch_weather_data, load_station_targets
from outputs import build_metrics_dashboard, build_plot, build_system_status
from self_test import run_self_test
from wrf_forecast import download_wrf_temperature, merge_wrf_into_forecast, wrf_enabled


def main():
    run_time = pd.Timestamp.now(tz="UTC")
    self_test_status = run_self_test()
    print(f"Forecast system self-test {self_test_status}", flush=True)

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

    climatology_by_target = {
        target_key: diurnal_climatology(station_targets, target_key, run_time)
        for target_key in FORECAST_TARGETS
    }
    forecast_archive = backfill_archive_climatology(
        load_forecast_archive(FORECAST_ARCHIVE_PATH),
        climatology_by_target,
    )
    verification_history = load_verification_history(FORECAST_VERIFICATION_PATH)
    historical_verification = merge_verification_history(
        verification_history,
        verify_forecast_archive(forecast_archive, station_targets),
        run_time,
    )
    print(
        f"Loaded {len(forecast_archive)} archived forecast row(s); "
        f"{len(historical_verification)} verified row(s) available for calibration "
        f"({len(verification_history)} from the cumulative history)",
        flush=True,
    )

    print(
        f"Retrieving ECMWF max/prom/min 2m temperature forecast for {STATION_NAME} "
        f"from sources {', '.join(ECMWF_SOURCES)}",
        flush=True,
    )
    raw_forecast = download_ecmwf_temperatures()
    print(f"Using ECMWF source {raw_forecast.attrs.get('ecmwf_source')}", flush=True)
    if wrf_enabled():
        print("Retrieving INAMHI WRF 2 m temperature", flush=True)
        wrf_forecast = download_wrf_temperature(run_time)
    else:
        print("WRF_SERVICE_URL not set; running on ECMWF only", flush=True)
        wrf_forecast = None
    raw_forecast = merge_wrf_into_forecast(raw_forecast, wrf_forecast)
    print(
        f"WRF temperature attached to {int(raw_forecast['wrf_temp_c'].notna().sum())} of {len(raw_forecast)} ECMWF step(s)",
        flush=True,
    )
    bias_by_target = {
        target_key: compute_local_bias(
            raw_forecast,
            station_targets,
            target_key,
            verification=historical_verification,
            run_time=run_time,
            climatology_by_hour=climatology_by_target[target_key],
        )
        for target_key in FORECAST_TARGETS
    }
    corrected_forecast = apply_bias_correction(raw_forecast, bias_by_target)
    corrected_forecast = mark_operational_forecast_rows(
        corrected_forecast,
        run_time,
        latest_observation_time,
    )
    corrected_forecast.to_csv(FORECAST_CSV_PATH, index=False)
    build_plot(station_targets, corrected_forecast, bias_by_target)

    forecast_archive = update_forecast_archive(
        forecast_archive,
        corrected_forecast,
        bias_by_target,
        run_time,
        latest_observation_time,
    )
    verification = merge_verification_history(
        historical_verification,
        verify_forecast_archive(forecast_archive, station_targets),
        run_time,
    )
    metrics = summarize_verification_metrics(verification)
    recent_since = run_time - pd.Timedelta(days=RECENT_METRICS_DAYS)
    recent_metrics = summarize_verification_metrics(verification, since=recent_since)
    method_summary = summarize_by_method(verification)
    save_verification_history(verification, FORECAST_VERIFICATION_PATH)
    metrics.to_csv(FORECAST_METRICS_PATH, index=False)
    build_metrics_dashboard(metrics)
    build_system_status(
        run_time=run_time,
        latest_observation_time=latest_observation_time,
        forecast_archive=forecast_archive,
        verification=verification,
        metrics=metrics,
        ecmwf_source=raw_forecast.attrs.get("ecmwf_source"),
        self_test_status=self_test_status,
        recent_metrics=recent_metrics,
        method_summary=method_summary,
        bias_by_target=bias_by_target,
    )

    calibration_summary = ", ".join(
        f"{FORECAST_TARGETS[target_key]['label']}: "
        + "/".join(f"{method} {len(profile)} h" for method, profile in bias_info["mos_profiles"].items())
        + f", offset {len(bias_info['hour_bias_c'])} h, fallback {bias_info['bias_c']:.2f} C"
        for target_key, bias_info in bias_by_target.items()
    )
    print(
        f"Saved {FORECAST_CSV_PATH} and {FORECAST_HTML_PATH}; "
        f"wrote {SYSTEM_STATUS_PATH} and {METRICS_HTML_PATH}; "
        f"archived {len(forecast_archive)} row(s), verified {len(verification)} row(s), "
        f"metrics {len(metrics)} row(s); calibration {calibration_summary}",
        flush=True,
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        print(f"Forecast system self-test {run_self_test()}", flush=True)
    else:
        main()
