import pandas as pd

from archive import build_forecast_archive_rows, future_forecast_rows, verify_forecast_archive
from calibration import apply_bias_correction, compute_local_bias
from forecast_config import FORECAST_TARGETS
from metrics import summarize_verification_metrics
from time_utils import to_utc_naive


def run_self_test():
    run_time = pd.Timestamp("2026-04-12T06:00:00Z")
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
    operational = future_forecast_rows(
        corrected,
        run_time,
        pd.Timestamp("2026-04-12T06:00:00Z"),
    )
    archive = build_forecast_archive_rows(operational, bias_by_target, run_time)
    verified = verify_forecast_archive(archive, station_targets)
    metrics = summarize_verification_metrics(verified)

    assert len(archive) == 12, f"expected 12 archive rows, got {len(archive)}"
    assert len(verified) == 12, f"expected 12 verified rows, got {len(verified)}"
    assert len(metrics) == 12, f"expected 12 metric rows, got {len(metrics)}"
    assert set(metrics["target"]) == {"max", "prom", "min"}
    assert set(metrics["lead_hour"]) == {3, 6, 9, 12}
    assert archive["valid_time"].min() > to_utc_naive(run_time)
    assert metrics["forecast_mae_c"].notna().all()
    assert metrics["raw_mae_c"].notna().all()
    assert metrics["coverage_rate"].between(0, 1).all()
    return "passed"
