import numpy as np
import pandas as pd

from calibration import independent_history
from time_utils import parse_timestamp_column, to_utc_naive


METRIC_COLUMNS = [
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
    "climatology_mae_c",
    "wrf_mae_c",
    "wrf_sample_count",
    "coverage_rate",
    "mae_improvement_vs_raw_c",
    "mae_improvement_vs_persistence_c",
    "mae_improvement_vs_climatology_c",
    "mae_improvement_vs_wrf_c",
]


def _mae(errors):
    errors = pd.to_numeric(errors, errors="coerce").dropna()
    return float(errors.abs().mean()) if len(errors) else np.nan


def _rmse(errors):
    errors = pd.to_numeric(errors, errors="coerce").dropna()
    return float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan


def independent_verification(verified, since=None):
    """One row per target, valid time and ECMWF cycle, optionally limited to recent valid times."""
    if verified.empty:
        return verified

    scored = verified.copy()
    scored = parse_timestamp_column(scored, "run_time")
    scored = parse_timestamp_column(scored, "valid_time")
    for column in ["lead_hours", "forecast_error_c", "raw_error_c", "persistence_error_c", "climatology_error_c", "wrf_error_c"]:
        if column in scored.columns:
            scored[column] = pd.to_numeric(scored[column], errors="coerce")
        else:
            scored[column] = np.nan
    scored = scored.dropna(subset=["lead_hours", "forecast_error_c", "raw_error_c"])
    if since is not None:
        scored = scored[scored["valid_time"] >= to_utc_naive(since)]
    if scored.empty:
        return scored
    return independent_history(scored)


def summarize_verification_metrics(verified, since=None):
    if verified.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    scored = independent_verification(verified, since=since)
    if scored.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    scored["lead_hour"] = scored["lead_hours"].round().astype(int)
    scored["within_band"] = scored["within_band"].astype(str).str.lower().isin(["true", "1", "1.0"])
    metrics = scored.groupby(["target", "target_label", "lead_hour"]).agg(
        sample_count=("forecast_error_c", "count"),
        forecast_bias_c=("forecast_error_c", "mean"),
        forecast_mae_c=("forecast_error_c", _mae),
        forecast_rmse_c=("forecast_error_c", _rmse),
        raw_bias_c=("raw_error_c", "mean"),
        raw_mae_c=("raw_error_c", _mae),
        raw_rmse_c=("raw_error_c", _rmse),
        persistence_mae_c=("persistence_error_c", _mae),
        climatology_mae_c=("climatology_error_c", _mae),
        wrf_mae_c=("wrf_error_c", _mae),
        wrf_sample_count=("wrf_error_c", "count"),
        coverage_rate=("within_band", "mean"),
    ).reset_index()
    # WRF skill is only comparable on the rows where WRF was available.
    wrf_rows = scored[scored["wrf_error_c"].notna()]
    if not wrf_rows.empty:
        paired = wrf_rows.groupby(["target", "target_label", "lead_hour"])["forecast_error_c"].apply(_mae)
        paired = paired.rename("forecast_mae_on_wrf_rows_c").reset_index()
        metrics = metrics.merge(paired, on=["target", "target_label", "lead_hour"], how="left")
        metrics["mae_improvement_vs_wrf_c"] = metrics["wrf_mae_c"] - metrics["forecast_mae_on_wrf_rows_c"]
    else:
        metrics["mae_improvement_vs_wrf_c"] = np.nan
    metrics["mae_improvement_vs_raw_c"] = metrics["raw_mae_c"] - metrics["forecast_mae_c"]
    metrics["mae_improvement_vs_persistence_c"] = (
        metrics["persistence_mae_c"] - metrics["forecast_mae_c"]
    )
    metrics["mae_improvement_vs_climatology_c"] = (
        metrics["climatology_mae_c"] - metrics["forecast_mae_c"]
    )
    return metrics[METRIC_COLUMNS].sort_values(["target", "lead_hour"]).reset_index(drop=True)


def summarize_by_method(verified, since=None):
    """MAE by correction method so a model change can be judged against its predecessor."""
    if verified.empty or "correction_method" not in verified.columns:
        return pd.DataFrame(columns=["correction_method", "sample_count", "forecast_mae_c", "coverage_rate"])

    scored = independent_verification(verified, since=since)
    if scored.empty:
        return pd.DataFrame(columns=["correction_method", "sample_count", "forecast_mae_c", "coverage_rate"])

    scored["within_band"] = scored["within_band"].astype(str).str.lower().isin(["true", "1", "1.0"])
    scored["correction_method"] = scored["correction_method"].fillna("unknown").astype(str)
    summary = scored.groupby("correction_method").agg(
        sample_count=("forecast_error_c", "count"),
        forecast_mae_c=("forecast_error_c", _mae),
        raw_mae_c=("raw_error_c", _mae),
        climatology_mae_c=("climatology_error_c", _mae),
        wrf_mae_c=("wrf_error_c", _mae),
        coverage_rate=("within_band", "mean"),
    ).reset_index()
    return summary.sort_values("sample_count", ascending=False).reset_index(drop=True)


def _empty_summary():
    return {
        "rows": 0,
        "samples": 0,
        "forecast_mae_c": np.nan,
        "forecast_rmse_c": np.nan,
        "raw_mae_c": np.nan,
        "persistence_mae_c": np.nan,
        "climatology_mae_c": np.nan,
        "wrf_mae_c": np.nan,
        "wrf_samples": 0,
        "coverage_rate": np.nan,
        "mae_improvement_vs_raw_c": np.nan,
        "mae_improvement_vs_persistence_c": np.nan,
        "mae_improvement_vs_climatology_c": np.nan,
        "mae_improvement_vs_wrf_c": np.nan,
        "leads_better_than_raw": 0,
        "leads_better_than_persistence": 0,
        "leads_better_than_climatology": 0,
    }


def weighted_summary(metrics):
    if metrics.empty:
        return _empty_summary()

    weights = metrics["sample_count"]
    sample_count = float(weights.sum())
    if sample_count == 0:
        return _empty_summary()

    def weighted(column):
        values = metrics[column]
        mask = values.notna()
        if not mask.any():
            return np.nan
        return float((values[mask] * weights[mask]).sum() / weights[mask].sum())

    def weighted_by(column, weight_column):
        if column not in metrics or weight_column not in metrics:
            return np.nan
        values = metrics[column]
        column_weights = metrics[weight_column].fillna(0)
        mask = values.notna() & (column_weights > 0)
        if not mask.any():
            return np.nan
        return float((values[mask] * column_weights[mask]).sum() / column_weights[mask].sum())

    return {
        "rows": int(len(metrics)),
        "samples": int(sample_count),
        "forecast_mae_c": weighted("forecast_mae_c"),
        "forecast_rmse_c": float(np.sqrt(weighted_square(metrics, "forecast_rmse_c"))),
        "raw_mae_c": weighted("raw_mae_c"),
        "persistence_mae_c": weighted("persistence_mae_c"),
        "climatology_mae_c": weighted("climatology_mae_c"),
        "wrf_mae_c": weighted_by("wrf_mae_c", "wrf_sample_count"),
        "wrf_samples": int(metrics["wrf_sample_count"].fillna(0).sum()) if "wrf_sample_count" in metrics else 0,
        "coverage_rate": weighted("coverage_rate"),
        "mae_improvement_vs_raw_c": weighted("mae_improvement_vs_raw_c"),
        "mae_improvement_vs_persistence_c": weighted("mae_improvement_vs_persistence_c"),
        "mae_improvement_vs_climatology_c": weighted("mae_improvement_vs_climatology_c"),
        "mae_improvement_vs_wrf_c": weighted_by("mae_improvement_vs_wrf_c", "wrf_sample_count"),
        "leads_better_than_raw": int((metrics["mae_improvement_vs_raw_c"] > 0).sum()),
        "leads_better_than_persistence": int((metrics["mae_improvement_vs_persistence_c"] > 0).sum()),
        "leads_better_than_climatology": int((metrics["mae_improvement_vs_climatology_c"] > 0).sum()),
    }


def weighted_square(metrics, column):
    values = metrics[column]
    weights = metrics["sample_count"]
    mask = values.notna()
    if not mask.any():
        return np.nan
    return float((np.square(values[mask]) * weights[mask]).sum() / weights[mask].sum())
