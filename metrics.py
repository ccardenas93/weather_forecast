import numpy as np
import pandas as pd


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


def weighted_summary(metrics):
    if metrics.empty:
        return {
            "rows": 0,
            "samples": 0,
            "forecast_mae_c": np.nan,
            "raw_mae_c": np.nan,
            "persistence_mae_c": np.nan,
            "coverage_rate": np.nan,
            "mae_improvement_vs_raw_c": np.nan,
            "mae_improvement_vs_persistence_c": np.nan,
            "leads_better_than_raw": 0,
            "leads_better_than_persistence": 0,
        }

    weights = metrics["sample_count"]
    sample_count = float(weights.sum())
    if sample_count == 0:
        return weighted_summary(pd.DataFrame())

    return {
        "rows": int(len(metrics)),
        "samples": int(sample_count),
        "forecast_mae_c": float((metrics["forecast_mae_c"] * weights).sum() / sample_count),
        "raw_mae_c": float((metrics["raw_mae_c"] * weights).sum() / sample_count),
        "persistence_mae_c": float((metrics["persistence_mae_c"] * weights).sum() / sample_count),
        "coverage_rate": float((metrics["coverage_rate"] * weights).sum() / sample_count),
        "mae_improvement_vs_raw_c": float(
            (metrics["mae_improvement_vs_raw_c"] * weights).sum() / sample_count
        ),
        "mae_improvement_vs_persistence_c": float(
            (metrics["mae_improvement_vs_persistence_c"] * weights).sum() / sample_count
        ),
        "leads_better_than_raw": int((metrics["mae_improvement_vs_raw_c"] > 0).sum()),
        "leads_better_than_persistence": int((metrics["mae_improvement_vs_persistence_c"] > 0).sum()),
    }
