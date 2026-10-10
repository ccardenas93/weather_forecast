import pandas as pd

from forecast_config import RECENT_METRICS_DAYS, SYSTEM_STATUS_PATH
from metrics import weighted_summary
from time_utils import to_utc_naive


OVERALL_HEADERS = [
    "samples",
    "forecast_mae_c",
    "forecast_rmse_c",
    "raw_mae_c",
    "persistence_mae_c",
    "climatology_mae_c",
    "wrf_mae_c",
    "wrf_samples",
    "vs_raw_c",
    "vs_persistence_c",
    "vs_climatology_c",
    "vs_wrf_c",
    "coverage",
    "leads_better_raw",
    "leads_better_climatology",
]

TARGET_HEADERS = [
    "target",
    "samples",
    "forecast_mae_c",
    "raw_mae_c",
    "persistence_mae_c",
    "climatology_mae_c",
    "vs_raw_c",
    "vs_climatology_c",
    "coverage",
    "leads_better_raw",
    "leads_better_climatology",
]

PROBLEM_HEADERS = [
    "target",
    "lead_hour",
    "samples",
    "forecast_mae_c",
    "raw_mae_c",
    "climatology_mae_c",
    "vs_raw_c",
    "vs_climatology_c",
    "coverage",
]

METHOD_HEADERS = [
    "correction_method",
    "samples",
    "forecast_mae_c",
    "raw_mae_c",
    "climatology_mae_c",
    "coverage",
]


def format_float(value, digits=2, suffix=""):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def markdown_table(rows, headers):
    if not rows:
        return "_No rows._"

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(table)


def summary_row(summary):
    return {
        "samples": summary["samples"],
        "forecast_mae_c": format_float(summary["forecast_mae_c"], 2),
        "forecast_rmse_c": format_float(summary["forecast_rmse_c"], 2),
        "raw_mae_c": format_float(summary["raw_mae_c"], 2),
        "persistence_mae_c": format_float(summary["persistence_mae_c"], 2),
        "climatology_mae_c": format_float(summary["climatology_mae_c"], 2),
        "wrf_mae_c": format_float(summary.get("wrf_mae_c"), 2),
        "wrf_samples": summary.get("wrf_samples", 0),
        "vs_raw_c": format_float(summary["mae_improvement_vs_raw_c"], 2),
        "vs_persistence_c": format_float(summary["mae_improvement_vs_persistence_c"], 2),
        "vs_climatology_c": format_float(summary["mae_improvement_vs_climatology_c"], 2),
        "vs_wrf_c": format_float(summary.get("mae_improvement_vs_wrf_c"), 2),
        "coverage": format_float(summary["coverage_rate"] * 100, 1, "%"),
        "leads_better_raw": f"{summary['leads_better_than_raw']}/{summary['rows']}",
        "leads_better_climatology": f"{summary['leads_better_than_climatology']}/{summary['rows']}",
    }


def build_target_metric_rows(metrics):
    rows = []
    if metrics.empty:
        return rows

    for target_key, group in metrics.groupby("target"):
        summary = weighted_summary(group)
        row = summary_row(summary)
        row["target"] = target_key
        row["leads_better_raw"] = f"{summary['leads_better_than_raw']}/{len(group)}"
        row["leads_better_climatology"] = f"{summary['leads_better_than_climatology']}/{len(group)}"
        rows.append(row)
    return rows


def build_problem_lead_rows(metrics):
    if metrics.empty:
        return []

    problem_leads = metrics[
        (metrics["mae_improvement_vs_raw_c"] < 0)
        | (metrics["mae_improvement_vs_climatology_c"] < 0)
        | (metrics["coverage_rate"] < 0.70)
    ].copy()
    if problem_leads.empty:
        return []

    problem_leads["sort_risk"] = problem_leads[[
        "mae_improvement_vs_raw_c",
        "mae_improvement_vs_climatology_c",
    ]].min(axis=1)
    problem_leads = problem_leads.sort_values(["sort_risk", "coverage_rate"]).head(12)
    rows = []
    for _, row in problem_leads.iterrows():
        rows.append({
            "target": row["target"],
            "lead_hour": int(row["lead_hour"]),
            "samples": int(row["sample_count"]),
            "forecast_mae_c": format_float(row["forecast_mae_c"], 2),
            "raw_mae_c": format_float(row["raw_mae_c"], 2),
            "climatology_mae_c": format_float(row["climatology_mae_c"], 2),
            "vs_raw_c": format_float(row["mae_improvement_vs_raw_c"], 2),
            "vs_climatology_c": format_float(row["mae_improvement_vs_climatology_c"], 2),
            "coverage": format_float(row["coverage_rate"] * 100, 1, "%"),
        })
    return rows


def build_method_rows(method_summary):
    rows = []
    if method_summary is None or method_summary.empty:
        return rows
    for _, row in method_summary.iterrows():
        rows.append({
            "correction_method": row["correction_method"],
            "samples": int(row["sample_count"]),
            "forecast_mae_c": format_float(row["forecast_mae_c"], 2),
            "raw_mae_c": format_float(row["raw_mae_c"], 2),
            "climatology_mae_c": format_float(row["climatology_mae_c"], 2),
            "coverage": format_float(row["coverage_rate"] * 100, 1, "%"),
        })
    return rows


def describe_calibration(bias_by_target):
    lines = []
    if not bias_by_target:
        return lines
    for target_key, bias_info in bias_by_target.items():
        profiles = bias_info.get("mos_profiles", {})
        fitted = ", ".join(f"{method} `{len(profile)}` h" for method, profile in profiles.items()) or "none"
        offset_hours = len(bias_info.get("hour_bias_c", {}))
        clim_hours = len(bias_info.get("climatology_by_hour", {}))
        anomaly = bias_info.get("latest_observation_anomaly_c")
        lines.append(
            f"- `{target_key}`: MOS fits {fitted}; hour-offset hours `{offset_hours}`; "
            f"climatology hours `{clim_hours}`; latest observed anomaly `{format_float(anomaly, 2)}` C; "
            f"training samples `{bias_info.get('historical_sample_count', 0)}`"
        )
    return lines


def build_system_status(
    run_time,
    latest_observation_time,
    forecast_archive,
    verification,
    metrics,
    ecmwf_source,
    self_test_status,
    recent_metrics=None,
    method_summary=None,
    bias_by_target=None,
):
    summary = weighted_summary(metrics)
    recent_summary = weighted_summary(recent_metrics) if recent_metrics is not None else None
    target_rows = build_target_metric_rows(metrics)
    problem_rows = build_problem_lead_rows(recent_metrics if recent_metrics is not None else metrics)
    method_rows = build_method_rows(method_summary)

    if forecast_archive.empty:
        archive_window = "n/a"
    else:
        archive_run_times = pd.to_datetime(forecast_archive["run_time"], errors="coerce")
        archive_window = f"{archive_run_times.min()} to {archive_run_times.max()} UTC"

    if verification.empty:
        verification_window = "n/a"
    else:
        verification_times = pd.to_datetime(verification["valid_time"], errors="coerce")
        verification_window = f"{verification_times.min()} to {verification_times.max()} UTC"

    run_time_text = to_utc_naive(run_time)
    latest_obs_text = to_utc_naive(latest_observation_time)
    if not summary["samples"]:
        status = "warming up"
    elif summary["mae_improvement_vs_raw_c"] > 0 and (
        pd.isna(summary["mae_improvement_vs_climatology_c"])
        or summary["mae_improvement_vs_climatology_c"] > 0
    ):
        status = "green"
    else:
        status = "yellow"

    lines = [
        "# Forecast System Status",
        "",
        "This file is regenerated by the hourly forecast workflow.",
        "",
        "## Current Status",
        "",
        f"- Status: `{status}`",
        f"- Last successful run time: `{run_time_text} UTC`",
        f"- Latest observation used: `{latest_obs_text} UTC`",
        f"- ECMWF source used: `{ecmwf_source or 'unknown'}`",
        f"- Self-test: `{self_test_status}`",
        f"- Archive rows: `{len(forecast_archive)}`",
        f"- Verified forecast rows (all hourly runs): `{len(verification)}`",
        f"- Independent verified samples (one per target, valid time and ECMWF cycle): `{summary['samples']}`",
        f"- Metric rows: `{len(metrics)}`",
        f"- Archive run window: `{archive_window}`",
        f"- Verification valid-time window: `{verification_window}`",
        "",
        "## Calibration In Use",
        "",
        *(describe_calibration(bias_by_target) or ["_No calibration details._"]),
        "",
        "## Overall Metrics (all retained verification, independent samples)",
        "",
        markdown_table([summary_row(summary)], OVERALL_HEADERS),
        "",
        "Positive `vs_raw_c`, `vs_persistence_c` and `vs_climatology_c` mean the corrected forecast is better.",
        "`climatology_mae_c` is the error of a forecast that only uses the trailing 30-day mean temperature",
        "for that hour of day. It is the honest no-skill benchmark for this station; persistence is much weaker.",
        "`wrf_mae_c` is the raw INAMHI WRF error on the rows where WRF was available, and `vs_wrf_c` compares",
        "the corrected forecast with WRF on those same rows.",
        "",
        f"## Last {RECENT_METRICS_DAYS} Days (independent samples)",
        "",
        markdown_table([summary_row(recent_summary)], OVERALL_HEADERS) if recent_summary else "_No recent rows._",
        "",
        "## Metrics By Target (all retained verification)",
        "",
        markdown_table(target_rows, TARGET_HEADERS),
        "",
        "## Metrics By Correction Method",
        "",
        markdown_table(method_rows, METHOD_HEADERS),
        "",
        f"## Lead Hours To Watch (last {RECENT_METRICS_DAYS} days)",
        "",
        markdown_table(problem_rows, PROBLEM_HEADERS),
        "",
        "## Plots And Raw Files",
        "",
        "- Forecast page: [forecast_Inaquito.html](forecast_Inaquito.html)",
        "- Metrics dashboard: [forecast_metrics.html](forecast_metrics.html)",
        "- Current forecast CSV: [forecast_output.csv](forecast_output.csv)",
        "- Forecast archive: [forecast_archive.csv](forecast_archive.csv)",
        "- Forecast verification: [forecast_verification.csv](forecast_verification.csv)",
        "- Forecast metrics: [forecast_metrics.csv](forecast_metrics.csv)",
        "",
        "## Operational Notes",
        "",
        "- The workflow should stay green even when one ECMWF mirror rate-limits; it tries Azure, Google, then AWS.",
        "- A red workflow should be classified as data-source outage, ECMWF mirror outage, dependency issue, Python exception, or git push conflict before changing model logic.",
        "- Verification is cumulative. The observation cache only covers about 30 days, so the file is appended to each run and retained for 400 days.",
        "- Sample counts are per ECMWF cycle, not per hourly run. Hourly reruns reuse the same 06Z cycle and are not independent.",
        "- Judge any model change by `vs_climatology_c` and by the correction-method table, not by `vs_persistence_c`.",
        "",
    ]
    SYSTEM_STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")
