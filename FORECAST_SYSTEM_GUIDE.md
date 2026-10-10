# Forecast System Guide

This project is building an operational forecast system for the Inaquito station, not just a static chart.

## What The System Does

Every hourly run does five things:

1. Gets recent INAMHI observations for Inaquito (the API only returns about 30 days).
2. Gets ECMWF forecast data near the station, using cloud mirrors by default to avoid direct-portal rate limits.
3. Gets the INAMHI WRF 2 m temperature at the station from the INAMHI-GEOGLOWS GeoTIFF service (`services.geoglows.org/api/met-data-explorer/donwload-geotiff`, enabled by default; set `WRF_SERVICE_URL=off` to disable).
4. Builds a statistical correction (MOS) per hour of day from the verified history and applies it.
5. Saves the forecast, the web page, the forecast archive, the cumulative verification file and the metrics.

The public page is:

```text
https://ccardenas93.github.io/weather_forecast/
```

The root page redirects to:

```text
forecast_Inaquito.html
```

## Forecast Tabs

The page has three temperature tabs:

- `Max`: based on ECMWF `mx2t3`, maximum 2 m temperature in the last 3 hours, verified against the 3-hour rolling max of the station.
- `Prom`: based on ECMWF `2t`, 2 m temperature, verified against the hourly station value.
- `Min`: based on ECMWF `mn2t3`, minimum 2 m temperature in the last 3 hours, verified against the 3-hour rolling min of the station.

The raw station `MAX`, `PROM` and `MIN` hourly columns are almost identical (median spread 0.05 C), which is why the
max/min targets are built as 3-hour rolling extrema in `observations.py`. Verifying against the raw columns gives wrong results.

## The Correction Model

For each target and each UTC hour of the valid time, `calibration.py` fits a least-squares regression on the trailing
`MOS_TRAINING_DAYS` (60) days of independent verified cases:

```text
observed = a + b * raw ECMWF + c * climatology + d * anomaly [+ e * WRF]
```

- `climatology`: mean observation for that hour of day over the trailing 30 days. This is the no-skill benchmark.
- `anomaly`: latest observation minus its climatology, decayed with `exp(-hours since observation / 36)`.
- `WRF`: INAMHI WRF 2 m temperature at the same valid time (0.027 degree grid, valid hours in local time every 3 h out to +70 h, usually published for the previous day's initialization), used only once at least 25 verified cases with WRF exist per hour.

Predictor sets are tried in order `mos_wrf`, `mos`, `mos_basic`; if none has enough samples for an hour the system falls
back to an hour-of-day median offset, and finally to the current-run overlap bias with 36-hour decay. The method used
is stored per row as `correction_method` so the status page can compare methods over time.

Why this design, from six months of verification (April to October 2026, 7130 independent cases):

| method | MAE C |
| --- | --- |
| raw ECMWF | 2.81 |
| persistence | 3.29 |
| 30-day diurnal climatology | 1.20 |
| legacy per-lead median offset | 1.13 |
| MOS on ECMWF + climatology | 1.01 |
| production cascade with decayed observed anomaly | 0.89 (0.82 on the 79% of rows where the full MOS applied) |

The uncertainty band is the 80th percentile of the training residuals times `MOS_UNCERTAINTY_INFLATION` (1.1), which
gave 83% coverage out of sample (RMSE 1.20 C against 1.45 C for the legacy offset).

## Output Files

- `merged_data_export.csv`: latest INAMHI observation cache (about 30 days).
- `forecast_output.csv`: the latest forecast values used by the page, including climatology, WRF and the method per row.
- `forecast_Inaquito.html`: the interactive Plotly forecast page.
- `forecast_archive.csv`: operational future rows from every forecast run in long format (180-day retention).
- `forecast_verification.csv`: cumulative archive of forecasts matched against observations (400-day retention). It is
  appended to every run, never rebuilt from the 30-day cache, so the training window really is 60 days.
- `forecast_metrics.csv`: performance metrics by target and lead hour over independent samples (one per target, valid time and ECMWF cycle).
- `SYSTEM_STATUS.md`: regenerated operational status report with all-history and last-30-day metrics, method comparison and calibration state.
- `forecast_metrics.html`: interactive verification dashboard with MAE and coverage plots by target and lead hour.

## Code Layout

- `script.py`: orchestration entrypoint used by GitHub Actions.
- `forecast_config.py`: environment settings, station metadata, paths, target definitions, INAMHI table names, WRF settings.
- `observations.py`: INAMHI fetch and station target loading.
- `ecmwf_forecast.py`: ECMWF open-data retrieval and GRIB extraction.
- `wrf_forecast.py`: INAMHI WRF GeoTIFF retrieval and point sampling.
- `calibration.py`: climatology, MOS regression, fallbacks and forecast correction.
- `archive.py`: forecast archive, verification joins and the cumulative verification history.
- `metrics.py`: independent-sample verification metrics with raw, persistence, climatology and WRF baselines.
- `plot_outputs.py`, `metrics_dashboard.py`, `status_output.py`: generated HTML and Markdown outputs.
- `self_test.py`: dependency-light test of the fallback pipeline, the MOS fit and the WRF path.

## How To Read The Metrics

The most important fields in `forecast_metrics.csv` and `SYSTEM_STATUS.md` are:

- `sample_count`: independent verified cases (hourly reruns of the same ECMWF cycle count once).
- `forecast_mae_c` / `forecast_rmse_c`: corrected forecast error.
- `raw_mae_c`, `persistence_mae_c`, `climatology_mae_c`, `wrf_mae_c`: the baselines.
- `coverage_rate`: how often the observation fell inside the uncertainty band (target 80%).
- `mae_improvement_vs_climatology_c`: the number that matters. At an equatorial station persistence is a weak baseline
  because the diurnal cycle dominates; a model only has skill if it beats the hour-of-day climatology.

## Known Limitations

- One station cannot describe spatial patterns across Quito.
- ECMWF and WRF grid cells are much coarser than the station environment.
- Afternoon hours (16:00 to 19:00 local) remain the hardest: cloud and convection timing drive 1.2 to 1.5 C MAE there versus about 0.9 C overnight.
- Regime transitions (April, October) degrade the 60-day training window; the anomaly predictor helps but does not remove this.
- Observation outages (three in six months) leave the system on stale anomalies; the decay term limits the damage.

## Next Steps

1. Let the WRF predictor accumulate about 25 days of verified cases per hour; `mos_wrf` activates by itself after that.
2. Add ECMWF cloud cover, radiation and dewpoint as predictors for the afternoon hours.
3. Add nearby stations to constrain the spatial pattern.
4. Add ECMWF ensemble spread to make the uncertainty band flow dependent.

## Practical Rule

Do not trust a model because it looks sophisticated. Trust it when `SYSTEM_STATUS.md` shows it beats the diurnal
climatology, raw ECMWF and WRF by lead hour and target over enough independent samples.
