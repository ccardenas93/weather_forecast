# Forecast System Guide

This project is building an operational forecast system for the Inaquito station, not just a static chart.

## What The System Does

Every hourly run does four things:

1. Gets recent INAMHI observations for Inaquito.
2. Gets ECMWF forecast data near the station, using the AWS mirror by default to avoid direct-portal rate limits.
3. Applies a local correction so the ECMWF forecast is adjusted to the station behavior.
4. Saves the forecast, the web page, and the verification data needed to improve future runs.

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

- `Max`: based on ECMWF `mx2t3`, maximum 2 m temperature in the last 3 hours.
- `Prom`: based on ECMWF `2t`, 2 m temperature.
- `Min`: based on ECMWF `mn2t3`, minimum 2 m temperature in the last 3 hours.

This matters because max/min should not be guessed from a single average temperature line. ECMWF has specific products for 3-hour maximum and minimum 2 m temperature, so the system uses those.

## What Variables Are Used Right Now

The correction model currently forecasts these station variables:

- `TEMPERATURA AIRE MAX`
- `TEMPERATURA AIRE PROM`
- `TEMPERATURA AIRE MIN`

The current ECMWF predictors used directly are:

- `mx2t3`
- `2t`
- `mn2t3`

The workflow sets `ECMWF_SOURCE=aws`. If AWS has a temporary problem, the script can be run with another source such as `azure`, `google`, or `ecmwf`, but the direct `ecmwf` source can hit HTTP 429 when the open-data portal is busy.

The script still downloads and stores other INAMHI observation columns in `merged_data_export.csv`, such as humidity, precipitation, pressure, and radiation. They are not yet used in the forecast correction model. They should be added deliberately after the verification archive is stable.

## Output Files

- `merged_data_export.csv`: latest INAMHI observation cache.
- `forecast_output.csv`: the latest forecast values used by the page.
- `forecast_Inaquito.html`: the interactive Plotly forecast page.
- `forecast_archive.csv`: every forecast run in long format, with run time, valid time, target, lead hour, raw ECMWF value, corrected forecast, and uncertainty band.
- `forecast_verification.csv`: archived forecasts matched against observations once the observed valid time is available.
- `forecast_metrics.csv`: objective performance metrics by target and lead hour.

## How To Read The Metrics

The most important fields in `forecast_metrics.csv` are:

- `sample_count`: how many verified forecast cases are available.
- `forecast_bias_c`: average corrected forecast error. Positive means the forecast is too warm.
- `forecast_mae_c`: average absolute corrected forecast error.
- `forecast_rmse_c`: root mean square corrected forecast error, which penalizes large errors.
- `raw_mae_c`: raw ECMWF absolute error before correction.
- `persistence_mae_c`: error from using the latest observation as the forecast.
- `coverage_rate`: how often the observation fell inside the uncertainty band.
- `mae_improvement_vs_raw_c`: positive means the corrected forecast beat raw ECMWF.
- `mae_improvement_vs_persistence_c`: positive means the corrected forecast beat persistence.

The system is only improving if it beats both raw ECMWF and persistence over enough samples.

## What Errors We Are Still Committing

The current system is better structured than the first version, but it still has real meteorological limitations:

- One station cannot describe spatial patterns across Quito.
- ECMWF grid resolution is much coarser than the station environment.
- `mx2t3` and `mn2t3` are 3-hour extrema, while the station target may represent a different aggregation window.
- Local valley wind, clouds, radiation timing, and urban effects are not fully represented yet.
- Bias correction needs an archive of verified forecasts before it becomes statistically meaningful.
- External data outages can still break or degrade a run.

## Next Steps

The next serious upgrades should be done in this order:

1. Let the archive accumulate verified forecast cases for several days.
2. Add tests around archive, verification, and metrics.
3. Add ECMWF predictors for dewpoint, cloud cover, radiation, precipitation, wind, and pressure.
4. Add hour-of-day and lead-time calibration.
5. Add more nearby stations and station metadata, especially elevation.
6. Train a small MOS/downscaling model only after there are enough verified samples.
7. Add probabilistic forecasts using ensemble data, so the output is not only one deterministic line.

## Practical Rule

Do not trust a model because it looks sophisticated. Trust it when the verification file shows it beats raw ECMWF and persistence by lead hour, target, and enough samples.
