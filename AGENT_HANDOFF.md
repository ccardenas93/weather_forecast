# Agent Handoff

This repo builds an automated temperature forecast page for the INAMHI Inaquito station using ECMWF Open Data.

## Current State

- Main branch: `master`
- Main script: `script.py`
- GitHub workflow: `.github/workflows/forecast.yml`
- Public page entry point: `index.html`, redirecting to `forecast_Inaquito.html`
- Current pushed feature: max/prom/min forecast tabs plus forecast archive and verification system

The workflow runs hourly and on pushes to `script.py`, `.github/workflows/forecast.yml`, and `requirements.txt`.

## Forecast Pipeline

`script.py` currently does this:

1. Fetches INAMHI hourly data for station `63777` Inaquito.
2. Falls back to cached `merged_data_export.csv` if INAMHI is temporarily down and `ALLOW_STALE_OBS=1`.
3. Loads three observed target series:
   - `TEMPERATURA AIRE MAX`
   - `TEMPERATURA AIRE PROM`
   - `TEMPERATURA AIRE MIN`
4. Downloads ECMWF IFS Open Data for:
   - `mx2t3`: max 2 m temperature in the previous 3 hours
   - `2t`: instantaneous 2 m temperature
   - `mn2t3`: min 2 m temperature in the previous 3 hours
   - Default sources are `azure,google,aws` via `ECMWF_SOURCES`, because individual mirrors can rate-limit under load.
5. Applies local bias correction:
   - Uses verified historical lead-time bias when enough samples exist.
   - Falls back to the latest observation/current-run overlap with exponential decay.
6. Writes:
   - `forecast_output.csv`: current corrected forecast
   - `forecast_Inaquito.html`: Plotly page with Max / Prom / Min buttons
   - `forecast_archive.csv`: long-format archive of every forecast run and lead time
   - `forecast_verification.csv`: archived forecasts matched to later observations
   - `forecast_metrics.csv`: MAE, RMSE, bias, coverage, and improvement by target and lead hour
   - `SYSTEM_STATUS.md`: latest operational report regenerated after each successful run
   - `forecast_metrics.html`: interactive metrics dashboard regenerated after each successful run

## Important Meteorological Notes

- `mx2t3` and `mn2t3` are 3-hour extrema, not exact station hourly extrema. This is more honest than deriving max/min from only `2t`, but the target definition mismatch still matters.
- The forecast is currently only for one point. It does not yet learn spatial gradients, elevation effects, valley effects, urban heat island effects, or neighboring-station consistency.
- The system now has a memory layer, but it needs a few successful hourly runs before historical lead-time calibration becomes active.
- Do not reintroduce SARIMAX/LSTM as the core forecaster. With one station and limited data, that adds complexity without improving the meteorological foundation.

## Checks To Run

Fast local checks:

```bash
python3 -c "import ast, pathlib; ast.parse(pathlib.Path('script.py').read_text()); print('syntax ok')"
ruby -e "require 'yaml'; YAML.load_file('.github/workflows/forecast.yml'); puts 'yaml ok'"
git diff --check
```

If dependencies are installed, run the built-in no-network self-test:

```bash
python3 script.py --self-test
```

Synthetic logic test without network or ECMWF dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import sys, types

plotly = types.ModuleType('plotly')
go = types.ModuleType('plotly.graph_objects')
class Figure:
    pass
class Scatter:
    def __init__(self, *args, **kwargs):
        pass
go.Figure = Figure
go.Scatter = Scatter
sys.modules['plotly'] = plotly
sys.modules['plotly.graph_objects'] = go

ecmwf = types.ModuleType('ecmwf')
opendata = types.ModuleType('ecmwf.opendata')
class Client:
    pass
opendata.Client = Client
sys.modules['ecmwf'] = ecmwf
sys.modules['ecmwf.opendata'] = opendata
sys.modules['cfgrib'] = types.ModuleType('cfgrib')

import pandas as pd
import script

run_time = pd.Timestamp('2026-04-12T06:00:00Z')
valid_time = pd.date_range('2026-04-12T09:00:00', periods=4, freq='3h')
forecast = pd.DataFrame({
    'valid_time': valid_time,
    'lead_hours': [3, 6, 9, 12],
    'ecmwf_max_c': [20.0, 21.0, 22.0, 21.5],
    'ecmwf_prom_c': [14.0, 15.0, 16.0, 15.5],
    'ecmwf_min_c': [10.0, 10.5, 11.0, 10.8],
})
station_targets = pd.DataFrame({
    'observed_max_c': [21.0, 22.0, 22.5, 21.0],
    'observed_prom_c': [15.0, 15.5, 16.5, 15.2],
    'observed_min_c': [9.0, 10.0, 10.6, 10.4],
}, index=valid_time)

bias = {
    key: script.compute_local_bias(forecast, station_targets, key, verification=pd.DataFrame(), run_time=run_time)
    for key in script.FORECAST_TARGETS
}
corrected = script.apply_bias_correction(forecast, bias)
archive = script.build_forecast_archive_rows(corrected, bias, run_time)
verified = script.verify_forecast_archive(archive, station_targets)
metrics = script.summarize_verification_metrics(verified)
assert len(archive) == 12
assert len(verified) == 12
assert len(metrics) == 12
print("synthetic forecast system test ok")
PY
```

Full local run, only if dependencies are installed and network is allowed:

```bash
python3 -m pip install -r requirements.txt
python3 script.py
```

GitHub Actions check:

```bash
curl -s --max-time 20 'https://api.github.com/repos/ccardenas93/weather_forecast/actions/runs?branch=master&per_page=3'
```

If a run fails, inspect whether it failed from external data availability or code:

- INAMHI failure usually shows HTTP 502 or no data. The script should continue with cached observations if a cache exists.
- ECMWF failure can happen from open-data portal limits or a missing parameter. If logs show HTTP 429 or S3 `SlowDown`, keep `ECMWF_SOURCES=azure,google,aws`; avoid defaulting back to direct `ecmwf` unless diagnosing manually.
- Code failures will show Python tracebacks in `script.py`; fix those first.

## Next Engineering Steps

1. Confirm the latest Action after the verification commit finishes.
2. Pull the Action output commit if it creates `forecast_archive.csv`, `forecast_verification.csv`, or `forecast_metrics.csv`.
3. Add unit tests in a real test file, not just the synthetic shell snippet.
4. Add more ECMWF predictors to the archive before using them in a model:
   - `2d` dewpoint
   - `tcc` total cloud cover
   - `ssrd` solar radiation downwards
   - `tp` precipitation
   - `10u`, `10v` wind
   - `sp` or `msl` pressure
5. Build a proper MOS/downscaling model after enough archive rows exist:
   - Start with lead-hour and hour-of-day grouped bias.
   - Then use a small tree or quantile model such as LightGBM/CatBoost if dependencies are acceptable.
   - Score against raw ECMWF and persistence before trusting it.
6. Add multi-station support:
   - Create a `stations.csv` or `stations.yml`.
   - Loop over stations and generate one page/file per station.
   - Include lat/lon/elevation metadata.
7. Consider moving long-term archives out of Git once files grow too much.

## Current Known Risk

The system depends on two external live data sources. A red Action is not automatically a model failure. First classify the failure as:

- data-source outage,
- ECMWF rate/availability issue,
- dependency install issue,
- code exception,
- git push conflict from hourly Action commits.

## Resume Here Tomorrow

User is pausing on 2026-04-12 and plans to continue in a new conversation.

Latest known good state:

- Latest pushed commit when this note was written: `d9b0152ad Fallback across ECMWF cloud sources`.
- User reported the workflow is working after the multi-source ECMWF fallback.
- The system now has Max / Prom / Min tabs and writes forecast archive, verification, and metrics files when the Action produces them.
- The current forecast should use `ECMWF_SOURCES=azure,google,aws`, not direct `ecmwf`, because direct ECMWF returned HTTP 429 and AWS returned S3 `SlowDown` under load.

When resuming:

1. Run `git pull --rebase origin master` first. The hourly Action may have pushed `Updated forecast data` commits while the user was asleep.
2. Check whether these files now exist on `master`:
   - `forecast_archive.csv`
   - `forecast_verification.csv`
   - `forecast_metrics.csv`
3. Check the latest Forecast Automation run status. If it failed, classify it before editing code:
   - INAMHI outage or stale data,
   - all ECMWF mirrors unavailable,
   - missing ECMWF parameter,
   - Python exception,
   - git conflict from concurrent hourly commits.
4. If the run is green, do not make the model more complex yet. First inspect `forecast_metrics.csv`.
5. Recommended next implementation:
   - Add stale ECMWF fallback: if all ECMWF sources fail, use the last successful `forecast_output.csv`/archive rows and mark the HTML as stale instead of failing the Action.
   - Add real tests for archive, verification, metrics, and historical lead-time bias.
6. Next meteorological upgrades after enough verified samples exist:
   - Add ECMWF predictors `2d`, `tcc`, `ssrd`, `tp`, `10u`, `10v`, and `sp`/`msl` to the archive.
   - Add hour-of-day calibration.
   - Add multi-station support with station metadata.
   - Only then consider a small MOS/downscaling model. Do not jump to LSTM.

Priority tomorrow: operational robustness first, smarter calibration second, more predictors third.
