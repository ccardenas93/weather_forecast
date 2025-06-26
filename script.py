import pandas as pd
import requests

table_names = [
    "009010101h", "009010201h", "009010401h",  # HUMEDAD RELATIVA DEL AIRE (MAX, MIN, PROM)
    "017140801h",  # PRECIPITACION (SUM)
    "018070201h", "018070401h", "018070101h",  # PRESION ATMOSFERICA (MIN, PROM, MAX)
    "021200201h", "021200101h", "021200401h", "021200801h",  # RADIACION SOLAR GLOBAL (MIN, MAX, PROM, SUM)
    "022200201h", "022200401h", "022200101h",  # RADIACION SOLAR REFLEJADA (MIN, PROM, MAX)
    "029030101h", "029030401h", "029030201h",  # TEMPERATURA AIRE (MAX, PROM, MIN)
    #"004020101h", "037110101h", "004020201h", "037110201h", "037110401h"  # VIENTO DIRECCION - VIENTO VELOCIDAD
]

# Mapping from table code to descriptive variable name returned by the previous
# API. The new API only returns the code in the "nemonico" field, so we keep
# using human readable names in the resulting dataframe.
NAME_MAP = {
    "009010101h": "HUMEDAD RELATIVA DEL AIRE MAX",
    "009010201h": "HUMEDAD RELATIVA DEL AIRE MIN",
    "009010401h": "HUMEDAD RELATIVA DEL AIRE PROM",
    "017140801h": "PRECIPITACION SUM",
    "018070201h": "PRESION ATMOSFERICA MIN",
    "018070401h": "PRESION ATMOSFERICA PROM",
    "018070101h": "PRESION ATMOSFERICA MAX",
    "021200201h": "RADIACION SOLAR GLOBAL MIN",
    "021200101h": "RADIACION SOLAR GLOBAL MAX",
    "021200401h": "RADIACION SOLAR GLOBAL PROM",
    "021200801h": "RADIACION SOLAR GLOBAL SUM",
    "022200201h": "RADIACION SOLAR REFLEJADA MIN",
    "022200401h": "RADIACION SOLAR REFLEJADA MAX",
    "022200101h": "RADIACION SOLAR REFLEJADA PROM",
    "029030101h": "TEMPERATURA AIRE MAX",
    "029030401h": "TEMPERATURA AIRE PROM",
    "029030201h": "TEMPERATURA AIRE MIN",
}
def fetch_weather_data(start_date, end_date, station_id, table_names):
    """Retrieve hourly data from INAMHI API.

    The API endpoint changed in 2024.  The new service uses
    ``/station_data_hour/get_data_hour/`` and requires an ``id_aplication``
    parameter.  The response now returns the measurement code in the
    ``nemonico`` field and ``fecha_toma_dato`` for the timestamp.
    """

    url = "https://inamhi.gob.ec/api_rest/station_data_hour/get_data_hour/"
    payload = {
        "id_estacion": station_id,
        "table_names": table_names,
        "id_aplication": 1,
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data: {response.status_code}")

    data = response.json()

    flattened_data = []
    for measurement in data:
        code = measurement.get('nemonico')
        variable_name = NAME_MAP.get(code, code)
        for entry in measurement['data']:
            flattened_data.append({
                "fecha": entry['fecha_toma_dato'],
                variable_name: entry['valor']
            })

    df = pd.DataFrame(flattened_data)

    # Ensure that data is aligned properly by grouping by date and merging similar columns
    df = df.groupby('fecha').first().reset_index()

    return df

# Example usage
start_date = "2001-01-29T00:00:00"
end_date = "2024-12-30T00:00:00"
station_id = 63777
df_weather = fetch_weather_data(start_date, end_date, station_id, table_names)
df_weather.to_csv('merged_data_export.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import cfgrib
import xarray as xr

# Load and clean the dataset
file_path = 'merged_data_export.csv'  # Replace with the correct path
data = pd.read_csv(file_path, delimiter=',')

# Drop any unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Drop the 'TEMPERATURA AIRE MIN' column if present
if 'TEMPERATURA AIRE MIN' in data.columns:
    data_cleaned = data.drop(columns=['TEMPERATURA AIRE MIN'])
else:
    data_cleaned = data.copy()

# Convert the 'fecha' column to datetime format and set it as the index
data_cleaned['fecha'] = pd.to_datetime(data_cleaned['fecha'])
data_cleaned.set_index('fecha', inplace=True)

# Set the frequency to hourly ('H')
data_cleaned = data_cleaned.asfreq('H')

# Check and handle missing data
data_cleaned = data_cleaned.fillna(method='ffill').fillna(method='bfill')

# Select temperature data and exogenous variables
temperature_data = data_cleaned['TEMPERATURA AIRE MAX'].dropna()
exog_vars = data_cleaned.drop(columns=['TEMPERATURA AIRE MAX'])

# Align temperature data with exog_vars using reindex
temperature_data = temperature_data.reindex(exog_vars.index)

# Fit SARIMAX model with exogenous variables
model_sarimax = SARIMAX(temperature_data, exog=exog_vars,
                        order=(1, 1, 1), seasonal_order=(1, 1, 1, 24),
                        enforce_stationarity=False, enforce_invertibility=False)
results_sarimax = model_sarimax.fit(disp=False, maxiter=500)

# Forecast the next 24 hours using SARIMAX
forecast_steps = 24
forecast_sarimax = results_sarimax.get_forecast(steps=forecast_steps, exog=exog_vars[-forecast_steps:])
forecast_index = pd.date_range(start=temperature_data.index[-1] + pd.Timedelta(hours=1), periods=forecast_steps, freq='H')
forecast_sarimax_values = forecast_sarimax.predicted_mean

# Normalize only the temperature data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_data_normalized = scaler.fit_transform(temperature_data.values.reshape(-1, 1))

# Function to create dataset for LSTM model
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 24
X, Y = create_dataset(temperature_data_normalized, time_step)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Check if X_train and X_test have three dimensions, reshape if needed
if X_train.ndim == 2:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model_lstm = Sequential()
model_lstm.add(Input(shape=(time_step, 1)))
model_lstm.add(LSTM(50, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(25))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, Y_train, batch_size=16, epochs=100, verbose=1)

# Prepare input for LSTM forecast
X_input = temperature_data_normalized[-time_step:].reshape(1, time_step, 1)

# Generate LSTM forecast
forecasted_temps_lstm = []
for _ in range(forecast_steps):
    forecasted_temp_normalized = model_lstm.predict(X_input)
    forecasted_temps_lstm.append(forecasted_temp_normalized[0][0])
    X_input = np.roll(X_input, -1, axis=1)
    X_input[0, -1, 0] = forecasted_temp_normalized

# Convert LSTM forecast back to original scale
forecasted_temps_lstm = np.array(forecasted_temps_lstm).reshape(-1, 1)
forecasted_temps_lstm = scaler.inverse_transform(forecasted_temps_lstm)

# Calculate SARIMAX residuals and fit LSTM on them for the hybrid model
residuals = results_sarimax.resid
residuals_normalized = scaler.fit_transform(residuals.values.reshape(-1, 1))

X_residuals, Y_residuals = create_dataset(residuals_normalized, time_step)
X_residuals = X_residuals.reshape(X_residuals.shape[0], X_residuals.shape[1], 1)

# Split the residual data into training and testing sets
X_train_res, X_test_res = X_residuals[:train_size], X_residuals[train_size:]
Y_train_res, Y_test_res = Y_residuals[:train_size], Y_residuals[train_size:]

# Build LSTM model for residuals
model_lstm_res = Sequential()
model_lstm_res.add(Input(shape=(time_step, 1)))
model_lstm_res.add(LSTM(50, return_sequences=True))
model_lstm_res.add(Dropout(0.2))
model_lstm_res.add(LSTM(50, return_sequences=False))
model_lstm_res.add(Dropout(0.2))
model_lstm_res.add(Dense(25))
model_lstm_res.add(Dense(1))

model_lstm_res.compile(optimizer='adam', loss='mean_squared_error')
model_lstm_res.fit(X_train_res, Y_train_res, batch_size=16, epochs=100, verbose=1)

# Forecast residuals using LSTM
X_input_res = residuals_normalized[-time_step:].reshape(1, time_step, 1)
forecasted_residuals = []

for _ in range(forecast_steps):
    forecasted_residual = model_lstm_res.predict(X_input_res)
    forecasted_residuals.append(forecasted_residual[0][0])
    X_input_res = np.roll(X_input_res, -1, axis=1)
    X_input_res[0, -1, 0] = forecasted_residual

# Convert residuals back to original scale
forecasted_residuals = np.array(forecasted_residuals).reshape(-1, 1)
forecasted_residuals = scaler.inverse_transform(forecasted_residuals)

# Combine SARIMAX and LSTM residuals for hybrid forecast
hybrid_forecast = forecast_sarimax_values + forecasted_residuals.flatten()

# Download ECMWF Data
from ecmwf.opendata import Client

client = Client(source="ecmwf")
client.retrieve(
    time=6,
    type="fc",
    step=[0, 3, 6, 9, 12, 15, 18, 21, 24,27,30,33,36,39,42,45,48],
    param="2t",
    stream="oper",
    target="surface_temp_multi.grib2"
)

ds = cfgrib.open_dataset('surface_temp_multi.grib2')

lat = -0.178300
lon = -78.487700
temperature = ds['t2m'].sel(latitude=lat, longitude=lon, method='nearest')
temperature_celsius = temperature - 273.15
valid_times = ds['valid_time']

ecmwf_df = pd.DataFrame({'Time': valid_times.values, 'Temperature': temperature_celsius.values})

# Plot SARIMAX, LSTM, Hybrid, and ECMWF data
fig = go.Figure()
fig.add_trace(go.Scatter(x=temperature_data.index[-100:], y=temperature_data.iloc[-100:],
                         mode='lines', name='Observed TEMPERATURA_AIRE_MAX', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast_sarimax_values,
                         mode='lines', name='SARIMAX Forecast', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=forecast_index, y=forecasted_temps_lstm.flatten(),
                         mode='lines', name='LSTM Forecast', line=dict(color='red')))
fig.add_trace(go.Scatter(x=forecast_index, y=hybrid_forecast,
                         mode='lines', name='Hybrid SARIMAX-LSTM Forecast', line=dict(color='purple')))
fig.add_trace(go.Scatter(x=forecast_index,
                         y=forecast_sarimax.conf_int().iloc[:, 0],
                         mode='lines', name='SARIMAX Lower CI', line=dict(color='blue', dash='dash'), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_index,
                         y=forecast_sarimax.conf_int().iloc[:, 1],
                         mode='lines', name='SARIMAX Upper CI', line=dict(color='blue', dash='dash'), fill='tonexty', showlegend=False))
fig.add_trace(go.Scatter(x=ecmwf_df['Time'], y=ecmwf_df['Temperature'],
                         mode='lines', name='ECMWF Forecast', line=dict(color='green')))

# Add buttons to toggle SARIMAX confidence intervals on/off
fig.update_layout(
    title="TEMPERATURA_AIRE_MAX Forecast with SARIMAX, LSTM, Hybrid, and ECMWF Models INAQUITO",
    xaxis_title="Date",
    yaxis_title="TEMPERATURA_AIRE_MAX (°C)",
    legend=dict(x=0.01, y=0.99),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [{"visible": [True, True, True, True, False, False, True]}],
                    "label": "Hide SARIMAX CI",
                    "method": "update",
                },
                {
                    "args": [{"visible": [True, True, True, True, True, True, True]}],
                    "label": "Show SARIMAX CI",
                    "method": "update",
                }
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.17,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }
    ]
)

#save figure
fig.write_html('forecast_Inaquito.html')
