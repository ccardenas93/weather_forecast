import pandas as pd

from forecast_config import DEFAULT_TARGET_KEY, FORECAST_HTML_PATH, FORECAST_TARGETS, STATION_NAME


def diagnostic_text(bias_info):
    if pd.notna(bias_info["raw_rmse_c"]):
        rmse_text = f"{bias_info['raw_rmse_c']:.2f} C"
    else:
        rmse_text = "n/a"
    history_text = ""
    if bias_info["historical_sample_count"]:
        history_text = (
            f" Verified archive samples: {bias_info['historical_sample_count']}; "
            f"calibrated leads: {len(bias_info['lead_bias_c'])}."
        )
    return (
        f"Bias correction: {bias_info['bias_c']:.2f} C ({bias_info['bias_source']}). "
        f"Current-run overlap RMSE: {rmse_text}.{history_text}"
    )


def target_title(target_key):
    spec = FORECAST_TARGETS[target_key]
    target_note = "3-hour " if target_key in {"max", "min"} else ""
    return (
        f"{STATION_NAME} {target_note}{spec['label']} Temperature Forecast: "
        "ECMWF with Local Bias Correction"
    )


def target_annotations(target_key, bias_info):
    latest_observation_time = bias_info["latest_observation_time"].to_pydatetime()
    return [
        dict(
            text="latest observation",
            x=latest_observation_time,
            y=1,
            xref="x",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        ),
        dict(
            text=diagnostic_text(bias_info),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.18,
            showarrow=False,
            align="left",
        ),
    ]


def build_plot(station_targets, forecast, bias_by_target):
    import plotly.graph_objects as go

    target_order = ["max", "prom", "min"]
    default_index = target_order.index(DEFAULT_TARGET_KEY)
    fig = go.Figure()

    for target_index, target_key in enumerate(target_order):
        spec = FORECAST_TARGETS[target_key]
        bias_info = bias_by_target[target_key]
        observed_column = f"observed_{target_key}_c"
        observed = station_targets[observed_column].dropna()
        if "is_operational_forecast" in forecast.columns:
            future = forecast[forecast["is_operational_forecast"]].copy()
        else:
            future = forecast[forecast["valid_time"] > bias_info["latest_observation_time"]].copy()
        if future.empty:
            future = forecast.copy()

        visible = target_index == default_index
        plot_values = pd.concat([
            observed.iloc[-120:],
            forecast[spec["raw_column"]],
            future[spec["lower_column"]],
            future[spec["upper_column"]],
            future[spec["forecast_column"]],
        ]).dropna()
        if plot_values.empty:
            marker_min, marker_max = 0, 1
        else:
            marker_min, marker_max = float(plot_values.min()), float(plot_values.max())
        latest_observation_time = bias_info["latest_observation_time"].to_pydatetime()

        fig.add_trace(go.Scatter(
            x=observed.index[-120:],
            y=observed.iloc[-120:],
            mode="lines",
            name=f"Observed {spec['label']}",
            visible=visible,
            line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=forecast["valid_time"],
            y=forecast[spec["raw_column"]],
            mode="lines+markers",
            name=f"Raw ECMWF {spec['label']}",
            visible=visible,
            line=dict(color="#2ca02c", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["upper_column"]],
            mode="lines",
            name=f"{spec['label']} upper band",
            visible=visible,
            showlegend=False,
            line=dict(color="rgba(214, 39, 40, 0.15)"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["lower_column"]],
            mode="lines",
            name=f"{spec['label']} uncertainty band",
            visible=visible,
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.15)",
            line=dict(color="rgba(214, 39, 40, 0.15)"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["forecast_column"]],
            mode="lines+markers",
            name=f"Bias-corrected {spec['label']}",
            visible=visible,
            line=dict(color="#d62728"),
        ))
        fig.add_trace(go.Scatter(
            x=future["valid_time"],
            y=future[spec["persistence_column"]],
            mode="lines",
            name=f"{spec['label']} persistence",
            visible=visible,
            line=dict(color="#7f7f7f", dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[latest_observation_time, latest_observation_time],
            y=[marker_min, marker_max],
            mode="lines",
            name="latest observation",
            visible=visible,
            showlegend=False,
            line=dict(color="#444444", dash="dot"),
        ))

    buttons = []
    traces_per_target = 7
    for target_index, target_key in enumerate(target_order):
        visible = [False] * len(fig.data)
        start = target_index * traces_per_target
        for trace_index in range(start, start + traces_per_target):
            visible[trace_index] = True
        buttons.append({
            "label": FORECAST_TARGETS[target_key]["label"],
            "method": "update",
            "args": [
                {"visible": visible},
                {
                    "title": target_title(target_key),
                    "annotations": target_annotations(target_key, bias_by_target[target_key]),
                },
            ],
        })

    fig.update_layout(
        title=target_title(DEFAULT_TARGET_KEY),
        xaxis_title="UTC time",
        yaxis_title="Temperature (C)",
        legend=dict(x=0.01, y=0.99),
        annotations=target_annotations(DEFAULT_TARGET_KEY, bias_by_target[DEFAULT_TARGET_KEY]),
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "right",
                "showactive": True,
                "active": default_index,
                "x": 0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        margin=dict(b=110),
    )
    fig.write_html(str(FORECAST_HTML_PATH))
