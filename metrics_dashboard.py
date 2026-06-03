from forecast_config import METRICS_HTML_PATH, STATION_NAME


def build_metrics_dashboard(metrics):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if metrics.empty:
        METRICS_HTML_PATH.write_text(
            "<html><body><h1>Forecast Metrics</h1><p>No verified metrics yet.</p></body></html>\n",
            encoding="utf-8",
        )
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("MAE by Lead Hour", "Coverage by Lead Hour"),
    )
    target_colors = {
        "max": "#d62728",
        "prom": "#1f77b4",
        "min": "#2ca02c",
    }
    series_styles = {
        "forecast_mae_c": ("forecast", "solid"),
        "raw_mae_c": ("raw ECMWF", "dash"),
        "persistence_mae_c": ("persistence", "dot"),
    }
    for target_key, group in metrics.groupby("target"):
        group = group.sort_values("lead_hour")
        color = target_colors.get(target_key, "#444444")
        for column, (label, dash) in series_styles.items():
            fig.add_trace(
                go.Scatter(
                    x=group["lead_hour"],
                    y=group[column],
                    mode="lines+markers",
                    name=f"{target_key} {label}",
                    line=dict(color=color, dash=dash),
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=group["lead_hour"],
                y=group["coverage_rate"],
                mode="lines+markers",
                name=f"{target_key} coverage",
                line=dict(color=color),
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0.8, row=2, col=1, line=dict(color="#777777", dash="dash"))
    fig.update_yaxes(title_text="MAE (C)", row=1, col=1)
    fig.update_yaxes(title_text="Coverage rate", row=2, col=1, range=[0, 1.05])
    fig.update_xaxes(title_text="Lead hour", row=2, col=1)
    fig.update_layout(
        title=f"{STATION_NAME} Forecast Verification Metrics",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(b=150),
    )
    fig.write_html(str(METRICS_HTML_PATH))
