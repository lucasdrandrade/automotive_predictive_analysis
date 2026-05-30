import os
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

from data_loader import load_data, compute_stats, SENSOR_LABELS, ALL_FEATURE_LABELS
from charts import (
    condition_donut,
    sensor_means_bar,
    sensor_boxplots,
    correlation_heatmap,
    scatter_plot,
    line_chart,
    feature_importance_bar,
    derived_distributions,
    health_score_gauge,
    regression_importance_bar,
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "Data_Set", "engine_data.csv",
)
df_full = load_data(DATA_PATH)
rpm_min = int(df_full["engine_rpm"].min())
rpm_max = int(df_full["engine_rpm"].max())

sensor_options = [
    {"label": label.replace(" (°C)", ""), "value": col}
    for col, label in ALL_FEATURE_LABELS.items()
]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Engine Analytics",
)
server = app.server


def kpi_card(title, value_id, color="dark"):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted small mb-1"),
            html.H4(id=value_id, className=f"text-{color} fw-bold mb-0"),
        ]),
        className="shadow-sm h-100",
    )


app.layout = dbc.Container([

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.H3("Engine Health Analytics", className="mb-0 fw-bold"),
        html.Small("Automotive Predictive Maintenance Dashboard", className="text-white-50"),
    ], className="py-3 px-2")), className="bg-dark text-white mb-4 rounded-3"),

    # Filters
    dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([
            html.Label("Engine Condition", className="fw-semibold small text-muted"),
            dbc.RadioItems(
                id="condition-filter",
                options=[
                    {"label": "  All",     "value": "all"},
                    {"label": "  Healthy", "value": "1"},
                    {"label": "  Faulty",  "value": "0"},
                ],
                value="all",
                inline=True,
                className="mt-1",
            ),
        ], md=4),
        dbc.Col([
            html.Label("Engine RPM Range", className="fw-semibold small text-muted"),
            dcc.RangeSlider(
                id="rpm-range",
                min=rpm_min,
                max=rpm_max,
                step=10,
                value=[rpm_min, rpm_max],
                marks={rpm_min: str(rpm_min), rpm_max: str(rpm_max)},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mt-2",
            ),
        ], md=8),
    ])), className="shadow-sm mb-4"),

    # KPI Cards
    dbc.Row([
        dbc.Col(kpi_card("Total Readings",   "kpi-total"),            md=3, className="mb-3"),
        dbc.Col(kpi_card("Healthy Readings", "kpi-healthy", "success"), md=3, className="mb-3"),
        dbc.Col(kpi_card("Faulty Readings",  "kpi-faulty",  "danger"),  md=3, className="mb-3"),
        dbc.Col(kpi_card("Fault Rate",       "kpi-rate",    "warning"), md=3, className="mb-3"),
    ], className="g-3 mb-1"),

    # Feature Importance — Pearson + Logistic Regression side by side
    dbc.Row([
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.P(
                    "Univariate — measures each feature independently against the target. "
                    "Saturated = derived features, muted = raw sensors.",
                    className="text-muted small mb-0",
                ),
                dcc.Graph(id="feature-importance-chart", config={"displayModeBar": False}),
            ]), className="shadow-sm h-100"),
            md=6, className="mb-4",
        ),
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.P(
                    "Multivariate — coefficients from Logistic Regression with all features "
                    "standardised (mean=0, std=1). Controls for inter-feature correlations.",
                    className="text-muted small mb-0",
                ),
                dcc.Graph(id="regression-importance-chart", config={"displayModeBar": False}),
            ]), className="shadow-sm h-100"),
            md=6, className="mb-4",
        ),
    ], className="g-3"),

    # Donut + Grouped bar
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="donut-chart", config={"displayModeBar": False}), className="shadow-sm"), md=5, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id="bar-chart",   config={"displayModeBar": False}), className="shadow-sm"), md=7, className="mb-4"),
    ], className="g-3"),

    # Sensor box plots
    dbc.Row(dbc.Col(
        dbc.Card(dcc.Graph(id="box-chart"), className="shadow-sm"),
        className="mb-4",
    )),

    # Derived Features Analysis
    dbc.Row([
        dbc.Col(
            dbc.Card(dcc.Graph(id="health-gauge", config={"displayModeBar": False}), className="shadow-sm"),
            md=4, className="mb-4",
        ),
        dbc.Col(
            dbc.Card(dcc.Graph(id="derived-dist-chart"), className="shadow-sm"),
            md=8, className="mb-4",
        ),
    ], className="g-3"),

    # Correlation heatmap + Scatter
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="heatmap-chart"), className="shadow-sm"), md=6, className="mb-4"),
        dbc.Col(dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("X Axis", className="small fw-semibold text-muted"),
                    dcc.Dropdown(id="scatter-x", options=sensor_options, value="engine_rpm", clearable=False),
                ], md=6),
                dbc.Col([
                    html.Label("Y Axis", className="small fw-semibold text-muted"),
                    dcc.Dropdown(id="scatter-y", options=sensor_options, value="fuel_pressure", clearable=False),
                ], md=6),
            ], className="mb-2"),
            dcc.Graph(id="scatter-chart"),
        ]), className="shadow-sm"), md=6, className="mb-4"),
    ], className="g-3"),

    # Sequential line chart
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        dbc.Row(dbc.Col([
            html.Label("Sensor", className="small fw-semibold text-muted"),
            dcc.Dropdown(
                id="line-sensor",
                options=sensor_options,
                value="engine_rpm",
                clearable=False,
                style={"maxWidth": "300px"},
            ),
        ]), className="mb-2"),
        dcc.Graph(id="line-chart"),
    ]), className="shadow-sm"), className="mb-4")),

    # Data table
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.H6("Data Explorer", className="mb-0 fw-bold")),
            dbc.Col(html.Small(id="table-note", className="text-muted"), className="text-end"),
        ], className="mb-3 align-items-center"),
        html.Div(id="data-table"),
    ]), className="shadow-sm"), className="mb-4")),

], fluid=True, className="px-4 py-3", style={"backgroundColor": "#f0f2f5", "minHeight": "100vh"})


# --- Helpers ---

def apply_filters(condition, rpm_range):
    dff = df_full.copy()
    if condition != "all":
        dff = dff[dff["engine_condition"] == int(condition)]
    dff = dff[(dff["engine_rpm"] >= rpm_range[0]) & (dff["engine_rpm"] <= rpm_range[1])]
    return dff


# --- Callbacks ---

@app.callback(
    Output("kpi-total",    "children"),
    Output("kpi-healthy",  "children"),
    Output("kpi-faulty",   "children"),
    Output("kpi-rate",     "children"),
    Output("donut-chart",  "figure"),
    Output("bar-chart",    "figure"),
    Output("box-chart",    "figure"),
    Output("heatmap-chart","figure"),
    Input("condition-filter", "value"),
    Input("rpm-range",        "value"),
)
def update_main(condition, rpm_range):
    dff = apply_filters(condition, rpm_range)
    s = compute_stats(dff)
    return (
        f"{s['total']:,}",
        f"{s['healthy']:,}  ({s['healthy_rate']:.1f}%)",
        f"{s['faulty']:,}  ({s['fault_rate']:.1f}%)",
        f"{s['fault_rate']:.1f}%",
        condition_donut(dff),
        sensor_means_bar(dff),
        sensor_boxplots(dff),
        correlation_heatmap(dff),
    )


@app.callback(
    Output("scatter-chart", "figure"),
    Input("scatter-x",        "value"),
    Input("scatter-y",        "value"),
    Input("condition-filter", "value"),
    Input("rpm-range",        "value"),
)
def update_scatter(x_col, y_col, condition, rpm_range):
    return scatter_plot(apply_filters(condition, rpm_range), x_col, y_col)


@app.callback(
    Output("line-chart",   "figure"),
    Input("line-sensor",      "value"),
    Input("condition-filter", "value"),
    Input("rpm-range",        "value"),
)
def update_line(sensor_col, condition, rpm_range):
    return line_chart(apply_filters(condition, rpm_range), sensor_col)


@app.callback(
    Output("data-table", "children"),
    Output("table-note", "children"),
    Input("condition-filter", "value"),
    Input("rpm-range",        "value"),
)
def update_table(condition, rpm_range):
    dff = apply_filters(condition, rpm_range)
    total = len(dff)
    limit = 500
    dff_display = dff.head(limit).copy()
    dff_display["Status"] = dff_display["engine_condition"].map({1: "Healthy", 0: "Faulty"})

    rename = {k: v.replace(" (°C)", "") for k, v in SENSOR_LABELS.items()}
    rename["engine_condition"] = "Condition"
    dff_display = dff_display.rename(columns=rename)

    ordered = [rename.get(c, c) for c in ["engine_rpm", "lub_oil_pressure", "fuel_pressure",
               "coolant_pressure", "lub_oil_temp", "coolant_temp", "engine_condition"]] + ["Status"]
    dff_display = dff_display[[c for c in ordered if c in dff_display.columns]]

    table = dash_table.DataTable(
        data=dff_display.round(3).to_dict("records"),
        columns=[{"name": c, "id": c} for c in dff_display.columns],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontWeight": "bold",
            "fontSize": "13px",
        },
        style_cell={"fontSize": "12px", "padding": "8px", "fontFamily": "monospace"},
        style_data_conditional=[
            {"if": {"filter_query": '{Status} = "Faulty"'},  "backgroundColor": "#fff5f5", "color": "#c0392b"},
            {"if": {"filter_query": '{Status} = "Healthy"'}, "backgroundColor": "#f0faf4"},
        ],
    )
    note = f"Showing {min(limit, total):,} of {total:,} rows"
    return table, note


@app.callback(
    Output("feature-importance-chart",  "figure"),
    Output("regression-importance-chart", "figure"),
    Output("health-gauge",              "figure"),
    Output("derived-dist-chart",        "figure"),
    Input("condition-filter",           "value"),
    Input("rpm-range",                  "value"),
)
def update_analysis(condition, rpm_range):
    dff = apply_filters(condition, rpm_range)
    return (
        feature_importance_bar(dff),
        regression_importance_bar(dff),
        health_score_gauge(dff, df_full),
        derived_distributions(dff),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
