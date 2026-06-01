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
    title="Análise de Motor",
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

    # Cabeçalho
    dbc.Row(dbc.Col(html.Div([
        html.H3("Análise de Saúde do Motor", className="mb-0 fw-bold"),
        html.Small("Dashboard de Manutenção Preditiva Automotiva", className="text-white-50"),
    ], className="py-3 px-2")), className="bg-dark text-white mb-4 rounded-3"),

    # Filtros
    dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([
            html.Label("Condição do Motor", className="fw-semibold small text-muted"),
            dbc.RadioItems(
                id="condition-filter",
                options=[
                    {"label": "  Todos",     "value": "all"},
                    {"label": "  Saudável",  "value": "1"},
                    {"label": "  Com Falha", "value": "0"},
                ],
                value="all",
                inline=True,
                className="mt-1",
            ),
        ], md=4),
        dbc.Col([
            html.Label("Faixa de RPM do Motor", className="fw-semibold small text-muted"),
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

    # KPIs
    dbc.Row([
        dbc.Col(kpi_card("Total de Leituras",  "kpi-total"),            md=3, className="mb-3"),
        dbc.Col(kpi_card("Leituras Saudáveis", "kpi-healthy", "success"), md=3, className="mb-3"),
        dbc.Col(kpi_card("Leituras com Falha", "kpi-faulty",  "danger"),  md=3, className="mb-3"),
        dbc.Col(kpi_card("Taxa de Falhas",     "kpi-rate",    "warning"), md=3, className="mb-3"),
    ], className="g-3 mb-1"),

    # Importância de Variáveis — Pearson + Regressão Logística lado a lado
    dbc.Row([
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.P(
                    "Univariada — mede cada variável independentemente em relação ao alvo. "
                    "Cores saturadas = variáveis derivadas, suaves = sensores brutos.",
                    className="text-muted small mb-0",
                ),
                dcc.Graph(id="feature-importance-chart", config={"displayModeBar": False}),
            ]), className="shadow-sm h-100"),
            md=6, className="mb-4",
        ),
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.P(
                    "Multivariada — coeficientes de Regressão Logística com todas as variáveis "
                    "padronizadas (média=0, desvio=1). Controla correlações entre variáveis.",
                    className="text-muted small mb-0",
                ),
                dcc.Graph(id="regression-importance-chart", config={"displayModeBar": False}),
            ]), className="shadow-sm h-100"),
            md=6, className="mb-4",
        ),
    ], className="g-3"),

    # Rosca + Barras agrupadas
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="donut-chart", config={"displayModeBar": False}), className="shadow-sm"), md=5, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id="bar-chart",   config={"displayModeBar": False}), className="shadow-sm"), md=7, className="mb-4"),
    ], className="g-3"),

    # Box plots dos sensores
    dbc.Row(dbc.Col(
        dbc.Card(dcc.Graph(id="box-chart"), className="shadow-sm"),
        className="mb-4",
    )),

    # Análise de Variáveis Derivadas
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

    # Mapa de correlação + Dispersão
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="heatmap-chart"), className="shadow-sm"), md=6, className="mb-4"),
        dbc.Col(dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Eixo X", className="small fw-semibold text-muted"),
                    dcc.Dropdown(id="scatter-x", options=sensor_options, value="engine_rpm", clearable=False),
                ], md=6),
                dbc.Col([
                    html.Label("Eixo Y", className="small fw-semibold text-muted"),
                    dcc.Dropdown(id="scatter-y", options=sensor_options, value="fuel_pressure", clearable=False),
                ], md=6),
            ], className="mb-2"),
            dcc.Graph(id="scatter-chart"),
        ]), className="shadow-sm"), md=6, className="mb-4"),
    ], className="g-3"),

    # Gráfico de linha sequencial
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

    # Tabela de dados
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.H6("Explorador de Dados", className="mb-0 fw-bold")),
            dbc.Col(html.Small(id="table-note", className="text-muted"), className="text-end"),
        ], className="mb-3 align-items-center"),
        html.Div(id="data-table"),
    ]), className="shadow-sm"), className="mb-4")),

], fluid=True, className="px-4 py-3", style={"backgroundColor": "#f0f2f5", "minHeight": "100vh"})


# --- Auxiliares ---

def apply_filters(condition, rpm_range):
    dff = df_full.copy()
    if condition != "all":
        dff = dff[dff["engine_condition"] == int(condition)]
    dff = dff[(dff["engine_rpm"] >= rpm_range[0]) & (dff["engine_rpm"] <= rpm_range[1])]
    return dff


# --- Callbacks (retornos de chamada) ---

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
    dff_display["Status"] = dff_display["engine_condition"].map({1: "Saudável", 0: "Com Falha"})

    rename = {k: v.replace(" (°C)", "") for k, v in SENSOR_LABELS.items()}
    rename["engine_condition"] = "Condição"
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
            {"if": {"filter_query": '{Status} = "Com Falha"'}, "backgroundColor": "#fff5f5", "color": "#c0392b"},
            {"if": {"filter_query": '{Status} = "Saudável"'}, "backgroundColor": "#f0faf4"},
        ],
    )
    note = f"Exibindo {min(limit, total):,} de {total:,} registros"
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
