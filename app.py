"""
Olist E-Commerce Analytics Dashboard
=====================================
Run locally:   python app.py
Deploy:        gunicorn app:server

Expects processed/ directory with:
  - master_orders.csv
  - churn_predictions.csv
  - customer_segments.csv
  - churn_with_segments.csv
  - demand_forecasts.csv
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ── THEME ───────────────────────────────────────────────────────────────
BG = "#111410"
BG_CARD = "#181a16"
GREEN = "#4ade80"
BLUE = "#60a5fa"
ORANGE = "#fb923c"
PURPLE = "#c084fc"
YELLOW = "#fbbf24"
TEXT = "#e4e8e0"
MUTED = "#6a7068"
BORDER = "#2a2e28"

PLOT_LAYOUT = dict(
    plot_bgcolor=BG,
    paper_bgcolor=BG_CARD,
    font=dict(color=TEXT, size=12),
    margin=dict(t=40, b=30, l=10, r=10),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

SEGMENT_COLORS = {
    "Champions": GREEN,
    "Loyal Customers": BLUE,
    "At Risk": ORANGE,
    "One-Time Buyers": PURPLE,
    "New Customers": YELLOW,
}


# ── LOAD DATA ────────────────────────────────────────────────────────────
def load_data():
    master = pd.read_csv(
        "processed/master_orders.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    churn = pd.read_csv("processed/churn_predictions.csv")
    segs = pd.read_csv("processed/customer_segments.csv")
    churn_s = pd.read_csv("processed/churn_with_segments.csv")
    forecasts = pd.read_csv("processed/demand_forecasts.csv", parse_dates=["ds"])
    return master, churn, segs, churn_s, forecasts


try:
    master, churn, segs, churn_s, forecasts = load_data()
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    LOAD_ERROR = str(e)

# ── PRE-COMPUTE SERIES ────────────────────────────────────────────────────
if DATA_LOADED:
    delivered = master[master["order_status"] == "delivered"].copy()
    reviewed = delivered[delivered["review_score"].notna()].copy()
    revenue_df = delivered[delivered["order_value"].notna()].copy()

    # monthly GMV
    monthly = (
        revenue_df.set_index("order_purchase_timestamp")
        .resample("ME")["order_value"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "gmv", "count": "n_orders"})
        .iloc[1:-1]
    )

    # delay buckets
    bins = [-999, -7, 0, 3, 7, 14, 999]
    labels = [
        "Early >7d",
        "On time",
        "1-3d late",
        "4-7d late",
        "8-14d late",
        ">14d late",
    ]
    reviewed["delay_bucket"] = pd.cut(
        reviewed["delivery_delay_days"], bins=bins, labels=labels
    )

    bucket_stats = (
        reviewed.groupby("delay_bucket", observed=True)["review_score"]
        .mean()
        .reset_index()
        .rename(columns={"review_score": "avg_review"})
    )

    # segment summary
    seg_summary = (
        segs.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .round(1)
    )

    # churn by segment
    seg_churn = (
        (
            churn_s.groupby("segment")
            .agg(avg_churn_prob=("churn_probability", "mean"))
            .reset_index()
            .round(3)
        )
        if "segment" in churn_s.columns
        else pd.DataFrame()
    )


# ── LAYOUT HELPERS ────────────────────────────────────────────────────────
def card(children, style=None):
    base = {
        "background": BG_CARD,
        "borderRadius": "8px",
        "border": f"1px solid {BORDER}",
        "padding": "16px 20px",
        "marginBottom": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def kpi(label, value, color=GREEN):
    return html.Div(
        [
            html.Div(
                value, style={"fontSize": "28px", "fontWeight": "700", "color": color}
            ),
            html.Div(
                label, style={"fontSize": "12px", "color": MUTED, "marginTop": "2px"}
            ),
        ],
        style={"textAlign": "center", "flex": "1"},
    )


def section_title(text):
    return html.H5(
        text,
        style={
            "color": TEXT,
            "marginBottom": "12px",
            "fontWeight": "600",
            "fontSize": "14px",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
        },
    )


# ── TABS CONTENT ─────────────────────────────────────────────────────────
def tab_overview():
    if not DATA_LOADED:
        return html.Div(
            f"Data load error: {LOAD_ERROR}", style={"color": ORANGE, "padding": "20px"}
        )

    total_orders = len(delivered)
    total_gmv = revenue_df["order_value"].sum()
    avg_review = reviewed["review_score"].mean()
    late_pct = (delivered["delivery_delay_days"] > 0).mean() * 100

    fig_gmv = go.Figure()
    fig_gmv.add_trace(
        go.Bar(
            x=monthly["order_purchase_timestamp"],
            y=monthly["gmv"],
            marker_color=GREEN,
            name="GMV",
        )
    )
    fig_gmv.update_layout(title="Monthly GMV (R$)", height=280, **PLOT_LAYOUT)

    fig_reviews = go.Figure()
    fig_reviews.add_trace(
        go.Bar(
            x=bucket_stats["delay_bucket"].astype(str),
            y=bucket_stats["avg_review"],
            marker_color=[GREEN, GREEN, YELLOW, ORANGE, ORANGE, "#7f1d1d"],
        )
    )
    fig_reviews.update_layout(
        title="Avg Review Score by Delivery Delay", height=280, **PLOT_LAYOUT
    )

    return html.Div(
        [
            # KPI row
            card(
                html.Div(
                    [
                        kpi("Total Orders", f"{total_orders:,}"),
                        kpi("Total GMV", f"R$ {total_gmv/1e6:.1f}M", BLUE),
                        kpi("Avg Review Score", f"{avg_review:.2f}", YELLOW),
                        kpi("% Orders Late", f"{late_pct:.1f}%", ORANGE),
                    ],
                    style={"display": "flex", "gap": "8px"},
                )
            ),
            # charts row
            html.Div(
                [
                    html.Div(
                        card(
                            [
                                section_title("Revenue Over Time"),
                                dcc.Graph(
                                    figure=fig_gmv, config={"displayModeBar": False}
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        card(
                            [
                                section_title("Delivery vs Customer Satisfaction"),
                                dcc.Graph(
                                    figure=fig_reviews, config={"displayModeBar": False}
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px"},
            ),
        ]
    )


def tab_churn():
    if not DATA_LOADED:
        return html.Div("Data not loaded", style={"color": ORANGE})

    churn_rate = churn["churned"].mean() * 100
    avg_prob = churn["churn_probability"].mean() * 100

    fig_hist = px.histogram(
        churn,
        x="churn_probability",
        nbins=50,
        color_discrete_sequence=[PURPLE],
        labels={"churn_probability": "Predicted Churn Probability"},
        title="Distribution of Churn Probability Scores",
    )
    fig_hist.update_layout(height=300, **PLOT_LAYOUT)

    rows = []
    if not seg_churn.empty:
        fig_seg = px.bar(
            seg_churn.sort_values("avg_churn_prob", ascending=False),
            x="segment",
            y="avg_churn_prob",
            color="avg_churn_prob",
            color_continuous_scale="RdYlGn_r",
            title="Avg Churn Probability by Segment",
        )
        fig_seg.update_layout(height=300, **PLOT_LAYOUT)
        rows.append(
            card(
                [
                    section_title("Churn Risk by Segment"),
                    dcc.Graph(figure=fig_seg, config={"displayModeBar": False}),
                ]
            )
        )

    return html.Div(
        [
            card(
                html.Div(
                    [
                        kpi("Actual Churn Rate", f"{churn_rate:.1f}%", ORANGE),
                        kpi("Avg Predicted Churn Prob", f"{avg_prob:.1f}%", PURPLE),
                        kpi("Customers Scored", f"{len(churn):,}"),
                    ],
                    style={"display": "flex", "gap": "8px"},
                )
            ),
            card(
                [
                    section_title("Churn Score Distribution"),
                    dcc.Graph(figure=fig_hist, config={"displayModeBar": False}),
                ]
            ),
            *rows,
        ]
    )


def tab_segments():
    if not DATA_LOADED:
        return html.Div("Data not loaded", style={"color": ORANGE})

    colors = [SEGMENT_COLORS.get(s, MUTED) for s in seg_summary["segment"]]

    fig_size = go.Figure(
        go.Bar(
            x=seg_summary["segment"], y=seg_summary["n_customers"], marker_color=colors
        )
    )
    fig_size.update_layout(title="Customers per Segment", height=280, **PLOT_LAYOUT)

    fig_spend = go.Figure(
        go.Bar(
            x=seg_summary["segment"], y=seg_summary["avg_monetary"], marker_color=colors
        )
    )
    fig_spend.update_layout(
        title="Avg Total Spend per Segment (R$)", height=280, **PLOT_LAYOUT
    )

    sample = segs.sample(min(3000, len(segs)), random_state=42)
    seg_colors_list = [SEGMENT_COLORS.get(s, MUTED) for s in sample["segment"]]

    fig_scatter = go.Figure(
        go.Scatter(
            x=sample["recency"],
            y=sample["monetary"],
            mode="markers",
            marker=dict(color=seg_colors_list, size=4, opacity=0.5),
            text=sample["segment"],
        )
    )
    fig_scatter.update_layout(
        title="Recency vs Monetary (sampled 3k)",
        xaxis_title="Recency (days)",
        yaxis_title="Total Spend (R$)",
        height=350,
        **PLOT_LAYOUT,
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        card(
                            [
                                section_title("Segment Size"),
                                dcc.Graph(
                                    figure=fig_size, config={"displayModeBar": False}
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        card(
                            [
                                section_title("Avg Spend"),
                                dcc.Graph(
                                    figure=fig_spend, config={"displayModeBar": False}
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px"},
            ),
            card(
                [
                    section_title("Recency vs Spend by Segment"),
                    dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}),
                ]
            ),
        ]
    )


def tab_forecast():
    if not DATA_LOADED:
        return html.Div("Data not loaded", style={"color": ORANGE})

    cats = forecasts["category"].unique().tolist()
    default_cat = cats[0] if cats else None

    return html.Div(
        [
            card(
                [
                    section_title("Select Category"),
                    dcc.Dropdown(
                        id="forecast-category-dropdown",
                        options=[{"label": c, "value": c} for c in cats],
                        value=default_cat,
                        style={"background": BG, "color": TEXT, "borderColor": BORDER},
                        className="dash-dropdown",
                    ),
                ]
            ),
            card(
                [
                    section_title("Demand Forecast"),
                    dcc.Graph(id="forecast-chart", config={"displayModeBar": False}),
                ]
            ),
        ]
    )


# ── APP INIT ──────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)
server = app.server  # for gunicorn

NAV_STYLE = {
    "background": BG_CARD,
    "borderBottom": f"1px solid {BORDER}",
    "padding": "12px 24px",
    "display": "flex",
    "alignItems": "center",
    "gap": "32px",
}

TAB_STYLE = {
    "color": MUTED,
    "padding": "8px 16px",
    "border": "none",
    "background": "none",
    "cursor": "pointer",
    "fontSize": "13px",
    "fontWeight": "500",
    "letterSpacing": "0.03em",
}

app.layout = html.Div(
    style={"background": BG, "minHeight": "100vh", "fontFamily": "Inter, sans-serif"},
    children=[
        # nav bar
        html.Div(
            [
                html.Span(
                    "Olist Analytics",
                    style={"color": GREEN, "fontWeight": "700", "fontSize": "16px"},
                ),
                dcc.Tabs(
                    id="main-tabs",
                    value="overview",
                    style={"flex": "1"},
                    children=[
                        dcc.Tab(
                            label="Overview",
                            value="overview",
                            style=TAB_STYLE,
                            selected_style={
                                **TAB_STYLE,
                                "color": GREEN,
                                "borderBottom": f"2px solid {GREEN}",
                            },
                        ),
                        dcc.Tab(
                            label="Churn",
                            value="churn",
                            style=TAB_STYLE,
                            selected_style={
                                **TAB_STYLE,
                                "color": GREEN,
                                "borderBottom": f"2px solid {GREEN}",
                            },
                        ),
                        dcc.Tab(
                            label="Segments",
                            value="segments",
                            style=TAB_STYLE,
                            selected_style={
                                **TAB_STYLE,
                                "color": GREEN,
                                "borderBottom": f"2px solid {GREEN}",
                            },
                        ),
                        dcc.Tab(
                            label="Forecasting",
                            value="forecast",
                            style=TAB_STYLE,
                            selected_style={
                                **TAB_STYLE,
                                "color": GREEN,
                                "borderBottom": f"2px solid {GREEN}",
                            },
                        ),
                    ],
                ),
            ],
            style=NAV_STYLE,
        ),
        # main content
        html.Div(
            id="tab-content",
            style={"padding": "20px 28px", "maxWidth": "1400px", "margin": "0 auto"},
        ),
    ],
)


# ── CALLBACKS ──────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "overview":
        return tab_overview()
    if tab == "churn":
        return tab_churn()
    if tab == "segments":
        return tab_segments()
    if tab == "forecast":
        return tab_forecast()
    return html.Div("Unknown tab")


@app.callback(
    Output("forecast-chart", "figure"),
    Input("forecast-category-dropdown", "value"),
    prevent_initial_call=True,
)
def update_forecast(category):
    if not category or not DATA_LOADED:
        return go.Figure()

    fc = forecasts[forecasts["category"] == category].sort_values("ds")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color=GREEN, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(74,222,128,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",
        )
    )
    fig.update_layout(
        title=f"Demand Forecast — {category}",
        xaxis_title="Date",
        yaxis_title="Daily Orders",
        height=400,
        **PLOT_LAYOUT,
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
