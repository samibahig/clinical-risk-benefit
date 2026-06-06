import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import io
import base64

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Clinical Benefit–Risk",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


# ─── Decision Engine ─────────────────────────────────────────────────────────

def classify_benefit_risk(
    df: pd.DataFrame,
    efficacy_thresh: float = 0.50,
    risk_thresh: float = 0.30,
    n_min: int = 50,
) -> pd.DataFrame:
    """
    Classify drug-indication pairs into recommendation classes.

    Rules:
      uncertain  → sample_size < n_min  (insufficient evidence)
      preferred  → efficacy ≥ thresh AND risk ≤ thresh
      conditional→ one condition met (moderate tradeoff)
      avoid      → efficacy < thresh AND risk > thresh
    """
    df = df.copy()

    high_eff  = df["efficacy"] >= efficacy_thresh
    low_risk  = df["risk"]     <= risk_thresh
    enough_n  = df["sample_size"] >= n_min

    df["decision_class"] = "avoid"
    df.loc[high_eff &  low_risk,  "decision_class"] = "preferred"
    df.loc[high_eff & ~low_risk,  "decision_class"] = "conditional"
    df.loc[~high_eff & low_risk,  "decision_class"] = "conditional"
    df.loc[~enough_n,             "decision_class"] = "uncertain"

    # Net benefit score (for ranking table)
    df["net_benefit"] = (
        df["efficacy"] - df["risk"]
    ) * np.log(df["sample_size"].clip(lower=1))

    # Benefit-risk ratio
    df["br_ratio"] = df["efficacy"] / (df["risk"].clip(lower=0.01))

    return df


# ─── Sample Data ─────────────────────────────────────────────────────────────

def generate_sample_data() -> pd.DataFrame:
    np.random.seed(7)

    entries = []

    # Oncology
    onco = [
        ("Pembrolizumab",    "NSCLC",           0.82, 0.18, 520),
        ("Nivolumab",        "Melanoma",         0.79, 0.22, 418),
        ("Ipilimumab",       "Melanoma",         0.48, 0.52, 212),
        ("Bevacizumab",      "CRC",              0.55, 0.38, 340),
        ("Cisplatin",        "Ovarian cancer",   0.61, 0.72, 290),
        ("Carboplatin",      "Lung cancer",      0.54, 0.48, 280),
        ("Paclitaxel",       "Breast cancer",    0.59, 0.55, 310),
        ("Olaparib",         "BRCA+ breast ca.", 0.77, 0.28, 195),
        ("Trastuzumab",      "HER2+ breast ca.", 0.84, 0.15, 580),
        ("Erlotinib",        "EGFR+ NSCLC",      0.70, 0.31, 265),
        ("Sunitinib",        "RCC",              0.62, 0.60, 225),
        ("Sorafenib",        "HCC",              0.44, 0.58, 182),
        ("Bortezomib",       "Multiple myeloma", 0.73, 0.42, 340),
        ("Lenalidomide",     "Multiple myeloma", 0.78, 0.34, 390),
        ("Rituximab",        "DLBCL",            0.80, 0.20, 512),
    ]

    # Cardiology
    cardio = [
        ("Sacubitril/Val.",  "Heart failure",    0.78, 0.14, 8442),
        ("Empagliflozin",    "HFrEF",            0.75, 0.12, 3730),
        ("Dapagliflozin",    "HFrEF",            0.73, 0.13, 4744),
        ("Amiodarone",       "AFib",             0.68, 0.61, 1220),
        ("Warfarin",         "AFib stroke prev.", 0.72, 0.44, 6500),
        ("Apixaban",         "AFib stroke prev.", 0.79, 0.19, 18201),
        ("Rivaroxaban",      "AFib stroke prev.", 0.76, 0.22, 14264),
        ("Atorvastatin",     "Primary prevention",0.64, 0.08, 10305),
        ("Rosuvastatin",     "Secondary prev.",  0.71, 0.09, 17802),
        ("Metoprolol",       "Post-MI",          0.69, 0.11, 3840),
        ("Spironolactone",   "Heart failure",    0.65, 0.25, 1663),
        ("Ivabradine",       "Chronic HF",       0.52, 0.16, 6505),
        ("Dronedarone",      "AFib",             0.51, 0.38, 4628),
        ("Digoxin",          "Heart failure",    0.40, 0.41, 3397),
        ("Nesiritide",       "Acute HF",         0.35, 0.58, 7141),
    ]

    # Psychiatry / Neurology
    psych = [
        ("Sertraline",       "Major depression", 0.65, 0.15, 4200),
        ("Escitalopram",     "GAD",              0.67, 0.12, 3800),
        ("Venlafaxine",      "MDD",              0.68, 0.20, 2900),
        ("Aripiprazole",     "Schizophrenia",    0.64, 0.24, 2300),
        ("Clozapine",        "Refractory schiz.",0.80, 0.55, 890),
        ("Haloperidol",      "Acute psychosis",  0.70, 0.62, 1200),
        ("Valproate",        "Bipolar I",        0.72, 0.41, 1850),
        ("Lithium",          "Bipolar I",        0.76, 0.38, 3100),
        ("Lamotrigine",      "Bipolar II",       0.68, 0.18, 2600),
        ("Lecanemab",        "Early Alzheimer's",0.37, 0.25, 1795),
        ("Aducanumab",       "Alzheimer's",      0.22, 0.32,  856),
        ("Methylphenidate",  "ADHD",             0.78, 0.19, 5200),
        ("Atomoxetine",      "ADHD",             0.62, 0.17, 3100),
    ]

    # Infectious disease
    infect = [
        ("Dolutegravir",     "HIV-1",            0.92, 0.08, 4200),
        ("Bictegravir",      "HIV-1",            0.91, 0.07, 3800),
        ("Sofosbuvir",       "Hepatitis C",      0.96, 0.05, 5200),
        ("Velpatasvir",      "Hepatitis C",      0.97, 0.06, 4800),
        ("Nirmatrelvir",     "COVID-19",         0.88, 0.12, 2246),
        ("Molnupiravir",     "COVID-19",         0.48, 0.12, 1433),
        ("Remdesivir",       "COVID-19 severe",  0.38, 0.22, 1062),
        ("Cefazolin",        "Surgical prophyl.",0.85, 0.06, 8200),
        ("Vancomycin",       "MRSA",             0.78, 0.28, 2900),
        ("Linezolid",        "VRE",              0.74, 0.35, 1400),
        ("Fluconazole",      "Candidiasis",      0.82, 0.10, 3600),
        ("Amphotericin B",   "Aspergillosis",    0.68, 0.64, 1100),
    ]

    # Add random noise to simulate realistic variability
    all_entries = onco + cardio + psych + infect
    for drug, indication, eff, risk, n in all_entries:
        eff_noisy  = float(np.clip(eff  + np.random.normal(0, 0.03), 0.01, 0.99))
        risk_noisy = float(np.clip(risk + np.random.normal(0, 0.02), 0.01, 0.99))
        n_noisy    = int(max(10, n + np.random.randint(-int(n * 0.05), int(n * 0.05) + 1)))
        ci_width   = float(1.96 * np.sqrt(eff_noisy * (1 - eff_noisy) / n_noisy) * 2)
        entries.append({
            "drug":        drug,
            "indication":  indication,
            "efficacy":    round(eff_noisy, 3),
            "risk":        round(risk_noisy, 3),
            "sample_size": n_noisy,
            "ci_width":    round(ci_width, 4),
        })

    return pd.DataFrame(entries)


SAMPLE_DF = generate_sample_data()


# ─── Design tokens ───────────────────────────────────────────────────────────

BG       = "#f8fafc"
SURFACE  = "#ffffff"
BORDER   = "#e2e8f0"
TEXT_PRI = "#0f172a"
TEXT_SEC = "#64748b"
TEXT_TER = "#94a3b8"
ACCENT   = "#2563eb"

COLOR_MAP = {
    "preferred":   "#16a34a",
    "conditional": "#d97706",
    "avoid":       "#dc2626",
    "uncertain":   "#94a3b8",
}

FILL_MAP = {
    "preferred":   "rgba(22, 163, 74, 0.07)",
    "conditional": "rgba(217, 119, 6, 0.07)",
    "avoid":       "rgba(220, 38, 38, 0.07)",
    "uncertain":   "rgba(148, 163, 184, 0.05)",
}

X_OPTIONS = [
    {"label": "Efficacy (response rate)",             "value": "efficacy"},
    {"label": "Inverse Risk (1 – AE rate)",           "value": "inv_risk"},
    {"label": "Benefit-Risk Ratio",                   "value": "br_ratio"},
]

Y_OPTIONS = [
    {"label": "Risk (adverse event rate)",            "value": "risk"},
    {"label": "Inverse Efficacy (1 – response rate)", "value": "inv_efficacy"},
]

SIZE_OPTIONS = [
    {"label": "Sample size (evidence weight)",        "value": "sample_size"},
    {"label": "Inverse CI width (precision)",         "value": "inv_ci_width"},
]

INDICATION_OPTIONS = [
    {"label": "All areas", "value": "all"},
    {"label": "Oncology",  "value": "onco"},
    {"label": "Cardiology","value": "cardio"},
    {"label": "Psychiatry / Neurology", "value": "psych"},
    {"label": "Infectious disease",     "value": "infect"},
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def label(text):
    return html.P(text, style={
        "fontSize": "0.67rem", "color": TEXT_SEC,
        "letterSpacing": "1.2px", "fontWeight": 700,
        "textTransform": "uppercase", "marginBottom": "5px", "marginTop": "12px",
    })


# ─── Layout ──────────────────────────────────────────────────────────────────

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "0",
           "fontFamily": "Inter, system-ui, sans-serif"},
    children=[
        dcc.Store(id="data-store"),

        # Header
        html.Div([
            html.Div([
                html.Div([
                    html.Div("⚖️", style={"fontSize": "1.7rem", "marginRight": "12px"}),
                    html.Div([
                        html.H1("Clinical Benefit–Risk Analyzer",
                                style={"margin": 0, "fontSize": "1.35rem", "fontWeight": 700,
                                       "color": TEXT_PRI, "letterSpacing": "-0.3px"}),
                        html.P("Efficacy vs. risk decision surface · HTA · Oncology · Cardiology · Neurology",
                               style={"margin": 0, "fontSize": "0.78rem", "color": TEXT_SEC}),
                    ]),
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div([
                    dbc.Badge("HTA workflows", color="light", text_color="secondary",
                              className="me-2 border", style={"fontSize": "0.72rem"}),
                    dbc.Badge("EMA benefit-risk framework", color="primary",
                              style={"fontSize": "0.72rem"}),
                ]),
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "padding": "16px 28px",
                "backgroundColor": SURFACE,
                "borderBottom": f"1px solid {BORDER}",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
            }),
        ]),

        dbc.Row([
            # ── Left sidebar ─────────────────────────────────────────────────
            dbc.Col([
                html.Div([
                    label("Data Source"),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            html.Div("📂", style={"fontSize": "1.3rem"}),
                            html.Div("Drop CSV or click to upload",
                                     style={"fontSize": "0.8rem", "color": TEXT_SEC, "marginTop": "4px"}),
                            html.Div("Columns: drug, indication, efficacy, risk, sample_size",
                                     style={"fontSize": "0.68rem", "color": TEXT_TER, "marginTop": "2px"}),
                        ], style={"textAlign": "center", "padding": "12px"}),
                        style={"border": f"1.5px dashed {BORDER}", "borderRadius": "8px",
                               "cursor": "pointer", "backgroundColor": "#f1f5f9", "marginBottom": "8px"},
                    ),
                    dbc.Button("Load sample data (55 drugs)", id="load-sample",
                               size="sm", color="primary", outline=True,
                               className="w-100 mb-1", style={"fontSize": "0.75rem"}),

                    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

                    label("X-Axis"),
                    dcc.Dropdown(id="x-metric", options=X_OPTIONS, value="efficacy",
                                 clearable=False, style={"fontSize": "0.82rem", "marginBottom": "4px"}),

                    label("Y-Axis"),
                    dcc.Dropdown(id="y-metric", options=Y_OPTIONS, value="risk",
                                 clearable=False, style={"fontSize": "0.82rem"}),

                    label("Point Size"),
                    dcc.Dropdown(id="size-metric", options=SIZE_OPTIONS, value="sample_size",
                                 clearable=False, style={"fontSize": "0.82rem"}),

                    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

                    label("Decision Thresholds"),
                    html.Div("Min efficacy (preferred)", style={"fontSize": "0.77rem", "color": TEXT_SEC, "marginBottom": "3px"}),
                    dcc.Slider(id="eff-thresh", min=0.2, max=0.9, step=0.05, value=0.50,
                               marks={0.2: ".2", 0.5: ".5", 0.9: ".9"},
                               tooltip={"placement": "bottom", "always_visible": True}),

                    html.Div("Max risk (preferred)", style={"fontSize": "0.77rem", "color": TEXT_SEC,
                                                             "marginBottom": "3px", "marginTop": "10px"}),
                    dcc.Slider(id="risk-thresh", min=0.1, max=0.7, step=0.05, value=0.30,
                               marks={0.1: ".1", 0.3: ".3", 0.7: ".7"},
                               tooltip={"placement": "bottom", "always_visible": True}),

                    html.Div("Min sample size (n)", style={"fontSize": "0.77rem", "color": TEXT_SEC,
                                                            "marginBottom": "3px", "marginTop": "10px"}),
                    dbc.Input(id="n-thresh", type="number", value=50, min=1, step=10,
                              size="sm", style={"marginBottom": "4px", "border": f"1px solid {BORDER}"}),

                    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),

                    label("Recommendation Filter"),
                    dcc.Checklist(
                        id="class-filter",
                        options=[
                            {"label": html.Span([html.Span("●", style={"color": COLOR_MAP["preferred"],   "marginRight": "6px"}), "Preferred"]),   "value": "preferred"},
                            {"label": html.Span([html.Span("●", style={"color": COLOR_MAP["conditional"], "marginRight": "6px"}), "Conditional"]), "value": "conditional"},
                            {"label": html.Span([html.Span("●", style={"color": COLOR_MAP["avoid"],       "marginRight": "6px"}), "Avoid"]),       "value": "avoid"},
                            {"label": html.Span([html.Span("●", style={"color": COLOR_MAP["uncertain"],   "marginRight": "6px"}), "Uncertain"]),   "value": "uncertain"},
                        ],
                        value=["preferred", "conditional", "avoid", "uncertain"],
                        className="mt-1",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "marginBottom": "7px",
                                    "fontSize": "0.82rem", "color": TEXT_PRI, "cursor": "pointer"},
                    ),

                ], style={
                    "padding": "18px 14px",
                    "height": "calc(100vh - 62px)",
                    "overflowY": "auto",
                    "backgroundColor": SURFACE,
                    "borderRight": f"1px solid {BORDER}",
                }),
            ], width=2),

            # ── Main content ─────────────────────────────────────────────────
            dbc.Col([
                dbc.Row(id="kpi-row", className="g-2 mb-3 mt-3 px-3"),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Loading(
                                dcc.Graph(
                                    id="br-scatter",
                                    config={"displayModeBar": True, "toImageButtonOptions": {
                                        "format": "png", "filename": "benefit_risk_plane", "scale": 2,
                                    }},
                                    style={"height": "490px"},
                                ),
                                type="circle", color=ACCENT,
                            ),
                        ], style={
                            "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
                            "borderRadius": "10px", "overflow": "hidden",
                        }),
                    ], width=8),

                    dbc.Col([
                        html.Div([
                            html.P("DECISION SURFACE", style={
                                "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                                "fontWeight": 700, "marginBottom": "10px",
                            }),
                            *[html.Div([
                                html.Span(icon, style={"fontSize": "1rem", "marginRight": "10px"}),
                                html.Div([
                                    html.Div(lbl, style={"fontSize": "0.78rem", "fontWeight": 600, "color": TEXT_PRI}),
                                    html.Div(desc, style={"fontSize": "0.7rem", "color": TEXT_SEC}),
                                ]),
                            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})
                              for icon, lbl, desc in [
                                ("🟢", "Preferred",   "High efficacy · Low risk"),
                                ("🟡", "Conditional", "Moderate tradeoff · use with caution"),
                                ("🔴", "Avoid",       "Low efficacy · High risk"),
                                ("⚪", "Uncertain",   "Insufficient evidence (low n)"),
                            ]],

                            html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),

                            html.P("DECISION RULE", style={
                                "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                                "fontWeight": 700, "marginBottom": "8px",
                            }),
                            *[html.Div(
                                html.Code(rule, style={
                                    "fontSize": "0.73rem", "whiteSpace": "pre", "display": "block",
                                    "color": color,
                                }),
                                style={
                                    "backgroundColor": bg, "border": f"1px solid {brd}",
                                    "borderRadius": "5px", "padding": "8px 10px", "marginBottom": "6px",
                                },
                            ) for rule, color, bg, brd in [
                                ("eff ≥ t  AND risk ≤ t → preferred",  "#16a34a", "#f0fdf4", "#bbf7d0"),
                                ("eff ≥ t  OR  risk ≤ t → conditional","#d97706", "#fffbeb", "#fde68a"),
                                ("eff < t  AND risk > t → avoid",       "#dc2626", "#fef2f2", "#fecaca"),
                                ("n < n_min             → uncertain",    "#64748b", "#f8fafc", "#e2e8f0"),
                            ]],

                            html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),

                            html.P("AXIS SEMANTICS", style={
                                "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                                "fontWeight": 700, "marginBottom": "10px",
                            }),
                            *[html.Div([
                                html.Code(axis, style={
                                    "fontSize": "0.74rem", "fontWeight": 700, "color": ACCENT,
                                    "backgroundColor": "#eff6ff", "padding": "1px 5px",
                                    "borderRadius": "3px", "marginRight": "6px",
                                }),
                                html.Span(defn, style={"fontSize": "0.72rem", "color": TEXT_SEC}),
                            ], style={"marginBottom": "6px"})
                              for axis, defn in [
                                ("X →", "Higher = more efficacious"),
                                ("Y ↑", "Higher = more risky"),
                                ("Size", "Evidence weight (sample n)"),
                                ("Color","Recommendation class"),
                            ]],
                        ], style={
                            "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
                            "borderRadius": "10px", "padding": "18px",
                            "height": "490px", "overflowY": "auto",
                        }),
                    ], width=4),
                ], className="px-3 mb-3"),

                # Table
                html.Div([
                    html.Div([
                        html.P("RANKED BENEFIT–RISK TABLE", style={
                            "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                            "fontWeight": 700, "margin": 0,
                        }),
                        html.P("Sorted by net benefit score · click headers · use filter row",
                               style={"fontSize": "0.72rem", "color": TEXT_TER, "margin": 0}),
                    ], style={"marginBottom": "12px"}),
                    html.Div(id="br-table-container"),
                ], style={
                    "margin": "0 12px 24px 12px",
                    "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
                    "borderRadius": "10px", "padding": "18px 20px",
                }),
            ], width=10),
        ], style={"margin": 0}),
    ],
)


# ─── Callbacks ───────────────────────────────────────────────────────────────

@callback(
    Output("data-store", "data"),
    Input("load-sample", "n_clicks"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=False,
)
def load_data(n_clicks, contents, filename):
    if contents:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            required = {"drug", "indication", "efficacy", "risk", "sample_size"}
            if required.issubset(df.columns):
                if "ci_width" not in df.columns:
                    df["ci_width"] = 1.96 * np.sqrt(
                        df["efficacy"] * (1 - df["efficacy"]) / df["sample_size"].clip(lower=1)
                    ) * 2
                return df.to_json(date_format="iso", orient="split")
        except Exception:
            pass
    return SAMPLE_DF.to_json(date_format="iso", orient="split")


@callback(
    Output("br-scatter", "figure"),
    Output("kpi-row", "children"),
    Output("br-table-container", "children"),
    Input("data-store", "data"),
    Input("x-metric", "value"),
    Input("y-metric", "value"),
    Input("size-metric", "value"),
    Input("eff-thresh", "value"),
    Input("risk-thresh", "value"),
    Input("n-thresh", "value"),
    Input("class-filter", "value"),
)
def update_dashboard(data_json, x_metric, y_metric, size_metric,
                     eff_thresh, risk_thresh, n_thresh, class_filter):
    eff_thresh  = float(eff_thresh  or 0.50)
    risk_thresh = float(risk_thresh or 0.30)
    n_thresh    = int(n_thresh or 50)

    df_raw = pd.read_json(io.StringIO(data_json), orient="split")

    if "inv_ci_width" not in df_raw.columns:
        df_raw["inv_ci_width"] = 1 / df_raw["ci_width"].clip(lower=0.01)

    df = classify_benefit_risk(df_raw, eff_thresh, risk_thresh, n_thresh)

    # Derived axes
    df["inv_risk"]     = 1 - df["risk"]
    df["inv_efficacy"] = 1 - df["efficacy"]

    df_filtered = df[df["decision_class"].isin(class_filter)]

    x_label_map = {"efficacy": "Efficacy (response rate)", "inv_risk": "Inverse Risk (1−AE rate)", "br_ratio": "Benefit-Risk Ratio"}
    y_label_map = {"risk": "Risk (AE rate)", "inv_efficacy": "Inverse Efficacy (1−response rate)"}
    x_label = x_label_map.get(x_metric, x_metric)
    y_label = y_label_map.get(y_metric, y_metric)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Quadrant shading (only for standard efficacy vs risk view)
    if x_metric == "efficacy" and y_metric == "risk":
        quads = [
            (eff_thresh, 1.01, 0, risk_thresh,   "rgba(22,163,74,0.06)"),   # preferred (top-right is low risk, high eff)
            (eff_thresh, 1.01, risk_thresh, 1.01,"rgba(217,119,6,0.06)"),   # conditional top
            (0, eff_thresh,    0, risk_thresh,    "rgba(217,119,6,0.06)"),   # conditional bottom
            (0, eff_thresh,    risk_thresh, 1.01, "rgba(220,38,38,0.06)"),   # avoid
        ]
        for x0, x1, y0, y1, col in quads:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                          fillcolor=col, line_width=0, layer="below")

    # Threshold lines
    if x_metric == "efficacy":
        fig.add_vline(x=eff_thresh, line_dash="dash", line_color="#16a34a",
                      line_width=1.2, opacity=0.5,
                      annotation_text=f"Eff = {eff_thresh:.2f}",
                      annotation_position="top left",
                      annotation_font=dict(color="#16a34a", size=10))
    if y_metric == "risk":
        fig.add_hline(y=risk_thresh, line_dash="dash", line_color="#dc2626",
                      line_width=1.2, opacity=0.5,
                      annotation_text=f"Risk = {risk_thresh:.2f}",
                      annotation_position="right",
                      annotation_font=dict(color="#dc2626", size=10))

    # Points per class (background → conditional → preferred on top)
    draw_order = ["uncertain", "avoid", "conditional", "preferred"]
    for cls in draw_order:
        sub = df_filtered[df_filtered["decision_class"] == cls]
        if sub.empty:
            continue

        x_vals    = sub[x_metric]
        y_vals    = sub[y_metric]
        size_vals = sub[size_metric]
        s_min, s_max = size_vals.min(), size_vals.max()
        sizes = (8 + 24 * (size_vals - s_min) / (s_max - s_min)) if s_max > s_min else pd.Series([14] * len(sub))

        label_map = {
            "preferred":   "Preferred",
            "conditional": "Conditional",
            "avoid":       "Avoid",
            "uncertain":   "Uncertain",
        }

        show_labels = sub["decision_class"].isin(["preferred", "avoid"])

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text" if cls in ["preferred", "avoid"] else "markers",
            name=label_map[cls],
            marker=dict(
                size=sizes,
                color=COLOR_MAP[cls],
                opacity=0.88 if cls in ["preferred", "avoid"] else (0.72 if cls == "conditional" else 0.45),
                line=dict(width=1.0, color="rgba(255,255,255,0.9)"),
            ),
            customdata=sub[["drug", "indication", "efficacy", "risk",
                            "sample_size", "br_ratio", "net_benefit", "ci_width"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<i>%{customdata[1]}</i><br><br>"
                "<b>Efficacy:</b> %{customdata[2]:.1%}<br>"
                "<b>Risk:</b> %{customdata[3]:.1%}<br>"
                "<b>Sample size:</b> %{customdata[4]:,}<br>"
                "<b>B/R ratio:</b> %{customdata[5]:.2f}<br>"
                "<b>Net benefit:</b> %{customdata[6]:.2f}"
                "<extra></extra>"
            ),
            text=sub["drug"] if cls in ["preferred", "avoid"] else [""] * len(sub),
            textposition="top center",
            textfont=dict(size=8, color=COLOR_MAP[cls]),
        ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor="#fafbfc",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_PRI),
        title=dict(
            text="Benefit–Risk Decision Plane",
            font=dict(size=13, color=TEXT_PRI), x=0.01,
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=11, color=TEXT_SEC)),
            gridcolor="#f1f5f9", linecolor=BORDER, showgrid=True,
            range=[0, 1.05] if x_metric in ["efficacy", "inv_risk"] else None,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=11, color=TEXT_SEC)),
            gridcolor="#f1f5f9", linecolor=BORDER, showgrid=True,
            range=[0, 1.05] if y_metric in ["risk", "inv_efficacy"] else None,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT_PRI)),
        hoverlabel=dict(bgcolor=SURFACE, font_size=12, bordercolor=BORDER,
                        font_family="Inter, monospace", font_color=TEXT_PRI),
        margin=dict(l=55, r=20, t=50, b=50),
    )

    # Axis direction annotation (preferred zone corner)
    if x_metric == "efficacy" and y_metric == "risk":
        fig.add_annotation(
            x=0.97, y=0.03, xref="paper", yref="paper",
            text="🟢 Preferred zone",
            showarrow=False,
            font=dict(size=10, color="#16a34a"),
            bgcolor="rgba(240,253,244,0.9)",
            bordercolor="#bbf7d0", borderwidth=1, borderpad=4,
        )
        fig.add_annotation(
            x=0.03, y=0.97, xref="paper", yref="paper",
            text="🔴 Avoid zone",
            showarrow=False,
            font=dict(size=10, color="#dc2626"),
            bgcolor="rgba(254,242,242,0.9)",
            bordercolor="#fecaca", borderwidth=1, borderpad=4,
        )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    n_pref  = (df["decision_class"] == "preferred").sum()
    n_cond  = (df["decision_class"] == "conditional").sum()
    n_avoid = (df["decision_class"] == "avoid").sum()
    n_unc   = (df["decision_class"] == "uncertain").sum()
    top_drug = df.nlargest(1, "net_benefit").iloc[0]["drug"] + " / " + \
               df.nlargest(1, "net_benefit").iloc[0]["indication"] if not df.empty else "—"
    avg_br = df["br_ratio"].median() if not df.empty else 0

    kpis = [
        _kpi("Preferred",   str(n_pref),        COLOR_MAP["preferred"],   "#16a34a"),
        _kpi("Conditional", str(n_cond),         COLOR_MAP["conditional"], "#d97706"),
        _kpi("Avoid",       str(n_avoid),        COLOR_MAP["avoid"],       "#dc2626"),
        _kpi("Uncertain",   str(n_unc),          COLOR_MAP["uncertain"],   "#94a3b8"),
        _kpi("Median B/R",  f"{avg_br:.2f}",     "#7c3aed",                "#7c3aed"),
        _kpi("Top profile", top_drug,             ACCENT,                   ACCENT, small=True),
    ]

    # ── Table ─────────────────────────────────────────────────────────────────
    tbl = df.nlargest(20, "net_benefit").copy()
    tbl["efficacy"] = tbl["efficacy"].map(lambda v: f"{v:.1%}")
    tbl["risk"]     = tbl["risk"].map(lambda v: f"{v:.1%}")
    tbl["br_ratio"] = tbl["br_ratio"].round(2)
    tbl["net_benefit"] = tbl["net_benefit"].round(2)
    tbl = tbl[["drug", "indication", "decision_class", "efficacy", "risk",
               "sample_size", "br_ratio", "net_benefit"]]
    tbl.columns = ["Drug", "Indication", "Class", "Efficacy", "Risk",
                   "n", "B/R Ratio", "Net Benefit"]

    cond_styles = [
        {"if": {"filter_query": '{Class} = "preferred"',  "column_id": "Class"}, "color": COLOR_MAP["preferred"],   "fontWeight": 700},
        {"if": {"filter_query": '{Class} = "conditional"', "column_id": "Class"}, "color": COLOR_MAP["conditional"], "fontWeight": 700},
        {"if": {"filter_query": '{Class} = "avoid"',      "column_id": "Class"}, "color": COLOR_MAP["avoid"],       "fontWeight": 700},
        {"if": {"filter_query": '{Class} = "uncertain"',  "column_id": "Class"}, "color": COLOR_MAP["uncertain"],   "fontWeight": 700},
        {"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"},
    ]

    table = dash_table.DataTable(
        data=tbl.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl.columns],
        sort_action="native",
        filter_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#f8fafc", "color": TEXT_SEC,
            "fontWeight": 700, "fontSize": "0.72rem",
            "letterSpacing": "0.6px", "border": f"1px solid {BORDER}",
            "textTransform": "uppercase",
        },
        style_cell={
            "backgroundColor": SURFACE, "color": TEXT_PRI,
            "fontSize": "0.8rem", "border": f"1px solid {BORDER}",
            "padding": "8px 14px", "fontFamily": "Inter, system-ui, sans-serif",
        },
        style_data_conditional=cond_styles,
    )

    return fig, kpis, table


def _kpi(label_text, value, value_color, accent_color, small=False):
    return dbc.Col(
        html.Div([
            html.Div(value, style={
                "fontSize": "0.88rem" if small else "1.45rem",
                "fontWeight": 700, "color": value_color,
                "lineHeight": 1.1, "marginBottom": "4px",
                "fontFamily": "Inter, monospace",
                "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
            }),
            html.Div(label_text, style={
                "fontSize": "0.67rem", "color": TEXT_SEC,
                "letterSpacing": "0.6px", "textTransform": "uppercase",
            }),
        ], style={
            "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
            "borderRadius": "8px", "padding": "14px 16px",
            "borderLeft": f"3px solid {accent_color}",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
        }),
        style={"padding": "0 6px"},
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
