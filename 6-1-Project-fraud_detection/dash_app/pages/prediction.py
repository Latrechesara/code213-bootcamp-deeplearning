import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import joblib

dash.register_page(__name__, path="/prediction")

# Load model
artifact = joblib.load("models/fraud_mlp_pipeline.joblib")
model = artifact["model"]
threshold = artifact["threshold"]

# Features
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "TRANSACTION INPUTS",
                                    className="text-white mb-4 fw-bold text-center",
                                ),

                                # Inputs (SHORTER)
                                *[
                                    dbc.Input(
                                        id=f"feat-{feat}",
                                        placeholder=feat,
                                        type="number",
                                        className="mb-2 bg-dark-glass",
                                        size="sm",  # 🔹 shorter input
                                    )
                                    for feat in feature_names
                                ],

                                # Button (SHORTER)
                                dbc.Button(
                                    "PREDICT FRAUD",
                                    id="predict-btn",
                                    className="w-100 py-2 mt-3 fw-bold border-0",
                                    style={
                                        "background": "linear-gradient(90deg, #7a0c0c 0%, #b11212 100%)",
                                        "borderRadius": "14px",
                                    },
                                ),

                                # Prediction Output
                                html.Div(
                                    id="predict-output",
                                    className="mt-4 p-3 result-display text-center",
                                ),
                            ],
                            className="div-user-controls p-4 my-5",  # 🔹 margin top & bottom
                            style={
                                "background": "rgba(30, 10, 40, 0.35)",
                                "backdropFilter": "blur(20px)",
                                "borderRadius": "22px",
                                "boxShadow": "0 20px 40px rgba(0,0,0,0.55)",
                            },
                        )
                    ],
                    lg=6,
                    md=8,
                    sm=12,
                )
            ],
            justify="center",
        )
    ],
    fluid=True,
)

@callback(
    Output("predict-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"feat-{feat}", "value") for feat in feature_names],
)
def predict_fraud(n, *values):
    if n is None:
        return ""

    try:
        df = pd.DataFrame([values], columns=feature_names)
        proba = model.predict_proba(df)[:, 1][0]

        prediction = (
            "🔴 Fraud Detected!" if proba >= threshold else "🟢 Legit Transaction"
        )

        return html.Div(
            [
                html.H5(
                    f"Fraud Probability: {proba:.2f}",
                    className="text-info",
                ),
                html.H3(
                    prediction,
                    className="text-white fw-bold",
                ),
            ]
        )

    except Exception as e:
        return html.Div(
            f"❌ Error: {str(e)}",
            className="text-danger",
        )
