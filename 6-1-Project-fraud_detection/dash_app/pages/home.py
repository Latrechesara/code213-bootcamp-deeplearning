import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')

layout = html.Div([
    dbc.Container([

        # SECTION 1: HERO
        dbc.Row([
            dbc.Col([
                html.H1(
                    "THE CREDIT CARD FRAUD PORTAL",
                    className="text-white fw-bold display-3 mt-5",
                    style={"letterSpacing": "5px"}
                ),
                html.P(
                    "Uncovering hidden patterns in transactions to protect customers and banks from fraud.",
                    className="text-light-gray fs-5 mb-5"
                ),
            ], width=12)
        ]),

        # SECTION 2: THE PROBLEM (Introduction)
        dbc.Row([
            dbc.Col([
                html.Img(
                    src="/assets/unnamed__1_-removebg-preview.png",
                    className="intro-img mb-4"
                ),
            ], lg=6),
            dbc.Col([
                html.H3(
                    "The Fraud Challenge",
                    className="text-light-purple fw-bold mb-3"
                ),
                html.P([
                    "Credit card fraud can cost customers and financial institutions millions. Every transaction carries a risk, but ",
                    html.B("less than 0.2% of transactions are actually fraudulent", className="text-light-purple"),
                    ". Our AI model analyzes hundreds of thousands of transactions, detecting subtle anomalies in patterns that human eyes might miss. ",
                    html.B("Early detection prevents losses, protects customers, and strengthens trust in banking.", className="text-light-purple"),
                ], className="text-light-gray fs-5"),
                
                html.Div([
                    html.Span("492 Frauds in 284,807 Transactions", className="data-badge"),
                    html.Span("PCA Feature Analysis (V1-V28)", className="data-badge"),
                    html.Span("Real-time Detection Potential", className="data-badge"),
                ], className="mt-4"),
                
                dbc.Button(
                    "START DETECTING FRAUD",
                    href="/prediction",
                    className="mt-5 py-3 px-5 fw-bold border-0",
                    style={
                        "background": "linear-gradient(90deg, #9b59b6 0%, #6c3483 100%)",
                        "borderRadius": "30px",
                        "color": "#e0e0e0"
                    }
                )
            ], lg=6, className="ps-lg-5 d-flex flex-column justify-content-center")
        ], className="mb-5"),

        # SECTION 3: THE DATASET EXPLAINED
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(
                        "Understanding the Dataset",
                        className="text-light-purple mb-4"
                    ),
                    html.P(
                        "This dataset contains anonymized credit card transactions over two days in September 2013. "
                        "Features V1-V28 are the result of PCA transformation, while 'Time' and 'Amount' remain interpretable. "
                        "'Class' indicates fraud (1) or normal (0).",
                        className="text-light-gray"
                    ),

                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Feature", className="text-light-purple"),
                            html.Th("Significance", className="text-light-purple")
                        ])),

                        html.Tbody([
                            html.Tr([
                                html.Td("Time", className="text-purple"),
                                html.Td("Seconds elapsed from the first transaction. Can reveal suspicious patterns over time.", className="text-light-gray")
                            ]),
                            html.Tr([
                                html.Td("Amount", className="text-purple"),
                                html.Td("The monetary value of the transaction. High amounts may indicate higher fraud risk.", className="text-light-gray")
                            ]),
                            html.Tr([
                                html.Td("V1-V28", className="text-purple"),
                                html.Td("Principal components capturing hidden patterns in transaction behavior.", className="text-light-gray")
                            ]),
                            html.Tr([
                                html.Td("Class", className="text-purple"),
                                html.Td("Target variable: 1 = Fraud, 0 = Legitimate transaction.", className="text-light-gray")
                            ]),
                        ])
                    ], className="modern-table mt-3", bordered=False, hover=False)

                ], className="p-5 result-display")
            ], width=12)
        ], className="mb-5 py-5")

    ], className="welcome-container", fluid=True)
])
