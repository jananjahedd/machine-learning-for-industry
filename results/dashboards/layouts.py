from dash import dcc, html, callback, Output, Input

def create_layout(linear_metrics, current_data, predictions, assets_dir, health_recommendations):
    return html.Div(
        style={'backgroundColor': '#f9fafb', 'padding': '30px', 'fontFamily': 'Arial, sans-serif', 'position': 'relative'},
        children=[
            # Button to toggle Admin Dashboard
            html.Button(
                "Admin Dashboard",
                id="admin-button",
                n_clicks=0,
                style={
                    'position': 'absolute', 'top': '10px', 'right': '10px',
                    'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 
                    'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                },
            ),
            
            dcc.Tabs([
                dcc.Tab(
                    label='User Dashboard',
                    style={'backgroundColor': '#fff', 'border': 'none', 'color': '#007bff'},
                    selected_style={
                        'backgroundColor': '#007bff', 'color': '#fff', 'borderRadius': '10px 10px 0 0', 
                        'fontWeight': 'bold', 'padding': '10px 15px'
                    },
                    children=[
                        html.Div(
                            style={'padding': '30px', 'backgroundColor': '#fff', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.05)'},
                            children=[
                                # Health recommendation display
                                html.Div([
                                    html.H3("Health Recommendations", style={'color': '#333', 'marginBottom': '15px'}),
                                    html.Ul(
                                        [html.Li(rec, style={'fontSize': '16px', 'color': '#555', 'lineHeight': '1.6'}) for rec in health_recommendations],
                                        style={'listStyleType': 'disc', 'paddingLeft': '25px'}
                                    )
                                ], style={'marginBottom': '30px'}),

                                # AQI Table
                                html.Div([
                                    html.H3("Air Quality Index (AQI) Table", style={'color': '#333', 'marginBottom': '15px'}),
                                    html.Table([
                                        html.Thead(
                                            html.Tr([
                                                html.Th("Name", style={'padding': '10px', 'backgroundColor': '#f9fafb', 'color': '#333', 'fontWeight': 'bold'}),
                                                html.Th("Index Value", style={'padding': '10px', 'backgroundColor': '#f9fafb', 'color': '#333', 'fontWeight': 'bold'}),
                                                html.Th("Advisory", style={'padding': '10px', 'backgroundColor': '#f9fafb', 'color': '#333', 'fontWeight': 'bold'})
                                            ])
                                        ),
                                        html.Tbody([
                                            html.Tr([
                                                html.Td("Good", style={'padding': '10px', 'backgroundColor': '#00e400', 'color': 'white'}),
                                                html.Td("0 to 50", style={'padding': '10px'}),
                                                html.Td("None", style={'padding': '10px'})
                                            ]),
                                            html.Tr([
                                                html.Td("Moderate", style={'padding': '10px', 'backgroundColor': '#ffff00', 'color': 'black'}),
                                                html.Td("51 to 100", style={'padding': '10px'}),
                                                html.Td("Usually sensitive individuals should consider limiting prolonged outdoor exertion.", style={'padding': '10px'})
                                            ]),
                                            html.Tr([
                                                html.Td("Unhealthy for Sensitive Groups", style={'padding': '10px', 'backgroundColor': '#ff7e00', 'color': 'black'}),
                                                html.Td("101 to 150", style={'padding': '10px'}),
                                                html.Td("Children, active adults, and people with respiratory disease, such as asthma, should limit prolonged outdoor exertion.", style={'padding': '10px'})
                                            ]),
                                            html.Tr([
                                                html.Td("Unhealthy", style={'padding': '10px', 'backgroundColor': '#ff0000', 'color': 'white'}),
                                                html.Td("151 to 200", style={'padding': '10px'}),
                                                html.Td("Children, active adults, and people with respiratory disease, such as asthma, should avoid outdoor exertion; everyone else should limit prolonged outdoor exertion.", style={'padding': '10px'})
                                            ]),
                                            html.Tr([
                                                html.Td("Very Unhealthy", style={'padding': '10px', 'backgroundColor': '#8f3f97', 'color': 'white'}),
                                                html.Td("201 to 300", style={'padding': '10px'}),
                                                html.Td("Children, active adults, and people with respiratory disease, such as asthma, should avoid outdoor exertion; everyone else should limit outdoor exertion.", style={'padding': '10px'})
                                            ]),
                                            html.Tr([
                                                html.Td("Hazardous", style={'padding': '10px', 'backgroundColor': '#7e0023', 'color': 'white'}),
                                                html.Td("301 to 500", style={'padding': '10px'}),
                                                html.Td("Everyone should avoid all physical activity outdoors.", style={'padding': '10px'})
                                            ])
                                        ])
                                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd', 'marginTop': '10px'})
                                ], style={'marginBottom': '30px'}),
                                
                                # Current pollutant concentrations display
                                html.Div([
                                    html.H3("Current Pollutant Concentrations", style={'color': '#333', 'marginBottom': '15px'}),
                                    html.Ul([
                                        html.Li(f"NO2: {round(current_data['Avg_NO2'].item(), 3)} µg/m³", style={'fontSize': '16px', 'color': '#555'}),
                                        html.Li(f"O3: {round(current_data['Avg_O3'].item(), 3)} µg/m³", style={'fontSize': '16px', 'color': '#555'}),
                                    ], style={'listStyleType': 'disc', 'paddingLeft': '25px'})
                                ], style={'marginBottom': '30px'}),

                                # Predictions display
                                html.Div([
                                    html.H3("Predicted Pollutant Concentrations for the Next 3 Days", style={'color': '#333', 'marginBottom': '15px'}),
                                    html.Table([
                                        html.Thead(html.Tr([html.Th("NO2", style={'padding': '10px', 'textAlign': 'center'}), html.Th("O3", style={'padding': '10px', 'textAlign': 'center'})],
                                            style={'color': '#fff', 'backgroundColor': '#007bff', 'padding': '12px'})),
                                        html.Tbody([
                                            html.Tr([
                                                html.Td(round(pred[0], 3), style={'padding': '12px', 'color': '#333', 'textAlign': 'center'}),
                                                html.Td(round(pred[1], 3), style={'padding': '12px', 'color': '#333', 'textAlign': 'center'})
                                            ])
                                            for pred in predictions
                                        ])
                                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd', 'marginTop': '10px'})
                                ])
                            ]
                        )
                    ]
                )
            ]),

            # Admin Dashboard
            html.Div(
                id="admin-dashboard",
                style={'display': 'block', 'padding': '20px', 'marginTop': '30px', 'backgroundColor': '#fff', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'},
                children=[
                    html.H1("Admin Dashboard", style={'textAlign': 'center', 'color': '#28a745', 'marginBottom': '30px'}),
                    html.Div(
                        style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'flex-start'},
                        children=[
                            html.Div(
                                style={
                                    'width': '45%', 'border': '1px solid #e0e4e8', 'padding': '25px',
                                    'backgroundColor': '#f9fafb', 'borderRadius': '10px', 'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.08)'
                                },
                                children=[
                                    html.H2("Linear Regression Results", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                                    dcc.Markdown(f"""
                                        - **Test MSE:** {linear_metrics['test_mse']}
                                        - **Test R² Score:** {linear_metrics['test_r2']}
                                        - **Test RMSE:** {linear_metrics['test_rmse']}
                                    """, style={'fontSize': '16px', 'color': '#555', 'lineHeight': '1.6'}),
                                    html.Img(
                                        src='/assets/linear_regression_plot.png',
                                        style={"width": "100%", "borderRadius": "10px", 'marginTop': '20px', 'objectFit': 'cover'}
                                    )
                                ]
                            ),
                            html.Div([
                                html.H3("Predicted Pollutant Concentrations for the Next 3 Days", style={'color': '#333', 'marginBottom': '15px'}),
                                html.Table([
                                    html.Thead(html.Tr([
                                        html.Th("NO2", style={'padding': '10px', 'textAlign': 'center'}), 
                                        html.Th("O3", style={'padding': '10px', 'textAlign': 'center'})
                                    ], style={'color': '#fff', 'backgroundColor': '#28a745', 'padding': '12px'})),
                                    html.Tbody([
                                        html.Tr([
                                            html.Td(pred[0], style={'padding': '12px', 'color': '#333', 'textAlign': 'center'}),
                                            html.Td(pred[1], style={'padding': '12px', 'color': '#333', 'textAlign': 'center'})
                                        ])
                                        for pred in predictions
                                    ])
                                ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd', 'marginTop': '10px'})
                            ], style={'width': '45%'})

                        ]
                    )
                ]
            )
        ]
    )

# Callback to toggle admin dashboard visibility
@callback(
    Output('admin-dashboard', 'style'),
    Input('admin-button', 'n_clicks')
)
def toggle_admin_dashboard(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    return {'display': 'none'}
