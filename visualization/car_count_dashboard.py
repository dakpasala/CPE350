#!/usr/bin/env python3
"""
car_count_dashboard.py

Web dashboard for vehicle exit direction analysis with query parameter support.
Displays interactive charts using Plotly and Dash.

URL Examples:
- http://localhost:8053/?location=patterson&time_range=hour
- http://localhost:8053/?time_range=day&only_type=car&min_speed=5
- http://localhost:8053/?location=all&time_range=week&interval=1H
"""

import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from datetime import datetime
from urllib.parse import parse_qs
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Optional, Dict, Any


# =========================
# Config
# =========================

BACKEND_API_URL = "http://127.0.0.1:8000"
DEFAULT_PORT = 8053


# =========================
# Dash App
# =========================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = "Traffic Exit Direction Analysis"


# =========================
# Helper Functions
# =========================

def heading_to_cardinal(h: float) -> str:
    """N=[315,360)∪[0,45), E=[45,135), S=[135,225), W=[225,315)"""
    h = float(h) % 360.0
    if h >= 315.0 or h < 45.0:
        return "N"
    if h < 135.0:
        return "E"
    if h < 225.0:
        return "S"
    return "W"


def fetch_combined_stats(
    time_range: str,
    limit: int,
    location: Optional[str] = None,
    timeout_s: int = 30
) -> Dict[str, Any]:
    """Fetch data from FastAPI endpoint."""
    try:
        url = f"{BACKEND_API_URL}/stats/combined"
        params: Dict[str, Any] = {"time_range": time_range, "limit": int(limit)}
        if location and location != 'all':
            params["location"] = location

        print(f"[API] GET {url}")
        print(f"[API] Params: {params}")
        
        resp = requests.get(url, params=params, timeout=timeout_s)
        resp.raise_for_status()
        
        data = resp.json()
        print(f"[SUCCESS] Fetched {data.get('count', 0)} records")
        return data
    
    except Exception as e:
        print(f"[ERROR] API request failed: {e}")
        return {"count": 0, "data": []}


def process_exit_data(
    df: pd.DataFrame,
    only_type: str = "car",
    min_speed: float = 0.0,
    confident_only: bool = False,
    interval: Optional[str] = None
):
    """Process dataframe to extract exit direction counts."""
    
    if df.empty:
        return None, None
    
    # Validate required columns
    required = ["timestamp", "object_id", "heading_deg"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return None, None
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]
    
    if df.empty:
        return None, None
    
    # Type filter
    if only_type.lower() != "any" and "detected_type" in df.columns:
        df["detected_type"] = df["detected_type"].astype("string")
        df = df[df["detected_type"].str.lower() == only_type.lower()]
    
    # Confident filter
    if confident_only and "is_confident" in df.columns:
        df = df[df["is_confident"] == True]
    
    # Speed filter
    if "speed_mps" in df.columns:
        df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce").fillna(0.0)
        df = df[df["speed_mps"] >= float(min_speed)]
    
    if df.empty:
        return None, None
    
    # Get exit record (latest timestamp per object)
    df = df.sort_values("timestamp")
    df_exit = df.groupby("object_id", as_index=False).tail(1).copy()
    
    # Process headings
    df_exit["heading_deg"] = pd.to_numeric(df_exit["heading_deg"], errors="coerce")
    df_exit = df_exit[df_exit["heading_deg"].notna()]
    
    if df_exit.empty:
        return None, None
    
    df_exit["direction"] = df_exit["heading_deg"].apply(heading_to_cardinal)
    
    # Total counts by direction
    total = (
        df_exit.groupby("direction", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("direction")
    )
    
    # Timeseries (if interval specified)
    timeseries = None
    if interval:
        df_exit["time_bin"] = df_exit["timestamp"].dt.floor(interval)
        timeseries = (
            df_exit.groupby(["time_bin", "direction"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("time_bin")
        )
    
    return total, timeseries


# =========================
# Layout
# =========================

app.layout = dbc.Container([
    # URL location component
    dcc.Location(id='url', refresh=False),
    
    # Hidden stores
    dcc.Store(id='config-store'),
    dcc.Store(id='data-store'),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("EXIT DIRECTION ANALYSIS", className="display-4 mb-0"),
                html.P(id="subtitle", children="Vehicle count by exit direction", 
                       className="lead text-muted")
            ], className="text-center py-4")
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FILTERS", className="font-weight-bold"),
                dbc.CardBody([
                    dbc.Row([
                        # Location
                        dbc.Col([
                            html.Label("Location:", className="text-muted small"),
                            dcc.Dropdown(
                                id='location-dropdown',
                                options=[
                                    {'label': 'All Cameras', 'value': 'all'},
                                    {'label': 'Patterson', 'value': 'patterson'},
                                    {'label': 'Downtown', 'value': 'downtown'},
                                    {'label': 'Highway 1', 'value': 'highway1'},
                                ],
                                value='all',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=4),
                        
                        # Time Range
                        dbc.Col([
                            html.Label("Time Range:", className="text-muted small"),
                            dcc.Dropdown(
                                id='time-dropdown',
                                options=[
                                    {'label': 'Last 1 hour', 'value': 'hour'},
                                    {'label': 'Last 6 hours', 'value': '6hours'},
                                    {'label': 'Last 12 hours', 'value': '12hours'},
                                    {'label': 'Last 1 day', 'value': 'day'},
                                    {'label': 'Last 1 week', 'value': 'week'},
                                ],
                                value='hour',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=4),
                        
                        # Vehicle Type
                        dbc.Col([
                            html.Label("Vehicle Type:", className="text-muted small"),
                            dcc.Dropdown(
                                id='type-dropdown',
                                options=[
                                    {'label': 'All Types', 'value': 'any'},
                                    {'label': 'Car', 'value': 'car'},
                                    {'label': 'Truck', 'value': 'truck'},
                                    {'label': 'Bus', 'value': 'bus'},
                                ],
                                value='car',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=4),
                    ]),
                    
                    dbc.Row([
                        # Timeseries Interval
                        dbc.Col([
                            html.Label("Timeseries Interval (optional):", className="text-muted small"),
                            dcc.Dropdown(
                                id='interval-dropdown',
                                options=[
                                    {'label': 'None', 'value': ''},
                                    {'label': '1 minute', 'value': '1min'},
                                    {'label': '5 minutes', 'value': '5min'},
                                    {'label': '15 minutes', 'value': '15min'},
                                    {'label': '1 hour', 'value': '1H'},
                                ],
                                value='',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=12),
                    ]),
                    
                    # Apply button
                    dbc.Button(
                        "Apply Filters & Generate Charts",
                        id="apply-btn",
                        color="primary",
                        className="w-100"
                    )
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Stats Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="total-vehicles", className="mb-0 text-primary"),
                    html.P("Total Vehicles", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="north-count", className="mb-0 text-info"),
                    html.P("North ↑", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="east-count", className="mb-0 text-success"),
                    html.P("East →", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="south-count", className="mb-0 text-warning"),
                    html.P("South ↓", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="west-count", className="mb-0 text-danger"),
                    html.P("West ←", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=2),
    ], className="mb-4"),
    
    # Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Total Exit Direction Counts", className="font-weight-bold"),
                dbc.CardBody([
                    dcc.Graph(id='total-chart')
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    # Timeseries Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Exit Direction Over Time", className="font-weight-bold"),
                dbc.CardBody([
                    dcc.Graph(id='timeseries-chart')
                ])
            ])
        ])
    ], className="mb-4")
    
], fluid=True, className="p-4")


# =========================
# Callbacks
# =========================

@app.callback(
    [
        Output('config-store', 'data'),
        Output('url', 'search')  # Add this to update URL
    ],
    [
        Input('url', 'search'),
        Input('apply-btn', 'n_clicks')
    ],
    [
        State('location-dropdown', 'value'),
        State('time-dropdown', 'value'),
        State('type-dropdown', 'value'),
        State('interval-dropdown', 'value'),
    ]
)
def update_config(url_search, n_clicks, location, time_range, vehicle_type, interval):
    """Update configuration from URL params or UI controls."""
    
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    # Parse URL on initial load
    if triggered_id == 'url' and url_search:
        params = parse_qs(url_search.lstrip('?'))
        location = params.get('location', [location])[0]
        time_range = params.get('time_range', [time_range])[0]
        vehicle_type = params.get('only_type', [vehicle_type])[0]
        interval = params.get('interval', [interval])[0]
    
    config = {
        'location': location,
        'time_range': time_range,
        'only_type': vehicle_type,
        'interval': interval if interval else None,
        'updated_at': datetime.utcnow().isoformat()
    }
    
    print(f"[CONFIG] {config}")
    
    # Build new URL query string when apply button is clicked
    new_url = ""
    if triggered_id == 'apply-btn':
        params = []
        if location != 'all':
            params.append(f"location={location}")
        params.append(f"time_range={time_range}")
        if vehicle_type != 'any':
            params.append(f"only_type={vehicle_type}")
        if interval:
            params.append(f"interval={interval}")
        
        new_url = f"?{'&'.join(params)}" if params else ""
    
    return config, new_url


@app.callback(
    Output('subtitle', 'children'),
    Input('config-store', 'data')
)
def update_subtitle(config):
    """Update subtitle based on config."""
    if not config:
        return "Vehicle count by exit direction"
    
    location = config.get('location', 'all')
    time_range = config.get('time_range', 'hour')
    vehicle_type = config.get('only_type', 'any')
    
    location_text = location.upper() if location != 'all' else 'ALL CAMERAS'
    type_text = vehicle_type.upper() if vehicle_type != 'any' else 'ALL TYPES'
    
    return f"{location_text} • {time_range.upper()} • {type_text}"


@app.callback(
    Output('data-store', 'data'),
    Input('config-store', 'data')
)
def fetch_and_process_data(config):
    """Fetch data from API and process it."""
    
    if not config:
        return None
    
    # Fetch data
    location = config.get('location')
    time_range = config.get('time_range', 'hour')
    
    # Dynamic limit based on time range
    limit_map = {
        'hour': 10000,
        '6hours': 50000,
        '12hours': 100000,
        'day': 200000,
        'week': 500000,
    }
    limit = limit_map.get(time_range, 10000)
    
    data = fetch_combined_stats(time_range, limit, location)
    
    if not data.get('data'):
        return {'total': None, 'timeseries': None}
    
    # Convert to DataFrame
    df = pd.DataFrame(data['data'])
    
    # Process data
    total, timeseries = process_exit_data(
        df,
        only_type=config.get('only_type', 'car'),
        min_speed=0.0,  # Fixed at 0
        confident_only=False,  # Fixed at False
        interval=config.get('interval')
    )
    
    # Convert to dict for storage
    result = {
        'total': total.to_dict('records') if total is not None else None,
        'timeseries': timeseries.to_dict('records') if timeseries is not None else None
    }
    
    return result


@app.callback(
    [
        Output('total-vehicles', 'children'),
        Output('north-count', 'children'),
        Output('east-count', 'children'),
        Output('south-count', 'children'),
        Output('west-count', 'children'),
    ],
    Input('data-store', 'data')
)
def update_stats(data):
    """Update stat cards."""
    
    if not data or not data.get('total'):
        return "0", "0", "0", "0", "0"
    
    total_df = pd.DataFrame(data['total'])
    
    # Get counts by direction
    counts = {row['direction']: row['count'] for _, row in total_df.iterrows()}
    
    total = sum(counts.values())
    north = counts.get('N', 0)
    east = counts.get('E', 0)
    south = counts.get('S', 0)
    west = counts.get('W', 0)
    
    return str(total), str(north), str(east), str(south), str(west)


@app.callback(
    Output('total-chart', 'figure'),
    Input('data-store', 'data')
)
def update_total_chart(data):
    """Update bar chart."""
    
    if not data or not data.get('total'):
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(data['total'])
    
    fig = px.bar(
        df,
        x='direction',
        y='count',
        title='Vehicle Counts by Exit Direction',
        labels={'direction': 'Exit Direction', 'count': 'Vehicle Count'},
        color='direction',
        color_discrete_map={'N': '#17a2b8', 'E': '#28a745', 'S': '#ffc107', 'W': '#dc3545'}
    )
    
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        xaxis_title="Direction",
        yaxis_title="Count"
    )
    
    return fig


@app.callback(
    Output('timeseries-chart', 'figure'),
    Input('data-store', 'data')
)
def update_timeseries_chart(data):
    """Update timeseries chart."""
    
    if not data or not data.get('timeseries'):
        return go.Figure().add_annotation(
            text="No timeseries data (select an interval in filters)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(data['timeseries'])
    df['time_bin'] = pd.to_datetime(df['time_bin'])
    
    fig = px.line(
        df,
        x='time_bin',
        y='count',
        color='direction',
        title='Exit Direction Over Time',
        labels={'time_bin': 'Time', 'count': 'Vehicle Count'},
        color_discrete_map={'N': '#17a2b8', 'E': '#28a745', 'S': '#ffc107', 'W': '#dc3545'}
    )
    
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title="Time",
        yaxis_title="Count"
    )
    
    return fig


# =========================
# Main
# =========================

def main():
    """Start the dashboard."""
    print(f"Starting Exit Direction Dashboard on http://0.0.0.0:{DEFAULT_PORT}")
    print(f"Backend API: {BACKEND_API_URL}")
    
    app.run(
        host='0.0.0.0',
        port=DEFAULT_PORT,
        debug=False
    )


if __name__ == "__main__":
    main()
