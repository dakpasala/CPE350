#!/usr/bin/env python3
"""
incidents_dashboard.py

Web dashboard for incident analysis with query parameter support.
Displays incident counts, severity, and timeseries using Plotly and Dash.

URL Examples:
- http://localhost:8055/?location=patterson&minutes=60
- http://localhost:8055/?minutes=1440&incident_type=collision
- http://localhost:8055/?location=all&minutes=10080&interval=1H
"""

import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
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
DEFAULT_PORT = 8055


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

app.title = "Incident Analysis Dashboard"


# =========================
# Helper Functions
# =========================

def time_range_to_minutes(time_range: str) -> int:
    """Convert time range string to minutes."""
    mapping = {
        '15min': 15,
        '30min': 30,
        '1hour': 60,
        '6hours': 360,
        '12hours': 720,
        '1day': 1440,
        '3days': 4320,
        '1week': 10080,
        '2weeks': 20160,
        '1month': 43200
    }
    return mapping.get(time_range, 60)


def minutes_to_label(minutes: int) -> str:
    """Convert minutes to human-readable label."""
    if minutes < 60:
        return f"Last {minutes} min"
    elif minutes < 1440:
        hours = minutes // 60
        return f"Last {hours} hour{'s' if hours > 1 else ''}"
    else:
        days = minutes // 1440
        return f"Last {days} day{'s' if days > 1 else ''}"


def fetch_incidents(
    minutes: int,
    location: Optional[str] = None,
    timeout_s: int = 30
) -> Dict[str, Any]:
    """Fetch incidents from FastAPI endpoint."""
    try:
        url = f"{BACKEND_API_URL}/incidents/timerange"
        params: Dict[str, Any] = {"minutes": int(minutes)}
        if location and location != 'all':
            params["location"] = location

        print(f"[API] GET {url}")
        print(f"[API] Params: {params}")
        
        resp = requests.get(url, params=params, timeout=timeout_s)
        resp.raise_for_status()
        
        data = resp.json()
        print(f"[SUCCESS] Fetched {data.get('count', 0)} incidents")
        return data
    
    except Exception as e:
        print(f"[ERROR] API request failed: {e}")
        return {"count": 0, "incidents": []}


def process_incident_data(
    incidents: list,
    incident_type: Optional[str] = None,
    min_severity: float = 0.0,
    interval: Optional[str] = None
):
    """Process incidents data for visualization."""
    
    if not incidents:
        return None, None, None
    
    df = pd.DataFrame(incidents)
    
    if df.empty:
        return None, None, None
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]
    
    if df.empty:
        return None, None, None
    
    # Filter by incident type
    if incident_type and incident_type != 'all':
        df = df[df["incident_type"] == incident_type]
    
    # Filter by severity
    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0)
        df = df[df["severity"] >= min_severity]
    
    if df.empty:
        return None, None, None
    
    # Total counts by type
    type_counts = (
        df.groupby("incident_type", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("incident_type")
    )
    
    # Severity stats by type
    severity_stats = None
    if "severity" in df.columns:
        severity_stats = (
            df.groupby("incident_type", as_index=False)
            .agg(
                avg_severity=("severity", "mean"),
                max_severity=("severity", "max"),
                count=("severity", "size")
            )
            .sort_values("incident_type")
        )
    
    # Timeseries (if interval specified)
    timeseries = None
    if interval:
        df["time_bin"] = df["timestamp"].dt.floor(interval)
        timeseries = (
            df.groupby(["time_bin", "incident_type"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("time_bin")
        )
    
    return type_counts, severity_stats, timeseries


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
                html.H1("INCIDENT ANALYSIS", className="display-4 mb-0"),
                html.P(id="subtitle", children="Traffic incident monitoring and analysis", 
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
                                    {'label': 'Last 15 minutes', 'value': '15min'},
                                    {'label': 'Last 30 minutes', 'value': '30min'},
                                    {'label': 'Last 1 hour', 'value': '1hour'},
                                    {'label': 'Last 6 hours', 'value': '6hours'},
                                    {'label': 'Last 12 hours', 'value': '12hours'},
                                    {'label': 'Last 1 day', 'value': '1day'},
                                    {'label': 'Last 3 days', 'value': '3days'},
                                    {'label': 'Last 1 week', 'value': '1week'},
                                    {'label': 'Last 2 weeks', 'value': '2weeks'},
                                    {'label': 'Last 1 month', 'value': '1month'},
                                ],
                                value='1hour',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=4),
                        
                        # Incident Type
                        dbc.Col([
                            html.Label("Incident Type:", className="text-muted small"),
                            dcc.Dropdown(
                                id='type-dropdown',
                                options=[
                                    {'label': 'All Types', 'value': 'all'},
                                    {'label': 'Collision', 'value': 'collision'},
                                    {'label': 'Near Miss', 'value': 'near_miss'},
                                ],
                                value='all',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=4),
                    ]),
                    
                    dbc.Row([
                        # Timeseries Interval
                        dbc.Col([
                            html.Label("Timeseries Interval:", className="text-muted small"),
                            dcc.Dropdown(
                                id='interval-dropdown',
                                options=[
                                    {'label': '1 minute', 'value': '1min'},
                                    {'label': '5 minutes', 'value': '5min'},
                                    {'label': '15 minutes', 'value': '15min'},
                                    {'label': '1 hour', 'value': '1H'},
                                    {'label': '1 day', 'value': '1D'},
                                ],
                                value='15min',  # Default selected
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
                    html.H4("0", id="total-incidents", className="mb-0 text-danger"),
                    html.P("Total Incidents", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="collision-count", className="mb-0 text-danger"),
                    html.P("Collisions", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="near-miss-count", className="mb-0 text-warning"),
                    html.P("Near Misses", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0.0", id="avg-severity", className="mb-0 text-info"),
                    html.P("Avg Severity", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
    ], className="mb-4"),
    
    # Charts Row 1 - REMOVED
    
    # Timeseries Chart - NOW SCATTER PLOT
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Incidents Over Time", className="font-weight-bold"),
                dbc.CardBody([
                    dcc.Graph(id='timeseries-chart', style={'height': '600px'})
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
        Output('url', 'search')
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
def update_config(url_search, n_clicks, location, time_range, incident_type, interval):
    """Update configuration from URL params or UI controls."""
    
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    # Parse URL on initial load
    if triggered_id == 'url' and url_search:
        params = parse_qs(url_search.lstrip('?'))
        location = params.get('location', [location])[0]
        
        # Handle both 'minutes' and 'time_range' params
        if 'minutes' in params:
            minutes = int(params['minutes'][0])
            # Find closest time_range
            for tr, mins in [('15min', 15), ('30min', 30), ('1hour', 60), 
                            ('6hours', 360), ('12hours', 720), ('1day', 1440),
                            ('3days', 4320), ('1week', 10080), ('2weeks', 20160), ('1month', 43200)]:
                if mins == minutes:
                    time_range = tr
                    break
        else:
            time_range = params.get('time_range', [time_range])[0]
        
        incident_type = params.get('incident_type', [incident_type])[0]
        interval = params.get('interval', [interval])[0]
    
    config = {
        'location': location,
        'time_range': time_range,
        'minutes': time_range_to_minutes(time_range),
        'incident_type': incident_type,
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
        params.append(f"minutes={config['minutes']}")
        if incident_type != 'all':
            params.append(f"incident_type={incident_type}")
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
        return "Traffic incident monitoring and analysis"
    
    location = config.get('location', 'all')
    time_range = config.get('time_range', '1hour')
    incident_type = config.get('incident_type', 'all')
    
    location_text = location.upper() if location != 'all' else 'ALL CAMERAS'
    type_text = incident_type.upper().replace('_', ' ') if incident_type != 'all' else 'ALL TYPES'
    time_text = minutes_to_label(config.get('minutes', 60))
    
    return f"{location_text} • {time_text} • {type_text}"


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
    minutes = config.get('minutes', 60)
    
    data = fetch_incidents(minutes, location)
    
    if not data.get('incidents'):
        return {'type_counts': None, 'severity_stats': None, 'timeseries': None}
    
    # Process data
    type_counts, severity_stats, timeseries = process_incident_data(
        data['incidents'],
        incident_type=config.get('incident_type', 'all'),
        min_severity=0.0,
        interval=config.get('interval')
    )
    
    # Convert to dict for storage
    result = {
        'type_counts': type_counts.to_dict('records') if type_counts is not None else None,
        'severity_stats': severity_stats.to_dict('records') if severity_stats is not None else None,
        'timeseries': timeseries.to_dict('records') if timeseries is not None else None
    }
    
    return result


@app.callback(
    [
        Output('total-incidents', 'children'),
        Output('collision-count', 'children'),
        Output('near-miss-count', 'children'),
        Output('avg-severity', 'children'),
    ],
    Input('data-store', 'data')
)
def update_stats(data):
    """Update stat cards."""
    
    if not data or not data.get('type_counts'):
        return "0", "0", "0", "0.0"
    
    counts_df = pd.DataFrame(data['type_counts'])
    
    # Get counts by type
    counts = {row['incident_type']: row['count'] for _, row in counts_df.iterrows()}
    
    total = sum(counts.values())
    collisions = counts.get('collision', 0)
    near_misses = counts.get('near_miss', 0)
    
    # Calculate average severity
    avg_severity = 0.0
    if data.get('severity_stats'):
        severity_df = pd.DataFrame(data['severity_stats'])
        total_weighted = sum(row['avg_severity'] * row['count'] for _, row in severity_df.iterrows())
        total_count = sum(row['count'] for _, row in severity_df.iterrows())
        if total_count > 0:
            avg_severity = total_weighted / total_count
    
    return str(total), str(collisions), str(near_misses), f"{avg_severity:.2f}"


@app.callback(
    Output('timeseries-chart', 'figure'),
    Input('data-store', 'data')
)
def update_timeseries_chart(data):
    """Update timeseries scatter plot."""
    
    if not data or not data.get('timeseries'):
        return go.Figure().add_annotation(
            text="No data with those parameters, apply filters again if this is an error",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6c757d')
        )
    
    df = pd.DataFrame(data['timeseries'])
    df['time_bin'] = pd.to_datetime(df['time_bin'])
    
    # Use scatter plot instead of line
    fig = px.scatter(
        df,
        x='time_bin',
        y='count',
        color='incident_type',
        title='Incidents Over Time',
        labels={'time_bin': 'Time', 'count': 'Incident Count', 'incident_type': 'Type'},
        color_discrete_map={'collision': '#dc3545', 'near_miss': '#ffc107'},
        size_max=15
    )
    
    # Make points larger with no connecting lines
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='white')),
        mode='markers'  # Only markers, no lines
    )
    
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title="Time",
        yaxis_title="Incident Count",
        legend=dict(
            title="Incident Type",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


# =========================
# Main
# =========================

def main():
    """Start the dashboard."""
    print(f"Starting Incidents Dashboard on http://0.0.0.0:{DEFAULT_PORT}")
    print(f"Backend API: {BACKEND_API_URL}")
    
    app.run(
        host='0.0.0.0',
        port=DEFAULT_PORT,
        debug=False
    )


if __name__ == "__main__":
    main()
