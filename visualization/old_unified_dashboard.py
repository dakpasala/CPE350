#!/usr/bin/env python3
"""
unified_dashboard.py

Comprehensive traffic monitoring dashboard combining:
- Heatmap visualization (deck.gl + Mapbox)
- Exit direction counts
- Speed analysis
- Incident tracking

URL Examples:
- http://localhost:8060/?location=patterson&time_range=hour&source=historical
- http://localhost:8060/?location=all&time_range=day&interval=1H
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
import json
from collections import deque
import numpy as np
from typing import Optional, Dict, Any


# =========================
# Config
# =========================

MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN_HERE"
BACKEND_API_URL = "http://127.0.0.1:8000"
DEFAULT_PORT = 8052  # Changed from 8060 to 8052
MPS_TO_MPH = 2.2369362920544

# Caltrans Colors
CALTRANS_BLUE = "#003366"
CALTRANS_GREEN = "#007B5F"

# This will be injected by client.py for live streaming
get_latest_data = None

# Heatmap settings
HEATMAP_HISTORY_SIZE = 200
INCIDENT_DISPLAY_TIME = 300
vehicle_history = deque(maxlen=HEATMAP_HISTORY_SIZE)
incident_history = []

DATA_SOURCE_LIVE = "live"
DATA_SOURCE_HISTORICAL = "historical"


# =========================
# Dash App
# =========================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Traffic Monitoring Dashboard"


# =========================
# Helper Functions
# =========================

def heading_to_cardinal(h: float) -> str:
    """Convert heading to cardinal direction."""
    h = float(h) % 360.0
    if h >= 315.0 or h < 45.0:
        return "N"
    if h < 135.0:
        return "E"
    if h < 225.0:
        return "S"
    return "W"


def time_range_to_minutes(time_range_str):
    """Convert time_range to minutes."""
    mapping = {'hour': 60, '6hours': 360, '12hours': 720, 'day': 1440, 'week': 10080}
    return mapping.get(time_range_str, 60)


def fetch_historical_data(location=None, time_range='hour'):
    """Fetch all data from backend APIs."""
    try:
        print(f"[FETCH] location={location}, time_range={time_range}")
        
        limit_map = {'hour': 10000, '6hours': 50000, '12hours': 100000, 'day': 200000, 'week': 625000}
        
        # Increase timeout for larger queries
        timeout_map = {'hour': 10, '6hours': 20, '12hours': 30, 'day': 45, 'week': 90}
        timeout = timeout_map.get(time_range, 10)
        
        # Fetch stats
        stats_params = {'time_range': time_range, 'limit': limit_map.get(time_range, 10000)}
        if location and location != 'all':
            stats_params['location'] = location
        
        print(f"[API] Fetching stats with timeout={timeout}s...")
        stats_response = requests.get(f"{BACKEND_API_URL}/stats/combined", params=stats_params, timeout=timeout)
        stats_data = stats_response.json() if stats_response.ok else {}
        print(f"[API] Stats fetched: {len(stats_data.get('data', []))} records")
        
        # Fetch incidents
        minutes = time_range_to_minutes(time_range)
        incident_params = {'minutes': minutes}
        if location and location != 'all':
            incident_params['location'] = location
        
        print(f"[API] Fetching incidents with timeout={timeout}s...")
        incidents_response = requests.get(f"{BACKEND_API_URL}/incidents/timerange", params=incident_params, timeout=timeout)
        incidents_data = incidents_response.json() if incidents_response.ok else {}
        print(f"[API] Incidents fetched: {len(incidents_data.get('incidents', []))} records")
        
        # Process vehicles
        vehicles = []
        if stats_data.get('data'):
            for row in stats_data['data']:
                if row.get('lat') and row.get('lon'):
                    vehicles.append({
                        'id': row.get('object_id', 'unknown'),
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'speed_mps': row.get('speed_mps', 0),
                        'heading_deg': row.get('heading_deg', 0),
                        'timestamp': row.get('timestamp'),
                        'detected_type': row.get('detected_type', 'unknown'),
                        'location': row.get('location', 'unknown')
                    })
        
        print(f"[SUCCESS] Processed {len(vehicles)} vehicles, {len(incidents_data.get('incidents', []))} incidents")
        
        return {
            'vehicles': vehicles,
            'incidents': incidents_data.get('incidents', []),
            'raw_stats': stats_data.get('data', [])
        }
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out - the backend took too long to respond")
        print(f"[HINT] Try a smaller time range or wait for the backend to finish processing")
        return {'vehicles': [], 'incidents': [], 'raw_stats': [], 'error': 'timeout'}
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {'vehicles': [], 'incidents': [], 'raw_stats': [], 'error': str(e)}


def process_exit_counts(df, only_type='car', interval=None):
    """Process exit direction counts."""
    if df.empty:
        return None, None
    
    required = ["timestamp", "object_id", "heading_deg"]
    if not all(col in df.columns for col in required):
        return None, None
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]
    
    if only_type.lower() != "any" and "detected_type" in df.columns:
        df = df[df["detected_type"].str.lower() == only_type.lower()]
    
    if df.empty:
        return None, None
    
    df = df.sort_values("timestamp")
    df_exit = df.groupby("object_id", as_index=False).tail(1).copy()
    df_exit["heading_deg"] = pd.to_numeric(df_exit["heading_deg"], errors="coerce")
    df_exit = df_exit[df_exit["heading_deg"].notna()]
    
    if df_exit.empty:
        return None, None
    
    df_exit["direction"] = df_exit["heading_deg"].apply(heading_to_cardinal)
    
    total = df_exit.groupby("direction", as_index=False).size().rename(columns={"size": "count"}).sort_values("direction")
    
    timeseries = None
    if interval:
        df_exit["time_bin"] = df_exit["timestamp"].dt.floor(interval)
        timeseries = df_exit.groupby(["time_bin", "direction"], as_index=False).size().rename(columns={"size": "count"}).sort_values("time_bin")
    
    return total, timeseries


def process_speed_data(df, only_type='car', interval=None):
    """Process speed statistics."""
    if df.empty:
        return None, None
    
    required = ["timestamp", "object_id", "heading_deg", "speed_mps"]
    if not all(col in df.columns for col in required):
        return None, None
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]
    
    if only_type.lower() != "any" and "detected_type" in df.columns:
        df = df[df["detected_type"].str.lower() == only_type.lower()]
    
    df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce")
    df = df[df["speed_mps"].notna()]
    df = df[df["speed_mps"] >= 0.0]
    
    if df.empty:
        return None, None
    
    df = df.sort_values("timestamp")
    df_exit = df.groupby("object_id", as_index=False).tail(1).copy()
    df_exit["heading_deg"] = pd.to_numeric(df_exit["heading_deg"], errors="coerce")
    df_exit = df_exit[df_exit["heading_deg"].notna()]
    
    if df_exit.empty:
        return None, None
    
    df_exit["direction"] = df_exit["heading_deg"].apply(heading_to_cardinal)
    df_exit["speed_mph"] = df_exit["speed_mps"] * MPS_TO_MPH
    
    total = df_exit.groupby("direction", as_index=False).agg(
        avg_speed_mph=("speed_mph", "mean"),
        count=("speed_mph", "size")
    ).sort_values("direction")
    
    timeseries = None
    if interval:
        df_exit["time_bin"] = df_exit["timestamp"].dt.floor(interval)
        timeseries = df_exit.groupby(["time_bin", "direction"], as_index=False).agg(
            avg_speed_mph=("speed_mph", "mean"),
            count=("speed_mph", "size")
        ).sort_values("time_bin")
    
    return total, timeseries


def process_incident_data(incidents, incident_type=None, interval=None):
    """Process incidents."""
    if not incidents:
        return None, None
    
    df = pd.DataFrame(incidents)
    if df.empty:
        return None, None
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]
    
    if incident_type and incident_type != 'all':
        df = df[df["incident_type"] == incident_type]
    
    if df.empty:
        return None, None
    
    type_counts = df.groupby("incident_type", as_index=False).size().rename(columns={"size": "count"}).sort_values("incident_type")
    
    timeseries = None
    if interval:
        df["time_bin"] = df["timestamp"].dt.floor(interval)
        timeseries = df.groupby(["time_bin", "incident_type"], as_index=False).size().rename(columns={"size": "count"}).sort_values("time_bin")
    
    return type_counts, timeseries


def generate_heatmap_html(heatmap_points, incidents, current_vehicles, is_dark=True):
    """Generate deck.gl heatmap HTML with theme support."""
    
    heatmap_data = json.dumps([[p['lon'], p['lat'], p.get('speed', 1)] for p in heatmap_points])
    incident_data = json.dumps([
        {'position': [inc.get('lon', 0), inc.get('lat', 0)], 'type': inc.get('incident_type', 'unknown'), 'severity': inc.get('severity', 0.5)}
        for inc in incidents if inc.get('lat') and inc.get('lon')
    ])
    vehicle_data = json.dumps([
        {'position': [v['lon'], v['lat']], 'speed': v.get('speed_mps', 0), 'heading': v.get('heading_deg', 0)}
        for v in current_vehicles if v.get('lat') and v.get('lon')
    ])
    
    if heatmap_points:
        center_lat = np.mean([p['lat'] for p in heatmap_points])
        center_lon = np.mean([p['lon'] for p in heatmap_points])
    else:
        center_lat, center_lon = 35.2828, -120.6596
    
    # Theme-aware styles
    bg_color = "#0a0e27" if is_dark else "#f5f7fa"
    map_style = "mapbox://styles/mapbox/dark-v11" if is_dark else "mapbox://styles/mapbox/light-v11"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/deck.gl@^8.9.0/dist.min.js"></script>
        <script src="https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.js"></script>
        <link href="https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.css" rel="stylesheet" />
        <style>
            body {{ margin: 0; padding: 0; background: {bg_color}; overflow: hidden; }}
            #map {{ width: 100vw; height: 100vh; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            const {{HeatmapLayer, ScatterplotLayer, MapboxOverlay}} = deck;
            mapboxgl.accessToken = '{MAPBOX_TOKEN}';
            
            const map = new mapboxgl.Map({{
                container: 'map',
                style: '{map_style}',
                center: [{center_lon}, {center_lat}],
                zoom: 14,
                pitch: 45,
                bearing: 0
            }});
            
            const layers = [
                new HeatmapLayer({{
                    id: 'heatmap',
                    data: {heatmap_data},
                    getPosition: d => d,
                    getWeight: d => d[2],
                    radiusPixels: 60,
                    intensity: 1.5,
                    threshold: 0.03,
                    colorRange: [[0,0,255,100], [0,255,0,150], [255,255,0,200], [255,128,0,255], [255,0,0,255]]
                }}),
                new ScatterplotLayer({{
                    id: 'vehicles',
                    data: {vehicle_data},
                    getPosition: d => d.position,
                    getFillColor: [0,255,255,200],
                    getRadius: 8,
                    radiusMinPixels: 5,
                    radiusMaxPixels: 15
                }}),
                new ScatterplotLayer({{
                    id: 'incidents',
                    data: {incident_data},
                    getPosition: d => d.position,
                    getFillColor: d => d.type === 'collision' ? [255,0,0,255] : [255,165,0,255],
                    getRadius: d => 50 + d.severity * 100,
                    radiusMinPixels: 25,
                    radiusMaxPixels: 100,
                    stroked: true,
                    lineWidthMinPixels: 4,
                    getLineColor: [255,255,255,255]
                }})
            ];
            
            map.on('load', () => {{
                map.addControl(new MapboxOverlay({{ interleaved: true, layers }}));
                map.addControl(new mapboxgl.NavigationControl(), 'bottom-right');
            }});
        </script>
    </body>
    </html>
    """


# =========================
# Layout
# =========================

# =========================
# CSS Styling
# =========================

CUSTOM_CSS = """
body { margin: 0; padding: 0; overflow-x: hidden; }

.main-container {
    padding: 30px;
    background: #f5f7fa;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    transition: background 0.2s ease, color 0.2s ease;
}
[data-theme="dark"] .main-container {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
}

.chart-card {
    transition: all 0.3s ease;
}
.chart-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 51, 102, 0.2) !important;
}

/* Theme Toggle Button - Caltrans Style */
.theme-toggle-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 18px;
    border-radius: 4px;
    border: 2px solid rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.1);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: white;
    transition: all 0.25s ease;
    user-select: none;
    white-space: nowrap;
}
.theme-toggle-btn:hover {
    background: rgba(255,255,255,0.2);
}
.theme-toggle-btn .toggle-track {
    width: 38px;
    height: 22px;
    border-radius: 11px;
    background: #ccc;
    position: relative;
    transition: background 0.25s ease;
    flex-shrink: 0;
}
.theme-toggle-btn .toggle-track.active {
    background: #007B5F;
}
.theme-toggle-btn .toggle-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
    position: absolute;
    top: 2px;
    left: 2px;
    transition: transform 0.25s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.theme-toggle-btn .toggle-track.active .toggle-thumb {
    transform: translateX(16px);
}

/* Home Button */
.home-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 4px;
    border: 2px solid rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.1);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: white;
    transition: all 0.25s ease;
    text-decoration: none;
}
.home-btn:hover {
    background: rgba(255,255,255,0.2);
    color: white;
}
"""

TOGGLE_JS = """
window.DASHBOARD_THEME_KEY = 'dashboard_theme';
(function() {
    var match = document.cookie.match('(?:^|; )dashboard_theme=([^;]*)');
    var saved = match ? match[1] : null;
    if (saved) document.documentElement.setAttribute('data-theme', saved);
})();
"""

INDEX_STRING = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + CUSTOM_CSS + '''</style>
        <script>''' + TOGGLE_JS + '''</script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.index_string = INDEX_STRING


# =========================
# Layout Helper Functions
# =========================

def get_card_style(is_dark=False):
    """Get card styling based on theme - Caltrans formal design."""
    return {
        "background": "#2d2d2d" if is_dark else "white",
        "padding": "20px",
        "borderRadius": "4px",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
        "border": f"1px solid {'#404040' if is_dark else '#eee'}",
        "marginBottom": "20px"
    }


def make_filter_card(is_dark=False, current_values=None):
    """Create the filters card."""
    if current_values is None:
        current_values = {
            'location': 'all',
            'time': 'hour',
            'type': 'car',
            'interval': '15min',
            'source': DATA_SOURCE_HISTORICAL
        }
    
    text_color = "#e0e0e0" if is_dark else "#333"
    bg_color = "#1f1f1f" if is_dark else "#f8f9fa"
    
    return html.Div([
        html.H3("FILTERS", style={
            "margin": "0 0 20px 0",
            "fontSize": "18px",
            "fontWeight": "600",
            "color": text_color,
            "borderBottom": f"2px solid {'#404040' if is_dark else '#eee'}",
            "paddingBottom": "12px"
        }),
        html.Div([
            html.Div([
                html.Label("Location", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginBottom": "5px", "display": "block"}),
                dcc.Dropdown(
                    id='location-dropdown',
                    options=[
                        {'label': 'All Cameras', 'value': 'all'},
                        {'label': 'Patterson', 'value': 'patterson'},
                        {'label': 'Downtown', 'value': 'downtown'},
                        {'label': 'Highway 1', 'value': 'highway1'},
                    ],
                    value=current_values['location'],  # Use preserved value
                    clearable=False,
                    style={"marginBottom": "15px"}
                )
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Time Range", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginBottom": "5px", "display": "block"}),
                dcc.Dropdown(
                    id='time-dropdown',
                    options=[
                        {'label': 'Last 1 hour', 'value': 'hour'},
                        {'label': 'Last 6 hours', 'value': '6hours'},
                        {'label': 'Last 12 hours', 'value': '12hours'},
                        {'label': 'Last 1 day', 'value': 'day'},
                        {'label': 'Last 1 week', 'value': 'week'},
                    ],
                    value=current_values['time'],  # Use preserved value
                    clearable=False,
                    style={"marginBottom": "15px"}
                )
            ], style={"flex": "1", "marginLeft": "15px"}),
            html.Div([
                html.Label("Vehicle Type", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginBottom": "5px", "display": "block"}),
                dcc.Dropdown(
                    id='type-dropdown',
                    options=[
                        {'label': 'All Types', 'value': 'any'},
                        {'label': 'Car', 'value': 'car'},
                        {'label': 'Truck', 'value': 'truck'},
                        {'label': 'Bus', 'value': 'bus'},
                    ],
                    value=current_values['type'],  # Use preserved value
                    clearable=False,
                    style={"marginBottom": "15px"}
                )
            ], style={"flex": "1", "marginLeft": "15px"}),
            html.Div([
                html.Label("Interval", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginBottom": "5px", "display": "block"}),
                dcc.Dropdown(
                    id='interval-dropdown',
                    options=[
                        {'label': '1 min', 'value': '1min'},
                        {'label': '5 min', 'value': '5min'},
                        {'label': '15 min', 'value': '15min'},
                        {'label': '1 hour', 'value': '1h'},
                    ],
                    value=current_values['interval'],  # Use preserved value
                    clearable=False,
                    style={"marginBottom": "15px"}
                )
            ], style={"flex": "1", "marginLeft": "15px"}),
            html.Div([
                html.Label("Source", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginBottom": "5px", "display": "block"}),
                dcc.Dropdown(
                    id='source-dropdown',
                    options=[
                        {'label': 'Live', 'value': DATA_SOURCE_LIVE},
                        {'label': 'Historical', 'value': DATA_SOURCE_HISTORICAL},
                    ],
                    value=current_values['source'],  # Use preserved value
                    clearable=False,
                    style={"marginBottom": "15px"}
                )
            ], style={"flex": "0.8", "marginLeft": "15px"}),
        ], style={"display": "flex", "alignItems": "flex-end"}),
        html.Button("Apply Filters", id="apply-btn", n_clicks=0, style={
            "width": "100%",
            "padding": "12px",
            "background": CALTRANS_BLUE, # "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "color": "white",
            "border": "none",
            "borderRadius": "8px",
            "fontSize": "14px",
            "fontWeight": "600",
            "cursor": "pointer",
            "marginTop": "10px",
            "transition": "all 0.3s ease"
        })
    ], style=get_card_style(is_dark))


def make_theme_toggle(toggle_id="theme-toggle"):
    """Return the theme toggle button component."""
    return html.Div(
        id=f"{toggle_id}-wrapper",
        children=[
            dcc.Checklist(
                id=toggle_id,
                options=[{"label": "", "value": "dark"}],
                value=[],
                style={"display": "none"},
            ),
            html.Button(
                id=f"{toggle_id}-btn",
                children=[
                    html.Div([
                        html.Div(className="toggle-thumb"),
                    ], id=f"{toggle_id}-track", className="toggle-track"),
                    html.Span("Theme: Light", id=f"{toggle_id}-label"),
                ],
                className="theme-toggle-btn",
                n_clicks=0,
            ),
        ],
        style={
            "position": "fixed",
            "top": "25px",
            "right": "30px",
            "zIndex": 10000,
        }
    )


def make_home_button():
    """Return the home button component."""
    return html.A(
        href="http://127.0.0.1:8050",
        children=[
            html.Span("← Home"),
        ],
        className="home-btn",
        style={
            "position": "fixed",
            "top": "25px",
            "left": "30px",
            "zIndex": 10000,
        }
    )


# =========================
# Layout
# =========================

app.layout = html.Div([
    make_theme_toggle("theme-toggle"),
    make_home_button(),
    
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='config-store'),
    dcc.Store(id='data-store'),
    dcc.Store(id='theme-store', data='light'),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    
    # Caltrans-style header
    html.Div([
        html.H1("TRAFFIC ANALYSIS DASHBOARD", style={
            "color": "white",
            "margin": "0",
            "fontSize": "32px",
            "fontWeight": "800",
            "letterSpacing": "1.5px"
        }),
        html.Div(style={
            "width": "50px",
            "height": "4px",
            "background": CALTRANS_GREEN,
            "margin": "15px auto"
        }),
        html.P("DIVISION OF TRAFFIC OPERATIONS • COMPREHENSIVE ANALYTICS PLATFORM", style={
            "color": "rgba(255,255,255,0.9)",
            "margin": "0",
            "fontSize": "13px",
            "letterSpacing": "2px",
            "fontWeight": "500"
        }),
    ], style={
        "padding": "45px 30px",
        "background": CALTRANS_BLUE,
        "borderRadius": "4px",
        "marginBottom": "30px",
        "boxShadow": "0 4px 15px rgba(0,0,0,0.1)",
        "textAlign": "center",
        "borderBottom": f"6px solid {CALTRANS_GREEN}"
    }),
    
    html.Div(id='filters-container'),
    html.Div(id='dashboard-content'),
    
], id="main-container", className="main-container")


# =========================
# Callbacks
# =========================

@app.callback(
    Output('filters-container', 'children'),
    Input('theme-store', 'data'),
)
def render_filters(theme):
    is_dark = theme == "dark"
    # Use default values on initial load
    current_values = {
        'location': 'all',
        'time': 'hour',
        'type': 'car',
        'interval': '15min',
        'source': DATA_SOURCE_HISTORICAL
    }
    return make_filter_card(is_dark, current_values)


@app.callback(
    Output('dashboard-content', 'children'),
    [Input('data-store', 'data'), Input('theme-store', 'data')]
)
def render_content(data, theme):
    is_dark = theme == "dark"
    card_style = get_card_style(is_dark)
    
    if not data:
        return html.Div("Loading...", style=card_style)
    
    return html.Div([
        # Heatmap
        html.Div([
            html.H3("HEATMAP", style={
                "margin": "0 0 15px 0",
                "fontSize": "18px",
                "fontWeight": "600",
                "color": "#e0e0e0" if is_dark else "#333"
            }),
            html.Iframe(id='heatmap-frame', style={'width': '100%', 'height': '500px', 'border': 'none', 'borderRadius': '8px'})
        ], style=card_style, className="chart-card"),
        
        # Charts Row
        html.Div([
            html.Div([
                html.H3("EXIT DIRECTION COUNTS", style={
                    "margin": "0 0 15px 0",
                    "fontSize": "18px",
                    "fontWeight": "600",
                    "color": "#e0e0e0" if is_dark else "#333"
                }),
                dcc.Graph(id='count-chart', style={'height': '350px'}, config={'displayModeBar': False})
            ], style={**card_style, "flex": "1", "marginRight": "10px"}, className="chart-card"),
            
            html.Div([
                html.H3("AVERAGE SPEED BY DIRECTION", style={
                    "margin": "0 0 15px 0",
                    "fontSize": "18px",
                    "fontWeight": "600",
                    "color": "#e0e0e0" if is_dark else "#333"
                }),
                dcc.Graph(id='speed-chart', style={'height': '350px'}, config={'displayModeBar': False})
            ], style={**card_style, "flex": "1", "marginLeft": "10px"}, className="chart-card"),
        ], style={"display": "flex"}),
        
        # Incidents Chart
        html.Div([
            html.H3("INCIDENTS OVER TIME", style={
                "margin": "0 0 15px 0",
                "fontSize": "18px",
                "fontWeight": "600",
                "color": "#e0e0e0" if is_dark else "#333"
            }),
            dcc.Graph(id='incidents-chart', style={'height': '400px'}, config={'displayModeBar': False})
        ], style=card_style, className="chart-card"),
    ])


@app.callback(
    [Output('config-store', 'data'), Output('url', 'search')],
    [Input('url', 'search'), Input('apply-btn', 'n_clicks')],
    [State('location-dropdown', 'value'), State('time-dropdown', 'value'), 
     State('type-dropdown', 'value'), State('interval-dropdown', 'value'), State('source-dropdown', 'value')]
)
def update_config(url_search, n_clicks, location, time_range, vehicle_type, interval, source):
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    print(f"[CONFIG UPDATE] Triggered by: {triggered_id}")
    print(f"[CONFIG UPDATE] n_clicks: {n_clicks}")
    print(f"[CONFIG UPDATE] Inputs: location={location}, time_range={time_range}, type={vehicle_type}, interval={interval}, source={source}")
    
    if triggered_id == 'url' and url_search:
        params = parse_qs(url_search.lstrip('?'))
        location = params.get('location', [location])[0]
        time_range = params.get('time_range', [time_range])[0]
        vehicle_type = params.get('only_type', [vehicle_type])[0]
        interval = params.get('interval', [interval])[0]
        source = params.get('source', [source])[0]
    
    config = {
        'location': location,
        'time_range': time_range,
        'only_type': vehicle_type,
        'interval': interval,
        'source': source,
        'updated_at': datetime.utcnow().isoformat()
    }
    
    print(f"[CONFIG] New config: {config}")
    
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
        params.append(f"source={source}")
        new_url = f"?{'&'.join(params)}"
        print(f"[CONFIG] New URL: {new_url}")
    
    return config, new_url


@app.callback(
    Output('subtitle', 'children'),
    Input('config-store', 'data')
)
def update_subtitle(config):
    if not config:
        return "Comprehensive traffic analysis"
    
    loc = config.get('location', 'all').upper() if config.get('location') != 'all' else 'ALL CAMERAS'
    return f"{loc} • {config.get('time_range', 'hour').upper()} • {config.get('source', DATA_SOURCE_HISTORICAL).upper()}"


@app.callback(
    Output('data-store', 'data'),
    Input('config-store', 'data')
)
def fetch_data(config):
    if not config:
        return None
    
    source = config.get('source', DATA_SOURCE_HISTORICAL)
    location = config.get('location')
    time_range = config.get('time_range', 'hour')
    
    if source == DATA_SOURCE_HISTORICAL:
        data = fetch_historical_data(location, time_range)
    else:
        if get_latest_data is None:
            return None
        latest = get_latest_data()
        if not latest:
            return None
        data = {
            'vehicles': latest.get('vehicles', []),
            'incidents': latest.get('incidents', []),
            'raw_stats': []
        }
    
    # Process all analyses
    df_stats = pd.DataFrame(data['raw_stats']) if data['raw_stats'] else pd.DataFrame()
    
    exit_counts, exit_ts = process_exit_counts(df_stats, config.get('only_type', 'car'), config.get('interval'))
    speed_stats, speed_ts = process_speed_data(df_stats, config.get('only_type', 'car'), config.get('interval'))
    incident_counts, incident_ts = process_incident_data(data['incidents'], None, config.get('interval'))
    
    return {
        'vehicles': data['vehicles'],
        'incidents': data['incidents'],
        'exit_counts': exit_counts.to_dict('records') if exit_counts is not None else None,
        'exit_ts': exit_ts.to_dict('records') if exit_ts is not None else None,
        'speed_stats': speed_stats.to_dict('records') if speed_stats is not None else None,
        'speed_ts': speed_ts.to_dict('records') if speed_ts is not None else None,
        'incident_counts': incident_counts.to_dict('records') if incident_counts is not None else None,
        'incident_ts': incident_ts.to_dict('records') if incident_ts is not None else None
    }


@app.callback(
    Output('heatmap-frame', 'srcDoc'),
    [Input('data-store', 'data'), Input('theme-store', 'data')]
)
def update_heatmap(data, theme):
    is_dark = theme == "dark"
    bg_color = "#0a0e27" if is_dark else "#f5f7fa"
    text_color = "#00ffff" if is_dark else "#333"
    
    if not data or not data.get('vehicles'):
        return f"<html><body style='background:{bg_color};color:{text_color};display:flex;align-items:center;justify-content:center;height:100vh;'>No data available</body></html>"
    
    # Add to history
    for v in data['vehicles']:
        if v.get('lat') and v.get('lon'):
            vehicle_history.append({'lat': v['lat'], 'lon': v['lon'], 'speed': v.get('speed_mps', 0)})
    
    return generate_heatmap_html(list(vehicle_history), data.get('incidents', []), data['vehicles'], is_dark)


@app.callback(
    Output('count-chart', 'figure'),
    [Input('data-store', 'data'), Input('theme-store', 'data')]
)
def update_count_chart(data, theme):
    is_dark = theme == "dark"
    
    if not data or not data.get('exit_counts'):
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            template='plotly_dark' if is_dark else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    df = pd.DataFrame(data['exit_counts'])
    fig = px.bar(df, x='direction', y='count', color='direction',
                 color_discrete_map={'N': '#17a2b8', 'E': '#28a745', 'S': '#ffc107', 'W': '#dc3545'})
    fig.update_layout(
        template='plotly_dark' if is_dark else 'plotly_white',
        showlegend=False,
        xaxis_title="Direction",
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


@app.callback(
    Output('speed-chart', 'figure'),
    [Input('data-store', 'data'), Input('theme-store', 'data')]
)
def update_speed_chart(data, theme):
    is_dark = theme == "dark"
    
    if not data or not data.get('speed_stats'):
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            template='plotly_dark' if is_dark else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    df = pd.DataFrame(data['speed_stats'])
    fig = px.bar(df, x='direction', y='avg_speed_mph', color='direction',
                 color_discrete_map={'N': '#17a2b8', 'E': '#28a745', 'S': '#ffc107', 'W': '#dc3545'},
                 hover_data={'count': True})
    fig.update_layout(
        template='plotly_dark' if is_dark else 'plotly_white',
        showlegend=False,
        xaxis_title="Direction",
        yaxis_title="Speed (mph)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


@app.callback(
    Output('incidents-chart', 'figure'),
    [Input('data-store', 'data'), Input('theme-store', 'data')]
)
def update_incidents_chart(data, theme):
    is_dark = theme == "dark"
    
    if not data or not data.get('incident_ts'):
        fig = go.Figure()
        fig.add_annotation(text="No timeseries data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            template='plotly_dark' if is_dark else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    df = pd.DataFrame(data['incident_ts'])
    df['time_bin'] = pd.to_datetime(df['time_bin'])
    fig = px.scatter(df, x='time_bin', y='count', color='incident_type',
                     color_discrete_map={'collision': '#dc3545', 'near_miss': '#ffc107'})
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='white')), mode='markers')
    fig.update_layout(
        template='plotly_dark' if is_dark else 'plotly_white',
        hovermode='x unified',
        xaxis_title="Time",
        yaxis_title="Incident Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# =========================
# Theme Toggle Callbacks
# =========================

# Wire up toggle button clicks
app.clientside_callback(
    """
    function(n_clicks, current_value) {
        if (n_clicks === undefined || n_clicks === null) {
            return window.dash_clientside.no_update;
        }
        var isDark = current_value && current_value.includes('dark');
        var newVal = isDark ? [] : ['dark'];
        return newVal;
    }
    """,
    Output("theme-toggle", "value"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-toggle", "value"),
    prevent_initial_call=True,
)

# Load theme from cookie on page mount
app.clientside_callback(
    """
    function(id) {
        var match = document.cookie.match('(?:^|; )dashboard_theme=([^;]*)');
        var saved = match ? match[1] : null;
        if (saved === 'dark') return ['dark'];
        return [];
    }
    """,
    Output("theme-toggle", "value", allow_duplicate=True),
    Input("theme-toggle", "id"),
    prevent_initial_call='initial_duplicate',
)

# Update button appearance + save to cookie when value changes
app.clientside_callback(
    """
    function(value) {
        var isDark = value && value.includes('dark');
        var theme = isDark ? 'dark' : 'light';
        document.cookie = 'dashboard_theme=' + theme + ';path=/;max-age=31536000;SameSite=Lax';
        document.documentElement.setAttribute('data-theme', theme);
        
        var btn = document.getElementById('theme-toggle-btn');
        var track = document.getElementById('theme-toggle-track');
        var label = document.getElementById('theme-toggle-label');
        if (btn && track && label) {
            if (isDark) {
                track.classList.add('active');
                label.textContent = 'Theme: Dark';
                btn.classList.add('dark-mode');
            } else {
                track.classList.remove('active');
                label.textContent = 'Theme: Light';
                btn.classList.remove('dark-mode');
            }
        }
        return theme;
    }
    """,
    Output("theme-store", "data"),
    Input("theme-toggle", "value"),
)


# =========================
# Main
# =========================

def main():
    print(f"Starting Unified Dashboard on http://0.0.0.0:{DEFAULT_PORT}")
    print(f"Backend API: {BACKEND_API_URL}")
    app.run(host='0.0.0.0', port=DEFAULT_PORT, debug=False)


if __name__ == "__main__":
    main()
