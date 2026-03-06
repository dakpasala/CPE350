#!/usr/bin/env python3
"""
heatmap_viewer_enhanced.py

Real-time traffic heatmap with query parameter support.
Supports filtering by location, time period, and data source.

URL Examples:
- http://localhost:8052/?location=patterson&time_range=hour&source=historical
- http://localhost:8052/?location=all&time_range=day&source=historical
- http://localhost:8052/?time_range=week&source=historical
"""

import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse
import json
from collections import defaultdict, deque
import numpy as np
import requests


# =========================
# Config
# =========================

# ⚠️ ADD YOUR MAPBOX API KEY HERE ⚠️
MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN_HERE"  # Get free token at https://www.mapbox.com/

# Backend API configuration
BACKEND_API_URL = "http://127.0.0.1:8000"

# This will be injected by client.py for live streaming
get_latest_data = None

# Heatmap settings
HEATMAP_HISTORY_SIZE = 200  # Keep last N data points for heatmap
INCIDENT_DISPLAY_TIME = 300  # Show incidents for 5 minutes

# Historical data storage
vehicle_history = deque(maxlen=HEATMAP_HISTORY_SIZE)
incident_history = []

# Data source modes
DATA_SOURCE_LIVE = "live"  # WebSocket streaming
DATA_SOURCE_HISTORICAL = "historical"  # Fetch from API


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

app.title = "Traffic Heatmap Monitor"


# =========================
# Helper Functions
# =========================

def time_range_to_minutes(time_range_str):
    """
    Convert server time_range string to minutes for incident API.
    Maps directly to what the server expects.
    """
    mapping = {
        'hour': 60,
        '6hours': 360,
        '12hours': 720,
        'day': 1440,
        'week': 10080,
    }
    return mapping.get(time_range_str, 60)  # Default to 1 hour


def fetch_historical_data(location=None, time_range='hour'):
    """
    Fetch historical data from backend API.
    
    Args:
        location: Camera location (or None for all)
        time_range: Server time_range value ('hour', 'day', 'week', '6hours', '12hours')
    
    Returns:
        dict with 'vehicles' and 'incidents' keys
    """
    try:
        print(f"[FETCH] Fetching data: location={location}, time_range={time_range}")
        
        # Adaptive limit based on time range
        limit_map = {
            'hour': 10000,
            '6hours': 50000,
            '12hours': 100000,
            'day': 200000,
            'week': 625000,
        }
        
        # Fetch combined stats (vehicle data)
        stats_params = {
            'time_range': time_range,
            'limit': limit_map.get(time_range, 10000)
        }
        
        if location and location != 'all':
            stats_params['location'] = location
        
        print(f"[API] GET {BACKEND_API_URL}/stats/combined")
        print(f"[API] Params: {stats_params}")
        
        stats_response = requests.get(
            f"{BACKEND_API_URL}/stats/combined",
            params=stats_params,
            timeout=10
        )
        
        if not stats_response.ok:
            print(f"[ERROR] Stats API returned: {stats_response.status_code}")
            print(f"[ERROR] Response: {stats_response.text[:200]}")
        
        stats_data = stats_response.json() if stats_response.ok else {}
        
        # Fetch incidents
        minutes = time_range_to_minutes(time_range)
        incident_params = {'minutes': minutes}
        if location and location != 'all':
            incident_params['location'] = location
        
        print(f"[API] GET {BACKEND_API_URL}/incidents/timerange")
        print(f"[API] Params: {incident_params}")
        
        incidents_response = requests.get(
            f"{BACKEND_API_URL}/incidents/timerange",
            params=incident_params,
            timeout=10
        )
        
        if not incidents_response.ok:
            print(f"[ERROR] Incidents API returned: {incidents_response.status_code}")
        
        incidents_data = incidents_response.json() if incidents_response.ok else {}
        
        # Transform stats data to vehicle format
        vehicles = []
        raw_count = len(stats_data.get('data', []))
        
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
        
        incidents = incidents_data.get('incidents', [])
        
        print(f"[SUCCESS] Fetched {raw_count} raw records -> {len(vehicles)} vehicles with coords")
        print(f"[SUCCESS] Fetched {len(incidents)} incidents")
        
        return {
            'vehicles': vehicles,
            'incidents': incidents,
            'source': 'historical',
            'location': location or 'all',
            'time_range': time_range
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch historical data: {e}")
        import traceback
        traceback.print_exc()
        return {
            'vehicles': [],
            'incidents': [],
            'source': 'historical',
            'error': str(e)
        }


# =========================
# Layout
# =========================

app.layout = dbc.Container([
    # URL location component (for reading query params)
    dcc.Location(id='url', refresh=False),
    
    # Hidden stores for state
    dcc.Store(id='map-data-store'),
    dcc.Store(id='config-store'),  # Stores query params
    
    # Update interval
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every 1 second
        n_intervals=0
    ),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("TRAFFIC HEATMAP", className="display-4 mb-0"),
                html.P(id="subtitle", children="Real-time vehicle density & incident monitoring", 
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
                    # Location selector
                    dbc.Row([
                        dbc.Col([
                            html.Label("Camera Location:", className="text-muted small"),
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
                        ], width=6),
                        
                        # Time period selector - MATCHES SERVER API
                        dbc.Col([
                            html.Label("Time Period:", className="text-muted small"),
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
                        ], width=6),
                    ]),
                    
                    # Data source selector
                    dbc.Row([
                        dbc.Col([
                            html.Label("Data Source:", className="text-muted small"),
                            dbc.RadioItems(
                                id='source-radio',
                                options=[
                                    {'label': 'Live Stream', 'value': DATA_SOURCE_LIVE},
                                    {'label': 'Historical', 'value': DATA_SOURCE_HISTORICAL},
                                ],
                                value=DATA_SOURCE_HISTORICAL,  # Default to historical for testing
                                inline=True,
                                className="mb-3"
                            )
                        ])
                    ]),
                    
                    # Apply button
                    dbc.Button(
                        "Apply Filters",
                        id="apply-filters-btn",
                        color="primary",
                        className="w-100"
                    )
                ])
            ])
        ], width=12)
    ], className="mb-3"),
    
    # Stats Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="vehicle-count", className="mb-0 text-primary"),
                    html.P("Active Vehicles", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="incident-count", className="mb-0 text-danger"),
                    html.P("Active Incidents", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("--", id="last-update", className="mb-0 text-info"),
                    html.P("Last Update", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="heatmap-points", className="mb-0 text-warning"),
                    html.P("Heatmap Points", className="text-muted small mb-0")
                ])
            ], className="text-center")
        ], width=3),
    ], className="mb-4"),
    
    # Map Container
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Iframe(
                        id='heatmap-frame',
                        style={
                            'width': '100%',
                            'height': '800px',
                            'border': 'none',
                            'borderRadius': '8px'
                        }
                    )
                ], className="p-0")
            ])
        ])
    ]),
    
    # Legend
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Legend", className="mb-3"),
                    html.Div([
                        html.Span("🔵 Low Density", className="me-3"),
                        html.Span("🟢 Medium Density", className="me-3"),
                        html.Span("🟡 High Density", className="me-3"),
                        html.Span("🔴 Extreme Density", className="me-3"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("🔴 Collision", className="me-3"),
                        html.Span("🟠 Near Miss", className="me-3"),
                    ])
                ])
            ])
        ])
    ], className="mt-4")
    
], fluid=True, className="p-4")


# =========================
# Callbacks
# =========================

@app.callback(
    [
        Output('config-store', 'data'),
        Output('url', 'search')  # ← Add URL updating
    ],
    [
        Input('url', 'search'),
        Input('apply-filters-btn', 'n_clicks')
    ],
    [
        State('location-dropdown', 'value'),
        State('time-dropdown', 'value'),
        State('source-radio', 'value')
    ]
)
def update_config(url_search, n_clicks, location, time_range, source):
    """
    Update configuration from URL params or UI controls.
    """
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    # Parse URL on initial load
    if triggered_id == 'url' and url_search:
        params = parse_qs(url_search.lstrip('?'))
        location = params.get('location', [location])[0]
        time_range = params.get('time_range', [time_range])[0]
        source = params.get('source', [source])[0]
    
    config = {
        'location': location,
        'time_range': time_range,
        'source': source,
        'updated_at': datetime.utcnow().isoformat()
    }
    
    print(f"[CONFIG] Updated: {config}")
    
    # Build new URL query string when apply button is clicked
    new_url = ""
    if triggered_id == 'apply-filters-btn':
        params = []
        if location != 'all':
            params.append(f"location={location}")
        params.append(f"time_range={time_range}")
        params.append(f"source={source}")
        
        new_url = f"?{'&'.join(params)}" if params else ""
    
    return config, new_url


@app.callback(
    Output('subtitle', 'children'),
    Input('config-store', 'data')
)
def update_subtitle(config):
    """Update subtitle based on current config."""
    if not config:
        return "Real-time vehicle density & incident monitoring"
    
    location = config.get('location', 'all')
    time_range = config.get('time_range', 'hour')
    source = config.get('source', DATA_SOURCE_LIVE)
    
    location_text = location.upper() if location != 'all' else 'ALL CAMERAS'
    source_text = "LIVE STREAM" if source == DATA_SOURCE_LIVE else "HISTORICAL DATA"
    
    return f"{source_text} • {location_text} • {time_range.upper()}"


@app.callback(
    [
        Output('vehicle-count', 'children'),
        Output('incident-count', 'children'),
        Output('last-update', 'children'),
        Output('heatmap-points', 'children'),
        Output('map-data-store', 'data')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('config-store', 'data')
    ]
)
def update_data(n, config):
    """Update stats and prepare map data based on source."""
    
    if not config:
        return "0", "0", "--", "0", None
    
    source = config.get('source', DATA_SOURCE_LIVE)
    location = config.get('location')
    time_range = config.get('time_range', 'hour')
    
    # Fetch data based on source
    if source == DATA_SOURCE_HISTORICAL:
        # Fetch from API
        data = fetch_historical_data(location, time_range)
        vehicles = data['vehicles']
        incidents = data['incidents']
    else:
        # Get from WebSocket (live streaming)
        if get_latest_data is None:
            return "0", "0", "--", "0", None
        
        latest = get_latest_data()
        if latest is None:
            return "0", "0", "--", "0", None
        
        vehicles = latest.get('vehicles', [])
        incidents = latest.get('incidents', [])
        
        # Filter by location if specified
        if location and location != 'all':
            vehicles = [v for v in vehicles if v.get('location') == location]
            incidents = [i for i in incidents if i.get('location') == location]
    
    # Add to history for heatmap
    for vehicle in vehicles:
        if vehicle.get('lat') and vehicle.get('lon'):
            vehicle_history.append({
                'lat': vehicle['lat'],
                'lon': vehicle['lon'],
                'speed': vehicle.get('speed_mps', 0),
                'timestamp': vehicle.get('timestamp')
            })
    
    # Update incident history
    now = datetime.utcnow()
    for incident in incidents:
        incident_time = incident.get('timestamp')
        if isinstance(incident_time, str):
            try:
                incident_time = datetime.fromisoformat(incident_time.replace('Z', '+00:00'))
            except:
                incident_time = now
        incident['_added_at'] = incident_time
        incident_history.append(incident)
    
    # Clean old incidents
    cutoff = now.timestamp() - INCIDENT_DISPLAY_TIME
    cleaned_incidents = [
        inc for inc in incident_history
        if inc.get('_added_at', now).timestamp() > cutoff
    ]
    incident_history.clear()
    incident_history.extend(cleaned_incidents)
    
    # Format timestamp
    time_str = datetime.utcnow().strftime('%H:%M:%S')
    
    # Prepare map data
    map_data = {
        'heatmap_points': list(vehicle_history),
        'incidents': cleaned_incidents,
        'current_vehicles': vehicles
    }
    
    return (
        str(len(vehicles)),
        str(len(cleaned_incidents)),
        time_str,
        str(len(vehicle_history)),
        map_data
    )


@app.callback(
    Output('heatmap-frame', 'srcDoc'),
    Input('map-data-store', 'data')
)
def update_map(data):
    """Generate deck.gl heatmap HTML."""
    
    if not data:
        return generate_empty_map()
    
    heatmap_points = data.get('heatmap_points', [])
    incidents = data.get('incidents', [])
    current_vehicles = data.get('current_vehicles', [])
    
    return generate_heatmap_html(heatmap_points, incidents, current_vehicles)


# =========================
# Map Generation
# =========================

def generate_empty_map():
    """Generate empty map HTML with Mapbox."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/deck.gl@^8.9.0/dist.min.js"></script>
        <script src="https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.js"></script>
        <link href="https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.css" rel="stylesheet" />
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{ 
                margin: 0; 
                padding: 0; 
                font-family: 'JetBrains Mono', monospace;
                background: #0a0e27;
                overflow: hidden;
            }}
            #map {{ 
                width: 100vw; 
                height: 100vh; 
            }}
            #loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #00ffff;
                font-size: 18px;
                text-transform: uppercase;
                letter-spacing: 3px;
                text-align: center;
                z-index: 1000;
            }}
            .loader {{
                width: 60px;
                height: 60px;
                border: 4px solid rgba(0, 255, 255, 0.1);
                border-top: 4px solid #00ffff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div id="loading">
            <div class="loader"></div>
            <div>Waiting for data...</div>
        </div>
        <script>
            mapboxgl.accessToken = '{MAPBOX_TOKEN}';
            
            const map = new mapboxgl.Map({{
                container: 'map',
                style: 'mapbox://styles/mapbox/dark-v11',
                center: [-120.6596, 35.2828],
                zoom: 12,
                pitch: 45,
                bearing: 0
            }});
            
            map.on('load', () => {{
                console.log('[MAP] Map loaded - waiting for data');
            }});
        </script>
    </body>
    </html>
    """


def generate_heatmap_html(heatmap_points, incidents, current_vehicles):
    """Generate deck.gl heatmap with Mapbox GL base."""
    
    # Convert data to JSON
    heatmap_data = json.dumps([
        [p['lon'], p['lat'], p.get('speed', 1)] 
        for p in heatmap_points
    ])
    
    incident_data = json.dumps([
        {
            'position': [inc.get('lon', 0), inc.get('lat', 0)],
            'type': inc.get('incident_type', 'unknown'),
            'severity': inc.get('severity', 0.5)
        }
        for inc in incidents
        if inc.get('lat') and inc.get('lon')
    ])
    
    vehicle_data = json.dumps([
        {
            'position': [v['lon'], v['lat']],
            'speed': v.get('speed_mps', 0),
            'heading': v.get('heading_deg', 0)
        }
        for v in current_vehicles
        if v.get('lat') and v.get('lon')
    ])
    
    # Calculate center point
    if heatmap_points:
        center_lat = np.mean([p['lat'] for p in heatmap_points])
        center_lon = np.mean([p['lon'] for p in heatmap_points])
    else:
        center_lat, center_lon = 35.2828, -120.6596  # San Luis Obispo
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/deck.gl@^8.9.0/dist.min.js"></script>
        <script src="https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.js"></script>
        <link href="https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.css" rel="stylesheet" />
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {{ 
                margin: 0; 
                padding: 0; 
                font-family: 'JetBrains Mono', monospace;
                background: #0a0e27;
                overflow: hidden;
            }}
            #map {{ 
                width: 100vw; 
                height: 100vh; 
            }}
            #info {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(10, 14, 39, 0.95);
                color: #00ff88;
                padding: 15px 20px;
                border-radius: 8px;
                font-size: 12px;
                border: 2px solid #00ff88;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
                z-index: 1000;
                backdrop-filter: blur(10px);
            }}
            #info h4 {{
                margin: 0 0 10px 0;
                color: #00ff88;
                text-transform: uppercase;
                letter-spacing: 2px;
                font-size: 14px;
            }}
            #info p {{
                margin: 5px 0;
                color: #88ffcc;
            }}
            .pulse {{
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div id="info">
            <h4 class="pulse">LIVE HEATMAP</h4>
            <p>Points: {len(heatmap_points)}</p>
            <p>Incidents: {len(incidents)}</p>
            <p>Vehicles: {len(current_vehicles)}</p>
        </div>
        <script>
            const {{DeckGL, HeatmapLayer, ScatterplotLayer, MapboxOverlay}} = deck;
            
            mapboxgl.accessToken = '{MAPBOX_TOKEN}';
            
            // Initialize Mapbox
            const map = new mapboxgl.Map({{
                container: 'map',
                style: 'mapbox://styles/mapbox/dark-v11',
                center: [{center_lon}, {center_lat}],
                zoom: 14,
                pitch: 45,
                bearing: 0,
                antialias: true
            }});
            
            // Data
            const heatmapData = {heatmap_data};
            const incidentData = {incident_data};
            const vehicleData = {vehicle_data};
            
            // Create layers
            const layers = [
                // Heatmap layer
                new HeatmapLayer({{
                    id: 'heatmap',
                    data: heatmapData,
                    getPosition: d => d,
                    getWeight: d => d[2],
                    radiusPixels: 60,
                    intensity: 1.5,
                    threshold: 0.03,
                    colorRange: [
                        [0, 0, 255, 100],      // Blue
                        [0, 255, 0, 150],      // Green
                        [255, 255, 0, 200],    // Yellow
                        [255, 128, 0, 255],    // Orange
                        [255, 0, 0, 255]       // Red
                    ]
                }}),
                
                // Current vehicles
                new ScatterplotLayer({{
                    id: 'vehicles',
                    data: vehicleData,
                    getPosition: d => d.position,
                    getFillColor: [0, 255, 255, 200],
                    getRadius: 8,
                    radiusScale: 1,
                    radiusMinPixels: 5,
                    radiusMaxPixels: 15,
                    pickable: true
                }}),
                
                // Incidents
                new ScatterplotLayer({{
                    id: 'incidents',
                    data: incidentData,
                    getPosition: d => d.position,
                    getFillColor: d => d.type === 'collision' ? [255, 0, 0, 255] : [255, 165, 0, 255],
                    getRadius: d => 50 + d.severity * 100,
                    radiusScale: 1,
                    radiusMinPixels: 25,
                    radiusMaxPixels: 100,
                    pickable: true,
                    stroked: true,
                    lineWidthMinPixels: 4,
                    getLineColor: [255, 255, 255, 255]
                }})
            ];
            
            // Add deck.gl overlay to map
            map.on('load', () => {{
                const deckOverlay = new MapboxOverlay({{
                    interleaved: true,
                    layers: layers
                }});
                
                map.addControl(deckOverlay);
                
                console.log('[HEATMAP] Loaded with', heatmapData.length, 'points');
            }});
            
            // Add navigation controls
            map.addControl(new mapboxgl.NavigationControl(), 'bottom-right');
            
            // Add fullscreen control
            map.addControl(new mapboxgl.FullscreenControl(), 'bottom-right');
        </script>
    </body>
    </html>
    """


# =========================
# Main
# =========================

def main():
    """Start the heatmap viewer."""
    app.run(
        host='0.0.0.0',  # Listen on all network interfaces
        port=8052,
        debug=False
    )


if __name__ == "__main__":
    main()
