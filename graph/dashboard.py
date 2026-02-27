#!/usr/bin/env python3
"""
dashboard.py

Multi-camera security room dashboard.
Shows live previews of all active camera feeds in a grid layout.
"""

import os
import math
from collections import defaultdict
from dotenv import load_dotenv

import pandas as pd
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


# =========================
# Data source (injected by client.py)
# =========================

def get_latest_data():
    """Placeholder - replaced by client.py at runtime."""
    return None


load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN is missing in .env")

ANCHOR = {"lat": 34.441560, "lon": -119.808362}
MAP_STYLE = "mapbox://styles/mapbox/satellite-streets-v12"

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8050"))

COLOR_MAP = {
    "car": "rgb(255,0,0)",
    "truck": "rgb(255,140,0)",
    "person": "rgb(0,255,0)",
    "bus": "rgb(255,255,0)",
}
DEFAULT_COLOR = "rgb(0,128,255)"
INCIDENT_COLOR = "rgb(255,0,255)"


# =========================
# Data Processing
# =========================

def process_camera_feeds(raw_data):
    """
    Group data by location/camera for grid display.
    
    Returns:
        Dict of {location: {vehicles, incidents, latest_timestamp}}
    """
    if not raw_data:
        return {}
    
    vehicles = raw_data.get("vehicles", [])
    incidents = raw_data.get("incidents", [])
    
    if not vehicles:
        return {}
    
    # Group by location
    cameras = defaultdict(lambda: {"vehicles": [], "incidents": [], "latest_timestamp": None})
    
    for v in vehicles:
        loc = v.get("location", "unknown")
        cameras[loc]["vehicles"].append(v)
        
        # Track latest timestamp
        ts = v.get("timestamp")
        if ts:
            if cameras[loc]["latest_timestamp"] is None or ts > cameras[loc]["latest_timestamp"]:
                cameras[loc]["latest_timestamp"] = ts
    
    # Add incidents
    for inc in incidents:
        loc = inc.get("location", "unknown")
        if loc in cameras:
            cameras[loc]["incidents"].append(inc)
    
    return dict(cameras)


def compute_center(vehicles):
    """Calculate center point from vehicle positions."""
    pts = [(v.get("lat"), v.get("lon")) for v in vehicles]
    pts = [(lat, lon) for lat, lon in pts if lat is not None and lon is not None
           and math.isfinite(lat) and math.isfinite(lon)]
    
    if not pts:
        return ANCHOR
    
    lats = [lat for lat, _ in pts]
    lons = [lon for _, lon in pts]
    
    return {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}


def build_preview_map(vehicles, incidents, location):
    """Build small preview map for camera feed."""
    fig = go.Figure()
    
    if vehicles:
        lons, lats, colors = [], [], []
        
        for v in vehicles:
            lat, lon = v.get("lat"), v.get("lon")
            if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            
            has_incident = len(v.get("incidents", [])) > 0
            
            if has_incident:
                color = INCIDENT_COLOR
            else:
                typ = str(v.get("detected_type", "unknown")).lower()
                color = COLOR_MAP.get(typ, DEFAULT_COLOR)
            
            lats.append(lat)
            lons.append(lon)
            colors.append(color)
        
        if lats:
            fig.add_trace(go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode="markers",
                marker=dict(size=5, opacity=0.8, color=colors),
                hoverinfo="skip",
            ))
    
    # Add incident markers
    if incidents and vehicles:
        inc_lats, inc_lons = [], []
        
        for inc in incidents:
            vehicle_ids = inc.get("vehicles", [])
            if not vehicle_ids:
                continue
            
            first_vehicle = next(
                (v for v in vehicles if str(v.get("id")) == str(vehicle_ids[0])),
                None
            )
            
            if not first_vehicle:
                continue
            
            lat, lon = first_vehicle.get("lat"), first_vehicle.get("lon")
            if lat is None or lon is None:
                continue
            
            inc_lats.append(lat)
            inc_lons.append(lon)
        
        if inc_lats:
            fig.add_trace(go.Scattermapbox(
                lon=inc_lons,
                lat=inc_lats,
                mode="markers+text",
                marker=dict(size=12, opacity=0.8, color="rgb(255,0,0)"),
                text=["!"] * len(inc_lats),
                hoverinfo="skip",
            ))
    
    center = compute_center(vehicles) if vehicles else ANCHOR
    
    fig.update_layout(
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN,
            style=MAP_STYLE,
            center=dict(lat=center["lat"], lon=center["lon"]),
            zoom=16,
            pitch=45,
            bearing=0,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        hovermode=False,
    )
    
    return fig


# =========================
# Dash App
# =========================

def main():
    print("Starting security dashboard...")
    
    app = Dash(__name__)
    app.title = "Security Dashboard"
    
    app.layout = html.Div([
        # Dark mode toggle
        html.Div([
            html.Label([
                html.Span("☀️", style={"marginRight": "8px", "fontSize": "18px"}),
                "Light",
            ], style={"marginRight": "10px", "color": "inherit"}),
            html.Label([
                dcc.Checklist(
                    id="dark-mode-toggle",
                    options=[{"label": "", "value": "dark"}],
                    value=[],
                    style={"margin": "0"}
                ),
            ], style={"marginRight": "10px"}),
            html.Label([
                "Dark",
                html.Span("🌙", style={"marginLeft": "8px", "fontSize": "18px"}),
            ], style={"color": "inherit"}),
        ], style={
            "position": "fixed",
            "top": "20px",
            "right": "20px",
            "zIndex": 10000,
            "display": "flex",
            "alignItems": "center",
            "background": "white",
            "padding": "10px 15px",
            "borderRadius": "25px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.1)",
            "fontSize": "14px",
            "fontWeight": "500"
        }),
        
        # Header
        html.Div([
            html.H1("LIVE TRAFFIC MONITORING", style={
                "color": "white",
                "margin": "0",
                "fontSize": "36px",
                "fontWeight": "bold",
                "textShadow": "2px 2px 4px rgba(0,0,0,0.3)"
            }),
            html.P("Multi-Camera Security Dashboard", style={
                "color": "rgba(255,255,255,0.9)",
                "margin": "10px 0 0 0",
                "fontSize": "16px"
            }),
        ], style={
            "padding": "30px",
            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "borderRadius": "12px",
            "marginBottom": "30px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.1)",
            "textAlign": "center"
        }),
        
        # Camera Grid
        html.Div(id="camera-grid", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",  # 4 columns
            "gap": "20px",
            "marginBottom": "30px"
        }),
        
        # Refresh interval
        dcc.Interval(id="refresh-interval", interval=1000, n_intervals=0),
        
        # Store for camera data
        dcc.Store(id="cameras-store", data={}),
        
        # Store for theme
        dcc.Store(id="theme-store", data="light"),
        
        html.Div(
            f"Security Dashboard • http://{HOST}:{PORT}",
            style={"textAlign": "center", "color": "#999", "fontSize": "13px", "marginTop": "20px"}
        ),
    ], id="main-container", style={
        "padding": "30px",
        "background": "#f5f7fa",
        "minHeight": "100vh",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    })
    
    @app.callback(
        Output("cameras-store", "data"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_camera_data(n):
        """Fetch latest data and group by camera."""
        raw_data = get_latest_data()
        cameras = process_camera_feeds(raw_data)
        return cameras
    
    @app.callback(
        Output("theme-store", "data"),
        Output("main-container", "style"),
        Input("dark-mode-toggle", "value"),
    )
    def toggle_theme(dark_mode):
        """Toggle between light and dark mode."""
        is_dark = "dark" in (dark_mode or [])
        theme = "dark" if is_dark else "light"
        
        if is_dark:
            # Dark mode styles
            style = {
                "padding": "30px",
                "background": "#1a1a1a",
                "minHeight": "100vh",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                "color": "#e0e0e0"
            }
        else:
            # Light mode styles
            style = {
                "padding": "30px",
                "background": "#f5f7fa",
                "minHeight": "100vh",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
            }
        
        return theme, style
    
    @app.callback(
        Output("camera-grid", "children"),
        Input("cameras-store", "data"),
        Input("theme-store", "data"),
    )
    def render_camera_grid(cameras, theme):
        """Render grid of camera previews with theme support."""
        is_dark = theme == "dark"
        
        # Theme colors
        if is_dark:
            bg_color = "#2d2d2d"
            text_color = "#e0e0e0"
            border_color = "#404040"
            card_hover_shadow = "0 8px 30px rgba(255,255,255,0.1)"
            no_data_bg = "#2d2d2d"
        else:
            bg_color = "white"
            text_color = "#333"
            border_color = "#eee"
            card_hover_shadow = "0 8px 30px rgba(0,0,0,0.12)"
            no_data_bg = "white"
        
        if not cameras:
            return html.Div([
                html.Div([
                    html.H2("No Active Cameras", style={
                        "color": "#999" if is_dark else "#666",
                        "marginBottom": "10px",
                        "fontSize": "28px"
                    }),
                    html.P("Waiting for camera data...", style={
                        "color": "#777" if is_dark else "#999",
                        "fontSize": "16px"
                    }),
                    html.Div("", style={
                        "width": "50px",
                        "height": "50px",
                        "border": "5px solid #404040" if is_dark else "5px solid #f3f3f3",
                        "borderTop": "5px solid #667eea",
                        "borderRadius": "50%",
                        "animation": "spin 1s linear infinite",
                        "margin": "20px auto"
                    }),
                ], style={
                    "background": no_data_bg,
                    "padding": "60px",
                    "borderRadius": "12px",
                    "boxShadow": "0 4px 20px rgba(0,0,0,0.1)",
                    "textAlign": "center",
                    "gridColumn": "1 / -1"
                })
            ])
        
        camera_cards = []
        
        for location, data in sorted(cameras.items()):
            vehicles = data["vehicles"]
            incidents = data["incidents"]
            unique_vehicles = len(set(v.get("id") for v in vehicles))
            incident_count = len(incidents)
            
            # Determine status
            is_live = len(vehicles) > 0
            status_color = "#4CAF50" if is_live else "#999"
            status_text = "LIVE" if is_live else "NO DATA"
            
            # Build preview map
            preview_fig = build_preview_map(vehicles, incidents, location)
            
            # Create camera card (entire card is clickable)
            card = html.A([
                # Camera name header
                html.Div([
                    html.H3(location.upper(), style={
                        "margin": "0",
                        "fontSize": "18px",
                        "fontWeight": "600",
                        "color": text_color
                    }),
                    html.Div([
                        html.Span("●", style={
                            "color": status_color,
                            "fontSize": "10px",
                            "marginRight": "5px"
                        }),
                        html.Span(status_text, style={
                            "color": status_color,
                            "fontSize": "11px",
                            "fontWeight": "600"
                        }),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "12px",
                    "paddingBottom": "12px",
                    "borderBottom": f"2px solid {border_color}"
                }),
                
                # Preview map
                html.Div([
                    dcc.Graph(
                        figure=preview_fig,
                        config={'displayModeBar': False},
                        style={"height": "180px", "width": "100%"}
                    ),
                ], style={
                    "borderRadius": "8px",
                    "overflow": "hidden",
                    "marginBottom": "12px",
                    "background": "#f8f9fa" if not is_dark else "#1a1a1a"
                }),
                
                # Stats
                html.Div([
                    html.Div([
                        html.Span(str(unique_vehicles), style={
                            "fontSize": "20px",
                            "fontWeight": "bold",
                            "color": "#667eea"
                        }),
                        html.Span(" Vehicles", style={
                            "fontSize": "13px",
                            "color": "#999" if is_dark else "#666",
                            "marginLeft": "5px"
                        }),
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span(str(incident_count), style={
                            "fontSize": "20px",
                            "fontWeight": "bold",
                            "color": "#f5576c" if incident_count > 0 else "#4CAF50"
                        }),
                        html.Span(" Incidents", style={
                            "fontSize": "13px",
                            "color": "#999" if is_dark else "#666",
                            "marginLeft": "5px"
                        }),
                    ]),
                ], style={
                    "marginBottom": "0",
                    "padding": "12px",
                    "background": "#f8f9fa" if not is_dark else "#1a1a1a",
                    "borderRadius": "8px"
                }),
            ], 
            href=f"http://127.0.0.1:8053?location={location}",
            target="_blank",
            style={
                "background": bg_color,
                "padding": "16px",
                "borderRadius": "12px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.08)" if not is_dark else "0 4px 20px rgba(0,0,0,0.3)",
                "border": f"2px solid {border_color}",
                "transition": "all 0.3s ease",
                "textDecoration": "none",
                "display": "block",
                "cursor": "pointer"
            }, 
            className="camera-card")
            
            camera_cards.append(card)
        
        return camera_cards
    
    # Add CSS for spinner and hover effects
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .camera-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3) !important;
                    border-color: #667eea !important;
                }
                html[data-theme="dark"] .camera-card:hover {
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
                }
            </style>
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
    
    print("\n" + "=" * 60)
    print("Security Dashboard - Multi-Camera View")
    print("=" * 60)
    print(f"Open: http://{HOST}:{PORT}")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host=HOST, port=PORT, use_reloader=False)


if __name__ == "__main__":
    main()