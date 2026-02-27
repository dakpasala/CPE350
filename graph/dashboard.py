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
    if not raw_data:
        return {}
    
    vehicles = raw_data.get("vehicles", [])
    incidents = raw_data.get("incidents", [])
    
    if not vehicles:
        return {}
    
    cameras = defaultdict(lambda: {"vehicles": [], "incidents": [], "latest_timestamp": None})
    
    for v in vehicles:
        loc = v.get("location", "unknown")
        cameras[loc]["vehicles"].append(v)
        ts = v.get("timestamp")
        if ts:
            if cameras[loc]["latest_timestamp"] is None or ts > cameras[loc]["latest_timestamp"]:
                cameras[loc]["latest_timestamp"] = ts
    
    for inc in incidents:
        loc = inc.get("location", "unknown")
        if loc in cameras:
            cameras[loc]["incidents"].append(inc)
    
    return dict(cameras)


def compute_center(vehicles):
    pts = [(v.get("lat"), v.get("lon")) for v in vehicles]
    pts = [(lat, lon) for lat, lon in pts if lat is not None and lon is not None
           and math.isfinite(lat) and math.isfinite(lon)]
    
    if not pts:
        return ANCHOR
    
    lats = [lat for lat, _ in pts]
    lons = [lon for _, lon in pts]
    
    return {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}


def build_preview_map(vehicles, incidents, location):
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

TOGGLE_CSS = """
body { margin: 0; padding: 0; overflow-x: hidden; }

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ---- Theme Toggle Button ---- */
.theme-toggle-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 18px;
    border-radius: 50px;
    border: 2px solid rgba(0,0,0,0.12);
    background: #ffffff;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #333;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    transition: all 0.25s ease;
    user-select: none;
    white-space: nowrap;
}
.theme-toggle-btn:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    transform: translateY(-1px);
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
    background: #667eea;
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

/* dark mode button variant */
.theme-toggle-btn.dark-mode {
    background: #2d2d2d;
    border-color: rgba(255,255,255,0.15);
    color: #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}

.camera-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3) !important;
    border-color: #667eea !important;
}
"""

TOGGLE_JS = """
// Theme persistence across pages
window.DASHBOARD_THEME_KEY = 'dashboard_theme';

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
}

// Apply saved theme immediately on load (before React renders)
(function() {
    var saved = localStorage.getItem(window.DASHBOARD_THEME_KEY);
    if (saved) applyTheme(saved);
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
        <style>''' + TOGGLE_CSS + '''</style>
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


def make_theme_toggle(toggle_id="theme-toggle"):
    """Return the theme toggle button component."""
    return html.Div(
        id=f"{toggle_id}-wrapper",
        children=[
            # Hidden checklist to track state (Dash state management)
            dcc.Checklist(
                id=toggle_id,
                options=[{"label": "", "value": "dark"}],
                value=[],
                style={"display": "none"},
            ),
            # Visual button (clientside will wire clicks)
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
            "top": "20px",
            "right": "20px",
            "zIndex": 10000,
        }
    )


def main():
    print("Starting security dashboard...")
    
    app = Dash(__name__)
    app.title = "Security Dashboard"
    app.index_string = INDEX_STRING

    app.layout = html.Div([
        make_theme_toggle("theme-toggle"),
        
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
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "20px",
            "marginBottom": "30px"
        }),
        
        dcc.Interval(id="refresh-interval", interval=1000, n_intervals=0),
        dcc.Store(id="cameras-store", data={}),
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

    # ---- Clientside: wire up toggle button clicks + localStorage sync ----
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

    # Load theme from localStorage on page load
    app.clientside_callback(
        """
        function(id) {
            var saved = localStorage.getItem(window.DASHBOARD_THEME_KEY || 'dashboard_theme');
            if (saved === 'dark') return ['dark'];
            return [];
        }
        """,
        Output("theme-toggle", "value", allow_duplicate=True),
        Input("theme-toggle", "id"),
        prevent_initial_call='initial_duplicate',
    )

    # Update button appearance + save to localStorage when value changes
    app.clientside_callback(
        """
        function(value) {
            var isDark = value && value.includes('dark');
            var theme = isDark ? 'dark' : 'light';
            localStorage.setItem(window.DASHBOARD_THEME_KEY || 'dashboard_theme', theme);
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


    app.clientside_callback(
        """
        function(theme) {
            var isDark = theme === 'dark';
            return {
                padding: '30px',
                background: isDark ? '#1a1a1a' : '#f5f7fa',
                minHeight: '100vh',
                fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                color: isDark ? '#e0e0e0' : '#333',
            };
        }
        """,
        Output("main-container", "style"),
        Input("theme-store", "data"),
    )


    @app.callback(
        Output("cameras-store", "data"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_camera_data(n):
        raw_data = get_latest_data()
        cameras = process_camera_feeds(raw_data)
        return cameras

    @app.callback(
        Output("camera-grid", "children"),
        Input("cameras-store", "data"),
        Input("theme-store", "data"),
    )
    def render_camera_grid(cameras, theme):
        is_dark = theme == "dark"
        
        bg_color = "#2d2d2d" if is_dark else "white"
        text_color = "#e0e0e0" if is_dark else "#333"
        border_color = "#404040" if is_dark else "#eee"
        
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
                    "background": bg_color,
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
            
            is_live = len(vehicles) > 0
            status_color = "#4CAF50" if is_live else "#999"
            status_text = "LIVE" if is_live else "NO DATA"
            
            preview_fig = build_preview_map(vehicles, incidents, location)
            
            card = html.A([
                html.Div([
                    html.H3(location.upper(), style={
                        "margin": "0",
                        "fontSize": "18px",
                        "fontWeight": "600",
                        "color": text_color
                    }),
                    html.Div([
                        html.Span("●", style={"color": status_color, "fontSize": "10px", "marginRight": "5px"}),
                        html.Span(status_text, style={"color": status_color, "fontSize": "11px", "fontWeight": "600"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "12px",
                    "paddingBottom": "12px",
                    "borderBottom": f"2px solid {border_color}"
                }),
                
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
                    "background": "#1a1a1a" if is_dark else "#f8f9fa"
                }),
                
                html.Div([
                    html.Div([
                        html.Span(str(unique_vehicles), style={"fontSize": "20px", "fontWeight": "bold", "color": "#667eea"}),
                        html.Span(" Vehicles", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginLeft": "5px"}),
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span(str(incident_count), style={
                            "fontSize": "20px",
                            "fontWeight": "bold",
                            "color": "#f5576c" if incident_count > 0 else "#4CAF50"
                        }),
                        html.Span(" Incidents", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginLeft": "5px"}),
                    ]),
                ], style={
                    "padding": "12px",
                    "background": "#1a1a1a" if is_dark else "#f8f9fa",
                    "borderRadius": "8px"
                }),
            ],
            href=f"http://127.0.0.1:8053?location={location}",
            target="_blank",
            style={
                "background": bg_color,
                "padding": "16px",
                "borderRadius": "12px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.3)" if is_dark else "0 4px 20px rgba(0,0,0,0.08)",
                "border": f"2px solid {border_color}",
                "transition": "all 0.3s ease",
                "textDecoration": "none",
                "display": "block",
                "cursor": "pointer"
            },
            className="camera-card")
            
            camera_cards.append(card)
        
        return camera_cards

    print("\n" + "=" * 60)
    print("Security Dashboard - Multi-Camera View")
    print("=" * 60)
    print(f"Open: http://{HOST}:{PORT}")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host=HOST, port=PORT, use_reloader=False)


if __name__ == "__main__":
    main()