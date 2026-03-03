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

# Caltrans Colors
CALTRANS_BLUE = "#003366"
CALTRANS_GREEN = "#007B5F"

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

/* Main container — themed via data-theme, no inline style override */
.main-container {
    padding: 30px;
    background: #f5f7fa;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #333;
    transition: background 0.2s ease, color 0.2s ease;
}
[data-theme="dark"] .main-container {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
}

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
    border-radius: 4px; /* Formality adjustment */
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

.camera-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0, 51, 102, 0.2) !important;
    border-color: #003366 !important;
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


def main():
    print("Starting security dashboard...")
    
    app = Dash(__name__)
    app.title = "Security Dashboard"
    app.index_string = INDEX_STRING

    app.layout = html.Div([
        make_theme_toggle("theme-toggle"),
        
        # --- RE-STYLED HEADER FOR CALTRANS ---
        html.Div([
            html.H1("LIVE TRAFFIC MONITORING", style={
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
            html.P("DIVISION OF TRAFFIC OPERATIONS • MULTI-CAMERA SECURITY DASHBOARD", style={
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
            f"Official System Dashboard • http://{HOST}:{PORT}",
            style={"textAlign": "center", "color": "#999", "fontSize": "13px", "marginTop": "20px"}
        ),
    ], id="main-container", className="main-container")

    # ---- Clientside: wire up toggle button clicks ----
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
                } else {
                    track.classList.remove('active');
                    label.textContent = 'Theme: Light';
                }
            }
            return theme;
        }
        """,
        Output("theme-store", "data"),
        Input("theme-toggle", "value"),
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
                    html.H2("System Standby", style={
                        "color": CALTRANS_BLUE if not is_dark else "#999",
                        "marginBottom": "10px",
                        "fontSize": "28px",
                        "fontWeight": "700"
                    }),
                    html.P("Waiting for camera telemetry...", style={
                        "color": "#777" if is_dark else "#999",
                        "fontSize": "16px"
                    }),
                    html.Div("", style={
                        "width": "50px",
                        "height": "50px",
                        "border": "5px solid #404040" if is_dark else "5px solid #f3f3f3",
                        "borderTop": f"5px solid {CALTRANS_GREEN}",
                        "borderRadius": "50%",
                        "animation": "spin 1s linear infinite",
                        "margin": "20px auto"
                    }),
                ], style={
                    "background": bg_color,
                    "padding": "60px",
                    "borderRadius": "4px",
                    "boxShadow": "0 4px 20px rgba(0,0,0,0.1)",
                    "textAlign": "center",
                    "gridColumn": "1 / -1",
                    "border": f"1px solid {border_color}"
                })
            ])
        
        camera_cards = []
        
        for location, data in sorted(cameras.items()):
            vehicles = data["vehicles"]
            incidents = data["incidents"]
            unique_vehicles = len(set(v.get("id") for v in vehicles))
            incident_count = len(incidents)
            
            is_live = len(vehicles) > 0
            status_color = CALTRANS_GREEN if is_live else "#999"
            status_text = "LIVE" if is_live else "NO DATA"
            
            preview_fig = build_preview_map(vehicles, incidents, location)
            
            card = html.Div([
                html.Div([
                    html.H3(location.upper(), style={
                        "margin": "0",
                        "fontSize": "16px",
                        "fontWeight": "700",
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
                    "borderBottom": f"1px solid {border_color}"
                }),
                
                html.Div([
                    dcc.Graph(
                        figure=preview_fig,
                        config={'displayModeBar': False},
                        style={"height": "180px", "width": "100%"}
                    ),
                ], style={
                    "borderRadius": "2px",
                    "overflow": "hidden",
                    "marginBottom": "12px",
                    "background": "#1a1a1a" if is_dark else "#f8f9fa"
                }),
                
                html.Div([
                    html.Div([
                        html.Span(str(unique_vehicles), style={"fontSize": "20px", "fontWeight": "bold", "color": CALTRANS_BLUE if not is_dark else "#667eea"}),
                        html.Span(" Vehicles", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginLeft": "5px"}),
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span(str(incident_count), style={
                            "fontSize": "20px",
                            "fontWeight": "bold",
                            "color": "#f5576c" if incident_count > 0 else CALTRANS_GREEN
                        }),
                        html.Span(" Incidents", style={"fontSize": "13px", "color": "#999" if is_dark else "#666", "marginLeft": "5px"}),
                    ]),
                ], style={
                    "padding": "12px",
                    "background": "#1a1a1a" if is_dark else "#f8f9fa",
                    "borderRadius": "4px"
                }),
            ],
            id={"type": "camera-card", "location": location},
            n_clicks=0,
            style={
                "background": bg_color,
                "padding": "16px",
                "borderRadius": "4px",
                "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
                "border": f"1px solid {border_color}",
                "transition": "all 0.3s ease",
                "cursor": "pointer",
            },
            className="camera-card")
            
            camera_cards.append(card)
        
        return camera_cards

    # Navigate in the same tab when a camera card is clicked.
    app.clientside_callback(
        """
        function(n_clicks_list, cameras) {
            var ctx = window.dash_clientside.callback_context;
            if (!ctx.triggered || ctx.triggered.length === 0) {
                return window.dash_clientside.no_update;
            }
            var trigger = ctx.triggered[0];
            var clicks = trigger.value;
            if (!clicks || clicks === 0) return window.dash_clientside.no_update;

            try {
                var id = JSON.parse(trigger.prop_id.split('.')[0]);
                var location = id.location;
                window.location.href = 'http://127.0.0.1:8053?location=' + encodeURIComponent(location);
            } catch(e) {}
            return window.dash_clientside.no_update;
        }
        """,
        Output("cameras-store", "data", allow_duplicate=True),
        Input({"type": "camera-card", "location": dash.ALL}, "n_clicks"),
        State("cameras-store", "data"),
        prevent_initial_call=True,
    )

    app.run(debug=False, host=HOST, port=PORT, use_reloader=False)


if __name__ == "__main__":
    main()